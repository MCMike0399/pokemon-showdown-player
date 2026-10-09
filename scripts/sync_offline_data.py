#!/usr/bin/env python3
"""Export a read-only, deduplicated DiveMac corpus; verify every transferred file."""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import json
import platform
import shlex
import shutil
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

FORMATS = ('gen9championsvgc2026regmc', 'gen9championsvgc2026regmb')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def export(source, output, episode_limit):
    source, output = Path(source).expanduser().resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect((source/'data/ml/experience.sqlite3').as_uri()+'?mode=ro', uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.execute('BEGIN')
    inventory = {table: [dict(r) for r in db.execute(sql)] for table, sql in {
        'episodes': 'SELECT format,source,status,count(*) AS count FROM episodes GROUP BY format,source,status',
        'public_battles': "SELECT format,CASE WHEN source LIKE 'https://replay.%' THEN 'showdown-replays' ELSE source END AS source,count(*) AS count FROM public_battles GROUP BY format,2",
    }.items()}
    counts = {}
    (output/'inputs').mkdir()
    target = sqlite3.connect(output/'inputs/experience.sqlite3')
    for table in ('research_teams','documents','feed_state','team_metadata','public_battles'):
        target.execute(db.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?', ('table', table)).fetchone()[0])
        query = f'SELECT * FROM {table}'
        args = ()
        if table in ('research_teams','documents','public_battles'):
            query += ' WHERE format IN (?,?)'
            args = FORMATS
        if table == 'public_battles':
            query += " AND source!='local-simulation' ORDER BY format,id"
        rows = db.execute(query, args)
        count = 0
        for row in rows:
            values = tuple(row)
            target.execute(f'INSERT INTO {table} VALUES ({",".join("?" for _ in values)})', values)
            count += 1
        counts[table] = count
    target.commit()
    target.close()
    # Exact historic requests are retained for diagnosis, NEVER passed as fresh PPO.
    counts['historic_ladder_episodes'] = 0
    with gzip.open(output/'historic-ladder-episodes.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in db.execute("SELECT data FROM episodes WHERE source='ladder' AND status='complete' AND format=? ORDER BY created DESC LIMIT ?", (FORMATS[0], episode_limit)):
            stream.write(row[0]+'\n')
            counts['historic_ladder_episodes'] += 1
    db.close()
    for name in ('models','scouts'):
        (output/name).mkdir()
        for fmt in FORMATS:
            path = source/'data/ml'/name/(fmt+'.pt')
            if path.exists():
                shutil.copy2(path, output/name/path.name)
    if (source/'teams.json').exists():
        shutil.copy2(source/'teams.json', output/'teams.json')
    for fmt in FORMATS:
        if (source/'cache/dex'/fmt).exists():
            shutil.copytree(source/'cache/dex'/fmt, output/'dex'/fmt)
    manifest = {'exported_at': datetime.now(timezone.utc).isoformat(), 'host': platform.node(),
        'source_root': str(source), 'formats': FORMATS, 'snapshot': 'SQLite read transaction; no source writes',
        'inventory': inventory, 'selected_counts': counts,
        'source_commit': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip(),
        'use': {'public_battles':'executed-public-move scouting; pre-turn features, battle-disjoint split',
                'historic_ladder_episodes':'diagnosis only; stale probabilities/revisions excluded from new PPO',
                'models':'optional frozen incumbent/warm start; never installed in production'},
        'licenses': {'vgc-bench-champions-mb':'MIT (owner declaration); exact archive provenance retained in feed_state',
                     'showdown-replays':'public battle evidence; no blanket dataset license asserted',
                     'research_teams':'original URL/player/event attribution retained; unknown spreads stay unknown'},
        'files': {str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':digest(p)}
                  for p in sorted(output.rglob('*')) if p.is_file()}}
    for node in ast.parse((source/'ml/feeds.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ARCHIVES' for t in node.targets):
            manifest['declared_upstream_archives'] = ast.literal_eval(node.value)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

def sync(host, source, destination, episode_limit):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('use a fresh destination; existing exports are immutable')
    remote = '/tmp/pokemon-offline-export-'+uuid.uuid4().hex
    code = Path(__file__).read_text()
    command = shlex.join(['python3','-','--export-only','--source',source,'--destination',remote,
                          '--episode-limit',str(episode_limit)])
    subprocess.run(['ssh','-o','BatchMode=yes',host,command], input=code, text=True, check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['scp','-q','-r',f'{host}:{remote}',str(destination)], check=True)
    manifest = json.loads((destination/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = destination/name
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('transfer checksum mismatch: '+name)
    print(json.dumps({'destination':str(destination),'verified_files':len(manifest['files']),
                      'counts':manifest['selected_counts'],
                      'bytes':sum(f['bytes'] for f in manifest['files'].values())}, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='DiveMac')
    p.add_argument('--source', default='~/Developer/pokemon-showdown-player')
    p.add_argument('--destination', default='data/offline-corpus/divemac-20261009')
    p.add_argument('--episode-limit', type=int, default=2000)
    p.add_argument('--export-only', action='store_true')
    a = p.parse_args()
    if a.episode_limit < 1:
        p.error('episode limit must be positive')
    if a.export_only:
        print(json.dumps(export(a.source, a.destination, a.episode_limit)['selected_counts']))
    else:
        sync(a.host, a.source, a.destination, a.episode_limit)
