"""Ladder-replica opponent teams: each live opponent's previewed six, sets drawn from
usage priors conditioned on what that opponent revealed, validated natively."""
from __future__ import annotations

import json
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battle_state import to_id  # noqa: E402
from search.priors import sample_set  # noqa: E402
from search.tracker import Tracker  # noqa: E402

FMT = 'gen9championsvgc2026regmc'


def our_side(log):
    for line in log:
        p = line.split('|')
        if len(p) > 4 and p[1] == 'switch':
            hp = p[4].split()[0]
            if '/' in hp and hp.split('/')[1] != '100':
                return p[2][:2]
    return None


def to_set(s):
    names = json.loads((ROOT / 'cache/dex' / FMT / 'moves.json').read_text())
    return s


def main(out_path, per_game=1, seed=7):
    db = sqlite3.connect(f'file:{ROOT}/data/ml/experience.sqlite3?mode=ro', uri=True)
    rows = db.execute("select id, log from public_battles where source='own-live-game' order by fetched").fetchall()
    rng = random.Random(seed)
    candidates = []
    for bid, log in rows:
        log = json.loads(log)
        me = our_side(log)
        if not me:
            continue
        t = Tracker(me).feed(log)
        foe = t.foe
        species = t.preview.get(foe) or []
        if len(species) < 4:
            continue
        seen = {m.base_species: m for m in t.side_mons(foe)}
        for k in range(per_game):
            for attempt in range(8):
                team = []
                used_items = set()
                for sp in species:
                    m = seen.get(sp) or next((v for b, v in seen.items() if to_id(b) == to_id(sp)), None)
                    obs = {}
                    if m:
                        obs = {'moves': m.moves, 'item': m.item if m.item_known else None, 'ability': m.ability,
                               'mega': m.mega, 'mega_species': m.species if m.mega else None}
                    s = None
                    for _ in range(12):
                        s = sample_set(rng, sp, obs, FMT, '0')
                        if s.get('item') not in used_items or not s.get('item'):
                            break
                    used_items.add(s.get('item'))
                    s = {k2: v for k2, v in s.items() if k2 in ('species', 'item', 'ability', 'nature', 'evs', 'moves', 'level')}
                    team.append(s)
                candidates.append({'battle': bid, 'team': team, 'attempt': attempt})
                break
    # Native validation in one pass; retry failures by resampling individually.
    teams = [c['team'] for c in candidates]
    res = subprocess.run(['node', str(ROOT / 'search/validate_many.cjs')], input=json.dumps({'format': FMT, 'teams': teams}),
                         capture_output=True, text=True, check=True)
    errors = json.loads(res.stdout)
    for rnd in range(6):
        bad_idx = [i for i, e in enumerate(errors) if e]
        if not bad_idx:
            break
        for i in bad_idx:
            c = candidates[i]
            # Resample the whole team with a different stream.
            rng2 = random.Random(seed * 1000 + i * 10 + rnd)
            log = None
            team = []
            used = set()
            for s in c['team']:
                sp = s['species']
                for _ in range(20):
                    s2 = sample_set(rng2, sp, {}, FMT, '0')
                    if not s2.get('item') or s2['item'] not in used:
                        break
                used.add(s2.get('item'))
                team.append({k2: v for k2, v in s2.items() if k2 in ('species', 'item', 'ability', 'nature', 'evs', 'moves', 'level')})
            # Keep observed facts for mons when possible: prefer original sets that individually validate later.
            candidates[i]['team'] = team
        res = subprocess.run(['node', str(ROOT / 'search/validate_many.cjs')],
                             input=json.dumps({'format': FMT, 'teams': [candidates[i]['team'] for i in bad_idx]}),
                             capture_output=True, text=True, check=True)
        for i, e in zip(bad_idx, json.loads(res.stdout)):
            errors[i] = e
    valid = [c for c, e in zip(candidates, errors) if not e]
    bad = [(c, e) for c, e in zip(candidates, errors) if e]
    print('candidates', len(candidates), 'valid', len(valid), 'invalid', len(bad))
    from collections import Counter
    print(Counter(e[0][:60] for _, e in bad).most_common(15))
    Path(out_path).write_text(json.dumps([{'battle': c['battle'], 'sets': c['team']} for c in valid], indent=0))


if __name__ == '__main__':
    main(sys.argv[1])
