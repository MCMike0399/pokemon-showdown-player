"""Freeze terminal campaign evidence and fit executed-move behavior frequencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.postgame import archived_log, knowledge_from_reviews, review


def run(root, campaign, output, cutoff):
    rows = [json.loads(line) for line in (campaign / 'games.jsonl').read_text().splitlines() if line.strip()][:cutoff]
    if len(rows) != cutoff or len({r['room'] for r in rows}) != cutoff:
        raise ValueError('cutoff requires the exact count of unique terminal rooms')
    fmt = json.loads((campaign / 'manifest.json').read_text())['format']
    db = sqlite3.connect((root / 'experience.sqlite3').as_uri() + '?mode=ro', uri=True)
    reports = []
    excluded = []
    episode_ids = []
    try:
        for ordinal, row in enumerate(rows, 1):
            episodes = [json.loads(r[0]) for r in db.execute("SELECT data FROM episodes WHERE json_extract(data,'$.room')=?", (row['room'],))]
            terminal = [e for e in episodes if e['status'] == 'complete' and e['format'] == fmt]
            if len(terminal) != 1:
                raise ValueError('ambiguous terminal room')
            episode = terminal[0]
            log = archived_log(episode, db)
            if not log:
                excluded.append({'ordinal': ordinal, 'reason': 'terminal public log unavailable'})
                continue
            item = review(episode, log)
            item.update(ordinal=ordinal, team=episode['team'], revision=episode['revision'],
                        log_sha256=hashlib.sha256(json.dumps(log).encode()).hexdigest())
            reports.append(item)
            episode_ids.append(episode['id'])
    finally:
        db.close()
    knowledge = knowledge_from_reviews(reports, fmt)
    knowledge['learned_episode_ids'] = sorted(episode_ids)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'reviews.json').write_text(json.dumps(reports, indent=2))
    (output / 'knowledge.json').write_text(json.dumps(knowledge, indent=2))
    summary = {'cutoff': cutoff, 'games_with_logs': len(reports), 'excluded': excluded,
               'wins': sum(r['outcome'] == 1 for r in reports),
               'captured_move_components': sum(r['captured_move_components'] for r in reports),
               'components_without_matching_execution': sum(len(r['components_without_matching_execution']) for r in reports),
               'dry_electro_shot_selections': sum(r['dry_electro_shot_selections'] for r in reports),
               'opponent_move_samples': knowledge['executed_samples'], 'opponent_species': len(knowledge['moves']),
               'knowledge_sha256': hashlib.sha256((output / 'knowledge.json').read_bytes()).hexdigest(),
               'limitations': ['Legacy observations cannot recover unavailable private pre-decision snapshots.',
                               'All cut-off games inform behavior evidence; none are fabricated PPO or expert labels.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, default=500)
    a = parser.parse_args()
    print(json.dumps(run(a.root.resolve(), a.campaign.resolve(), a.output.resolve(), a.cutoff), indent=2))
