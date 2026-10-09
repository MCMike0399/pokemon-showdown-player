"""Read-only inventory and bounded, stratified training-quality audit."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.continuous import LearningConfig
from ml.data_budget import storage_status
from ml.data_quality import trajectory_issues
from ml.storage import DEFAULT_ROOT, now


def report(root: Path, sample_per_group: int = 40) -> dict:
    if not 1 <= sample_per_group <= 500:
        raise ValueError('sample_per_group must be 1..500')
    config = LearningConfig.load(root)
    db = sqlite3.connect((root / 'experience.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('BEGIN')
        corpus = [dict(r) for r in db.execute('''SELECT format,source,status,COUNT(*) games,
            SUM(trained) consumed FROM episodes GROUP BY format,source,status''')]
        public = [dict(r) for r in db.execute('''SELECT format,
            CASE WHEN source='own-live-game' THEN 'own-live-game'
                 WHEN source='local-simulation' THEN 'local-simulation' ELSE 'external-replay' END source,
            COUNT(*) games FROM public_battles GROUP BY 1,2''')]
        quality = []
        for group in corpus:
            if group['status'] != 'complete':
                continue
            ids = [r[0] for r in db.execute("SELECT id FROM episodes WHERE format=? AND source=? AND status='complete' ORDER BY created,id",
                                            (group['format'], group['source']))]
            # Spread across history, plus the freshest records. Deterministic,
            # bounded and explicitly a sample, never an all-corpus quality claim.
            n = min(len(ids), sample_per_group)
            positions = {round(i * (len(ids)-1) / max(1, n-1)) for i in range(n)}
            selected = {ids[i] for i in positions} | set(ids[-min(10, n):])
            defects, limits, usage = Counter(), Counter(), Counter()
            defective_records = []
            for key in sorted(selected):
                episode = json.loads(db.execute('SELECT data FROM episodes WHERE id=?', (key,)).fetchone()[0])
                issues = trajectory_issues(episode)
                defects.update(issues)
                if issues:
                    defective_records.append({'id': key, 'issues': issues})
                usage['structurally_valid'] += not issues
                usage['on_policy_flag'] += bool(episode.get('on_policy'))
                usage['with_collecting_checkpoint'] += bool(episode.get('collecting_policy', {}).get('checkpoint_archive'))
                if not episode.get('on_policy'):
                    limits['not-ppo-rollout'] += 1
                if not episode.get('terminal_log'):
                    limits['missing-terminal-log-evidence'] += 1
                if any(not step.get('snapshot') for step in episode.get('steps', [])):
                    limits['legacy-encoded-only-steps'] += 1
            quality.append({'format': group['format'], 'source': group['source'],
                            'sampled_games': len(selected), 'total_games': len(ids),
                            'coverage': dict(usage), 'defects': dict(defects), 'defective_records': defective_records, 'limitations': dict(limits)})
        scout = [dict(r) for r in db.execute('''SELECT format,COUNT(*) samples,
            COUNT(DISTINCT battle) battles,COUNT(DISTINCT species) species FROM scout_samples GROUP BY format''')]
        for group in scout:
            group['orphaned_samples'] = db.execute('''SELECT COUNT(*) FROM scout_samples s WHERE s.format=?
                AND NOT EXISTS(SELECT 1 FROM public_battles p WHERE p.format=s.format AND p.digest=s.battle)''',
                (group['format'],)).fetchone()[0]
        return {'checked_at': now(), 'storage': storage_status(root, config.resource['max_disk_gb']),
                'episodes': corpus, 'public_battles': public, 'scout': scout, 'quality_sample': quality,
                'collection_policy': {'current_formats': config.formats, 'curriculum': config.curriculum,
                    'simulation_games_per_batch': config.simulation_games, 'backlog_steps': config.max_training_backlog_steps,
                    'max_daily_replays_per_format': config.max_replays},
                'interpretation': [
                    'Original trajectories, private requests, collecting checkpoints, replay bodies and evaluation reports are retained.',
                    'Structural validation is sampled; exact on-policy eligibility also requires matching revision, features and collecting likelihoods.',
                    'Public logs and search games supply scouting evidence, not private PPO actions or expert labels.',
                    'Historical formats remain separately labeled; evaluation cases are not training trajectories.',
                    'Background admission reserves 10% for live finalization; this is cooperative accounting, not an OS filesystem quota.']}
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--sample-per-group', type=int, default=40)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root, args.sample_per_group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
