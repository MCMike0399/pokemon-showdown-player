"""Read-only corpus quality, campaign provenance and learning coverage audit."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data_quality import trajectory_issues
from ml.mechanics import charging
from ml.recording import log_prefix
from ml.storage import now
from battle_state import to_id


def audit_data(campaign: Path, root: Path):
    import torch
    manifest = json.loads((campaign / 'manifest.json').read_text())
    fmt = manifest['format']
    db = sqlite3.connect((root / 'experience.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    checkpoint = torch.load(root / 'models' / (fmt + '.pt'), map_location='cpu', weights_only=True)
    records = [json.loads(line) for line in (campaign / 'games.jsonl').read_text().splitlines() if line.strip()]
    if len({r['room'] for r in records}) != len(records):
        raise ValueError('duplicate campaign rooms')
    counts, patterns, issues, episodes, consumed = Counter(), Counter(), [], [], []
    values, lengths = [], []
    fingerprints = {}
    try:
        corpus = [dict(row) for row in db.execute('''SELECT format,source,status,COUNT(*) episodes,
                SUM(trained) consumed,SUM(length(data)) json_bytes FROM episodes GROUP BY format,source,status''')]
        scout = [dict(row) for row in db.execute('''SELECT format,COUNT(*) samples,COUNT(DISTINCT battle) battles,
                COUNT(DISTINCT species) species,COUNT(DISTINCT move) moves FROM scout_samples GROUP BY format''')]
        for group in scout:
            group['orphaned_samples'] = db.execute('''SELECT COUNT(*) FROM scout_samples s WHERE s.format=?
                AND NOT EXISTS (SELECT 1 FROM public_battles p WHERE p.format=s.format AND p.digest=s.battle)''',
                (group['format'],)).fetchone()[0]
        for ordinal, record in enumerate(records, 1):
            rows = db.execute("SELECT id,trained,data FROM episodes WHERE json_extract(data,'$.room')=?", (record['room'],)).fetchall()
            complete = [(row, json.loads(row['data'])) for row in rows if json.loads(row['data'])['status'] == 'complete']
            if len(complete) != 1:
                raise ValueError('campaign room must have one terminal episode')
            row, episode = complete[0]
            fingerprints[episode['id']] = {'trained': row['trained'], 'sha256': hashlib.sha256(row['data'].encode()).hexdigest()}
            counts['terminal_games'] += 1
            counts['wins' if episode['outcome'] == 1 else 'losses' if episode['outcome'] == -1 else 'ties'] += 1
            counts['consumed_episodes'] += bool(row['trained'])
            counts['episodes_with_terminal_log'] += bool(episode.get('terminal_log'))
            counts['episodes_with_policy_provenance'] += bool(episode.get('collecting_policy'))
            if len(rows) != 1:
                counts['fragmented_excluded'] += 1
                issues.append({'game': ordinal, 'id': episode['id'], 'issues': ['fragmented-room'], 'outcome_retained': True})
                continue
            defects = trajectory_issues(episode)
            if defects:
                issues.append({'game': ordinal, 'id': episode['id'], 'issues': defects})
            else:
                counts['structurally_usable_episodes'] += 1
            current = episode['revision'] == checkpoint['revision'] and episode.get('on_policy') and episode.get('feature_profile', 'legacy') == checkpoint.get('feature_profile', 'legacy')
            counts['current_policy_episodes'] += current and not defects
            counts['current_policy_unconsumed_episodes'] += current and not defects and not row['trained']
            lengths.append(len(episode['steps']))
            episodes.append({'id': episode['id'], 'game': ordinal, 'team': episode['team'], 'revision': episode['revision'],
                'source': episode['source'], 'steps': len(episode['steps']), 'outcome': episode['outcome'],
                'consumed': bool(row['trained']), 'issues': defects})
            for step in episode['steps']:
                counts['encoded_steps'] += 1
                values.append(step['value'])
                snapshot = step.get('snapshot')
                if not snapshot:
                    counts['legacy_encoded_only_steps'] += 1
                    continue
                counts['rich_snapshots'] += 1
                counts['verified_prefixes'] += not any('log-prefix' in d or 'observation' in d for d in defects)
                observation = snapshot['observation']
                counts['snapshots_with_open_sheets'] += bool(observation.get('opp_team_sheet'))
                chosen = snapshot['legal_choices'][step['index']]
                counts['alternative_entries'] += len(snapshot['legal_choices'])
                if chosen['kind'] == 'team':
                    counts['previews'] += 1
                    counts['previews_without_rain_setter'] += 'Politoed' not in chosen['pokemon']
                    continue
                for component in chosen['components']:
                    if component['kind'] != 'move':
                        continue
                    name = to_id(component.get('move', ''))
                    patterns['selected/' + name] += 1
                    if name == 'electroshot' and observation.get('weather') not in ('RainDance', 'PrimordialSea'):
                        i = component['slot'] - 1
                        actor = observation['my_actives'][i]
                        move = snapshot['request']['active'][i]['moves'][component['move_index'] - 1]
                        ctx = {'state': observation, 'public_log': log_prefix(episode, snapshot['public_log'])}
                        patterns['electroshot_dry_charge' if charging(ctx, actor, move) else 'electroshot_dry_release_or_herb'] += 1
        jobs, panels = Counter(), []
        for row in db.execute('SELECT id,kind,result FROM jobs WHERE format=? AND result IS NOT NULL', (fmt,)):
            result = json.loads(row['result'])
            training = result.get('training') or {}
            gate = result.get('evaluation') or {}
            jobs['with_errors'] += bool(result.get('error'))
            jobs['trained_candidates'] += bool(training.get('trained'))
            jobs['single_episode_candidates'] += bool(training.get('trained')) and training.get('episodes') == 1
            jobs['completed_evaluations'] += bool(gate.get('complete'))
            jobs['passed_gates'] += bool(gate.get('passed'))
            jobs['staged_candidates'] += bool(result.get('staged_for_between_game_promotion'))
            if gate:
                panels.append({'job': row['id'], 'kind': row['kind'], 'episodes': training.get('episodes'),
                    'steps': training.get('steps'), 'incumbent_wins': gate.get('incumbent_wins'),
                    'candidate_wins': gate.get('candidate_wins'), 'passed': gate.get('passed'),
                    'games': gate.get('games_per_policy')})
        for path in (root / 'candidates').glob('*/training-state.json'):
            state = json.loads(path.read_text())
            covered = [key for key in state.get('training', {}).get('consumed', []) if key in fingerprints]
            if covered:
                consumed.append({'candidate': path.parent.name, 'campaign_episodes': covered})
        return {'checked_at': now(), 'format': fmt, 'collecting_revision': checkpoint['revision'],
            'campaign': dict(counts), 'issues': issues, 'patterns': dict(patterns), 'corpus': corpus, 'scout': scout,
            'training_jobs': dict(jobs), 'evaluation_panels': panels, 'candidate_consumption': consumed,
            'episode_evidence': fingerprints, 'episodes': episodes,
            'value_summary': {'min': min(values, default=None), 'max': max(values, default=None),
                              'median': statistics.median(values) if values else None},
            'median_steps': statistics.median(lengths) if lengths else None,
            'limitations': ['Consumed means used to train a candidate, not promoted or improved.',
                'No raw private requests exist for encoded-only historical decisions.',
                'Exact collecting logprobs require the original checkpoint; structural checks alone do not prove on-policy provenance.',
                'Public executed moves train scouting, not private PPO masks or expert command labels.',
                'Archived M-B scout samples remain separate from current M-C actor episodes.']}
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('data/ml'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_data(args.campaign, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('campaign', 'training_jobs', 'value_summary', 'median_steps', 'issues')}, indent=2))
