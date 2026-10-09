"""Isolated campaign PPO and mechanical ablation with fresh paired holdouts.

No login, production consumption reset, automatic promotion or live games.
The output directory retains a frozen plan and resumes completed comparisons.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]

from ml.continuous import LearningConfig
from ml.model import Model
from ml.promotion import paired_gate
from ml.resources import ResourcePolicy
from ml.storage import Store, now
from ml.worker import atomic_json, candidate_teams, freeze_inputs, parallel_games, snapshot


def source_hashes():
    files = list((REPO / 'ml').glob('*.py')) + [REPO / name for name in
        ('battle_state.py', 'harness.py', 'simulator.cjs', 'scripts/improve_campaign.py')]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


async def improve(campaign, root, output, development=96, final=200, seed=1250000, max_seconds=1800):
    from harness import TeamStore
    from ml.simulator import validate_team
    from ml.teams import team_id
    if not 4 <= development <= 1000 or not 4 <= final <= 1000 or not 10 <= max_seconds <= 1800:
        raise ValueError('evaluation panels must be 4..1000 games and budget 10..1800 seconds')
    output.mkdir(parents=True, exist_ok=True)
    fmt = json.loads((campaign / 'manifest.json').read_text())['format']
    policy = ResourcePolicy(**{**LearningConfig.load(root).resource, 'max_workers': 2})
    policy.apply_background()
    deadline = time.monotonic() + max_seconds
    suite_path = output / 'suite.json'
    if suite_path.exists() and not (output / 'checkpoint-hashes.json').exists():
        raise ValueError('experiment initialization was interrupted; retain it and use a fresh output directory')
    if not suite_path.exists():
        live = Store(root)
        try:
            incumbent = Model(root, fmt)
            for label in ('incumbent', 'pooled-ppo', 'mechanics'):
                snapshot(incumbent.path, output / label, fmt)
            freeze_inputs(live, output / 'inputs', fmt)
            focus_name = LearningConfig.load(root).training_teams.get(fmt)
            if not focus_name:
                raise ValueError('campaign experiment requires an explicit training team')
            focused = TeamStore().get(focus_name)
            if focused['format'] != fmt or (await validate_team(fmt, focused['sets']))['errors']:
                raise ValueError('focused team format/legality mismatch')
            focused_id = team_id(fmt, focused['sets'])
            teams = []
            for sets in candidate_teams(live, fmt):
                if not (await validate_team(fmt, sets))['errors']:
                    teams.append(sets)
            if not teams:
                raise ValueError('no legal opponent teams')
            records = [json.loads(line) for line in (campaign / 'games.jsonl').read_text().splitlines() if line.strip()]
            if len({r['room'] for r in records}) != len(records):
                raise ValueError('duplicate campaign rooms')
            episodes, evidence, excluded = [], [], []
            for record in records:
                rows = live.db.execute("SELECT id,trained,data FROM episodes WHERE json_extract(data,'$.room')=?",
                                       (record['room'],)).fetchall()
                if len(rows) != 1:
                    excluded.append({'room': record['room'], 'reason': 'fragmented'}); continue
                episode = json.loads(rows[0]['data'])
                if (episode['status'] != 'complete' or episode['revision'] != incumbent.revision or
                        episode.get('team') != focused_id or not episode.get('on_policy') or
                        episode.get('feature_profile', 'legacy') != incumbent.feature_profile):
                    continue
                evidence.append({'id': episode['id'], 'trained': rows[0]['trained'],
                                 'sha256': hashlib.sha256(rows[0]['data'].encode()).hexdigest(),
                                 'steps': len(episode['steps']), 'outcome': episode['outcome']})
                # Immutable encoded collecting inputs, not reconstructed private data.
                for step in episode['steps']:
                    step.pop('snapshot', None)
                episodes.append(episode)
            cases = {}
            opponent_cycle = ('self', 'heuristic', 'tactical')
            for phase, n, offset in (('development', development, 0), ('final', final, 10000)):
                cases[phase] = []
                for i in range(n):
                    opponent = opponent_cycle[(i + i // len(teams)) % len(opponent_cycle)]
                    cases[phase].append({'root': str(output / 'inputs'), 'inference_root': str(output / 'inputs'),
                        'format': fmt, 'team1': focused['sets'], 'team2': teams[i % len(teams)],
                        'seed': seed + offset + i, 'side': 'p2' if (i + i // len(teams)) % 2 else 'p1',
                        'training': False, 'sample_actions': True, 'opponent': opponent,
                        'seed_encoding': 'full-v1',
                        'open_team_sheets': i % 12 == 0,
                        'opponent_checkpoint_root': str(output / 'incumbent') if opponent == 'self' else None})
            atomic_json(suite_path, {'declared_at': now(), 'format': fmt, 'parent_revision': incumbent.revision,
                'parent_sha256': hashlib.sha256(incumbent.path.read_bytes()).hexdigest(),
                'training_team': {'name': focus_name, 'fingerprint': focused_id}, 'cases': cases,
                'margin': .1, 'training_evidence': evidence, 'excluded': excluded,
                'source_hashes': source_hashes(),
                'selection': 'Development-passing candidate with most wins; independent final gate. No automatic promotion.',
                'changes': {'pooled-ppo': 'Pooled compatible ladder PPO, up to 8 epochs, KL .03; unchanged profile/temperatures.',
                            'mechanics': 'Turn-only mechanics-v1, unchanged weights/state/preview/temperatures.'},
                'limitations': ['Local opponents are scripted/frozen policies, not humans.',
                                'Fresh seeds; opponent team compositions can overlap practice.']})
            pooled = Model(output / 'pooled-ppo', fmt)
            training = pooled.train(episodes, epochs=8, deadline=deadline)
            atomic_json(output / 'training.json', training)
            if not training.get('trained'):
                raise ValueError('pooled training failed data validation; no candidate evaluation was started')
            mechanical = Model(output / 'mechanics', fmt)
            mechanical.feature_profile = 'mechanics-v1'
            mechanical.revision = uuid.uuid4().hex
            mechanical.save()
            atomic_json(output / 'checkpoint-hashes.json', {label: hashlib.sha256(
                (output / label / 'models' / (fmt + '.pt')).read_bytes()).hexdigest()
                for label in ('incumbent', 'pooled-ppo', 'mechanics')})
        finally:
            live.close()
    suite = json.loads(suite_path.read_text())
    hashes = json.loads((output / 'checkpoint-hashes.json').read_text())
    for label, expected in hashes.items():
        if hashlib.sha256((output / label / 'models' / (fmt + '.pt')).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen experiment checkpoint changed')

    async def evaluate(label, phase):
        path = output / (phase + '-' + label + '.json')
        results = json.loads(path.read_text())['games'] if path.exists() else []
        tasks = [{**task, 'checkpoint_root': str(output / label)} for task in suite['cases'][phase]]
        while len(results) < len(tasks) and time.monotonic() < deadline:
            if suite.get('source_hashes') and source_hashes() != suite['source_hashes']:
                raise ValueError('frozen experiment source changed; restore it or start a fresh output directory')
            report = await parallel_games(tasks[len(results):], policy, min(deadline, time.monotonic() + 45))
            results.extend(report['games'])
            atomic_json(path, {'games': results})
            atomic_json(output / 'status.json', {'phase': phase, 'policy': label, 'completed': len(results),
                                                'target': len(tasks), 'updated_at': now()})
            print(phase, label, len(results), '/', len(tasks), flush=True)
            if len(results) < len(tasks):
                await asyncio.sleep(2)
        return results

    report = {'parent_revision': suite['parent_revision'], 'promoted': False, 'development': {}}
    base = await evaluate('incumbent', 'development')
    for label in ('pooled-ppo', 'mechanics'):
        candidate = await evaluate(label, 'development')
        report['development'][label] = paired_gate(base, candidate, len(suite['cases']['development']), suite['margin'])
        atomic_json(output / 'report.json', report)
    passing = [k for k, v in report['development'].items() if v['passed']]
    report['selected'] = max(passing, key=lambda k: report['development'][k]['candidate_wins']) if passing else None
    if report['selected']:
        base = await evaluate('incumbent', 'final')
        candidate = await evaluate(report['selected'], 'final')
        report['final'] = paired_gate(base, candidate, len(suite['cases']['final']), suite['margin'])
        report['candidate_checkpoint'] = str(output / report['selected'] / 'models' / (fmt + '.pt'))
    else:
        report['final_untouched'] = True
    report['complete'] = all(v['complete'] for v in report['development'].values()) and (
        not report['selected'] or report['final']['complete'])
    atomic_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('data/ml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development', type=int, default=96)
    parser.add_argument('--final', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1250000)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(improve(args.campaign.resolve(), args.root.resolve(), args.output.resolve(),
        args.development, args.final, args.seed, args.max_seconds)), indent=2))
