"""Resumable isolated opening-profile PPO and paired development/final evaluation.

Uses no player connection. Production data/checkpoints are read, never consumed
or rewritten. A passing report can be staged separately at the live boundary.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.continuous import LearningConfig
from ml.model import Model
from ml.promotion import paired_gate
from ml.reload import source_generation
from ml.resources import BackgroundGuard, ResourceDeferred, ResourcePolicy
from ml.storage import Store, now
from ml.worker import atomic_json, candidate_teams, freeze_inputs, parallel_games, snapshot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(root, output, fmt, focus, practice, development, final, seed, seconds,
              rounds=1, matchups=False, initial=None, profile='opening-v1'):
    from harness import TeamStore
    from ml.simulator import validate_team, load_dex
    from ml.teams import team_id
    config = LearningConfig.load(root)
    policy = ResourcePolicy(**config.resource)
    # Serialize this experiment's recordings; production uses the other shared slots.
    policy._simulator_cap = 1
    policy.apply_background()
    deadline = time.monotonic() + seconds
    output.mkdir(parents=True, exist_ok=True)
    isolated_config = LearningConfig(**{**config.__dict__, 'enabled': True})
    isolated_config.save(output / 'experience')
    isolated_config.save(output / 'evaluation')
    plan_path = output / 'plan.json'
    if not plan_path.exists():
        source = root / 'models' / (fmt + '.pt')
        if not source.exists():
            raise ValueError('existing incumbent required')
        snapshot(source, output / 'incumbent', fmt)
        incumbent = Model(output / 'incumbent', fmt)
        team = TeamStore().get(focus)
        if team['format'] != fmt or (await validate_team(fmt, team['sets']))['errors']:
            raise ValueError('invalid focused team')
        db = sqlite3.connect((root / 'experience.sqlite3').as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            live = SimpleNamespace(root=root, db=db)
            freeze_inputs(live, output / 'inputs', fmt)
            teams = []
            for sets in candidate_teams(live, fmt):
                if not (await validate_team(fmt, sets))['errors']:
                    teams.append(sets)
        finally:
            db.close()
        if not teams:
            raise ValueError('no legal frozen opponent pool')
        features = await load_dex(fmt)
        plan = {'declared_at': now(), 'format': fmt, 'focus': focus, 'team': team['sets'],
                'team_fingerprint': team_id(fmt, team['sets']), 'opponents': teams,
                'parent_revision': incumbent.revision, 'parent_sha256': digest(incumbent.path),
                'source_generation': source_generation(), 'dex_sha256': features.signature(),
                'practice': practice, 'development': development, 'final': final, 'seed': seed,
                'rounds': rounds, 'matchups': matchups, 'profile': profile,
                'initial_checkpoint_sha256': digest(initial) if initial else None,
                'margin': .1, 'selection': 'Development-passing trained candidate; independent final gate before staging.',
                'notes': ['New-profile stochastic own rollouts only, no legacy PPO relabeling.',
                          'Counterfactual prior and Mega-only development ablations do not open final partitions.',
                          'Seeds, sides, research/scout, team pool and opponents are frozen.']}
        for name, feature_profile in [('policy', profile), ('mega', 'mega-v1')]:
            snapshot(initial if initial and name == 'policy' else incumbent.path, output / name, fmt)
            model = Model(output / name, fmt)
            if matchups and name == 'policy':
                model.enable_matchups()
                model.preview_training_weight = 4
            model.feature_profile = feature_profile
            model.preview_temperature = .35 if feature_profile.startswith('opening-') else incumbent.preview_temperature
            model.revision = uuid.uuid4().hex
            model.save()
            plan[name + '_sha256'] = digest(model.path)
        atomic_json(plan_path, plan)
    plan = json.loads(plan_path.read_text())
    if plan['source_generation'] != source_generation():
        raise ValueError('experiment source changed; retain evidence and use fresh output')
    if digest(output / 'incumbent' / 'models' / (fmt + '.pt')) != plan['parent_sha256']:
        raise ValueError('incumbent changed')
    for name in ('policy', 'mega'):
        if digest(output / name / 'models' / (fmt + '.pt')) != plan[name + '_sha256']:
            raise ValueError('frozen policy changed')
    cycle = ('self', 'tactical', 'heuristic', 'self', 'tactical', 'heuristic', 'random', 'tactical')

    def tasks(label, count, offset, training):
        result = []
        for i in range(count):
            opponent = cycle[(i // len(plan['opponents'])) % len(cycle)]
            result.append({'root': str(output / ('experience' if training else 'evaluation')),
                'simulator_slots_root': str(root),
                'inference_root': str(output / 'inputs'), 'checkpoint_root': str(output / label),
                'source_generation': plan['source_generation'], 'format': fmt,
                'team1': plan['team'], 'team2': plan['opponents'][i % len(plan['opponents'])],
                'seed': plan['seed'] + offset + i, 'seed_encoding': 'full-v1',
                'side': 'p2' if (i + i // len(plan['opponents'])) % 2 else 'p1',
                'training': training, 'sample_actions': True, 'open_team_sheets': i % 12 == 0,
                'opponent': opponent, 'opponent_checkpoint_root': str(output / 'incumbent') if opponent == 'self' else None})
        return result

    async def collect(label, phase, count, offset, training=False):
        path = output / (phase + '-' + label + '.json')
        rows = json.loads(path.read_text())['games'] if path.exists() else []
        work = tasks(label, count, offset, training)
        while len(rows) < count and time.monotonic() < deadline:
            if source_generation() != plan['source_generation']:
                raise ValueError('source changed during frozen experiment')
            report = await parallel_games(work[len(rows):], policy, min(deadline, time.monotonic() + 30))
            if report['games']:
                rows.extend(report['games'])
                atomic_json(path, {'games': rows})
            status = {'phase': phase, 'policy': label, 'completed': len(rows), 'target': count,
                      'updated_at': now(), 'deferred': report.get('deferred'), 'resources': report.get('resources')}
            atomic_json(output / 'status.json', status)
            print(json.dumps({k:v for k,v in status.items() if k != 'resources'}), flush=True)
            if any(g.get('unfinished') or g.get('rejected_actions') for g in rows):
                raise ValueError('unclean simulation; preserve evidence and investigate before training/evaluation')
            if len(rows) < count:
                await asyncio.sleep(5 if report.get('deferred') else .1)
        return rows

    # Each update gets fresh episodes from its own immutable collecting checkpoint.
    training_path = output / 'training.json'
    reports = []
    for round_index in range(plan.get('rounds', 1)):
        label = 'policy' if round_index == 0 else 'collecting-' + str(round_index + 1)
        if round_index:
            source = output / ('trained-' + str(round_index)) / 'models' / (fmt + '.pt')
            target = output / label / 'models' / (fmt + '.pt')
            if not target.exists():
                snapshot(source, output / label, fmt)
            if digest(source) != digest(target):
                raise ValueError('collecting round checkpoint changed')
        games = await collect(label, 'practice-' + str(round_index + 1), plan['practice'], round_index * 1000, True)
        if len(games) != plan['practice']:
            return {'complete': False, 'reason': 'practice pending', 'output': str(output)}
        round_path = output / ('training-' + str(round_index + 1) + '.json')
        round_root = output / ('trained-' + str(round_index + 1))
        if round_path.exists():
            report = json.loads(round_path.read_text())
            if digest(round_root / 'models' / (fmt + '.pt')) != report['checkpoint_sha256']:
                raise ValueError('trained round checkpoint changed')
            reports.append(report)
            continue
        while time.monotonic() < deadline:
            guard = BackgroundGuard(policy, deadline)
            try:
                guard()
                snapshot(output / label / 'models' / (fmt + '.pt'), round_root, fmt)
                model = Model(round_root, fmt)
                # The small network benefits from CPU preparation; MPS optimizers stay bounded.
                model.set_device('mps' if policy.backend != 'cpu' and __import__('torch').backends.mps.is_available() else 'cpu')
                store = Store(output / 'experience')
                try:
                    episodes = store.episodes(fmt, model.revision, untrained=False)
                    report = model.train(episodes, epochs=8, duty_fraction=policy.gpu_duty_fraction,
                                         target_kl=.03, deadline=deadline, checkpoint=guard)
                finally:
                    store.close()
                if not report.get('trained'):
                    raise ValueError('new-profile PPO training had no validated data')
                report['checkpoint_sha256'] = digest(model.path)
                atomic_json(round_path, report)
                reports.append(report)
                break
            except ResourceDeferred:
                atomic_json(output / 'status.json', {'phase': 'training', 'deferred': True, 'resources': policy.sample(), 'updated_at': now()})
                await asyncio.sleep(5)
        if not round_path.exists():
            return {'complete': False, 'reason': 'training deferred'}
    snapshot(output / ('trained-' + str(plan.get('rounds', 1))) / 'models' / (fmt + '.pt'), output / 'trained', fmt)
    training = {**reports[-1], 'rounds': reports,
                'total_episodes': sum(r['episodes'] for r in reports), 'total_steps': sum(r['steps'] for r in reports)}
    atomic_json(training_path, training)
    if digest(output / 'trained' / 'models' / (fmt + '.pt')) != training['checkpoint_sha256']:
        raise ValueError('trained candidate changed')
    base = await collect('incumbent', 'development', plan['development'], 10000)
    result = {'parent_revision': plan['parent_revision'], 'source_generation': plan['source_generation'],
              'training': training, 'development': {}, 'promoted': False}
    for label in ('policy', 'mega', 'trained'):
        candidate = await collect(label, 'development', plan['development'], 10000)
        result['development'][label] = paired_gate(base, candidate, plan['development'], plan['margin'])
        atomic_json(output / 'report.json', result)
    if not all(r['complete'] for r in result['development'].values()):
        result.update(complete=False, reason='development pending')
    elif result['development']['trained']['passed']:
        base = await collect('incumbent', 'final', plan['final'], 20000)
        candidate = await collect('trained', 'final', plan['final'], 20000)
        result['final'] = paired_gate(base, candidate, plan['final'], plan['margin'])
        result['complete'] = result['final']['complete']
        result['passed'] = result['final']['passed']
        result['candidate_checkpoint'] = str(output / 'trained' / 'models' / (fmt + '.pt'))
        result['candidate_sha256'] = training['checkpoint_sha256']
    else:
        result.update(complete=True, passed=False, final_untouched=True)
    atomic_json(output / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', default='gen9championsvgc2026regmc')
    parser.add_argument('--focus', default='Rain-Recife-special-stat-fix')
    parser.add_argument('--practice', type=int, default=128)
    parser.add_argument('--development', type=int, default=96)
    parser.add_argument('--final', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1600000)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--matchups', action='store_true')
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--profile', choices=['opening-v1', 'opening-v2'], default='opening-v1')
    args = parser.parse_args()
    if not all(4 <= n <= 1000 for n in (args.practice, args.development, args.final)) or not 10 <= args.seconds <= 7200 or not 1 <= args.rounds <= 8:
        parser.error('panels must be 4..1000 and seconds 10..7200')
    print(json.dumps(asyncio.run(run(args.root.resolve(), args.output.resolve(), args.format, args.focus,
          args.practice, args.development, args.final, args.seed, args.seconds, args.rounds,
          args.matchups, args.initial_checkpoint.resolve() if args.initial_checkpoint else None, args.profile)), indent=2))
