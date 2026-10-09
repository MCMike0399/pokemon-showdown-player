"""Resumable strategic-prior ablations and unopened final paired validation.

Reads production checkpoints/evidence; uses only isolated official simulations.
No account, source deployment, promotion or production PPO consumption occurs.
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
from ml.resources import ResourcePolicy
from ml.storage import now
from ml.worker import atomic_json, candidate_teams, freeze_inputs, parallel_games, snapshot


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(root, output, knowledge_path, development, final, seed, seconds, profiles):
    from harness import TeamStore
    from ml.simulator import validate_team, load_dex
    from ml.teams import team_id
    fmt = 'gen9championsvgc2026regmc'
    config = LearningConfig.load(root)
    resource = ResourcePolicy(**config.resource)
    resource._simulator_cap = 1
    resource.apply_background()
    deadline = time.monotonic() + seconds
    output.mkdir(parents=True, exist_ok=True)
    isolated = LearningConfig(**{**config.__dict__, 'enabled': True})
    isolated.save(output / 'evaluation')
    plan_path = output / 'plan.json'
    if not plan_path.exists():
        parent = root / 'models' / (fmt + '.pt')
        snapshot(parent, output / 'incumbent', fmt)
        incumbent = Model(output / 'incumbent', fmt)
        team = TeamStore().get('Rain-Recife-special-stat-fix')['sets']
        db = sqlite3.connect((root / 'experience.sqlite3').as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            source = SimpleNamespace(root=root, db=db)
            freeze_inputs(source, output / 'inputs', fmt)
            opponents = [sets for sets in candidate_teams(source, fmt) if not (await validate_team(fmt, sets))['errors']]
        finally:
            db.close()
        if not opponents or (await validate_team(fmt, team))['errors']:
            raise ValueError('legal focus and opponent pool required')
        features = await load_dex(fmt)
        knowledge = json.loads(knowledge_path.read_text())
        if knowledge.get('format') != fmt:
            raise ValueError('behavior knowledge exact-format mismatch')
        plan = {'declared_at': now(), 'source_generation': source_generation(), 'format': fmt,
                'parent_revision': incumbent.revision, 'parent_sha256': sha(incumbent.path),
                'team': team, 'team_fingerprint': team_id(fmt, team), 'opponents': opponents,
                'knowledge_sha256': sha(knowledge_path), 'dex_sha256': features.signature(),
                'development': development, 'final': final, 'seed': seed, 'profiles': profiles,
                'margin': .1, 'selection': 'Highest development wins among passing candidates; one unopened independent final panel.',
                'hypotheses': ['Joint initiative/denial improves conversion of chosen actions into execution.',
                               'Matchup opening and field/entry plans improve four-Pokemon cohesion.',
                               'Executed opponent behavior is statistical evidence, never expert/PPO relabeling.']}
        for profile in profiles:
            snapshot(incumbent.path, output / profile, fmt)
            model = Model(output / profile, fmt)
            model.feature_profile = profile
            model.preview_temperature = .25 if profile != 'strategic-turn-v1' else incumbent.preview_temperature
            model.strategy_knowledge = knowledge
            model.revision = uuid.uuid4().hex
            model.save()
            plan[profile + '_sha256'] = sha(model.path)
        atomic_json(plan_path, plan)
    plan = json.loads(plan_path.read_text())
    if source_generation() != plan['source_generation']:
        raise ValueError('source changed; preserve results and declare a fresh experiment')
    for label in ['incumbent'] + plan['profiles']:
        expected = plan['parent_sha256'] if label == 'incumbent' else plan[label + '_sha256']
        if sha(output / label / 'models' / (fmt + '.pt')) != expected:
            raise ValueError('immutable experiment checkpoint changed')

    async def panel(label, phase, count, offset):
        path = output / (phase + '-' + label + '.json')
        games = json.loads(path.read_text())['games'] if path.exists() else []
        cycle = ('self', 'tactical', 'heuristic', 'self', 'tactical', 'heuristic', 'random', 'tactical')
        tasks = []
        for i in range(count):
            opponent = cycle[(i // len(plan['opponents'])) % len(cycle)]
            tasks.append({'root': str(output / 'evaluation'), 'simulator_slots_root': str(root),
                'inference_root': str(output / 'inputs'), 'checkpoint_root': str(output / label),
                'source_generation': plan['source_generation'], 'format': fmt, 'team1': plan['team'],
                'team2': plan['opponents'][i % len(plan['opponents'])], 'seed': plan['seed'] + offset + i,
                'seed_encoding': 'full-v1', 'side': 'p2' if (i + i // len(plan['opponents'])) % 2 else 'p1',
                'training': False, 'sample_actions': True, 'open_team_sheets': i % 12 == 0,
                'opponent': opponent, 'opponent_checkpoint_root': str(output / 'incumbent') if opponent == 'self' else None})
        while len(games) < count and time.monotonic() < deadline:
            if source_generation() != plan['source_generation']:
                raise ValueError('experiment source changed during panel')
            result = await parallel_games(tasks[len(games):], resource, min(deadline, time.monotonic() + 30))
            games.extend(result['games'])
            atomic_json(path, {'games': games})
            status = {'phase': phase, 'policy': label, 'completed': len(games), 'target': count,
                      'deferred': result.get('deferred'), 'updated_at': now()}
            atomic_json(output / 'status.json', status)
            print(json.dumps(status), flush=True)
            if any(g.get('unfinished') or g.get('rejected_actions') for g in games):
                raise ValueError('unclean simulation; inspect preserved evidence')
            if len(games) < count:
                await asyncio.sleep(5 if result.get('deferred') else .1)
        return games

    base = await panel('incumbent', 'development', plan['development'], 0)
    report = {'source_generation': plan['source_generation'], 'parent_revision': plan['parent_revision'],
              'development': {}, 'promoted': False}
    for label in plan['profiles']:
        candidate = await panel(label, 'development', plan['development'], 0)
        report['development'][label] = paired_gate(base, candidate, plan['development'], plan['margin'])
        atomic_json(output / 'report.json', report)
    if not all(g['complete'] for g in report['development'].values()):
        report.update(complete=False, passed=False, reason='development pending')
    else:
        passed = [label for label, gate in report['development'].items() if gate['passed']]
        if not passed:
            report.update(complete=True, passed=False, final_untouched=True)
        else:
            label = max(passed, key=lambda x: report['development'][x]['candidate_wins'])
            base = await panel('incumbent', 'final', plan['final'], 100000)
            candidate = await panel(label, 'final', plan['final'], 100000)
            report['selected'] = label
            report['final'] = paired_gate(base, candidate, plan['final'], plan['margin'])
            report.update(complete=report['final']['complete'], passed=report['final']['passed'],
                          candidate_checkpoint=str(output / label / 'models' / (fmt + '.pt')),
                          candidate_sha256=plan[label + '_sha256'])
    atomic_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--knowledge', type=Path, required=True)
    p.add_argument('--development', type=int, default=64)
    p.add_argument('--final', type=int, default=160)
    p.add_argument('--seed', type=int, default=3100000)
    p.add_argument('--seconds', type=int, default=3600)
    p.add_argument('--profiles', nargs='+', choices=['strategic-v1', 'strategic-turn-v1', 'strategic-preview-v1'], default=['strategic-v1', 'strategic-turn-v1', 'strategic-preview-v1'])
    a = p.parse_args()
    if not 16 <= a.development <= 1000 or not 64 <= a.final <= 1000 or not 30 <= a.seconds <= 7200:
        p.error('development16..1000/final64..1000/seconds30..7200 required')
    print(json.dumps(asyncio.run(run(a.root.resolve(), a.output.resolve(), a.knowledge.resolve(),
                     a.development, a.final, a.seed, a.seconds, a.profiles)), indent=2))
