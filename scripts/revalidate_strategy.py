"""Re-execute a preselected strategy's whole final panel after a code correction.

Immutable actors and frozen inputs are reused; no results cross generations.
The final cases were reserved independently of development selection. Their
entire base/candidate panels execute again under the declared current source.
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.continuous import LearningConfig
from ml.promotion import paired_gate
from ml.reload import source_generation
from ml.resources import ResourcePolicy
from ml.worker import atomic_json, parallel_games


async def run(root, original, output, selected, capacity, seconds):
    original_plan = json.loads((original / 'plan.json').read_text())
    development = json.loads((original / 'report.json').read_text())['development']
    if not development.get(selected, {}).get('passed'):
        raise ValueError('candidate must have passed its declared development gate')
    fmt = original_plan['format']
    config = LearningConfig.load(root)
    policy = ResourcePolicy(**config.resource)
    if capacity:
        policy.max_workers = capacity
    policy._simulator_cap = 1
    policy.apply_background()
    output.mkdir(parents=True, exist_ok=True)
    LearningConfig(**{**config.__dict__, 'enabled': True}).save(output / 'evaluation')
    path = output / 'plan.json'
    if not path.exists():
        atomic_json(path, {**original_plan, 'source_generation': source_generation(),
                          'previous_source_generation': original_plan['source_generation'],
                          'selected': selected, 'revalidation': 'Entire original final panel under corrected source, no mixed-generation results.',
                          'original': str(original)})
    plan = json.loads(path.read_text())
    if plan['source_generation'] != source_generation():
        raise ValueError('revalidation source changed')
    for label, digest in [('incumbent', plan['parent_sha256']), (selected, plan[selected + '_sha256'])]:
        checkpoint = original / label / 'models' / (fmt + '.pt')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
            raise ValueError('immutable checkpoint changed')
    deadline = time.monotonic() + seconds
    panels = {}
    for label in ('incumbent', selected):
        path = output / ('final-' + label + '.json')
        rows = json.loads(path.read_text())['games'] if path.exists() else []
        tasks = []
        cycle = ('self', 'tactical', 'heuristic', 'self', 'tactical', 'heuristic', 'random', 'tactical')
        for i in range(plan['final']):
            opponent = cycle[(i // len(plan['opponents'])) % len(cycle)]
            tasks.append({'root': str(output / 'evaluation'), 'simulator_slots_root': str(root),
                'inference_root': str(original / 'inputs'), 'checkpoint_root': str(original / label),
                'source_generation': plan['source_generation'], 'format': fmt,
                'team1': plan['team'], 'team2': plan['opponents'][i % len(plan['opponents'])],
                'seed': plan['seed'] + 100000 + i, 'seed_encoding': 'full-v1',
                'side': 'p2' if (i + i // len(plan['opponents'])) % 2 else 'p1',
                'training': False, 'sample_actions': True, 'open_team_sheets': i % 12 == 0,
                'opponent': opponent, 'opponent_checkpoint_root': str(original / 'incumbent') if opponent == 'self' else None})
        while len(rows) < plan['final'] and time.monotonic() < deadline:
            if plan['source_generation'] != source_generation():
                raise ValueError('source changed during final revalidation')
            result = await parallel_games(tasks[len(rows):], policy, min(deadline, time.monotonic() + 30))
            rows.extend(result['games']);atomic_json(path, {'games': rows})
            status = {'phase': 'final', 'policy': label, 'completed': len(rows), 'target': plan['final'], 'deferred': result.get('deferred')}
            atomic_json(output / 'status.json', status);print(json.dumps(status), flush=True)
            if any(g.get('unfinished') or g.get('rejected_actions') for g in rows):
                raise ValueError('unclean simulation; preserve evidence')
            if len(rows) < plan['final']:
                await asyncio.sleep(5 if result.get('deferred') else .1)
        panels[label] = rows
    final = paired_gate(panels['incumbent'], panels[selected], plan['final'], plan['margin'])
    report = {'source_generation': plan['source_generation'], 'parent_revision': plan['parent_revision'],
              'selected': selected, 'final': final, 'complete': final['complete'], 'passed': final['passed'],
              'promoted': False, 'candidate_checkpoint': str(original / selected / 'models' / (fmt + '.pt')),
              'candidate_sha256': plan[selected + '_sha256'], 'prior_development_source': plan['previous_source_generation']}
    atomic_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True);p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True);p.add_argument('--selected', default='strategic-v1')
    p.add_argument('--capacity', type=int, default=0);p.add_argument('--seconds', type=int, default=3600)
    a = p.parse_args()
    if a.capacity < 0 or a.capacity > 4 or not 30 <= a.seconds <= 7200:
        p.error('capacity0..4/seconds30..7200 required')
    print(json.dumps(asyncio.run(run(a.root.resolve(), a.original.resolve(), a.output.resolve(), a.selected, a.capacity, a.seconds)), indent=2))
