"""Reproducible root-policy audit on frozen simulator worlds; not an Elo test.

    python -m search.belief_audit --workload workload.json --out audit-directory

The workload is a JSON list of engine search messages. Each output retains the
input hash and complete root matrices so the mathematical claims are replayable.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

from search.agent import Engine

OPTIONS = {'keep': 24, 'risk': .1, 'iterations': 4000, 'tolerance': .01}


def legacy_scores(results):
    values = {}
    for r in results:
        for a, v in r['values'].items():
            values.setdefault(a, []).append(v)
    return {a: sum(v) / len(v) - (.5 if len(v) < len(results) else 0.) for a, v in values.items()}


def security(matrices, policy):
    return sum(min(sum(policy.get(a, 0) * m['payoff'][i][j] for i, a in enumerate(m['ours']))
                   for j in range(len(m['theirs']))) for m in matrices) / len(matrices)


def quantiles(values):
    values = sorted(values)
    return {name: values[round((len(values) - 1) * q)] for name, q in
            [('min', 0), ('median', .5), ('p95', .95), ('max', 1)]} if values else {}


def summarize(rows):
    good = [r for r in rows if 'error' not in r]
    return {'positions': len(rows), 'good': len(good), 'errors': [r for r in rows if 'error' in r],
            'scope': 'Restricted sampled-game diagnostics; not measured wins or Elo.',
            'incomplete_incumbent_choices': sum(r['chosen_coverage'] < r['worlds'] for r in good),
            'common_actions_before': quantiles([r['common_before'] for r in good]),
            'coverage_changes_best': sum(r['coverage_choice'] != r['incumbent'] for r in good),
            'mixed_positions': sum(max(r['solution']['policy'].values()) < .95 for r in good),
            'security_gain': quantiles([r['solution']['security'] - r['incumbent_security'] for r in good]),
            'model_value_change': quantiles([r['solution']['model_value'] - r['incumbent_model_value'] for r in good]),
            'solver_gap': quantiles([r['solution']['gap'] for r in good]),
            'solve_ms': quantiles([r['solve_ms'] for r in good]),
            'completion_ms': quantiles([r['completion_ms'] for r in good]),
            'search_ms': quantiles([r['search_ms'] for r in good]),
            'simulation_overhead': (sum(r['completion_sims'] for r in good) /
                                    max(1, sum(r['search_sims'] for r in good)))}


async def audit(workload: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    raw = workload.read_bytes()
    protocol = {'workload_sha256': hashlib.sha256(raw).hexdigest(), 'options': OPTIONS,
                'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__), Path(__file__).with_name('engine.cjs'),
                             Path(__file__).with_name('belief.cjs')]}}
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('audit inputs or code changed; use a new output directory')
    protocol_path.write_text(json.dumps(protocol, indent=2))
    (out / 'workload.json').write_bytes(raw)
    rows = []
    engine = Engine()
    try:
        for index, msg in enumerate(json.loads(raw)):
            dest = out / f'{index:03}.json'
            if dest.exists():
                rows.append(json.loads(dest.read_text()))
                continue
            try:
                t = time.perf_counter()
                results = (await engine.call({**msg, 'matrix_detail': True}, timeout=300))['results']
                search_ms = (time.perf_counter() - t) * 1000
                if any('error' in r for r in results):
                    raise RuntimeError(str([r.get('error') for r in results]))
                scores = legacy_scores(results)
                incumbent = max(scores, key=scores.get)
                wanted = sorted(scores, key=scores.get, reverse=True)[:OPTIONS['keep']]
                before = set.intersection(*(set(r['values']) for r in results))
                jobs = [{'source': {**{k: v for k, v in msg.items() if k != 'worlds'},
                                    'world': w, 'world_offset': msg.get('world_offset', 0) + i},
                         'matrix': r['matrix'], 'wanted': wanted}
                        for i, (w, r) in enumerate(zip(msg['worlds'], results))]
                t = time.perf_counter()
                completed = (await engine.call({'type': 'complete_matrix', 'jobs': jobs}, timeout=300))['results']
                completion_ms = (time.perf_counter() - t) * 1000
                common = set(wanted).intersection(*(set(r['matrix']['ours']) for r in completed))
                actions = [a for a in wanted if a in common]
                if incumbent not in common:
                    raise RuntimeError('incumbent not legal in all completed worlds')
                matrices = [{**r['matrix'], 'ours': actions,
                             'payoff': [r['matrix']['payoff'][r['matrix']['ours'].index(a)] for a in actions]}
                            for r in completed]
                t = time.perf_counter()
                solution = await engine.call({'type': 'belief', 'matrices': matrices,
                                              'options': {k: v for k, v in OPTIONS.items() if k != 'keep'}})
                solve_ms = (time.perf_counter() - t) * 1000
                row = {'index': index, 'worlds': len(results), 'incumbent': incumbent,
                       'chosen_coverage': sum(incumbent in r['values'] for r in results),
                       'common_before': len(before), 'solution': solution, 'matrices': matrices,
                       'coverage_choice': max(solution['model_rows'], key=solution['model_rows'].get),
                       'incumbent_security': security(matrices, {incumbent: 1.}),
                       'incumbent_model_value': solution['model_rows'][incumbent],
                       'search_ms': search_ms, 'completion_ms': completion_ms, 'solve_ms': solve_ms,
                       'search_sims': sum(r['sims'] for r in results),
                       'completion_sims': sum(r['sims'] for r in completed)}
            except (RuntimeError, ValueError, KeyError) as exc:
                row = {'index': index, 'error': str(exc)}
            dest.write_text(json.dumps(row))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('matrices', 'solution')}), flush=True)
    finally:
        await engine.close()
    summary = summarize(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workload', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(audit(args.workload, args.out))


if __name__ == '__main__':
    main()
