"""Sweep the opponent model's alpha (equilibrium weight) and tau (greedy softmax temperature)
offline, on real ladder opponents' actions.

Input: `compare_live.py --dump` files (per world: the equilibrium column strategy `nash`, the
greedy softmax at the run's tau `greedy`, and the real action's indices `mask` in the kept
set). Greedy at temperature t is greedy_run ** (tau_run / t), renormalised, so one engine run
per leaf scale covers every tau. Scores NLL / top-1 / top-3 exactly as compare_live.py does
(a real action dropped by screening scores the floor). Output: one JSON table on stdout.

    .venv/bin/python search/opp_sweep.py artifacts/leaf/opp-linear.jsonl artifacts/leaf/opp-winprob.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search.human.compare_live import score  # noqa: E402


def sweep(rows: list[dict], alphas: list[float], taus: list[float], run_tau: float = 1.0) -> list[dict]:
    out = []
    for a in alphas:
        for t in taus:
            nll = t1 = t3 = 0.0
            for r in rows:
                nash, g = np.asarray(r['nash']), np.asarray(r['greedy'], dtype=float)
                logg = np.log(np.maximum(g, 1e-300)) * (run_tau / t)
                g_t = np.exp(logg - logg.max())
                g_t /= g_t.sum()
                q = a * nash + (1 - a) * g_t
                n, k1, k3 = score(q, r['mask'])
                nll += n; t1 += k1; t3 += k3
            m = max(1, len(rows))
            out.append({'alpha': a, 'tau': t, 'nll': round(nll / m, 4), 'top1': round(t1 / m, 4), 'top3': round(t3 / m, 4), 'n': len(rows)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dumps', nargs='+', type=Path)
    ap.add_argument('--alphas', default='0,0.25,0.5,0.75')
    ap.add_argument('--taus', default='0.25,0.5,1,2,4,8,16')
    ap.add_argument('--run-tau', type=float, default=1.0)
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(',')]
    taus = [float(x) for x in args.taus.split(',')]
    report = {}
    for path in args.dumps:
        rows = [json.loads(line) for line in path.open() if line.strip()]
        table = sweep(rows, alphas, taus, args.run_tau)
        best = min(table, key=lambda r: r['nll'])
        deployed = next(r for r in table if r['alpha'] == 0.5 and r['tau'] == 1.0) if 0.5 in alphas and 1.0 in taus else None
        report[path.stem] = {'best': best, 'deployed': deployed, 'table': table,
                             'uniform_nll': round(sum(-math.log(max(len(r['mask']) / len(r['nash']), 1e-6)) if r['mask'] else
                                                      -math.log(1e-6) for r in rows) / max(1, len(rows)), 4)}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
