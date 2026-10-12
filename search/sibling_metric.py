"""Does an evaluation rank sibling positions the way their playouts do?

    .venv/bin/python search/sibling_metric.py GROUPS.jsonl [...] [--value cand.json[:beta]] ... [--json]

A one-turn search compares positions one turn apart from the same root. Log loss
on game outcomes measures something else: whether the evaluation knows who is
winning across games. Groups from `selfplay.cjs --siblings` hold positions that
differ only in one side's action (same reply, same chance stream), each played
out to the end; this scores every evaluation on exactly the search's comparison.

Per evaluation, over sibling pairs with different playout scores:

- `pair_acc`: share of pairs ordered as the playouts order them, weighted by the
  playout difference (score ties count half); 0.5 is chance;
- `top1_regret`: mean playout score lost by picking the evaluation's best sibling
  instead of the playouts' best (0 is perfect);
- a 95% bootstrap interval over groups for `pair_acc`, and for every candidate its
  paired difference to the hand evaluation (`vs_hand`), with the same resamples.

Scores follow the engine (search/engine.cjs `evaluate`): finished positions keep
the hand evaluation's +-30; otherwise (1 - beta) * hand + beta * u, u the net in
hand units, clipped to +-20 (search/valuenet.cjs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_groups(paths):
    groups = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                if len(g['sibs']) >= 2 and len({s['y'] for s in g['sibs']}) > 1:
                    groups.append(g)
    return groups


class Net:
    def __init__(self, path: Path):
        w = json.loads(Path(path).read_text())
        self.layers = [(np.asarray(l['W'], np.float64), np.asarray(l['b'], np.float64)) for l in w['layers']]
        self.mean, self.std = np.asarray(w['mean']), np.asarray(w['std'])
        self.hand = w.get('hand')

    def hand_units(self, X: np.ndarray) -> np.ndarray:
        h = (X - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if k < len(self.layers) - 1:
                h = np.maximum(h, 0)
        z = h[:, 0]
        if not self.hand:
            return 8 * np.tanh(z / 2)
        return np.clip((z / (self.hand.get('temp') or 1) - self.hand['b']) / self.hand['a'], -20, 20)


def scorer(spec: str | None):
    if spec is None:
        return 'hand', None
    path, _, beta = spec.partition(':')
    return f'{Path(path).stem}@{beta or 0.5}', (Net(Path(path)), float(beta or 0.5))


def group_scores(groups, ev):
    """List of (scores, playout y) arrays per group."""
    out = []
    if ev is not None:
        net, beta = ev
        X = np.asarray([s['x'] for g in groups for s in g['sibs']], np.float64)
        U = net.hand_units(X)
    k = 0
    for g in groups:
        h = np.asarray([s['h'] for s in g['sibs']], np.float64)
        y = np.asarray([s['y'] for s in g['sibs']], np.float64)
        ended = np.asarray([s.get('n', 1) == 0 for s in g['sibs']])
        if ev is None:
            sc = h
        else:
            u = U[k:k + len(h)]
            sc = np.where(ended, h, (1 - beta) * h + beta * u)
        k += len(h)
        out.append((sc, y))
    return out


def per_group(scored):
    """(weighted correct, weight, regret) per group."""
    rows = []
    for sc, y in scored:
        correct = weight = 0.0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                dy = y[i] - y[j]
                if dy == 0:
                    continue
                ds = sc[i] - sc[j]
                hit = 0.5 if ds == 0 else float(np.sign(ds) == np.sign(dy))
                correct += abs(dy) * hit
                weight += abs(dy)
        best = np.flatnonzero(sc == sc.max())
        rows.append((correct, weight, y.max() - y[best].mean()))
    return np.asarray(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('groups', nargs='+')
    ap.add_argument('--value', action='append', default=[], help='candidate.json[:beta] (repeatable; beta default 0.5)')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    groups = load_groups(args.groups)
    if not groups:
        sys.exit('no informative groups (all siblings had the same playout score)')
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(groups), size=(args.boot, len(groups)))
    evals = [scorer(None)] + [scorer(v) for v in args.value]
    stats = {name: per_group(group_scores(groups, ev)) for name, ev in evals}

    def acc(rows, sel=None):
        r = rows if sel is None else rows[sel]
        return r[:, 0].sum() / r[:, 1].sum()

    hand_boot = np.array([acc(stats['hand'], s) for s in idx])
    report = {'groups': len(groups), 'siblings': sum(len(g['sibs']) for g in groups), 'evaluations': {}}
    for name, rows in stats.items():
        boot = np.array([acc(rows, s) for s in idx])
        entry = {'pair_acc': round(acc(rows), 4), 'ci95': [round(float(np.quantile(boot, q)), 4) for q in (.025, .975)],
                 'top1_regret': round(float(rows[:, 2].mean()), 4)}
        if name != 'hand':
            diff = boot - hand_boot
            entry['vs_hand'] = round(acc(rows) - acc(stats['hand']), 4)
            entry['vs_hand_ci95'] = [round(float(np.quantile(diff, q)), 4) for q in (.025, .975)]
        report['evaluations'][name] = entry
    if args.json:
        print(json.dumps(report))
        return report
    print(f"{report['groups']} informative groups, {report['siblings']} siblings")
    for name, e in report['evaluations'].items():
        extra = f"  vs hand {e['vs_hand']:+.4f} {e['vs_hand_ci95']}" if 'vs_hand' in e else ''
        print(f"{name:40s} pair_acc {e['pair_acc']:.4f} {e['ci95']}  top1_regret {e['top1_regret']:.4f}{extra}")
    return report


if __name__ == '__main__':
    main()
