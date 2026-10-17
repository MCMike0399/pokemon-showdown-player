"""Matchup value from human games: P(win | own six, opposing six, brought four, lead pair).

    .venv/bin/python search/human/preview_value.py train [--previews ...] [--out data/ml/human/models-preview]
    .venv/bin/python search/human/preview_value.py score --model P --team Rain-Recife-special-stat-fix [--games 400]

Trained on replay previews whose brought four are known (the result is the label),
with the preview policy's plan features (multi-hot species of own six, opposing
six, brought four, lead pair). It is a human-population matchup estimate, not
our agent's win rate; `score` reports, for the opponents our own games met, the
estimated win probability of the plan our actor chose versus the best plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from search.human.preview_model import PLANS, _sid, consistent, plan_features  # noqa: E402


def load(path: Path, val_frac: float = 0.15):
    X, Y, split, meta = [], [], [], []
    for line in open(path):
        r = json.loads(line)
        if r.get('winner') is None:
            continue
        for s in ('p1', 'p2'):
            o = 'p2' if s == 'p1' else 'p1'
            own, opp = r['six'][s], r['six'][o]
            if len(own) != 6 or len(opp) != 6 or len(r['brought'][s]) != 4:
                continue
            ok = consistent(own, r['leads'][s], r['brought'][s])
            if len(ok) != 1:
                continue
            X.append(plan_features([_sid(x) for x in own], [_sid(x) for x in opp], PLANS[ok[0]]))
            Y.append(float(r['winner'] == r['names'][s]))
            b = int(hashlib.sha256(r['game'].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            split.append('val' if b < val_frac else 'train')
            meta.append({'game': r['game'], 'side': s})
    return np.stack(X).astype(np.float32), np.asarray(Y, np.float32), np.asarray(split), meta


def train(X, Y, split, hidden=64, epochs=40, lr=1e-3, wd=1e-3, batch=256, seed=0, patience=5):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr = np.flatnonzero(split == 'train'); va = np.flatnonzero(split == 'val')
    Xt = torch.tensor(X, device=dev); Yt = torch.tensor(Y, device=dev)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    best, state, bad, hist = math.inf, None, 0, []
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm), batch):
            idx = torch.tensor(perm[i:i + batch], device=dev)
            loss = bce(net(Xt[idx]).squeeze(-1), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(bce(net(Xt[va]).squeeze(-1), Yt[va]))
        hist.append(round(vl, 4))
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net = net.cpu().eval()
    with torch.no_grad():
        P = torch.sigmoid(net(torch.tensor(X)).squeeze(-1)).numpy()
    return net, hist, P


def auc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


class PreviewValue:
    """One net ({'layers'}) or an ensemble ({'members': [{'layers'}, ...]}, probabilities
    averaged). Version 2 nets take `rating_features` extra inputs (rating difference and
    own rating, both 0 = equal 1300 players at inference), so the matchup estimate is not
    confounded by who tends to pick a plan."""

    def __init__(self, path):
        w = json.loads(Path(path).read_text())
        members = w.get('members') or [w]
        self.nets = [[(np.asarray(l['W']), np.asarray(l['b'])) for l in m['layers']] for m in members]
        self.extra = int(w.get('rating_features', 0))
        self.layers = self.nets[0]

    def probs(self, own: list[str], opp: list[str]) -> np.ndarray:
        own_ids = [_sid(x) for x in own]; opp_ids = [_sid(x) for x in opp]
        x = np.stack([plan_features(own_ids, opp_ids, p) for p in PLANS]).astype(np.float64)
        if self.extra:
            x = np.concatenate([x, np.zeros((len(PLANS), self.extra))], 1)
        out = np.zeros(len(PLANS))
        for layers in self.nets:
            h = x
            for k, (W, b) in enumerate(layers):
                h = h @ W.T + b
                if k < len(layers) - 1:
                    h = np.maximum(h, 0)
            out += 1 / (1 + np.exp(-h[:, 0]))
        return out / len(self.nets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['train', 'score'])
    ap.add_argument('--previews', type=Path, default=ROOT / 'data' / 'ml' / 'human' / 'previews-replays.jsonl')
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'ml' / 'human' / 'models-preview')
    ap.add_argument('--model', type=Path)
    ap.add_argument('--team', default='Rain-Recife-special-stat-fix')
    ap.add_argument('--games', type=int, default=400)
    args = ap.parse_args()
    if args.cmd == 'train':
        t0 = time.time()
        X, Y, split, meta = load(args.previews)
        net, hist, P = train(X, Y, split)
        va = split == 'val'
        base = float(Y[split == 'train'].mean())
        ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))  # noqa: E731
        report = {'rows': int(len(Y)), 'history': hist, 'val': {'n': int(va.sum()), 'logloss': round(ll(P[va], Y[va]), 4),
                  'logloss_base': round(ll(np.full(va.sum(), base), Y[va]), 4), 'auc': round(auc(P[va], Y[va]), 4)},
                  'seconds': round(time.time() - t0, 1)}
        layers = [m for m in net if hasattr(m, 'weight')]
        weights = {'layers': [{'W': l.weight.detach().tolist(), 'b': l.bias.detach().tolist()} for l in layers], 'version': 1,
                   'kind': 'human-preview-value'}
        sha = hashlib.sha256(json.dumps(weights, separators=(',', ':')).encode()).hexdigest()
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f'pvalue-{sha[:16]}.json'
        path.write_text(json.dumps({**weights, 'sha256': sha, 'meta': {'report': report}}, separators=(',', ':')))
        report['path'] = str(path)
        print(json.dumps(report))
    else:
        pv = PreviewValue(args.model)
        team = json.loads((ROOT / 'teams.json').read_text())[args.team]['sets']
        own = [s['species'] for s in team]
        games = [json.loads(l) for l in open(ROOT / 'data' / 'ml' / 'value' / 'teams' / 'rain-recife-special-stat-fix' / 'games.jsonl')][-args.games:]
        usual = [i for i, s in enumerate(own) if s in ('Archaludon', 'Politoed')] + [i for i, s in enumerate(own) if s in ('Incineroar', 'Golisopod')]
        usual_plan = next(k for k, p in enumerate(PLANS) if set(p[:2]) == set(usual[:2]) and set(p) == set(usual))
        gains, best_counts, won_usual = [], {}, []
        for g in games:
            opp = g.get('opp_preview') or []
            if len(opp) != 6:
                continue
            p = pv.probs(own, opp)
            k = int(np.argmax(p))
            gains.append(float(p[k] - p[usual_plan]))
            lead = tuple(sorted(own[i] for i in PLANS[k][:2]))
            best_counts[lead] = best_counts.get(lead, 0) + 1
            won_usual.append((float(p[usual_plan]), g.get('win')))
        pu = np.asarray([w[0] for w in won_usual]); yu = np.asarray([float(w[1]) for w in won_usual])
        print(json.dumps({'games': len(gains), 'mean_est_gain_best_vs_usual': round(float(np.mean(gains)), 4),
                          'usual_plan_est_mean': round(float(pu.mean()), 4), 'usual_plan_actual_winrate': round(float(yu.mean()), 4),
                          'auc_usual_est_vs_actual': round(auc(pu, yu), 4),
                          'best_leads': sorted(best_counts.items(), key=lambda kv: -kv[1])[:8]}))


if __name__ == '__main__':
    main()
