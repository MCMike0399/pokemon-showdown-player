"""Human team-preview policy: which four does a ladder player bring, and which two lead?

    .venv/bin/python search/human/preview_model.py train --previews data/ml/human/previews-replays.jsonl \
        --out data/ml/human/models-preview

Data: one record per replay (search/human/extract.py preview_record): both sixes,
the leads and the Pokemon each side brought (all four only when the game revealed
them). A plan is (lead pair, back pair) over the player's six: 15 fours x 6 lead
pairs = 90 plans. The label is the plan played; when only the leads (or three of
the four) are known, every consistent plan is a label and the loss marginalises
(same listwise rule as the move policy).

Features of a plan: own six, opposing six, the brought four and the lead pair as
multi-hot vectors over the species vocabulary (search/human/vocab.json), so one
MLP scores any plan of any team against any opponent. Splits are by game.
`PreviewPolicy` scores a live request (own species from the request, opposing
six from the public preview) and samples a 'team i,j,k,l' choice.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VOCAB = json.loads((Path(__file__).with_name('vocab.json')).read_text())
SPECIES = {s.lower().replace('-mega', '').replace(' ', ''): i for i, s in enumerate(VOCAB['species'])}
NS = len(VOCAB['species']) + 1


def _sid(species: str) -> int:
    s = re.sub(r'-Mega(-[XYZ])?$', '', species.split(',')[0].strip()).lower().replace(' ', '')
    s2 = s.replace('-', '')
    for key in (s, s2):
        for k, i in SPECIES.items():
            if k == key or k.replace('-', '') == key:
                return i
    return NS - 1


def multi_hot(ids) -> np.ndarray:
    v = np.zeros(NS, np.float32)
    for i in ids:
        v[i] = 1.0
    return v


def plans(n: int = 6):
    """All (lead pair, back pair) plans as index tuples (lead a, lead b, back c, back d)."""
    out = []
    for four in itertools.combinations(range(n), 4):
        for lead in itertools.combinations(four, 2):
            back = tuple(i for i in four if i not in lead)
            out.append(lead + back)
    return out


PLANS = plans()


def plan_features(own_ids, opp_ids, plan) -> np.ndarray:
    return np.concatenate([multi_hot(own_ids), multi_hot(opp_ids), multi_hot(own_ids[i] for i in plan),
                           multi_hot(own_ids[i] for i in plan[:2])])


def consistent(own: list[str], leads: list[str], brought: list[str]) -> list[int]:
    """Indices of plans consistent with the observed leads and brought set."""
    base = [re.sub(r'-Mega(-[XYZ])?$', '', s) for s in own]
    lead_idx = [base.index(s) for s in leads if s in base]
    brought_idx = [base.index(s) for s in brought if s in base]
    if len(lead_idx) != 2:
        return []
    ok = []
    for k, p in enumerate(PLANS):
        if set(p[:2]) != set(lead_idx):
            continue
        if not set(brought_idx) <= set(p):
            continue
        ok.append(k)
    return ok


def load(path: Path, val_frac: float = 0.1):
    X, Y, grp, split = [], [], [], []
    g = 0
    for line in open(path):
        r = json.loads(line)
        for s in ('p1', 'p2'):
            o = 'p2' if s == 'p1' else 'p1'
            own, opp = r['six'][s], r['six'][o]
            if len(own) != 6 or len(opp) != 6:
                continue
            ok = consistent(own, r['leads'][s], r['brought'][s])
            if not ok:
                continue
            own_ids = [_sid(x) for x in own]
            opp_ids = [_sid(x) for x in opp]
            mask = np.zeros(len(PLANS), np.float32)
            mask[ok] = 1
            X.append(np.stack([plan_features(own_ids, opp_ids, p) for p in PLANS]))
            Y.append(mask)
            grp.append(g)
            b = int(hashlib.sha256(r['game'].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            split.append('val' if b < val_frac else 'train')
            g += 1
    return np.stack(X), np.stack(Y), np.asarray(split)


def train(X, Y, split, hidden=128, epochs=30, lr=1e-3, wd=1e-4, batch=128, device='auto', seed=0, patience=4):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    device = device if device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dev = torch.device(device)
    tr = np.flatnonzero(split == 'train'); va = np.flatnonzero(split == 'val')
    mean = X[tr].reshape(-1, X.shape[-1]).mean(0); std = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-3
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32, device=dev); Yt = torch.tensor(Y, device=dev)
    net = nn.Sequential(nn.Linear(X.shape[-1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    def loss_of(idx):
        z = net(Xt[idx]).squeeze(-1)
        lse = torch.logsumexp(z, 1)
        num = torch.logsumexp(z + torch.log(Yt[idx] + 1e-12), 1)
        return -(num - lse).mean()

    best, state, bad, hist = math.inf, None, 0, []
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm), batch):
            loss = loss_of(torch.tensor(perm[i:i + batch], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_of(torch.tensor(va, device=dev))) if len(va) else float('nan')
        hist.append(round(vl, 4))
        if vl < best - 1e-4 or not len(va):
            best, bad, state = vl, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net = net.cpu().eval()
    with torch.no_grad():
        Z = net(torch.tensor((X - mean) / std, dtype=torch.float32)).squeeze(-1).numpy()
    return net, mean, std, hist, Z


def metrics(Z, Y, idx) -> dict:
    nll = top1 = top3 = lead1 = 0.0
    n = 0
    for i in idx:
        z = Z[i] - Z[i].max(); p = np.exp(z); p /= p.sum()
        mass = float(p[Y[i] > 0].sum())
        nll += -math.log(max(mass, 1e-9))
        order = np.argsort(-p)
        top1 += float(Y[i][order[0]] > 0); top3 += float(Y[i][order[:3]].max() > 0)
        # Lead pair accuracy: probability mass aggregated per lead pair.
        lead_p = {}
        for k, pk in enumerate(p):
            lead_p[PLANS[k][:2]] = lead_p.get(PLANS[k][:2], 0) + pk
        best_lead = max(lead_p, key=lead_p.get)
        true_leads = {PLANS[k][:2] for k in np.flatnonzero(Y[i] > 0)}
        lead1 += float(best_lead in true_leads)
        n += 1
    return {'n': n, 'nll': round(nll / max(1, n), 4), 'nll_uniform_90': round(math.log(90), 4), 'top1_plan': round(top1 / max(1, n), 4),
            'top3_plan': round(top3 / max(1, n), 4), 'top1_lead': round(lead1 / max(1, n), 4), 'uniform_lead': round(1 / 15, 4)}


class PreviewPolicy:
    """Samples a 'team a,b,c,d' choice for a request like a ladder player would."""

    def __init__(self, path, temp: float = 1.0, seed: int = 0):
        import random
        w = json.loads(Path(path).read_text())
        self.layers = [(np.asarray(l['W']), np.asarray(l['b'])) for l in w['layers']]
        self.mean, self.std = np.asarray(w['mean']), np.asarray(w['std'])
        self.temp = temp
        self.rng = random.Random(seed)

    def scores(self, own: list[str], opp: list[str]) -> np.ndarray:
        own_ids = [_sid(x) for x in own]; opp_ids = [_sid(x) for x in opp]
        X = np.stack([plan_features(own_ids, opp_ids, p) for p in PLANS])
        h = (X - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if k < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return h[:, 0]

    def choose(self, request: dict, opp_preview: list[str], legal: list[str], sample: bool = True) -> str:
        own = [p['details'].split(',')[0] for p in request['side']['pokemon']]
        if len(own) != 6 or len(opp_preview) < 4:
            return legal[0]
        z = self.scores(own, opp_preview) / self.temp
        z = z - z.max(); p = np.exp(z); p /= p.sum()
        legal_set = set(legal)
        order = list(np.argsort(-p))
        if sample:
            r, acc = self.rng.random(), 0.0
            pick = order[-1]
            for k in order:
                acc += p[k]
                if r <= acc:
                    pick = k
                    break
            order = [pick] + order
        for k in order:
            choice = 'team ' + ','.join(str(i + 1) for i in PLANS[k])
            if choice in legal_set:
                return choice
        return legal[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['train'])
    ap.add_argument('--previews', type=Path, default=ROOT / 'data' / 'ml' / 'human' / 'previews-replays.jsonl')
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'ml' / 'human' / 'models-preview')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    t0 = time.time()
    X, Y, split = load(args.previews)
    net, mean, std, hist, Z = train(X, Y, split, args.hidden, args.epochs, args.lr, args.wd, seed=args.seed)
    report = {'decisions': int(len(Y)), 'history': hist, 'exact': int((Y.sum(1) == 1).sum())}
    for s in ('train', 'val'):
        idx = np.flatnonzero(split == s)
        report[s] = metrics(Z, Y, idx)
        report[s + '_exact'] = metrics(Z, Y, [i for i in idx if Y[i].sum() == 1])
    layers = [m for m in net if hasattr(m, 'weight')]
    weights = {'layers': [{'W': l.weight.detach().tolist(), 'b': l.bias.detach().tolist()} for l in layers],
               'mean': mean.tolist(), 'std': std.tolist(), 'version': 1, 'kind': 'human-preview'}
    blob = json.dumps({k: weights[k] for k in ('layers', 'mean', 'std', 'version')}, separators=(',', ':')).encode()
    sha = hashlib.sha256(blob).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'preview-{sha[:16]}.json'
    report['seconds'] = round(time.time() - t0, 1)
    path.write_text(json.dumps({**weights, 'sha256': sha, 'meta': {'report': report, 'created': time.time()}}, separators=(',', ':')))
    report['path'] = str(path)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
