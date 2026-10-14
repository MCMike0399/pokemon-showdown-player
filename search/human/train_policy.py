"""Train the human opponent policy on MPS: which joint action does a ladder player pick?

    .venv/bin/python search/human/train_policy.py --data 'data/ml/human/regmc/replay-*.npz' \
        --live 'data/ml/human/regmc/live-*.npz' --out data/ml/human/models

Each decision is a group of legal joint actions (search/human/extract.py); the
loss is listwise: -log sum_{consistent} softmax(logits), where the consistent set
is the actions matching what was logged (one action, or every action sharing the
known slot's part when the other slot's choice was never revealed). The model is
a small MLP over the acting side's position features and the action features of
search/screenfeat.cjs, so the engine can score it (search/valuenet.cjs `logit`).

Splits are by game (hash), with our own live games as a separate held-out set:
those opponents are exactly the distribution the live search faces. Reports
top-1/top-3 accuracy and NLL against uniform play, and exports an immutable
`human-<sha16>.json` (layers, mean, std) the engine loads like a screening net.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _bucket(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


from search.human.humanfeat_dim import ACTION_DIM, NM, NS  # noqa: E402

BLOCKS = [NM, NS, NM, NS]  # one-hot blocks after the dense action features: move/species per slot


def split_actions(A: np.ndarray):
    """Dense part and the index inside each one-hot block (stored compactly, expanded in the model)."""
    dense = A[:, :ACTION_DIM].astype(np.float32)
    idx = np.zeros((len(A), len(BLOCKS)), np.int64)
    off = ACTION_DIM
    for k, n in enumerate(BLOCKS):
        idx[:, k] = A[:, off:off + n].argmax(1)
        off += n
    return dense, idx


def load(patterns: list[str], split_of, min_rating: int = 0, max_groups: int = 0):
    S, Ad, Ai, rec, y, split, meta = [], [], [], [], [], [], []
    recs = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with np.load(path, allow_pickle=False) as d:
                m = json.loads(str(d['meta']))
                s_, a_, r_, y_ = d['S'], d['A'], d['rec'], d['y']
            keep_rec = np.array([((x.get('rating') or 0) >= min_rating) for x in m], bool)
            if not keep_rec.any():
                continue
            row_keep = keep_rec[r_]
            old_to_new = np.cumsum(keep_rec) - 1
            dense, idx = split_actions(a_[row_keep])
            S.append(np.nan_to_num(s_[keep_rec].astype(np.float32))); Ad.append(np.nan_to_num(dense)); Ai.append(idx); y.append(y_[row_keep])
            rec.append(old_to_new[r_[row_keep]] + recs)
            kept_meta = [x for x, k in zip(m, keep_rec) if k]
            split.extend(split_of(x, path) for x in kept_meta)
            meta.extend(kept_meta)
            recs += int(keep_rec.sum())
            if max_groups and recs >= max_groups:
                break
    if not S:
        return None
    return {'S': np.concatenate(S), 'Ad': np.concatenate(Ad), 'Ai': np.concatenate(Ai),
            'rec': np.concatenate(rec).astype(np.int64), 'y': np.concatenate(y).astype(np.float32),
            'split': np.asarray(split), 'meta': meta}


def merge(a, b):
    off = len(a['S'])
    return {'S': np.concatenate([a['S'], b['S']]), 'Ad': np.concatenate([a['Ad'], b['Ad']]), 'Ai': np.concatenate([a['Ai'], b['Ai']]),
            'rec': np.concatenate([a['rec'], b['rec'] + off]), 'y': np.concatenate([a['y'], b['y']]),
            'split': np.concatenate([a['split'], b['split']]), 'meta': a['meta'] + b['meta']}


def inputs(data, rows):
    """Dense inputs (position + dense action features); one-hot blocks are looked up separately."""
    return np.concatenate([data['S'][data['rec'][rows]], data['Ad'][rows]], axis=1)


def group_bounds(rec: np.ndarray):
    """rec is sorted by construction (rows of a group are contiguous)."""
    starts = np.flatnonzero(np.r_[True, np.diff(rec) != 0])
    ends = np.r_[starts[1:], len(rec)]
    return starts, ends


def evaluate_logits(data, logits: np.ndarray, which: np.ndarray, exact_only: bool = True) -> dict:
    """NLL, top-1, top-3 over groups whose record index is in `which` (bool over records);
by default only decisions whose logged action is fully known."""
    starts, ends = group_bounds(data['rec'])
    nll = top1 = top3 = 0.0
    n = 0
    unif = 0.0
    for s, e in zip(starts, ends):
        r = data['rec'][s]
        if not which[r] or (exact_only and not data['meta'][r].get('exact', True)):
            continue
        z = logits[s:e]
        yy = data['y'][s:e]
        if yy.sum() <= 0:
            continue
        z = z - z.max()
        p = np.exp(z); p /= p.sum()
        mass = float(p[yy > 0].sum())
        nll += -math.log(max(mass, 1e-12))
        unif += -math.log(max(float(yy.sum()) / len(yy), 1e-12))
        order = np.argsort(-p)
        top1 += float(yy[order[0]] > 0)
        top3 += float(yy[order[:3]].max() > 0)
        n += 1
    if not n:
        return {}
    return {'groups': n, 'nll': round(nll / n, 4), 'nll_uniform': round(unif / n, 4),
            'top1': round(top1 / n, 4), 'top3': round(top3 / n, 4)}


class PolicyNet:
    """MLP whose first layer takes dense inputs plus one-hot blocks as embeddings (exported as one dense layer)."""

    def __init__(self, d_dense, hidden, layers, device):
        import torch
        import torch.nn as nn
        self.lin1 = nn.Linear(d_dense, hidden).to(device)
        self.emb = nn.ModuleList([nn.Embedding(n, hidden) for n in BLOCKS]).to(device)
        for e in self.emb:
            nn.init.normal_(e.weight, std=0.05)
        rest = []
        for _ in range(layers - 1):
            rest += [nn.Linear(hidden, hidden), nn.ReLU()]
        rest.append(nn.Linear(hidden, 1))
        self.rest = nn.Sequential(*rest).to(device)
        self.params = list(self.lin1.parameters()) + list(self.emb.parameters()) + list(self.rest.parameters())

    def __call__(self, x_dense, idx):
        import torch
        h = self.lin1(x_dense)
        for k, e in enumerate(self.emb):
            h = h + e(idx[:, k])
        return self.rest(torch.relu(h)).squeeze(-1)

    def state(self):
        return {'lin1': {k: v.detach().cpu().clone() for k, v in self.lin1.state_dict().items()},
                'emb': {k: v.detach().cpu().clone() for k, v in self.emb.state_dict().items()},
                'rest': {k: v.detach().cpu().clone() for k, v in self.rest.state_dict().items()}}

    def load(self, st):
        self.lin1.load_state_dict(st['lin1']); self.emb.load_state_dict(st['emb']); self.rest.load_state_dict(st['rest'])

    def train_mode(self, flag):
        for m in (self.lin1, self.emb, self.rest):
            m.train(flag)

    def export_layers(self, mean, std):
        """One dense first layer over [S, dense A, one-hot blocks]; one-hot columns are unnormalised."""
        W1 = np.concatenate([self.lin1.weight.detach().cpu().numpy()] + [e.weight.detach().cpu().numpy().T for e in self.emb], axis=1)
        layers = [{'W': W1.tolist(), 'b': self.lin1.bias.detach().cpu().tolist()}]
        for m in self.rest:
            if hasattr(m, 'weight'):
                layers.append({'W': m.weight.detach().cpu().tolist(), 'b': m.bias.detach().cpu().tolist()})
        n_onehot = sum(BLOCKS)
        return layers, np.concatenate([mean, np.zeros(n_onehot)]), np.concatenate([std, np.ones(n_onehot)])


def train(data, hidden, epochs, lr, wd, batch_groups, device, seed, patience, layers=2):
    import torch
    torch.manual_seed(seed)
    dev = torch.device(device)
    starts, ends = group_bounds(data['rec'])
    grp_split = data['split'][data['rec'][starts]]
    tr = np.flatnonzero(grp_split == 'train')
    va = np.flatnonzero(grp_split == 'val')
    sample_rows = np.random.default_rng(seed).choice(len(data['y']), min(len(data['y']), 300_000), replace=False)
    Xs = inputs(data, sample_rows)
    mean, std = Xs.mean(0), Xs.std(0) + 1e-3
    S = torch.tensor(data['S'], device=dev); Ad = torch.tensor(data['Ad'], device=dev); Ai = torch.tensor(data['Ai'], device=dev)
    rec = torch.tensor(data['rec'], device=dev); Y = torch.tensor(data['y'], device=dev)
    mu, sd = torch.tensor(mean, device=dev), torch.tensor(std, device=dev)
    st_t = torch.tensor(starts, device=dev); en_t = torch.tensor(ends, device=dev)

    def rows_of(groups):
        lens = (en_t[groups] - st_t[groups])
        gid = torch.repeat_interleave(torch.arange(len(groups), device=dev), lens)
        offs = torch.arange(int(lens.sum().item()), device=dev) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        rows = torch.repeat_interleave(st_t[groups], lens) + offs
        return rows, gid, len(groups)

    def loss_of(groups):
        rows, gid, G = rows_of(groups)
        x = (torch.cat([S[rec[rows]], Ad[rows]], 1) - mu) / sd
        z = net(x, Ai[rows])
        zmax = torch.full((G,), -1e9, device=dev).scatter_reduce(0, gid, z, 'amax')
        ez = torch.exp(z - zmax[gid])
        denom = torch.zeros(G, device=dev).scatter_add(0, gid, ez)
        num = torch.zeros(G, device=dev).scatter_add(0, gid, ez * Y[rows])
        return -(torch.log(num + 1e-12) - torch.log(denom)).mean()

    net = PolicyNet(Xs.shape[1], hidden, layers, dev)
    opt = torch.optim.AdamW(net.params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    tr_t, va_t = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    best, best_state, bad, history = math.inf, None, 0, []
    for ep in range(epochs):
        net.train_mode(True)
        perm = tr_t[torch.randperm(len(tr_t), generator=gen).to(dev)]
        for i in range(0, len(perm), batch_groups):
            loss = loss_of(perm[i:i + batch_groups])
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.train_mode(False)
        with torch.no_grad():
            vl = float(np.mean([loss_of(va_t[i:i + batch_groups]).item() for i in range(0, len(va_t), batch_groups)])) if len(va_t) else float('nan')
        history.append(round(vl, 4))
        if vl < best - 1e-4 or not len(va_t):
            best, bad = vl, 0
            best_state = net.state()
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load(best_state)
    net.train_mode(False)
    layers_out, mean_full, std_full = net.export_layers(mean, std)
    return layers_out, mean_full, std_full, history


def expand(data, rows) -> np.ndarray:
    """Full feature rows [S, dense A, one-hot blocks] as the engine computes them."""
    dense = inputs(data, rows)
    parts = [dense]
    for k, n in enumerate(BLOCKS):
        oh = np.zeros((len(rows), n), np.float32)
        oh[np.arange(len(rows)), data['Ai'][rows, k]] = 1
        parts.append(oh)
    return np.concatenate(parts, axis=1)


def forward(layers, mean, std, X):
    h = (np.asarray(X, np.float64) - mean) / std
    for k, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if k < len(layers) - 1:
            h = np.maximum(h, 0)
    return h[:, 0]


def predict(layers, mean, std, data, batch=32768):
    L = [(np.asarray(l['W']), np.asarray(l['b'])) for l in layers]
    out = np.zeros(len(data['y']), np.float32)
    for i in range(0, len(out), batch):
        rows = np.arange(i, min(len(out), i + batch))
        out[i:i + batch] = forward(L, mean, std, expand(data, rows))
    return out


def export(layers, mean, std, report, out_dir: Path, extra_meta: dict | None = None) -> Path:
    weights = {'layers': layers, 'mean': mean.tolist(), 'std': std.tolist(), 'version': 1, 'kind': 'human-policy'}
    blob = json.dumps({k: weights[k] for k in ('layers', 'mean', 'std', 'version')}, separators=(',', ':')).encode()
    sha = hashlib.sha256(blob).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'human-{sha[:16]}.json'
    if not path.exists():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({**weights, 'sha256': sha, 'meta': {'report': report, **(extra_meta or {}), 'created': time.time()}},
                                  separators=(',', ':')))
        tmp.replace(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', action='append', default=[], help='glob of replay shards')
    ap.add_argument('--live', action='append', default=[], help='glob of our own live-game shards (held out)')
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'ml' / 'human' / 'models')
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--min-rating', type=int, default=0)
    ap.add_argument('--max-groups', type=int, default=0)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch-groups', type=int, default=256)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--live-train-frac', type=float, default=0.0, help='fraction of live games moved into training')
    ap.add_argument('--live-holdout-newest', type=int, default=0, help='always hold out the newest N live games (by room number)')
    ap.add_argument('--replay-half', choices=['A', 'B'], default=None,
                    help='use only one hashed half of the replay games (gate opponent and candidate on disjoint data)')
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()

    def split_replay(m, path):
        if args.replay_half:
            half = 'A' if _bucket('half:' + m['game']) < 0.5 else 'B'
            if half != args.replay_half:
                return 'unused'
        return 'val' if _bucket(m['game']) < args.val_frac else 'train'

    newest: set = set()
    if args.live and args.live_holdout_newest:
        rooms = set()
        for pattern in args.live:
            for path in glob.glob(pattern):
                with np.load(path, allow_pickle=False) as d:
                    rooms.update(x['game'] for x in json.loads(str(d['meta'])))
        def room_no(g):
            import re
            m = re.search(r'-(\d+)', g)
            return int(m.group(1)) if m else 0
        newest = set(sorted(rooms, key=room_no)[-args.live_holdout_newest:])

    def split_live(m, path):
        if m['game'] in newest:
            return 'live'
        b = _bucket('live:' + m['game'])
        return 'train' if b < args.live_train_frac else 'live'

    data = load(args.data, split_replay, args.min_rating, args.max_groups)
    live = load(args.live, split_live, 0) if args.live else None
    if data is None:
        raise SystemExit('no data')
    if live is not None:
        data = merge(data, live)
    counts = {s: int((data['split'] == s).sum()) for s in np.unique(data['split'])}
    device = args.device if args.device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu')
    layers, mean, std, history = train(data, args.hidden, args.epochs, args.lr, args.wd, args.batch_groups, device,
                                       args.seed, args.patience, args.layers)
    logits = predict(layers, mean, std, data)
    report = {'device': device, 'history': history, 'groups': counts, 'rows': int(len(data['y'])),
              'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    for s in counts:
        report[s] = evaluate_logits(data, logits, data['split'] == s)
    report['seconds'] = round(time.time() - t0, 1)
    path = export(layers, mean, std, report, args.out)
    report['path'] = str(path)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    main()
