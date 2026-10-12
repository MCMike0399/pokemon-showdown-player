"""Train the search value MLP on self-play and live positions; export an immutable candidate.

    .venv/bin/python search/train_value.py --team Rain-Recife-special-stat-fix
    (defaults: data/ml/value/teams/<team>/{selfplay/*.jsonl, live, candidates}, see search/team_data.py)

Splits are by game (`g`), never by row: adjacent positions of one game share an
outcome, so a row split leaks labels into validation. Self-play games are split
by a hash of `g`. Live games are split by time: the newest `--live-test`
fraction is a held-out test on the deployment distribution, and older live
games join training with `--live-weight`.

Every metric is reported next to the baseline the search uses today, the hand
evaluation calibrated by a 2-parameter logistic fit on the training rows. A
candidate is `useful` only if it beats that baseline's log loss on both the
self-play validation games and the live test games. The candidate file is named
by the SHA-256 of its weights and is never overwritten; a promotion pointer
(`search/value_service.py`) decides what the live search uses.

Runs on MPS when available (the model is small; most time is data loading),
falls back to CPU on any device error.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bucket(g: str) -> float:
    return int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def load_rows(selfplay: list[str], live_dir: Path | None, val_frac: float, live_test: float,
              max_rows: int = 0, live_weight: float = 0.0, live_cal: float = 0.5):
    """Return dict split -> (X, Y, H, games, weights, V, F, K, SRC) with game-level splits.

V is the self-play turn's search value (hand units, NaN when absent), F 1 for the
focus team's rows, K the game's opponent style (0 search, 1 noisy, 2 random) and
SRC 0 self-play / 1 live. Self-play shards are read newest first until
`max_rows` (0 = all), so a long run trains on a recent window of the improving
self-play distribution. `live_sel` (newest half of the non-test live games, a
subset of the calibration or training games) is where competing settings are
compared, so the newest live games stay an untouched test.
"""
    from search.value_store import load_shard_columns
    files = []
    for pattern in selfplay:
        files.extend(glob.glob(pattern))
    files = sorted(set(files), key=lambda f: os.stat(f).st_mtime, reverse=True)
    parts = {k: [] for k in ('train', 'val', 'live_cal', 'live_test')}
    manifest, total, width = [], 0, None
    for path in files:
        if max_rows and total >= max_rows:
            break
        x, g, cols = load_shard_columns(Path(path))
        y, h = cols['y'], cols['h']
        if not len(y):
            continue
        if width is None:
            width = x.shape[1]
        if x.shape[1] != width:
            continue
        manifest.append({'path': os.path.relpath(path, ROOT), 'bytes': os.stat(path).st_size, 'rows': int(len(y))})
        total += len(y)
        val = np.array([_bucket(str(k)) < val_frac for k in g])
        garr = np.asarray(g)
        for split, mask in (('val', val), ('train', ~val)):
            if mask.any():
                parts[split].append((x[mask], y[mask], h[mask], garr[mask].tolist(), np.ones(int(mask.sum()), np.float32),
                                     cols['v'][mask], cols['f'][mask], np.nan_to_num(cols['k'][mask]),
                                     np.zeros(int(mask.sum()), np.float32)))
    live_games = []
    if live_dir and Path(live_dir).exists():
        for path in sorted(Path(live_dir).glob('live-*.jsonl')):
            rows = [json.loads(line) for line in path.open() if line.strip()]
            rows = [r for r in rows if r.get('h') is not None and (width is None or len(r['x']) == width)]
            if rows:
                live_games.append((min(r.get('time') or 0 for r in rows), path, rows))
        live_games.sort(key=lambda item: item[0])
    n_test = int(round(len(live_games) * live_test)) if live_games else 0
    n_old = len(live_games) - n_test
    sel_games = {r['g'] for _, _, rows in live_games[n_old - n_old // 2:n_old] for r in rows}
    for k, (_, path, rows) in enumerate(live_games):
        manifest.append({'path': os.path.relpath(path, ROOT), 'bytes': path.stat().st_size, 'rows': len(rows)})
        test = k >= len(live_games) - n_test
        block = (np.asarray([r['x'] for r in rows], np.float32), np.asarray([r['y'] for r in rows], np.float32),
                 np.asarray([r['h'] for r in rows], np.float32), [r['g'] for r in rows],
                 np.full(len(rows), 1.0 if test else live_weight, np.float32),
                 np.full(len(rows), np.nan, np.float32), np.ones(len(rows), np.float32),
                 np.zeros(len(rows), np.float32), np.ones(len(rows), np.float32))
        # Time order: [oldest: training (if live_weight > 0)] [calibration] [newest: test].
        # With live_weight 0 every non-test game calibrates. Calibration games are
        # never trained on, so the fitted temperature and blend are honest.
        cal_start = int(round(n_old * (1 - live_cal))) if live_weight > 0 else 0
        if test:
            parts['live_test'].append(block)
        elif k >= cal_start:
            parts['live_cal'].append(block)
        else:
            parts['train'].append(block)
    out = {}
    for split, blocks in parts.items():
        if blocks:
            out[split] = tuple([g for b in blocks for g in b[3]] if i == 3 else np.concatenate([b[i] for b in blocks])
                               for i in range(9))
        else:
            out[split] = tuple([] if i == 3 else np.zeros((0, width or 1) if i == 0 else 0, np.float32) for i in range(9))
    # live_sel: rows of the selection games, wherever they fell (training or calibration).
    sel = [(blk, np.array([g in sel_games for g in blk[3]], dtype=bool)) for name in ('train', 'live_cal') for blk in [out[name]]]
    out['live_sel'] = tuple([g for blk, m in sel for g, keep in zip(blk[3], m) if keep] if i == 3
                            else np.concatenate([blk[i][m] for blk, m in sel]) for i in range(9))
    return out, manifest, {'live_games': len(live_games), 'live_test_games': n_test,
                           'live_cal_games': len({g for b in parts['live_cal'] for g in b[3]}), 'selfplay_rows': total}


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    pos, neg = p[y > .5], p[y < .5]
    if not len(pos) or not len(neg):
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(p, y):
    if not len(y):
        return None
    return {'n': int(len(y)), 'logloss': round(logloss(p, y), 5), 'brier': round(float(np.mean((p - y) ** 2)), 5),
            'acc': round(float(np.mean((p > .5) == (y > .5))), 4), 'auc': round(auc(p, y), 4)}


def fit_hand(h, y, steps=400):
    """Logistic calibration p = sigmoid(a*h + b) by Newton's method (2 parameters)."""
    a, b = 0.1, 0.0
    for _ in range(steps):
        z = np.clip(a * h + b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((p - y) * h), np.sum(p - y)])
        w = p * (1 - p) + 1e-9
        H = np.array([[np.sum(w * h * h), np.sum(w * h)], [np.sum(w * h), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


def train(data, hidden, layers, epochs, lr, wd, batch, device, seed, patience):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xtr, Ytr, _, _, Wtr = data['train'][:5]
    Xva, Yva = data['val'][0], data['val'][1]
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-3
    dev = torch.device(device)
    to = lambda a: torch.tensor(a, device=dev)
    xt, yt, wt = to((Xtr - mean) / std), to(Ytr), to(Wtr)
    xv, yv = to((Xva - mean) / std), to(Yva)
    mods, d = [], Xtr.shape[1]
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, 1))
    net = nn.Sequential(*mods).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    lossf = nn.BCEWithLogitsLoss(reduction='none')
    best, best_state, bad, history = math.inf, None, 0, []
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), generator=gen).to(dev)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = (lossf(net(xt[b]).squeeze(-1), yt[b]) * wt[b]).sum() / wt[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(xv).squeeze(-1), yv).mean()) if len(xv) else float('nan')
        history.append(round(vl, 5))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net = net.cpu().eval()
    return net, mean, std, history


def train_distilled(data, k, hidden, layers, epochs, lr, wd, batch, device, seed, patience):
    """K teachers on different seeds, then one student of the same shape fitted to
their mean logit. The engine runs a single MLP; distillation keeps the ensemble's
lower seed variance in that one net."""
    if k <= 1:
        return train(data, hidden, layers, epochs, lr, wd, batch, device, seed, patience) + ({},)
    X = data['train'][0]
    logits, teachers = [], []
    for i in range(k):
        net, mean, std, history = train(data, hidden, layers, epochs, lr, wd, batch, device, seed + 1000 * (i + 1), patience)
        p = predict(net, mean, std, X)
        logits.append(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)))
        teachers.append({'seed': seed + 1000 * (i + 1), 'epochs': len(history), 'best_val': min(history)})
    soft = (1 / (1 + np.exp(-np.mean(logits, axis=0)))).astype(np.float32)
    student = dict(data)
    student['train'] = (X, soft, *data['train'][2:])
    net, mean, std, history = train(student, hidden, layers, epochs, lr, wd, batch, device, seed, patience)
    return net, mean, std, history, {'teachers': teachers}


def predict(net, mean, std, X):
    import torch
    if not len(X):
        return np.zeros(0, np.float32)
    with torch.no_grad():
        z = net(torch.tensor((X - mean) / std)).squeeze(-1).numpy()
    return 1 / (1 + np.exp(-z))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selfplay', action='append', default=[], help='glob of self-play shards (repeatable)')
    ap.add_argument('--team', default='Rain-Recife-special-stat-fix',
                    help='team whose data is used when --selfplay/--live/--out are not given')
    ap.add_argument('--live', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--live-test', type=float, default=0.4, help='newest fraction of live games held out')
    ap.add_argument('--live-weight', type=float, default=0.0, help='weight of the oldest live games in training')
    ap.add_argument('--live-cal', type=float, default=0.5,
                    help='with --live-weight > 0: newest fraction of non-test live games kept for calibration')
    ap.add_argument('--max-rows', type=int, default=1_500_000, help='newest self-play rows to use (0 = all)')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--ensemble', type=int, default=1, help='distill this many seeds into the exported net')
    ap.add_argument('--max-style', type=int, default=2,
                    help='drop self-play games whose opponent style exceeds this (0 search only, 1 + noisy, 2 all)')
    ap.add_argument('--focus-weight', type=float, default=1.0, help='training weight of the focus team\'s self-play rows')
    ap.add_argument('--target-mix', type=float, default=0.0,
                    help='self-play target (1-L)*outcome + L*calibrated search value of the turn (search-vs-search rows)')
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    if not (args.selfplay and args.live and args.out):
        from search.team_data import team_dir
        tdir = team_dir(args.team)  # data/ml/value/teams/<slug>/
        args.selfplay = args.selfplay or [str(tdir / 'selfplay' / '*.jsonl')]
        args.live = args.live or tdir / 'live'
        args.out = args.out or tdir / 'candidates'
    started = time.time()
    data, manifest, live_info = load_rows(args.selfplay, args.live, args.val_frac, args.live_test,
                                          args.max_rows, args.live_weight, args.live_cal)
    for split in ('train', 'val'):
        keep = data[split][7] <= args.max_style
        if not keep.all():
            data[split] = tuple([g for g, m in zip(data[split][3], keep) if m] if i == 3 else data[split][i][keep]
                                for i in range(9))
    train_data = dict(data)
    X, Y, H, G, W, V, F, K, SRC = data['train']
    W = W * np.where((SRC == 0) & (F == 1), args.focus_weight, 1.0).astype(np.float32)
    mixed = 0
    if args.target_mix > 0:
        # The turn's equilibrium value bootstraps a lower-variance target; it is on
        # the hand scale (self-play ran the hand evaluation), so calibrate it the same way.
        ah, bh = fit_hand(H, Y)
        use = (SRC == 0) & (K == 0) & np.isfinite(V)
        pv = 1 / (1 + np.exp(-np.clip(ah * np.nan_to_num(V) + bh, -30, 30)))
        Y = np.where(use, (1 - args.target_mix) * Y + args.target_mix * pv, Y).astype(np.float32)
        mixed = int(use.sum())
    train_data['train'] = (X, Y, H, G, W, V, F, K, SRC)
    if len(data['train'][0]) < 1000:
        raise SystemExit(json.dumps({'error': 'not enough training rows', 'rows': int(len(data['train'][0]))}))
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        try:
            torch.mps.set_per_process_memory_fraction(0.25)
        except Exception:
            pass
    try:
        net, mean, std, history, ensemble = train_distilled(train_data, args.ensemble, args.hidden, args.layers,
                                                            args.epochs, args.lr, args.wd, args.batch, device,
                                                            args.seed, args.patience)
    except RuntimeError as error:
        if device == 'cpu':
            raise
        print(json.dumps({'device_error': repr(error)[:300], 'fallback': 'cpu'}), file=sys.stderr)
        device = 'cpu'
        net, mean, std, history, ensemble = train_distilled(train_data, args.ensemble, args.hidden, args.layers,
                                                            args.epochs, args.lr, args.wd, args.batch, device,
                                                            args.seed, args.patience)
    a, b = fit_hand(data['train'][2], data['train'][1])
    report = {'device': device, 'seconds': None, 'history': history, **ensemble, 'hand_calibration': {'a': a, 'b': b},
              'rows': {k: int(len(v[1])) for k, v in data.items()},
              'games': {k: len(set(v[3])) for k, v in data.items()}, 'target_mixed_rows': mixed, **live_info}
    for split in ('val', 'live_sel', 'live_test'):
        X, Y, H = data[split][0], data[split][1], data[split][2]
        report[split] = {'learned': metrics(predict(net, mean, std, X), Y),
                         'hand': metrics(1 / (1 + np.exp(-np.clip(a * H + b, -30, 30))), Y)}
    # Self-play is omniscient and more decisive than ladder games, so the net is
    # overconfident live. Fit a temperature T on the older live games (z -> z/T),
    # then choose the blend weight beta there too: the engine plays
    # v = (1-beta)*hand + beta*u, i.e. probability sigmoid((1-beta)*(a*h+b) + beta*z/T).
    # The newest live games (live_test) are never used for any choice.
    def logit_of(split):
        p = predict(net, mean, std, data[split][0])
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))

    def blend(split, beta, temp):
        H = data[split][2]
        return 1 / (1 + np.exp(-np.clip((1 - beta) * (a * H + b) + beta * logit_of(split) / temp, -30, 30)))

    cal = 'live_cal' if len(set(data['live_cal'][3])) >= 10 else 'val'
    temp = 1.0
    if cal == 'live_cal':
        zc, yc = logit_of('live_cal'), data['live_cal'][1]
        grid = np.exp(np.linspace(np.log(0.5), np.log(8.0), 80))
        temp = float(min(grid, key=lambda t: logloss(1 / (1 + np.exp(-zc / t)), yc)))
    betas = (0.0, 0.25, 0.5, 0.75, 1.0)
    scan = {split: {str(beta): round(logloss(blend(split, beta, temp), data[split][1]), 5) for beta in betas}
            for split in ('val', 'live_cal', 'live_sel', 'live_test') if len(data[split][1])}
    beta = float(min(scan[cal], key=scan[cal].get))
    report.update(temperature=round(temp, 4), calibration_split=cal, beta_scan=scan, recommended_beta=beta)
    beats = []
    for split in ('val', 'live_sel', 'live_test'):
        if len(data[split][1]):
            report[split]['deployed_blend'] = metrics(blend(split, beta, temp), data[split][1])
        if split != 'live_sel' and len(data[split][1]):
            beats.append(report[split]['deployed_blend']['logloss'] < report[split]['hand']['logloss'])
    # Judge what the engine will play (the blend), not the raw net. The engine
    # divides by a (hand units); a flat or inverted fit is unusable.
    report['useful'] = bool(beats) and all(beats) and a > 0.05 and beta > 0
    layers = [m for m in net if hasattr(m, 'weight')]
    weights = {'layers': [{'W': l.weight.detach().tolist(), 'b': l.bias.detach().tolist()} for l in layers],
               'mean': mean.tolist(), 'std': std.tolist(), 'hand': {'a': a, 'b': b, 'temp': temp}}
    blob = json.dumps(weights, separators=(',', ':')).encode()
    sha = hashlib.sha256(blob).hexdigest()
    report['seconds'] = round(time.time() - started, 1)
    report['sha256'] = sha
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'value-{sha[:16]}.json'
    if not path.exists():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({**weights, 'meta': {'report': report, 'manifest': manifest, 'args': {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, 'created': time.time()}},
            separators=(',', ':')))
        tmp.replace(path)
    report['path'] = str(path)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    main()
