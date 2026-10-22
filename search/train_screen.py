"""Train the learned screening ranker on MPS and export an immutable candidate.

    .venv/bin/python search/train_screen.py --team Rain-Recife-special-stat-fix

The probe screening in `search/engine.cjs` simulates each candidate action
against four replies only to rank it; that is about half of every search's
simulations. This trainer fits an MLP that predicts the same ranking from the
position and the action (`search/screenfeat.cjs`), from probe scores the live
runner and the gate log as a by-product (`search/screen_data.py`).

Labels are standardized within each logged (world, side) group, because only
the order inside a group matters. Splits are by game file, never by row. The
newest live games are an untouched test on the deployment distribution.

The ranker becomes a prefilter: the engine keeps its top `pre` actions and runs
the usual probes on those only. The trainer picks, on held-out games, the
smallest `pre` that keeps the probe screening's choices:

- theirs: mean recall of the probe-kept 20 replies >= --recall-opp;
- ours: the world's final choice survives in >= --best-retention of worlds and
  mean recall of the probe-kept 24 >= --recall-ours.

A candidate is `useful` when both sizes exist and the estimated simulations per
world drop by at least --min-saving. Only the paired gate decides promotion.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEEP = {0: 24, 1: 20}  # deployed screening keeps (search/config.py): ours, theirs
PROBES = 4
GRID = (0, 4, 8, 12, 16, 24, 32, 48)


def _bucket(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def game_key(name: str) -> str:
    """One played game: a gate pair's two arms replay the same seed and worlds, and its
two seats see the same game, so all four files must fall on the same side of a split."""
    return re.sub(r'-(cand|inc)-(us|them)\.npz$', '', name)


def load(files: list[str], max_rows: int, val_frac: float, live_test: float):
    """Concatenate logged games, newest first, and split them by game."""
    from search.screen_data import load_records
    files = sorted(set(files), key=lambda f: os.stat(f).st_mtime, reverse=True)
    live = [f for f in files if Path(f).name.startswith('live-')]
    n_test = int(round(len(live) * live_test))
    test_files = set(live[:n_test])  # newest live games
    S, A, y, role, rec, best, split, manifest = [], [], [], [], [], [], [], []
    rows = recs = 0
    width = None
    for path in files:
        if max_rows and rows >= max_rows:
            break
        d = load_records(Path(path))
        if not d or not len(d['y']):
            continue
        shape = (d['S'].shape[-1], d['A'].shape[-1])
        width = width or shape
        if shape != width:
            continue
        name = Path(path).name
        s = 'live_test' if path in test_files else ('val' if _bucket(game_key(name)) < val_frac else 'train')
        S.append(d['S']); A.append(d['A']); y.append(d['y']); role.append(d['role']); best.append(d['best'])
        rec.append(d['rec'].astype(np.int64) + recs)
        split.append(np.full(len(d['y']), s))
        manifest.append({'path': os.path.relpath(path, ROOT), 'rows': int(len(d['y'])), 'split': s})
        rows += len(d['y'])
        recs += len(d['S'])
    if not S:
        return None
    return {'S': np.concatenate(S).astype(np.float32), 'A': np.concatenate(A).astype(np.float32),
            'y': np.concatenate(y), 'role': np.concatenate(role).astype(np.int64), 'rec': np.concatenate(rec),
            'best': np.concatenate(best), 'split': np.concatenate(split), 'manifest': manifest}


def groups(data) -> np.ndarray:
    """Group id per row: one (logged world, side) pair."""
    return data['rec'] * 2 + data['role']


def targets(data):
    """Within-group z-scores of the side's own value (theirs: lower p1 value is better)."""
    g = groups(data)
    v = np.where(data['role'] == 1, -data['y'], data['y']).astype(np.float64)
    ok = np.isfinite(v)
    z = np.zeros(len(v), np.float32)
    keep = np.zeros(len(v), bool)
    order = np.argsort(g, kind='stable')
    bounds = np.flatnonzero(np.diff(g[order])) + 1
    for idx in np.split(order, bounds):
        idx = idx[ok[idx]]
        if len(idx) < 3:
            continue
        sd = v[idx].std()
        if sd < 1e-6:
            continue
        z[idx] = np.clip((v[idx] - v[idx].mean()) / sd, -3, 3)
        keep[idx] = True
    return z, keep


def inputs(data, rows):
    S = data['S'][data['rec'][rows], data['role'][rows]]
    return np.concatenate([S, data['A'][rows], data['role'][rows, None].astype(np.float32)], axis=1)


def train(data, z, keep, hidden, epochs, lr, wd, batch, device, seed, patience):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    tr = np.flatnonzero(keep & (data['split'] == 'train'))
    va = np.flatnonzero(keep & (data['split'] == 'val'))
    sample = np.random.default_rng(seed).choice(tr, min(len(tr), 200_000), replace=False)
    Xs = inputs(data, sample)
    mean, std = Xs.mean(0), Xs.std(0) + 1e-3
    dev = torch.device(device)
    # Positions are stored once per world and gathered per batch on the device.
    S = torch.tensor(data['S'], device=dev)
    A = torch.tensor(data['A'], device=dev)
    rec = torch.tensor(data['rec'], device=dev)
    role = torch.tensor(data['role'], device=dev)
    Z = torch.tensor(z, device=dev)
    mu, sd = torch.tensor(mean, device=dev), torch.tensor(std, device=dev)

    def x_of(idx):
        r = role[idx]
        x = torch.cat([S[rec[idx], r], A[idx], r[:, None].float()], dim=1)
        return (x - mu) / sd

    d = Xs.shape[1]
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    tr_t, va_t = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    best, best_state, bad, history = math.inf, None, 0, []
    for _ in range(epochs):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen).to(dev)]
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = ((net(x_of(b)).squeeze(-1) - Z[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(sum(((net(x_of(va_t[i:i + batch])).squeeze(-1) - Z[va_t[i:i + batch]]) ** 2).sum()
                           for i in range(0, len(va_t), batch)) / max(1, len(va_t))) if len(va_t) else float('nan')
        history.append(round(vl, 5))
        if vl < best - 1e-5 or not len(va_t):
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net.cpu().eval(), mean, std, history


def predict(net, mean, std, data, rows, batch=65536):
    import torch
    out = np.zeros(len(rows), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = (inputs(data, rows[i:i + batch]) - mean) / std
            out[i:i + batch] = net(torch.tensor(x, dtype=torch.float32)).squeeze(-1).numpy()
    return out


def evaluate(data, pred_all, split: str) -> dict:
    """Recall of the probe-kept set and survival of the final choice, per prefilter size."""
    rows = np.flatnonzero(data['split'] == split)
    if not len(rows):
        return {}
    g = groups(data)[rows]
    v = np.where(data['role'][rows] == 1, -data['y'][rows], data['y'][rows])
    pred = pred_all[rows]
    order = np.argsort(g, kind='stable')
    bounds = np.flatnonzero(np.diff(g[order])) + 1
    stats = {k: {'groups': 0, 'size': [], **{n: {'recall': [], 'best': [], 'recall_enum': [], 'recall_random': []}
                                             for n in GRID}} for k in (0, 1)}
    world_sizes: dict[int, dict] = {}
    rng = np.random.default_rng(0)
    for idx in np.split(order, bounds):
        r = int(data['role'][rows[idx[0]]])
        K = KEEP[r]
        world_sizes.setdefault(int(data['rec'][rows[idx[0]]]), {})[r] = len(idx)
        vv = v[idx]
        valid = np.flatnonzero(np.isfinite(vv))
        if len(valid) <= K:
            continue
        true_top = set(valid[np.argsort(-vv[valid], kind='stable')[:K]].tolist())
        ranked = np.argsort(-pred[idx], kind='stable')
        best = np.flatnonzero(data['best'][rows[idx]] == 1)
        s = stats[r]
        s['groups'] += 1
        s['size'].append(len(idx))
        shuffled = rng.permutation(len(idx))
        for n in GRID:
            keep = K + n
            s[n]['recall'].append(len(true_top & set(ranked[:keep].tolist())) / K)
            s[n]['recall_enum'].append(len(true_top & set(range(keep))) / K)
            s[n]['recall_random'].append(len(true_top & set(shuffled[:keep].tolist())) / K)
            if r == 0 and len(best):
                s[n]['best'].append(float(best[0] in set(ranked[:keep].tolist())))
    out = {}
    for r, name in ((0, 'ours'), (1, 'theirs')):
        s = stats[r]
        out[name] = {'groups': s['groups'], 'mean_actions': round(float(np.mean(s['size'])), 1) if s['size'] else None,
                     'by_pre': {str(KEEP[r] + n): {k: round(float(np.mean(s[n][k])), 4) for k in
                                                     ('recall', 'best', 'recall_enum', 'recall_random') if s[n][k]}
                                for n in GRID}}
    out['worlds'] = world_sizes
    return out


LIVE_MIN_GROUPS = 50
LIVE_TOLERANCE = 0.03


def choose(metrics: dict, recall_opp: float, recall_ours: float, best_retention: float, live: dict | None = None) -> dict:
    """Smallest prefilter meeting the thresholds on validation games, and also (within
LIVE_TOLERANCE) on the held-out live games once there are enough of them."""
    def ok(name, r, m, slack):
        if name == 'theirs':
            return m.get('recall', 0) >= recall_opp - slack
        return m.get('recall', 0) >= recall_ours - slack and m.get('best', 0) >= best_retention - slack

    pre = {}
    for name, r in (('theirs', 1), ('ours', 0)):
        live_side = (live or {}).get(name, {})
        check_live = live_side.get('groups', 0) >= LIVE_MIN_GROUPS
        for n in GRID:
            key = str(KEEP[r] + n)
            m = metrics.get(name, {}).get('by_pre', {}).get(key, {})
            if not m or not ok(name, r, m, 0.0):
                continue
            if check_live and not ok(name, r, live_side.get('by_pre', {}).get(key, {}), LIVE_TOLERANCE):
                continue
            pre['opp' if name == 'theirs' else 'ours'] = KEEP[r] + n
            break
    return pre


def sims_per_world(m_ours: int, m_theirs: int, pre: dict | None) -> int:
    """Simulations of one world: probes (4 per screened action) plus the kept matrix."""
    mo, mt = m_ours, m_theirs
    if pre:
        mo, mt = min(mo, pre['ours']), min(mt, pre['opp'])
    probe = PROBES * (mt if mt > KEEP[1] else 0) + PROBES * (mo if mo > KEEP[0] else 0)
    return probe + min(mo, KEEP[0]) * min(mt, KEEP[1])


def check_holdout(metrics: dict, pre: dict, recall_opp=.92, recall_ours=.90, best_retention=.98) -> bool:
    """Check already-fixed prefilter sizes without adapting them to test results."""
    for name, key, threshold in (('ours', 'ours', recall_ours), ('theirs', 'opp', recall_opp)):
        side = metrics.get(name, {})
        if key not in pre or side.get('groups', 0) < LIVE_MIN_GROUPS:
            return False
        m = side.get('by_pre', {}).get(str(pre[key]), {})
        if m.get('recall', 0) < threshold - LIVE_TOLERANCE:
            return False
        if name == 'ours' and m.get('best', 0) < best_retention - LIVE_TOLERANCE:
            return False
    return True


def saving(world_sizes: dict, pre: dict) -> float | None:
    both = [w for w in world_sizes.values() if 0 in w and 1 in w]
    if not both or 'opp' not in pre or 'ours' not in pre:
        return None
    base = sum(sims_per_world(w[0], w[1], None) for w in both)
    new = sum(sims_per_world(w[0], w[1], pre) for w in both)
    return round(1 - new / base, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team', default='Rain-Recife-special-stat-fix')
    ap.add_argument('--data', action='append', default=[], help='glob of logged games (default: the team screen dir)')
    ap.add_argument('--out', type=Path, help='candidate dir (default: the team screen-candidates dir)')
    ap.add_argument('--max-rows', type=int, default=1_500_000, help='newest logged actions to use (memory: ~0.6 KB each)')
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--live-test', type=float, default=0.3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    ap.add_argument('--strict-holdout', action='store_true', help='choose sizes on validation only; test can reject but never retune')
    ap.add_argument('--mps-memory-fraction', type=float, default=.25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--recall-opp', type=float, default=0.92)
    ap.add_argument('--recall-ours', type=float, default=0.90)
    ap.add_argument('--best-retention', type=float, default=0.98)
    ap.add_argument('--min-saving', type=float, default=0.15)
    ap.add_argument('--min-groups', type=int, default=300)
    args = ap.parse_args(argv)
    if not 0 < args.mps_memory_fraction <= 1:
        ap.error('--mps-memory-fraction must be in (0, 1]')
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    from search.screen_data import screen_root
    root = screen_root(args.team)
    files = []
    for pattern in args.data or [str(root / '*.npz')]:
        files.extend(f for f in glob.glob(pattern) if not f.endswith('.tmp.npz'))
    data = load(files, args.max_rows, args.val_frac, args.live_test)
    if data is None:
        raise SystemExit(json.dumps({'error': 'no logged screening data', 'files': len(files)}))
    z, keep = targets(data)
    if not (keep & (data['split'] == 'train')).any():
        raise SystemExit(json.dumps({'error': 'not enough training groups', 'rows': int(len(z))}))
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        try:
            torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
        except Exception:
            pass
    fit = (args.hidden, args.epochs, args.lr, args.wd, args.batch)
    try:
        net, mean, std, history = train(data, z, keep, *fit, device, args.seed, args.patience)
    except Exception as error:  # any device error: same data on the CPU
        if device == 'cpu':
            raise
        print(json.dumps({'device_error': repr(error)[:300], 'fallback': 'cpu'}), file=sys.stderr)
        device = 'cpu'
        net, mean, std, history = train(data, z, keep, *fit, device, args.seed, args.patience)
    pred = predict(net, mean, std, data, np.arange(len(z)))
    report = {'device': device, 'history': history, 'rows': int(len(z)), 'files': len(data['manifest']),
              'splits': {s: int((data['split'] == s).sum()) for s in ('train', 'val', 'live_test')}}
    for split in ('val', 'live_test'):
        report[split] = evaluate(data, pred, split)
    pre = choose(report['val'], args.recall_opp, args.recall_ours, args.best_retention,
                 None if args.strict_holdout else report.get('live_test'))
    report['prefilter'] = pre
    report['estimated_sim_saving'] = {s: saving(report[s].get('worlds', {}), pre) for s in ('val', 'live_test') if report[s]}
    for s in ('val', 'live_test'):
        report.get(s, {}).pop('worlds', None)
    groups_val = min(report['val'].get('ours', {}).get('groups', 0), report['val'].get('theirs', {}).get('groups', 0)) if report['val'] else 0
    est = report['estimated_sim_saving'].get('val')
    report['useful'] = bool('opp' in pre and 'ours' in pre and est is not None and est >= args.min_saving
                            and groups_val >= args.min_groups)
    if args.strict_holdout:
        report['holdout_passed'] = check_holdout(report.get('live_test', {}), pre,
                                                args.recall_opp, args.recall_ours, args.best_retention)
        test_saving = report['estimated_sim_saving'].get('live_test')
        report['useful'] &= bool(report['holdout_passed'] and test_saving is not None and test_saving >= args.min_saving)
    layers = [m for m in net if hasattr(m, 'weight')]
    weights = {'layers': [{'W': l.weight.detach().tolist(), 'b': l.bias.detach().tolist()} for l in layers],
               'mean': mean.tolist(), 'std': std.tolist(),
               'prefilter': {'opp': int(pre.get('opp', KEEP[1] + GRID[-1])), 'ours': int(pre.get('ours', KEEP[0] + GRID[-1]))},
               'version': 1}
    blob = json.dumps(weights, separators=(',', ':')).encode()
    sha = hashlib.sha256(blob).hexdigest()
    report['seconds'] = round(time.time() - started, 1)
    report['sha256'] = sha
    out = args.out or root.parent / 'screen-candidates'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'screen-{sha[:16]}.json'
    if not path.exists():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({**weights, 'meta': {'report': report, 'manifest': data['manifest'][:2000], 'args': {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, 'created': time.time()}},
            separators=(',', ':')))
        tmp.replace(path)
    report['path'] = str(path)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    main()
