"""Freeze and fit one matchup-sensitive residual; no promotion or live writes.

prepare --replays 'data/replays/regmc*/replays-*.jsonl' --out DIR
train --out DIR

Inputs/protocol are immutable after prepare. Split by numeric battle id (a
chronological proxy), keep both players in the same split, and inspect the test
only once after validation-only checkpoint selection.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battle_state import to_id  # noqa: E402
from search.preview_residual import TARGET, RISKS, VERSION, features  # noqa: E402
from search.preview_guard import DEFAULT_MODEL  # noqa: E402
from search.human.preview_model import PLANS, _sid, plan_features, consistent  # noqa: E402
from search.human.preview_value import PreviewValue  # noqa: E402
from search.value_store import atomic_json  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def split_games(game_ids):
    games = sorted(set(game_ids), key=lambda g: (int(g.rsplit('-', 1)[1]), g))
    n = len(games)
    return {g: ('train' if i < int(n*.7) else 'val' if i < int(n*.85) else 'test')
            for i, g in enumerate(games)}


def logloss(p, y):
    p = np.clip(p, 1e-6, 1-1e-6)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def prepare(out, patterns):
    from search.human.extract import preview_record, player_info
    from search.elo_monitor import my_name
    if (out / 'protocol.json').exists():
        raise ValueError('protocol already frozen; do not retune this experiment')
    sources = sorted({Path(f) for pat in patterns for f in glob.glob(pat)})
    if not sources:
        raise ValueError('no replay files')
    out.mkdir(parents=True, exist_ok=True)
    # Written before extraction or model results, including the final acceptance rule.
    protocol = {'version': VERSION, 'seed': 20261023, 'created': time.time(),
        'sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sources],
        'code': {n: sha(ROOT/n) for n in ('search/train_preview_residual.py', 'search/preview_residual.py',
                                         'search/human/preview_model.py', 'search/human/preview_value.py')},
        'baseline': DEFAULT_MODEL, 'baseline_sha256': sha(ROOT / DEFAULT_MODEL),
        'target': list(TARGET), 'risks': list(RISKS), 'minimum_team_overlap': 4,
        'exclusions': 'our account; either player name contains bot (case/punctuation insensitive); Illusion games',
        'split': 'numeric battle id, 70/15/15 percent by whole game',
        'fit': {'hidden': 32, 'epochs': 40, 'patience': 5, 'batch': 1024, 'lr': .001, 'decay': .01, 'cap': .35},
        'rating_offset': 'log(10) * clipped rating difference [-400,400] / 400; zero live',
        'eligibility': {'max_incumbent_drop': .025, 'only_target_team_and_risk_matchups': True},
        'accept': '>=300 validation and test rows each; both logloss improve >=.002; test Brier no worse',
        'promotion': 'none; requires a separately frozen offline gate then fresh ladder A/B after c10',
        'limitations': 'Unobserved bot accounts may remain. Baseline may have seen historical test games. '
                       'Predictive performance is not counterfactual evidence of better plans.'}
    atomic_json(out / 'protocol.json', protocol)
    (out / 'protocol.sha256').write_text(sha(out / 'protocol.json'))
    seen, examples, excluded = set(), [], 0
    me = to_id(my_name())
    for path in sources:
        for line in path.open():
            rep = json.loads(line)
            gid = rep.get('id', '')
            if gid in seen or not re.fullmatch(r'gen9championsvgc2026regmc-\d+', gid):
                continue
            seen.add(gid)
            log = rep.get('log', '')
            log = log.split('\n') if isinstance(log, str) else log
            info = player_info(log)
            names = [to_id(info[s].get('name')) for s in ('p1', 'p2')]
            if me in names or any('bot' in n for n in names) or any(l.startswith('|replace|') for l in log):
                excluded += 1
                continue
            r = preview_record(gid, log, info)
            if not r or not r['winner']:
                continue
            for side, foe in (('p1', 'p2'), ('p2', 'p1')):
                own, opp = r['six'][side], r['six'][foe]
                ratings = [r['rating'][s] for s in (side, foe)]
                if (len(set(map(to_id, own)) & set(TARGET)) < 4 or not set(map(to_id, opp)) & set(RISKS)
                        or any(x is None for x in ratings) or len(r['brought'][side]) != 4):
                    continue
                plans = consistent(own, r['leads'][side], r['brought'][side])
                if len(plans) != 1:
                    continue
                examples.append({'game': gid, 'side': side, 'own': own, 'opp': opp, 'plan': plans[0],
                                 'ratings': ratings, 'y': float(r['winner'] == r['names'][side])})
    splits = split_games([r['game'] for r in examples])
    if len(splits) < 1000:
        raise ValueError(f'only {len(splits)} qualifying games; frozen protocol retained, no model fitted')
    baseline = PreviewValue(ROOT / DEFAULT_MODEL)
    X = np.stack([features(r['own'], r['opp'], PLANS[r['plan']]) for r in examples])
    bx = np.stack([plan_features([_sid(s) for s in r['own']], [_sid(s) for s in r['opp']], PLANS[r['plan']])
                   for r in examples])
    if baseline.extra:
        bx = np.concatenate([bx, np.zeros((len(bx), baseline.extra))], 1)
    probs = np.zeros(len(bx))
    for layers in baseline.nets:
        h = bx
        for i, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if i < len(layers)-1:
                h = np.maximum(h, 0)
        probs += 1/(1+np.exp(-np.clip(h[:, 0], -30, 30))) / len(baseline.nets)
    offset = np.log(np.clip(probs, 1e-6, 1-1e-6)/np.clip(1-probs, 1e-6, 1))
    offset += np.array([math.log(10)*np.clip(r['ratings'][0]-r['ratings'][1], -400, 400)/400 for r in examples])
    np.savez_compressed(out / 'data.npz', X=X, offset=offset.astype(np.float32),
                        y=np.array([r['y'] for r in examples], np.float32),
                        split=np.array([splits[r['game']] for r in examples]))
    atomic_json(out / 'manifest.json', {'games_seen': len(seen), 'games_excluded': excluded,
                                       'rows': [{**r, 'split': splits[r['game']]} for r in examples]})
    atomic_json(out / 'frozen.json', {n: sha(out/n) for n in ('protocol.json', 'data.npz', 'manifest.json')})
    print(json.dumps({'rows': len(examples), 'games': len(splits), 'split_rows':
                      {k: sum(splits[r['game']]==k for r in examples) for k in ('train', 'val', 'test')}}))


def train(out):
    import torch
    if (out / 'result.json').exists():
        raise ValueError('test has already been evaluated; no retuning on this snapshot')
    for n, h in json.loads((out / 'frozen.json').read_text()).items():
        if sha(out/n) != h:
            raise ValueError('frozen input changed: ' + n)
    protocol = json.loads((out / 'protocol.json').read_text())
    if sha(ROOT / protocol['baseline']) != protocol['baseline_sha256']:
        raise ValueError('baseline changed')
    for name, expected in protocol['code'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('preregistered training code changed: ' + name)
    if not torch.backends.mps.is_available():
        raise RuntimeError('this M5 experiment requires MPS')
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.5)
    torch.manual_seed(protocol['seed'])
    fit = protocol['fit']; started = time.time()
    data = np.load(out / 'data.npz', allow_pickle=False)
    X, y, offset = [torch.tensor(data[k], device='mps') for k in ('X', 'y', 'offset')]
    tr, va, te = [np.flatnonzero(data['split']==k) for k in ('train', 'val', 'test')]
    if min(len(va), len(te)) < 300:
        raise ValueError('insufficient frozen validation/test support')
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], fit['hidden']), torch.nn.ReLU(),
                              torch.nn.Linear(fit['hidden'], 1)).to('mps')
    torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=fit['lr'], weight_decay=fit['decay'])
    bce = torch.nn.BCEWithLogitsLoss(); rng = np.random.default_rng(protocol['seed'])
    best, bad, history, weights = float('inf'), 0, [], None
    def logits(indices):
        return offset[indices] + fit['cap']*torch.tanh(net(X[indices]).squeeze(-1))
    vi = torch.tensor(va, device='mps')
    for epoch in range(fit['epochs']):
        net.train(); perm = rng.permutation(tr)
        for j in range(0, len(perm), fit['batch']):
            ix = torch.tensor(perm[j:j+fit['batch']], device='mps')
            opt.zero_grad(); loss = bce(logits(ix), y[ix]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            loss = float(bce(logits(vi), y[vi]))
        history.append(loss)
        if loss < best-1e-5:
            best, bad = loss, 0
            weights = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= fit['patience']:
                break
    net.load_state_dict(weights)
    report = {'device': 'mps', 'history': history, 'rows': len(y), 'evaluation': {}}
    with torch.no_grad():
        for label, indices in (('val', va), ('test', te)):
            ix = torch.tensor(indices, device='mps')
            pred = torch.sigmoid(logits(ix)).cpu().numpy()
            base = torch.sigmoid(offset[ix]).cpu().numpy(); target = data['y'][indices]
            report['evaluation'][label] = {'n': len(indices), 'logloss': logloss(pred, target),
                'baseline_logloss': logloss(base, target), 'brier': float(np.mean((pred-target)**2)),
                'baseline_brier': float(np.mean((base-target)**2))}
    val, test = [report['evaluation'][s] for s in ('val', 'test')]
    report['qualified_for_offline_gate'] = bool(all(r['logloss'] <= r['baseline_logloss']-.002 for r in (val,test))
                                               and test['brier'] <= test['baseline_brier'])
    blob = {'version': VERSION, 'cap': fit['cap'], 'baseline_sha256': protocol['baseline_sha256'],
            'layers': [{'W': m.weight.detach().cpu().tolist(), 'b': m.bias.detach().cpu().tolist()}
                       for m in net if isinstance(m, torch.nn.Linear)]}
    digest = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()
    path = out / f'preview-residual-{digest[:16]}.json'
    atomic_json(path, blob)
    report.update(path=str(path.resolve()), sha256=sha(path), seconds=time.time()-started,
                  promoted=False, note='Predictive test only; no causal strength claim or live writes.')
    atomic_json(out / 'result.json', report)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['prepare', 'train'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--replays', nargs='+', default=['data/replays/regmc*/replays-*.jsonl'])
    args = ap.parse_args()
    if args.cmd == 'prepare':
        prepare(args.out, args.replays)
    else:
        train(args.out)
