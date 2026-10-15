"""How well does each opponent model predict what OUR ladder opponents really did?

    .venv/bin/python search/human/compare_live.py --model data/ml/human/models/human-<sha>.json \
        [--games 200] [--worlds 2] [--out artifacts/human-policy/compare-live.json]

For every move turn of our own live games (data/ml/browser-runs ledgers), the
position is rebuilt exactly as the live search rebuilt it (our request, the
opponent sampled from priors with what we had seen), the deployed search runs on
one or more worlds with `opp_detail`, and the opponent's real action (from the
room's public log) is located in the engine's enumeration. Then, per world:

- q        the deployed opponent model (alpha * equilibrium + (1-alpha) * greedy softmax)
- nash     the equilibrium column strategy alone
- greedy   the softmax prior alone
- uniform  over the enumerated (post-screening) actions
- human    the learned policy (softmax of the exported net's logits over the same actions)
- mixtures (1-w) * human + w * q

Reports NLL and top-1/top-3 on decisions whose real action is fully known, and
how often the real action was dropped by the probe screening (then every model
of the kept set gets it wrong: we count that as a miss with NLL of the floor).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from battle_state import legal_choices  # noqa: E402
from search.agent import SearchAgent  # noqa: E402
from search.config import agent_kwargs  # noqa: E402
from search.human.extract import match_choice, player_info, split_turns, turn_actions  # noqa: E402
from search.ledger_positions import _prefix, load_games  # noqa: E402
from search.tracker import Tracker  # noqa: E402

FMT = 'gen9championsvgc2026regmc'


def load_net(path: Path):
    w = json.loads(Path(path).read_text())
    layers = [(np.asarray(l['W'], np.float64), np.asarray(l['b'], np.float64)) for l in w['layers']]
    return {'layers': layers, 'mean': np.asarray(w['mean']), 'std': np.asarray(w['std'])}


def logits(net, X: np.ndarray) -> np.ndarray:
    h = (np.asarray(X, np.float64) - net['mean']) / net['std']
    for k, (W, b) in enumerate(net['layers']):
        h = h @ W.T + b
        if k < len(net['layers']) - 1:
            h = np.maximum(h, 0)
    return h[:, 0]


def softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


FLOOR = 1e-3


def score(p: np.ndarray, mask: list[int]) -> tuple[float, bool, bool]:
    mass = float(p[mask].sum()) if mask else 0.0
    order = np.argsort(-p)
    return -math.log(max(mass, FLOOR)), bool(mask and order[0] in set(mask)), bool(mask and set(order[:3]) & set(mask))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', action='append', type=Path, default=[])
    ap.add_argument('--games', type=int, default=150, help='newest live games to use')
    ap.add_argument('--worlds', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts' / 'human-policy' / 'compare-live.json')
    ap.add_argument('--team', default='Rain-Recife-special-stat-fix')
    ap.add_argument('--mix', default='0.1,0.2,0.3,0.5')
    ap.add_argument('--leaf', type=json.loads, help='win-probability leaf {"a":..,"b":..} (engine.cjs setLeaf)')
    ap.add_argument('--dump', type=Path, help='JSONL of per-world nash/greedy/real-action mask, for offline alpha/tau sweeps')
    args = ap.parse_args()
    nets = {p.stem: load_net(p) for p in args.model}
    team_sets = json.loads((ROOT / 'teams.json').read_text())[args.team]['sets']
    games = load_games(ROOT / 'data' / 'ml' / 'browser-runs')
    rooms = sorted(games, key=lambda r: games[r].get('time') or 0)[-args.games:]
    kw = agent_kwargs(engines=1, screen_log=0.0, adaptive_margin=None, adaptive_worlds=0, engine_timeout=600.0)
    agent = SearchAgent(fmt=FMT, seed=args.seed, **kw)
    mixes = [float(x) for x in args.mix.split(',') if x]
    names = ['q', 'nash', 'greedy', 'uniform'] + [f'human:{n}' for n in nets] + [f'mix{w}:{n}' for n in nets for w in mixes]
    acc = {n: {'nll': 0.0, 'top1': 0, 'top3': 0, 'n': 0} for n in names}
    counts = {'decisions': 0, 'exact': 0, 'screened_out': 0, 'not_found': 0, 'worlds': 0}
    t0 = time.time()
    dump = args.dump.open('w') if args.dump else None
    try:
        for gi, room in enumerate(rooms):
            game = games[room]
            log = game['log']
            info = player_info(log)
            turns = {t: (i, e) for t, i, e in split_turns(log)}
            agent.rng = random.Random(f'{args.seed}:{room}')
            by_turn = {}
            for item in sorted(game['decisions'].values(), key=lambda d: d.get('time') or 0):
                req = item['request']
                if req.get('teamPreview') or req.get('forceSwitch') or req.get('wait') or not req.get('active'):
                    continue
                by_turn[item.get('turn')] = item
            for turn, item in by_turn.items():
                if turn not in turns:
                    continue
                req = item['request']
                side = req['side']['id']
                foe = 'p2' if side == 'p1' else 'p1'
                prefix = _prefix(log, turn)
                if prefix is None:
                    continue
                i, e = turns[turn]
                acts = turn_actions(log[i + 1:e])
                if not any(k.startswith(foe) for k in acts):
                    continue
                tracker = Tracker(side).feed(prefix)
                ctx = {'request': req, 'choices': legal_choices(req), 'team_sets': team_sets, 'public_log': prefix}
                try:
                    our = agent._our_side(ctx, tracker)
                    if any(a.get('canMegaEvo') for a in req.get('active') or []):
                        tracker.mega_used[side] = False
                    field = agent._field(tracker)
                    our_choices = agent._prune_ours(ctx)
                    sim_map = {}
                    for c in our_choices:
                        sim_map.setdefault(agent._to_sim(c, req.get('active') or [], tracker), c)
                    worlds = [{'p1': our, 'p2': agent._their_side(tracker, agent.rng)} for _ in range(args.worlds)]
                    out = await agent.engines[0].call({'type': 'search', 'format': FMT, 'field': field, 'worlds': worlds,
                                                       'our_choices': list(sim_map), 'seeds': 1, 'alpha': kw['alpha'],
                                                       'opp_tau': kw['opp_tau'], 'screen': kw['screen'], 'crn': kw['crn'],
                                                       'opp_detail': True, 'world_offset': 0,
                                                       **({'leaf': args.leaf} if args.leaf else {})}, timeout=600)
                except Exception as error:
                    counts['not_found'] += 1
                    continue
                counts['decisions'] += 1
                for r in out['results']:
                    d = r.get('opp_detail')
                    if not d:
                        continue
                    counts['worlds'] += 1
                    # The real action, located in the full enumeration (before screening) and in the kept set.
                    full = {'choices': d['all'], 'moves': d['moves'], 'party': d['party']}
                    mask_all, why = match_choice(full, acts, foe, d['party'], d['alive'])
                    if not mask_all:
                        counts['not_found'] += 1
                        continue
                    exact = why == 'ok'
                    if not exact:
                        continue
                    counts['exact'] += 1
                    kept = {c: k for k, c in enumerate(d['theirs'])}
                    mask = [kept[d['all'][j]] for j in mask_all if d['all'][j] in kept]
                    if not mask:
                        counts['screened_out'] += 1
                    # Coverage: would the real action survive a human-probability screening of the
                    # same size as the probe screening (keep_opp), per model?
                    K = len(d['theirs'])
                    if nets and len(d['all']) > K:
                        Xa = np.concatenate([np.repeat(np.asarray(d['sx'])[None, :], len(d['all']), 0), np.asarray(d['a_all'])], axis=1)
                        for name, net in nets.items():
                            top = set(np.argsort(-logits(net, Xa))[:K].tolist())
                            cov = counts.setdefault('coverage', {}).setdefault(name, [0, 0])
                            cov[0] += int(bool(top & set(mask_all))); cov[1] += 1
                        cov = counts.setdefault('coverage', {}).setdefault('probe', [0, 0])
                        cov[0] += int(bool(mask)); cov[1] += 1
                    if dump is not None:
                        dump.write(json.dumps({'room': room, 'turn': turn, 'nash': [round(v, 6) for v in d['nash']],
                                               'greedy': [round(v, 8) for v in d['greedy']], 'mask': mask}) + '\n')
                    n = len(d['theirs'])
                    dists = {'q': np.asarray(d['q']), 'nash': np.asarray(d['nash']), 'greedy': np.asarray(d['greedy']),
                             'uniform': np.full(n, 1.0 / n)}
                    if nets:
                        X = np.concatenate([np.repeat(np.asarray(d['sx'])[None, :], n, 0), np.asarray(d['a'])], axis=1)
                        for name, net in nets.items():
                            h = softmax(logits(net, X))
                            dists[f'human:{name}'] = h
                            for w in mixes:
                                dists[f'mix{w}:{name}'] = (1 - w) * h + w * dists['q']
                    for name, p in dists.items():
                        nll, t1, t3 = score(p, mask)
                        a = acc[name]
                        a['nll'] += nll; a['top1'] += t1; a['top3'] += t3; a['n'] += 1
            if (gi + 1) % 10 == 0:
                print(json.dumps({'games': gi + 1, 'counts': counts, 'elapsed': round(time.time() - t0),
                                  'q_top1': round(acc['q']['top1'] / max(1, acc['q']['n']), 3)}), flush=True)
    finally:
        await agent.close()
        if dump is not None:
            dump.close()
    report = {'counts': counts, 'games': len(rooms), 'worlds_per_decision': args.worlds,
              'models': {n: {'nll': round(a['nll'] / max(1, a['n']), 4), 'top1': round(a['top1'] / max(1, a['n']), 4),
                             'top3': round(a['top3'] / max(1, a['n']), 4), 'n': a['n']} for n, a in acc.items()},
              'seconds': round(time.time() - t0)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == '__main__':
    asyncio.run(main())
