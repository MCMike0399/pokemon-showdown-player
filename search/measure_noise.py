"""Measure how much of a search decision is noise, on real live positions.

For sampled turn-start positions from the live ledgers, the decision is
recomputed several times and compared:

* chance noise: the SAME sampled worlds, different simulator seeds, with
  common random numbers off and on (`crn`);
* world noise: freshly sampled opponent worlds, with K and 2K worlds.

The flip rate (two repeats choosing different actions) is a direct,
statistically efficient measure of decision noise; a win-rate gate could not
detect a variance-reduction change of this size. Output: one JSON summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battle_state import legal_choices  # noqa: E402
from search.agent import SearchAgent  # noqa: E402
from search.config import agent_kwargs  # noqa: E402
from search.ledger_positions import FMT, _prefix, load_games  # noqa: E402
from search.tracker import Tracker  # noqa: E402


def positions(runs: Path, n: int, seed: int):
    out = []
    for room, game in load_games(runs).items():
        for item in game['decisions'].values():
            r = item['request']
            if r.get('teamPreview') or r.get('forceSwitch') or r.get('wait') or not r.get('active'):
                continue
            prefix = _prefix(game['log'], item.get('turn')) if item.get('turn') else None
            if prefix:
                out.append((room, item, prefix))
    random.Random(seed).shuffle(out)
    return out[:n]


def decide(results: list[dict]) -> tuple[str | None, float]:
    totals = {}
    good = [r for r in results if 'values' in r]
    for r in good:
        for c, v in r['values'].items():
            totals.setdefault(c, []).append(v)
    if not totals:
        return None, 0.0
    scored = {c: sum(v) / len(v) - (0.5 if len(v) < len(good) else 0.0) for c, v in totals.items()}
    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    return ranked[0][0], (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 0.0


async def run_worlds(agent, base, worlds, offset):
    n = len(agent.engines)
    chunks = [worlds[i::n] for i in range(n)]
    outs = await asyncio.gather(*(e.call({**base, 'worlds': c, 'world_offset': offset + k * 1000}, timeout=900)
                                  for k, (e, c) in enumerate(zip(agent.engines, chunks)) if c))
    return [r for o in outs for r in o['results']]


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, default=ROOT / 'data' / 'ml' / 'browser-runs')
    ap.add_argument('--positions', type=int, default=40)
    ap.add_argument('--engines', type=int, default=2)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--team', default='Rain-Recife-special-stat-fix')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--records', type=Path, help='per-position JSONL (appended)')
    args = ap.parse_args()
    team = json.loads((ROOT / 'teams.json').read_text())[args.team]['sets']
    kw = agent_kwargs(engines=args.engines)
    K = kw['worlds']
    agent = SearchAgent(fmt=FMT, seed=args.seed, **kw)
    stats = {k: [] for k in ('chance_crn_off', 'chance_crn_on', 'worlds_K', 'worlds_2K')}
    margins, ms = [], {k: [] for k in stats}
    started = time.time()
    done = set()
    if args.records and args.records.exists():  # resume: re-count finished positions, skip them
        for line in args.records.read_text().splitlines():
            rec = json.loads(line)
            if (rec['room'], rec['turn']) in done:
                continue
            done.add((rec['room'], rec['turn']))
            for key, flip in rec['flips'].items():
                stats[key].append(flip)
                stats.setdefault('regret_' + key, []).append(rec['regret'][key])
                ms[key].extend(rec['ms'][key])
            margins.append(rec['margin'])
    try:
        for idx, (room, item, prefix) in enumerate(positions(args.runs, args.positions, args.seed)):
            if (room, item.get('turn')) in done:
                continue
            request = item['request']
            side = request['side']['id']
            tracker = Tracker(side).feed(prefix)
            ctx = {'request': request, 'choices': legal_choices(request), 'team_sets': team, 'public_log': prefix}
            try:
                our_choices = agent._prune_ours(ctx)
                active = request.get('active') or []
                if any(a.get('canMegaEvo') for a in active):
                    tracker.mega_used[side] = False
                sim = list(dict.fromkeys(agent._to_sim(c, active, tracker) for c in our_choices))
                our = agent._our_side(ctx, tracker)
                field = agent._field(tracker)
            except Exception as error:
                print(json.dumps({'skip': room, 'error': repr(error)[:200]}), file=sys.stderr)
                continue
            if len(sim) < 2:
                continue
            try:
                record = await measure_one(agent, kw, K, args, idx, sim, our, field, tracker, stats, ms, margins)
            except Exception as error:  # a slow or failed position is skipped, never guessed
                print(json.dumps({'skip': room, 'error': repr(error)[:200]}), file=sys.stderr)
                continue
            if args.records:
                with args.records.open('a') as fh:
                    fh.write(json.dumps({'room': room, 'turn': item.get('turn'), **record}) + '\n')
            print(json.dumps({'i': idx, **{k: round(sum(v), 3) for k, v in stats.items()}, 'n': len(stats['worlds_K'])}),
                  file=sys.stderr, flush=True)
    finally:
        await agent.close()
    return finish(stats, ms, margins, K, args, started)


async def measure_one(agent, kw, K, args, idx, sim, our, field, tracker, stats, ms, margins):
    """Two repeats per configuration; stats are only updated once all succeed.

The reference value of each action is its mean over every run of this position
(4 x K fixed-world runs and K + K + 2K + 2K fresh worlds), so `regret` is how
much reference value a configuration's pick gives up, which separates harmful
flips from flips between equally good actions.
"""
    base = {'type': 'search', 'format': FMT, 'field': field, 'our_choices': sim, 'seeds': kw['seeds'],
            'alpha': kw['alpha'], 'opp_tau': kw['opp_tau'], 'screen': kw['screen']}
    rng = random.Random(f'{args.seed}:{idx}')
    sample = lambda k: [{'p1': our, 'p2': agent._their_side(tracker, rng)} for _ in range(k)]
    fixed = sample(K)
    picks, times, all_results, margin = {}, {}, [], None
    plan = [('chance_crn_off', False, lambda: fixed), ('chance_crn_on', True, lambda: fixed),
            ('worlds_K', True, lambda: sample(K)), ('worlds_2K', True, lambda: sample(2 * K))]
    for key, crn, worlds in plan:
        picks[key], times[key] = [], []
        for rep in range(2):
            t0 = time.perf_counter()
            res = await run_worlds(agent, {**base, 'crn': crn}, worlds(), (1 if key.startswith('chance') else 7) + rep * 50000)
            times[key].append((time.perf_counter() - t0) * 1000)
            c, m = decide(res)
            picks[key].append(c)
            all_results.extend(r for r in res if 'values' in r)
            if margin is None:
                margin = m
    ref = {}
    for r in all_results:
        for c, v in r['values'].items():
            ref.setdefault(c, []).append(v)
    ref = {c: sum(v) / len(v) for c, v in ref.items()}
    best = max(ref.values())
    record = {'margin': margin, 'flips': {}, 'regret': {}, 'ms': times}
    for key in picks:
        record['flips'][key] = picks[key][0] != picks[key][1]
        record['regret'][key] = sum(best - ref.get(c, best) for c in picks[key]) / 2
        stats[key].append(record['flips'][key])
        ms[key].extend(times[key])
        stats.setdefault('regret_' + key, []).append(record['regret'][key])
    margins.append(margin)
    return record


def finish(stats, ms, margins, K, args, started):
    n = len(stats['worlds_K'])
    summary = {'positions': n, 'worlds': K, 'engines': args.engines, 'seconds': round(time.time() - started, 1),
               'flip_rate': {k: round(sum(v) / max(1, len(v)), 3) for k, v in stats.items() if not k.startswith('regret')},
               'flips': {k: int(sum(v)) for k, v in stats.items() if not k.startswith('regret')},
               'mean_regret': {k[7:]: round(sum(v) / max(1, len(v)), 4) for k, v in stats.items() if k.startswith('regret')},
               'median_ms': {k: round(statistics.median(v), 1) if v else None for k, v in ms.items()},
               'median_margin': round(statistics.median(margins), 4) if margins else None}
    # Exact paired sign test for crn on vs off (discordant positions only).
    off, on = stats['chance_crn_off'], stats['chance_crn_on']
    b = sum(1 for x, y in zip(off, on) if x and not y)
    c = sum(1 for x, y in zip(off, on) if y and not x)
    from math import comb
    m = b + c
    summary['crn_sign_test'] = {'off_only': b, 'on_only': c,
                                'p_one_sided': round(sum(comb(m, i) for i in range(b, m + 1)) / 2 ** m, 4) if m else None}
    print(json.dumps(summary))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
