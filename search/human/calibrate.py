"""How strong is each imitation-opponent variant against the deployed search?

    .venv/bin/python search/human/calibrate.py --model M --preview P --games 24 --concurrency 4 \
        --variant t1.0 --variant t0.5 --variant argmax --variant topk3 --out artifacts/human-policy/calibrate.jsonl

The live ladder beats the deployed search about half the time. A gate opponent
that the search beats 95% of the time yields almost no discordant pairs, so the
imitation variant used in a gate should land near the ladder's strength.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from search.agent import SearchAgent  # noqa: E402
from search.arena import BrainAgent, play_game  # noqa: E402
from search.config import agent_kwargs  # noqa: E402
from search.human.imitation import ImitationAgent  # noqa: E402

FMT = 'gen9championsvgc2026regmc'
FROZEN = ROOT / 'artifacts' / 'search-breakthrough' / 'frozen'


def variant_kwargs(name: str) -> dict:
    if name == 'argmax':
        return {'sample': False}
    if name.startswith('topk'):
        return {'search_topk': int(name[4:]), 'sample': True}
    if name.startswith('t'):
        return {'temp': float(name[1:]), 'sample': True}
    raise ValueError(name)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--preview', default=None)
    ap.add_argument('--variant', action='append', default=[])
    ap.add_argument('--games', type=int, default=24)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--pool', type=Path, default=ROOT / 'data' / 'ml' / 'value' / 'pool-build' / 'pool-teams-new.json')
    ap.add_argument('--team', default='Rain-Recife-special-stat-fix')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    team = json.loads((ROOT / 'teams.json').read_text())[args.team]['sets']
    pool = [t['sets'] if isinstance(t, dict) else t for t in json.loads(args.pool.read_text())]
    brain = BrainAgent(FROZEN, explore=False)
    preview = brain.preview_fn()
    sem = asyncio.Semaphore(args.concurrency)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r['variant'], r['index']))
    tally = {}

    async def one(variant, i):
        if (variant, i) in done:
            return
        async with sem:
            seed = args.seed * 1000 + i
            us_side = 'p1' if i % 2 == 0 else 'p2'
            them_side = 'p2' if us_side == 'p1' else 'p1'
            us = SearchAgent(fmt=FMT, preview=preview, seed=seed, **agent_kwargs(engines=1, adaptive_max_ms=1e9, engine_timeout=900.0))
            them = ImitationAgent(args.model, fmt=FMT, preview=preview, seed=seed + 1, preview_model=args.preview, **variant_kwargs(variant))
            t0 = time.time()
            try:
                r = await play_game({us_side: us, them_side: them}, {us_side: team, them_side: pool[i % len(pool)]}, seed, timeout=300)
                rec = {'variant': variant, 'index': i, 'winner': 'us' if r['winner'] == us_side else 'them' if r['winner'] == them_side else 'tie',
                       'turns': r['turns'], 'seconds': round(time.time() - t0, 1), 'them_fallbacks': them.stats['fallbacks'],
                       'them_ms': round(r['decision_ms'][them_side] / max(1, r['decisions'][them_side]))}
            except Exception as error:
                rec = {'variant': variant, 'index': i, 'winner': None, 'error': repr(error)[:200]}
            finally:
                await us.close(); await them.close()
            with args.out.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')
            t = tally.setdefault(variant, {'us': 0, 'them': 0, 'tie': 0, 'n': 0})
            if rec.get('winner'):
                t[rec['winner']] += 1; t['n'] += 1
            print(json.dumps({'variant': variant, 'tally': t, 'seconds': rec.get('seconds')}), flush=True)

    await asyncio.gather(*(one(v, i) for v in args.variant for i in range(args.games)))
    print(json.dumps({'final': tally}))


if __name__ == '__main__':
    asyncio.run(main())
