"""Which team does the deployed search win most with, against the ladder-human imitation?

    .venv/bin/python search/human/team_arena.py --teams data/ml/human/candidate-teams-valid.json --games 30 \
        --out artifacts/human-policy/team-arena.jsonl

Every candidate team (plus our live team) is played by the deployed search with the
human preview policy (argmax) against the argmax imitation opponent piloting the real-team
pool with the same preview policy, same seeds and sides for every team. A human-population
comparison (docs/human-opponent.md): the pilot and the opponent model are fixed, only the
six Pokemon change.
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
from search.value_gate import make_preview  # noqa: E402

FMT = 'gen9championsvgc2026regmc'
FROZEN = ROOT / 'artifacts' / 'search-breakthrough' / 'frozen'


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teams', type=Path, required=True)
    ap.add_argument('--opp-model', default='data/ml/human/models-v4/human-449e6bd0e8869deb.json')
    ap.add_argument('--preview', default='data/ml/human/models-preview/preview-1455f4a55d18112a.json')
    ap.add_argument('--games', type=int, default=30)
    ap.add_argument('--concurrency', type=int, default=3)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--pool', type=Path, default=ROOT / 'data' / 'ml' / 'value' / 'pool-build' / 'pool-teams-new.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cands = json.loads(args.teams.read_text())
    ours = json.loads((ROOT / 'teams.json').read_text())['Rain-Recife-special-stat-fix']['sets']
    teams = [('ours', ours)] + [('-'.join(t['six']), t['sets']) for t in cands]
    pool = [t['sets'] if isinstance(t, dict) else t for t in json.loads(args.pool.read_text())]
    brain = BrainAgent(FROZEN, explore=False)
    fallback = brain.preview_fn()
    sem = asyncio.Semaphore(args.concurrency)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r['team'], r['index']))
    tally = {}

    async def one(name, sets, i):
        if (name, i) in done:
            return
        async with sem:
            seed = args.seed * 1000 + i
            us_side = 'p1' if i % 2 == 0 else 'p2'
            them_side = 'p2' if us_side == 'p1' else 'p1'
            us = SearchAgent(fmt=FMT, preview=make_preview(f'human:{args.preview}:argmax', fallback, seed), seed=seed,
                             **agent_kwargs(engines=1, adaptive_max_ms=1e9, engine_timeout=900.0))
            them = ImitationAgent(args.opp_model, fmt=FMT, preview=fallback, seed=seed + 1, sample=False, preview_model=args.preview)
            t0 = time.time()
            try:
                r = await play_game({us_side: us, them_side: them}, {us_side: sets, them_side: pool[i % len(pool)]}, seed, timeout=300)
                rec = {'team': name, 'index': i, 'winner': 'us' if r['winner'] == us_side else 'them' if r['winner'] == them_side else 'tie',
                       'turns': r['turns'], 'seconds': round(time.time() - t0, 1), 'us_fallbacks': us.stats['fallbacks'], 'them_fallbacks': them.stats['fallbacks']}
            except Exception as error:
                rec = {'team': name, 'index': i, 'winner': None, 'error': repr(error)[:200]}
            finally:
                await us.close(); await them.close()
            with args.out.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')
            t = tally.setdefault(name, {'us': 0, 'them': 0, 'tie': 0})
            if rec.get('winner'):
                t[rec['winner']] += 1
            print(json.dumps({'team': name[:50], 'tally': t}), flush=True)

    await asyncio.gather(*(one(n, s, i) for i in range(args.games) for n, s in teams))
    print(json.dumps({'final': tally}))


if __name__ == '__main__':
    asyncio.run(main())
