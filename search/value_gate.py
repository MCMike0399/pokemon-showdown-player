"""Paired gate for a learned search value: does it win more real-team games?

Both arms play the deployed search (`search.config.agent_kwargs`) with our live
team; they differ ONLY in the leaf evaluation (candidate value vs incumbent).
Pair i uses the same battle seed, the same opponent team from the real-team
pool, the same side, and agents reseeded from the pair seed, so both arms
start from identical worlds. The opponent is the search agent with the hand
evaluation piloting a team real ladder opponents brought; previews come from
the frozen incumbent actor, as live.

Decision rule (preregistered, no peeking): play all `--pairs`; one-sided exact
sign test on discordant pairs; pass iff p <= alpha AND gains > losses. Results
are appended to `<out>` (JSONL) and the run resumes after interruption.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search.arena import BrainAgent, play_game  # noqa: E402
from search.config import agent_kwargs  # noqa: E402
from search.team_data import DEFAULT_TEAM  # noqa: E402

FROZEN = ROOT / 'artifacts' / 'search-breakthrough' / 'frozen'
FMT = 'gen9championsvgc2026regmc'


def sign_test(gains: int, losses: int) -> float:
    n = gains + losses
    if not n:
        return 1.0
    return sum(comb(n, i) for i in range(gains, n + 1)) / 2 ** n


MAX_ATTEMPTS = 3
# Futility: from this many complete pairs on, a candidate with no more gains
# than losses is rejected without finishing the panel. The rule can only end a
# stage as a rejection, so it never raises the false-promotion rate; it frees
# hours of gate compute for the next candidate.
FUTILITY_FROM = 40


def _score(r: dict):
    """1 win, 0 loss, .5 tie or unfinished game; None for an errored attempt."""
    if r.get('error'):
        return None
    if r.get('score') is not None:
        return r['score']
    return None if r.get('win') is None else float(r['win'])


def summarize(rows: list[dict], pairs: int, alpha: float) -> dict:
    """A pair counts once both arms have a result; ties are concordant.

An arm that errored MAX_ATTEMPTS times voids its pair (never retried with a
different seed, which would break the pairing), so a gate always terminates.
"""
    by: dict[int, dict] = {}
    for r in rows:
        arm = by.setdefault(r['pair'], {}).setdefault(r['arm'], {'score': None, 'errors': 0})
        if r.get('error'):
            arm['errors'] += 1
        elif _score(r) is not None:
            arm['score'] = _score(r)
    complete, void = [], 0
    for v in by.values():
        if all(v.get(a, {}).get('score') is not None for a in ('cand', 'inc')):
            complete.append((v['cand']['score'], v['inc']['score']))
        elif any(v.get(a, {}).get('errors', 0) >= MAX_ATTEMPTS for a in ('cand', 'inc')):
            void += 1
    gains = sum(1 for c, i in complete if c > i)
    losses = sum(1 for c, i in complete if i > c)
    p = sign_test(gains, losses)
    futile = len(complete) >= min(FUTILITY_FROM, pairs) and gains <= losses and len(complete) + void < pairs
    done = len(complete) + void >= pairs or futile
    return {'pairs_complete': len(complete), 'pairs_void': void, 'pairs_planned': pairs,
            'candidate_score': sum(c for c, _ in complete), 'incumbent_score': sum(i for _, i in complete),
            'gains': gains, 'losses': losses, 'p_one_sided': round(p, 5),
            'errors': sum(1 for r in rows if r.get('error')), 'complete': done, 'futility_stop': futile,
            'passed': bool(done and not futile and p <= alpha and gains > losses)}


async def run_gate(candidate: dict, incumbent: dict, out: Path, pairs: int, concurrency: int, seed: int,
                   team: list[dict], pool: list[list[dict]], engines: int, alpha: float, log=print) -> dict:
    from search.agent import SearchAgent
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    attempts: dict = {}
    for r in rows:
        key = (r['pair'], r['arm'])
        attempts[key] = MAX_ATTEMPTS if _score(r) is not None else attempts.get(key, 0) + 1
    brain = BrainAgent(FROZEN, explore=False)
    preview = brain.preview_fn()
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    order = random.Random(seed).sample(range(len(pool)), len(pool))

    async def one(pair: int, arm: str):
        while attempts.get((pair, arm), 0) < MAX_ATTEMPTS:
            if summarize(rows, pairs, alpha)['complete']:
                return  # decided (futility stop): skip games not yet started
            attempts[(pair, arm)] = attempts.get((pair, arm), 0) + 1
            if await attempt(pair, arm):
                return

    async def attempt(pair: int, arm: str) -> bool:
        async with sem:
            game_seed = seed * 1_000_003 + pair
            us_side = 'p1' if pair % 2 == 0 else 'p2'
            them_side = 'p2' if us_side == 'p1' else 'p1'
            opp_team = pool[order[pair % len(order)]]
            value = candidate if arm == 'cand' else incumbent
            # Adaptive worlds are wall-clock bounded live; gate games run slower on
            # background cores, so lift the bound to measure the intended policy.
            us = SearchAgent(fmt=FMT, preview=preview, seed=game_seed,
                             **agent_kwargs(engines=engines, adaptive_max_ms=1e9, engine_timeout=900.0),
                             value_path=value.get('path'), value_beta=value.get('beta', 0.0))
            them = SearchAgent(fmt=FMT, preview=preview, seed=game_seed + 1,
                               **agent_kwargs(engines=1, adaptive_max_ms=1e9, engine_timeout=900.0))
            us.reseed(game_seed * 2 + 1)
            them.reseed(game_seed * 2 + 2)
            t0 = time.time()
            try:
                r = await play_game({us_side: us, them_side: them}, {us_side: team, them_side: opp_team},
                                    game_seed % (2 ** 63), timeout=300)
                winner = r.get('winner')
                score = 0.5 if winner in (None, 'tie') else float(winner == us_side)
                rec = {'pair': pair, 'arm': arm, 'win': score == 1.0, 'score': score, 'tie': winner == 'tie',
                       'unfinished': bool(r.get('unfinished')), 'turns': r.get('turns'),
                       'side': us_side, 'opp_team': order[pair % len(order)], 'seconds': round(time.time() - t0, 1),
                       'fallbacks': us.stats['fallbacks'], 'decisions': us.stats['decisions'],
                       'ms': round(us.stats['ms'] / max(1, us.stats['decisions']), 1)}
                if us.stats['fallbacks'] or them.stats['fallbacks']:
                    # A fallback move is an infrastructure failure, not the policy under
                    # test: retry this pair's seed (void after MAX_ATTEMPTS).
                    rec = {**rec, 'win': None, 'score': None,
                           'error': f"fallbacks us={us.stats['fallbacks']} them={them.stats['fallbacks']}: "
                                    + '; '.join((us.stats['errors'] + them.stats['errors'])[-2:])[:240]}
            except Exception as error:
                rec = {'pair': pair, 'arm': arm, 'win': None, 'error': repr(error)[:300]}
            finally:
                await us.close()
                await them.close()
            async with lock:
                with out.open('a') as fh:
                    fh.write(json.dumps(rec) + '\n')
                rows.append(rec)
                s = summarize(rows, pairs, alpha)
                log(json.dumps({'pair': pair, 'arm': arm, 'win': rec.get('win'), 'seconds': rec.get('seconds'),
                                'gains': s['gains'], 'losses': s['losses'], 'complete': s['pairs_complete']}))
            return 'error' not in rec

    # Interleave arms so an interruption leaves complete pairs, not one long arm.
    await asyncio.gather(*(one(p, arm) for p in range(pairs) for arm in ('cand', 'inc')))
    return summarize(rows, pairs, alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidate', required=True, help='value json path')
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--incumbent', help='incumbent value json (default: hand evaluation)')
    ap.add_argument('--incumbent-beta', type=float, default=0.0)
    ap.add_argument('--pairs', type=int, default=60)
    ap.add_argument('--concurrency', type=int, default=3)
    ap.add_argument('--engines', type=int, default=1, help='engines per searching agent (memory)')
    ap.add_argument('--seed', type=int, default=20261009)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--team', default=DEFAULT_TEAM)
    ap.add_argument('--pool', type=Path, default=ROOT / 'data' / 'ml' / 'value' / 'pool-teams.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    team = json.loads((ROOT / 'teams.json').read_text())[args.team]['sets']
    pool = [t['sets'] if isinstance(t, dict) else t for t in json.loads(args.pool.read_text())]
    cand = {'path': args.candidate, 'beta': args.beta}
    inc = {'path': args.incumbent, 'beta': args.incumbent_beta if args.incumbent else 0.0}
    result = asyncio.run(run_gate(cand, inc, args.out, args.pairs, args.concurrency, args.seed, team, pool,
                                  args.engines, args.alpha))
    print(json.dumps({'candidate': cand, 'incumbent': inc, **result}))


if __name__ == '__main__':
    main()
