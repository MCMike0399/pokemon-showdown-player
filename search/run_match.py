"""Run a batch of arena games between two agent configs, in parallel; JSONL results.

Example:
  .venv/bin/python search/run_match.py --a search --b incumbent --games 40 --concurrency 4 \
      --out artifacts/search-breakthrough/runs/mirror-v1.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search.arena import BrainAgent, RandomAgent, play_game  # noqa: E402

FROZEN = ROOT / 'artifacts' / 'search-breakthrough' / 'frozen'


def make_agent(spec: str, seed: int, shared: dict):
    """spec: 'incumbent' | 'random' | 'search[:key=val,...]'"""
    name, _, opts = spec.partition(':')
    kw = {}
    for part in filter(None, opts.split(';')):
        k, _, v = part.partition('=')
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    if name == 'incumbent':
        if 'incumbent' not in shared:
            shared['incumbent'] = BrainAgent(FROZEN)
        return shared['incumbent']
    if name == 'random':
        return RandomAgent(seed)
    if name == 'search':
        from search.agent import SearchAgent
        if 'incumbent' not in shared:
            shared['incumbent'] = BrainAgent(FROZEN)
        preview = shared['incumbent'].preview_fn()
        return SearchAgent(seed=seed, preview=preview, **kw)
    raise ValueError(spec)


def load_teams(spec: str) -> list[list[dict]]:
    if spec.endswith('.json') and Path(spec).exists():
        data = json.loads(Path(spec).read_text())
        if isinstance(data, dict):
            return [v['sets'] if isinstance(v, dict) else v for v in data.values()]
        return [t['sets'] if isinstance(t, dict) else t for t in data]
    teams = json.loads((ROOT / 'teams.json').read_text())
    return [teams[name]['sets'] for name in spec.split('+')]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--team-a', default='Rain-Recife-special-stat-fix')
    ap.add_argument('--team-b', default='Rain-Recife-special-stat-fix')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    teams_a = load_teams(args.team_a)
    teams_b = load_teams(args.team_b)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            done.add(json.loads(line)['index'])
    shared = {}
    sem = asyncio.Semaphore(args.concurrency)
    stats = {'a': 0, 'b': 0, 'tie': 0, 'n': 0}

    async def one(i):
        if i in done:
            return
        async with sem:
            seed = args.seed + i // 2
            a_side = 'p1' if i % 2 == 0 else 'p2'
            b_side = 'p2' if a_side == 'p1' else 'p1'
            ta = teams_a[(i // 2) % len(teams_a)]
            tb = teams_b[(i // 2) % len(teams_b)]
            agent_a = make_agent(args.a, seed * 7 + 1, shared)
            agent_b = make_agent(args.b, seed * 7 + 2, shared)
            t0 = time.time()
            try:
                r = await play_game({a_side: agent_a, b_side: agent_b}, {a_side: ta, b_side: tb}, seed)
            except Exception as error:
                r = {'error': repr(error)[:500], 'winner': None}
            finally:
                for ag in (agent_a, agent_b):
                    if ag is not shared.get('incumbent'):
                        await ag.close()
            winner = r.get('winner')
            label = 'a' if winner == a_side else 'b' if winner == b_side else 'tie'
            rec = {'index': i, 'seed': seed, 'a_side': a_side, 'result': label, 'turns': r.get('turns'),
                   'seconds': round(time.time() - t0, 1), 'error': r.get('error'),
                   'rejections': r.get('rejections'), 'decision_ms': r.get('decision_ms'),
                   'decisions': r.get('decisions'), 'team_b': (i // 2) % len(teams_b),
                   'search_stats': getattr(agent_a, 'stats', None) or getattr(agent_b, 'stats', None)}
            with out.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')
            stats[label] += 1
            stats['n'] += 1
            print(f"[{stats['n']}] game {i} {label} turns={r.get('turns')} {rec['seconds']}s  "
                  f"A {stats['a']} - B {stats['b']} (ties {stats['tie']})", flush=True)

    await asyncio.gather(*(one(i) for i in range(args.games)))


if __name__ == '__main__':
    asyncio.run(main())
