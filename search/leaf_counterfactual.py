"""Re-search recorded live positions under the linear leaf and the win-probability leaf.

Both arms run the deployed configuration (search/config.py) on the same sampled worlds
(same reseed per position, common random numbers), so a changed choice is caused by the
leaf alone. Positions come from the live ledgers: a `stall` sample (we chose Protect in a
position the search itself scored below `--behind`) and a `random` sample of all turn-start
positions. Output: one JSON line per position plus a summary on stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battle_state import legal_choices  # noqa: E402
from search.agent import SearchAgent, move_info  # noqa: E402
from search.config import agent_kwargs  # noqa: E402
from search.ledger_positions import FMT, _prefix, load_games  # noqa: E402
from search.team_data import load_teams  # noqa: E402
from search.tracker import PROTECT_MOVES  # noqa: E402

TEAM = 'Rain-Recife-special-stat-fix'


def protect_slots(request: dict, choice: str) -> int:
    act = request.get('active') or []
    n = 0
    for slot, part in enumerate(p.strip() for p in choice.split(',')):
        w = part.split()
        if w and w[0] == 'move' and slot < len(act):
            j = int(w[1]) - 1
            moves = act[slot].get('moves') or []
            if 0 <= j < len(moves) and (moves[j].get('id') or '') in PROTECT_MOVES:
                n += 1
    return n


def describe(request: dict, choice: str) -> str:
    act = request.get('active') or []
    out = []
    for slot, part in enumerate(p.strip() for p in choice.split(',')):
        w = part.split()
        if w and w[0] == 'move' and slot < len(act):
            j = int(w[1]) - 1
            moves = act[slot].get('moves') or []
            name = moves[j].get('move') if 0 <= j < len(moves) else '?'
            out.append(name + (' ->' + w[2] if len(w) > 2 else ''))
        else:
            out.append(part)
    return ' + '.join(out)


def positions(runs: list[Path], behind: float, n_stall: int, n_random: int, seed: int):
    with tempfile.TemporaryDirectory() as tmp:
        for r in runs:
            (Path(tmp) / r.name).symlink_to(r.resolve())
        games = load_games(Path(tmp))
    stall, rest = [], []
    for room, game in games.items():
        for item in game['decisions'].values():
            r = item['request']
            s = item.get('search') or {}
            if r.get('teamPreview') or r.get('forceSwitch') or r.get('wait') or not r.get('active') or s.get('kind') != 'move':
                continue
            prefix = _prefix(game['log'], item.get('turn')) if item.get('turn') else None
            if not prefix:
                continue
            won = game.get('winner') == r['side']['name']
            row = (room, item, prefix, won)
            top = (s.get('top') or [[None, 0]])[0][1]
            if protect_slots(r, item['choice']) and top < behind:
                stall.append(row)
            else:
                rest.append(row)
    rng = random.Random(seed)
    rng.shuffle(stall)
    rng.shuffle(rest)
    return [('stall', *x) for x in stall[:n_stall]] + [('random', *x) for x in rest[:n_random]]


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs='+', required=True, help='browser-run directories')
    ap.add_argument('--a', type=float, default=0.801)
    ap.add_argument('--b', type=float, default=-0.430)
    ap.add_argument('--behind', type=float, default=-2.0)
    ap.add_argument('--stall', type=int, default=60)
    ap.add_argument('--random', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cand', type=json.loads, help='candidate search overrides (JSON); default the leaf from --a/--b')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sets = load_teams()[TEAM]['sets']
    pos = positions([Path(r) for r in args.runs], args.behind, args.stall, args.random, args.seed)
    leaf = {'a': args.a, 'b': args.b}
    arms = {'linear': SearchAgent(fmt=FMT, **agent_kwargs()),
            'winprob': SearchAgent(fmt=FMT, **agent_kwargs(**(args.cand or {'leaf': leaf})))}
    summary = {}
    try:
        with args.out.open('w') as fh:
            for k, (kind, room, item, prefix, won) in enumerate(pos):
                request = item['request']
                ctx = {'request': request, 'choices': legal_choices(request), 'team_sets': sets, 'public_log': prefix}
                row = {'kind': kind, 'room': room, 'turn': item.get('turn'), 'won': won, 'live': item['choice'],
                       'live_top': ((item.get('search') or {}).get('top') or [[None, None]])[0][1]}
                for name, agent in arms.items():
                    agent.reseed(args.seed * 100003 + k)
                    choice = await agent.decide(ctx)
                    tr = agent.last_trace or {}
                    row[name] = {'choice': choice, 'say': describe(request, choice), 'protect': protect_slots(request, choice),
                                 'top': tr.get('top'), 'margin': tr.get('margin'), 'ms': tr.get('ms')}
                row['changed'] = row['linear']['choice'] != row['winprob']['choice']
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                s = summary.setdefault(kind, {'n': 0, 'changed': 0, 'protect_linear': 0, 'protect_winprob': 0})
                s['n'] += 1
                s['changed'] += row['changed']
                s['protect_linear'] += row['linear']['protect'] > 0
                s['protect_winprob'] += row['winprob']['protect'] > 0
                print(f"{k + 1}/{len(pos)} {kind} T{row['turn']} changed={row['changed']} "
                      f"linear=[{row['linear']['say']}] winprob=[{row['winprob']['say']}]", flush=True)
    finally:
        for agent in arms.values():
            await agent.close()
    print(json.dumps(summary))


if __name__ == '__main__':
    asyncio.run(main())
