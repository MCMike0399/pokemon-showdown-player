"""Turn finished live search games into labelled value-learning positions.

Every browser run keeps a private ledger (`data/ml/browser-runs/*/events.jsonl`)
with each decision's exact request and the room's full public log at the end.
This tool rebuilds the position the search saw at each turn-start decision:
the same tracker, the same usage-prior determinization and the same simulator
rebuild as `SearchAgent`, then labels it with the game's verified result.

Rows (one per decision per sampled world) are appended to
`data/ml/value/teams/<team>/live/live-<room>.jsonl` (search/team_data.py):
    {x, y, t, g, s, h, w, src: 'live', time, rating, team}
Each game is rebuilt with the exact sets of the team that played it.
`g` is the room, so training and evaluation can split by game. A room is
processed once; its file is written atomically.

Live rows are a held-out set on the deployment distribution (~10 positions and
one outcome per game). They measure whether a learned value predicts real
ladder results better than the hand evaluation; self-play supplies volume.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battle_state import legal_choices  # noqa: E402
from search.agent import SearchAgent  # noqa: E402
from search.tracker import Tracker  # noqa: E402

FMT = 'gen9championsvgc2026regmc'


def _as_list(log):
    if isinstance(log, list):
        return log
    if isinstance(log, str):
        try:
            value = ast.literal_eval(log)
            return value if isinstance(value, list) else None
        except (ValueError, SyntaxError):
            return None
    return None


def load_games(runs: Path) -> dict[str, dict]:
    """room -> {decisions: [...], log: [...], winner, me, side, time}; finished games only."""
    games: dict[str, dict] = {}
    for path in sorted(runs.glob('*/events.jsonl')):
        for line in path.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get('kind')
            if kind == 'decision':
                room = event.get('room')
                request = event.get('request') or {}
                decision = event.get('decision') or {}
                if not room or not decision.get('decider'):
                    continue  # only search-decided turns
                game = games.setdefault(room, {'decisions': {}})
                key = request.get('rqid') or json.dumps(request, sort_keys=True)[:200]
                game['decisions'][key] = {'request': request, 'time': event.get('time'),
                                          'turn': (event.get('state') or {}).get('turn'),
                                          'ms': decision.get('inference_ms'), 'choice': decision.get('choice'),
                                          'search': decision.get('search')}
            elif kind == 'terminal':
                result = event.get('result') or {}
                room = result.get('room')
                log = _as_list(event.get('log'))
                if not room or not log:
                    continue
                game = games.setdefault(room, {'decisions': {}})
                game.update(log=log, winner=result.get('winner'), time=event.get('time'))
    return {room: g for room, g in games.items() if g.get('log') and g['decisions']}


def _prefix(log: list[str], turn: int) -> list[str] | None:
    """Public log up to and including `|turn|<turn>` (what the search saw)."""
    marker = '|turn|' + str(turn)
    for i, line in enumerate(log):
        if line == marker:
            return log[:i + 1]
    return None


def _rating(log: list[str], name: str) -> int | None:
    for line in log:
        if line.startswith('|player|'):
            parts = line.split('|')
            if len(parts) > 5 and parts[3] == name and parts[5].isdigit():
                return int(parts[5])
    return None


async def game_rows(agent: SearchAgent, room: str, game: dict, team_sets: list[dict], worlds: int) -> tuple[list[dict], dict]:
    log = game['log']
    winner = game.get('winner')
    rows, skipped = [], {}
    by_turn = {}
    for item in sorted(game['decisions'].values(), key=lambda d: d.get('time') or 0):
        request = item['request']
        if request.get('teamPreview') or request.get('forceSwitch') or request.get('wait') or not request.get('active'):
            skipped['not_turn_start'] = skipped.get('not_turn_start', 0) + 1
            continue
        by_turn[item.get('turn')] = item  # a rejected proposal's retry replaces it
    for item in by_turn.values():
        request = item['request']
        side = request['side']['id']
        me = request['side']['name']
        turn = item.get('turn')  # the observed turn the decision answered
        prefix = _prefix(log, turn) if turn else None
        if prefix is None:
            skipped['no_turn'] = skipped.get('no_turn', 0) + 1
            continue
        tracker = Tracker(side).feed(prefix)
        ctx = {'request': request, 'choices': legal_choices(request), 'team_sets': team_sets, 'public_log': prefix}
        try:
            our = agent._our_side(ctx, tracker)
            if any(a.get('canMegaEvo') for a in request.get('active') or []):
                tracker.mega_used[side] = False
            field = agent._field(tracker)
            sampled = [{'p1': our, 'p2': agent._their_side(tracker, agent.rng)} for _ in range(worlds)]
            out = await agent.engines[0].call({'type': 'features', 'format': FMT, 'field': field, 'worlds': sampled})
        except Exception as error:  # a position we cannot rebuild is skipped, not guessed
            skipped['rebuild_error'] = skipped.get('rebuild_error', 0) + 1
            skipped.setdefault('errors', []).append(repr(error)[:200])
            continue
        y = 0.5 if not winner else (1.0 if winner == me else 0.0)
        for w, r in enumerate(out['results']):
            if 'error' in r:
                continue
            rows.append({'x': r['x'], 'y': y, 't': turn, 'g': room, 's': side, 'h': round(r['hand'], 4), 'w': w,
                         'src': 'live', 'time': item.get('time'), 'rating': _rating(log, me),
                         'choice': item.get('choice')})
    return rows, skipped


async def main():
    from search.team_data import TEAMS_ROOT, build_catalog, ensure_team, load_teams, scan_ledgers
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=ROOT / 'data' / 'ml' / 'browser-runs')
    ap.add_argument('--teams-root', type=Path, default=TEAMS_ROOT)
    ap.add_argument('--team', help='only this team (default: every team, each into its own directory)')
    ap.add_argument('--worlds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--force', action='store_true', help='rebuild rooms that already have a file')
    args = ap.parse_args()
    teams = load_teams()
    owners = {room: g.get('team') for room, g in scan_ledgers(args.runs).items()}
    agent = SearchAgent(fmt=FMT, worlds=args.worlds, engines=1, seed=args.seed)
    summary = {'rooms': 0, 'rows': 0, 'skipped': {}, 'new_rooms': 0, 'by_team': {}}
    try:
        for room, game in load_games(args.runs).items():
            summary['rooms'] += 1
            name = owners.get(room)
            if name not in teams or (args.team and name != args.team):
                summary['skipped']['unknown_team' if name not in teams else 'other_team'] = \
                    summary['skipped'].get('unknown_team' if name not in teams else 'other_team', 0) + 1
                continue
            path = ensure_team(name, args.teams_root) / 'live' / ('live-' + room.replace('/', '_') + '.jsonl')
            if path.exists() and not args.force:
                continue
            agent.rng = random.Random(f'{args.seed}:{room}')
            rows, skipped = await game_rows(agent, room, game, teams[name]['sets'], args.worlds)
            for k, v in skipped.items():
                if k != 'errors':
                    summary['skipped'][k] = summary['skipped'].get(k, 0) + v
            if not rows:
                continue
            for row in rows:
                row['team'] = name
            tmp = path.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            tmp.replace(path)
            summary['rows'] += len(rows)
            summary['new_rooms'] += 1
            summary['by_team'][name] = summary['by_team'].get(name, 0) + 1
    finally:
        await agent.close()
    summary['catalog'] = {n: t['games'] for n, t in build_catalog(args.teams_root, args.runs)['teams'].items()}
    print(json.dumps(summary))


if __name__ == '__main__':
    asyncio.run(main())
