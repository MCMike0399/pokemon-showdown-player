"""Play one arena game with full per-turn search traces for diagnosis."""
import asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from search.arena import play_game, BrainAgent
from search.run_match import make_agent, load_teams

async def main(spec_a, spec_b, seed, side, out, team_b='Rain-Recife-special-stat-fix', team_index=0):
    shared = {}
    a = make_agent(spec_a, seed, shared); b = make_agent(spec_b, seed + 1, shared)
    if hasattr(a, 'log'): a.log = []
    ta = load_teams('Rain-Recife-special-stat-fix')[0]
    tb = load_teams(team_b)[team_index]
    other = 'p2' if side == 'p1' else 'p1'
    r = await play_game({side: a, other: b}, {side: ta, other: tb}, seed)
    Path(out).write_text(json.dumps({'result': {k: v for k, v in r.items() if k != 'log_p1'}, 'a_side': side,
                                     'trace': getattr(a, 'log', None), 'log_p1': r['log_p1']}, indent=1))
    print(r['winner'], side, r['turns'])
    for ag in (a, b):
        await ag.close()

if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5],
                     *(sys.argv[6:7] or []), *([int(sys.argv[7])] if len(sys.argv) > 7 else [])))
