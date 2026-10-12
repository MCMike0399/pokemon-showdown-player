"""Weight ladder-replica teams by how the real game against them went.

    .venv/bin/python search/build_matchup_weights.py ladder-teams.json games.jsonl out.json [--loss-weight 3]

Each replica (search/build_ladder_teams.py) comes from one of our own live games
(`battle`). A replica of a game we lost gets weight `--loss-weight`, one we won 1,
an unknown result 1 (games before the team catalog, search/team_data.py). Self-play
(`--pool2 out.json`) then meets the matchups that beat us more often: prioritized
opponent sampling, as in prioritized fictitious self-play, without inventing teams.
Outcomes come from the catalog because stored public logs redact the winner.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def room_key(room: str) -> str:
    """battle-<format>-<number>: drops the `own-` prefix and a private room's password."""
    return '-'.join(room.removeprefix('own-').split('-')[:3])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('teams', type=Path)
    ap.add_argument('games', type=Path, help="the team's games.jsonl")
    ap.add_argument('out', type=Path)
    ap.add_argument('--loss-weight', type=float, default=3.0)
    args = ap.parse_args(argv)
    results = {}
    for line in args.games.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            if not g.get('tie'):
                results[room_key(g['room'])] = bool(g.get('win'))
    teams = json.loads(args.teams.read_text())
    counts = collections.Counter()
    for t in teams:
        res = results.get(room_key(t.get('battle') or ''))
        t['w'] = args.loss_weight if res is False else 1.0
        counts['lost' if res is False else 'won' if res else 'unknown'] += 1
    args.out.write_text(json.dumps(teams, indent=0))
    total = sum(t['w'] for t in teams)
    print(json.dumps({'teams': len(teams), **counts, 'loss_weight': args.loss_weight,
                      'share_of_draws_from_lost_games': round(counts['lost'] * args.loss_weight / total, 3)}))


if __name__ == '__main__':
    main()
