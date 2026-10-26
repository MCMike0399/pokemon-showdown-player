"""How much does the team itself win, after the players' ladder ratings are accounted for?

    .venv/bin/python search/human/team_effects.py --replays 'data/replays/regmc*/replays-*.jsonl' \
        --out data/ml/human/team-effects.json

Every public game gives two observations (one per side): the six Pokemon brought to
preview, the pre-game ladder rating of both players and the result. A team's effect is
its mean residual, score minus the Elo expectation from the rating difference, so a
team that is simply played by stronger players does not look strong. Residuals are
reported as Elo (the rating offset that would explain the mean residual at 50%).

Bots (name contains "bot") and our own account are excluded. A side's open team sheet
(|showteam|) is kept when present so the exact sets of a strong team can be reused.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ME = 'freepalestinamx'


def _id(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def is_bot(name: str) -> bool:
    return 'bot' in _id(name)


def parse(record: dict) -> dict | None:
    """Both sides of one public game, or None if it is unusable."""
    log = record.get('log') or ''
    names, ratings, six, sheets = {}, {}, defaultdict(list), {}
    winner, forfeit, turns = None, False, 0
    for line in log.split('\n'):
        if line.startswith('|player|'):
            parts = line.split('|')
            if len(parts) >= 4 and parts[3]:
                names[parts[2]] = parts[3]
                if len(parts) >= 6 and parts[5].strip().isdigit():
                    ratings[parts[2]] = int(parts[5])
        elif line.startswith('|poke|'):
            parts = line.split('|')
            six[parts[2]].append(parts[3].split(',')[0].strip())
        elif line.startswith('|showteam|'):
            parts = line.split('|', 3)
            sheets[parts[2]] = parts[3]
        elif line.startswith('|win|'):
            winner = line[5:].strip()
        elif line.startswith('|turn|'):
            turns = int(line[6:])
        elif 'forfeited' in line and line.startswith('|-message|'):
            forfeit = True
    if winner is None or set(names) != {'p1', 'p2'} or set(ratings) != {'p1', 'p2'}:
        return None
    if any(len(six[s]) != 6 for s in ('p1', 'p2')):
        return None
    sides = []
    for s, o in (('p1', 'p2'), ('p2', 'p1')):
        sides.append({'name': names[s], 'rating': ratings[s], 'opp_rating': ratings[o],
                      'six': sorted(six[s]), 'opp_six': sorted(six[o]),
                      'score': 1.0 if _id(winner) == _id(names[s]) else 0.0,
                      'bot': is_bot(names[s]), 'opp_bot': is_bot(names[o]),
                      'me': _id(names[s]) == ME or _id(names[o]) == ME,
                      'sheet': sheets.get(s)})
    return {'id': record.get('id'), 'time': record.get('uploadtime'), 'turns': turns,
            'forfeit': forfeit, 'sides': sides}


def parse_file(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                game = parse(json.loads(line))
            except (ValueError, KeyError, IndexError):
                continue
            if game:
                out.append(game)
    return out


def expected(r: float, opp: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp - r) / 400.0))


def elo(mean_residual: float) -> float:
    p = min(max(0.5 + mean_residual, 1e-3), 1 - 1e-3)
    return 400.0 * math.log10(p / (1 - p))


def summarise(rows: list[tuple[float, float]]) -> dict:
    n = len(rows)
    res = [s - e for s, e in rows]
    mean = sum(res) / n
    var = sum((x - mean) ** 2 for x in res) / max(n - 1, 1)
    se = math.sqrt(var / n)
    return {'n': n, 'win': round(sum(s for s, _ in rows) / n, 4), 'expected': round(sum(e for _, e in rows) / n, 4),
            'residual': round(mean, 4), 'se': round(se, 4), 'elo': round(elo(mean), 1),
            'elo_lo': round(elo(mean - 1.96 * se), 1), 'elo_hi': round(elo(mean + 1.96 * se), 1)}


def player_effects(per_player: dict, player_all: dict, key: str) -> dict:
    """Team effect with every player weighted equally, and within-player (team vs the same
    player's other teams), so one strong player climbing a fresh account cannot carry a team."""
    means = [sum(v) / len(v) for v in per_player.values()]
    n = len(means)
    mean = sum(means) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in means) / max(n - 1, 1) / n)
    top = max(len(v) for v in per_player.values()) / sum(len(v) for v in per_player.values())
    diffs = []
    for name, res in per_player.items():
        other = [r for k, r in player_all[name] if k != key]
        if len(other) >= 10 and len(res) >= 5:
            diffs.append(sum(res) / len(res) - sum(other) / len(other))
    out = {'players': n, 'top_player_share': round(top, 3), 'player_elo': round(elo(mean), 1),
           'player_elo_lo': round(elo(mean - 1.96 * se), 1), 'player_elo_hi': round(elo(mean + 1.96 * se), 1)}
    if len(diffs) >= 5:
        dm = sum(diffs) / len(diffs)
        dse = math.sqrt(sum((x - dm) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
        out.update(within_players=len(diffs), within_elo=round(elo(dm), 1), within_se_elo=round(elo(dse) , 1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replays', default='data/replays/regmc*/replays-*.jsonl')
    ap.add_argument('--out', type=Path, default=Path('data/ml/human/team-effects.json'))
    ap.add_argument('--min-rating', type=int, default=0, help='Only sides whose own rating is at least this')
    ap.add_argument('--min-team-games', type=int, default=150)
    ap.add_argument('--min-players', type=int, default=15, help='Rank only teams used by at least this many players')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(args.replays))
    with ProcessPoolExecutor(args.workers) as pool:
        games = [g for chunk in pool.map(parse_file, files) for g in chunk]
    seen, uniq = set(), []
    for g in games:
        if g['id'] not in seen:
            seen.add(g['id'])
            uniq.append(g)
    by_team, by_species, sheets = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(int))
    players, player_all = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    total = []
    for g in uniq:
        for s in g['sides']:
            if s['bot'] or s['opp_bot'] or s['me'] or s['rating'] < args.min_rating:
                continue
            row = (s['score'], expected(s['rating'], s['opp_rating']))
            total.append(row)
            key = '|'.join(s['six'])
            by_team[key].append(row)
            players[key][_id(s['name'])].append(row[0] - row[1])
            player_all[_id(s['name'])].append((key, row[0] - row[1]))
            for sp in s['six']:
                by_species[sp].append(row)
            if s['sheet']:
                sheets[key][s['sheet']] += 1
    teams = []
    for key, rows in by_team.items():
        if len(rows) >= args.min_team_games:
            entry = {'six': key.split('|'), **summarise(rows), **player_effects(players[key], player_all, key)}
            if sheets.get(key):
                sheet, count = max(sheets[key].items(), key=lambda kv: kv[1])
                entry['sheet'], entry['sheet_games'] = sheet, count
            teams.append(entry)
    teams.sort(key=lambda t: -(t['player_elo'] if t['players'] >= args.min_players else -1e9))
    species = sorted(({'species': sp, **summarise(rows)} for sp, rows in by_species.items() if len(rows) >= 500),
                     key=lambda t: -t['elo'])
    report = {'replays': len(uniq), 'sides': len(total), 'min_rating': args.min_rating,
              'overall': summarise(total) if total else None, 'teams': teams, 'species': species}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"{len(uniq)} replays, {len(total)} sides, {len(teams)} teams with >= {args.min_team_games} games")
    for t in teams[:25]:
        print(f"{t['player_elo']:+6.1f} [{t['player_elo_lo']:+.0f},{t['player_elo_hi']:+.0f}] pooled {t['elo']:+6.1f} "
              f"within {t.get('within_elo', float('nan')):+6.1f}/{t.get('within_players', 0):3d} n={t['n']:5d} "
              f"players={t['players']:4d} top={t['top_player_share']:.2f} sheet={t.get('sheet_games', 0):4d}  {' '.join(t['six'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
