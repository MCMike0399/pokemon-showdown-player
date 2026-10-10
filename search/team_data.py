"""Per-team organisation of everything the live search and its learning produce.

Every team our account plays gets one directory, named by a stable slug of its
`teams.json` name:

    data/ml/value/teams/<slug>/
        team.json          name, format, team_id (fingerprint of the exact sets), sets
        games.jsonl        one line per finished live game (catalog, rebuilt from ledgers)
        live/              labelled live positions, one file per room
        selfplay/          self-play shards with this team as the focus side
        candidates/        value nets trained for this team (immutable, by weights hash)
        gates/             paired gate results for those candidates
        current.json       the gated value the live search uses for this team
    data/ml/value/teams/index.json   all teams with game counts and date ranges

A value net is trained on one team's self-play and calibrated on that team's
live games, so candidates, gates and promotions are per team. The raw ledgers
stay in `data/ml/browser-runs/` (one run = one team); each terminal event and
each run's `team` event name the team, and older runs are identified by
matching our request's sets to `teams.json`.
"""
from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAMS_FILE = ROOT / 'teams.json'
TEAMS_ROOT = ROOT / 'data' / 'ml' / 'value' / 'teams'
RUNS = ROOT / 'data' / 'ml' / 'browser-runs'
DEFAULT_TEAM = 'Rain-Recife-special-stat-fix'
SUBDIRS = ('live', 'selfplay', 'candidates', 'gates')


def slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'unnamed'


def _id(text) -> str:
    return re.sub(r'[^a-z0-9]', '', str(text or '').lower())


def load_teams() -> dict:
    return json.loads(TEAMS_FILE.read_text())


def team_dir(name: str, root: Path | None = None) -> Path:
    path = Path(root or TEAMS_ROOT) / slug(name)
    for sub in SUBDIRS:
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def ensure_team(name: str, root: Path | None = None) -> Path:
    """Create the team's directory and its team.json (sets copied from teams.json)."""
    from ml.teams import team_id
    path = team_dir(name, root)
    target = path / 'team.json'
    try:
        team = load_teams()[name]
    except (OSError, ValueError, KeyError):
        # teams.json is private (gitignored); keep whatever the directory already has.
        if not target.exists():
            target.write_text(json.dumps({'name': name, 'slug': slug(name)}, indent=2))
        return path
    meta = {'name': name, 'slug': slug(name), 'format': team['format'],
            'team_id': team_id(team['format'], team['sets']), 'sets': team['sets']}
    if not target.exists() or json.loads(target.read_text()) != meta:
        target.write_text(json.dumps(meta, indent=2))
    return path


def team_sets(name: str) -> list[dict]:
    return load_teams()[name]['sets']


_STATS: dict[str, dict] = {}


def team_stats(teams: dict) -> dict:
    """name -> [(species id, exact level-50 stats)] from the pinned simulator (cached)."""
    missing = {n: t for n, t in teams.items() if n not in _STATS}
    if missing:
        import subprocess
        out = subprocess.run(['node', str(ROOT / 'search' / 'team_stats.cjs')], input=json.dumps(missing), capture_output=True, text=True,
                             timeout=120, check=True).stdout
        _STATS.update(json.loads(out))
    return {n: _STATS.get(n) for n in teams}


def identify(request: dict, teams: dict | None = None, prefer: str = DEFAULT_TEAM) -> str | None:
    """The teams.json name whose exact stats match our side of a request.

Several saved teams share species and items and differ only in stat points, so
species alone is ambiguous; the request's stats are exact. If identical teams
remain (true duplicates), `prefer` wins, then the first name.
"""
    teams = teams if teams is not None else load_teams()
    ours = []
    for mon in (request.get('side') or {}).get('pokemon') or []:
        species = _id(re.sub(r'-Mega(-[XYZ])?$', '', mon.get('details', '').split(',')[0]))
        cond = str(mon.get('condition', '')).split()[0] if mon.get('condition') else ''
        maxhp = int(cond.split('/')[1]) if '/' in cond and cond.split('/')[1].isdigit() else None
        ours.append((species, mon.get('stats') or {}, maxhp))
    if not ours:
        return None
    matches = []
    for name, stats in team_stats(teams).items():
        if not stats:
            continue
        table = {}
        for species, st in stats:
            table.setdefault(species, []).append(st)
        def fits(species, st, maxhp):
            for cand in table.get(species, []):
                if all(cand.get(k) == v for k, v in st.items()) and (maxhp is None or cand.get('hp') == maxhp):
                    return True
            # A Mega-evolved request reports Mega stats; fall back to species only.
            return species in table and not st
        if all(fits(*m) for m in ours):
            matches.append(name)
    if not matches:
        return None
    return prefer if prefer in matches else sorted(matches)[0]


def _player_line(log: list[str], side: str | None):
    for line in log:
        parts = line.split('|')
        if line.startswith('|player|') and len(parts) > 3 and parts[2] == side:
            return parts
    return None


def scan_ledgers(runs: Path | None = None) -> dict[str, dict]:
    """room -> finished-game summary with its team, from every run ledger."""
    teams = load_teams()
    games: dict[str, dict] = {}
    first_request: dict[str, dict] = {}  # across runs: a resumed room spans two ledgers
    for path in sorted(Path(runs or RUNS).glob('*/events.jsonl')):
        run_team = None
        for line in path.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get('kind')
            if kind == 'team':
                run_team = event.get('name')
            elif kind == 'decision':
                room = event.get('room')
                if room:
                    game = games.setdefault(room, {'decisions': 0, 'search_turns': 0})
                    game['decisions'] += 1
                    game['search_turns'] += bool((event.get('decision') or {}).get('decider'))
                    first_request.setdefault(room, event.get('request') or {})
            elif kind in ('team_sheets_accepted',):
                games.setdefault(event.get('room'), {'decisions': 0, 'search_turns': 0})['ots'] = True
            elif kind == 'terminal':
                result = event.get('result') or {}
                room = result.get('room')
                log = event.get('log')
                log = log if isinstance(log, list) else ast.literal_eval(log) if isinstance(log, str) else []
                side = event.get('side') or ((first_request.get(room) or {}).get('side') or {}).get('id')
                me = _player_line(log, side)
                foe = _player_line(log, {'p1': 'p2', 'p2': 'p1'}.get(side))
                name = event.get('team') or run_team or identify(first_request.get(room) or {}, teams)
                game = games.setdefault(room, {'decisions': 0, 'search_turns': 0})
                game.update(
                    room=room, time=event.get('time'), run=path.parent.name, team=name,
                    win=bool(me and result.get('winner') == me[3]), tie=result.get('winner') in (None, ''),
                    rating=int(me[5]) if me and len(me) > 5 and me[5].isdigit() else None,
                    opp_rating=int(foe[5]) if foe and len(foe) > 5 and foe[5].isdigit() else None,
                    turns=max([int(x.split('|')[2]) for x in log if x.startswith('|turn|')] or [0]),
                    opp_preview=[x.split('|')[3].split(',')[0] for x in log
                                 if foe and x.startswith('|poke|' + foe[2] + '|')])
    return {room: g for room, g in games.items() if g.get('time')}


def build_catalog(root: Path | None = None, runs: Path | None = None) -> dict:
    """Rewrite every team's games.jsonl and index.json from the ledgers (idempotent)."""
    root = Path(root or TEAMS_ROOT)
    games = scan_ledgers(runs)
    by_team: dict[str, list] = {}
    for game in games.values():
        by_team.setdefault(game.get('team') or 'unidentified', []).append(game)
    index = {'updated': time.time(), 'teams': {}}
    teams = load_teams()
    for name, rows in sorted(by_team.items()):
        rows.sort(key=lambda g: g['time'])
        path = ensure_team(name, root) if name in teams else team_dir(name, root)
        for g in rows:
            live = path / 'live' / ('live-' + g['room'] + '.jsonl')
            g['positions'] = sum(1 for _ in live.open()) if live.exists() else 0
        tmp = path / 'games.jsonl.tmp'
        tmp.write_text(''.join(json.dumps(g) + '\n' for g in rows))
        tmp.replace(path / 'games.jsonl')
        wins = sum(g['win'] for g in rows)
        index['teams'][name] = {'slug': slug(name), 'games': len(rows), 'wins': wins, 'losses': len(rows) - wins,
                                'first': rows[0]['time'], 'last': rows[-1]['time'],
                                'search_games': sum(1 for g in rows if g['search_turns']),
                                'positions': sum(g['positions'] for g in rows)}
    tmp = Path(root) / 'index.json.tmp'
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(Path(root) / 'index.json')
    return index


if __name__ == '__main__':
    print(json.dumps(build_catalog()['teams'], indent=2))


def migrate_legacy(value_root: Path | None = None, state_path: Path | None = None) -> dict:
    """Move pre-team value data into teams/<slug>/ (one-time; idempotent).

Live positions go to the team that played their room; self-play shards,
candidates and gates produced before teams existed all had the default team
as their focus. The service state's candidate paths are rewritten to match.
Run only while the value service and its producers/gates are stopped: they
write by path.
"""
    import shutil
    value_root = Path(value_root or TEAMS_ROOT.parent)
    moved = {'live': 0, 'selfplay': 0, 'candidates': 0, 'gates': 0, 'pointer': 0}
    owners = {room: g.get('team') for room, g in scan_ledgers().items()}
    teams = load_teams()
    for src in sorted((value_root / 'live').glob('live-*.jsonl')) if (value_root / 'live').exists() else []:
        room = src.stem[len('live-'):]
        name = owners.get(room) if owners.get(room) in teams else DEFAULT_TEAM
        dest = ensure_team(name, value_root / 'teams') / 'live' / src.name
        rows = [json.loads(line) for line in src.open() if line.strip()]
        dest.write_text(''.join(json.dumps({**r, 'team': name}) + '\n' for r in rows))
        src.unlink()
        moved['live'] += 1
    default = ensure_team(DEFAULT_TEAM, value_root / 'teams')
    for sub in ('selfplay', 'candidates', 'gates'):
        for src in sorted((value_root / sub).glob('*')) if (value_root / sub).exists() else []:
            if src.is_file():
                shutil.move(str(src), str(default / sub / src.name))
                moved[sub] += 1
    if (value_root / 'current.json').exists():
        shutil.move(str(value_root / 'current.json'), str(default / 'current.json'))
        moved['pointer'] = 1
    if (value_root / 'focus-team.json').exists():
        (value_root / 'focus-team.json').unlink()  # superseded by teams/<slug>/team.json
    state_path = Path(state_path or value_root / 'service-state.json')
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for cand in state.get('candidates', []):
            old = Path(cand.get('path', ''))
            if old.parent == value_root / 'candidates':
                cand['path'] = str(default / 'candidates' / old.name)
            cand.setdefault('team', DEFAULT_TEAM)
        state['producers'] = {}
        state['team'] = state.get('team') or DEFAULT_TEAM
        state_path.write_text(json.dumps(state, indent=2))
    for sub in ('live', 'selfplay', 'candidates', 'gates'):
        try:
            (value_root / sub).rmdir()
        except OSError:
            pass
    moved['catalog'] = {n: t['games'] for n, t in build_catalog(value_root / 'teams')['teams'].items()}
    return moved
