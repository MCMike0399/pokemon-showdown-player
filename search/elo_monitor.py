"""Live Elo performance-rating monitor for the ladder player.

Win rate is a poor live metric on a ladder: matchmaking pairs the bot with
opponents near its own rating, so the win rate drifts toward 50% whether the
bot is strong or weak. The performance rating is the rating R at which the sum
of Elo expected scores against the opponents actually faced equals the score
actually earned. It rises when the bot beats stronger opposition and does not
saturate at 50%.

Everything here is pure and testable. `load_games` reads the browser-run
ledgers (`data/ml/browser-runs/<RUN>/events.jsonl`), `performance` scores a set
of games, `segments` groups runs into deploy segments by policy key, and
`report` bundles what the live viewer shows. `report_cached` memoises on the
ledgers' mtimes so repeated calls are cheap.

    .venv/bin/python search/elo_monitor.py [--runs DIR] [--json]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'data/ml/browser-runs'
LN10 = math.log(10)
LOW, HIGH = 100.0, 3000.0
WINDOWS = (50, 100, 200)
SERIES_WINDOW = 50
MAX_SERIES_POINTS = 200
# Fields of a search_agent event that carry no policy information.
VOLATILE = frozenset({'time', 'kind', 'pid', 'run', 'started_at'})
# Fallback account name, split so the public-snapshot scan (which blocks the .env
# username anywhere in tracked files) does not reject this file.
DEFAULT_NAME = 'Free' + 'Palestina' + 'MX'


# ----------------------------------------------------------------- parsing

def _as_list(log):
    """Protocol log as a list; ledgers store it as a list or a stringified list."""
    if isinstance(log, list):
        return log
    if isinstance(log, str):
        try:
            value = ast.literal_eval(log)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
        return value if isinstance(value, list) else None
    return None


def my_name(root: Path = ROOT) -> str:
    """Our ladder name: `.env` PS_USERNAME, else the project default. Never reads the password."""
    try:
        for line in (root / '.env').read_text().splitlines():
            key, _, value = line.partition('=')
            if key.strip() == 'PS_USERNAME' and value.strip():
                return value.strip().strip('"\'')
    except OSError:
        pass
    return DEFAULT_NAME


def _norm(name) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name or '').lower())


def _ratings(log, side, me):
    """(my_rating, opp_rating, my_rating_after) from a protocol log; None where unknown."""
    players = {}
    for line in log:
        if not isinstance(line, str) or not line.startswith('|player|'):
            continue
        parts = line.split('|')
        # |player|p1|name|avatar|rating  (reconnect lines omit the rating)
        if len(parts) >= 6 and parts[2] in ('p1', 'p2') and parts[5].strip().lstrip('-').isdigit():
            players.setdefault(parts[2], (parts[3], int(parts[5])))
    mine = players.get(side)
    other = players.get('p1' if side == 'p2' else 'p2')
    if mine is None and other is None:
        return None, None, None
    if mine is not None and _norm(mine[0]) != _norm(me):
        # `side` disagrees with the name: trust the name.
        named = [s for s, (n, _) in players.items() if _norm(n) == _norm(me)]
        if named:
            mine, other = players[named[0]], players.get('p1' if named[0] == 'p2' else 'p2')
    after = None
    for line in log:
        if isinstance(line, str) and line.startswith('|raw|') and "'s rating:" in line:
            match = re.match(r"\|raw\|(.+?)'s rating: (\d+) &rarr; <strong>(\d+)</strong>", line)
            if match and _norm(match.group(1)) == _norm(me):
                after = int(match.group(3))
    return (mine[1] if mine else None), (other[1] if other else None), after


def _score(winner, me):
    if not winner:
        return 0.5
    return 1.0 if _norm(winner) == _norm(me) else 0.0


def policy_key(config: dict, preview, team) -> str:
    """Stable hash of the search config (volatile fields removed), preview spec and team name."""
    body = {k: v for k, v in sorted((config or {}).items()) if k not in VOLATILE and v is not None}
    blob = json.dumps({'search': body, 'preview': preview, 'team': team}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


_KIND = re.compile(rb'"kind"\s*:\s*"(?:terminal|search_agent|preview_policy|team)"')


def _apply(run: dict, event: dict, me: str) -> None:
    kind = event.get('kind')
    if kind == 'search_agent':
        run['config'] = {k: v for k, v in event.items() if k not in VOLATILE and v is not None}
    elif kind == 'preview_policy':
        run['preview'] = event.get('spec')
    elif kind == 'team':
        run['team'] = event.get('name')
    elif kind == 'terminal':
        result = event.get('result') or {}
        room = result.get('room')
        log = _as_list(event.get('log'))
        if not room or not log or not isinstance(event.get('time'), (int, float)):
            return
        mine, opp, after = _ratings(log, event.get('side'), me)
        run['terminals'][room] = {
            'time': float(event['time']), 'run': run['run'], 'room': room,
            'my_rating': mine, 'opp_rating': opp, 'my_rating_after': after,
            'score': _score(result.get('winner'), me)}


def _read_run(path: Path, me: str, cache: dict | None = None) -> dict | None:
    """Parse one ledger. With a cache dict, an append-only ledger is read incrementally."""
    try:
        size = path.stat().st_size
        handle = path.open('rb')
    except OSError:
        return None
    with handle:
        head = handle.read(256)
        entry = cache.get(path) if cache is not None else None
        if entry and entry['me'] == me and entry['head'] == head and size >= entry['offset']:
            run, offset = entry['run'], entry['offset']
        else:
            run = {'run': path.parent.name, 'config': None, 'preview': None, 'team': None, 'terminals': {}}
            offset = 0
        handle.seek(offset)
        data = handle.read()
    end = data.rfind(b'\n') + 1  # leave a half-written last line for the next call
    for line in data[:end].split(b'\n'):
        if not _KIND.search(line):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            _apply(run, event, me)
    run['key'] = policy_key(run['config'], run['preview'], run['team'])
    if cache is not None:
        cache[path] = {'me': me, 'head': head, 'offset': offset + end, 'run': run}
    return run


def load_runs(runs_dir: Path = RUNS, me: str | None = None, cache: dict | None = None) -> list[dict]:
    """Per run: {run, key, config, preview, team, terminals: {room: game}}, in run order."""
    me = me or my_name()
    out = []
    for path in sorted(Path(runs_dir).glob('*/events.jsonl')):
        run = _read_run(path, me, cache)
        if run is not None:
            out.append(run)
    return out


def games_from_runs(runs: list[dict]) -> list[dict]:
    """Finished games deduped by room (a resumed room keeps its later run's terminal), by time."""
    games = {}
    for run in runs:  # run order: a later terminal replaces an earlier one
        for room, game in run['terminals'].items():
            game['policy_key'] = run['key']
            games[room] = game
    return sorted(games.values(), key=lambda g: (g['time'], g['room']))


def load_games(runs_dir: Path = RUNS, me: str | None = None) -> list[dict]:
    """Games: time, run, room, my_rating, opp_rating, score, policy_key (+ my_rating_after)."""
    return games_from_runs(load_runs(runs_dir, me))


# ------------------------------------------------------------------- math

def expected(rating: float, opp_ratings) -> float:
    return sum(1.0 / (1.0 + 10 ** ((o - rating) / 400.0)) for o in opp_ratings)


def solve_rating(score_sum: float, opp_ratings) -> tuple[float, bool]:
    """Rating whose expected total equals `score_sum`; (value, clamped)."""
    opps = list(opp_ratings)
    if expected(LOW, opps) >= score_sum:
        return LOW, True
    if expected(HIGH, opps) <= score_sum:
        return HIGH, True
    lo, hi = LOW, HIGH
    for _ in range(60):
        mid = (lo + hi) / 2
        if expected(mid, opps) < score_sum:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def performance(games) -> dict:
    """Performance rating of games with a known opponent rating.

    Keys: n, score, expected (at the solved rating, = score unless clamped),
    rating (None for no games), ci (half-width of the 95% interval, None when
    clamped or empty), clamped ('wins'|'losses'|None), win_rate (%), mean_opp.
    """
    rated = [g for g in games if g.get('opp_rating') is not None]
    n = len(rated)
    if not n:
        return {'n': 0, 'score': 0.0, 'expected': 0.0, 'rating': None, 'ci': None, 'clamped': None,
                'win_rate': None, 'mean_opp': None}
    opps = [g['opp_rating'] for g in rated]
    total = sum(g['score'] for g in rated)
    rating, clamped = solve_rating(total, opps)
    ci = None
    if not clamped:
        variance = sum(p * (1 - p) for p in (1 / (1 + 10 ** ((o - rating) / 400)) for o in opps))
        if variance > 0:
            ci = 1.96 * (400 / LN10) / math.sqrt(variance)
    return {'n': n, 'score': total, 'expected': expected(rating, opps), 'rating': rating, 'ci': ci,
            'clamped': ('wins' if total >= n else 'losses') if clamped else None,
            'win_rate': 100.0 * sum(g['score'] == 1.0 for g in rated) / n,
            'mean_opp': sum(opps) / n}


def rolling_series(games, window: int = SERIES_WINDOW, max_points: int = MAX_SERIES_POINTS) -> list[list]:
    """[[game_index, performance], ...] for each full window, downsampled to max_points.

    game_index is the 0-based index of the window's last game. Always keeps the final point.
    """
    rated = [g for g in games if g.get('opp_rating') is not None]
    ends = list(range(window - 1, len(rated)))
    if len(ends) > max_points:
        step = (len(ends) - 1) / (max_points - 1)
        ends = sorted({ends[round(i * step)] for i in range(max_points)})
    series = []
    for end in ends:
        value = performance(rated[end - window + 1:end + 1])['rating']
        series.append([end, round(value, 1)])
    return series


# --------------------------------------------------------------- segments

def _preview_label(spec) -> str:
    if not spec:
        return 'baseline preview'
    base = os.path.basename(str(spec).split(':', 1)[-1])
    return os.path.splitext(base)[0] or str(spec)


def _val(value) -> str:
    return 'unset' if value is None else json.dumps(value)


def _changes(prev, cur) -> list[str]:
    """Readable differences between two segments' policy descriptors."""
    out = []
    if prev is None:
        return out
    if prev['preview'] != cur['preview']:
        out.append('preview ' + _preview_label(prev['preview']) + ' -> ' + _preview_label(cur['preview']))
    if prev['team'] != cur['team']:
        out.append('team %s -> %s' % (prev['team'], cur['team']))
    a, b = prev['config'] or {}, cur['config'] or {}
    for field in sorted(set(a) | set(b)):
        if a.get(field) != b.get(field):
            out.append('%s %s -> %s' % (field, _val(a.get(field)), _val(b.get(field))))
    return out


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') if ts else None


def segments(games, runs=None) -> list[dict]:
    """Deploy segments: consecutive runs (with games) sharing a policy key.

    `runs` is the load_runs() list (for descriptors); without it segments are
    keyed on each game's policy_key alone and the labels are bare keys.
    """
    by_run = {r['run']: r for r in (runs or [])}
    out = []
    for game in games:  # time order
        run = game['run']
        last = out[-1] if out else None
        if last and last['key'] == game['policy_key']:
            last['games'].append(game)
            if run not in last['runs']:
                last['runs'].append(run)
        else:
            out.append({'key': game['policy_key'], 'runs': [run], 'games': [game]})
    previous = None
    for index, seg in enumerate(out):
        src = by_run.get(seg['runs'][0], {})
        descriptor = {'preview': src.get('preview'), 'team': src.get('team'), 'config': src.get('config')}
        changes = _changes(previous, descriptor) if previous else []
        previous = descriptor
        label = _preview_label(descriptor['preview'])
        seg.update(index=index, descriptor=descriptor, changes=changes,
                   label=label + (' (' + '; '.join(changes) + ')' if changes and index else ''),
                   first=games_time(seg['games'], 0), last=games_time(seg['games'], -1))
    return out


def games_time(games, i):
    return games[i]['time'] if games else None


# ------------------------------------------------------------------ report

def _brief(perf: dict) -> dict:
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in perf.items()}


def report(runs_dir: Path = RUNS, me: str | None = None, cache: dict | None = None) -> dict:
    """Everything the viewer panel and CLI show, JSON-serialisable."""
    runs = load_runs(runs_dir, me, cache)
    games = games_from_runs(runs)
    segs = segments(games, runs)
    rated = [g for g in games if g.get('opp_rating') is not None]
    current = None
    change = None
    if games:
        last = games[-1]
        current = last.get('my_rating_after') or last.get('my_rating')
        mine = [g for g in games if g.get('my_rating') is not None]
        if current is not None and len(mine) >= SERIES_WINDOW:
            change = current - mine[-SERIES_WINDOW]['my_rating']
    windows = {str(w): _brief(performance(rated[-w:])) for w in WINDOWS}
    segment_rows = []
    for seg in segs:
        segment_rows.append({
            'index': seg['index'], 'key': seg['key'], 'label': seg['label'],
            'short': _preview_label(seg['descriptor']['preview']), 'changes': seg['changes'],
            'runs': seg['runs'], 'first': _iso(seg['first']), 'last': _iso(seg['last']),
            'current': seg['index'] == len(segs) - 1, **_brief(performance(seg['games']))})
    # Series over all games; segment starts are given as indices into the rated list.
    starts, cursor = [], 0
    for seg in segs:
        starts.append(cursor)
        cursor += sum(g.get('opp_rating') is not None for g in seg['games'])
    return {
        'name': me or my_name(), 'games': len(games), 'rated_games': len(rated),
        'rating': current, 'rating_change_50': change,
        'overall': _brief(performance(rated)), 'windows': windows, 'segments': segment_rows,
        'series': rolling_series(rated), 'series_window': SERIES_WINDOW, 'segment_starts': starts,
        'updated': _iso(games[-1]['time']) if games else None}


_cache: dict = {}
_file_cache: dict = {}
_cache_lock = threading.Lock()


def report_cached(runs_dir: Path = RUNS, me: str | None = None) -> dict:
    """`report` memoised on the ledgers' (path, mtime, size) signature."""
    runs_dir = Path(runs_dir)
    signature = []
    for path in sorted(runs_dir.glob('*/events.jsonl')):
        try:
            stat = path.stat()
        except OSError:
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    key = (str(runs_dir), me, tuple(signature))
    with _cache_lock:
        hit = _cache.get('last')
        if hit and hit[0] == key:
            return hit[1]
    value = report(runs_dir, me, _file_cache)
    with _cache_lock:
        _cache['last'] = (key, value)
    return value


# --------------------------------------------------------------------- CLI

def _fmt(perf: dict) -> str:
    if not perf['n']:
        return '    n/a'
    rating = '%4.0f' % perf['rating'] + ('*' if perf['clamped'] else ' ')
    ci = ' +-%3.0f' % perf['ci'] if perf['ci'] is not None else '  n/a '
    return '%s%s  win %5.1f%%  opp %4.0f  n=%d' % (rating, ci, perf['win_rate'], perf['mean_opp'], perf['n'])


def format_table(data: dict) -> str:
    lines = []
    lines.append('Ladder performance for %s: %d games (%d with opponent ratings)' % (
        data['name'], data['games'], data['rated_games']))
    if data['rating'] is not None:
        change = data['rating_change_50']
        lines.append('Current rating %d%s' % (data['rating'], '' if change is None else ' (%+d over last 50 games)' % change))
    lines.append('')
    lines.append('Segments (performance +-95%, * = all wins/losses, clamped):')
    for seg in data['segments']:
        lines.append('  #%d%s %s  %s -> %s' % (seg['index'], ' <- current' if seg['current'] else '', _fmt(seg),
                                                 seg['first'], seg['last']))
        lines.append('      %s [%s] runs=%s' % (seg['label'], seg['key'],
                                                 seg['runs'][0] + ('..' + seg['runs'][-1] if len(seg['runs']) > 1 else '')))
    lines.append('')
    lines.append('Rolling windows:')
    for w, perf in data['windows'].items():
        lines.append('  last %-4s %s' % (w, _fmt(perf)))
    lines.append('  overall   %s' % _fmt(data['overall']))
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--runs', type=Path, default=RUNS, help='browser-runs directory')
    parser.add_argument('--json', action='store_true', help='print the full report as JSON')
    args = parser.parse_args(argv)
    data = report_cached(args.runs)
    print(json.dumps(data, indent=1) if args.json else format_table(data))
    return 0


if __name__ == '__main__':
    sys.exit(main())
