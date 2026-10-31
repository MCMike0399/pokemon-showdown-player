"""Live interleaved A/B experiments on the ladder: a factorial design with one SPRT per factor.

Gates against the ladder-human imitation detect only large effects (about +9 pp per
400-pair stage as implemented), and that opponent Protects on 29% of its actions where
ladder humans do on 12% (docs/c9-live-ab.md). The live ladder is the real target. Here
every live game draws its own arm, so both arms meet the same opponents at the same
rating, with no deploy-segment confounds.

Design. The spec (`data/ml/value/ab/current.json`, read by the runner between games) names
factors. Each factor is a set of `search/config.py` overrides that is either on or off. Games
draw cells of the full factorial in balanced blocks: with k randomised factors, each block of
2**k games plays every cell once, in a seeded random order. Every game informs every
factor, so k factors need about as many games as one. A factor's main effect compares the
mean score of the games where it was on with those where it was off, over the games played
while it was randomised.

Decision per factor: a sequential probability ratio test of +elo0 against +elo1 Elo, using
the normal approximation of the score difference (GSPRT):

    LLR = (d1 - d0) / V * (d - (d0 + d1) / 2),  V = s^2 (1/n_on + 1/n_off)

d is the observed score difference, d0 and d1 are the score differences at elo0 and elo1
around 50%, and s^2 is the pooled score variance. The test accepts H1 at
LLR >= log((1 - beta) / alpha) and H0 at LLR <= log(beta / (1 - alpha)). It decides nothing
before `min_games`. At `max_games` it is inconclusive, which counts as not promoted. A
decided factor is frozen at its level (on for H1, off otherwise) for later games, and the
others keep running. Forfeit wins count as wins. Ties and unfinished games count 0.5.

The runner appends one line per game to `<spec dir>/<id>-arms.jsonl` (room, cell, the
randomised factors), so a resumed room keeps its arm. This module is pure apart from
those two files.

An optional `base` dict of overrides applies to every cell; a factor that is on replaces
the base keys it sets. Besides `search/config.py` keys, an override may name `team` (a
`teams.json` entry): the runner then plays that team for the game (campaign c13,
docs/c13-team-ladder.md), so a base team against a challenger team is one factor. It may
also name `preview`, a team-preview policy spec as for the runner's `--preview`
('pvalue:PATH' or 'human:PATH[:temp|argmax]'; campaign c14, docs/c14-preview.md).

    .venv/bin/python -m search.ab_test status [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC = ROOT / 'data/ml/value/ab/current.json'
DEFAULTS = {'elo0': 0.0, 'elo1': 40.0, 'alpha': 0.05, 'beta': 0.10, 'min_games': 300, 'max_games': 3000}
_NAME = re.compile(r'^[a-z][a-z0-9_]{0,23}$')


# ------------------------------------------------------------------ spec

def validate(spec: dict) -> dict:
    """The spec with defaults filled in; raises ValueError on a malformed one."""
    if not isinstance(spec, dict) or not spec.get('id') or not _NAME.match(str(spec['id'])):
        raise ValueError('spec needs an id like "c9"')
    factors = spec.get('factors')
    if not isinstance(factors, dict) or not factors:
        raise ValueError('spec needs at least one factor')
    seen = {}
    for name, options in factors.items():
        if not _NAME.match(name) or not isinstance(options, dict) or not options:
            raise ValueError(f'factor {name!r} must be a name with a non-empty options dict')
        for key in options:
            if key in seen:
                raise ValueError(f'factors {seen[key]!r} and {name!r} both set {key!r}')
            seen[key] = name
    if not isinstance(spec.get('base', {}), dict):
        raise ValueError('base must be a dict of overrides')
    for name, options in [('base', spec.get('base') or {})] + list(factors.items()):
        if 'team' in options and not (isinstance(options['team'], str) and options['team']):
            raise ValueError(f'{name}: team must be a teams.json name')
        if 'preview' in options and not (isinstance(options['preview'], str)
                                         and options['preview'].split(':')[0] in ('pvalue', 'human')):
            raise ValueError(f"{name}: preview must be 'pvalue:PATH' or 'human:PATH[:temp|argmax]'")
    frozen = spec.get('frozen') or {}
    if any(f not in factors or v not in (True, False) for f, v in frozen.items()):
        raise ValueError('frozen maps factor names to true/false')
    if len(factors) - len(frozen) > 6:
        raise ValueError('at most 6 randomised factors (blocks of 64 games)')
    out = {**DEFAULTS, **spec, 'frozen': dict(frozen)}
    if not 0 < out['alpha'] < 0.5 or not 0 < out['beta'] < 0.5 or out['elo1'] <= out['elo0']:
        raise ValueError('need 0 < alpha, beta < 0.5 and elo1 > elo0')
    return out


def load_spec(path: Path = SPEC) -> dict | None:
    """The active spec, or None (no file, inactive, or unreadable: the runner then plays LIVE)."""
    try:
        spec = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(spec, dict) or not spec.get('active'):
        return None
    return validate(spec)


def write_spec(spec: dict, path: Path = SPEC) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(spec, indent=1) + '\n')
    os.replace(tmp, path)


def arms_path(spec: dict, path: Path = SPEC) -> Path:
    return Path(path).parent / f"{spec['id']}-arms.jsonl"


def read_arms(spec: dict, path: Path = SPEC) -> list[dict]:
    rows = []
    try:
        lines = arms_path(spec, path).read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a half-written last line
        if isinstance(row, dict) and row.get('room'):
            rows.append(row)
    return rows


# ------------------------------------------------------------- assignment

def randomised(spec: dict) -> list[str]:
    return [f for f in spec['factors'] if f not in spec['frozen']]


def cell_at(spec: dict, n: int) -> dict:
    """The cell of the n-th game of the current phase: every block of 2**k games plays each cell once."""
    free = randomised(spec)
    size = 2 ** len(free)
    order = list(range(size))
    random.Random(f"{spec['id']}:{spec.get('seed', 0)}:{','.join(free)}:{n // size}").shuffle(order)
    bits = order[n % size]
    cell = {f: bool(bits >> i & 1) for i, f in enumerate(free)}
    cell.update(spec['frozen'])
    return {f: cell[f] for f in spec['factors']}


def overrides(spec: dict, cell: dict) -> dict:
    out = dict(spec.get('base') or {})
    for factor, on in cell.items():
        if on:
            out.update(spec['factors'][factor])
    return out


def next_arm(spec: dict, path: Path = SPEC) -> dict:
    """The arm for the next new game: games already recorded in this phase pick the block position."""
    free = randomised(spec)
    n = sum(1 for r in read_arms(spec, path) if r.get('random') == free)
    cell = cell_at(spec, n)
    return {'id': spec['id'], 'n': n, 'cell': cell, 'random': free, 'overrides': overrides(spec, cell)}


def arm_for_room(spec: dict, room: str, path: Path = SPEC) -> dict | None:
    """The arm a room was started with (a resumed game keeps it), or None."""
    for row in reversed(read_arms(spec, path)):
        if row.get('room') == room:
            cell = {f: bool(row['cell'].get(f)) for f in spec['factors']}
            return {**row, 'cell': cell, 'overrides': overrides(spec, cell)}
    return None


def record_arm(spec: dict, room: str, arm: dict, path: Path = SPEC) -> None:
    row = {'room': room, 'time': time.time(), 'id': spec['id'], 'n': arm['n'], 'cell': arm['cell'],
           'random': arm['random']}
    p = arms_path(spec, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(row) + '\n')


# --------------------------------------------------------------- analysis

def elo_score(elo: float) -> float:
    """Score difference against an equal opponent for an Elo advantage."""
    return 1.0 / (1.0 + 10 ** (-elo / 400.0)) - 0.5


def score_elo(d: float) -> float:
    d = max(-0.499, min(0.499, d))
    return -400.0 * math.log10(1.0 / (d + 0.5) - 1.0)


def sprt(on: list[float], off: list[float], elo0: float, elo1: float) -> dict:
    n1, n0 = len(on), len(off)
    if n1 < 2 or n0 < 2:
        return {'llr': 0.0, 'd': None, 'se': None}
    allv = on + off
    mean = sum(allv) / len(allv)
    s2 = sum((v - mean) ** 2 for v in allv) / (len(allv) - 1)
    m1, m0 = sum(on) / n1, sum(off) / n0
    d = m1 - m0
    var = max(s2, 1e-6) * (1 / n1 + 1 / n0)
    d0, d1 = elo_score(elo0), elo_score(elo1)
    return {'llr': (d1 - d0) / var * (d - (d0 + d1) / 2), 'd': d, 'se': math.sqrt(var)}


def bounds(spec: dict) -> tuple[float, float]:
    a, b = spec['alpha'], spec['beta']
    return math.log(b / (1 - a)), math.log((1 - b) / a)


def join(spec: dict, games: list[dict], arms: list[dict]) -> list[dict]:
    """Finished games of this experiment: {room, score, cell, random, opp_rating, time}."""
    by_room = {}
    for row in arms:  # a room recorded twice keeps its first arm
        by_room.setdefault(row['room'], row)
    out = []
    for g in games:
        row = by_room.get(g['room'])
        if row is None or g.get('score') is None:
            continue
        out.append({**g, 'cell': {f: bool(row['cell'].get(f)) for f in spec['factors']},
                    'random': list(row.get('random') or [])})
    return out


def analyse(spec: dict, rows: list[dict]) -> dict:
    """Per factor: games, scores on/off, Elo estimate, LLR and decision; per cell: games and score."""
    from search.elo_monitor import performance
    lo, hi = bounds(spec)
    factors = {}
    for f in spec['factors']:
        sample = [r for r in rows if f in r['random']]
        on = [r['score'] for r in sample if r['cell'][f]]
        off = [r['score'] for r in sample if not r['cell'][f]]
        t = sprt(on, off, spec['elo0'], spec['elo1'])
        n = len(sample)
        decision = None
        if n >= spec['min_games']:
            if t['llr'] >= hi:
                decision = 'H1'
            elif t['llr'] <= lo:
                decision = 'H0'
            elif n >= spec['max_games']:
                decision = 'cap'
        perf_on = performance([r for r in sample if r['cell'][f]])
        perf_off = performance([r for r in sample if not r['cell'][f]])
        factors[f] = {
            'games': n, 'on': len(on), 'off': len(off),
            'score_on': round(sum(on) / len(on), 4) if on else None,
            'score_off': round(sum(off) / len(off), 4) if off else None,
            'd': None if t['d'] is None else round(t['d'], 4),
            'se': None if t['se'] is None else round(t['se'], 4),
            'elo': None if t['d'] is None else round(score_elo(t['d']), 1),
            'perf_on': None if perf_on['rating'] is None else round(perf_on['rating']),
            'perf_off': None if perf_off['rating'] is None else round(perf_off['rating']),
            'llr': round(t['llr'], 3), 'bounds': [round(lo, 3), round(hi, 3)],
            'decision': decision, 'frozen': spec['frozen'].get(f)}
    cells = {}
    for r in rows:
        key = ''.join('+' if r['cell'][f] else '-' for f in spec['factors'])
        c = cells.setdefault(key, {'games': 0, 'score': 0.0})
        c['games'] += 1
        c['score'] += r['score']
    for c in cells.values():
        c['mean'] = round(c['score'] / c['games'], 3)
    return {'id': spec['id'], 'games': len(rows), 'factors': factors, 'cells': dict(sorted(cells.items())),
            'factor_order': list(spec['factors'])}


def status(spec_path: Path = SPEC, runs_dir: Path | None = None) -> dict | None:
    from search.elo_monitor import RUNS, load_games
    try:
        raw = json.loads(Path(spec_path).read_text())
    except (OSError, ValueError):
        return None
    spec = validate(raw)
    rows = join(spec, load_games(runs_dir or RUNS), read_arms(spec, spec_path))
    return {**analyse(spec, rows), 'active': bool(raw.get('active'))}


def format_status(st: dict) -> str:
    lines = [f"A/B {st['id']} ({'active' if st['active'] else 'inactive'}): {st['games']} finished games"]
    lines.append('  factor       games  on/off      score on/off   Elo (+-se)     perf on/off   LLR [lo, hi]          decision')
    for f, s in st['factors'].items():
        elo = '   n/a      ' if s['elo'] is None else '%+6.0f (+-%3.0f)' % (s['elo'], 695 * (s['se'] or 0))
        so = '  n/a       ' if s['score_on'] is None else '%.3f / %.3f' % (s['score_on'], s['score_off'] or 0)
        perf = '%s / %s' % (s['perf_on'], s['perf_off'])
        dec = (s['decision'] or '-') + ('' if s['frozen'] is None else f" (frozen {'on' if s['frozen'] else 'off'})")
        lines.append('  %-12s %5d  %4d/%-4d  %s  %s  %-12s %7.2f [%.2f, %.2f]  %s' % (
            f, s['games'], s['on'], s['off'], so, elo, perf, s['llr'], s['bounds'][0], s['bounds'][1], dec))
    lines.append('  cells (' + ' '.join(st['factor_order']) + '):')
    for key, c in st['cells'].items():
        lines.append(f"    {key}  games {c['games']:4d}  score {c['mean']:.3f}")
    return '\n'.join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('command', choices=['status'])
    ap.add_argument('--spec', type=Path, default=SPEC)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)
    st = status(args.spec)
    if st is None:
        print('no A/B spec at ' + str(args.spec))
        return 1
    print(json.dumps(st, indent=1) if args.json else format_status(st))
    return 0


if __name__ == '__main__':
    sys.exit(main())
