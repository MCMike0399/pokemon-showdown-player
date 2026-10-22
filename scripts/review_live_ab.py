"""Read-only campaign review with chronological slices and reproducible input snapshots.

    .venv/bin/python scripts/review_live_ab.py --snapshot /tmp/c9-review.json
    .venv/bin/python scripts/review_live_ab.py --from-snapshot /tmp/c9-review.json

Intervals are descriptive fixed-sample approximations, not sequential decision
boundaries. Only ab_test / ab_watch decide the experiment. This command never
writes the experiment spec, changes the player, or promotes a candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search import ab_test as ab, elo_monitor as em  # noqa: E402


def brief(rows):
    p = em.performance(rows)
    return {k: round(p[k], 1) if isinstance(p[k], float) else p[k]
            for k in ('n', 'rating', 'ci', 'win_rate', 'mean_opp')}


def review(data, timezone_name='America/Mexico_City'):
    spec = ab.validate(data['spec'])
    games = sorted(data['games'], key=lambda g: (g['time'], g['room']))
    arms = data['arms']
    rows = ab.join(spec, games, arms)
    status = ab.analyse(spec, rows)
    factors = {}
    for name, stats in status['factors'].items():
        sample = [r for r in rows if name in r['random']]
        t = ab.sprt([r['score'] for r in sample if r['cell'][name]],
                    [r['score'] for r in sample if not r['cell'][name]], spec['elo0'], spec['elo1'])
        interval = ([round(ab.score_elo(t['d'] + sign * 1.96 * t['se']), 1) for sign in (-1, 1)]
                    if t['d'] is not None else None)
        factors[name] = {**stats, 'descriptive_elo_ci95': interval}

    phases = []
    for phase in dict.fromkeys(tuple(r['random']) for r in rows):
        sample = [r for r in rows if tuple(r['random']) == phase]
        phases.append({'randomised': list(phase), 'overall': brief(sample), 'factors': {
            f: {'on': brief([r for r in sample if r['cell'][f]]),
                'off': brief([r for r in sample if not r['cell'][f]])} for f in phase}})
    tz = ZoneInfo(timezone_name)
    daily = {}
    for r in rows:
        day = datetime.fromtimestamp(r['time'], tz).date().isoformat()
        daily.setdefault(day, []).append(r)

    # Compare only the immediately preceding contiguous deployment segment.
    first_assignment = min((a['time'] for a in arms), default=0)
    prior = [g for g in games if g['time'] < first_assignment]
    baseline = []
    for g in reversed(prior):
        if baseline and g['policy_key'] != baseline[0]['policy_key']:
            break
        baseline.append(g)
    completed = {r['room'] for r in rows}
    counts = {}
    mismatches = []
    for a in arms:
        phase = tuple(a['random'])
        n = counts.get(phase, 0)
        phase_spec = {**spec, 'frozen': {f: a['cell'][f] for f in spec['factors'] if f not in phase}}
        if a['n'] != n or a['cell'] != ab.cell_at(phase_spec, n):
            mismatches.append(a['room'])
        counts[phase] = n + 1
    return {
        'captured_at': data.get('captured_at'), 'id': spec['id'], 'active': spec.get('active', False),
        'timezone': timezone_name, 'campaign': brief(rows), 'preceding_segment': brief(baseline),
        'last_200': brief(rows[-200:]), 'factors': factors, 'phases': phases,
        'daily': {day: brief(sample) for day, sample in daily.items()},
        'integrity': {'assigned': len(arms), 'completed': len(rows),
                      'duplicates': len(arms) - len({a['room'] for a in arms}),
                      'unresolved_rooms': [a['room'] for a in arms if a['room'] not in completed],
                      'assignment_mismatches': mismatches,
                      'missing_opponent_ratings': sum(r.get('opp_rating') is None for r in rows)},
        'max_games': spec['max_games'],
        'interval_note': 'Descriptive fixed-sample 95% intervals; freezing and repeated looks affect coverage. '
                         'The preregistered SPRT, not these intervals or phase slices, decides promotion.',
    }


def format_report(result):
    def perf(p):
        if p['rating'] is None:
            return 'n/a'
        if p['ci'] is None:
            return f"{p['rating']:.0f} (clamped), n={p['n']}"
        return f"{p['rating']:.0f} +/- {p['ci']:.0f}, n={p['n']}"

    lines = [f"Campaign {result['id']} — {'active' if result['active'] else 'inactive'}",
             'Performance Elo (approximate 95% intervals):',
             '  preceding deployment: ' + perf(result['preceding_segment']),
             '  whole campaign:       ' + perf(result['campaign']),
             '  last 200:             ' + perf(result['last_200']), '', 'Randomised factor effects:']
    for name, f in result['factors'].items():
        ci = f['descriptive_elo_ci95']
        estimate = 'n/a' if ci is None else f"{f['elo']:+.0f} Elo [{ci[0]:+.0f}, {ci[1]:+.0f}]"
        state = 'testing' if f['frozen'] is None else 'frozen ' + ('on' if f['frozen'] else 'off')
        lines.append(f"  {name}: {estimate}; n={f['games']}; LLR {f['llr']:+.3f}; {state}")
    lines += ['', 'Chronological phases (descriptive; not additional promotion tests):']
    for phase in result['phases']:
        lines.append('  ' + (', '.join(phase['randomised']) or 'all frozen') + ': ' + perf(phase['overall']))
        for name, levels in phase['factors'].items():
            lines.append(f"    {name} on {perf(levels['on'])}; off {perf(levels['off'])}")
    lines += ['', 'Daily performance (' + result['timezone'] + '):']
    lines += [f'  {day}: {perf(p)}' for day, p in result['daily'].items()]
    lines += ['', 'Integrity: ' + json.dumps(result['integrity']), '', result['interval_note']]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', type=Path, default=ab.SPEC)
    parser.add_argument('--runs', type=Path, default=em.RUNS)
    parser.add_argument('--snapshot', type=Path, help='save inputs for an exactly reproducible review')
    parser.add_argument('--from-snapshot', type=Path, help='review saved inputs instead of live files')
    parser.add_argument('--timezone', default='America/Mexico_City')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.from_snapshot:
        data = json.loads(args.from_snapshot.read_text())
    else:
        spec = ab.validate(json.loads(args.spec.read_text()))
        # Capture assignments first: a game started during this read stays outside the snapshot.
        arms = ab.read_arms(spec, args.spec)
        data = {'captured_at': datetime.now(timezone.utc).isoformat(), 'spec': spec,
                'arms': arms, 'games': em.load_games(args.runs)}
    if args.snapshot:
        args.snapshot.write_text(json.dumps(data, indent=1) + '\n')
    result = review(data, args.timezone)
    print(json.dumps(result, indent=1) if args.json else format_report(result))


if __name__ == '__main__':
    main()
