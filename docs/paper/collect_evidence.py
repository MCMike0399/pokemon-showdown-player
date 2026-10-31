"""Read-only campaign audit; publish aggregates, never battle/player records.

Verify the committed snapshot without private data:
    python3 docs/paper/collect_evidence.py --verify
Collect a NEW dated snapshot from this checkout's private campaign ledgers:
    python3 docs/paper/collect_evidence.py --collect --output /tmp/evidence.json
The paper's frozen snapshot must not be refreshed silently during a build.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).with_name('evidence-20261031.json')


def sign_tail(gains, losses):
    n = gains + losses
    return sum(math.comb(n, k) for k in range(gains, n + 1)) / 2**n if n else 1.0


def verify(data):
    """Independently recompute reported numerical claims from public aggregates."""
    lower, upper = math.log(.10 / .95), math.log(.90 / .05)
    for result in data['offline'].values():
        assert abs(sign_tail(result['gains'], result['losses']) - result['p_one_sided']) < .00001
        assert not result['passed']
    for row in data['c13_rounds']:
        s = row['stats']
        assert s['on'] + s['off'] == s['games']
        a = row['arms']['on']; b = row['arms']['off']
        assert a['n'] == s['on'] and b['n'] == s['off']
        d = a['score'] / a['n'] - b['score'] / b['n']
        assert abs(d - s['d']) < .0001
        elo = 400 * math.log10((.5 + d) / (.5 - d))
        assert abs(elo - s['elo']) < .051
        n = a['n'] + b['n']; total = a['score'] + b['score']
        squares = a['wins'] + b['wins'] + .25 * (a['ties'] + b['ties'])
        variance = (squares - total**2 / n) / (n - 1) * (1 / a['n'] + 1 / b['n'])
        d1 = 1 / (1 + 10**(-40 / 400)) - .5
        llr = d1 / variance * (d - d1 / 2)
        assert abs(llr - s['llr']) < .00051
        assert (llr >= upper if row['decision'] == 'H1' else llr <= lower)
    for row in data['c9_decisions']:
        assert row['decision'] == 'H0' and row['stats']['llr'] <= lower
    cohort = data['post_c13']
    assert cohort['rating'] - cohort['ci'] > 1200
    print('Verified: offline exact tails, C9 stopping decisions, both C13 arm scores,')
    print('Elo transforms, normal-approximation LLRs, and post-C13 performance interval.')


def collect():
    sys.path.insert(0, str(ROOT))
    from search import ab_test, elo_monitor

    cutoff = datetime.now(timezone.utc)
    value = ROOT / 'data/ml/value'
    manifests = {}

    def read(relative):
        content = (value / relative).read_bytes()
        manifests[relative] = sha256(content).hexdigest()
        return json.loads(content)

    def arms_summary(rows):
        p = elo_monitor.performance(rows)
        return {**p, 'wins': sum(r['score'] == 1 for r in rows),
                'ties': sum(r['score'] == .5 for r in rows)}

    runs = elo_monitor.load_runs()
    games = [g for g in elo_monitor.games_from_runs(runs) if g['time'] <= cutoff.timestamp()]
    c13 = read('campaign-c13/state.json')
    base = ab_test.validate(read('campaign-c13/spec-final-c13.json'))
    rounds = []
    for row in c13['history']:
        spec = {**base, 'id': 'c13r' + str(row['round'])}
        joined = ab_test.join(spec, games, ab_test.read_arms(spec))
        # The controller's first stopping decision excludes any later result.
        joined = joined[:row['stats']['games']]
        rounds.append({**row, 'arms': {
            label: arms_summary([g for g in joined if bool(g['cell']['team']) == enabled])
            for label, enabled in [('on', True), ('off', False)]}})
    post = ab_test.join(base, games, ab_test.read_arms(base))
    segments = []
    for seg in elo_monitor.segments(games, runs):
        segments.append({'index': seg['index'], 'policy_key': seg['key'],
                         'first': elo_monitor._iso(seg['first']),
                         'last': elo_monitor._iso(seg['last']),
                         **arms_summary(seg['games'])})
    offline_fields = ('pairs_complete', 'pairs_void', 'pairs_planned', 'candidate_score',
                      'incumbent_score', 'gains', 'losses', 'p_one_sided', 'rule',
                      'sims_ratio', 'errors', 'complete', 'futility_stop', 'passed')
    offline = {}
    for key, path in [('c10_preview', 'campaign-c10/preview-result.json'),
                      ('c10_endgame', 'campaign-c10/endgame-result.json'),
                      ('c12_belief', 'campaign-c12/belief-result.json')]:
        result = read(path)
        offline[key] = {k: result[k] for k in offline_fields}
    c9 = read('campaign-c9/spec-final-c9.json')
    current = ab_test.validate(read('ab/current.json'))
    current_rows = ab_test.join(current, games, ab_test.read_arms(current))
    return {'cutoff_utc': cutoff.isoformat(),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'scope': 'Completed terminal records only; deduplicated by room; aggregates only. '
                     'C13 rounds truncated at the controller stopping count. '
                     'C13final cohort joined by arm ledger, across runner restarts. '
                     'Unfinished assigned games are excluded, not scored as ties.',
            'games': len(games), 'overall': arms_summary(games), 'segments': segments,
            'offline': offline, 'c9_decisions': c9['history'], 'c13_rounds': rounds,
            'post_c13': arms_summary(post),
            'c14': {'id': current['id'], 'active': current['active'],
                    'completed_games': len(current_rows),
                    'decision': ab_test.analyse(current, current_rows)['factors']['preview']['decision']},
            'manifest_sha256': manifests}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--collect', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.collect:
        if args.verify or args.output is None:
            p.error('--collect requires --output and cannot be combined with --verify')
        data = collect()
        verify(data)
        args.output.write_text(json.dumps(data, indent=2) + '\n')
        print(args.output)
    else:
        verify(json.loads(SNAPSHOT.read_text()))


if __name__ == '__main__':
    main()
