"""Is each game paying off? One read-only report over live play and both learners.

    .venv/bin/python search/learning_report.py [--hours 24] [--json]

Live: games, record, rating, decision latency, how often close turns got extra
worlds, incidents, and how many labelled positions each finished game added.
Value learning: self-play volume, every candidate's predictive edge over the
hand evaluation on held-out live games (does more data help?), gate outcomes,
what live search uses now. PPO: recent paired evaluations and the job backlog.
Nothing here starts, stops or changes anything.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data' / 'ml'
VALUE = DATA / 'value'
INCIDENTS = ('interrupted', 'stalled', 'submit_retry', 'search_request_changed', 'submit_abandoned',
             'controls_never_shown', 'timer_on')


def live_section(since: float) -> dict:
    games, kinds, ms, extra = {}, {}, [], []
    for path in glob.glob(str(DATA / 'browser-runs' / '*' / 'events.jsonl')):
        if os.path.getmtime(path) < since:
            continue
        for line in open(path):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('time', 0) < since:
                continue
            kind = event.get('kind')
            if kind in INCIDENTS:
                kinds[kind] = kinds.get(kind, 0) + 1
            if kind == 'decision':
                trace = (event.get('decision') or {}).get('search') or {}
                if trace.get('kind') == 'move':
                    ms.append(trace.get('ms') or event['decision'].get('inference_ms') or 0)
                    extra.append(bool(trace.get('extra_worlds')))
                room = event.get('room')
                if room:
                    games.setdefault(room, {'decisions': 0})['decisions'] += 1
            if kind == 'terminal':
                result = event.get('result') or {}
                log = event.get('log')
                log = log if isinstance(log, list) else ast.literal_eval(log) if isinstance(log, str) else []
                me = event.get('side')
                rating = None
                for entry in log:
                    parts = entry.split('|')
                    if entry.startswith('|player|') and len(parts) > 5 and parts[2] == me and parts[5].isdigit():
                        rating = int(parts[5])
                game = games.setdefault(result.get('room'), {'decisions': 0})
                game.update(time=event['time'], win=result.get('winner') == _me(log, me), rating=rating)
    finished = sorted((g for g in games.values() if 'win' in g), key=lambda g: g['time'])
    positions = []
    for room, game in games.items():
        if 'win' not in game:
            continue
        path = VALUE / 'live' / ('live-' + room + '.jsonl')
        positions.append(sum(1 for _ in path.open()) if path.exists() else 0)
    ratings = [g['rating'] for g in finished if g.get('rating')]
    wins = sum(g['win'] for g in finished)
    return {'games': len(finished), 'wins': wins, 'losses': len(finished) - wins,
            'win_rate': round(wins / len(finished), 3) if finished else None,
            'rating_first_last': [ratings[0], ratings[-1]] if ratings else None,
            'unfinished_rooms': sum(1 for g in games.values() if 'win' not in g),
            'decision_ms_median': round(statistics.median(ms)) if ms else None,
            'decision_ms_p90': round(sorted(ms)[int(.9 * (len(ms) - 1))]) if ms else None,
            'close_turns_with_extra_worlds': round(sum(extra) / len(extra), 2) if extra else None,
            'incidents': kinds,
            'labelled_positions_per_game': round(statistics.mean(positions), 1) if positions else None,
            'games_without_positions': sum(1 for p in positions if not p)}


def _me(log: list[str], side: str | None) -> str | None:
    for entry in log:
        parts = entry.split('|')
        if entry.startswith('|player|') and len(parts) > 3 and parts[2] == side:
            return parts[3]
    return None


def value_section(since: float) -> dict:
    status = {}
    try:
        status = json.loads((VALUE / 'service-status.json').read_text())
    except (OSError, ValueError):
        pass
    candidates = []
    for path in sorted(glob.glob(str(VALUE / 'logs' / 'train-*.json')), key=os.path.getmtime):
        try:
            report = json.loads(Path(path).read_text().strip().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            continue
        test = report.get('live_test') or {}
        blend, hand = test.get('deployed_blend'), test.get('hand')
        if not blend or not hand:
            continue
        candidates.append({'sha': report['sha256'][:10], 'selfplay_rows': report.get('selfplay_rows'),
                           'live_test_games': (report.get('games') or {}).get('live_test'),
                           'logloss_edge': round(hand['logloss'] - blend['logloss'], 4),
                           'auc_edge': round(blend['auc'] - hand['auc'], 4), 'useful': report.get('useful')})
    from search.value_gate import summarize
    gates = []
    for path in sorted(glob.glob(str(VALUE / 'gates' / '*.jsonl')), key=os.path.getmtime):
        rows = [json.loads(x) for x in open(path) if x.strip()]
        s = summarize(rows, 80, 0.05)
        gates.append({'gate': Path(path).stem, **{k: s[k] for k in ('pairs_complete', 'gains', 'losses', 'p_one_sided',
                                                                     'complete', 'passed', 'futility_stop')}})
    edges = [c['logloss_edge'] for c in candidates]
    return {'selfplay_rows': status.get('selfplay_rows'), 'live_games_ingested': status.get('live_games'),
            'candidates_trained': len(candidates), 'candidates_beating_hand': sum(1 for c in candidates if c['useful']),
            'mean_logloss_edge_last5': round(statistics.mean(edges[-5:]), 4) if edges else None,
            'recent_candidates': candidates[-5:], 'gates': gates[-4:],
            'live_search_uses': status.get('current') or {'path': None}}


def ppo_section(since: float) -> dict:
    db = sqlite3.connect(f'file:{DATA / "experience.sqlite3"}?mode=ro', uri=True)
    try:
        cutoff = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(since))
        evals, promoted = [], 0
        for (result,) in db.execute("SELECT result FROM jobs WHERE kind='evaluate' AND status='complete' AND updated>? "
                                    "ORDER BY updated", (cutoff,)):
            r = json.loads(result or '{}')
            e = r.get('evaluation') or {}
            if 'paired_gained' in e:
                evals.append([e.get('paired_gained'), e.get('paired_lost'), e.get('passed')])
            promoted += bool(r.get('staged_for_between_game_promotion') or r.get('promoted'))
        queued = db.execute("SELECT COUNT(*) FROM jobs WHERE kind='learn' AND status='queued'").fetchone()[0]
        return {'evaluations': len(evals), 'passed': sum(1 for e in evals if e[2]), 'staged_or_promoted': promoted,
                'gained_lost_passed': evals[-6:], 'learn_jobs_queued': queued}
    finally:
        db.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hours', type=float, default=24)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    since = time.time() - args.hours * 3600
    report = {'window_hours': args.hours, 'live': live_section(since), 'value_learning': value_section(since),
              'ppo': ppo_section(since)}
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for section, body in report.items():
        if not isinstance(body, dict):
            print(f'{section}: {body}')
            continue
        print(f'\n== {section}')
        for key, value in body.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                print(f'  {key}:')
                for item in value:
                    print('    ' + json.dumps(item))
            else:
                print(f'  {key}: {json.dumps(value)}')


if __name__ == '__main__':
    main()
