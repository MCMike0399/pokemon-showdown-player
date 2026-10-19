"""Watch a live A/B experiment (search/ab_test.py): freeze decided factors, then promote the winners.

Every `--poll` seconds it reads the experiment's status. A factor whose SPRT has decided gets
frozen in the spec: on for H1, off for H0 or for an inconclusive test at `max_games`. The
runner reads the spec before each game, so later games stop randomising that factor and the
others go on. Once every factor is frozen, the factors frozen on are promoted together
through `scripts/auto_promote.py`'s `promote()`. That writes them into `search/config.py`
LIVE, runs the tests and the secret scan in a clean checkout, commits, pushes main, pins the
live worktree and restarts the runner gracefully. Then the spec is set inactive. With no
winner, the spec is just set inactive.

The promotion checks are re-run every `--preflight-hours` on the union of all factors'
options, so a blocker shows up long before the end. Every step goes to `<dir>/promote.log`,
and the outcome to `<dir>/promote.json`.

    nohup .venv/bin/python scripts/ab_watch.py --dir data/ml/value/campaign-c9 --label "campaign c9" &
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import auto_promote  # noqa: E402
from search import ab_test  # noqa: E402


def step(spec_path: Path, out: Path, label: str, runs_dir: Path | None = None, dry: bool = False,
         promote=auto_promote.promote) -> str:
    """One poll: 'waiting', 'running', 'promoted', 'finished' (no winner) or 'failed'."""
    raw = json.loads(spec_path.read_text())
    if not raw.get('active'):
        return 'waiting'
    spec = ab_test.validate(raw)
    st = ab_test.status(spec_path, runs_dir)
    (out / 'status.json').write_text(json.dumps({**st, 'time': datetime.now().isoformat(timespec='seconds')}, indent=1))
    changed = False
    for f, s in st['factors'].items():
        if f in spec['frozen'] or not s['decision']:
            continue
        on = s['decision'] == 'H1'
        raw.setdefault('frozen', {})[f] = on
        raw.setdefault('history', []).append({'time': datetime.now().isoformat(timespec='seconds'), 'factor': f,
                                              'decision': s['decision'], 'frozen': on, 'stats': s})
        auto_promote.log(out, f"{f}: {s['decision']} after {s['games']} games (Elo {s['elo']}, LLR {s['llr']}) "
                              f"-> frozen {'on' if on else 'off'}")
        changed = True
    if changed:
        ab_test.write_spec(raw, spec_path)
        spec = ab_test.validate(raw)
    if ab_test.randomised(spec):
        return 'running'
    winners = [f for f, on in spec['frozen'].items() if on]
    options = ab_test.overrides(spec, {f: f in winners for f in spec['factors']})
    outcome = {'label': label, 'winners': winners, 'options': options, 'history': raw.get('history', []),
               'time': datetime.now().isoformat(timespec='seconds')}
    result = 'finished'
    if winners:
        lines = [f"{h['factor']}: {h['decision']} after {h['stats']['games']} games, Elo {h['stats']['elo']:+.0f} "
                 f"(+-{695 * (h['stats']['se'] or 0):.0f}), performance on/off "
                 f"{h['stats']['perf_on']}/{h['stats']['perf_off']}" for h in raw.get('history', [])]
        message = (f'Promote {label} ({", ".join(winners)}) to the live search\n\n'
                   f'Live interleaved factorial A/B on the ladder (search/ab_test.py), one SPRT per factor '
                   f'({spec["elo0"]:+.0f} vs {spec["elo1"]:+.0f} Elo, alpha {spec["alpha"]}, beta {spec["beta"]}):\n'
                   + '\n'.join(lines) + '\nPromoted automatically by scripts/ab_watch.py.')
        try:
            promote(out, options, label, outcome, dry, message=message)
            outcome['deployed'] = not dry
            result = 'promoted'
        except Exception as e:  # noqa: BLE001
            outcome.update(deployed=False, error=repr(e))
            auto_promote.log(out, f'promotion FAILED: {e!r}')
            result = 'failed'
    if result != 'failed':
        raw['active'] = False
        ab_test.write_spec(raw, spec_path)
    (out / 'promote.json').write_text(json.dumps(outcome, indent=1))
    auto_promote.log(out, f'experiment {spec["id"]} {result}: winners {winners}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', type=Path, default=ab_test.SPEC)
    ap.add_argument('--dir', type=Path, required=True, help='campaign directory for logs and outcome')
    ap.add_argument('--label', required=True)
    ap.add_argument('--poll', type=float, default=1800)
    ap.add_argument('--preflight-hours', type=float, default=12.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    out = (ROOT / args.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    auto_promote.log(out, f'ab-watch {args.label}: watching {args.spec}')
    last_preflight = 0.0
    while True:
        if args.preflight_hours and time.time() - last_preflight > args.preflight_hours * 3600:
            last_preflight = time.time()
            try:
                spec = ab_test.validate(json.loads(args.spec.read_text()))
                union = ab_test.overrides(spec, {f: True for f in spec['factors']})
                auto_promote.preflight(out, union, args.label)
            except Exception as e:  # noqa: BLE001
                auto_promote.log(out, f'preflight could not run: {e!r}')
        try:
            result = step(args.spec, out, args.label, dry=args.dry_run)
        except Exception as e:  # noqa: BLE001 - a ledger or the spec can be mid-write
            auto_promote.log(out, f'status check failed, retrying: {e!r}')
            result = 'running'
        if result in ('promoted', 'finished', 'failed'):
            break
        time.sleep(args.poll)


if __name__ == '__main__':
    main()
