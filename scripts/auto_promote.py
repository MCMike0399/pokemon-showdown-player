"""Promote a gated search candidate to the live runner without a human in the loop.

Waits until a campaign's confirm stage and its supplementary standard gate (vs the
deployed search) have settled, applies the campaign rule, and only if it holds:
writes the candidate's options into `search/config.py` LIVE, runs the test suite and
the public-snapshot secret scan, commits that one file, pushes main, and restarts the
browser-player LaunchAgent gracefully (bootout lets the current game finish).

Rule (campaigns c4/c5): the confirm stage passed (value_gate `superior` summary) and
the standard gate shows no significant harm (one-sided sign test for the incumbent
p > alpha, or losses <= gains). Only final summaries count: a stage is settled when
its process has exited and its summary is complete. A standard stage that exited
unsettled is resumed with `--launch STAGE` (same seed, it skips finished pairs).

Every step is appended to `<gates>/promote.log`; the outcome goes to
`<gates>/promote.json`. Any failure before the push leaves config.py untouched.

    nohup .venv/bin/python scripts/auto_promote.py --gates data/ml/value/campaign-c5/gates \\
      --confirm c5-full-confirm --standard c5-full-standard --label "campaign c5 full" \\
      --launch "data/ml/value/campaign-c5/launch.sh full-standard 2" \\
      --options '{"leaf": {"a": 0.801, "b": -0.43}, "alpha": 0.25, ...}' &
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search.value_gate import sign_test, summarize  # noqa: E402

CONFIG = ROOT / 'search' / 'config.py'
PY = str(ROOT / '.venv' / 'bin' / 'python')
LABEL = 'dev.pokemon-showdown.browser-player'
PLIST = Path.home() / 'Library' / 'LaunchAgents' / f'{LABEL}.plist'


def log(gates: Path, msg: str):
    line = f'{datetime.now().isoformat(timespec="seconds")} {msg}'
    print(line, flush=True)
    with (gates / 'promote.log').open('a') as f:
        f.write(line + '\n')


def stage_state(gates: Path, stage: str, pairs: int, alpha: float) -> dict:
    path = gates / f'{stage}.jsonl'
    # Process first: once it has exited, the rows read next are final.
    running = subprocess.run(['pgrep', '-f', f'{stage}.jsonl'], capture_output=True).returncode == 0
    rows = []
    for line in (path.open() if path.exists() else []):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # blank or still being written
    sm = summarize(rows, pairs, alpha) if rows else {'complete': False}
    return {**sm, 'running': running, 'settled': bool(sm.get('complete')) and not running}


def live_dict_span(src: str) -> tuple[int, int]:
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'LIVE' for t in n.targets))
    lines = src.splitlines(keepends=True)
    start = sum(len(l) for l in lines[:node.value.lineno - 1]) + node.value.col_offset
    end = sum(len(l) for l in lines[:node.value.end_lineno - 1]) + node.value.end_col_offset
    return start, end


def apply_options(src: str, options: dict, note: str) -> str:
    """Rewrite single-line `'key': value,` entries of LIVE in place; append new keys before
    the closing brace under a comment. Raises if a key to replace spans several lines."""
    start, end = live_dict_span(src)
    body = src[start:end]
    lines = body.splitlines(keepends=True)
    added = []
    for key, value in options.items():
        hits = [i for i, l in enumerate(lines) if l.strip().startswith(f"'{key}':")]
        if hits:
            i = hits[0]
            line = lines[i]
            if not line.split('#', 1)[0].rstrip().endswith(','):
                raise ValueError(f'LIVE[{key!r}] spans several lines; edit it by hand')
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}'{key}': {value!r},  # {note}\n"
        else:
            added.append(f"    '{key}': {value!r},\n")
    if added:
        assert lines[-1].strip() == '}', 'LIVE must end with a closing brace on its own line'
        lines[-1:-1] = [f'    # {note}\n', *added]
    return src[:start] + ''.join(lines) + src[end:]


def check_config(options: dict):
    out = subprocess.run([PY, '-c', 'import json; from search.config import LIVE; print(json.dumps(LIVE))'],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    live = json.loads(out.stdout)
    bad = {k: (live.get(k), v) for k, v in options.items() if live.get(k) != v}
    if bad:
        raise ValueError(f'config.py did not take the options: {bad}')


def run(gates: Path, cmd: list[str], timeout: float = 3600) -> str:
    log(gates, '$ ' + ' '.join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    log(gates, '  -> ' + ' | '.join(tail))
    if p.returncode != 0:
        raise RuntimeError(f'{cmd[0]} failed ({p.returncode})')
    return p.stdout


def promote(gates: Path, options: dict, label: str, evidence: dict, dry: bool):
    run(gates, ['git', 'fetch', '-q', 'origin'])
    if subprocess.run(['git', 'symbolic-ref', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                      text=True).stdout.strip() != 'main':
        raise RuntimeError('workspace is not on main')
    run(gates, ['git', 'merge', '--ff-only', 'origin/main'])
    if subprocess.run(['git', 'diff', '--quiet', '--', str(CONFIG)], cwd=ROOT).returncode != 0:
        raise RuntimeError('search/config.py has uncommitted changes')
    original = CONFIG.read_text()
    note = f'{label}, promoted {datetime.now().date()} (docs/endgame-search.md)'
    CONFIG.write_text(apply_options(original, options, note))
    try:
        check_config(options)
        try:
            run(gates, [PY, '-m', 'pytest', '-q'])
        except RuntimeError:
            log(gates, 'tests failed; retrying once (load-sensitive tests)')
            run(gates, [PY, '-m', 'pytest', '-q'])
        with tempfile.TemporaryDirectory() as tmp:
            out = run(gates, [PY, 'scripts/public_snapshot.py', str(Path(tmp) / 'snap')])
            if '"secret_scan": "passed"' not in out:
                raise RuntimeError('secret scan did not pass')
        if dry:
            log(gates, 'dry run: config.py restored, nothing committed')
            CONFIG.write_text(original)
            return
        c, s = evidence['confirm'], evidence['standard']
        msg = (f'Promote {label} to the live search\n\n'
               f'Confirm gate vs the ladder-human imitation: {c["gains"]} gains, {c["losses"]} losses, '
               f'p {c["p_one_sided"]} ({c["pairs_complete"]} pairs).\n'
               f'Standard gate vs the deployed search: {s["gains"]} gains, {s["losses"]} losses '
               f'(no significant harm).\nPromoted automatically by scripts/auto_promote.py.')
        run(gates, ['git', 'commit', '-q', '-m', msg, '--', 'search/config.py'])
    except Exception:
        if subprocess.run(['git', 'diff', '--quiet', '--', str(CONFIG)], cwd=ROOT).returncode != 0:
            CONFIG.write_text(original)
        raise
    for attempt in range(1, 4):
        try:
            run(gates, ['git', 'push', '-q', 'origin', 'main'])
            break
        except RuntimeError:
            log(gates, f'push failed (attempt {attempt}); rebasing on origin/main')
            subprocess.run(['git', 'pull', '-q', '--rebase', 'origin', 'main'], cwd=ROOT)
            time.sleep(60)
    else:
        log(gates, 'push failed 3 times: the commit stays local (push it by hand); deploying anyway')
    restart(gates)


def restart(gates: Path):
    uid = os.getuid()
    subprocess.run(['launchctl', 'bootout', f'gui/{uid}/{LABEL}'], capture_output=True)
    log(gates, 'bootout sent; waiting for the current game to finish')
    deadline = time.time() + 3600
    while subprocess.run(['pgrep', '-f', 'scripts/browser_play.py'], capture_output=True).returncode == 0:
        if time.time() > deadline:
            log(gates, 'runner still running after an hour; bootstrapping anyway')
            break
        time.sleep(10)
    for attempt in range(3):  # never leave the ladder player down
        if subprocess.run(['launchctl', 'bootstrap', f'gui/{uid}', str(PLIST)], capture_output=True).returncode == 0:
            break
        time.sleep(30)
    time.sleep(60)
    if subprocess.run(['pgrep', '-f', 'scripts/browser_play.py'], capture_output=True).returncode != 0:
        raise RuntimeError('runner is not running after bootstrap')
    log(gates, 'runner restarted with the promoted configuration')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gates', type=Path, required=True)
    ap.add_argument('--confirm', required=True, help='confirm stage name (<gates>/<name>.jsonl)')
    ap.add_argument('--standard', required=True, help='standard stage name (<gates>/<name>.jsonl)')
    ap.add_argument('--options', required=True, help='JSON of search/config.py LIVE keys to set')
    ap.add_argument('--label', required=True)
    ap.add_argument('--launch', help='command that (re)starts the standard stage when it exited unsettled')
    ap.add_argument('--pairs', type=int, default=400)
    ap.add_argument('--standard-pairs', type=int, default=200)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--poll', type=float, default=300)
    ap.add_argument('--dry-run', action='store_true', help='test + scan, then restore config.py; no commit/push/restart')
    args = ap.parse_args()
    gates = (ROOT / args.gates).resolve()
    options = json.loads(args.options)
    log(gates, f'auto-promote {args.label}: waiting for {args.confirm} and {args.standard}')
    resumes = 0
    while True:
        try:
            c = stage_state(gates, args.confirm, args.pairs, args.alpha)
            s = stage_state(gates, args.standard, args.standard_pairs, args.alpha)
        except Exception as e:  # noqa: BLE001 - keep waiting; a gate file can be mid-rewrite
            log(gates, f'status check failed, retrying: {e!r}')
            time.sleep(args.poll)
            continue
        if c['settled'] and not c['passed']:
            break
        if not s['running'] and not s['settled'] and args.launch:
            if resumes >= 5:
                log(gates, 'standard stage keeps exiting unsettled; giving up')
                break
            resumes += 1
            log(gates, f'resume standard ({resumes})')
            subprocess.run(args.launch, shell=True, cwd=ROOT)
        if c['settled'] and s['settled']:
            break
        time.sleep(args.poll)
    harm_p = sign_test(s.get('losses', 0), s.get('gains', 0))
    harm = s.get('losses', 0) > s.get('gains', 0) and harm_p <= args.alpha
    ok = c['settled'] and c['passed'] and s['settled'] and not harm
    outcome = {'label': args.label, 'confirm': c, 'standard': {**s, 'harm_p': round(harm_p, 5), 'harm': harm},
               'promote': ok, 'options': options, 'time': datetime.now().isoformat(timespec='seconds')}
    log(gates, f'decision: promote={ok} confirm {c.get("gains")}-{c.get("losses")} p {c.get("p_one_sided")} '
               f'passed={c.get("passed")}; standard {s.get("gains")}-{s.get("losses")} harm_p {harm_p:.4f}')
    if ok:
        try:
            promote(gates, options, args.label, outcome, args.dry_run)
            outcome['deployed'] = not args.dry_run
        except Exception as e:  # noqa: BLE001
            outcome['deployed'] = False
            outcome['error'] = repr(e)
            log(gates, f'promotion FAILED: {e!r}')
    (gates / 'promote.json').write_text(json.dumps(outcome, indent=1))


if __name__ == '__main__':
    main()
