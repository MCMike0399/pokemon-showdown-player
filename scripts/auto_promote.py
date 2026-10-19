"""Promote a gated search candidate to the live runner without a human in the loop.

Waits until a campaign's confirm stage and its supplementary standard gate (vs the
deployed search) have settled, applies the campaign rule, and only if it holds:
writes the candidate's options into `search/config.py` LIVE, runs the test suite and
the public-snapshot secret scan, commits that one file, pushes main, pins the live
worktree (`scripts/live_deploy.py`) to that commit and restarts the browser-player
LaunchAgent gracefully (bootout lets the current game finish).

Rule (campaigns c4/c5): the confirm stage passed (value_gate `superior` summary) and
the standard gate shows no significant harm (one-sided sign test for the incumbent
p > alpha, or losses <= gains). Only final summaries count: a stage is settled when
its process has exited and its summary is complete. A standard stage that exited
unsettled is resumed with `--launch STAGE` (same seed, it skips finished pairs).

The checks run in a clean worktree of origin/main (tracked files only, like CI), and a
preflight repeats them every two hours while the gates run (`<gates>/preflight.json`), so a
blocker shows up long before promotion. A test failure blocks only if origin/main passes it.

Every step is appended to `<gates>/promote.log`; the outcome goes to
`<gates>/promote.json`. Any failure before the push leaves config.py untouched.

Other agents may work in the main checkout meanwhile: the runner runs from the pinned live
worktree, so their uncommitted edits never go live, and the main checkout is fast-forwarded
only when it is on `main` with no tracked changes (never another agent's branch).

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
sys.path.insert(0, str(ROOT / 'scripts'))

import live_deploy  # noqa: E402
from search.value_gate import sign_test, summarize  # noqa: E402

PY = str(ROOT / '.venv' / 'bin' / 'python')


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


def check_config(options: dict, cwd: Path = ROOT):
    out = subprocess.run([PY, '-c', 'import json; from search.config import LIVE; print(json.dumps(LIVE))'],
                         cwd=cwd, capture_output=True, text=True, check=True)
    live = json.loads(out.stdout)
    bad = {k: (live.get(k), v) for k, v in options.items() if live.get(k) != v}
    if bad:
        raise ValueError(f'config.py did not take the options: {bad}')


def run(gates: Path, cmd: list[str], timeout: float = 3600, cwd: Path = ROOT, check: bool = True):
    log(gates, '$ ' + ' '.join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    log(gates, '  -> ' + ' | '.join(tail))
    if check and p.returncode != 0:
        raise RuntimeError(f'{cmd[0]} failed ({p.returncode})')
    return p


def failed_tests(gates: Path, cwd: Path) -> set[str]:
    """Test ids that fail in a checkout (empty = suite green). One retry absorbs load-sensitive flakes."""
    fails = None
    for _ in range(2):
        p = run(gates, [PY, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'], cwd=cwd, check=False)
        now = {l.split()[1] for l in p.stdout.splitlines() if l.startswith(('FAILED ', 'ERROR '))}
        if p.returncode not in (0, 1):  # collection error, interrupted, no tests
            now.add(f'pytest exit {p.returncode}')
        fails = now if fails is None else fails & now
        if not fails:
            return set()
    return fails


def checkout(gates: Path, path: Path, ref: str = 'origin/main') -> Path:
    """A clean detached worktree of `ref` (only tracked files, as CI and the public repo see them)."""
    subprocess.run(['git', 'worktree', 'remove', '--force', str(path)], cwd=ROOT, capture_output=True)
    run(gates, ['git', 'worktree', 'add', '-q', '--detach', str(path), ref])
    if (ROOT / 'node_modules').exists():  # the simulator package, installed but untracked
        (path / 'node_modules').symlink_to(ROOT / 'node_modules')
    return path


def prepare(gates: Path, options: dict, label: str) -> Path:
    """Build the promotion commit's tree in a clean worktree and check it: config takes the options,
    the test suite introduces no new failure vs origin/main, the public snapshot scan passes.
    Only tracked files are tested and scanned, so local data or another agent's uncommitted
    files cannot block (or sneak into) the promotion. Returns the worktree."""
    run(gates, ['git', 'fetch', '-q', 'origin'])
    wt = checkout(gates, gates / '.promote-wt')
    cfg = wt / 'search' / 'config.py'
    cfg.write_text(apply_options(cfg.read_text(), options, f'{label}, promoted {datetime.now().date()}'))
    check_config(options, wt)
    fails = failed_tests(gates, wt)
    if fails:
        base = checkout(gates, gates / '.baseline-wt')
        try:
            new = fails - failed_tests(gates, base)
        finally:
            subprocess.run(['git', 'worktree', 'remove', '--force', str(base)], cwd=ROOT, capture_output=True)
        if new:
            raise RuntimeError(f'the promotion breaks tests: {sorted(new)}')
        log(gates, f'tests failing on origin/main too (not caused by the promotion): {sorted(fails)}')
    with tempfile.TemporaryDirectory() as tmp:
        p = run(gates, [PY, str(wt / 'scripts' / 'public_snapshot.py'), str(Path(tmp) / 'snap')], cwd=wt)
        if '"secret_scan": "passed"' not in p.stdout:
            raise RuntimeError('secret scan did not pass')
    return wt


def preflight(gates: Path, options: dict, label: str) -> dict:
    """Run prepare() while the gates are still running, so a blocker shows up hours before promotion."""
    state = {'time': datetime.now().isoformat(timespec='seconds')}
    try:
        live_deploy.check()
        prepare(gates, options, label)
        # Credentials: a dry-run push still authenticates against GitHub's receive-pack.
        run(gates, ['git', 'push', '--dry-run', '-q', 'origin', 'origin/main:refs/heads/main'])
        state['ok'] = True
        log(gates, 'preflight OK: clean checkout + options pass tests and the secret scan')
    except Exception as e:  # noqa: BLE001
        state.update(ok=False, error=repr(e))
        log(gates, f'PREFLIGHT FAILED (fix before the gates settle): {e!r}')
    subprocess.run(['git', 'worktree', 'remove', '--force', str(gates / '.promote-wt')], cwd=ROOT, capture_output=True)
    (gates / 'preflight.json').write_text(json.dumps(state, indent=1))
    return state


def promote(gates: Path, options: dict, label: str, evidence: dict, dry: bool, message: str | None = None):
    """`message` replaces the gate-evidence commit message (live A/B promotions, scripts/ab_watch.py)."""
    live_deploy.check()
    wt = prepare(gates, options, label)
    sha = None
    try:
        if dry:
            log(gates, 'dry run: checks passed, nothing committed')
            return
        if message:
            msg = message
        else:
            c, s = evidence['confirm'], evidence['standard']
            msg = (f'Promote {label} to the live search\n\n'
                   f'Confirm gate vs the ladder-human imitation: {c["gains"]} gains, {c["losses"]} losses, '
                   f'p {c["p_one_sided"]} ({c["pairs_complete"]} pairs).\n'
                   f'Standard gate vs the deployed search: {s["gains"]} gains, {s["losses"]} losses '
                   f'(no significant harm).\nPromoted automatically by scripts/auto_promote.py.')
        run(gates, ['git', 'commit', '-q', '-m', msg, '--', 'search/config.py'], cwd=wt)
        for attempt in range(1, 4):
            if run(gates, ['git', 'push', '-q', 'origin', 'HEAD:main'], cwd=wt, check=False).returncode == 0:
                break
            log(gates, f'push failed (attempt {attempt}); rebasing on origin/main')
            subprocess.run(['git', 'fetch', '-q', 'origin'], cwd=wt)
            subprocess.run(['git', 'rebase', '-q', 'origin/main'], cwd=wt)
            time.sleep(60)
        else:
            log(gates, 'push failed 3 times: commit kept on the worktree HEAD; deploying anyway')
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=wt, capture_output=True, text=True).stdout.strip()
    finally:
        if not dry:
            log(gates, f'promotion commit: {subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()}')
        subprocess.run(['git', 'worktree', 'remove', '--force', str(wt)], cwd=ROOT, capture_output=True)
    # Deploy the exact promotion commit (pushed or not: the object is in the shared repo).
    run(gates, ['git', 'fetch', '-q', 'origin'], check=False)
    log(gates, f'live worktree pinned to {live_deploy.pin(sha)}')
    check_config(options, live_deploy.LIVE)
    sync_workspace(gates)
    live_deploy.restart(log=lambda m: log(gates, m))


def sync_workspace(gates: Path):
    """Fast-forward the main checkout only if it is on main with no tracked changes; other
    agents' branches and work in progress are left alone (the runner does not read it)."""
    branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if branch != 'main' or dirty:
        log(gates, f'main checkout left alone (branch {branch}, tracked changes: {bool(dirty)})')
        return
    run(gates, ['git', 'merge', '-q', '--ff-only', 'origin/main'], check=False)


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
    ap.add_argument('--dry-run', action='store_true', help='run the promotion checks only; no commit/push/restart')
    ap.add_argument('--preflight-hours', type=float, default=2.0,
                    help='re-run the promotion checks this often while waiting (0 = never)')
    args = ap.parse_args()
    gates = (ROOT / args.gates).resolve()
    options = json.loads(args.options)
    log(gates, f'auto-promote {args.label}: waiting for {args.confirm} and {args.standard}')
    resumes, last_preflight = 0, 0.0
    while True:
        if args.preflight_hours and time.time() - last_preflight > args.preflight_hours * 3600:
            last_preflight = time.time()
            preflight(gates, options, args.label)
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
