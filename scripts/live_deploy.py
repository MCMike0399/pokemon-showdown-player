"""Run the live ladder player from a pinned worktree, never from the shared checkout.

Other agents work in the main checkout (UI, docs, experiments). If the runner ran
from there, a restart would put their uncommitted or half-finished edits live. So
the runner runs from its own detached worktree, `<main>.live`, and only a deploy
moves it, to an exact commit. Gitignored runtime inputs (credentials, data, usage
caches, the simulator package, private teams) are symlinks to the main checkout,
so the pinned code shares the same data, ladder lock and logs.

    .venv/bin/python scripts/live_deploy.py --status
    .venv/bin/python scripts/live_deploy.py --install            # one-time: create + point the LaunchAgent at it
    .venv/bin/python scripts/live_deploy.py --ref origin/main    # pin a commit and restart gracefully

A deploy refuses to move a worktree with tracked changes (nobody should edit it by
hand). The restart is graceful: bootout lets the current game finish (ExitTimeOut).
"""
from __future__ import annotations

import argparse
import json
import plistlib
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
# The main checkout, also when this file runs from a worktree.
MAIN = Path(subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=HERE,
                           capture_output=True, text=True, check=True).stdout.strip()).parent
LIVE = MAIN.parent / f'{MAIN.name}.live'
SHARED = ('.env', 'data', 'cache', 'node_modules', 'teams.json', 'artifacts')
LABEL = 'dev.pokemon-showdown.browser-player'
PLIST = Path.home() / 'Library' / 'LaunchAgents' / f'{LABEL}.plist'
SCRIPT = 'scripts/browser_play.py'


def _git(*args, cwd=None, check=True) -> str:
    return subprocess.run(['git', *args], cwd=cwd or MAIN, capture_output=True, text=True, check=check).stdout.strip()


def link_shared(path: Path | None = None) -> None:
    path = path or LIVE
    for name in SHARED:
        src, dst = MAIN / name, path / name
        if src.exists() and not dst.is_symlink() and not dst.exists():
            dst.symlink_to(src)


def ensure(ref: str = 'HEAD', path: Path | None = None) -> Path:
    """The live worktree, created detached at `ref` if missing, with its shared links."""
    path = path or LIVE
    if not (path / '.git').exists():
        _git('worktree', 'add', '-q', '--detach', str(path), _git('rev-parse', ref))
    link_shared(path)
    return path


def dirty(path: Path | None = None) -> str:
    return _git('status', '--porcelain', '--untracked-files=no', cwd=path or LIVE)


def pin(ref: str, path: Path | None = None) -> str:
    """Move the live worktree to `ref` (a commit or ref resolved in the shared repo). Returns the sha."""
    path = path or LIVE
    ensure(ref, path)
    if changes := dirty(path):
        raise RuntimeError(f'live worktree {path} has tracked changes; inspect before deploying:\n{changes}')
    sha = _git('rev-parse', ref)
    _git('checkout', '-q', '--detach', sha, cwd=path)
    link_shared(path)
    return sha


def plist_points_live(plist: Path | None = None, path: Path | None = None) -> bool:
    plist, path = plist or PLIST, path or LIVE
    data = plistlib.loads(plist.read_bytes())
    return data.get('WorkingDirectory') == str(path) and str(path / SCRIPT) in data.get('ProgramArguments', [])


def install(plist: Path | None = None, path: Path | None = None) -> bool:
    """Point the LaunchAgent at the live worktree (backup kept next to it). True if it changed."""
    plist, path = plist or PLIST, path or LIVE
    data = plistlib.loads(plist.read_bytes())
    args = data['ProgramArguments']
    script = next(i for i, a in enumerate(args) if a.endswith(SCRIPT))
    want = str(path / SCRIPT)
    if args[script] == want and data.get('WorkingDirectory') == str(path):
        return False
    shutil.copy2(plist, plist.with_name(f'{plist.name}.bak-{datetime.now():%Y%m%d-%H%M%S}'))
    args[script] = want
    data['WorkingDirectory'] = str(path)
    plist.write_bytes(plistlib.dumps(data))
    return True


def status(path: Path | None = None) -> dict:
    path = path or LIVE
    out = {'live': str(path), 'exists': (path / '.git').exists(),
           'plist_points_live': PLIST.exists() and plist_points_live(PLIST, path)}
    if out['exists']:
        out.update(head=_git('rev-parse', 'HEAD', cwd=path), dirty=bool(dirty(path)),
                   missing_links=[n for n in SHARED if (MAIN / n).exists() and not (path / n).exists()],
                   behind_origin_main=int(_git('rev-list', '--count', 'HEAD..origin/main', cwd=path, check=False) or 0))
    return out


def check(path: Path | None = None) -> None:
    """Raise unless a deploy can run unattended: worktree present and clean, links in place, LaunchAgent on it."""
    s = status(path or LIVE)
    problems = [p for p, bad in (('missing live worktree', not s['exists']),
                                 ('LaunchAgent does not run the live worktree (run --install)', not s['plist_points_live']),
                                 ('live worktree has tracked changes', s.get('dirty')),
                                 (f'missing shared links {s.get("missing_links")}', s.get('missing_links'))) if bad]
    if problems:
        raise RuntimeError('; '.join(problems))


def restart(log=print) -> None:
    """Graceful restart: bootout (the current game finishes), wait for exit, bootstrap, verify."""
    import os
    uid = os.getuid()
    subprocess.run(['launchctl', 'bootout', f'gui/{uid}/{LABEL}'], capture_output=True)
    log('bootout sent; waiting for the current game to finish')
    deadline = time.time() + 3600
    while subprocess.run(['pgrep', '-f', SCRIPT], capture_output=True).returncode == 0:
        if time.time() > deadline:
            log('runner still running after an hour; bootstrapping anyway')
            break
        time.sleep(10)
    for _ in range(3):  # never leave the ladder player down
        if subprocess.run(['launchctl', 'bootstrap', f'gui/{uid}', str(PLIST)], capture_output=True).returncode == 0:
            break
        time.sleep(30)
    time.sleep(60)
    if subprocess.run(['pgrep', '-f', SCRIPT], capture_output=True).returncode != 0:
        raise RuntimeError('runner is not running after bootstrap')
    log('runner restarted')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--install', action='store_true', help='create the worktree (at the running commit) and point the LaunchAgent at it')
    ap.add_argument('--ref', help='commit/ref to pin, e.g. origin/main (fetches first)')
    ap.add_argument('--no-restart', action='store_true')
    args = ap.parse_args()
    if args.status:
        print(json.dumps(status(), indent=1))
        return
    changed = False
    if args.install:
        ensure(args.ref or 'HEAD')
        changed = install()
        print('LaunchAgent now runs', LIVE if changed else f'{LIVE} (already)')
    if args.ref:
        _git('fetch', '-q', 'origin', check=False)
        print('pinned', pin(args.ref))
        changed = True
    if changed and not args.no_restart:
        restart()


if __name__ == '__main__':
    main()
