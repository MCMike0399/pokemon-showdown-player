#!/usr/bin/env python3
"""Yield registered offline search workers to games/transcription without altering pinned experiments.

macOS taskpolicy changes background QoS, not SIGSTOP: live search retains sole
ownership of its existing pause/resume protocol. Never targets the live player.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import psutil


def needs_headroom(state, now):
    if not -5 <= now-state.get('at', 0) <= 90:
        return False
    if (state.get('worker') or {}).get('active'):
        return True
    return state.get('realtimeBusy') is True


def worker_active(token_file):
    try:
        request = urllib.request.Request('http://127.0.0.1:18510/workloads/v1/status',
            headers={'Authorization':'Bearer '+token_file.read_text().strip()})
        with urllib.request.urlopen(request,timeout=2) as response:
            return bool(json.loads(response.read(65536)).get('active'))
    except (OSError, ValueError): return False


def is_offline_leader(args, root):
    # Exact executable path inside this project's pinned sibling worktrees.
    if len(args) < 3 or 'worker' not in args or '--dir' not in args:
        return False
    paths = [Path(arg) for arg in args if arg.endswith('/search/campaign.py')]
    if len(paths) != 1:
        return False
    checkout = paths[0].parent.parent
    if checkout.parent != root.parent or not checkout.name.startswith(root.name+'.c'):
        return False
    try:
        directory = Path(args[args.index('--dir')+1]).resolve()
        return directory.is_relative_to((root/'data/ml/value').resolve())
    except (IndexError, OSError):
        return False


def targets(root):
    result = {}
    registry = root/'data/ml/value/extra-pgids.json'
    try: leaders = json.loads(registry.read_text())
    except (OSError, ValueError): return result
    for pid in leaders:
        if type(pid) is not int or pid <= 1: continue
        try:
            process = psutil.Process(pid)
            if not is_offline_leader(process.cmdline(), root): continue
            for child in [process, *process.children(recursive=True)]:
                result[str(child.pid)] = child.create_time()
        except (psutil.Error, OSError): continue
    return result


def apply(pid, created, background):
    try:
        process = psutil.Process(int(pid))
        if process.create_time() != created: return False
        return subprocess.run(['/usr/sbin/taskpolicy', '-b' if background else '-B', '-p', str(pid)],
                              capture_output=True, timeout=5).returncode == 0
    except (psutil.Error, OSError, subprocess.SubprocessError): return False


def reconcile(owned, current, busy, setter=apply):
    next_owned = dict(owned)
    for pid, created in owned.items():
        if not busy or current.get(pid) != created:
            setter(pid, created, False)
            next_owned.pop(pid, None)
    if busy:
        for pid, created in current.items():
            if next_owned.get(pid) != created and setter(pid, created, True):
                next_owned[pid] = created
    return next_owned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    try: owned = json.loads(args.receipt.read_text()).get('owned', {})
    except (OSError, ValueError): owned = {}
    while True:
        try: state = json.loads(args.state.read_text())
        except (OSError, ValueError): state = {}
        busy = needs_headroom(state, time.time()) or worker_active(args.state.parent/'worker.token')
        owned = reconcile(owned, targets(args.root), busy)
        receipt = {'at':time.time(), 'yielding':busy, 'owned':owned}
        temp = args.receipt.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt)); temp.replace(args.receipt)
        if args.once: break
        time.sleep(5)


if __name__ == '__main__': main()
