"""Install/resume the pinned c10 campaign supervisor on this Mac."""
import argparse
import fcntl
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import live_deploy

LABEL = 'dev.pokemon-showdown.campaign-c10'


def scale(out, slots):
    """Change only the protocol's mutable slot count; resume the same pinned ledgers."""
    from search.campaign import verify_protocol, command
    from search.value_store import atomic_json
    if slots not in (2, 4, 6, 8):
        raise ValueError('slots must be 2, 4, 6 or 8')
    lock = (out.parent / 'compute-benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = json.loads((out/'spec.json').read_text())
    state = json.loads((out/'state.json').read_text())
    verify_protocol(out, spec)
    if state['phase'] != 'offline_gates' or not spec['active']:
        raise ValueError('slot scaling is only supported during active offline gates')
    if slots == spec['slots']:
        return {'unchanged': True, 'slots': slots}
    jobs = [state['jobs'][k] for k in ('preview', 'endgame') if not (out/f'{k}-result.json').exists()]
    if any(j['attempts'] >= 3 for j in jobs):
        raise ValueError('a worker has exhausted its start budget; inspect before scaling')
    domain = f'gui/{os.getuid()}'
    plist = Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
    receipt = {'time': time.time(), 'from': spec['slots'], 'to': slots,
               'source_commit': spec['source_commit'], 'protocol_hash': spec['protocol_hash'],
               'note': 'Planned worker restart; completed rows retained, interrupted seeds replayed unchanged.'}
    subprocess.run(['launchctl', 'bootout', domain+'/'+LABEL], check=True)
    try:
        deadline = time.time()+30
        while 'search/campaign.py run' in command(state['pid']):
            if time.time() > deadline:
                raise RuntimeError('supervisor did not stop; slot setting retained')
            time.sleep(.5)
        # No supervisor can now start a duplicate or overwrite its job state.
        for j in jobs:
            if 'search/campaign.py worker' in command(j['pid']) and str(out) in command(j['pid']):
                os.killpg(j['pid'], signal.SIGTERM)
                try:
                    os.killpg(j['pid'], signal.SIGCONT)  # a live decision may have stopped it
                except ProcessLookupError:
                    pass
        deadline = time.time()+30
        while any('search/campaign.py worker' in command(j['pid']) for j in jobs):
            if time.time() > deadline:
                raise RuntimeError('workers did not exit; slot setting retained')
            time.sleep(.5)
        spec['slots'] = slots
        verify_protocol(out, spec)
        atomic_json(out/'spec.json', spec)
        atomic_json(out/f'scale-{int(receipt["time"])}.json', receipt)
    finally:
        subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], check=True)
        lock.close()
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation', choices=['install', 'status', 'scale'])
    ap.add_argument('--slots', type=int, choices=[2, 4, 6, 8])
    args = ap.parse_args()
    main = live_deploy.MAIN
    out = main / 'data/ml/value/campaign-c10'
    domain = f'gui/{os.getuid()}'
    if args.operation == 'scale':
        if args.slots is None:
            ap.error('scale requires --slots')
        print(json.dumps(scale(out, args.slots), indent=1))
        return
    if args.operation == 'status':
        subprocess.run(['launchctl', 'print', domain + '/' + LABEL], check=False)
        subprocess.run([str(main / '.venv/bin/python'), str(main / 'search/campaign.py'), 'status', '--dir', str(out)])
        return
    if sys.platform != 'darwin':
        ap.error('launchd installation requires macOS')
    if subprocess.run(['launchctl', 'print', domain + '/' + LABEL], capture_output=True).returncode == 0:
        print('Supervisor already installed; its state and running jobs were retained.')
        return
    pin = main.parent / (main.name + '.c10')
    ref = (json.loads((out / 'spec.json').read_text())['source_commit'] if (out / 'spec.json').exists()
           else live_deploy._git('rev-parse', 'HEAD'))
    live_deploy.ensure(ref, pin)
    if live_deploy._git('rev-parse', 'HEAD', cwd=pin) != ref or live_deploy.dirty(pin):
        raise RuntimeError('existing c10 worktree differs from the frozen campaign; inspect it before resuming')
    if not (pin / '.venv').exists():
        (pin / '.venv').symlink_to(main / '.venv')
    py = str(main / '.venv/bin/python')
    if not (out / 'spec.json').exists():
        subprocess.run([py, str(pin / 'search/campaign.py'), 'init', '--dir', str(out), '--main', str(main)],
                       cwd=pin, check=True, stdout=subprocess.DEVNULL)
    plist = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')
    config = {'Label': LABEL, 'ProgramArguments': [py, '-u', str(pin / 'search/campaign.py'), 'run', '--dir', str(out)],
              'WorkingDirectory': str(pin), 'RunAtLoad': True, 'KeepAlive': True, 'ThrottleInterval': 30,
              'StandardOutPath': str(out / 'supervisor.log'), 'StandardErrorPath': str(out / 'supervisor-errors.log'),
              'EnvironmentVariables': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                      'PATH': str(Path(shutil.which('node')).parent) + ':/usr/bin:/bin:/usr/sbin:/sbin'}}
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(plistlib.dumps(config))
    subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], check=True)
    print('Started ' + LABEL + ' from ' + ref + '; state: ' + str(out / 'state.json'))


if __name__ == '__main__':
    main()
