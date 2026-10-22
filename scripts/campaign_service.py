"""Install/resume the pinned c10 campaign supervisor on this Mac."""
import argparse
import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import live_deploy

LABEL = 'dev.pokemon-showdown.campaign-c10'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation', choices=['install', 'status'])
    args = ap.parse_args()
    main = live_deploy.MAIN
    out = main / 'data/ml/value/campaign-c10'
    domain = f'gui/{os.getuid()}'
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
