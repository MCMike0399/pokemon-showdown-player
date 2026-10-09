"""Install the continuous local-model/Playwright player as a launchd service."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time
import re

HERE = Path(__file__).resolve().parents[1]
LABEL = 'dev.pokemon-showdown.browser-player'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('install', 'status', 'remove'))
    parser.add_argument('--team', default='Rain-Recife-special-stat-fix')
    parser.add_argument('--room', help='Resume this exact room when starting the service')
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('launchd requires macOS; run browser_play.py --continuous elsewhere')
    domain = f'gui/{os.getuid()}'
    agent = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')
    if args.operation == 'status':
        subprocess.run(['launchctl', 'print', domain + '/' + LABEL], check=False)
        return
    if args.operation == 'remove':
        # SIGTERM lets browser_play finish the active game before exiting.
        subprocess.run(['launchctl', 'kill', 'SIGTERM', domain + '/' + LABEL], check=False)
        deadline = time.monotonic() + 1200
        while True:
            status = subprocess.run(['launchctl', 'print', domain + '/' + LABEL], capture_output=True, text=True)
            if status.returncode != 0 or not re.search(r'\bpid = \d+', status.stdout):
                break
            if time.monotonic() >= deadline:
                parser.error('active player retained; graceful stop has not finished')
            time.sleep(2)
        subprocess.run(['launchctl', 'bootout', domain, str(agent)], check=False)
        agent.unlink(missing_ok=True)
        print('Removed ' + LABEL + ' after the active game finished')
        return
    existing = subprocess.run(['launchctl', 'print', domain + '/' + LABEL], capture_output=True)
    if existing.returncode == 0:
        parser.error('player service already installed; do not replace an active battle owner')
    logs = HERE / 'data/ml/logs'
    logs.mkdir(parents=True, exist_ok=True)
    command = [str(HERE / '.venv/bin/python'), '-u', str(HERE / 'scripts/browser_play.py'),
               '--team', args.team, '--continuous', '--resume-active', '--chrome',
               '--search', '--search-worlds', '6', '--search-engines', '3']
    if args.room:
        command.extend(['--room', args.room])
    config = {'Label': LABEL, 'ProgramArguments': command, 'WorkingDirectory': str(HERE),
              'RunAtLoad': True, 'ExitTimeOut': 1200,
              'StandardOutPath': str(logs / 'browser-player.log'),
              'StandardErrorPath': str(logs / 'browser-player-errors.log'),
              'EnvironmentVariables': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                      'PATH': str(Path(shutil.which('node') or '/opt/homebrew/bin/node').parent) + ':/usr/bin:/bin:/usr/sbin:/sbin'}}
    # No automatic retry on server rejection or ambiguous battle ownership.
    # The continuous loop itself retains the process; failures need inspection.
    agent.parent.mkdir(parents=True, exist_ok=True)
    agent.write_bytes(plistlib.dumps(config))
    agent.chmod(0o600)
    subprocess.run(['launchctl', 'bootstrap', domain, str(agent)], check=True)
    print('Started ' + LABEL + ': continuous local inference/search and Playwright clicks')


if __name__ == '__main__':
    main()
