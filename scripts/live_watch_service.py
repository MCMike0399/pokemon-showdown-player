#!/usr/bin/env python3
"""Install or inspect the independent macOS live viewer LaunchAgent."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
LABEL = 'dev.pokemon-showdown.live-watch'
PLIST = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'status', 'uninstall'))
    parser.add_argument('--port', type=int, default=18489)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('LaunchAgent management requires macOS; run live_watch.py directly elsewhere')
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    domain = f'gui/{os.getuid()}'
    service = f'{domain}/{LABEL}'
    if args.action == 'status':
        result = subprocess.run(['launchctl', 'print', service])
        return result.returncode
    if args.action == 'uninstall':
        subprocess.run(['launchctl', 'bootout', service], check=False, capture_output=True)
        PLIST.unlink(missing_ok=True)
        return 0
    logs = PROJECT / 'data/ml/logs'
    logs.mkdir(parents=True, exist_ok=True)
    config = {
        'Label': LABEL, 'ProgramArguments': [str(PROJECT / '.venv/bin/python'),
            str(PROJECT / 'scripts/live_watch.py'), '--port', str(args.port)],
        'WorkingDirectory': str(PROJECT), 'RunAtLoad': True, 'KeepAlive': True,
        'ThrottleInterval': 10, 'ProcessType': 'Background', 'LowPriorityIO': True,
        'StandardOutPath': str(logs / 'live-watch.log'),
        'StandardErrorPath': str(logs / 'live-watch-errors.log'),
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    temporary = PLIST.with_suffix('.tmp')
    temporary.write_bytes(plistlib.dumps(config))
    temporary.replace(PLIST)
    subprocess.run(['launchctl', 'bootout', service], check=False, capture_output=True)
    subprocess.run(['launchctl', 'bootstrap', domain, str(PLIST)], check=True)
    artifact = PROJECT / 'artifacts/live-watch'
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / 'service.json').write_text(json.dumps({
        'label': LABEL, 'port': args.port, 'url': f'http://localhost:{args.port}/',
        'read_only': True, 'inline_renderer': True, 'player_relay': True,
        'transport': 'server-sent-events', 'supervised': True,
        'source': 'scripts/live_watch.py', 'assets': 'web/live-watch',
    }, indent=2) + '\n')
    print(f'Installed {LABEL} on 127.0.0.1:{args.port}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
