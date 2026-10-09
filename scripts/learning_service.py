"""Install/status/remove this project's bounded launchd agent; no other services change."""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LABEL = "dev.pokemon-showdown.learning"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("install", "status", "remove"))
    parser.add_argument('--pipeline', action='store_true', help='retain independent collection, learning and evaluation lanes')
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("launchd installer requires macOS; elsewhere schedule python -m ml.worker --schedule with your job runner")
    agent = Path.home()/"Library/LaunchAgents"/(LABEL+".plist")
    domain = f"gui/{os.getuid()}"
    if args.operation == "status":
        subprocess.run(["launchctl", "print", domain+"/"+LABEL], check=False)
        return
    if agent.exists():
        subprocess.run(["launchctl", "bootout", domain, str(agent)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if args.operation == "remove":
        agent.unlink(missing_ok=True)
        return
    logs = HERE/"data/ml/logs"
    logs.mkdir(parents=True, exist_ok=True)
    agent.parent.mkdir(parents=True, exist_ok=True)
    config = {"Label": LABEL, "ProgramArguments": [str(HERE/".venv/bin/python"), "-m", *(['ml.pipeline'] if args.pipeline else ['ml.worker', '--schedule'])],
              "WorkingDirectory": str(HERE), "RunAtLoad": True,
              "Nice": 10, "LowPriorityIO": True, "ProcessType": "Background",
              "StandardOutPath": str(logs/"schedule.log"), "StandardErrorPath": str(logs/"schedule-errors.log"),
              "EnvironmentVariables": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                                       "PATH": str(Path(shutil.which("node") or "/opt/homebrew/bin/node").parent)+":/usr/bin:/bin:/usr/sbin:/sbin"}}
    if args.pipeline:
        config.update(KeepAlive=True, ThrottleInterval=30, ExitTimeOut=1200)
    else:
        config['StartInterval'] = 900
    agent.write_bytes(plistlib.dumps(config))
    agent.chmod(0o600)
    subprocess.run(["launchctl", "bootstrap", domain, str(agent)], check=True)
    print(f"Installed {LABEL}: " + ('retained CPU collection, GPU learning and CPU evaluation lanes' if args.pipeline else 'bounded 15-minute practice cycles'))


if __name__ == "__main__":
    main()
