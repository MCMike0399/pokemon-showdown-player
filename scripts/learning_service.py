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
    config = {"Label": LABEL, "ProgramArguments": [str(HERE/".venv/bin/python"), "-m", "ml.worker", "--schedule"],
              "WorkingDirectory": str(HERE), "RunAtLoad": True, "StartInterval": 900,
              "Nice": 10, "LowPriorityIO": True, "ProcessType": "Background",
              "StandardOutPath": str(logs/"schedule.log"), "StandardErrorPath": str(logs/"schedule-errors.log"),
              "EnvironmentVariables": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                                       "PATH": str(Path(shutil.which("node") or "/opt/homebrew/bin/node").parent)+":/usr/bin:/bin:/usr/sbin:/sbin"}}
    agent.write_bytes(plistlib.dumps(config))
    agent.chmod(0o600)
    subprocess.run(["launchctl", "bootstrap", domain, str(agent)], check=True)
    print(f"Installed {LABEL}: bounded 15-minute practice cycles, daily feeds, no other services changed")


if __name__ == "__main__":
    main()
