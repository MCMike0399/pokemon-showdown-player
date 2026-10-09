"""Export only reviewable source into a fresh directory; fail closed on private data.

Never copies Git history, credentials, account-specific logs, datasets or weights.
The public repository can be initialized in the resulting directory after scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FILES = [".env.example", ".gitignore", "README.md", "LICENSE", "requirements.txt", "requirements-ml.txt", "requirements-browser.txt",
         "package.json", "package-lock.json", "pytest.ini", "battle_state.py", "harness.py", "policy.py",
         "ps_client.py", "ps_mcp_server.py", "simulator.cjs", "selftest.py", "mcp_selftest.py"]
GLOBS = ["ml/*.py", "docs/*.md", "docs/paper/*.tex", "docs/paper/*.bib", "docs/paper/*.md",
         "web/live-watch/*.html", "web/live-watch/*.js",
         "docs/paper/*.pdf", "docs/paper/.gitignore", "examples/*.json", "scripts/*.py", "tests/test_offline_*.py",
         "tests/conftest.py", "tests/test_teams.py", ".github/workflows/*.yml",
         "search/*.py", "search/*.cjs"]
PUBLIC_AGENTS = """# Agent instructions

This harness gives an agent one regular Pokemon Showdown player connection.

For OpenClaw battle/team decisions, read `docs/agent-workflow.md` before playing.
For CPU/MPS training, schedules or feeds, read `docs/continuous-learning.md`.
For corpus changes, read `docs/continuous-learning-research.md`; retain source,
license and exact regulation, and validate observable-state labels.

Use the configured account in `.env`, log in once, and reuse its connection.
Read waiting requests before choosing a complete legal joint action. Keep the
collecting checkpoint fixed during a recorded game. Training uses background
resource budgets and candidate evaluation before promotion.

Keep credentials in `.env` or environment variables. Keep datasets, replays,
team stores, weights, reports and machine-specific settings under gitignored
local directories. Respect Showdown's rules and use the offline simulator for
bulk training. Use the shared host's configured compute headroom.

Run `.venv/bin/python -m pytest -q` after battle, model or feed changes. Run
`scripts/public_snapshot.py` before publishing a clean source snapshot.
"""


def private_values():
    values = [str(Path.home()), "/" + "Users" + "/"]
    env = HERE/".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                value = value.strip().strip("\"'")
                if key.strip() in ("PS_USERNAME", "PS_PASSWORD", "BRAVE_SEARCH_API_KEY") and value:
                    values.append(value)
    return values


def scan(root: Path):
    problems = []
    forbidden = private_values()
    patterns = [r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", r"gh[pousr]_[A-Za-z0-9]{20,}",
                r"AKIA[0-9A-Z]{16}", r"sk-[A-Za-z0-9_-]{24,}"]
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        relative = str(path.relative_to(root))
        if (path.name.startswith(".env") and path.name != ".env.example") or path.suffix in (".pt", ".pth", ".log", ".sqlite3", ".key", ".pem", ".zip"):
            problems.append({"file": relative, "reason": "private artifact type"})
            continue
        text = path.read_text(errors="replace")
        if any(value in text for value in forbidden) or any(re.search(pattern, text) for pattern in patterns):
            problems.append({"file": relative, "reason": "private value or secret pattern detected"})
    if problems:
        raise ValueError("snapshot rejected: " + json.dumps(problems))


def export(destination: Path):
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("destination must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    selected = {HERE/f for f in FILES}
    for pattern in GLOBS:
        selected.update(HERE.glob(pattern))
    manifest = []
    for source in sorted(selected):
        relative = source.relative_to(HERE)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append({"file": str(relative), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination/"AGENTS.md").write_text(PUBLIC_AGENTS)
    scan(destination)
    return {"files": len(manifest)+1, "clean_snapshot": str(destination), "history_copied": False,
            "secret_scan": "passed", "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    if args.scan_only:
        scan(args.destination)
        print(json.dumps({"secret_scan": "passed"}))
    else:
        print(json.dumps(export(args.destination), indent=2))
