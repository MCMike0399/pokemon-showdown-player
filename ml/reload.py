"""Pin Python battle modules and detect deployments at safe work boundaries."""
from __future__ import annotations

import hashlib
import importlib
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNTIME_GENERATION = None


def source_generation(root: Path = REPO) -> str:
    paths = [root / name for name in ('ps_client.py', 'ps_mcp_server.py', 'harness.py',
             'battle_state.py', 'policy.py', 'simulator.cjs', 'package-lock.json',
             'scripts/campaign_ladder.py')]
    paths.extend(sorted((root / 'ml').glob('*.py')))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SourceChanged(Exception):
    """Finish the current owned operation, then restart on the new generation."""


def pin_runtime(expected: str):
    """Load optional battle logic before a game, avoiding lazy mid-game imports.

Entrypoints are excluded. Data/checkpoints stay in their existing root; this
pins module objects, not a copy of credentials or a second player connection.
"""
    global RUNTIME_GENERATION
    for path in sorted((REPO / 'ml').glob('*.py')):
        if path.stem not in ('__init__', 'cli', 'worker', 'pipeline', 'reload'):
            importlib.import_module('ml.' + path.stem)
    if source_generation() != expected:
        raise SourceChanged('source changed during runtime initialization')
    RUNTIME_GENERATION = expected


def runtime_generation() -> str:
    return RUNTIME_GENERATION or source_generation()


class SourceGuard:
    def __init__(self, generation: str):
        self.generation = generation
        self.next_check = 0.0

    def __call__(self, force: bool = False):
        if not force and time.monotonic() < self.next_check:
            return
        if source_generation() != self.generation:
            raise SourceChanged('source changed; restart at the next work boundary')
        self.next_check = time.monotonic() + 5
