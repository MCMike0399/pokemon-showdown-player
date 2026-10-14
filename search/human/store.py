"""Promotion pointer for the human opponent policy, per team (docs/human-opponent.md).

    data/ml/value/teams/<slug>/opp-current.json   {path, sha256, w, temp, promoted_at, evidence}

Like the value and screening pointers, a damaged or edited file falls back to the
equilibrium/greedy opponent model, and every read verifies the weights' SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from search.value_store import atomic_json


def opp_pointer(team: str) -> Path:
    from search.team_data import TEAMS_ROOT, slug
    return TEAMS_ROOT / slug(team) / 'opp-current.json'


def weights_sha(path: Path) -> str:
    data = json.loads(Path(path).read_text())
    weights = {k: data[k] for k in ('layers', 'mean', 'std', 'version') if k in data}
    return hashlib.sha256(json.dumps(weights, separators=(',', ':')).encode()).hexdigest()


NONE = {'path': None, 'w': None, 'temp': None, 'sha256': None}


def current_opp(pointer: Path) -> dict:
    try:
        data = json.loads(Path(pointer).read_text())
    except (OSError, ValueError):
        return dict(NONE)
    path = data.get('path')
    if not path:
        return dict(NONE)
    from search.value_store import localize
    path = localize(path)
    try:
        if weights_sha(Path(path)) != data.get('sha256'):
            return {**NONE, 'rejected': 'sha mismatch'}
    except (OSError, ValueError):
        return {**NONE, 'rejected': 'unreadable'}
    return {'path': str(path), 'w': float(data.get('w', 0.3)), 'temp': float(data.get('temp', 1.0)), 'sha256': data['sha256']}


def promote_opp(candidate: Path, w: float, temp: float, evidence: dict, pointer: Path) -> dict:
    record = {'path': str(Path(candidate).resolve()), 'sha256': weights_sha(candidate), 'w': w, 'temp': temp,
              'promoted_at': time.time(), 'evidence': evidence}
    atomic_json(pointer, record)
    with (pointer.parent / 'opp-promotions.jsonl').open('a') as fh:
        fh.write(json.dumps(record) + '\n')
    return record


def model_from_file(path: Path | None, w: float = 0.3, temp: float = 1.0) -> dict | None:
    if not path:
        return None
    return {'path': str(Path(path).resolve()), 'w': w, 'temp': temp, 'sha256': weights_sha(Path(path))}
