"""Logged probe-screening results and the learned-screening promotion pointer.

Every search decision runs a probe screening (engine.cjs): each candidate action
is simulated against a few replies only to rank it. Those probe scores are free
training labels for a ranker that orders the actions without simulating them
(`search/train_screen.py`, MPS). The live runner and the gate log them here:

    data/ml/value/teams/<slug>/screen/<source>-<id>.npz   one file per game
    data/ml/value/teams/<slug>/screen-candidates/screen-<sha16>.json
    data/ml/value/teams/<slug>/screen-current.json        what live search uses

A file holds R logged worlds and N candidate actions:

    S     (R, 2, F) float16   position features from p1's and p2's side
    A     (N, D)    float16   action features (search/screenfeat.cjs)
    y     (N,)      float32   probe score, p1's value (nan: the action was illegal)
    role  (N,)      int8      0 our action (scored vs their top replies), 1 theirs
    rec   (N,)      int32     row of S
    best  (N,)      int8      1 for the world's final choice
    turn  (R,)      int16
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from search.value_store import atomic_json


def screen_root(team: str) -> Path:
    from search.team_data import TEAMS_ROOT, slug
    return TEAMS_ROOT / slug(team) / 'screen'


def screen_pointer(team: str) -> Path:
    from search.team_data import TEAMS_ROOT, slug
    return TEAMS_ROOT / slug(team) / 'screen-current.json'


def write_records(records: list[dict], path: Path, game: str) -> Path | None:
    """One game's logged worlds -> one npz (atomic). Returns None if nothing usable."""
    S, A, y, role, rec, best, turns = [], [], [], [], [], [], []
    for item in records:
        r = item.get('record') or {}
        sx = r.get('sx')
        if not sx or len(sx) != 2:
            continue
        rows = []
        for k, key in enumerate(('ours', 'theirs')):
            part = r.get(key)
            if not part:
                continue
            feats, scores = part.get('a') or [], part.get('s') or []
            if len(feats) != len(scores):
                continue
            for i, (f, s) in enumerate(zip(feats, scores)):
                rows.append((f, np.nan if s is None else float(s), k, int(k == 0 and i == part.get('best', -1))))
        if not rows:
            continue
        idx = len(S)
        S.append(sx)
        turns.append(int(item.get('turn') or 0))
        for f, s, k, b in rows:
            A.append(f); y.append(s); role.append(k); rec.append(idx); best.append(b)
    if not S:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, S=np.asarray(S, np.float16), A=np.asarray(A, np.float16), y=np.asarray(y, np.float32),
                        role=np.asarray(role, np.int8), rec=np.asarray(rec, np.int32), best=np.asarray(best, np.int8),
                        turn=np.asarray(turns, np.int16), game=np.asarray(game), version=np.asarray(1))
    os.replace(tmp, path)
    return path


def load_records(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except (OSError, ValueError, KeyError, EOFError):
        return None


# ------------------------------------------------------------ pointer
def weights_sha(path: Path) -> str:
    """SHA-256 over the ranker and its prefilter sizes, exactly as the trainer hashed them."""
    data = json.loads(Path(path).read_text())
    weights = {k: data[k] for k in ('layers', 'mean', 'std', 'prefilter', 'version') if k in data}
    return hashlib.sha256(json.dumps(weights, separators=(',', ':')).encode()).hexdigest()


NONE = {'path': None, 'pre_opp': None, 'pre_ours': None, 'sha256': None}


def current_screen(pointer: Path) -> dict:
    """The promoted ranker, verified; probe screening (path None) if none or damaged."""
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
        model = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {**NONE, 'rejected': 'unreadable'}
    pre = model.get('prefilter') or {}
    return {'path': str(path), 'pre_opp': int(pre['opp']), 'pre_ours': int(pre['ours']), 'sha256': data['sha256']}


def promote_screen(candidate: Path, evidence: dict, pointer: Path) -> dict:
    record = {'path': str(Path(candidate).resolve()), 'sha256': weights_sha(candidate),
              'promoted_at': time.time(), 'evidence': evidence}
    atomic_json(pointer, record)
    with (pointer.parent / 'screen-promotions.jsonl').open('a') as fh:
        fh.write(json.dumps(record) + '\n')
    return record


def model_from_file(path: Path | None) -> dict | None:
    """A screening candidate file as the model dict SearchAgent.set_screen takes."""
    if not path:
        return None
    pre = json.loads(Path(path).read_text())['prefilter']
    return {'path': str(Path(path).resolve()), 'pre_opp': int(pre['opp']), 'pre_ours': int(pre['ours']),
            'sha256': weights_sha(Path(path))}


def engine_model(screen: dict | None) -> dict | None:
    """The `learned` part of an engine screen message, or None for probe screening."""
    if not screen or not screen.get('path'):
        return None
    return {'path': screen['path'], 'pre_opp': int(screen['pre_opp']), 'pre_ours': int(screen['pre_ours'])}
