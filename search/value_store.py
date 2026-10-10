"""Files shared by the value service, the trainer, the gate and the live runner.

Layout under data/ml/value/:
    selfplay/*.jsonl      self-play shards (one producer process each), plus a
                          `.npz` float32 cache written once a shard is finished
    live/live-<room>.jsonl rebuilt live positions (search/ledger_positions.py)
    candidates/value-<sha16>.json   immutable trained candidates
    gates/<sha16>-<stage>.jsonl     paired gate results (resumable)
    current.json          the promotion pointer the live runner reads
    promotions.jsonl      every promotion, append-only
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VALUE_ROOT = ROOT / 'data' / 'ml' / 'value'
POINTER = VALUE_ROOT / 'current.json'


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def weights_sha(path: Path) -> str:
    """SHA-256 of the weights exactly as the trainer hashed them (meta excluded)."""
    data = json.loads(Path(path).read_text())
    weights = {k: data[k] for k in ('layers', 'mean', 'std', 'hand') if k in data}
    return hashlib.sha256(json.dumps(weights, separators=(',', ':')).encode()).hexdigest()


def current_value(pointer: Path = POINTER) -> dict:
    """The promoted value for live search, verified; hand evaluation if none.

Returns {'path': str|None, 'beta': float, 'sha256': str|None}. A pointer whose
file is missing or whose weights hash differs is ignored (hand evaluation), so
a damaged candidate can never reach live play.
"""
    try:
        data = json.loads(Path(pointer).read_text())
    except (OSError, ValueError):
        return {'path': None, 'beta': 0.0, 'sha256': None}
    path = data.get('path')
    beta = float(data.get('beta') or 0)
    if not path or not beta:
        return {'path': None, 'beta': 0.0, 'sha256': None}
    try:
        if weights_sha(Path(path)) != data.get('sha256'):
            return {'path': None, 'beta': 0.0, 'sha256': None, 'rejected': 'sha mismatch'}
    except (OSError, ValueError):
        return {'path': None, 'beta': 0.0, 'sha256': None, 'rejected': 'unreadable'}
    return {'path': str(path), 'beta': beta, 'sha256': data['sha256']}


def promote(candidate: Path, beta: float, evidence: dict, pointer: Path = POINTER) -> dict:
    sha = weights_sha(candidate)
    record = {'path': str(Path(candidate).resolve()), 'sha256': sha, 'beta': float(beta),
              'promoted_at': time.time(), 'evidence': evidence}
    atomic_json(pointer, record)
    with (pointer.parent / 'promotions.jsonl').open('a') as fh:
        fh.write(json.dumps(record) + '\n')
    return record


FIELDS = ('y', 'h', 't', 's', 'f')


def pack_shard(path: Path) -> Path | None:
    """Cache a finished JSONL shard as float32 arrays (rebuildable, lossless to 1e-7)."""
    path = Path(path)
    xs, cols, games = [], {k: [] for k in FIELDS}, []
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'g' not in r or r.get('h') is None:
                continue
            xs.append(r['x'])
            for k in FIELDS:
                v = r.get(k)
                cols[k].append(np.nan if v is None else float(v))
            games.append(r['g'])
    if not xs:
        return None
    out = path.with_suffix('.npz')
    st = path.stat()
    tmp = out.with_name(out.stem + '.tmp.npz')
    np.savez(tmp, x=np.asarray(xs, np.float32), g=np.asarray(games), source_bytes=st.st_size,
             **{k: np.asarray(v, np.float32) for k, v in cols.items()})
    os.replace(tmp, out)
    return out


def load_shard(path: Path):
    """(x, y, h, g) for a shard, from its cache when the cache matches the JSONL size."""
    path = Path(path)
    cache = path.with_suffix('.npz')
    if cache.exists():
        try:
            data = np.load(cache, allow_pickle=False)
            if int(data['source_bytes']) == path.stat().st_size:
                return data['x'], data['y'], data['h'], data['g'].tolist()
        except (OSError, ValueError, KeyError):
            pass
    xs, ys, hs, gs = [], [], [], []
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a shard being appended may end mid-line
            if 'g' not in r or r.get('h') is None:
                continue  # legacy rows without a game id cannot be split safely
            xs.append(r['x']); ys.append(float(r['y'])); hs.append(float(r['h'])); gs.append(r['g'])
    if not xs:
        return np.zeros((0, 0), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32), []
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(hs, np.float32), gs


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).glob('*') if p.is_file()) if Path(path).exists() else 0


# ------------------------------------------------------------ live priority
# Background simulation runs at normal QoS so it can use every core, and the
# live runner pauses it for the few seconds it spends deciding a turn. The
# value service publishes its children's process groups; the runner stops them
# (SIGSTOP) while searching and resumes them (SIGCONT) afterwards. If a runner
# dies mid-decision, the service resumes everything once the marker is stale.
BACKGROUND_PGIDS = VALUE_ROOT / 'background-pgids.json'
PAUSE_MARKER = VALUE_ROOT / 'live-deciding'
PAUSE_STALE_SECONDS = 45
MARKER = 'search/'  # every pausable leader runs search/selfplay.cjs or search/value_gate.py


def _signal_groups(sig) -> int:
    import signal as _signal  # noqa: F401
    try:
        pgids = json.loads(BACKGROUND_PGIDS.read_text())
    except (OSError, ValueError):
        return 0
    pgids = [int(p) for p in pgids if str(p).isdigit()]
    if not pgids:
        return 0
    # Only signal groups whose leader is still one of our search programs: a
    # published pid may have been reused since the service last wrote the list.
    ours = [pid for pid in pgids if MARKER in _command(pid)]
    sent = 0
    for pgid in ours:
        try:
            os.killpg(pgid, sig)
            sent += 1
        except OSError:
            pass
    return sent


def _command(pid: int) -> str:
    """Command line of a process ('' if gone): /proc where available, else ps."""
    proc = Path('/proc') / str(pid) / 'cmdline'
    if proc.parent.exists():
        try:
            return proc.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            return ''
    import subprocess
    try:
        return subprocess.run(['ps', '-ww', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ''


class LivePriority:
    """`async with LivePriority():` around a live decision; never raises."""

    async def __aenter__(self):
        import signal
        try:
            PAUSE_MARKER.write_text(str(time.time()))
            self.paused = _signal_groups(signal.SIGSTOP)
        except OSError:
            self.paused = 0
        return self

    async def __aexit__(self, *exc):
        import signal
        _signal_groups(signal.SIGCONT)
        try:
            PAUSE_MARKER.unlink()
        except OSError:
            pass
        return False


def resume_if_stale() -> bool:
    """Service failsafe: resume background work if no live decision is in progress."""
    import signal
    try:
        started = float(PAUSE_MARKER.read_text())
    except (OSError, ValueError):
        started = None
    if started is not None and time.time() - started < PAUSE_STALE_SECONDS:
        return False
    _signal_groups(signal.SIGCONT)
    if started is not None:
        try:
            PAUSE_MARKER.unlink()
        except OSError:
            pass
    return True
