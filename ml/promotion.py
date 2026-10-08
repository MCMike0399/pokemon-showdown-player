"""Durable candidate promotion at an explicit boundary before matchmaking."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

from ml.model import Model
from ml.storage import Store, WriterBusy, now


def paired_gate(base: list[dict], candidate: list[dict], games: int, margin: float) -> dict:
    complete = len(base) == len(candidate) == games
    keys = ('seed', 'learner_side', 'opponent', 'opponent_revision', 'learner_team', 'opponent_team', 'open_team_sheets')
    paired = complete and all(tuple(a.get(key) for key in keys) == tuple(b.get(key) for key in keys)
                              and a.get('seed_encoding', 'legacy') == b.get('seed_encoding', 'legacy')
                              and a.get('simulator_seed') == b.get('simulator_seed')
                              for a, b in zip(base, candidate))
    clean = paired and all(not g.get('unfinished') and not g.get('rejected_actions') and
                           (g.get('winner') or g.get('tie')) for g in base + candidate)
    wins = lambda rows: sum(g.get('winner') == 'LocalBrain' for g in rows)
    gained = sum(a.get('winner') != 'LocalBrain' and b.get('winner') == 'LocalBrain' for a, b in zip(base, candidate))
    lost = sum(a.get('winner') == 'LocalBrain' and b.get('winner') != 'LocalBrain' for a, b in zip(base, candidate))
    discordant = gained + lost
    # Exact one-sided paired sign test; a tiny noisy win margin is insufficient.
    p = sum(math.comb(discordant, i) for i in range(gained, discordant + 1)) / 2**discordant if discordant else 1.0
    return {'complete': complete, 'clean': clean, 'same_seeds_and_sides': paired,
            'incumbent_wins': wins(base), 'candidate_wins': wins(candidate),
            'paired_gained': gained, 'paired_lost': lost, 'paired_p_value': p,
            'passed': clean and wins(candidate) - wins(base) >= math.ceil(games * margin) and p <= .05}


def stage(store: Store, fmt: str, parent_revision: str, checkpoint: Path, evaluation: dict) -> bool:
    """Caller holds writer lock; rejected candidates never enter the ready slot."""
    if not evaluation.get('passed'):
        return False
    from ml.reload import source_generation
    evaluation = {'source_generation': source_generation(), **evaluation}
    folder = store.root / 'ready'
    folder.mkdir(exist_ok=True)
    path = folder / (fmt + '.json')
    if path.exists():
        return False
    data = {'format': fmt, 'parent_revision': parent_revision, 'checkpoint': str(checkpoint.resolve()),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'evaluation': evaluation,
            'staged_at': now()}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)
    return True


def promote_ready(store: Store, fmt: str) -> dict:
    """Called before search, never inside a recorded battle or by the worker."""
    path = store.root / 'ready' / (fmt + '.json')
    if not path.exists():
        return {'promoted': False, 'reason': 'no staged candidate'}
    try:
        with store.writer():
            return _promote_locked(store, fmt, path)
    except WriterBusy:
        return {'promoted': False, 'reason': 'learning writer busy; promotion deferred'}


def _promote_locked(store: Store, fmt: str, path: Path) -> dict:
    if not path.exists():
        return {'promoted': False, 'reason': 'no staged candidate'}
    data = json.loads(path.read_text())
    from ml.reload import source_generation
    evaluated_source = data['evaluation'].get('source_generation')
    if evaluated_source != source_generation():
        path.unlink()  # Checkpoint/report remain; only its obsolete ready slot expires.
        return {'promoted': False, 'reason': 'source changed since evaluation; candidate retained for audit'}
    if store.db.execute("SELECT 1 FROM episodes WHERE format=? AND source='ladder' AND status='pending' LIMIT 1", (fmt,)).fetchone():
        return {'promoted': False, 'reason': 'pending ladder recording'}
    incumbent = Model(store.root, fmt)
    if incumbent.revision != data['parent_revision']:
        path.unlink()
        return {'promoted': False, 'reason': 'stale parent revision'}
    candidate_path = Path(data['checkpoint'])
    if data['format'] != fmt or not data['evaluation'].get('passed') or hashlib.sha256(candidate_path.read_bytes()).hexdigest() != data['sha256']:
        raise ValueError('staged candidate integrity/gate mismatch; incumbent retained')
    candidate = Model(candidate_path.parent.parent, fmt)
    temporary = incumbent.path.with_suffix('.promote.tmp')
    shutil.copy2(candidate.path, temporary)
    temporary.replace(incumbent.path)
    path.unlink()
    result = {'promoted': True, 'previous_revision': incumbent.revision, 'revision': candidate.revision,
              'evaluation': data['evaluation'], 'promoted_at': now()}
    (store.root / 'ready' / (fmt + '-last-promotion.json')).write_text(json.dumps(result, indent=2))
    return result
