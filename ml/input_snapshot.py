"""Small immutable inference inputs, shared by content rather than by job.

Canonical training vectors and replay bodies remain in the experience store.
Inference only uses research, distinct move support and scout weights. Canonical
sample identities and bodies stay in the main store, rather than every run.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from ml.storage import Store, fingerprint


def freeze_inputs(store: Store, destination: Path, fmt: str):
    destination = Path(destination)
    if destination.exists():
        if (destination / 'experience.sqlite3').is_file():
            return  # A retry must keep the original declared inputs.
        raise ValueError('incomplete frozen input destination; inspect before retry')
    pool = store.root / 'input-snapshots'
    pool.mkdir(exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=pool, prefix='.freeze-') as temporary:
        work = Path(temporary) / 'inputs'
        frozen = Store(work)
        source = sqlite3.connect((store.root / 'experience.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
        hasher = hashlib.sha256(b'inference-inputs-v2\0' + fmt.encode())
        counts = {}
        try:
            source.execute('BEGIN')
            with frozen.db:
                for table, projection in (
                    ('research_teams', '*'), ('documents', '*'),
                ):
                    counts[table] = 0
                    hasher.update(table.encode() + b'\0')
                    cursor = source.execute(f'SELECT {projection} FROM {table} WHERE format=? ORDER BY rowid', (fmt,))
                    while rows := cursor.fetchmany(1000):
                        placeholders = ','.join('?' for _ in rows[0])
                        frozen.db.executemany(f'INSERT INTO {table} VALUES ({placeholders})', rows)
                        for row in rows:
                            hasher.update(json.dumps(row, separators=(',', ':'), ensure_ascii=True).encode() + b'\n')
                        counts[table] += len(rows)
                support = list(source.execute('SELECT DISTINCT species,move FROM scout_samples WHERE format=? ORDER BY species,move', (fmt,)))
                rows = [(fingerprint([fmt, species, move]), fmt, species, move, '[]', '') for species, move in support]
                frozen.db.executemany('INSERT INTO scout_samples VALUES(?,?,?,?,?,?)', rows)
                counts['scout_support_pairs'] = len(rows)
                hasher.update(b'move-support\0' + json.dumps(support, separators=(',', ':')).encode())
            scout = store.root / 'scouts' / (fmt + '.pt')
            if scout.exists():
                # Read once: the trainer replaces its checkpoint atomically.
                payload = scout.read_bytes()
                (work / 'scouts').mkdir()
                (work / 'scouts' / scout.name).write_bytes(payload)
                hasher.update(b'scout\0' + payload)
            frozen.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            frozen.db.execute('PRAGMA journal_mode=DELETE')
        finally:
            source.close()
            frozen.close()
        identity = hasher.hexdigest()
        manifest = {'schema': 'inference-inputs-v2', 'sha256': identity, 'format': fmt,
                    'counts': counts, 'training_allowed': False,
                    'omitted_bodies': ['scout_samples training rows', 'public_battles'],
                    'canonical_root': str(store.root.resolve())}
        (work / 'inference-only.json').write_text(json.dumps(manifest, indent=2))
        target = pool / identity
        with (pool / 'publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                for path in work.rglob('*'):
                    if path.is_file():
                        path.chmod(0o444)
                work.rename(target)
            elif json.loads((target / 'inference-only.json').read_text())['sha256'] != identity:
                raise ValueError('invalid shared inference snapshot')
            destination.symlink_to(os.path.relpath(target.resolve(), destination.parent.resolve()), target_is_directory=True)
