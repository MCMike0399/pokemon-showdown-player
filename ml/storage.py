"""Durable episodes, demonstrations and sourced research, with atomic SQLite writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import fcntl
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "ml"


class WriterBusy(ValueError):
    """Temporary contention, distinguishable from invalid training state."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


class Store:
    def __init__(self, root: Path = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # MCP serializes brain operations with its async lock; training runs in
        # a worker thread so the live websocket reader remains responsive.
        self.db = sqlite3.connect(self.root / "experience.sqlite3", check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, format TEXT NOT NULL, team TEXT NOT NULL,
                source TEXT NOT NULL, revision TEXT NOT NULL, created TEXT NOT NULL,
                outcome REAL, status TEXT NOT NULL DEFAULT 'pending', trained INTEGER DEFAULT 0,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS episodes_room_side ON episodes
                (json_extract(data,'$.room'), json_extract(data,'$.side'));
            CREATE INDEX IF NOT EXISTS episodes_training ON episodes
                (format,status,trained,revision,team);
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, url TEXT NOT NULL, format TEXT NOT NULL,
                title TEXT NOT NULL, fetched TEXT NOT NULL, published TEXT,
                text TEXT NOT NULL, digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_teams (
                id TEXT PRIMARY KEY, format TEXT NOT NULL, url TEXT NOT NULL,
                player TEXT, event TEXT, fetched TEXT NOT NULL, sets TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, format TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued', created TEXT NOT NULL,
                updated TEXT NOT NULL, attempts INTEGER DEFAULT 0,
                lease_until REAL DEFAULT 0, payload TEXT NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS public_battles (
                id TEXT PRIMARY KEY, format TEXT NOT NULL, source TEXT NOT NULL,
                fetched TEXT NOT NULL, digest TEXT NOT NULL, log TEXT NOT NULL,
                UNIQUE(format, digest)
            );
            CREATE TABLE IF NOT EXISTS scout_samples (
                id TEXT PRIMARY KEY, format TEXT NOT NULL, species TEXT NOT NULL,
                move TEXT NOT NULL, vector TEXT NOT NULL, battle TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS team_metadata (
                id TEXT PRIMARY KEY, data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feed_state (
                id TEXT PRIMARY KEY, checked TEXT NOT NULL, result TEXT NOT NULL
            );
        """)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(scout_samples)")}
        if "battle" not in columns:
            with self.db:
                self.db.execute("ALTER TABLE scout_samples ADD COLUMN battle TEXT NOT NULL DEFAULT ''")

    def close(self):
        self.db.close()

    def retain_checkpoint(self, model) -> str:
        """Keep the collecting artifact once by digest, even after promotion.

Only local model artifacts are retained; no replay labels or private requests
are reconstructed. Atomic creation also tolerates concurrent local collectors.
"""
        target = self.root / 'models' / 'collected' / model.fmt / (model.checkpoint_sha256 + '.pt')
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != model.checkpoint_sha256:
            raise ValueError('retained collecting checkpoint digest mismatch')
        if not target.exists():
            payload = model.path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != model.checkpoint_sha256:
                raise ValueError('collecting checkpoint changed before archival; reload between games')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return str(target.relative_to(self.root))

    @contextmanager
    def writer(self):
        """Only one learning process can mutate a root at a time (Mac/Linux)."""
        with (self.root / "writer.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WriterBusy("another learning process is writing this data root; use a separate --root") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def save_episode(self, episode: dict):
        # Finalization is idempotent. Never overwrite a completed game on retry.
        existing = self.db.execute("SELECT status FROM episodes WHERE id=?", (episode["id"],)).fetchone()
        if existing and existing["status"] == "complete":
            return
        with self.db:
            self.db.execute("""INSERT OR REPLACE INTO episodes
                (id, format, team, source, revision, created, outcome, status, trained, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (episode["id"], episode["format"], episode.get("team", ""), episode["source"],
                 episode["revision"], episode.get("created", now()), episode.get("outcome"),
                 episode.get("status", "pending"), json.dumps(episode)))

    def episodes(self, fmt: str, revision: str | None = None, untrained: bool = True) -> list[dict]:
        query = "SELECT data FROM episodes WHERE format=? AND status='complete'"
        args = [fmt]
        if untrained:
            query += " AND trained=0"
        if revision is not None:
            query += " AND revision=?"
            args.append(revision)
        return [json.loads(r[0]) for r in self.db.execute(query + " ORDER BY created", args)]

    def room_episodes(self, room: str, side: str | None = None) -> list[dict]:
        """Recorded sessions for an exact room/perspective, including terminal ones."""
        query = "SELECT data FROM episodes WHERE json_extract(data,'$.room')=?"
        args = [room]
        if side is not None:
            query += " AND json_extract(data,'$.side')=?"
            args.append(side)
        return [json.loads(row[0]) for row in self.db.execute(query + " ORDER BY created", args)]

    def mark_trained(self, ids: list[str]):
        with self.db:
            self.db.executemany("UPDATE episodes SET trained=1 WHERE id=?", [(i,) for i in ids])

    def stats(self) -> dict:
        rows = self.db.execute("SELECT format, source, status, COUNT(*) n, AVG(outcome) score FROM episodes GROUP BY format, source, status")
        return {"episodes": [dict(r) for r in rows], "documents": self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "research_teams": self.db.execute("SELECT COUNT(*) FROM research_teams").fetchone()[0],
                "public_battles": self.db.execute("SELECT COUNT(*) FROM public_battles").fetchone()[0],
                "scout_samples": self.db.execute("SELECT COUNT(*) FROM scout_samples").fetchone()[0],
                "jobs": [dict(r) for r in self.db.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status")]}

    def enqueue(self, key: str, kind: str, fmt: str, payload: dict | None = None):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO jobs (id,kind,format,created,updated,payload) VALUES (?,?,?,?,?,?)",
                            (key, kind, fmt, now(), now(), json.dumps(payload or {})))
        return key

    def claim(self, lease_seconds: int = 3600, kinds: tuple[str, ...] | None = None):
        import time
        if kinds is not None and not kinds:
            return None
        self.db.execute("BEGIN IMMEDIATE")
        try:
            filter_sql = " AND kind IN (" + ",".join("?" for _ in kinds) + ")" if kinds is not None else ""
            row = self.db.execute("SELECT * FROM jobs WHERE (status='queued' OR (status='running' AND lease_until<?)) AND attempts<3" + filter_sql + " ORDER BY created LIMIT 1", (time.time(), *(kinds or ()))).fetchone()
            if row:
                self.db.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated=?,lease_until=? WHERE id=?",
                                (now(), time.time() + lease_seconds, row["id"]))
            self.db.commit()
            return {**dict(row), "payload": json.loads(row["payload"])} if row else None
        except BaseException:
            self.db.rollback()
            raise

    def complete_job(self, key: str, result: dict, retry: bool = False):
        with self.db:
            self.db.execute("UPDATE jobs SET status=?,result=?,updated=?,lease_until=0,attempts=CASE WHEN ? THEN MAX(0,attempts-1) ELSE attempts END WHERE id=?",
                            ("queued" if retry else "complete", json.dumps(result), now(), retry, key))
