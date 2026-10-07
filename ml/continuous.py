"""Configuration and durable handoff from live battle decisions to background learning."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ml.resources import ResourcePolicy
from ml.storage import Store, DEFAULT_ROOT, now


@dataclass
class LearningConfig:
    enabled: bool = True
    formats: list[str] = field(default_factory=lambda: ["gen9championsvgc2026regmc"])
    feed_daily: bool = True
    simulation_games: int = 24
    evaluation_games: int = 20
    promotion_margin: float = 0.10
    max_seconds: int = 360
    max_replays: int = 24
    max_tournament_teams: int = 12
    resource: dict = field(default_factory=lambda: asdict(ResourcePolicy()))
    archives: list[str] = field(default_factory=list)

    def __post_init__(self):
        from battle_state import to_id
        if not all(f and f == to_id(f) for f in self.formats):
            raise ValueError("all formats must be Showdown format ids")
        if not 0 <= self.simulation_games <= 256 or not 4 <= self.evaluation_games <= 100:
            raise ValueError("simulation games 0..256 and evaluation games 4..100")
        if not 0.05 <= self.promotion_margin <= 0.5 or not 10 <= self.max_seconds <= 1800:
            raise ValueError("promotion margin .05..5 and run budget 10..1800 seconds")
        if not 0 <= self.max_replays <= 100 or not 1 <= self.max_tournament_teams <= 30:
            raise ValueError("replays 0..100 and tournament teams 1..30")
        ResourcePolicy(**self.resource)
        from ml.feeds import ARCHIVES
        if any(name not in ARCHIVES for name in self.archives):
            raise ValueError("archive must be an approved pinned source")

    @classmethod
    def load(cls, root: Path = DEFAULT_ROOT):
        path = Path(root)/"autopilot.json"
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, root: Path):
        path = Path(root)/"autopilot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        path.chmod(0o600)


def enqueue_battle(store: Store, fmt: str, episode_id: str):
    if LearningConfig.load(store.root).enabled:
        return store.enqueue("battle-"+episode_id, "learn", fmt, {"trigger": "completed-battle"})
    return None


def queue_daily(store: Store, config: LearningConfig):
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not config.enabled:
        return
    for fmt in config.formats:
        if config.feed_daily:
            store.enqueue(f"feed-{day}-{fmt}", "feed", fmt)
        clock = datetime.now(timezone.utc)
        store.enqueue(f"practice-{day}-{clock.hour}-{clock.minute//15}-{fmt}", "practice", fmt)


def kick_worker(root: Path):
    """Nonblocking, one bounded process; OS lock prevents duplicate workers."""
    if os.environ.get("PS_DISABLE_WORKER_KICK") == "1":
        return
    log_dir = Path(root)/"logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir/"worker.log").open("a") as log:
        subprocess.Popen([sys.executable, "-m", "ml.worker", "--root", str(root), "--once"],
                         cwd=str(Path(__file__).resolve().parents[1]), stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)


class BattleObserver:
    """Finalizes completed games even when OpenClaw forgets a separate finish call."""
    def __init__(self, session, lock: asyncio.Lock):
        self.session, self.lock = session, lock
        self.task = None

    def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())

    async def run(self):
        while True:
            await asyncio.sleep(1)
            async with self.lock:
                if self.session.watch_room:
                    self.session.publish_watch(self.session.watch_room)
                rooms = {room for room, _ in self.session.brain.pending}
                from battle_state import to_id
                user = to_id(self.session.player.c.user or "")
                if user:
                    for room in self.session.player.battles():
                        names = self.session.player.c.battle_summary(room).get("players", {}).values()
                        if user in {to_id(name) for name in names}:
                            rooms.add(room)
                for room in rooms:
                    if self.session.player.finished(room) and room not in self.session.finished_rooms:
                        self.session.finish(room)
