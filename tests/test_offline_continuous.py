import asyncio
import json
from types import SimpleNamespace

import pytest

from ml.continuous import LearningConfig, queue_daily
from ml.feeds import parse_limitless
from ml.resources import ResourcePolicy
from ml.scout import ingest_public, public_lines, Scout
from ml.storage import Store
from ml.features import Features
from ml.teams import TeamPlanner
from harness import TeamStore

FMT = "gen9championsvgc2026regmc"


def test_guard_defers_under_cpu_or_memory_pressure(monkeypatch):
    monkeypatch.setattr("ml.resources.psutil.cpu_percent", lambda interval: 95)
    monkeypatch.setattr("ml.resources.psutil.virtual_memory", lambda: SimpleNamespace(available=8*2**30))
    assert ResourcePolicy().sample()["deferred"]
    monkeypatch.setattr("ml.resources.psutil.cpu_percent", lambda interval: 10)
    monkeypatch.setattr("ml.resources.psutil.virtual_memory", lambda: SimpleNamespace(available=1*2**30))
    assert ResourcePolicy().sample(resident_workers=4)["deferred"]
    monkeypatch.setattr("ml.resources.psutil.virtual_memory", lambda: SimpleNamespace(available=8*2**30))
    result = ResourcePolicy().sample()
    assert 1 <= result["allowed_workers"] <= 4


def test_durable_jobs_dedupe_and_recover_leases(tmp_path):
    store = Store(tmp_path)
    store.enqueue("same", "learn", FMT)
    store.enqueue("same", "learn", FMT)
    assert store.claim()["id"] == "same"
    assert store.claim() is None
    with store.db:
        store.db.execute("UPDATE jobs SET lease_until=0 WHERE id='same'")
    assert store.claim()["id"] == "same"
    store.complete_job("same", {"promoted": False})
    assert store.claim() is None
    queue_daily(store, LearningConfig())
    first = store.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    queue_daily(store, LearningConfig())
    assert store.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == first
    store.close()


def test_public_scout_uses_pre_turn_state_and_no_private_lines(tmp_path):
    features = Features({"moves": {"protect": {}, "hurricane": {}}})
    store = Store(tmp_path)
    prefix = ["|switch|p1a: One|Pelipper, L50|100/100", "|switch|p2a: Two|Pelipper, L50|100/100", "|turn|1"]
    log = prefix + ["|request|secret private request", "|c|User|private chat", "|move|p1a: One|Hurricane|p2a: Two",
                    "|-damage|p2a: Two|10/100", "|move|p2a: Two|Protect|p2a: Two", "|win|PrivateUser"]
    result = ingest_public(store, FMT, log, "test", "one", features)
    assert result["samples"] == 2
    rows = list(store.db.execute("SELECT move, vector FROM scout_samples"))
    vectors = {r["move"]: json.loads(r["vector"]) for r in rows}
    # Changing a reveal after the turn began must not change either target's input.
    other = Store(tmp_path/"other")
    changed = [l.replace("10/100", "90/100") for l in log]
    ingest_public(other, FMT, changed, "test", "two", features)
    assert {r["move"]:json.loads(r["vector"]) for r in other.db.execute("SELECT move,vector FROM scout_samples")} == vectors
    stored = store.db.execute("SELECT log FROM public_battles").fetchone()[0]
    assert "private" not in stored and "PrivateUser" not in stored
    assert not ingest_public(store, FMT, log, "test", "duplicate", features)["imported"]
    store.close()
    other.close()


def test_public_scout_ignores_called_moves_and_ambiguous_switches(tmp_path):
    features = Features({"moves": {"protect": {}}})
    store = Store(tmp_path)
    log = ["|switch|p2a: Old|Pelipper|100/100", "|turn|1", "|switch|p2a: New|Swampert|100/100",
           "|move|p2a: New|Protect|p2a: New", "|move|p2a: Old|Protect|p2a: Old|[from] move: Sleep Talk", "|tie|"]
    assert ingest_public(store, FMT, log, "test", "one", features)["samples"] == 0
    store.close()


def test_limitless_feed_attests_format_and_missing_spreads():
    mons = "".join('<div class="pkmn"><div class="name">Pelipper</div><div class="details"><div class="item">Focus Sash</div></div><div class="ability">Ability: Drizzle</div><div class="nature">Modest Nature</div><ul class="moves"><li>Protect</li></ul></div>' for _ in range(6))
    html = ('<title>Event</title>Regulation Set M-C<div class="tournament-decklist"><div class="teamlist-toggle">1st Trainer</div>'+mons+'</div>').encode()
    teams = parse_limitless(html, FMT)
    assert teams[0]["player"] == "1st Trainer"
    assert teams[0]["metadata"]["stat_points"] == "not_disclosed"
    assert "evs" not in teams[0]["sets"][0]
    with pytest.raises(ValueError, match="regulation"):
        parse_limitless(html, "gen9championsvgc2026regmb")


def test_team_plan_materializes_validated_hypothesis_without_overwrite(tmp_path, monkeypatch):
    store = Store(tmp_path)
    teams = TeamStore(tmp_path/"teams.json")
    teams.create("Base", [{"species": "Pelipper", "moves": ["Protect"], "nature": "Modest"}], FMT)
    async def validate(fmt, sets):
        return {"errors": [], "packed": "packed"}
    monkeypatch.setattr("ml.simulator.validate_team", validate)
    result = asyncio.run(TeamPlanner(store, teams).plan(FMT, "Base", save_as="Planned"))
    assert result["selected"]["stat_points"] == "generated_hypothesis"
    assert sum(result["selected"]["sets"][0]["evs"].values()) == 66
    assert "evs" not in teams.get("Base")["sets"][0]
    assert "evs" in teams.get("Planned")["sets"][0]
    store.close()
