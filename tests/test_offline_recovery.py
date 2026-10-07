"""Exercise real recording/submission objects across a simulated process restart."""
import asyncio
import json

import pytest

from harness import Player, TeamStore
from ml.brain import Brain
from ml.features import Features
from ml.live import LiveSession
from ml.model import Model
from ml.storage import Store
from ps_client import PSClient

FMT = "gen9championsvgc2026regmc"
ROOM = "battle-" + FMT + "-123"


def request(rqid, side="p1"):
    return {"rqid": rqid, "side": {"id": side, "pokemon": [
        {"ident": side + ": Pelipper", "details": "Pelipper, L50", "condition": "100/100", "active": True}]},
        "active": [{"moves": [{"move": "Hurricane", "target": "normal", "pp": 10},
                              {"move": "Protect", "target": "self", "pp": 10}]}]}


@pytest.fixture
def session(tmp_path):
    stores = []
    def new_brain():
        store = Store(tmp_path)
        stores.append(store)
        return Brain(store, Features())
    client = PSClient(username="configured", password="unused")
    client.logged_in, client.user = True, "Player"
    client.battles[ROOM] = {"log": ["|request|" + json.dumps(request(1))]}
    player = Player(client, TeamStore(tmp_path / "teams.json"))
    sent = []
    async def choose(room, choice):
        sent.append((room, choice))
    player.choose = choose
    yield new_brain, player, sent
    for store in stores:
        store.close()


def test_restart_continues_the_original_episode(session):
    make, player, sent = session
    first = make()
    asyncio.run(LiveSession(first, player).choose(ROOM, FMT))
    original = first.pending[(ROOM, "p1")]
    player.c.battles[ROOM]["log"].append("|request|" + json.dumps(request(2)))
    restarted = make()
    asyncio.run(LiveSession(restarted, player).choose(ROOM, FMT))
    restored = restarted.pending[(ROOM, "p1")]
    assert restored["id"] == original["id"]
    assert restored["steps"][0] == original["steps"][0]
    assert [s["request_id"] for s in restored["steps"]] == [1, 2]
    assert restarted.store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1


def test_restart_resubmits_same_request_without_resampling(session):
    make, player, sent = session
    first = make()
    decision = asyncio.run(LiveSession(first, player).choose(ROOM, FMT))
    original = first.pending[(ROOM, "p1")]
    restarted = make()
    recovered = asyncio.run(LiveSession(restarted, player).choose(ROOM, FMT))
    assert recovered["choice"] == decision["choice"]
    assert recovered["recovered"]
    restored = restarted.pending[(ROOM, "p1")]
    assert restored["id"] == original["id"]
    assert restored["steps"] == original["steps"]
    assert len(sent) == 2


def test_terminal_reconnect_finalizes_original_once_without_request(session):
    make, player, sent = session
    first = make()
    # A player can occupy p2, and the terminal server request can be null.
    player.c.battles[ROOM]["log"] = ["|request|" + json.dumps(request(1, "p2"))]
    asyncio.run(LiveSession(first, player).choose(ROOM, FMT))
    episode = first.pending[(ROOM, "p2")]
    player.c.battles[ROOM]["log"].extend(["|win|Player", "|request|null"])
    restarted = make()
    finished = LiveSession(restarted, player).finish(ROOM)
    assert finished["experience"]["id"] == episode["id"]
    assert finished["experience"]["outcome"] == 1
    assert finished["experience"]["steps"] == 1
    again = LiveSession(make(), player).finish(ROOM)
    assert again["experience"]["id"] == episode["id"]
    assert again["experience"]["already_recorded"]
    assert restarted.store.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert restarted.store.db.execute("SELECT COUNT(*) FROM episodes WHERE status='complete'").fetchone()[0] == 1


def test_restart_refuses_changed_collecting_checkpoint(session):
    make, player, sent = session
    first = make()
    asyncio.run(LiveSession(first, player).choose(ROOM, FMT))
    changed = Model(first.store.root, FMT)
    changed.revision = "different-revision"
    changed.save()
    with pytest.raises(ValueError, match="revision|checkpoint"):
        asyncio.run(LiveSession(make(), player).choose(ROOM, FMT))
    assert len(sent) == 1
    assert first.store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1


def test_restart_refuses_ambiguous_pending_segments(session):
    make, player, sent = session
    first = make()
    asyncio.run(LiveSession(first, player).choose(ROOM, FMT))
    original = first.pending[(ROOM, "p1")]
    first.store.save_episode({**original, "id": "other-segment"})
    with pytest.raises(ValueError, match="multiple|ambiguous"):
        asyncio.run(LiveSession(make(), player).choose(ROOM, FMT))
    assert len(sent) == 1


def test_terminal_recovery_excludes_unsent_proposal(session):
    from ml.live import context
    make, player, sent = session
    first = make()
    first.decide(context(player, ROOM, FMT), explore=True, record=True)
    player.c.battles[ROOM]["log"].extend(["|win|Player", "|request|null"])
    restarted = make()
    result = LiveSession(restarted, player).finish(ROOM)
    assert result["experience"]["steps"] == 0
    assert not restarted.train(FMT)["trained"]
    assert not sent


def test_restart_refuses_changed_team(session):
    make, player, sent = session
    first = make()
    asyncio.run(LiveSession(first, player).choose(ROOM, FMT))
    player.teams.create("Different", [{"species": "Pelipper", "moves": ["Protect"]}], FMT)
    with pytest.raises(ValueError, match="team"):
        asyncio.run(LiveSession(make(), player).choose(ROOM, FMT, team="Different"))
    assert len(sent) == 1


def test_reward_uses_own_battle_name_when_ladder_anonymizes_players(session):
    make, player, sent = session
    brain = make()
    player.c.battles[ROOM]["log"] = ["|player|p1|Opponent|1", "|player|p2|AnonymousPlayer|2",
                                   "|request|" + json.dumps(request(1, "p2"))]
    live = LiveSession(brain, player)
    asyncio.run(live.choose(ROOM, FMT))
    player.c.battles[ROOM]["log"].append("|win|AnonymousPlayer")
    assert live.finish(ROOM)["experience"]["outcome"] == 1


def test_watch_relay_excludes_requests_chat_and_auth(session):
    make, player, sent = session
    brain = make()
    player.c.battles[ROOM]["log"] = ["|player|p1|Player|1", "|player|p2|Opponent|2",
        "|request|" + json.dumps(request(1)), "|challstr|fixture", "|c|Player|private chat",
        "|turn|1", "|move|p1a: Pelipper|Protect|p1a: Pelipper"]
    live = LiveSession(brain, player)
    live.publish_watch(ROOM)
    data = json.loads((brain.store.root / "live-watch/current.json").read_text())
    assert data["source"] == "player-relay" and data["live"]
    assert "|player|p1|Your agent|1" in data["log"]
    assert "|turn|1" in data["log"]
    assert not any(line.startswith(("|request|", "|challstr|", "|c|")) for line in data["log"])


def test_play_reports_no_job_when_learning_disabled(session):
    from ml.continuous import LearningConfig
    make, player, sent = session
    brain = make()
    LearningConfig(enabled=False).save(brain.store.root)
    live = LiveSession(brain, player)
    async def run():
        async def send(message):
            pass
        player.c.send = send
        async def choose(room, choice):
            player.c.battles[ROOM]['log'].append('|win|Player')
        player.choose = choose
        return await live.play(ROOM, FMT)
    output = asyncio.run(run())
    assert output['experience']['recorded']
    assert output['background_learning']['job'] is None
    assert output['training']['queued'] is False
    assert output['training']['job'] is None
    assert brain.store.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
