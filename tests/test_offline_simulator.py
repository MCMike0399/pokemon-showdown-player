import asyncio
import json
from pathlib import Path

import pytest

from ml.brain import Brain
from ml.simulator import BRIDGE, load_dex, play_local, validate_team
from ml.storage import Store

FMT = "gen9championsvgc2026regmc"
BUILT = BRIDGE.parent / "node_modules/pokemon-showdown/dist/sim/index.js"
pytestmark = pytest.mark.skipif(not BUILT.exists(), reason="run npm install && npm run simulator:build")


def fixture_team():
    return json.loads((BRIDGE.parent / "examples/champions-rain.json").read_text())["sets"]


def test_champions_validation_and_packing_match_official_simulator():
    team = [dict(mon, level=50) for mon in fixture_team()]
    result = asyncio.run(validate_team(FMT, team))
    assert result["errors"] == []
    from harness import pack_team
    # Showdown packName preserves capitalization; our identifiers are lowercase.
    assert pack_team(team).lower() == result["packed"].lower()
    invalid = [dict(mon) for mon in team]
    invalid[0]["item"] = "Assault Vest"
    assert asyncio.run(validate_team(FMT, invalid))["errors"]


def test_real_champions_game_records_ppo_and_reload(tmp_path):
    async def run():
        features = await load_dex(FMT)
        store = Store(tmp_path)
        brain = Brain(store, features)
        team = fixture_team()
        result = await play_local(brain, FMT, team, team, seed=4)
        assert result.get("winner") or result.get("tie")
        assert result["rejected_actions"] == 0
        episodes = store.episodes(FMT)
        assert len(episodes) == 1 and len(episodes[0]["steps"]) > 1
        update = brain.train(FMT)
        assert update["trained"]
        report = await play_local(brain, FMT, team, team, seed=5, training=False, learner_side="p2")
        assert report["rejected_actions"] == 0
        assert len(store.episodes(FMT, untrained=False)) == 1
        store.close()
    asyncio.run(run())
