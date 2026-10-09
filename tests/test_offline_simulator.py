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




def test_hidden_trap_retries_corrected_request_without_credit(monkeypatch, tmp_path):
    from test_offline_learning import context
    from ml.features import Features
    request = context()['request']
    request['side']['pokemon'].append({'ident': 'p1: Swampert', 'details': 'Swampert, L50',
                                       'condition': '100/100', 'active': False})
    corrected = json.loads(json.dumps(request))
    corrected['rqid'] = 2
    corrected['active'][0]['trapped'] = True
    events = [{'side': 'p1', 'lines': [], 'request': request},
              {'side': 'p1', 'lines': ["|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"],
               'request': corrected},
              {'side': 'p1', 'lines': ['|win|LocalBrain']}]
    sent = []
    class Pipe:
        def write(self, data):
            sent.append(json.loads(data))
        async def drain(self):
            pass
        async def readline(self):
            return (json.dumps(events.pop(0)) + '\n').encode()
    class Process:
        stdin = Pipe()
        stdout = Pipe()
        returncode = None
        def terminate(self):
            self.returncode = 0
        async def wait(self):
            return 0
    async def spawn(*args, **kwargs):
        return Process()
    monkeypatch.setattr('ml.simulator.asyncio.create_subprocess_exec', spawn)
    store = Store(tmp_path)
    brain = Brain(store, Features())
    original = brain.decide
    def decide(ctx, **kwargs):
        choice = 'switch 2' if 'switch 2' in ctx['choices'] else ctx['choices'][0]
        return original(ctx, demonstration=choice)
    brain.decide = decide
    result = asyncio.run(play_local(brain, FMT, seed=12))
    assert result['winner'] == 'LocalBrain'
    assert result['unavailable_choices'] == 1
    assert result['rejected_actions'] == 0
    steps = store.episodes(FMT)[0]['steps']
    assert len(steps) == 1 and steps[0]['choice'].startswith('move ')
    assert [c['choice'] for c in sent if c['type'] == 'choose'][0] == 'switch 2'
    store.close()
