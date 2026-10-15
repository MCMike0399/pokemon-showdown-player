"""Offline checks for speed inference (search/speed.py), the win-probability leaf and the
Protect second turn (search/engine.cjs), and the cleared decision trace (search/agent.py)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from search.speed import effective_speed, observations, raw_speed, violations

ROOT = Path(__file__).resolve().parents[1]
OURS = [
    {'species': 'Garchomp', 'item': 'Garchompite Z', 'ability': 'Rough Skin', 'nature': 'Modest',
     'moves': ['Dragon Pulse', 'Earth Power', 'Power Gem', 'Protect'], 'level': 50, 'evs': {'spa': 32, 'spe': 32, 'hp': 2}},
    {'species': 'Archaludon', 'item': 'Leftovers', 'ability': 'Stamina', 'nature': 'Timid',
     'moves': ['Dragon Pulse', 'Electro Shot', 'Snarl', 'Protect'], 'level': 50, 'evs': {'spa': 32, 'spe': 32, 'hp': 2}},
]
HEAD = ['|player|p1|Me|1', '|player|p2|Foe|2', '|gametype|doubles', '|start',
        '|switch|p1a: Garchomp|Garchomp, L50, M|185/185', '|switch|p1b: Archaludon|Archaludon, L50, F|167/167',
        '|switch|p2a: Garchomp|Garchomp, L50, M|100/100', '|switch|p2b: Incineroar|Incineroar, L50, M|100/100']


def test_champions_speed_formula_matches_live_request():
    # Live requests: Timid 32-point Archaludon 150, Modest 32-point Garchomp 154.
    assert raw_speed('Archaludon', 'Timid', {'spe': 32}) == 150
    assert raw_speed('Garchomp', 'Modest', {'spe': 32}) == 154
    assert raw_speed('Garchomp', 'Jolly', {'spe': 32}) == 169
    ctx = {'boost': 1, 'tailwind': True, 'status': 'par'}
    assert effective_speed('Archaludon', 'Timid', {'spe': 32}, 'choicescarf', '', ctx) == pytest.approx(150 * 1.5 * 2 * 1.5 * 0.5)
    rain = {'weather': 'raindance'}
    assert effective_speed('Archaludon', 'Timid', {'spe': 32}, '', 'swiftswim', rain) == 300


def test_observations_bound_opposing_speed_and_keep_same_names_apart():
    # Our Garchomp (154) moves before theirs: their Garchomp is at most 154. Same nickname on
    # both sides must not share a snapshot.
    log = HEAD + ['|turn|1', '|move|p1a: Garchomp|Dragon Pulse|p2b: Incineroar',
                  '|move|p2a: Garchomp|Dragon Claw|p1b: Archaludon', '|turn|2']
    obs = observations(log, 'p1', OURS)
    cons = obs['Garchomp']
    assert len(cons) == 1 and cons[0]['bound'] == 154 and cons[0]['ge'] is False
    assert violations({'nature': 'Jolly', 'evs': {'spe': 32}, 'item': '', 'ability': 'roughskin'}, cons) == 1
    assert violations({'nature': 'Adamant', 'evs': {'spe': 0}, 'item': '', 'ability': 'roughskin'}, cons) == 0
    # Scarf: 122 x 1.5 = 183 would have outsped our 154.
    assert violations({'nature': 'Adamant', 'evs': {'spe': 0}, 'item': 'choicescarf', 'ability': 'roughskin'}, cons) == 1
    assert violations({'nature': 'Jolly', 'evs': {'spe': 32}, 'item': 'choicescarf', 'ability': 'roughskin'}, cons) == 1


def test_trick_room_inverts_and_speed_events_and_quick_claw_stop_inference():
    tr = HEAD + ['|-fieldstart|move: Trick Room|[of] p2b: Incineroar', '|turn|1',
                 '|move|p1a: Garchomp|Dragon Pulse|p2b: Incineroar', '|move|p2a: Garchomp|Dragon Claw|p1b: Archaludon']
    assert observations(tr, 'p1', OURS)['Garchomp'][0]['ge'] is True
    boosted = HEAD + ['|turn|1', '|move|p1a: Garchomp|Dragon Pulse|p2b: Incineroar',
                      '|-unboost|p2a: Garchomp|spe|1', '|move|p1b: Archaludon|Snarl|p2a: Garchomp',
                      '|move|p2a: Garchomp|Dragon Claw|p1b: Archaludon']
    # Pairs whose earlier move precedes the Speed change still count; the later ones do not.
    assert len(observations(boosted, 'p1', OURS)['Garchomp']) == 1
    claw = HEAD + ['|turn|1', '|-activate|p2a: Garchomp|item: Quick Claw',
                   '|move|p2a: Garchomp|Dragon Claw|p1b: Archaludon', '|move|p1a: Garchomp|Dragon Pulse|p2b: Incineroar']
    assert observations(claw, 'p1', OURS) == {}
    # Different priority (Fake Out +3) says nothing about Speed.
    fake = HEAD + ['|turn|1', '|move|p2b: Incineroar|Fake Out|p1a: Garchomp', '|move|p1a: Garchomp|Dragon Pulse|p2b: Incineroar']
    assert 'Incineroar' not in observations(fake, 'p1', OURS)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_engine_leaf_and_protect_helpers():
    script = """
const E = require('./search/engine.cjs');
const src = require('fs').readFileSync('./search/engine.cjs', 'utf8');
console.log(JSON.stringify({leaf: /setLeaf\\(msg\\.leaf\\)/.test(src), protect: /protect_d2/.test(src)}));
"""
    out = subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, check=True)
    flags = json.loads(out.stdout)
    assert flags == {'leaf': True, 'protect': True}


def test_preview_decision_clears_previous_trace():
    import asyncio

    from search.agent import SearchAgent

    async def run():
        agent = SearchAgent(worlds=1)
        agent.last_trace = {'kind': 'move', 'turn': 9}
        choice = await agent.decide({'request': {'teamPreview': True, 'side': {'id': 'p1', 'pokemon': []}},
                                     'choices': ['team 1234', 'team 1243']})
        await agent.close()
        return choice, agent.last_trace

    choice, trace = asyncio.run(run())
    assert choice == 'team 1234' and trace is None


def test_tracker_follows_ally_switch():
    from search.tracker import Tracker
    log = HEAD + ['|turn|1', '|move|p2b: Incineroar|Ally Switch|p2b: Incineroar',
                  '|swap|p2b: Incineroar|0|[from] move: Ally Switch',
                  '|move|p2b: Garchomp|Dragon Claw|p1b: Archaludon', '|turn|2']
    t = Tracker('p1').feed(log)
    assert t.slots['p2a'].name == 'Incineroar' and t.slots['p2b'].name == 'Garchomp'
    assert t.mons[('p2', 'Garchomp')].active_slot == 'p2b' and t.mons[('p2', 'Garchomp')].last_move == 'dragonclaw'
    assert t.mons[('p2', 'Incineroar')].last_move == 'allyswitch'
