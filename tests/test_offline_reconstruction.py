"""Regressions from live and c10 invalid-action fallbacks."""
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from search.agent import SearchAgent
from search.tracker import Tracker

ROOT = Path(__file__).resolve().parents[1]
FMT = 'gen9championsvgc2026regmc'
SIM = shutil.which('node') and (ROOT / 'node_modules/pokemon-showdown/dist').exists()


def mon(species, moves):
    return {'set': {'species': species, 'moves': moves, 'level': 50}, 'state': {}}


def engine(message):
    proc = subprocess.run(['node', str(ROOT / 'search/engine.cjs')],
                          input=json.dumps(message) + '\n', text=True, capture_output=True,
                          check=True, timeout=30)
    return json.loads(proc.stdout)


ILLUSION = [
    '|poke|p2|Alcremie, L50|', '|poke|p2|Blaziken, L50|',
    '|poke|p2|Farigiraf, L50|', '|poke|p2|Zoroark, L50|',
    '|switch|p2a: Alcremie|Alcremie, L50|100/100',
    '|switch|p2b: Blaziken|Blaziken, L50|100/100', '|turn|1',
    '|move|p2b: Blaziken|Night Daze|p1a: Politoed',
    '|-damage|p2b: Blaziken|0 fnt',
    # Real Showdown replace messages need not contain an HP field.
    '|replace|p2b: Zoroark|Zoroark, L50', '|faint|p2b: Zoroark',
    '|switch|p2b: Blaziken|Blaziken, L50|100/100', '|turn|2',
    '|faint|p2a: Alcremie', '|switch|p2a: Farigiraf|Farigiraf, L50|100/100',
    '|turn|3', '|faint|p2a: Farigiraf', '|turn|4',
]


def test_illusion_reveal_keeps_the_real_survivor_and_observations():
    tracker = Tracker('p1').feed(ILLUSION)
    survivor = tracker.slots['p2b']
    assert not survivor.fainted and survivor.hp == 1
    assert 'nightdaze' not in survivor.moves
    zoroark = tracker.mons[('p2', 'Zoroark')]
    assert zoroark.fainted and 'nightdaze' in zoroark.moves
    assert len(tracker.side_mons('p2')) == 4


@pytest.mark.skipif(not SIM, reason='official simulator is not built')
def test_illusion_world_search_does_not_fall_back_to_pass_pass(monkeypatch):
    import search.agent as module
    monkeypatch.setattr(module, 'sample_set', lambda rng, species, *a, **kw:
                        mon(species, ['tackle'])['set'])
    monkeypatch.setattr(module, 'sample_bench', lambda rng, preview, revealed, n, *a, **kw:
                        [species for species in preview if species not in revealed][:n])
    agent = SearchAgent(worlds=1, seed=1)
    world = {'p1': [mon('Politoed', ['tackle']), mon('Archaludon', ['tackle'])],
             'p2': agent._their_side(Tracker('p1').feed(ILLUSION), random.Random(1))}
    out = engine({'type': 'search', 'format': FMT, 'worlds': [world], 'field': {'turn': 4},
                  'screen': {'probe': 1, 'keep_ours': 2, 'keep_opp': 2}})['results'][0]
    assert 'error' not in out, out
    assert out['values'] and out['sims'] > 0


def test_recharge_tracks_consumption_and_switch_out():
    log = ['|switch|p2a: Sylveon|Sylveon, L50|100/100', '|turn|1',
           '|move|p2a: Sylveon|Hyper Beam|p1a: Politoed', '|-mustrecharge|p2a: Sylveon', '|turn|2']
    tracker = Tracker('p1').feed(log)
    assert 'mustrecharge' in tracker.slots['p2a'].volatiles
    tracker.line('|cant|p2a: Sylveon|recharge')
    assert 'mustrecharge' not in tracker.slots['p2a'].volatiles
    tracker.feed(['|-mustrecharge|p2a: Sylveon', '|switch|p2a: Politoed|Politoed, L50|100/100'])
    assert 'mustrecharge' not in tracker.mons[('p2', 'Sylveon')].volatiles


@pytest.mark.skipif(not SIM, reason='official simulator is not built')
def test_private_recharge_request_rebuilds_real_recharge_turn():
    # Private request is authoritative even after reconnect without the public event.
    sets = [mon('Sylveon', ['hyperbeam', 'hypervoice'])['set'], mon('Politoed', ['tackle'])['set']]
    request = {'side': {'id': 'p1', 'pokemon': [
        {'ident': 'p1: ' + s['species'], 'details': s['species'] + ', L50',
         'condition': '100/100', 'active': True, 'moves': s['moves']} for s in sets]},
        'active': [{'moves': [{'id': 'recharge', 'move': 'Recharge'}]},
                   {'moves': [{'id': 'tackle', 'move': 'Tackle'}]}]}
    ours = SearchAgent(worlds=1)._our_side({'request': request, 'team_sets': sets}, Tracker('p1'))
    world = {'p1': ours, 'p2': [mon('Garchomp', ['tackle']), mon('Archaludon', ['tackle'])]}
    choice = SearchAgent._to_sim('move 1, move 1 1', request['active'], Tracker('p1'))
    result = engine({'type': 'search', 'format': FMT, 'worlds': [world], 'field': {'turn': 2},
                     'our_choices': [choice],
                     'screen': {'probe': 1, 'keep_ours': 2, 'keep_opp': 2}})['results'][0]
    assert 'error' not in result, result
    assert choice in result['values']


@pytest.mark.parametrize('hp', ['', '|35/100'])
def test_illusion_preserves_prior_disguise_facts_and_current_visitor_state(hp):
    tracker = Tracker('p1').feed([
        '|switch|p2a: Blaziken|Blaziken, L50|100/100',
        '|move|p2a: Blaziken|Blaze Kick|p1a: Politoed',
        '|-damage|p2a: Blaziken|80/100',
        '|switch|p2a: Sylveon|Sylveon, L50|100/100',
        '|switch|p2a: Blaziken|Blaziken, L50|100/100',
        '|move|p2a: Blaziken|Night Daze|p1a: Politoed',
        '|-damage|p2a: Blaziken|35/100',
        '|-boost|p2a: Blaziken|spa|2',
        '|replace|p2a: Zoroark|Zoroark, L50' + hp,
    ])
    real = tracker.mons[('p2', 'Blaziken')]
    visitor = tracker.slots['p2a']
    assert real.hp == .8 and real.moves == ['blazekick'] and not real.active_slot
    assert visitor.species == 'Zoroark' and visitor.hp == .35
    assert visitor.moves == ['nightdaze'] and visitor.boosts == {'spa': 2}
