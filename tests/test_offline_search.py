"""Offline checks for the determinized simulator search agent (search/)."""
import asyncio
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from search import priors
from search.agent import Engine, SearchAgent
from search.tracker import Tracker

ROOT = Path(__file__).resolve().parents[1]
FMT = 'gen9championsvgc2026regmc'

LOG = [
    '|player|p1|Me|1', '|player|p2|Foe|2', '|gametype|doubles',
    '|poke|p1|Politoed, L50, M|', '|poke|p1|Archaludon, L50, M|',
    '|poke|p2|Gardevoir, L50, F|', '|poke|p2|Incineroar, L50, M|', '|poke|p2|Rillaboom, L50, M|',
    '|start',
    '|switch|p1a: Politoed|Politoed, L50, M|167/167', '|switch|p1b: Archaludon|Archaludon, L50, M|167/167',
    '|switch|p2a: Gardy|Gardevoir, L50, F|100/100', '|switch|p2b: Incineroar|Incineroar, L50, M|100/100',
    '|-weather|RainDance|[from] ability: Drizzle|[of] p1a: Politoed',
    '|-ability|p2a: Gardy|Drizzle|[from] ability: Trace|[of] p1a: Politoed',
    '|-ability|p2b: Incineroar|Intimidate|boost', '|-unboost|p1a: Politoed|atk|1',
    '|turn|1',
    '|detailschange|p2a: Gardy|Gardevoir-Mega, L50, F',
    '|move|p2b: Incineroar|Fake Out|p1a: Politoed', '|-damage|p1a: Politoed|150/167',
    '|move|p1b: Archaludon|Protect|p1b: Archaludon',
    '|move|p2a: Gardy|Hyper Voice|p1a: Politoed|[spread] p1a,p1b', '|-damage|p1a: Politoed|90/167',
    '|-sidestart|p2: Foe|move: Tailwind',
    '|-enditem|p2b: Incineroar|Sitrus Berry|[eat]', '|-heal|p2b: Incineroar|60/100|[from] item: Sitrus Berry',
    '|turn|2',
]


def test_tracker_reads_public_state():
    t = Tracker('p1').feed(LOG)
    gardy = t.mons[('p2', 'Gardy')]
    assert gardy.species == 'Gardevoir-Mega' and gardy.mega and t.mega_used['p2']
    # Trace copied Drizzle; the holder's own ability is Trace, not Drizzle.
    assert gardy.ability == 'trace'
    inc = t.mons[('p2', 'Incineroar')]
    assert inc.item == 'sitrusberry' and inc.item_gone and inc.ability == 'intimidate'
    assert inc.hp == pytest.approx(.6) and inc.moves == ['fakeout']
    pol = t.mons[('p1', 'Politoed')]
    assert pol.boosts == {'atk': -1} and pol.hp_exact == 90
    arch = t.mons[('p1', 'Archaludon')]
    assert arch.last_move == 'protect' and arch.last_move_turn == 1
    field = t.field_state()
    assert field['weather']['id'] == 'raindance' and field['weather']['duration'] == 4
    assert field['sides']['p2']['tailwind']['duration'] == 3
    assert t.preview['p2'] == ['Gardevoir', 'Incineroar', 'Rillaboom']


def test_locked_request_maps_to_move_id():
    t = Tracker('p1').feed(LOG)
    # A locked request lists only the locked move; index 1 is not the set's first move.
    active = [{'moves': [{'move': 'Dragon Pulse', 'id': 'dragonpulse'}]},
              {'moves': [{'move': 'Muddy Water', 'id': 'muddywater', 'target': 'allAdjacentFoes'},
                         {'move': 'Ice Beam', 'id': 'icebeam', 'target': 'normal'}]}]
    sim = SearchAgent._to_sim('move 1, move 2 2', active, t)
    assert sim == 'move dragonpulse 1, move icebeam 2'


def _fake_usage(tmp_path, monkeypatch):
    (tmp_path / 'cache' / 'usage').mkdir(parents=True)
    (tmp_path / 'cache' / 'dex' / FMT).mkdir(parents=True)
    dex = {'salamence': {'name': 'Salamence', 'abilities': {'0': 'Intimidate', '1': 'Moxie'}},
           'salamencemega': {'name': 'Salamence-Mega', 'baseSpecies': 'Salamence', 'requiredItem': 'Salamencite',
                             'abilities': {'0': 'Aerilate'}}}
    (tmp_path / 'cache' / 'dex' / FMT / 'pokedex.json').write_text(json.dumps(dex))
    data = {'data': {
        'Salamence': {'Raw count': 10, 'Moves': {'tailwind': 5, 'protect': 5, 'dracometeor': 4, 'flamethrower': 2},
                      'Items': {'lifeorb': 1}, 'Abilities': {'intimidate': 1}, 'Spreads': {'Timid:2/0/0/32/0/32': 1},
                      'Teammates': {}},
        'Salamence-Mega': {'Raw count': 90, 'Moves': {'hypervoice': 9, 'protect': 8, 'tailwind': 6, 'doubleedge': 3},
                           'Items': {'salamencite': 1}, 'Abilities': {'aerilate': 1},
                           'Spreads': {'Timid:2/0/0/32/0/32': 1}, 'Teammates': {}}}}
    (tmp_path / 'cache' / 'usage' / f'{FMT}-0.json').write_text(json.dumps(data))
    monkeypatch.setattr(priors, 'ROOT', tmp_path)
    monkeypatch.setattr(priors, 'STATS', tmp_path / 'cache' / 'usage')
    priors.load_usage.cache_clear()


def test_priors_respect_observations(tmp_path, monkeypatch):
    _fake_usage(tmp_path, monkeypatch)
    rng = random.Random(1)
    sets = [priors.sample_set(rng, 'Salamence') for _ in range(200)]
    megas = [s for s in sets if s['item'] == 'salamencite']
    assert 150 < len(megas) < 200
    # Mega sets carry the base forme's ability; Mega Evolution swaps it in battle.
    assert all(s['ability'] == 'intimidate' for s in megas)
    revealed = priors.sample_set(rng, 'Salamence', {'item': 'lifeorb', 'moves': ['flamethrower']})
    assert revealed['item'] == 'lifeorb' and 'flamethrower' in revealed['moves'] and len(revealed['moves']) == 4
    priors.load_usage.cache_clear()


SET = lambda species, item, ability, moves, nature='Modest', evs=None: {
    'species': species, 'item': item, 'ability': ability, 'moves': moves, 'nature': nature, 'level': 50,
    'evs': evs or {'hp': 2, 'spa': 32, 'spe': 32}}


@pytest.mark.skipif(shutil.which('node') is None or not (ROOT / 'node_modules' / 'pokemon-showdown' / 'dist').exists(),
                    reason='official simulator is not built')
def test_engine_searches_rebuilt_battle():
    ours = [SET('Politoed', 'Mystic Water', 'Drizzle', ['muddywater', 'icebeam', 'weatherball', 'protect']),
            SET('Archaludon', 'Leftovers', 'Stamina', ['dragonpulse', 'electroshot', 'snarl', 'protect']),
            SET('Incineroar', 'Sitrus Berry', 'Intimidate', ['fakeout', 'flareblitz', 'partingshot', 'darkestlariat'],
                'Adamant', {'hp': 32, 'atk': 32, 'spd': 2}),
            SET('Farigiraf', 'Sitrus Berry', 'Armor Tail', ['psychic', 'trickroom', 'helpinghand', 'protect'])]
    theirs = [SET('Garchomp', 'Life Orb', 'Rough Skin', ['earthquake', 'dragonclaw', 'rockslide', 'protect'],
                  'Jolly', {'hp': 2, 'atk': 32, 'spe': 32}),
              SET('Sylveon', 'Leftovers', 'Pixilate', ['hypervoice', 'protect', 'moonblast', 'yawn']),
              SET('Rillaboom', 'Miracle Seed', 'Grassy Surge', ['fakeout', 'grassyglide', 'woodhammer', 'protect'],
                  'Adamant', {'hp': 32, 'atk': 32, 'spd': 2}),
              SET('Kingambit', 'Black Glasses', 'Defiant', ['kowtowcleave', 'suckerpunch', 'ironhead', 'protect'],
                  'Adamant', {'hp': 32, 'atk': 32, 'spd': 2})]
    world = {'p1': [{'set': s, 'state': {'hp_abs': None, 'turn': 3}} for s in ours],
             'p2': [{'set': s, 'state': {'hp': .5 if i == 0 else 1.0, 'turn': 3}} for i, s in enumerate(theirs)]}
    world['p1'][0]['state'].update(status='par', boosts={'spa': 1})
    field = {'turn': 3, 'weather': {'id': 'raindance', 'duration': 2}, 'terrain': None, 'pseudo': {'trickroom': 3},
             'sides': {'p1': {'tailwind': {'duration': 2}}, 'p2': {}}, 'mega_used': {'p1': False, 'p2': False}}
    proc = subprocess.run(['node', str(ROOT / 'search' / 'engine.cjs')], capture_output=True, text=True, timeout=120,
                          input=json.dumps({'type': 'debug', 'id': 0, 'format': FMT, 'worlds': [world], 'field': field}) + '\n' +
                          json.dumps({'type': 'search', 'id': 1, 'format': FMT, 'worlds': [world], 'field': field,
                                      'our_choices': ['move muddywater, move dragonpulse 1', 'move protect, move protect',
                                                      'move icebeam 1, move dragonpulse 1'],
                                      'screen': {'probe': 3, 'keep_opp': 8, 'keep_ours': 8}}) + '\n')
    debug, result = [json.loads(line) for line in proc.stdout.splitlines()]
    pol = debug['p1'][0]
    assert pol['status'] == 'par' and pol['boosts']['spa'] == 1 and pol['active']
    assert abs(debug['p2'][0]['hp'] - debug['p2'][0]['maxhp'] * .5) <= .5
    assert debug['field']['weather'] == 'raindance' and debug['field']['wd'] == 2
    assert 'trickroom' in debug['field']['pseudo'] and debug['field']['sides'][0] == ['tailwind']
    values = result['results'][0]['values']
    assert set(values) == {'move muddywater, move dragonpulse 1', 'move protect, move protect',
                           'move icebeam 1, move dragonpulse 1'}
    import math
    assert all(math.isfinite(v) for v in values.values()) and len(set(values.values())) > 1
    r = result['results'][0]
    assert r['sims'] > 0 and r['n_theirs'] >= 1 and r['top_opp']
