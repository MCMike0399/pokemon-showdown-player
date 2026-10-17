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


@pytest.mark.skipif(shutil.which('node') is None or not (ROOT / 'node_modules' / 'pokemon-showdown' / 'dist').exists(),
                    reason='official simulator is not built')
def test_two_mega_capable_actives_can_both_attack():
    # Both actives hold a stone: combos must Mega exactly one slot. Forcing ' mega' on
    # both left only "one of them switches" combos for such a side.
    ours = [SET('Politoed', 'Mystic Water', 'Drizzle', ['muddywater', 'icebeam', 'weatherball', 'protect']),
            SET('Archaludon', 'Leftovers', 'Stamina', ['dragonpulse', 'electroshot', 'snarl', 'protect']),
            SET('Incineroar', 'Sitrus Berry', 'Intimidate', ['fakeout', 'flareblitz', 'partingshot', 'darkestlariat']),
            SET('Farigiraf', 'Sitrus Berry', 'Armor Tail', ['psychic', 'trickroom', 'helpinghand', 'protect'])]
    theirs = [SET('Charizard', 'Charizardite Y', 'Blaze', ['heatwave', 'airslash', 'solarbeam', 'protect']),
              SET('Raichu', 'Raichunite Y', 'Lightning Rod', ['thunderbolt', 'fakeout', 'protect', 'voltswitch']),
              SET('Garchomp', 'Choice Scarf', 'Rough Skin', ['earthquake', 'rockslide', 'dragonclaw', 'stompingtantrum']),
              SET('Incineroar', 'Sitrus Berry', 'Intimidate', ['fakeout', 'flareblitz', 'knockoff', 'partingshot'])]
    world = {'p1': [{'set': s} for s in ours], 'p2': [{'set': s} for s in theirs]}
    script = f"""
const E = require('./search/engine.cjs');
const w = {json.dumps(world)};
const b = E.buildBattle('{FMT}', w, {{turn: 1}});
const two = E.sideChoices(b, b.sides[1]), one = E.sideChoices(b, b.sides[0]);
const parts = c => c.split(',').map(p => p.trim());
console.log(JSON.stringify({{
  bothAttack: two.filter(c => parts(c).every(p => p.startsWith('move'))).length,
  maxMegas: Math.max(...two.map(c => parts(c).filter(p => p.endsWith(' mega')).length)),
  plainMoves: two.filter(c => parts(c).some(p => p.startsWith('move')) && !/ mega/.test(c)).length,
  oursNoStone: one.length}}));
"""
    out = json.loads(subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    assert out['bothAttack'] > 0 and out['maxMegas'] == 1 and out['plainMoves'] == 0
    assert out['oursNoStone'] > 0


def test_own_set_found_for_a_formes_mega():
    from search.agent import _our_set
    team = [{'species': 'Floette-Eternal', 'item': 'Floettite'}, {'species': 'Garchomp', 'item': 'Garchompite Z'}]
    assert _our_set(team, 'Floette-Mega')['species'] == 'Floette-Eternal'
    assert _our_set(team, 'Garchomp-Mega-Z')['species'] == 'Garchomp'
    assert _our_set(team, 'Floette-Eternal')['species'] == 'Floette-Eternal'
    assert _our_set(team, 'Politoed') is None


def test_failed_protect_resets_the_chain():
    log = LOG[:10] + ['|turn|1', '|move|p1a: Politoed|Protect|p1a: Politoed', '|-singleturn|p1a: Politoed|Protect',
                     '|turn|2', '|move|p1a: Politoed|Protect||[still]', '|-fail|p1a: Politoed', '|turn|3']
    mon = Tracker('p1').feed(log).slots['p1a']
    assert mon.last_move == 'protect' and mon.protect_failed
    # A successful Protect afterwards is chained again.
    mon = Tracker('p1').feed(log + ['|move|p1a: Politoed|Protect|p1a: Politoed', '|turn|4']).slots['p1a']
    assert not mon.protect_failed


@pytest.mark.skipif(not (ROOT / 'cache' / 'usage').exists(), reason='usage statistics not downloaded')
def test_floette_eternal_can_hold_its_mega_stone():
    priors.load_usage.cache_clear()
    rng = random.Random(3)
    items = [priors.sample_set(rng, 'Floette-Eternal')['item'] for _ in range(60)]
    assert items.count('floettite') > 40
    mega = priors.sample_set(rng, 'Floette-Mega', {'mega': True, 'mega_species': 'Floette-Mega'})
    assert mega['species'] == 'Floette-Eternal' and mega['item'] == 'floettite'


def test_our_harmful_status_moves_never_target_the_partner():
    agent = SearchAgent.__new__(SearchAgent)
    agent.keep_nonmega = True
    moves = [{'move': 'Fake Out', 'id': 'fakeout', 'target': 'normal'},
             {'move': 'Parting Shot', 'id': 'partingshot', 'target': 'normal'}]
    request = {'active': [{'moves': moves}, {'moves': [{'move': 'Helping Hand', 'id': 'helpinghand', 'target': 'adjacentAlly'}]}]}
    choices = ['move 2 -2, move 1 -1', 'move 2 1, move 1 -1', 'move 1 1, move 1 -1']
    assert agent._prune_ours({'request': request, 'choices': choices}) == choices[1:]


def test_struggle_request_empties_the_rebuilt_moves():
    agent = SearchAgent.__new__(SearchAgent)
    sets = [SET('Politoed', 'Mystic Water', 'Drizzle', ['muddywater', 'icebeam', 'weatherball', 'protect'])]
    request = {'side': {'id': 'p1', 'pokemon': [{'ident': 'p1: Politoed', 'details': 'Politoed, L50, M',
                                                 'condition': '50/167', 'active': True,
                                                 'moves': ['muddywater', 'icebeam', 'weatherball', 'protect']}]},
               'active': [{'moves': [{'move': 'Struggle', 'id': 'struggle', 'target': 'randomNormal'}]}]}
    side = agent._our_side({'request': request, 'team_sets': sets}, Tracker('p1').feed(LOG[:10]))
    assert side[0]['state']['pp'] == {'muddywater': 0, 'icebeam': 0, 'weatherball': 0, 'protect': 0}


@pytest.mark.skipif(not (ROOT / 'cache' / 'dex').exists(), reason='dex cache not built')
def test_mega_priority_keeps_the_stone_for_garchomp():
    agent = SearchAgent.__new__(SearchAgent)
    agent.keep_nonmega = False
    agent.mega_avoid = ['golisopod']
    party = [{'details': 'Golisopod, L50, F', 'item': 'golisopite', 'condition': '100/100', 'active': True},
             {'details': 'Incineroar, L50, M', 'item': 'choplefberry', 'condition': '100/100', 'active': True},
             {'details': 'Garchomp, L50, M', 'item': 'garchompitez', 'condition': '100/100'},
             {'details': 'Politoed, L50, F', 'item': 'mysticwater', 'condition': '100/100'}]
    move = {'move': 'Leech Life', 'id': 'leechlife', 'target': 'normal'}
    request = {'side': {'pokemon': party},
               'active': [{'moves': [move], 'canMegaEvo': True}, {'moves': [dict(move, id='fakeout')]}]}
    choices = ['move 1 1 mega, move 1 2', 'move 1 1, move 1 2', 'switch 3, move 1 2']
    assert agent._prune_ours({'request': request, 'choices': choices}) == ['move 1 1, move 1 2', 'switch 3, move 1 2']
    # Once Garchomp has fainted, Golisopod takes the Mega again (and it stays forced).
    party[2]['condition'] = '0 fnt'
    assert agent._prune_ours({'request': request, 'choices': choices}) == ['move 1 1 mega, move 1 2']


@pytest.mark.skipif(shutil.which('node') is None or not (ROOT / 'node_modules' / 'pokemon-showdown' / 'dist').exists(),
                    reason='official simulator is not built')
def test_eval_terms_price_boosts_and_weather_owner():
    ours = [SET('Politoed', 'Mystic Water', 'Drizzle', ['muddywater', 'icebeam', 'weatherball', 'protect']),
            SET('Archaludon', 'Leftovers', 'Stamina', ['dragonpulse', 'electroshot', 'snarl', 'protect'])]
    theirs = [SET('Charizard', 'Charizardite Y', 'Blaze', ['heatwave', 'airslash', 'solarbeam', 'protect']),
              SET('Garchomp', 'Life Orb', 'Rough Skin', ['earthquake', 'dragonclaw', 'rockslide', 'protect'],
                  'Jolly', {'hp': 2, 'atk': 32, 'spe': 32})]
    world = {'p1': [{'set': s} for s in ours],
             'p2': [{'set': theirs[0]}, {'set': theirs[1], 'state': {'boosts': {'atk': 2}, 'turn': 2}}]}
    field = {'turn': 2, 'weather': {'id': 'sunnyday', 'duration': 3}}
    terms = {'boost': 0.5, 'foe_boost': 0.15, 'speed': 0.4, 'evasion': 0.3, 'weather': 0.4}
    msgs = [{'type': 'debug', 'id': 0, 'format': FMT, 'worlds': [world], 'field': field},
            {'type': 'debug', 'id': 1, 'format': FMT, 'worlds': [world], 'field': field, 'eval_terms': terms},
            {'type': 'debug', 'id': 2, 'format': FMT, 'worlds': [world], 'field': field}]
    proc = subprocess.run(['node', str(ROOT / 'search' / 'engine.cjs')], capture_output=True, text=True, timeout=120,
                          input=''.join(json.dumps(m) + '\n' for m in msgs))
    plain, priced, again = [json.loads(line) for line in proc.stdout.splitlines()]
    assert priced['p2'][1]['boosts']['atk'] == 2 and priced['field']['weather'] == 'sunnyday'
    # +2 Atk on a full-HP foe: (0.5 + 0.15 - 0.12) * 2 more; their sun (Charizardite Y): 0.4 more.
    assert abs((plain['value'] - priced['value']) - (0.53 * 2 + 0.4)) < 1e-6
    # Terms are per message: the next call without them is the deployed evaluation again.
    assert again['value'] == plain['value']


@pytest.mark.skipif(shutil.which('node') is None or not (ROOT / 'node_modules' / 'pokemon-showdown' / 'dist').exists(),
                    reason='official simulator is not built')
def test_endgame_second_turn_triggers_with_two_left():
    ours = [SET('Politoed', 'Mystic Water', 'Drizzle', ['muddywater', 'icebeam', 'weatherball', 'protect']),
            SET('Archaludon', 'Leftovers', 'Stamina', ['dragonpulse', 'electroshot', 'snarl', 'protect'])]
    theirs = [SET('Garchomp', 'Life Orb', 'Rough Skin', ['earthquake', 'dragonclaw', 'rockslide', 'protect'],
                  'Jolly', {'hp': 2, 'atk': 32, 'spe': 32}),
              SET('Sylveon', 'Leftovers', 'Pixilate', ['hypervoice', 'protect', 'moonblast', 'yawn'])]
    world = {'p1': [{'set': s, 'state': {'turn': 6}} for s in ours], 'p2': [{'set': s, 'state': {'turn': 6}} for s in theirs]}
    msg = {'type': 'search', 'id': 1, 'format': FMT, 'worlds': [world], 'field': {'turn': 6}, 'crn': True,
           'endgame_d2': {'alive': 2, 'top_ours': 3, 'top_opp': 3, 'keep': 6, 'probes': 2}}
    proc = subprocess.run(['node', str(ROOT / 'search' / 'engine.cjs')], capture_output=True, text=True, timeout=300,
                          input=json.dumps(msg) + '\n')
    r = json.loads(proc.stdout)['results'][0]
    assert r['depth2']['trigger'] == 'endgame' and r['depth2']['cells'] > 0


@pytest.mark.skipif(not (ROOT / 'cache' / 'usage').exists(), reason='usage statistics not downloaded')
def test_back_rate_bench_prior_samples_unseen_members():
    six = ['Rillaboom', 'Kommo-o', 'Incineroar', 'Sneasler', 'Garchomp', 'Pelipper']
    rng = random.Random(5)
    picks = [tuple(priors.sample_bench(rng, six, six[:2], 2, mode='back_rate', leads=six[:2])) for _ in range(200)]
    assert all(len(set(p)) == 2 and not set(p) & set(six[:2]) for p in picks)
    # Ladder usage made Rillaboom ~18x Kommo-o; bring odds keep them comparable.
    usage = [tuple(priors.sample_bench(rng, ['Rillaboom', 'Kommo-o'] + six[2:4], six[2:4], 1)) for _ in range(400)]
    odds = [tuple(priors.sample_bench(rng, ['Rillaboom', 'Kommo-o'] + six[2:4], six[2:4], 1, mode='back_rate',
                                      leads=six[2:4])) for _ in range(400)]
    assert sum(p == ('Kommo-o',) for p in odds) > 3 * sum(p == ('Kommo-o',) for p in usage)


def test_preview_value_averages_an_ensemble(tmp_path):
    import numpy as np
    from search.human.preview_model import PLANS, _sid, plan_features
    from search.human.preview_value import PreviewValue
    own = ['Garchomp', 'Politoed', 'Incineroar', 'Archaludon', 'Farigiraf', 'Golisopod']
    opp = ['Sneasler', 'Kingambit', 'Rillaboom', 'Incineroar', 'Garchomp', 'Charizard']
    d = len(plan_features([_sid(x) for x in own], [_sid(x) for x in opp], PLANS[0]))
    rng = np.random.default_rng(0)
    net = lambda: {'layers': [{'W': rng.normal(size=(4, d + 2)).tolist(), 'b': [0.0] * 4},  # noqa: E731
                              {'W': rng.normal(size=(1, 4)).tolist(), 'b': [0.0]}]}
    a, b = net(), net()
    for name, w in (('a', {**a, 'rating_features': 2}), ('b', {**b, 'rating_features': 2}),
                    ('ab', {'members': [a, b], 'rating_features': 2})):
        (tmp_path / f'{name}.json').write_text(json.dumps(w))
    pa, pb, pab = (PreviewValue(tmp_path / f'{n}.json').probs(own, opp) for n in ('a', 'b', 'ab'))
    assert pab.shape == (len(PLANS),) and np.allclose(pab, (pa + pb) / 2)
