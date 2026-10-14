"""Human opponent policy: log parsing, labelling, engine enumeration, blending, imitation."""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import numpy as np
import pytest

from search.human.extract import match_choice, player_info, split_turns, turn_actions

ROOT = Path(__file__).resolve().parents[1]
FMT = 'gen9championsvgc2026regmc'

TURN = [
    '|turn|3', '|', '|t:|1', '|detailschange|p1a: Staraptor|Staraptor-Mega, L50, F', '|-mega|p1a: Staraptor|Staraptor|Staraptite',
    '|move|p2a: Politoed|Protect|p2a: Politoed', '|-singleturn|p2a: Politoed|Protect',
    '|move|p1b: Rillaboom|Grassy Glide|p2a: Politoed', '|-activate|p2a: Politoed|move: Protect',
    '|move|p1a: Staraptor|Close Combat|p2b: Golisopod', '|-damage|p2b: Golisopod|59/100',
    '|cant|p2b: Golisopod|flinch', '|', '|upkeep', '|', '|switch|p1b: Rotom|Rotom-Wash, L50|100/100', '|turn|4',
]


def test_turn_actions_reads_voluntary_actions_only():
    acts = turn_actions(TURN[1:-1])
    assert acts['p1a'] == {'kind': 'move', 'move': 'closecombat', 'target': 'p2b', 'mega': True, 'notarget': False}
    assert acts['p1b']['move'] == 'grassyglide' and acts['p1b']['target'] == 'p2a' and not acts['p1b']['mega']
    assert acts['p2a']['move'] == 'protect'
    assert acts['p2b'] == {'kind': 'unknown'}  # flinched: the chosen move is not logged
    assert 'switch' not in json.dumps(acts)  # the post-upkeep switch replaced a fainted Pokemon


def test_match_choice_exact_and_marginal():
    out = {'choices': ['move 1 1, move 2 1', 'move 1 1 mega, move 2 1', 'move 1 2 mega, move 2 1', 'move 1 2 mega, switch 3',
                       'switch 3, move 2 1'],
           'moves': [['closecombat', 'bravebird'], ['fakeout', 'grassyglide']], 'party': ['Staraptor', 'Rillaboom', 'Rotom-Wash']}
    acts = turn_actions(TURN[1:-1])
    mask, why = match_choice(out, acts, 'p1', out['party'], [True, True])
    assert why == 'ok' and mask == [2]
    # The partner's action unknown: every move part for that slot is consistent.
    mask, why = match_choice(out, {'p1a': acts['p1a']}, 'p1', out['party'], [True, True])
    assert why == "marginal" and mask == [2]  # a switch resolves before moves, so the unknown slot moved
    # A Mega Evolution seen in the log restricts the unknown slot to Mega parts.
    mask, why = match_choice(out, {'p1b': acts['p1b'], 'p1a': {'kind': 'unknown', 'mega': True}}, 'p1', out['party'], [True, True])
    assert why == 'marginal' and mask == [1, 2]
    # An empty slot must be 'pass'.
    assert match_choice({'choices': ['move 1 1, pass'], 'moves': [['closecombat'], []], 'party': ['Staraptor']},
                        {'p1a': {'kind': 'move', 'move': 'closecombat', 'target': 'p2a', 'mega': False, 'notarget': False}},
                        'p1', ['Staraptor'], [True, False]) == ([0], 'ok')


def _world():
    from search.priors import sample_set
    rng = random.Random(3)
    team = json.loads((ROOT / 'teams.json').read_text())['Rain-Recife-special-stat-fix']['sets']
    ours = [{'set': s, 'state': {'hp': 1.0, 'turn': 1}} for s in team[:4]]
    theirs = [{'set': sample_set(rng, sp, {}, FMT), 'state': {'hp': 1.0, 'turn': 1}}
              for sp in ('Rillaboom', 'Incineroar', 'Sneasler', 'Kingambit')]
    field = {'turn': 1, 'weather': None, 'terrain': None, 'pseudo': {}, 'sides': {'p1': {}, 'p2': {}},
             'mega_used': {'p1': False, 'p2': False}}
    return {'p1': ours, 'p2': theirs}, field


def _call(*msgs):
    """Run engine calls in one event loop (the subprocess pipes belong to that loop)."""
    from search.agent import Engine

    async def go():
        e = Engine()
        try:
            return [await e.call(m) for m in msgs]
        finally:
            await e.close()
    return asyncio.run(go())


def test_engine_human_message_enumerates_with_features():
    from search.human.humanfeat_dim import DIM
    world, field = _world()
    out = _call({'type': 'human', 'format': FMT, 'field': field, 'worlds': [world]})[0]
    assert out['choices'] and len(out['a']) == len(out['choices'])
    assert len(out['a'][0]) == DIM and len(out['sx']) == 146
    assert out['moves'][0] and out['party'][0] == world['p1'][0]['set']['species']
    assert any(c.endswith(' mega') or ' mega,' in c for c in out['choices']) or not any(out['can_mega'])


def _tiny_model(tmp_path: Path) -> Path:
    from search.human.humanfeat_dim import DIM
    d = 146 + DIM
    rng = np.random.default_rng(0)
    weights = {'layers': [{'W': (rng.normal(size=(8, d)) * 0.05).tolist(), 'b': [0.0] * 8},
                          {'W': (rng.normal(size=(1, 8)) * 0.1).tolist(), 'b': [0.0]}],
               'mean': [0.0] * d, 'std': [1.0] * d, 'version': 1}
    p = tmp_path / 'human-test.json'
    p.write_text(json.dumps(weights))
    return p


def test_engine_opp_model_screens_and_blends(tmp_path):
    world, field = _world()
    model = _tiny_model(tmp_path)
    base = {'type': 'search', 'format': FMT, 'field': field, 'worlds': [world], 'seeds': 1, 'alpha': 0.5,
            'opp_tau': 1.0, 'screen': {'probe': 4, 'keep_opp': 8, 'keep_ours': 10}, 'crn': True, 'opp_detail': True}
    plain, human = [r['results'][0] for r in _call(base, {**base, 'opp_model': {'path': str(model), 'w': 0.3, 'temp': 1.0, 'sha256': 'x'}})]
    assert 'error' not in plain and 'error' not in human
    assert human['opp_model'] == 'x' and plain['opp_model'] is None
    assert human['n_theirs'] <= 8 and abs(sum(human['opp_detail']['q']) - 1) < 1e-6
    # Their probe screening is replaced by the policy: fewer screening simulations.
    assert human['screen_sims'] < plain['screen_sims']
    # Our side is still probe-screened to the same size (against their likeliest replies now).
    assert len(human['values']) == len(plain['values']) == 10


def test_store_pointer_verifies_sha(tmp_path):
    from search.human.store import current_opp, model_from_file, promote_opp
    model = _tiny_model(tmp_path)
    pointer = tmp_path / 'opp-current.json'
    assert current_opp(pointer)['path'] is None
    promote_opp(model, 0.3, 1.0, {'test': True}, pointer)
    cur = current_opp(pointer)
    assert cur['path'] == str(model.resolve()) and cur['w'] == 0.3 and cur['sha256'] == model_from_file(model)['sha256']
    model.write_text(model.read_text().replace('"version": 1', '"version": 2'))
    assert current_opp(pointer)['path'] is None and current_opp(pointer).get('rejected') == 'sha mismatch'


def test_imitation_agent_returns_legal_choice(tmp_path):
    from battle_state import legal_choices
    from search.human.imitation import ImitationAgent
    model = _tiny_model(tmp_path)
    team = json.loads((ROOT / 'teams.json').read_text())['Rain-Recife-special-stat-fix']['sets']
    request = json.loads((ROOT / 'tests' / 'fixtures' / 'human-request.json').read_text()) if (ROOT / 'tests' / 'fixtures' / 'human-request.json').exists() else None
    if request is None:
        pytest.skip('no request fixture')
    log = request.pop('_public_log')
    ctx = {'request': request, 'choices': legal_choices(request), 'team_sets': team, 'public_log': log}
    async def go():
        agent = ImitationAgent(model, fmt=FMT, seed=1)
        try:
            return await agent.decide(ctx), agent.stats
        finally:
            await agent.close()
    choice, stats = asyncio.run(go())
    assert choice in ctx['choices']
    assert stats['fallbacks'] == 0


def test_player_info_and_turns():
    log = ['|player|p1|A|1|1200', '|player|p2|B|2|1100', '|turn|1', '|move|p1a: X|Protect|p1a: X', '|turn|2', '|win|B']
    info = player_info(log)
    assert info['p1'] == {'name': 'A', 'rating': 1200} and info['winner'] == 'B'
    assert split_turns(log) == [(1, 2, 4), (2, 4, 6)]
