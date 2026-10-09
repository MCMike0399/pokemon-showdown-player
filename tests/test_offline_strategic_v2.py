import copy

import pytest

from ml.features import Features
from ml.strategic_mechanics import accuracy, estimate, move_data, sleep_probability, unproductive_solo_protect
from ml.strategy import grounded, hit, hypotheses, score, prepare, component, simulate
from test_offline_strategy import context


def test_body_press_uses_defense_and_foul_play_uses_targets_attack():
    f, ctx = context()
    a = {**ctx['state']['my_actives'][0], 'stats': {'atk': 50, 'def': 200}, 'boosts': {'atk': -6, 'def': 2}}
    t = {**ctx['state']['opp_actives'][0], 'stats': {'atk': 50, 'def': 100}, 'boosts': {'atk': 2}}
    d = {'category': 'Physical', 'type': 'Normal', 'basePower': 80, 'accuracy': 100}
    first = estimate(a, t, 'bodypress', d, ctx['state'], f)
    assert estimate({**a, 'stats': {'atk': 500, 'def': 200}}, t, 'bodypress', d, ctx['state'], f) == first
    assert estimate({**a, 'boosts': {'atk': 6, 'def': 0}}, t, 'bodypress', d, ctx['state'], f) < first / 1.8
    low = estimate(a, t, 'foulplay', d, ctx['state'], f)
    assert estimate(a, {**t, 'stats': {'atk': 200, 'def': 100}}, 'foulplay', d, ctx['state'], f) > low * 3


def test_stored_power_and_terrain_spread_are_resolved_once_without_mutation():
    f, ctx = context(); a = ctx['state']['my_actives'][0]
    d = {'category': 'Special', 'type': 'Psychic', 'basePower': 20, 'target': 'normal'}
    assert move_data('storedpower', d, {**a, 'boosts': {'def': 2, 'spe': 2, 'atk': -3}}, ctx['state'], True)['basePower'] == 100
    state = {**ctx['state'], 'field': ['move: Psychic Terrain']}
    d['basePower'] = 80
    resolved = move_data('expandingforce', d, a, state, True)
    assert resolved['basePower'] == 120 and resolved['target'] == 'allAdjacentFoes'
    assert move_data('expandingforce', resolved, a, state, True) == resolved
    assert move_data('expandingforce', d, a, state, False)['basePower'] == 80
    assert d['basePower'] == 80 and d['target'] == 'normal'


def test_darkest_lariat_and_psyshock_use_the_correct_defensive_stat():
    f, ctx = context(); a = ctx['state']['my_actives'][0];t = ctx['state']['opp_actives'][0]
    d = {'category': 'Physical', 'type': 'Normal', 'basePower': 85, 'accuracy': 100}
    assert estimate(a, {**t, 'boosts': {'def': 6}}, 'darkestlariat', d, ctx['state'], f) == estimate(a, t, 'darkestlariat', d, ctx['state'], f)
    t = {**t, 'stats': {'def': 50, 'spd': 200}}
    d['category'] = 'Special'
    assert estimate(a, t, 'psyshock', d, ctx['state'], f) > estimate(a, t, 'psychic', d, ctx['state'], f) * 3


def test_accuracy_accounts_for_visible_stages_and_rain_always_hit():
    assert accuracy({'boosts': {'accuracy': -1}}, {}, {'accuracy': 100}) == .75
    assert accuracy({'boosts': {'accuracy': 1}}, {}, {'accuracy': 60}) == .8
    assert accuracy({'boosts': {'accuracy': -6}}, {'boosts': {'evasion': 6}}, {'accuracy': True}) == 1
    assert accuracy({'ability': 'Compound Eyes'}, {}, {'accuracy': 75}) == pytest.approx(.975)


def test_sleep_forecasts_are_targeted_and_respect_immunities():
    f, ctx = context();a = ctx['state']['opp_actives'][0];t = ctx['state']['my_actives'][0]
    d = {'category': 'Status', 'accuracy': 60, 'target': 'normal'}
    assert sleep_probability(a, t, 'hypnosis', d, ctx['state'], f) == .6
    assert sleep_probability(a, {**t, 'condition': '170/170 par'}, 'hypnosis', d, ctx['state'], f) == 0
    for protected in ({**t, 'ability': 'Insomnia'}, {**t, 'item': 'Safety Goggles'}):
        assert sleep_probability(a, protected, 'sleeppowder', d, ctx['state'], f) == 0
    assert sleep_probability(a, t, 'hypnosis', d, {**ctx['state'], 'field': ['ElectricTerrain']}, f) == 0
    assert sleep_probability(a, {**t, 'ability': 'Levitate'}, 'hypnosis', d, {**ctx['state'], 'field': ['ElectricTerrain']}, f) == .6
    f.dex['moves']['hypnosis'] = d
    ctx['state']['opp_team_sheet'] = [{'species': 'Enemy', 'moves': ['Hypnosis']}]
    assert not any(n == 'hypnosis' for n, *_ in hypotheses(ctx, a, [t], f))
    ctx['state']['_strategic_v2'] = True
    result = hypotheses(ctx, a, [t], f)
    assert result and all(n == 'hypnosis' and index == 0 for n, _, index, _ in result)


def test_no_move_evidence_still_produces_bounded_attack_hypotheses():
    f, ctx = context();ctx['state']['_strategic_v2'] = True
    mon = {**ctx['state']['opp_actives'][0], 'moves_known': []}
    result = hypotheses(ctx, mon, ctx['state']['my_actives'], f)
    assert result and all(data.get('basePower') for _, data, _, _ in result)
    assert sum(weight for _, _, _, weight in result) == pytest.approx(1)


def test_psyshock_does_not_use_assault_vest_special_defense():
    f, ctx = context();ctx['state']['_strategic_v2'] = True
    actor=ctx['state']['my_actives'][0];target=ctx['state']['opp_actives'][0]
    data={'category': 'Special', 'type': 'Normal', 'basePower': 80, 'accuracy': 100}
    vest={**target,'item':'Assault Vest'}
    assert hit(actor,vest,'psyshock',data,ctx['state'],f,'mine') == hit(actor,target,'psyshock',data,ctx['state'],f,'mine')
    assert hit(actor,vest,'psychic',data,ctx['state'],f,'mine') < hit(actor,target,'psychic',data,ctx['state'],f,'mine')


def test_fast_sleep_denies_expected_attack_but_protect_stops_sleep():
    f, ctx = context();ctx['state'].update(_strategic_v2=True)
    ctx['state']['opp_actives'][0]['stats'] = {'spe': 300}
    p = prepare(ctx, f)
    attack = [component(ctx, 0, 'move 1 1', f, p), component(ctx, 1, 'pass', f, p)]
    protected = [component(ctx, 0, 'move 3', f, p), component(ctx, 1, 'pass', f, p)]
    sleep = [('hypnosis', {'category': 'Status', 'accuracy': 100, 'target': 'normal'}, 0, 1)]
    idle = [('', {}, None, 1)]
    assert simulate(ctx, attack, sleep, p, f) < simulate(ctx, attack, idle, p, f) - .2
    assert simulate(ctx, protected, sleep, p, f) == simulate(ctx, protected, idle, p, f)


def test_expanding_force_resolves_both_foes_from_a_single_target_hypothesis():
    f, ctx = context();ctx['state'].update(_strategic_v2=True, field=['PsychicTerrain'])
    f.dex['moves']['expandingforce'] = {'category': 'Special', 'type': 'Psychic', 'basePower': 80, 'accuracy': 100, 'target': 'normal'}
    p = prepare(ctx, f)
    ours = [component(ctx, i, 'pass', f, p) for i in range(2)]
    enemy = [('expandingforce', f.dex['moves']['expandingforce'], 0, 1)]
    spread = simulate(ctx, ours, enemy, p, f)
    ctx['state']['field'] = []
    single = simulate(ctx, ours, enemy, p, f)
    assert spread < single * 1.7


def test_solo_protect_cost_preserves_residual_recovery_and_timers():
    f, ctx = context();ctx['state']['my_party'] = ctx['state']['my_actives'][:1]
    p = prepare(ctx, f);ours = [component(ctx, 0, 'move 3', f, p)];foes=p['enemy']
    assert unproductive_solo_protect(ctx, ours, foes) > 0
    assert unproductive_solo_protect(ctx, ours, [{**foes[0], 'condition': '100/100 tox'}]) == 0
    ctx['state']['field'] = ['TrickRoom']
    assert unproductive_solo_protect(ctx, ours, foes) == 0
    ctx['state']['field'] = [];ctx['state']['my_party'][0]['item'] = 'Leftovers'
    assert unproductive_solo_protect(ctx, ours, foes) == 0


def test_v2_scoring_never_changes_observed_state_and_uses_cached_context():
    f, ctx = context();ctx['feature_profile'] = 'strategic-v2';before=copy.deepcopy(ctx['state'])
    first = score(ctx, 'move 1 1, move 2', f)
    assert score(ctx, 'move 1 1, move 2', f) == first
    assert ctx['state'] == before
