import copy

import pytest

from ml.features import Features
from test_offline_rain import rain_context


def test_locked_electro_shot_release_is_not_a_second_charge():
    from ml.mechanics import charging
    _, ctx = rain_context()
    ctx['state']['weather'] = None
    actor = ctx['state']['my_actives'][0]
    assert charging(ctx, actor, {'id': 'electroshot', 'target': 'normal'})
    # Showdown omits the target from a forced release request.
    assert not charging(ctx, actor, {'id': 'electroshot'})
    assert not charging(ctx, actor, {'id': 'electroshot', 'target': 'scripted'})
    ctx['state']['weather'] = 'RainDance'
    assert not charging(ctx, actor, {'id': 'electroshot', 'target': 'normal'})


def test_support_scores_the_actual_joint_partner_and_weather_ball():
    f, ctx = rain_context()
    ctx['feature_profile'] = 'mechanics-v1'
    helped = f.action(ctx, 'move 2 1, move 1 -1')[-1]
    attack = f.action(ctx, 'move 2 1, move 2')[-1]
    assert helped > attack
    assert f.action(ctx, 'move 3, move 1 -1')[-1] < attack
    ctx['state']['my_actives'][0]['moves'][1]['id'] = 'weatherball'
    f.dex['moves']['weatherball'] = {'type': 'Normal', 'category': 'Special', 'basePower': 50,
                                   'accuracy': 100, 'target': 'normal'}
    assert f.action(ctx, 'move 2 1, move 2')[-16] == 1
    assert f.action({**ctx, 'feature_profile': 'legacy'}, 'move 2 1, move 2')[-16] == .5


def test_observed_immunity_blocks_ground_and_priority_but_not_support():
    f, ctx = rain_context()
    ctx['feature_profile'] = 'mechanics-v1'
    foe = ctx['state']['opp_actives'][0]
    f.dex['moves']['dragonpulse']['type'] = 'Ground'
    regular = f.action(ctx, 'move 2 1, move 2')[-1]
    foe['ability'] = 'Levitate'
    assert f.action(ctx, 'move 2 1, move 2')[-1] < regular
    f.dex['moves']['dragonpulse'].update(type='Dragon', priority=1)
    foe['ability'] = 'Armor Tail'
    assert f.action(ctx, 'move 2 1, move 2')[-1] < regular
    # Ally-targeted Helping Hand remains usable through opposing Armor Tail.
    f.dex['moves']['dragonpulse']['priority'] = 0
    assert f.action(ctx, 'move 2 1, move 1 -1')[-1] > f.action(ctx, 'move 2 1, move 2')[-1]


def test_mechanics_profile_preserves_legacy_state_and_preview_and_input_objects():
    f, ctx = rain_context()
    ctx['state']['request_type'] = 'move'
    ctx['choices'] = ['move 2 1, move 2']
    before = copy.deepcopy(ctx)
    import numpy as np
    assert np.array_equal(f.state({**ctx, 'feature_profile': 'legacy'}),
                          f.state({**ctx, 'feature_profile': 'mechanics-v1'}))
    assert np.array_equal(f.action({**ctx, 'feature_profile': 'legacy'}, 'team 1,2'),
                          f.action({**ctx, 'feature_profile': 'mechanics-v1'}, 'team 1,2'))
    f.action({**ctx, 'feature_profile': 'mechanics-v1'}, ctx['choices'][0])
    assert ctx == before


def test_lineup_prior_requires_bringing_the_setter_and_rewards_compatible_leads():
    from ml.preview import rain_lineup_score
    f = Features()
    party = [{'species': 'Politoed', 'baseAbility': 'Drizzle', 'moves': ['weatherball']},
             {'species': 'Archaludon', 'moves': ['electroshot']},
             {'species': 'Incineroar', 'moves': ['flareblitz']},
             {'species': 'Garchomp', 'moves': ['dragonpulse']}]
    ctx = {'state': {'my_party': party, 'opp_preview': []}}
    assert rain_lineup_score(ctx, 'team 1,2,3', f) > rain_lineup_score(ctx, 'team 3,2,1', f)
    assert rain_lineup_score(ctx, 'team 3,2,1', f) > rain_lineup_score(ctx, 'team 3,2,4', f)
    assert rain_lineup_score(ctx, 'team 1,2,3,4', f) == rain_lineup_score(ctx, 'team 1,2,4,3', f)


@pytest.mark.parametrize('open_sheets', [False, True])
def test_simulation_respects_sheet_visibility_and_returns_pair_identity(tmp_path, open_sheets):
    import asyncio
    from ml.brain import Brain
    from ml.storage import Store
    from ml.simulator import play_local
    from test_offline_simulator import fixture_team, FMT
    store = Store(tmp_path)
    brain = Brain(store, Features.cached(FMT))
    result = asyncio.run(play_local(brain, FMT, fixture_team(), fixture_team(),
                                   opponent='tactical', seed=31, open_team_sheets=open_sheets))
    episode = store.episodes(FMT)[0]
    sheets = [s['snapshot']['observation']['opp_team_sheet'] for s in episode['steps']]
    assert any(sheets) == open_sheets
    assert result['open_team_sheets'] == open_sheets
    assert result['learner_team'] == result['opponent_team']
    assert result['rejected_actions'] == 0 and not result.get('unfinished')
    store.close()
