import copy
import pytest

from ml.teams import generated_spreads

FMT = 'gen9championsvgc2026regmc'


def test_generated_spread_follows_special_moves_on_physical_base_species():
    mon = {'species': 'Garchomp', 'item': 'Garchompite Z', 'nature': 'Modest',
           'moves': ['Dragon Pulse', 'Earth Power', 'Power Gem', 'Protect']}
    sets, generated = generated_spreads([mon], FMT)
    assert generated
    assert sets[0]['evs'] == {'spa': 32, 'spe': 32, 'hp': 2}
    assert 'evs' not in mon


def test_generated_physical_and_support_spreads_and_explicit_spreads_are_preserved():
    physical = {'species': 'Garchomp', 'nature': 'Adamant', 'moves': ['Dragon Claw', 'Earthquake']}
    support = {'species': 'Farigiraf', 'nature': 'Calm', 'moves': ['Psychic', 'Trick Room']}
    explicit = {**physical, 'evs': {'hp': 32, 'atk': 32, 'spe': 2}}
    before = copy.deepcopy([physical, support, explicit])
    sets, _ = generated_spreads([physical, support, explicit], FMT)
    assert sets[0]['evs'] == {'atk': 32, 'spe': 32, 'hp': 2}
    assert sets[1]['evs'] == {'hp': 32, 'spd': 32, 'def': 2}
    assert sets[2] == explicit
    assert [physical, support, explicit] == before


def test_generated_standard_evs_follow_moves_too():
    mon = {'species': 'Garchomp', 'nature': 'Modest', 'moves': ['Dragon Pulse', 'Earth Power']}
    sets, _ = generated_spreads([mon], 'gen9vgc2026regi')
    assert sets[0]['evs'] == {'spa': 252, 'spe': 252, 'hp': 4}


def test_exact_team_plan_retains_generated_spread_provenance_and_limits_variants(tmp_path, monkeypatch):
    import asyncio
    import json
    from harness import TeamStore
    from ml.storage import Store
    from ml.teams import TeamPlanner, team_id
    store = Store(tmp_path / 'ml')
    teams = TeamStore(tmp_path / 'teams.json')
    mon = {'species': 'Garchomp', 'moves': ['Dragon Pulse'], 'evs': {'spa': 32, 'spe': 32, 'hp': 2}}
    teams.create('Corrected', [mon], FMT)
    key = team_id(FMT, [mon])
    provenance = {'stat_points': 'generated_hypothesis', 'source_url': 'https://example.org/sheet'}
    with store.db:
        store.db.execute('INSERT INTO team_metadata VALUES (?,?)', (key, json.dumps(provenance)))
        store.db.execute('INSERT INTO research_teams VALUES (?,?,?,?,?,?,?)',
                         ('alternate', FMT, 'https://example.org/alternate', 'Trainer', 'Event', 'today',
                          json.dumps([{**mon, 'moves': ['Earth Power']}])) )
    calls = []
    async def validate(fmt, sets):
        calls.append(sets)
        return {'errors': [], 'packed': 'packed'}
    monkeypatch.setattr('ml.simulator.validate_team', validate)
    result = asyncio.run(TeamPlanner(store, teams).plan(FMT, 'Corrected', max_candidates=1))
    assert calls == [[mon]]
    assert result['selected']['id'] == key
    assert result['selected']['stat_points'] == 'generated_hypothesis'
    assert result['selected']['source_url'] == provenance['source_url']
    assert result['alternatives'] == []
    store.close()


def rain_context():
    from ml.features import Features
    dex = {'pokedex': {'archaludon': {'types': ['Dragon', 'Steel'], 'baseStats': {'spa': 125}},
                      'farigiraf': {'types': ['Normal', 'Psychic']},
                      'target': {'types': ['Normal'], 'baseStats': {'hp': 100, 'spd': 100}}},
           'moves': {'electroshot': {'type': 'Electric', 'category': 'Special', 'basePower': 130, 'accuracy': 100, 'target': 'normal'},
                     'dragonpulse': {'type': 'Dragon', 'category': 'Special', 'basePower': 40, 'accuracy': 100, 'target': 'normal'},
                     'helpinghand': {'category': 'Status', 'basePower': 0, 'target': 'adjacentAlly'},
                     'protect': {'category': 'Status', 'basePower': 0, 'target': 'self'}}}
    actor = {'species': 'Archaludon', 'slot': 'p1a', 'condition': '100/100', 'stats': {'spa': 150},
             'moves': [{'id': 'electroshot'}, {'id': 'dragonpulse'}, {'id': 'protect'}]}
    helper = {'species': 'Farigiraf', 'slot': 'p1b', 'condition': '100/100',
              'moves': [{'id': 'helpinghand'}, {'id': 'protect'}]}
    ctx = {'feature_profile': 'rain-v1', 'state': {'turn': 1, 'weather': 'RainDance',
           'my_party': [actor, helper], 'my_actives': [actor, helper],
           'opp_actives': [{'species': 'Target', 'slot': 'p2a', 'condition': '100/100'}]}}
    return Features(dex), ctx


def test_rain_charge_is_not_scored_as_immediate_damage():
    from ml.rain import score
    f, ctx = rain_context()
    immediate = score(ctx, 'move 1 1, move 2', f)
    ctx['state']['weather'] = None
    charging = score(ctx, 'move 1 1, move 2', f)
    assert charging == pytest.approx(.25)
    assert immediate > charging
    # The observable generic two-turn volatile establishes a release phase.
    ctx['state']['my_actives'][0]['volatiles'] = ['twoturnmove']
    assert score(ctx, 'move 1 1, move 2', f) > charging


def test_helping_hand_is_joint_support_and_does_not_reward_partner_protect():
    from ml.rain import score
    f, ctx = rain_context()
    attack = score(ctx, 'move 2 1, move 2', f)
    helped = score(ctx, 'move 2 1, move 1 -1', f)
    assert helped > attack
    assert score(ctx, 'move 3, move 1 -1', f) == pytest.approx(.15)
    legacy = {**ctx, 'feature_profile': 'legacy'}
    assert f.action(legacy, 'move 2 1, move 1 -1')[-1] == pytest.approx(float(f.action(legacy, 'move 2 1, move 2')[-1]) - .15)


def test_observed_fake_out_entry_resets_eligibility():
    from ml.rain import fake_out_available
    _, ctx = rain_context()
    actor = ctx['state']['my_actives'][0]
    ctx['public_log'] = ['|switch|p1a: Mon|Archaludon|100/100', '|turn|1', '|turn|2']
    ctx['state']['turn'] = 2
    assert not fake_out_available(ctx, actor)
    ctx['public_log'] += ['|switch|p1a: Mon|Archaludon|100/100', '|turn|3']
    ctx['state']['turn'] = 3
    assert fake_out_available(ctx, actor)
    ctx['state']['turn'] = 4
    assert not fake_out_available(ctx, actor)


def test_mega_action_uses_selected_form_types_and_ability_without_mutating_actor():
    from ml.rain import mega_actor
    from ml.features import Features
    f = Features({'pokedex': {'garchomp': {'baseStats': {'spa': 80}},
                  'garchompmegaz': {'name': 'Garchomp-Mega-Z', 'baseSpecies': 'Garchomp',
                                   'requiredItem': 'Garchompite Z', 'types': ['Dragon'],
                                   'baseStats': {'spa': 141}, 'abilities': {'0': 'Levitate'}}}})
    actor = {'species': 'Garchomp', 'item': 'Garchompite Z', 'ability': 'Rough Skin', 'stats': {'spa': 144}}
    assert mega_actor(actor, ['move', '1'], f) is actor
    mega = mega_actor(actor, ['move', '1', 'mega'], f)
    assert mega['species'] == 'Garchomp-Mega-Z'
    assert mega['ability'] == 'Levitate'
    assert mega['stats']['spa'] > actor['stats']['spa']
    assert actor['species'] == 'Garchomp'


def test_rain_profile_records_exact_likelihood_and_rejects_legacy_training(tmp_path):
    import math
    import torch
    from ml.brain import Brain
    from ml.model import Model
    from ml.storage import Store
    f, ctx = rain_context()
    store = Store(tmp_path)
    brain = Brain(store, f)
    model = brain.model(FMT)
    model.feature_profile = 'rain-v1'
    model.revision = 'rain-fixture'
    model.save()
    ctx.update(room='local-rain', format=FMT, source='local', choices=['move 2 1, move 2', 'move 2 1, move 1 -1'],
               request={'rqid': 1, 'side': {'id': 'p1'}})
    ctx['state']['request_type'] = 'move'
    brain.decide(ctx, explore=True, record=True)
    episode = next(iter(brain.pending.values()))
    step = episode['steps'][0]
    states, actions, mask, indices = model.batch([step])
    logits, _ = model.net(states, actions, mask)
    actual = torch.distributions.Categorical(logits=model.training_logits(logits, [step])).log_prob(indices)
    assert math.isclose(float(actual[0].detach()), step['logprob'], abs_tol=1e-6)
    assert step['snapshot']['feature_profile'] == 'rain-v1'
    assert Model(tmp_path, FMT).feature_profile == 'rain-v1'
    brain.finish(ctx['room'], {'winner': 'Player'}, 'Player')
    model.feature_profile = 'legacy'
    assert not model.train(store.episodes(FMT))['trained']
    store.close()


def test_focused_practice_and_evaluation_keep_our_team_and_vary_opponents(tmp_path):
    from ml.worker import game_tasks
    from ml.continuous import LearningConfig
    focus = [{'species': 'Politoed'}]
    opponents = [[{'species': 'Charizard'}], [{'species': 'Tyranitar'}], [{'species': 'Gengar'}]]
    before = game_tasks(tmp_path, tmp_path / 'incumbent', FMT, opponents, 24, False, 100, tmp_path / 'fixed', focus)
    after = game_tasks(tmp_path, tmp_path / 'candidate', FMT, opponents, 24, False, 100, tmp_path / 'fixed', focus)
    assert all(t['team1'] == focus for t in before + after)
    assert {t['team2'][0]['species'] for t in before} == {'Charizard', 'Tyranitar', 'Gengar'}
    assert {t['side'] for t in before} == {'p1', 'p2'}
    for a, b in zip(before, after):
        assert {k: v for k, v in a.items() if k != 'checkpoint_root'} == {k: v for k, v in b.items() if k != 'checkpoint_root'}
    config = LearningConfig(training_teams={FMT: 'Corrected rain'})
    config.save(tmp_path)
    assert LearningConfig.load(tmp_path).training_teams == {FMT: 'Corrected rain'}
    with pytest.raises(ValueError, match='training_teams'):
        LearningConfig(training_teams={FMT: True})


def test_focused_learning_preserves_other_team_experience_without_training_it(tmp_path, monkeypatch):
    import asyncio
    from harness import TeamStore
    from ml.continuous import LearningConfig
    from ml.model import Model
    from ml.storage import Store
    from ml.teams import team_id
    from ml.worker import learn
    store = Store(tmp_path / 'ml')
    model = Model(store.root, FMT)
    team = [{'species': 'Politoed', 'moves': ['Weather Ball'], 'evs': {'spa': 32}}]
    teams = TeamStore(tmp_path / 'teams.json')
    teams.create('Rain', team, FMT)
    monkeypatch.setattr('harness.TeamStore', lambda: teams)
    monkeypatch.setattr('ml.worker.candidate_teams', lambda *args: [team])
    async def validate(*args):
        return {'errors': [], 'packed': 'packed'}
    monkeypatch.setattr('ml.simulator.validate_team', validate)
    for name, key, count in [('rain', team_id(FMT, team), 1), ('other', 'other-team', 70)]:
        store.save_episode({'id': name, 'format': FMT, 'source': 'ladder', 'team': key,
                            'revision': model.revision, 'status': 'complete', 'on_policy': True,
                            'steps': [{}] * count, 'outcome': 1})
    class Policy:
        def sample(self):
            return {'deferred': False}
    result = asyncio.run(learn(store, {'id': 'focus', 'format': FMT, 'kind': 'learn', 'payload': {}},
                              LearningConfig(training_teams={FMT: 'Rain'}), Policy(), float('inf')))
    assert result['reason'] == 'accumulating compatible experience'
    assert result['steps'] == 1
    assert result['training_team'] == {'name': 'Rain', 'fingerprint': team_id(FMT, team)}
    assert len(store.episodes(FMT)) == 2
    assert Model(store.root, FMT).revision == model.revision
    store.close()
