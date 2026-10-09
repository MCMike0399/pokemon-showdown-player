import pytest

from battle_state import observe, legal_choices
from ml.features import Features

PROFILE = 'weather-v1'


def weather_context(weather='RainDance'):
    req = {'rqid': 1, 'side': {'id': 'p1', 'pokemon': [
        {'ident': 'p1: Politoed', 'details': 'Politoed, L50', 'condition': '100/100', 'active': True}]},
        'active': [{'moves': [{'move': 'Weather Ball', 'id': 'weatherball', 'target': 'normal', 'pp': 10}]}]}
    log = ['|switch|p2a: Gengar|Gengar, L50|100/100'] + (['|-weather|' + weather] if weather else [])
    return {'request': req, 'state': observe(req, log), 'choices': legal_choices(req), 'feature_profile': PROFILE}


def features():
    return Features({'pokedex': {'politoed': {'types': ['Water']}, 'gengar': {'types': ['Ghost', 'Poison']}},
                     'moves': {'weatherball': {'type': 'Normal', 'basePower': 50, 'accuracy': 100, 'category': 'Special'}},
                     'typechart': {'Ghost': {'damageTaken': {'Normal': 3}}}})


def test_rain_weather_ball_is_water_100_power_in_candidate_features():
    f = features()
    action = f.action(weather_context(), 'move 1')
    assert action[-16] == 1.0
    assert action[-1] == pytest.approx(1.5)
    assert f.dex['moves']['weatherball']['type'] == 'Normal'
    assert f.dex['moves']['weatherball']['basePower'] == 50


def test_legacy_and_unknown_weather_keep_original_encoding():
    f = features()
    ctx = weather_context()
    ctx['feature_profile'] = 'legacy'
    assert f.action(ctx, 'move 1')[-1] == 0
    ctx['feature_profile'] = PROFILE
    ctx['state']['weather'] = None
    assert f.action(ctx, 'move 1')[-16] == .5
    assert f.action(ctx, 'move 1')[-1] == 0


def test_known_weather_suppression_disables_candidate_weather_changes():
    ctx = weather_context()
    ctx['state']['opp_actives'][0]['ability'] = 'Cloud Nine'
    assert features().action(ctx, 'move 1')[-1] == 0


@pytest.mark.parametrize('weather,type_name', [('RainDance', 'Water'), ('SunnyDay', 'Fire'),
                                               ('Sandstorm', 'Rock'), ('Snowscape', 'Ice')])
def test_weather_ball_uses_observable_weather_type(weather, type_name):
    f = features()
    ctx = weather_context(weather)
    data = f.move_data(ctx, 'weatherball', f.dex['moves']['weatherball'])
    assert data['type'] == type_name and data['basePower'] == 100


def test_hurricane_and_thunder_weather_accuracy():
    f = features()
    for name in ('hurricane', 'thunder'):
        data = {'accuracy': 70, 'basePower': 110, 'type': 'Flying' if name == 'hurricane' else 'Electric'}
        assert f.move_data(weather_context(), name, data)['accuracy'] is True
        assert f.move_data(weather_context('SunnyDay'), name, data)['accuracy'] == 50
        assert f.move_data(weather_context(None), name, data)['accuracy'] == 70


def test_known_umbrella_on_affected_mon_suppresses_rain_but_not_sand():
    f = features()
    ctx = weather_context()
    ctx['state']['my_actives'][0]['item'] = 'Utility Umbrella'
    assert f.action(ctx, 'move 1')[-1] == 0
    ctx['state']['weather'] = 'Sandstorm'
    assert f.action(ctx, 'move 1')[-16] == 1
    target = dict(ctx['state']['opp_actives'][0], item='Utility Umbrella')
    ctx['state']['weather'] = 'RainDance'
    assert f.move_data(ctx, 'hurricane', {'accuracy': 70}, targets=[target])['accuracy'] == 70


def test_candidate_profile_is_checkpointed_recorded_and_isolated_from_old_data(tmp_path):
    from ml.brain import Brain
    from ml.model import Model
    from ml.storage import Store
    store = Store(tmp_path)
    brain = Brain(store, features())
    model = brain.model('gen9championsvgc2026regmc')
    ctx = weather_context()
    ctx.update(room='local-weather', format=model.fmt, source='local')
    # Legacy checkpoint forces the legacy encoder even if a caller adds a flag.
    decision = brain.decide(ctx, explore=True, record=True)
    episode = next(iter(brain.pending.values()))
    assert episode['feature_profile'] == 'legacy'
    assert episode['steps'][0]['actions'][0][-16] == .5
    brain.finish(ctx['room'], {'winner': 'Player'}, 'Player')
    model.feature_profile = PROFILE
    model.save()
    # Matching revision is insufficient when the collecting encoder differs.
    assert not model.train(store.episodes(model.fmt))['trained']
    assert len(store.episodes(model.fmt)) == 1
    assert Model(tmp_path, model.fmt).feature_profile == PROFILE
    ctx['room'] = 'local-weather-new'
    brain.decide(ctx, explore=True, record=True)
    episode = next(iter(brain.pending.values()))
    assert episode['feature_profile'] == PROFILE
    assert episode['steps'][0]['snapshot']['feature_profile'] == PROFILE
    assert episode['steps'][0]['actions'][0][-16] == 1
    store.close()


def test_profile_cannot_change_inside_an_existing_trajectory(tmp_path):
    from ml.brain import Brain
    from ml.storage import Store
    store = Store(tmp_path)
    brain = Brain(store, features())
    model = brain.model('gen9championsvgc2026regmc')
    ctx = weather_context()
    ctx.update(room='local-weather', format=model.fmt, source='local')
    brain.decide(ctx, explore=True, record=True)
    model.feature_profile = PROFILE
    with pytest.raises(ValueError, match='feature profile'):
        brain.decide(ctx, explore=True, record=True)
    assert len(next(iter(brain.pending.values()))['steps']) == 1
    store.close()
