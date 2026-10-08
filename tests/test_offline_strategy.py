import copy
import pytest

from ml.features import Features
from ml.strategy import hit, speed, turn_score


def context():
    dex = {'pokedex': {name: {'types': [kind], 'baseStats': {'hp': 80, 'atk': 100, 'def': 100, 'spa': 100, 'spd': 100, 'spe': spe}}
           for name, kind, spe in [('attacker', 'Electric', 120), ('helper', 'Water', 70), ('enemy', 'Water', 90)]},
           'moves': {'thunderbolt': {'type': 'Electric', 'category': 'Special', 'basePower': 90, 'accuracy': 100, 'target': 'normal'},
                     'electroshot': {'type': 'Electric', 'category': 'Special', 'basePower': 130, 'accuracy': 100, 'target': 'normal'},
                     'protect': {'category': 'Status', 'target': 'self', 'priority': 4},
                     'helpinghand': {'category': 'Status', 'target': 'adjacentAlly', 'priority': 5},
                     'tackle': {'type': 'Normal', 'category': 'Physical', 'basePower': 90, 'accuracy': 100, 'target': 'normal'},
                     'fakeout': {'type': 'Normal', 'category': 'Physical', 'basePower': 40, 'accuracy': 100, 'target': 'normal', 'priority': 3}},
           'typechart': {'Water': {'damageTaken': {'Electric': 1}}, 'Electric': {'damageTaken': {}}}}
    f = Features(dex)
    own = [{'species': 'Attacker', 'condition': '170/170', 'slot': 'p1a', 'stats': {'spa': 160, 'spe': 150, 'def': 110},
            'moves': [{'id': 'thunderbolt', 'target': 'normal'}, {'id': 'electroshot', 'target': 'normal'}, {'id': 'protect', 'target': 'self'}]},
           {'species': 'Helper', 'condition': '170/170', 'slot': 'p1b', 'stats': {'atk': 90, 'spe': 80, 'def': 110},
            'moves': [{'id': 'helpinghand', 'target': 'adjacentAlly'}, {'id': 'protect', 'target': 'self'}, {'id': 'tackle', 'target': 'normal'}]}]
    enemy = [{'species': 'Enemy', 'condition': '100/100', 'slot': 'p2a', 'moves_known': ['Tackle']}]
    state = {'my_actives': own, 'my_party': own, 'opp_actives': enemy, 'opp_preview': ['Enemy'],
             'opp_team_sheet': [], 'weather': None, 'field': [], 'hazards': {}, 'turn': 1, 'history': [], 'request_type': 'move'}
    return f, {'format': 'gen9championsvgc2026regmc', 'feature_profile': 'strategic-v1', 'state': state, 'choices': []}


def test_partner_weather_switch_enables_electroshot_this_turn():
    f, ctx = context()
    setter = {'species': 'Helper', 'ability': 'Drizzle', 'condition': '170/170', 'moves': ['Tackle']}
    ctx['state']['my_party'] = ctx['state']['my_party'] + [setter]
    dry = turn_score(copy.deepcopy(ctx), 'move 2 1, move 2', f)
    entry = turn_score(copy.deepcopy(ctx), 'move 2 1, switch 3', f)
    assert entry > dry + .5


def test_helping_hand_is_only_valuable_when_partner_attack_executes():
    f, ctx = context()
    assert turn_score(copy.deepcopy(ctx), 'move 1 1, move 1 -1', f) > turn_score(copy.deepcopy(ctx), 'move 1 1, move 2', f)
    assert turn_score(copy.deepcopy(ctx), 'move 3, move 1 -1', f) < turn_score(copy.deepcopy(ctx), 'move 1 1, move 1 -1', f)


def test_damage_denominator_is_preserved_for_double_target_attacks():
    from ml.strategy import prepare, component, simulate
    f, ctx = context()
    ctx['state']['opp_actives'] *= 2
    ctx['state']['opp_actives'] = copy.deepcopy(ctx['state']['opp_actives'])
    ctx['state']['opp_actives'][1]['slot'] = 'p2b'
    prepared = prepare(ctx, f)
    ours = [component(ctx, i, 'pass', f, prepared) for i in range(2)]
    move = f.dex['moves']['tackle']
    one = simulate(ctx, ours, [('tackle', move, 0, 1), ('', {}, None, 1)], prepared, f)
    two = simulate(ctx, ours, [('tackle', move, 0, 1), ('tackle', move, 0, 1)], prepared, f)
    assert two < one * 1.7


def test_priority_blocking_terrain_depends_on_grounding():
    f, ctx = context();a = ctx['state']['my_actives'][0];t = ctx['state']['opp_actives'][0]
    ctx['state']['field'] = ['move: Psychic Terrain']
    assert hit(a, t, 'fakeout', f.dex['moves']['fakeout'], ctx['state'], f, 'mine') == 0
    flying = {**t, 'ability': 'Levitate'}
    assert hit(a, flying, 'fakeout', f.dex['moves']['fakeout'], ctx['state'], f, 'mine') > 0


def test_speed_combines_visible_weather_tailwind_paralysis_and_scarf():
    f, ctx = context();m = {**ctx['state']['my_actives'][0], 'ability': 'Swift Swim', 'item': 'Choice Scarf', 'condition': '170/170 par'}
    ctx['state'].update(weather='RainDance', hazards={'mine': ['Tailwind']})
    assert speed(m, 'mine', ctx['state'], f) == 450


def test_sequence_denies_attack_after_fast_knockout_and_does_not_mutate_observations():
    from ml.strategy import prepare, component, simulate
    f, ctx = context();ctx['state']['opp_actives'][0]['condition'] = '10/100'
    before = copy.deepcopy(ctx['state'])
    prepared = prepare(ctx, f)
    ours = [component(ctx, 0, 'move 1 1', f, prepared), component(ctx, 1, 'pass', f, prepared)]
    fast = simulate(ctx, ours, [('tackle', f.dex['moves']['tackle'], 0, 1)], prepared, f)
    ctx['state']['my_actives'][0]['stats']['spe'] = 1
    slow = simulate(ctx, ours, [('tackle', f.dex['moves']['tackle'], 0, 1)], prepared, f)
    assert fast > slow
    ctx['state']['my_actives'][0]['stats']['spe'] = 150
    assert ctx['state'] == before


def test_immutable_behavior_evidence_survives_checkpoint_reload(tmp_path):
    from ml.model import Model
    m = Model(tmp_path, 'gen9championsvgc2026regmc')
    m.feature_profile = 'strategic-v1';m.strategy_knowledge = {'version': 1, 'moves': {'enemy': {'protect': 20}}};m.save()
    restored = Model(tmp_path, m.fmt)
    assert restored.strategy_knowledge == m.strategy_knowledge and restored.feature_profile == 'strategic-v1'


def test_postgame_only_labels_executed_opponent_moves_and_distinguishes_missing_execution():
    from ml.postgame import review, knowledge_from_reviews
    f, ctx = context()
    episode = {'side': 'p1', 'outcome': -1, 'steps': [{'choice': 'move 1 1, move 2',
        'snapshot': {'observation': ctx['state']}}]}
    log = ['|switch|p1a: A|Attacker, L50|170/170', '|switch|p2a: E|Enemy, L50|100/100',
           '|turn|1', '|move|p2a: E|Tackle|p1a: A', '|faint|p1a: A', '|win|Opponent']
    r = review(episode, log)
    assert r['opponent_executed_moves'] == {'enemy': {'tackle': 1}}
    assert len(r['components_without_matching_execution']) == 2
    assert knowledge_from_reviews([r], ctx['format'])['executed_samples'] == 1


def test_authoritative_support_only_sheet_never_creates_a_fake_attack():
    from ml.strategy import hypotheses
    f, ctx = context();mon = ctx['state']['opp_actives'][0]
    ctx['state']['opp_team_sheet'] = [{'species': 'Enemy', 'moves': ['Protect', 'Helping Hand']}]
    result = hypotheses(ctx, mon, ctx['state']['my_actives'], f)
    assert result and all(not data.get('basePower') for _, data, _, _ in result)


def test_new_postgame_behavior_is_folded_once_and_does_not_rewrite_live_checkpoint(tmp_path):
    from ml.postgame import refresh_knowledge
    from ml.model import Model
    from ml.storage import Store
    store = Store(tmp_path)
    model = Model(tmp_path / 'candidate', 'gen9championsvgc2026regmc')
    before = model.path.read_bytes()
    episode = {'id': 'new-game', 'format': model.fmt, 'revision': model.revision, 'schema': 1,
               'source': 'ladder', 'status': 'complete', 'created': '2026-10-08T00:00:00Z',
               'steps': [], 'postgame': {'version': 1, 'opponent_executed_moves': {'enemy': {'tackle': 2}}, 'opponent_leads': ['enemy']}}
    store.save_episode(episode)
    assert refresh_knowledge(store, model)['new_ladder_games'] == 1
    assert refresh_knowledge(store, model)['new_ladder_games'] == 0
    assert model.strategy_knowledge['moves']['enemy']['tackle'] == 2
    assert model.path.read_bytes() == before
    store.close()


def test_speed_tie_scenarios_average_both_initiative_orders():
    from ml.strategy import prepare, component, simulate
    f, ctx = context();ctx['state']['my_actives'][0]['stats']['spe'] = 125
    ctx['state']['opp_actives'][0]['condition'] = '10/100'
    prepared = prepare(ctx, f)
    ours = [component(ctx, 0, 'move 1 1', f, prepared), component(ctx, 1, 'pass', f, prepared)]
    rows = [('tackle', f.dex['moves']['tackle'], 0, 1)]
    assert simulate(ctx, ours, rows, prepared, f) > simulate(ctx, ours, rows, prepared, f, True)


def test_locked_release_never_hits_both_opponents_as_a_spread_move():
    from ml.strategy import prepare, component, simulate
    f, ctx = context();ctx['state']['my_actives'][0]['moves'][1]['target'] = 'scripted'
    ctx['state']['opp_actives'] += [{'species': 'Enemy', 'condition': '100/100', 'slot': 'p2b'}]
    prepared = prepare(ctx, f)
    ours = [component(ctx, 0, 'move 2', f, prepared), component(ctx, 1, 'pass', f, prepared)]
    value = simulate(ctx, ours, [('', {}, None, 1), ('', {}, None, 1)], prepared, f)
    assert value < 2  # One target's full HP plus KO reward, not two.


def test_opponent_joint_hypotheses_obey_single_mega_and_unique_switches():
    from ml.strategy import prepare
    f, ctx = context()
    f.dex['pokedex']['enemymega'] = {'name': 'Enemy-Mega', 'baseSpecies': 'Enemy', 'requiredItem': 'Enemyite',
        'isMega': True, 'abilities': {'0': 'Drought'}, 'types': ['Water'], 'baseStats': {'spe': 90}}
    a = {**ctx['state']['opp_actives'][0], 'ident': 'p2a: E', 'item': 'Enemyite'}
    b = {**a, 'slot': 'p2b', 'ident': 'p2b: F'}
    reserve = {**a, 'ident': 'p2b: Reserve', 'item': '', 'boosts': {'atk': -2}, 'volatiles': ['confusion']}
    ctx['state'].update(opp_actives=[a, b], opp_revealed=[a, b, reserve])
    prepared = prepare(ctx, f)
    for rows, _ in prepared['scenarios']:
        assert sum(bool(r[1].get('hypothesis_form')) for r in rows) <= 1
        incoming = [r[1]['hypothesis_switch'] for r in rows if r[1].get('hypothesis_switch')]
        assert len({m['ident'] for m in incoming}) == len(incoming)
        assert all(not m['boosts'] and not m['volatiles'] for m in incoming)


def test_known_used_opposing_mega_excludes_further_evolutions():
    from ml.strategy import hypotheses
    f, ctx = context()
    f.dex['pokedex']['enemymega'] = {'name': 'Enemy-Mega', 'baseSpecies': 'Enemy', 'requiredItem': 'Enemyite',
        'isMega': True, 'abilities': {'0': 'Drought'}, 'types': ['Water'], 'baseStats': {'spe': 90}}
    mon = {**ctx['state']['opp_actives'][0], 'item': 'Enemyite'}
    ctx['public_log'] = ['|-mega|p2b: Other|Other|Otherite']
    assert not any(d.get('hypothesis_form') for _, d, _, _ in hypotheses(ctx, mon, ctx['state']['my_actives'], f))


def test_mega_transition_preserves_intimidate_applied_by_partner_switch(monkeypatch):
    import ml.strategy as strategy
    f, ctx = context()
    incoming = {'species': 'Helper', 'ability': 'Intimidate', 'condition': '170/170', 'moves': ['Tackle']}
    ctx['state']['my_party'] += [incoming]
    prepared = strategy.prepare(ctx, f)
    ours = [strategy.component(ctx, 0, 'pass', f, prepared), strategy.component(ctx, 1, 'switch 3', f, prepared)]
    seen = []
    original = strategy.hit
    def capture(actor, *args, **kwargs):
        if actor.get('species') == 'Enemy-Mega':
            seen.append(actor.get('boosts', {}).get('atk'))
        return original(actor, *args, **kwargs)
    monkeypatch.setattr(strategy, 'hit', capture)
    form = {**prepared['enemy'][0], 'species': 'Enemy-Mega', 'ability': 'Drought', 'stats': {}}
    strategy.simulate(ctx, ours, [('tackle', {**f.dex['moves']['tackle'], 'hypothesis_form': form}, 0, 1)], prepared, f)
    assert seen and all(stage == -1 for stage in seen)


def test_switching_terrain_setter_blocks_priority_at_execution():
    from ml.strategy import prepare, component, simulate
    f, ctx = context()
    incoming = {'species': 'Helper', 'ability': 'Psychic Surge', 'condition': '170/170', 'moves': ['Tackle']}
    ctx['state']['my_party'] += [incoming]
    prepared = prepare(ctx, f)
    ours = [component(ctx, 0, 'pass', f, prepared), component(ctx, 1, 'switch 3', f, prepared)]
    enemy = [('fakeout', f.dex['moves']['fakeout'], 0, 1)]
    psychic = simulate(ctx, ours, enemy, prepared, f)
    incoming['ability'] = '';prepared['components'].clear()
    ours[1] = component(ctx, 1, 'switch 3', f, prepared)
    neutral = simulate(ctx, ours, enemy, prepared, f)
    assert psychic > neutral


def test_grassy_glide_priority_tracks_terrain_and_grounding():
    from ml.strategy import move_priority
    f, ctx = context();m = ctx['state']['my_actives'][0]
    d = {'priority': 0, 'category': 'Physical'}
    assert move_priority(m, 'grassyglide', d, ctx['state'], f) == 0
    ctx['state']['field'] = ['GrassyTerrain']
    assert move_priority(m, 'grassyglide', d, ctx['state'], f) == 1
    assert move_priority({**m, 'ability': 'Levitate'}, 'grassyglide', d, ctx['state'], f) == 0


def test_partner_attack_retargets_after_first_foe_faints():
    from ml.strategy import prepare, component, simulate
    f, ctx = context();ctx['state']['opp_actives'][0]['condition'] = '1/100'
    ctx['state']['opp_actives'] += [{'species': 'Enemy', 'condition': '100/100', 'slot': 'p2b'}]
    prepared = prepare(ctx, f)
    first = component(ctx, 0, 'move 1 1', f, prepared)
    passive = component(ctx, 1, 'pass', f, prepared)
    partner = component(ctx, 1, 'move 3 1', f, prepared)
    rows = [('', {}, None, 1), ('', {}, None, 1)]
    assert simulate(ctx, [first, partner], rows, prepared, f) > simulate(ctx, [first, passive], rows, prepared, f)


def test_drain_healing_changes_survival_instead_of_only_adding_damage():
    from ml.strategy import prepare, component, simulate
    f, ctx = context()
    f.dex['moves']['gigadrain'] = {**f.dex['moves']['thunderbolt'], 'type': 'Electric', 'basePower': 90}
    ctx['state']['my_actives'][0]['condition'] = '60/170'
    ctx['state']['my_actives'][0]['moves'].append({'id': 'gigadrain', 'target': 'normal'})
    prepared = prepare(ctx, f)
    partner = component(ctx, 1, 'pass', f, prepared)
    attack = component(ctx, 0, 'move 1 1', f, prepared)
    drain = component(ctx, 0, 'move 4 1', f, prepared)
    rows = [('tackle', f.dex['moves']['tackle'], 0, 1)]
    assert simulate(ctx, [drain, partner], rows, prepared, f) > simulate(ctx, [attack, partner], rows, prepared, f)


@pytest.mark.parametrize('review_fails', [False, True])
def test_brain_terminal_feedback_persists_with_immutable_completed_records(tmp_path, monkeypatch, review_fails):
    from ml.brain import Brain
    from ml.storage import Store
    import ml.postgame
    store = Store(tmp_path)
    brain = Brain(store)
    episode = {'id': 'finished', 'room': 'battle-test', 'side': 'p1', 'format': 'gen9championsvgc2026regmc',
               'revision': 'r', 'source': 'ladder', 'status': 'pending', 'steps': [], 'schema': 1}
    brain.pending[('battle-test', 'p1')] = episode;store.save_episode(episode)
    if review_fails:
        def fail(*args):
            raise RuntimeError('auxiliary failure')
        monkeypatch.setattr(ml.postgame, 'review', fail)
    result = brain.finish('battle-test', {'winner': 'Player'}, 'Player', public_log=['|win|Player'])
    assert result['recorded']
    saved = store.room_episodes('battle-test')[0]
    assert saved['status'] == 'complete' and saved['outcome'] == 1
    assert saved['postgame']['version'] == 1
    assert saved['postgame'].get('error') == ('RuntimeError' if review_fails else None)
    assert brain.finish('battle-test', {'winner': 'Other'}, 'Player')['already_recorded']
    assert store.room_episodes('battle-test')[0]['outcome'] == 1
    store.close()
