import pytest
from battle_state import hp_fraction


def test_public_colored_hp_is_not_full_health():
    assert hp_fraction('24/100y') == pytest.approx(.24)
    assert hp_fraction('6/100r par') == pytest.approx(.06)
    assert hp_fraction('91/100g') == pytest.approx(.91)
    assert hp_fraction('0 fnt') == 0


def test_legacy_features_keep_collecting_inputs_while_tactics_sees_correct_hp():
    import numpy as np
    from ml.features import Features, STATE_DIM
    f = Features()
    a = np.zeros(STATE_DIM, dtype=np.float32)
    b = np.zeros(STATE_DIM, dtype=np.float32)
    f._mon(a, 'opponent', {'condition':'24/100y'}, 'legacy')
    f._mon(b, 'opponent', {'condition':'100/100'}, 'legacy')
    assert np.array_equal(a, b)
    f._mon(b := np.zeros(STATE_DIM,dtype=np.float32), 'opponent', {'condition':'24/100y'}, 'tactics-v1')
    assert not np.array_equal(a, b)


def test_observed_boosts_and_burn_change_damage_estimate():
    from ml.tactics import damage
    from ml.features import Features
    f = Features({'pokedex': {'actor':{'types':['Normal'],'baseStats':{'atk':100}},
                               'target':{'types':['Normal'],'baseStats':{'hp':90,'def':100}}}})
    actor={'species':'Actor','condition':'100/100','stats':{'atk':150}}
    target={'species':'Target','condition':'100/100'}
    move={'category':'Physical','type':'Normal','basePower':100,'accuracy':100}
    base=damage(actor,target,move,{},f)
    assert damage({**actor,'boosts':{'atk':-2}},target,move,{},f)<base
    assert damage(actor,{**target,'boosts':{'def':2}},move,{},f)<base
    assert damage({**actor,'condition':'100/100 brn'},target,move,{},f)==pytest.approx(base*.5)


def test_joint_targeting_stops_rewarding_overkill():
    from ml.features import Features
    f = Features({'pokedex':{'actor':{'types':['Normal'],'baseStats':{'atk':100}},
                             'target':{'types':['Normal'],'baseStats':{'hp':90,'def':100}}},
                  'moves':{'tackle':{'type':'Normal','category':'Physical','basePower':100,'accuracy':100,'target':'normal'}}})
    actor={'species':'Actor','condition':'100/100','stats':{'atk':200},'moves':[{'id':'tackle','move':'Tackle'}]}
    ctx={'feature_profile':'tactics-v1','state':{'my_party':[], 'my_actives':[actor,actor],
         'opp_actives':[{'slot':'p2a','species':'Target','condition':'10/100r'},
                        {'slot':'p2b','species':'Target','condition':'100/100'}]}}
    from ml.tactics import score
    assert score(ctx,'move 1 1, move 1 2',f)>score(ctx,'move 1 1, move 1 1',f)


def test_fixed_opponent_identity_is_part_of_the_paired_gate():
    from ml.promotion import paired_gate
    base=[{'seed':i,'learner_side':'p1','opponent':'self','opponent_revision':'fixed',
           'winner':'LocalOpponent','rejected_actions':0} for i in range(20)]
    candidate=[{**g,'winner':'LocalBrain','opponent_revision':'different'} for g in base]
    assert not paired_gate(base,candidate,20,.1)['passed']
    candidate=[{**g,'winner':'LocalBrain'} for g in base]
    assert paired_gate(base,candidate,20,.1)['passed']


def test_real_game_uses_explicit_frozen_opponent(tmp_path):
    import asyncio,json
    from ml.brain import Brain
    from ml.model import Model
    from ml.features import Features
    from ml.storage import Store
    from ml.simulator import play_local,BRIDGE
    built=BRIDGE.parent/'node_modules/pokemon-showdown/dist/sim/index.js'
    if not built.exists():pytest.skip('build pinned simulator')
    fmt='gen9championsvgc2026regmc'
    store=Store(tmp_path/'actor')
    brain=Brain(store,Features.cached(fmt))
    fixed=Model(tmp_path/'fixed',fmt)
    fixed.feature_profile='tactics-v1'
    fixed.save()
    team=json.loads((BRIDGE.parent/'examples/champions-rain.json').read_text())['sets']
    result=asyncio.run(play_local(brain,fmt,team,team,opponent='self',opponent_model=fixed,
                                  training=False,sample_actions=True,seed=19))
    assert result.get('winner') or result.get('tie')
    assert result['rejected_actions']==0
    assert result['opponent_revision']==fixed.revision
    assert not store.episodes(fmt)
    assert brain.model(fmt).revision!=fixed.revision
    store.close()


def test_preview_coverage_prefers_a_relevant_attacker():
    from ml.features import Features
    from ml.preview import score
    f=Features({'pokedex':{'watermon':{'types':['Water']},'firemon':{'types':['Fire']},'groundmon':{'types':['Ground']}},
                'typechart':{'Ground':{'damageTaken':{'Water':1,'Fire':0}},'Water':{'damageTaken':{'Ground':0}},'Fire':{'damageTaken':{'Ground':1}}},
                'moves':{'watermove':{'basePower':100,'type':'Water','accuracy':100,'category':'Special'},
                         'firemove':{'basePower':100,'type':'Fire','accuracy':100,'category':'Special'}}})
    ctx={'feature_profile':'preview-v1','state':{'my_party':[{'species':'Watermon','moves':['watermove']},
                                                          {'species':'Firemon','moves':['firemove']}],
                                              'opp_preview':['Groundmon']}}
    assert score(ctx,'team 1',f)>score(ctx,'team 2',f)
    assert score({**ctx,'state':{**ctx['state'],'opp_preview':[]}},'team 1',f)==0


def test_preview_profile_preserves_turn_action_encoding():
    import numpy as np
    from ml.features import Features
    from test_offline_learning import context
    ctx=context()
    f=Features()
    old=f.action(ctx,'move 1')
    new=f.action({**ctx,'feature_profile':'preview-v1'},'move 1')
    assert np.array_equal(old,new)


def test_bounded_preview_preserves_differences_that_clipping_erases():
    from ml.features import Features
    f=Features({'pokedex':{'watermon':{'types':['Water']},'firemon':{'types':['Fire']},'groundmon':{'types':['Ground']}},
                'typechart':{'Ground':{'damageTaken':{'Water':1}},'Water':{'damageTaken':{}},'Fire':{'damageTaken':{}}},
                'moves':{'watermove':{'basePower':300,'type':'Water','accuracy':100,'category':'Special'},
                         'firemove':{'basePower':300,'type':'Fire','accuracy':100,'category':'Special'}}})
    party=[{'species':'Watermon','moves':['watermove']},{'species':'Firemon','moves':['firemove']}]
    ctx={'feature_profile':'preview-v1','state':{'my_party':party,'opp_preview':['Groundmon']}}
    assert f.action(ctx,'team 1')[-1]==f.action(ctx,'team 2')[-1]==4
    ctx['feature_profile']='preview-v2'
    assert 0<f.action(ctx,'team 2')[-1]<f.action(ctx,'team 1')[-1]<4
