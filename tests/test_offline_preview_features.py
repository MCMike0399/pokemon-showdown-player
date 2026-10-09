import copy

import numpy as np

from ml.preview_features import public_summary, plan_summary
from test_offline_strategy import context


def test_public_preview_descriptors_are_order_independent_and_do_not_guess_sets():
    f,ctx=context()
    a=public_summary(['Enemy','Attacker'],f)
    assert a==public_summary(['Attacker','Enemy'],f)
    assert a['opponent/type/Water']>0 and a['opponent/type/Electric']>0
    assert not any(word in key for key in a for word in ('item','ability','move','iv','ev'))


def test_new_preview_state_exposes_public_type_and_base_speed_changes():
    f,ctx=context();ctx['choices']=['team 1,2'];ctx['feature_profile']='strategic-v3'
    first=f.state(ctx)
    other=copy.deepcopy(ctx);other['state']['opp_preview']=['Attacker']
    assert not np.array_equal(first,f.state(other))
    # Descriptors are explicitly opt-in; prior checkpoints retain their arrays.
    legacy=copy.deepcopy(ctx);legacy['feature_profile']='strategic-v2'
    assert not np.array_equal(first,f.state(legacy))


def test_plan_expresses_rain_dependency_without_forcing_rain_or_mutating_state():
    f,ctx=context();ctx['state']['my_party'][1]['ability']='Drizzle'
    before=copy.deepcopy(ctx['state'])
    summary=plan_summary(ctx,'team 1,2',f)
    assert summary['plan/rain_brought']==1 and summary['plan/rain_dependencies']>0
    assert ctx['state']==before
    dry=copy.deepcopy(ctx);dry.pop('_v3_preview');dry['state']['my_party'][1]['ability']=''
    assert plan_summary(dry,'team 1,2',f)['plan/rain_brought']==0


def test_counter_structure_changes_when_visible_opposing_typing_changes():
    f,ctx=context()
    wet=plan_summary(ctx,'team 1,2',f)
    other=copy.deepcopy(ctx);other.pop('_v3_preview');other['state']['opp_preview']=['Attacker']
    electric=plan_summary(other,'team 1,2',f)
    assert 'plan/base/coverage/type/Water' in wet
    assert 'plan/base/coverage/type/Electric' in electric
    assert wet['plan/base/coverage']!=electric['plan/base/coverage']


def test_new_action_features_preserve_bench_order_equivalence():
    f,ctx=context();ctx['feature_profile']='strategic-v3'
    ctx['state']['my_party']+=copy.deepcopy(ctx['state']['my_party'])
    first=f.action(ctx,'team 1,2,3,4')
    assert np.array_equal(first,f.action(ctx,'team 1,2,4,3'))


def test_exact_one_mega_mode_never_adds_two_evolutions():
    f,ctx=context();ctx['state']['my_party'][0]['item']='Alphaite';ctx['state']['my_party'][1]['item']='Betite'
    for base,item in [('Attacker','Alphaite'),('Helper','Betite')]:
        f.dex['pokedex'][(base+'Mega').lower()]={'name':base+'Mega','baseSpecies':base,'requiredItem':item,'types':['Electric'],'baseStats':{'atk':160,'spa':160,'spe':180},'abilities':{'0':'Tough Claws'}}
    summary=plan_summary(ctx,'team 1,2',f)
    assert summary['plan/mega_options']==1
    assert summary['plan/single_mega/coverage']>=summary['plan/base/coverage']
    assert ctx['state']['my_party'][0]['species']=='Attacker'
