import copy

from ml.continuation import boosts, variable_power
from ml.strategy import hypotheses,prepare,component,simulate
from test_offline_strategy import context


def test_setup_effects_match_public_move_rules_and_preserve_observations():
    mon={'boosts':{'spa':5},'ability':'Simple'};before=copy.deepcopy(mon)
    assert boosts(mon,'quiverdance')=={'spa':6,'spd':2,'spe':2}
    assert boosts({'ability':'Contrary'},'calmmind')=={'spa':-1,'spd':-1}
    assert boosts({'item':'White Herb'},'shellsmash')=={'atk':2,'spa':2,'spe':2,'def':0,'spd':0}
    assert mon==before


def test_hp_power_changes_at_execution_and_leaves_metadata_unmodified():
    data={'basePower':150,'category':'Special'}
    assert variable_power('waterspout',data,{'condition':'50/100'})['basePower']==75
    assert variable_power('eruption',data,{'condition':'25/100'})['basePower']==37
    assert variable_power('dragonenergy',data,{'condition':'1/100'})['basePower']==1
    assert data['basePower']==150


def test_visible_setup_and_recovery_can_be_forecast_without_private_sets():
    f,ctx=context();ctx['state'].update(_strategic_v2=True,_strategic_v4=True)
    f.dex['moves'].update(calmmind={'category':'Status','basePower':0,'target':'self'},recover={'category':'Status','basePower':0,'target':'self'})
    mon={**ctx['state']['opp_actives'][0],'moves_known':['Calm Mind','Recover'],'condition':'40/100'}
    result=hypotheses(ctx,mon,ctx['state']['my_actives'],f)
    assert {'calmmind','recover'}<={name for name,*_ in result}
    full={**mon,'condition':'100/100'}
    assert 'recover' not in {name for name,*_ in hypotheses(ctx,full,ctx['state']['my_actives'],f)}


def test_opponent_setup_costs_future_position_even_without_immediate_damage():
    f,ctx=context();ctx['feature_profile']='strategic-v4';ctx['state'].update(_strategic_v2=True,_strategic_v4=True)
    p=prepare(ctx,f);ours=[component(ctx,i,'pass',f,p) for i in range(2)];before=copy.deepcopy(ctx['state'])
    idle=simulate(ctx,ours,[('',{},None,1)],p,f)
    setup=simulate(ctx,ours,[('swordsdance',{'category':'Status','basePower':0},None,1)],p,f)
    assert setup<idle and ctx['state']==before


def test_fast_recovery_prevents_a_previously_estimated_knockout():
    f,ctx=context();ctx['state'].update(_strategic_v2=True,_strategic_v4=True)
    ctx['state']['my_actives'][0]['stats']['spa']=80
    ctx['state']['opp_actives'][0].update(condition='40/100',stats={'spe':300,'def':110})
    p=prepare(ctx,f);ours=[component(ctx,0,'move 1 1',f,p),component(ctx,1,'pass',f,p)]
    attack_only=simulate(ctx,ours,[('',{},None,1)],p,f)
    recovered=simulate(ctx,ours,[('recover',{'category':'Status','basePower':0},None,1)],p,f)
    assert recovered<attack_only


def test_known_support_only_sheet_never_gets_an_invented_setup_attack():
    from ml.continuation import attacking_moves
    f,ctx=context();mon={'species':'Enemy','moves':['Protect','Helping Hand']}
    assert attacking_moves(mon,f)==[]
