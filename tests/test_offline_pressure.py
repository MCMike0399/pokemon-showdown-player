import copy

from ml.pressure import score
from test_offline_strategy import context


def test_fixed_opponent_uses_setup_then_stops_at_large_boosts():
    f,ctx=context();f.dex['moves']['calmmind']={'category':'Status','basePower':0,'target':'self'}
    ctx['state']['my_actives'][0]['moves']=[{'id':'calmmind','target':'self'}]
    before=copy.deepcopy(ctx['state']);initial=score(ctx,'move 1, pass',f)
    assert ctx['state']==before
    ctx['state']['my_actives'][0]['boosts']={'spa':6,'spd':6}
    assert initial>score(ctx,'move 1, pass',f)+1


def test_fixed_opponent_pairs_redirection_with_its_own_setup_choice():
    f,ctx=context();f.dex['moves'].update(calmmind={'category':'Status','basePower':0,'target':'self'},followme={'category':'Status','basePower':0,'target':'self'})
    ctx['state']['my_actives'][0]['moves']=[{'id':'calmmind','target':'self'}]
    ctx['state']['my_actives'][1]['moves']=[{'id':'followme','target':'self'}]
    assert score(ctx,'move 1, move 1',f)>score(ctx,'pass, move 1',f)+2


def test_fixed_recovery_policy_values_healing_only_when_damaged():
    f,ctx=context();f.dex['moves']['recover']={'category':'Status','basePower':0,'target':'self'}
    ctx['state']['my_actives'][0]['moves']=[{'id':'recover','target':'self'}]
    full=score(ctx,'move 1, pass',f)
    ctx['state']['my_actives'][0]['condition']='50/170'
    assert score(ctx,'move 1, pass',f)>full+1
