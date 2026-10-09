import copy

import pytest

from harness import pack_team
from ml.teams import team_id
from scripts.audit_match_collection import audit_episode,disclosed_sheet


def episode(source='ladder'):
    return {'id':'fixture','status':'complete','source':source,'side':'p1','format':'gen9championsvgc2026regmc',
        'team_sets':[{'species':'Politoed','nature':'Modest','evs':{'hp':32,'spa':32,'spe':2},'moves':['Weather Ball'],'ability':'Drizzle','item':'Mystic Water'}],
        'steps':[{'snapshot':{'request':{'side':{'pokemon':[{'ident':'p1: Politoed','details':'Politoed, L50','condition':'167/167','stats':{'spa':156,'spe':122}}]}}}}]}


def test_own_declared_nature_and_stat_points_are_preserved_without_fabricated_ivs():
    e=episode();original=copy.deepcopy(e);r=audit_episode(e,{'champions':True,'level_clause_mod':False},[])
    assert r['mechanics']['ev_field_units']=='stat_points' and r['mechanics']['ivs_affect_stats'] is False
    assert r['own_submitted_sets'][0]['nature']=={'value':'Modest','provenance':'submitted-own-team'}
    assert r['own_submitted_sets'][0]['ivs']=={'value':None,'provenance':'unknown'}
    assert r['own_observed_stats'][0]['stats_as_requested']['spa']==156
    assert r['own_observed_stats'][0]['observed_max_hp']==167
    assert e==original


def test_open_sheet_blanks_do_not_become_zero_evs_or_perfect_ivs():
    packed=pack_team([{'species':'Milotic','nature':'Bold','ability':'Competitive','moves':['Hypnosis'],'level':50}])
    r=disclosed_sheet(packed)[0]
    assert r['nature']['value']=='Bold'
    assert r['evs']['value'] is None and r['ivs']['value'] is None
    assert r['level']['value']==50


def test_explicit_public_spreads_include_zero_values_and_default_only_inside_transmitted_field():
    packed=pack_team([{'species':'Milotic','nature':'Bold','evs':{'hp':32,'def':32,'spe':0},'ivs':{'atk':0,'spe':0},'moves':['Hypnosis']}])
    r=disclosed_sheet(packed)[0]
    assert r['evs']['value']=={'hp':32,'atk':0,'def':32,'spa':0,'spd':0,'spe':0}
    assert r['ivs']['value']=={'hp':31,'atk':0,'def':31,'spa':31,'spd':31,'spe':0}


def test_only_opponents_actual_public_sheet_is_added():
    e=episode();packed=pack_team([{'species':'Milotic','nature':'Bold','moves':['Hypnosis']}])
    r=audit_episode(e,{'champions':True},['|showteam|p1|'+packed,'|turn|1','|showteam|p2|'+packed])
    assert len(r['opponent_public_sheets'])==1 and r['opponent_public_sheets'][0]['first_public_turn']==1
    assert r['opponent_public_sheets'][0]['pokemon'][0]['nature']['value']=='Bold'


def test_fixture_truth_is_postgame_only_and_never_applies_to_live_opponent():
    sets=[{'species':'Milotic','nature':'Bold','evs':{'def':32},'moves':['Hypnosis']}]
    e=episode('local');e['simulation']={'opponent_team':team_id(e['format'],sets)}
    plan={'format':e['format'],'opponents':[sets]}
    r=audit_episode(e,{'champions':True},[],plan)
    assert r['offline_fixture_diagnostics']['scope']=='postgame-diagnostics-only'
    assert r['offline_fixture_diagnostics']['opponent_declared_sets']==sets
    e['source']='ladder';assert 'offline_fixture_diagnostics' not in audit_episode(e,{'champions':True},[],plan)
    e['status']='pending'
    with pytest.raises(ValueError,match='terminal'):audit_episode(e,{'champions':True},[],plan)


def test_legacy_missing_fields_stay_unknown_and_stat_transitions_are_compact():
    e=episode();e['team_sets']=None;e['steps']+=copy.deepcopy(e['steps'])
    r=audit_episode(e,{},[])
    assert r['own_submitted_sets']==[] and r['mechanics']['ivs_affect_stats'] is None
    assert len(r['own_observed_stats'])==1 and r['coverage']['private_request_decisions']==2
