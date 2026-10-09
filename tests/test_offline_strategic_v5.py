import copy
import pytest
from ml.mega_uncertainty import alternatives,expand,envelope,catalog
from ml.strategy import prepare as prepare_turn,speed
from ml.preview_v5 import prepare as preview, bounded
from test_offline_strategy import context


def mega_context():
    f,c=context();f.dex['pokedex']['enemymega']={'name':'Enemy-Mega','baseSpecies':'Enemy','requiredItem':'Enemyite','abilities':{'0':'Tough Claws'},'types':['Water'],'baseStats':{'hp':80,'atk':140,'def':120,'spa':100,'spd':100,'spe':180}}
    f._v5_mega_catalogs={c['format']:{'enemy':['EnemyMega']}}
    c['state'].update(_strategic_v2=True,_strategic_v4=True,_strategic_v5=True)
    return f,c


def test_unknown_item_forecasts_speed_changing_mega_without_revealing_it():
    f,c=mega_context();before=copy.deepcopy(c['state']);mon=c['state']['opp_actives'][0]
    rows=expand(c,mon,[('tackle',f.dex['moves']['tackle'],0,1)],f)
    form=next(r[1]['hypothesis_form'] for r in rows if 'hypothesis_form' in r[1])
    assert 0<sum(r[3] for r in rows if 'hypothesis_form' in r[1])<1
    assert speed(form,'theirs',c['state'],f)>150>speed(mon,'theirs',c['state'],f)
    assert form['stats']=={} and c['state']==before and 'item' not in mon


@pytest.mark.parametrize('item',['Leftovers',''])
def test_known_nonstone_or_removed_item_excludes_unknown_mega(item):
    f,c=mega_context();assert alternatives(c,{**c['state']['opp_actives'][0],'item':item},f)==([],0.0)


def test_already_used_opposing_mega_excludes_second_but_own_mega_does_not():
    f,c=mega_context();m=c['state']['opp_actives'][0]
    c['public_log']=['|-mega|p1a: Mine|Attacker|Alphaite'];assert alternatives(c,m,f)[0]
    c['public_log']=['|-mega|p2a: Foe|Enemy|Enemyite'];assert alternatives(c,m,f)==([],0.0)


def test_joint_scenarios_preserve_possible_mega_and_forbid_two():
    f,c=mega_context();other=copy.deepcopy(c['state']['opp_actives'][0]);other['slot']='p2b';c['state']['opp_actives'].append(other)
    prepared=prepare_turn(c,f)
    assert any(any(r[1].get('hypothesis_form') for r in rows) for rows,_ in prepared['scenarios'])
    assert all(sum(bool(r[1].get('hypothesis_form')) for r in rows)<=1 for rows,_ in prepared['scenarios'])
    assert sum(p for _,p in prepared['scenarios'])==pytest.approx(1)


def test_coverage_distinguishes_attack_category_against_special_bulk():
    f,c=context()
    f.dex['pokedex']['enemy']['baseStats']['def']=40;f.dex['pokedex']['enemy']['baseStats']['spd']=220
    f.dex['moves']['thunderpunch']={'type':'Electric','category':'Physical','basePower':90,'accuracy':100,'target':'normal'}
    c['state']['my_party'][0]['stats']['atk']=160;c['state']['my_party'][0]['moves']=['Thunderbolt']
    special=preview(c,f)['tables'][False][0][0]['attack'][0]
    other=copy.deepcopy(c);other.pop('_v5_preview');other['state']['my_party'][0]['moves']=['Thunder Punch']
    physical=preview(other,f)['tables'][False][0][0]['attack'][0]
    assert physical>2*special
    assert bounded(3)>bounded(2)>bounded(1)


def test_native_catalog_uses_format_species_and_item_restrictions():
    from ml.features import Features
    f=Features();forms=catalog('gen9championsvgc2026regmc',f)
    assert 'Metagross-Mega' in forms['metagross']
    assert 'Garchomp-Mega-Z' in forms['garchomp']
    # The catalog is auxiliary public format data, not injected into old dex signatures.
    assert f.dex=={}


def test_missing_item_none_never_creates_a_base_form_as_a_fake_mega():
    f,c=mega_context();mon={**c['state']['opp_actives'][0],'item':None}
    rows=expand(c,mon,[('tackle',f.dex['moves']['tackle'],0,1)],f)
    forms=[row[1]['hypothesis_form'] for row in rows if row[1].get('hypothesis_form')]
    assert forms and all(form['species']=='Enemy-Mega' for form in forms)
    assert all(row[1].get('form_provenance')=='uncertain-format-eligibility' for row in rows if row[1].get('hypothesis_form'))
