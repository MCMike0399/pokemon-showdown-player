import copy
import asyncio
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

FMT = 'gen9championsvgc2026regmc'
BUILT = Path(__file__).resolve().parents[1] / 'node_modules/pokemon-showdown/dist/sim/index.js'
pytestmark = pytest.mark.skipif(not BUILT.exists(), reason='run npm install && npm run simulator:build')


@lru_cache(maxsize=1)
def native_features():
    from ml.simulator import load_dex
    return asyncio.run(load_dex(FMT))


def context():
    f = native_features()
    # Explicit own sets are fixtures, not attributed tournament spreads.
    sets = [{'species': 'Garchomp', 'item': 'Garchompite Z', 'nature': 'Modest',
             'ability': 'Rough Skin', 'evs': {'spa': 32, 'spe': 32, 'hp': 2},
             'moves': ['Dragon Pulse', 'Earth Power', 'Power Gem', 'Protect']},
            {'species': 'Golisopod', 'item': 'Golisopite', 'nature': 'Adamant',
             'ability': 'Emergency Exit', 'evs': {'atk': 32, 'spe': 32, 'hp': 2},
             'moves': ['Leech Life', 'Drill Run', 'Iron Head', 'Protect']}]
    party = [{'species': 'Garchomp', 'item': 'Garchompite Z', 'ability': 'Rough Skin',
              'condition': '185/185', 'stats': {'atk': 135, 'def': 115, 'spa': 145, 'spd': 105, 'spe': 154},
              'slot': 'p1a', 'moves': [{'id': n, 'target': 'normal'} for n in ('dragonpulse','earthpower','powergem','protect')]},
             {'species': 'Golisopod', 'item': 'Golisopite', 'ability': 'Emergency Exit',
              'condition': '152/152', 'stats': {'atk': 194, 'def': 160, 'spa': 72, 'spd': 110, 'spe': 92},
              'slot': 'p1b', 'moves': [{'id': n, 'target': 'normal'} for n in ('leechlife','drillrun','ironhead','protect')]}]
    ctx = {'format': FMT, 'feature_profile': 'opening-v1', 'team_sets': sets,
           'state': {'request_type': 'move', 'turn': 1, 'my_party': party, 'my_actives': party,
                     'opp_actives': [{'species': 'Clefable', 'condition': '100/100', 'slot': 'p2a'}],
                     'opp_preview': ['Clefable'], 'field': [], 'hazards': {}, 'history': []}}
    return f, ctx


def test_exact_champions_mega_stats_and_form_are_not_additive_approximations():
    from ml.opening import own_form
    f, ctx = context()
    original = copy.deepcopy(ctx)
    garch = own_form(ctx, ctx['state']['my_actives'][0], f, True)
    goli = own_form(ctx, ctx['state']['my_actives'][1], f, True)
    assert garch['stats'] == {'atk': 135, 'def': 105, 'spa': 212, 'spd': 105, 'spe': 203}
    assert goli['stats'] == {'atk': 222, 'def': 195, 'spa': 81, 'spd': 140, 'spe': 92}
    assert garch['ability'] == 'Levitate' and goli['ability'] == 'Tough Claws'
    assert garch['prospective_stats_exact'] and goli['prospective_stats_exact']
    assert ctx == original
    missing = {**ctx, 'team_sets': None}
    unknown = own_form(missing, ctx['state']['my_actives'][0], f, True)
    assert not unknown['prospective_stats_exact']
    assert unknown['stats'] == ctx['state']['my_actives'][0]['stats']


def test_mega_scoring_values_steel_contact_gain_and_fire_defensive_cost():
    from ml.opening import turn_score
    f, ctx = context()
    normal = turn_score(ctx, 'move 4, move 3 1', f)
    mega = turn_score(ctx, 'move 4, move 3 1 mega', f)
    assert mega > normal
    ctx['state']['opp_actives'] = [{'species': 'Volcarona', 'condition': '100/100', 'slot': 'p2a',
                                   'moves_known': ['Flamethrower']}]
    # Evolution sharply changes incoming Fire pressure; it is not a free bonus.
    from ml.opening import _threat, own_form
    enemy = ctx['state']['opp_actives'][0];goli=ctx['state']['my_actives'][1]
    assert _threat(enemy, own_form(ctx,goli,f,True),ctx,f) > _threat(enemy,goli,ctx,f)


def test_joint_mega_prior_is_symmetric_when_active_slots_swap():
    from ml.opening import turn_score
    f, ctx = context()
    score = turn_score(ctx, 'move 4, move 3 1 mega', f)
    other = copy.deepcopy(ctx);other['state']['my_actives'].reverse()
    assert turn_score(other, 'move 3 1 mega, move 4', f) == pytest.approx(score)


def test_charge_release_and_helping_hand_use_the_joint_action():
    from ml.opening import turn_score
    f, ctx = context()
    actor = {'species': 'Archaludon', 'slot': 'p1a', 'condition': '167/167', 'stats': {'spa': 177,'spe':150},
             'moves': [{'id':'electroshot','target':'normal'},{'id':'protect','target':'self'}]}
    helper = {'species':'Farigiraf','slot':'p1b','condition':'227/227','stats':{'spa':130,'spe':80},
              'moves':[{'id':'helpinghand','target':'adjacentAlly'},{'id':'protect','target':'self'}]}
    ctx['state']['my_actives']=[actor,helper];ctx['state']['weather']='RainDance'
    helped=turn_score(ctx,'move 1 1, move 1 -1',f)
    assert helped>turn_score(ctx,'move 1 1, move 2',f)
    ctx['state']['weather']=None
    charge=turn_score(ctx,'move 1 1, move 1 -1',f)
    actor['moves'][0]['target']='scripted'
    assert turn_score(ctx,'move 1, move 1 -1',f)>charge


def test_preview_prior_uses_matchup_and_ignores_bench_order():
    from ml.opening import preview_score
    f, ctx = context()
    ctx['state']['my_party'] += [
        {'species':'Politoed','ability':'Drizzle','moves':['Muddy Water','Weather Ball'],'condition':'167/167','stats':{'spa':156,'spe':122}},
        {'species':'Archaludon','ability':'Stamina','moves':['Electro Shot','Dragon Pulse'],'condition':'167/167','stats':{'spa':177,'spe':150}},
        {'species':'Farigiraf','ability':'Armor Tail','moves':['Psychic','Helping Hand'],'condition':'227/227','stats':{'spa':130,'spe':80}},
        {'species':'Incineroar','ability':'Intimidate','moves':['Flare Blitz','Fake Out'],'condition':'172/172','stats':{'atk':183,'spe':112}}]
    ctx['state']['opp_preview']=['Primarina','Milotic','Pelipper','Rillaboom','Incineroar','Basculegion']
    rain=preview_score(ctx,'team 3,4,1,2',f)
    assert rain==pytest.approx(preview_score(ctx,'team 3,4,2,1',f))
    assert rain>preview_score(ctx,'team 5,4,1,2',f)
    other=copy.deepcopy(ctx);other.pop('_opening_preview');other['state']['opp_preview']=['Kingambit','Gholdengo','Metagross','Excadrill','Scizor','Aggron']
    assert preview_score(other,'team 3,4,1,2',f)!=pytest.approx(rain)


def test_new_profile_keeps_legacy_encodings_and_training_separate(tmp_path):
    from ml.model import Model
    f,ctx=context();ctx['choices']=['move 4, move 3 1','move 4, move 3 1 mega']
    old={**ctx,'feature_profile':'legacy'}
    before=f.encode(old)
    f.encode(ctx)
    after=f.encode(old)
    assert all(np.array_equal(a,b) for a,b in zip(before,after))
    mega_only={**ctx,'feature_profile':'mega-v1'}
    assert np.array_equal(f.action(old,ctx['choices'][0]),f.action(mega_only,ctx['choices'][0]))
    assert np.array_equal(f.state(old),f.state(mega_only))
    m=Model(tmp_path,FMT);m.feature_profile='opening-v1';m.save()
    assert Model(tmp_path,FMT).feature_profile=='opening-v1'
    assert not m.train([{'format':FMT,'status':'complete','revision':m.revision,'on_policy':True,
                        'feature_profile':'legacy'}])['trained']


def test_matchup_migration_preserves_logits_and_old_checkpoint_loads(tmp_path):
    import torch
    from ml.model import Model
    f,ctx=context();ctx['choices']=['move 4, move 3 1','move 4, move 3 1 mega']
    state,actions=f.encode(ctx)
    m=Model(tmp_path,FMT)
    # Exercise an artifact without new metadata, as well as a newly saved one.
    legacy=torch.load(m.path,weights_only=True);legacy.pop('architecture');legacy.pop('preview_training_weight')
    torch.save(legacy,m.path)
    m=Model(tmp_path,FMT);before=m.predict(state,actions)
    old_revision=m.revision;m.enable_matchups()
    assert m.revision!=old_revision and m.architecture=='matchup-v1'
    assert m.predict(state,actions)==before
    m.preview_training_weight=4;m.save();reloaded=Model(tmp_path,FMT)
    assert reloaded.architecture=='matchup-v1' and reloaded.preview_training_weight==4
    assert reloaded.predict(state,actions)==before
    # A learned residual can change relative actions as the opposing state changes.
    with torch.no_grad():
        reloaded.net.matchup_score.weight.fill_(.1)
    changed=state.copy();changed[:12]+=2
    a=np.array(reloaded.predict(state,actions)['probabilities'])
    b=np.array(reloaded.predict(changed,actions)['probabilities'])
    assert np.max(np.abs(a-b))>1e-6


def test_frozen_inference_reader_never_initializes_or_writes_schema(tmp_path):
    import sqlite3
    from ml.storage import Store
    writer=Store(tmp_path)
    writer.db.execute('BEGIN IMMEDIATE')
    writer.db.execute("INSERT INTO feed_state VALUES ('uncommitted','now','{}')")
    reader=Store.read_only(tmp_path)
    assert reader.db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]==0
    assert reader.db.execute('SELECT COUNT(*) FROM feed_state').fetchone()[0]==0
    with pytest.raises(sqlite3.OperationalError,match='readonly'):
        reader.db.execute("INSERT INTO feed_state VALUES ('x','now','{}')")
    reader.close();writer.db.rollback();writer.close()
    with pytest.raises(sqlite3.OperationalError):
        Store.read_only(tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def test_focused_profile_preserves_nonmega_turns_and_canonicalizes_bench():
    f,ctx=context();legacy={**ctx,'feature_profile':'legacy'};focused={**ctx,'feature_profile':'opening-v2'}
    command='move 4, move 3 1'
    assert np.array_equal(f.state(legacy),f.state(focused))
    assert np.array_equal(f.action(legacy,command),f.action(focused,command))
    assert not np.array_equal(f.action(legacy,command+' mega'),f.action(focused,command+' mega'))
    ctx['state']['my_party']+=copy.deepcopy(ctx['state']['my_party'])
    a=f.action(focused,'team 1,2,3,4');b=f.action(focused,'team 1,2,4,3')
    assert np.array_equal(a,b)
