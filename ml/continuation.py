"""Bounded next-turn threat estimates for visible setup/recovery hypotheses.

No hidden spreads, opponent requests or fixture truths are read. This adds a
continuation utility to the one-turn planner, not an exact multi-turn simulator.
"""
from battle_state import hp_fraction, to_id
from ml.tactics import stat

SETUP={
    'calmmind':{'spa':1,'spd':1}, 'quiverdance':{'spa':1,'spd':1,'spe':1},
    'shellsmash':{'atk':2,'spa':2,'spe':2,'def':-1,'spd':-1},
    'coil':{'atk':1,'def':1,'accuracy':1}, 'irondefense':{'def':2},
    'swordsdance':{'atk':2}, 'nastyplot':{'spa':2},
    'bulkup':{'atk':1,'def':1}, 'dragondance':{'atk':1,'spe':1},
}
RECOVERY={'recover','slackoff','softboiled','milkdrink','healorder'}


def boosts(mon, name):
    result=dict(mon.get('boosts',{}));ability=to_id(mon.get('ability') or mon.get('baseAbility'))
    factor=-1 if ability=='contrary' else 2 if ability=='simple' else 1
    for key,delta in SETUP[name].items():result[key]=max(-6,min(6,result.get(key,0)+factor*delta))
    if to_id(mon.get('item'))=='whiteherb' and any(value<0 for value in result.values()):
        result={key:max(0,value) for key,value in result.items()}
    return result


def attacking_moves(mon, features):
    names=mon.get('moves') or mon.get('moves_known') or []
    result=[]
    for move in names:
        name=to_id(move if isinstance(move,str) else move.get('id',move.get('move','')))
        data=features.dex.get('moves',{}).get(name,{})
        if data.get('basePower'):result.append((name,data))
    if not result and not mon.get('moves'):
        # A partial public move history does not prove a support-only set.
        category='Physical' if stat(mon,'atk',features)>=stat(mon,'spa',features) else 'Special'
        result=[('estimated'+kind,{'type':kind,'category':category,'basePower':90,'accuracy':100})
                for kind in features.species(mon.get('species','')).get('types',['Normal'])]
    return result


def potential(mon, targets, state, features, side, hit, speed):
    alive=[target for target in targets if hp_fraction(target.get('condition'))>0]
    if not alive:return 0.0
    moves=attacking_moves(mon,features);other='theirs' if side=='mine' else 'mine'
    pressure=sum(min(1.5,max((hit(mon,target,name,data,state,features,side) for name,data in moves),default=0)) for target in alive)/len(alive)
    incoming=sum(min(1.5,max((hit(target,mon,name,data,state,features,other) for name,data in attacking_moves(target,features)),default=0)) for target in alive)/len(alive)
    room=any(to_id(value).removeprefix('move')=='trickroom' for value in state.get('field',[]))
    own_speed=speed(mon,side,state,features)
    initiative=sum((own_speed<speed(target,other,state,features)) if room else (own_speed>speed(target,other,state,features)) for target in alive)/len(alive)
    return .6*pressure-.2*incoming+.2*initiative


def variable_power(name, data, mon):
    if name not in ('waterspout','eruption','dragonenergy'):return data
    return {**data,'basePower':max(1,int(150*hp_fraction(mon.get('condition'))))}
