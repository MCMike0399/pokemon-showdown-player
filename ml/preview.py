"""Opponent-aware lineup prior from our exact sets and the visible species preview."""
from __future__ import annotations

from battle_state import to_id


def score(ctx, choice, features):
    party=ctx['state'].get('my_party',[])
    enemies=ctx['state'].get('opp_preview',[])
    if not enemies:return 0.0
    indices=[int(x)-1 for x in choice[5:].split(',')]
    picked=[party[i] for i in indices if 0<=i<len(party)]
    if not picked:return 0.0
    rain=any(to_id(m.get('ability') or m.get('baseAbility') or '')=='drizzle' for m in picked)
    lead_rain=any(to_id(m.get('ability') or m.get('baseAbility') or '')=='drizzle' for m in picked[:2])
    def attack(mon, enemy):
        own_types=features.species(mon.get('species','')).get('types',[])
        values=[]
        for move in mon.get('moves',[]):
            name=to_id(move if isinstance(move,str) else move.get('id',move.get('move','')))
            data=features.dex.get('moves',{}).get(name,{})
            if not data.get('basePower'):continue
            kind=data.get('type','');power=data.get('basePower',0)
            if name=='weatherball' and rain:kind,power='Water',100
            accuracy=data.get('accuracy',100)
            if rain and name in ('thunder','hurricane'):accuracy=True
            accuracy=1 if accuracy is True else float(accuracy or 100)/100
            stat='atk' if data.get('category')=='Physical' else 'spa'
            strength=mon.get('stats',{}).get(stat,features.species(mon.get('species','')).get('baseStats',{}).get(stat,80)+35)
            values.append(power/100*accuracy*(1.5 if kind in own_types else 1)*features.effectiveness(kind,{'species':enemy})*(strength/140)**.5)
        return max(values,default=0)
    coverage=0.0;leads=0.0;exposure=0.0
    for enemy in enemies:
        values=sorted((attack(m,enemy) for m in picked),reverse=True)
        coverage+=values[0]+.25*(values[1] if len(values)>1 else 0)
        leads+=max((attack(m,enemy) for m in picked[:2]),default=0)
        for mon in picked[:2]:
            incoming=[features.effectiveness(kind,mon) for kind in features.species(enemy).get('types',[])]
            exposure+=max(incoming,default=1)
    synergy=0.0
    if lead_rain:
        for mon in picked[:2]:
            names={to_id(m if isinstance(m,str) else m.get('id',m.get('move',''))) for m in mon.get('moves',[])}
            if names & {'electroshot','weatherball','hurricane','thunder'}:synergy+=.35
    return coverage/len(enemies)+.5*leads/len(enemies)-.15*exposure/len(enemies)+synergy
