"""Public preview structure and own-set matchup features for strategic-v3.

Opponent types/base stats are dex facts. Coverage, exposure and relative speed
are estimates; they do not reconstruct spreads, items, abilities or brought four.
Prospective forms use our supplied sets with at most one Mega per plan.
"""
from collections import Counter

from battle_state import to_id
from ml.opening import own_form
from ml.tactics import stat


def public_summary(species, features):
    rows=[features.species(name) for name in species]
    count=max(1,len(rows));result={}
    types=Counter(kind for row in rows for kind in row.get('types',[]))
    result.update({'opponent/type/'+kind:value/count for kind,value in types.items()})
    for name in ('hp','atk','def','spa','spd','spe'):
        result['opponent/base/'+name]=sum(row.get('baseStats',{}).get(name,0) for row in rows)/count/200
    result['opponent/base_attack_bias']=sum(row.get('baseStats',{}).get('atk',0)>row.get('baseStats',{}).get('spa',0) for row in rows)/count
    return result


def prepare(ctx, features):
    if '_v3_preview' in ctx:
        return ctx['_v3_preview']
    state=ctx['state'];party=state.get('my_party',[])
    enemies=[{'species':name,'condition':'100/100'} for name in state.get('opp_preview',[])]
    forms=[(mon,own_form(ctx,mon,features,True)) for mon in party]
    tables={}
    for rain in (False,True):
        table=[]
        for pair in forms:
            variants=[]
            for mon in pair:
                attacks=[];exposure=[]
                own_types=features.species(mon.get('species','')).get('types',[])
                for enemy in enemies:
                    best=0
                    for move in mon.get('moves',[]):
                        name=to_id(move if isinstance(move,str) else move.get('id',move.get('move','')))
                        data=features.dex.get('moves',{}).get(name,{})
                        if not data.get('basePower'):continue
                        kind=data.get('type','');power=data['basePower']
                        if name=='weatherball' and rain:kind,power='Water',100
                        charge=.35 if name=='electroshot' and not rain else 1
                        attack_key='def' if name=='bodypress' else 'atk' if data.get('category')=='Physical' else 'spa'
                        strength=stat(enemy,'atk',features) if name=='foulplay' else stat(mon,attack_key,features)
                        field_scale=1.5 if rain and kind=='Water' else .5 if rain and kind=='Fire' else 1
                        hit=power/100*(strength/160)*(1.5 if kind in own_types else 1)*features.effectiveness(kind,enemy)*charge*field_scale
                        best=max(best,hit)
                    attacks.append(min(2,best))
                    exposure.append(max((features.effectiveness(kind,mon) for kind in features.species(enemy['species']).get('types',[])),default=1))
                variants.append({'attack':attacks,'exposure':exposure,'speed':stat(mon,'spe',features)})
            table.append(variants)
        tables[rain]=table
    ctx['_v3_preview']={'party':party,'enemies':enemies,'forms':forms,'tables':tables}
    return ctx['_v3_preview']


def plan_summary(ctx, choice, features):
    cache=prepare(ctx,features);enemies=cache['enemies'];party=cache['party']
    if not enemies:return {}
    order=[int(value)-1 for value in choice[5:].split(',')]
    picked=[party[i] for i in order]
    setter=lambda m:to_id(m.get('ability') or m.get('baseAbility'))=='drizzle'
    rain=any(setter(m) for m in picked);lead_rain=any(setter(m) for m in picked[:2])
    options=[i for i in order if cache['forms'][i][1] is not party[i]]
    result={'plan/rain_brought':float(rain),'plan/rain_lead':float(lead_rain),'plan/mega_options':len(options)/2}
    names=[{to_id(m if isinstance(m,str) else m.get('id',m.get('move',''))) for m in mon.get('moves',[])} for mon in picked]
    result['plan/rain_dependencies']=sum(bool(n&{'electroshot','hurricane','thunder'}) for n in names)/4
    result['plan/trickroom_lead']=float(any('trickroom' in n for n in names[:2]))
    result['plan/priority_shield']=float(any(to_id(m.get('ability') or m.get('baseAbility')) in ('armortail','dazzling','queenlymajesty') for m in picked))

    def evaluate(mega):
        rows=[cache['tables'][rain][i][int(i==mega)] for i in order]
        leads=[cache['tables'][lead_rain][i][int(i==mega)] for i in order[:2]]
        metrics=Counter();type_count=Counter();uncovered=sole=0
        for j,enemy in enumerate(enemies):
            values=[row['attack'][j] for row in rows]
            best=max(values);uncovered+=best==0;sole+=sum(value>=1 for value in values)==1
            metrics['coverage']+=best/len(enemies)
            metrics['lead_pressure']+=sum(row['attack'][j] for row in leads)/2/len(enemies)
            metrics['lead_type_exposure']+=sum(row['exposure'][j] for row in leads)/4/len(enemies)
            metrics['lead_speed_fraction']+=sum(row['speed']>stat(enemy,'spe',features) for row in leads)/2/len(enemies)
            for kind in features.species(enemy['species']).get('types',[]):
                metrics['coverage/type/'+kind]+=best;type_count[kind]+=1
        for kind,count in type_count.items():metrics['coverage/type/'+kind]/=count
        metrics['uncovered']=uncovered/len(enemies);metrics['sole_estimated_answer']=sole/len(enemies)
        return dict(metrics)
    base=evaluate(None)
    result.update({'plan/base/'+key:value for key,value in base.items()})
    # This is one legal prospective plan, not simultaneous evolution of both.
    modes=[evaluate(i) for i in options]
    selected=max(modes,key=lambda row:row['coverage']-.25*row['lead_type_exposure']) if modes else base
    result.update({'plan/single_mega/'+key:value for key,value in selected.items()})
    return result


def add_state(vector, ctx, features):
    if not any(choice.startswith('team ') for choice in ctx.get('choices',[])):
        return
    for name,value in public_summary(ctx['state'].get('opp_preview',[]),features).items():
        features.add(vector,'v3/'+name,value)


def add_action(vector, ctx, choice, features):
    if choice.startswith('team '):
        for name,value in plan_summary(ctx,choice,features).items():
            features.add(vector,'v3/'+name,value)
