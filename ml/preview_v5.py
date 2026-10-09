"""Defense/HP-aware estimated answer coverage with one prospective own Mega."""
from collections import Counter
from battle_state import to_id
from ml.opening import own_form
from ml.mechanics import known_mon
from ml.tactics import stat
from ml.continuation import attacking_moves


def bounded(value):return max(0,value)/(1+max(0,value))


def prepare(ctx,features):
    if '_v5_preview' in ctx:return ctx['_v5_preview']
    from ml.strategy import hit
    party=ctx['state'].get('my_party',[]);enemies=[known_mon({'species':name,'condition':'100/100'},ctx['state'],features) for name in ctx['state'].get('opp_preview',[])]
    forms=[(mon,own_form(ctx,mon,features,True)) for mon in party];tables={}
    for rain in (False,True):
        table=[]
        state={**ctx['state'],'weather':'RainDance' if rain else None,'_strategic_v2':True,'_strategic_v4':True}
        for pair in forms:
            variants=[]
            for mon in pair:
                attacks=[];exposure=[]
                for enemy in enemies:
                    best=0
                    for name,data in attacking_moves(mon,features):
                        charge=.35 if name=='electroshot' and not rain else 1
                        actor={**mon,'boosts':{**mon.get('boosts',{}),'spa':min(6,mon.get('boosts',{}).get('spa',0)+1)}} if name=='electroshot' else mon
                        spread=data.get('target') in ('allAdjacent','allAdjacentFoes')
                        best=max(best,hit(actor,enemy,name,data,state,features,'mine',spread)*charge)
                    attacks.append(best)
                    exposure.append(max((hit(enemy,mon,name,data,state,features,'theirs',data.get('target') in ('allAdjacent','allAdjacentFoes')) for name,data in attacking_moves(enemy,features)),default=0))
                variants.append({'attack':attacks,'exposure':exposure,'speed':stat(mon,'spe',features)})
            table.append(variants)
        tables[rain]=table
    ctx['_v5_preview']={'party':party,'enemies':enemies,'forms':forms,'tables':tables};return ctx['_v5_preview']


def plan_summary(ctx,choice,features):
    cache=prepare(ctx,features);party=cache['party'];enemies=cache['enemies']
    if not enemies:return {}
    order=[int(x)-1 for x in choice[5:].split(',')];picked=[party[i] for i in order]
    setter=lambda mon:to_id(mon.get('ability') or mon.get('baseAbility'))=='drizzle'
    rain=any(setter(m) for m in picked);lead_rain=any(setter(m) for m in picked[:2]);options=[i for i in order if cache['forms'][i][1] is not party[i]]
    result={'plan/rain_brought':float(rain),'plan/rain_lead':float(lead_rain),'plan/mega_options':len(options)/2}
    names=[{to_id(m if isinstance(m,str) else m.get('id',m.get('move',''))) for m in mon.get('moves',[])} for mon in picked]
    result['plan/rain_dependencies']=sum(bool(n & {'electroshot','hurricane','thunder'}) for n in names)/max(1,len(picked))
    result['plan/trickroom_lead']=float(any('trickroom' in n for n in names[:2]))
    shield=lambda mon:to_id(mon.get('ability') or mon.get('baseAbility')) in ('armortail','dazzling','queenlymajesty')
    result['plan/priority_shield']=float(any(shield(m) for m in picked))
    result['plan/priority_shield_lead']=float(any(shield(m) for m in picked[:2]))
    def evaluate(mega):
        rows=[cache['tables'][rain][i][int(i==mega)] for i in order];leads=[cache['tables'][lead_rain][i][int(i==mega)] for i in order[:2]];metrics=Counter();coverage=[];sole=weak=0
        for j,enemy in enumerate(enemies):
            values=[r['attack'][j] for r in rows];best=max(values);coverage.append(bounded(best));sole+=sum(v>=1 for v in values)==1;weak+=best<.5
            metrics['coverage']+=bounded(best)/len(enemies);metrics['lead_pressure']+=sum(bounded(r['attack'][j]) for r in leads)/len(leads)/len(enemies)
            metrics['lead_exposure']+=sum(bounded(r['exposure'][j]) for r in leads)/len(leads)/len(enemies)
            metrics['lead_speed_fraction']+=sum(r['speed']>stat(enemy,'spe',features) for r in leads)/len(leads)/len(enemies)
        metrics['minimum_coverage']=min(coverage);metrics['weak_answer_fraction']=weak/len(enemies);metrics['sole_estimated_ko_answer']=sole/len(enemies);return dict(metrics)
    base=evaluate(None);modes=[evaluate(i) for i in options];selected=max(modes,key=lambda m:.6*m['coverage']+.4*m['minimum_coverage']-.15*m['lead_exposure']) if modes else base
    result.update({'plan/base/'+k:v for k,v in base.items()});result.update({'plan/single_mega/'+k:v for k,v in selected.items()});return result


def add_action(vector,ctx,choice,features):
    if choice.startswith('team '):
        for key,value in plan_summary(ctx,choice,features).items():features.add(vector,'v5/'+key,value)
