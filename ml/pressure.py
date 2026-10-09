"""Fixed scripted opponent using only its own request and public battle view."""
from battle_state import hp_fraction,to_id
from ml.continuation import SETUP,RECOVERY
from ml.mechanics import score as tactical_score
from ml.preview import score as preview_score


def names(mon):
    return {to_id(m if isinstance(m,str) else m.get('id',m.get('move',''))) for m in mon.get('moves',[])}


def score(ctx,choice,features):
    state=ctx['state']
    if choice.startswith('team '):
        picked=[state['my_party'][int(i)-1] for i in choice[5:].split(',')[:2]]
        sets=[names(mon) for mon in picked]
        setup=any(moves & SETUP.keys() for moves in sets)
        support=any(moves & {'followme','ragepowder'} for moves in sets)
        return preview_score(ctx,choice,features)+4.0*setup+4.0*(setup and support)
    value=tactical_score({**ctx,'feature_profile':'mechanics-v1'},choice,features)
    own=state.get('my_actives',[]);selected=[]
    for index,text in enumerate(choice.split(',')):
        parts=text.strip().split();name=''
        if parts[0]=='move' and index<len(own):
            move=own[index]['moves'][int(parts[1])-1];name=to_id(move.get('id',move.get('move','')))
        selected.append(name)
    for i,name in enumerate(selected):
        if i>=len(own):continue
        mon=own[i];health=hp_fraction(mon.get('condition'))
        if name in SETUP:
            # Prefer a limited setup window, rather than dancing forever at +6.
            positive=max((mon.get('boosts',{}).get(key,0) for key,delta in SETUP[name].items() if delta>0),default=0)
            value+=4.0*health*max(0,1-positive/3)
        elif name in RECOVERY:
            value+=2.5*min(.5,1-health)
        elif name in ('followme','ragepowder'):
            partner=next((n for j,n in enumerate(selected) if j!=i),'')
            value+=2.0*health if partner in SETUP or partner in RECOVERY else .3*health
    return value
