"""Public format eligibility is a hypothesis, never a revealed opposing item/set."""
import json,subprocess
from pathlib import Path
from battle_state import to_id

CATALOG_QUERY = r'''
const {Dex,TeamValidator}=require('pokemon-showdown');
const fmt=process.argv[1], v=new TeamValidator(fmt), d=Dex.forFormat(fmt), forms={};
for(const f of d.species.all()){
 if(!f.isMega || !f.requiredItem)continue;
 const b=d.species.get(f.baseSpecies), item=d.items.get(f.requiredItem), set={name:b.name,species:b.name,item:item.name,ability:b.abilities['0']}, has={};
 if(!b.exists || !item.exists || v.checkSpecies(set,b,f,has) || v.checkItem(set,item,has))continue;
 (forms[b.id] ||= []).push(f.name);
}
process.stdout.write(JSON.stringify(forms));
'''


def catalog(fmt, features):
    caches=getattr(features,'_v5_mega_catalogs',{})
    if fmt not in caches:
        root=Path(__file__).resolve().parents[1]
        caches[fmt]=json.loads(subprocess.check_output(['node','-e',CATALOG_QUERY,fmt],cwd=root,timeout=30))
        features._v5_mega_catalogs=caches
    return caches[fmt]


def spent(ctx, features):
    mine=ctx['state'].get('side_id',ctx.get('request',{}).get('side',{}).get('id','p1'))
    return any(line.startswith('|-mega|') and not line.split('|')[2].startswith(mine) for line in ctx.get('public_log',[])) or any(features.species(m.get('species','')).get('requiredItem') for m in ctx['state'].get('opp_actives',[])+ctx['state'].get('opp_revealed',[]))


def alternatives(ctx, mon, features):
    if spent(ctx,features) or features.species(mon.get('species','')).get('requiredItem'):
        return [],0.0
    forms=[features.species(name) for name in catalog(ctx['format'],features).get(to_id(mon.get('species','')),[])]
    # An explicit empty item means publicly removed/absent, not an unknown item.
    known='item' in mon and mon['item'] is not None
    if known:
        forms=[form for form in forms if to_id(form.get('requiredItem'))==to_id(mon['item'])]
    return forms[:2],(.5 if known else .35) if forms else 0.0


def expand(ctx, mon, rows, features):
    forms,mass=alternatives(ctx,mon,features)
    if not forms:return rows
    movable=sum(w for name,_,_,w in rows if name!='switch')
    if not movable:return rows
    result=[(name,data,target,w*(1-mass) if name!='switch' else w) for name,data,target,w in rows]
    for form in forms:
        prospective={**mon,'species':form['name'],'ability':form.get('abilities',{}).get('0',''),'stats':{}}
        result.extend((name,{**data,'hypothesis_form':prospective,'form_provenance':'disclosed-stone' if mon.get('item') else 'uncertain-format-eligibility'},target,w*mass/len(forms)) for name,data,target,w in rows if name!='switch')
    return result


def envelope(joint, limit=8):
    """Keep plausible Mega alternatives from disappearing under top-k truncation."""
    selected=joint[:limit];covered=set()
    for rows in joint:
        for index,row in enumerate(rows):
            form=row[1].get('hypothesis_form')
            key=(index,form['species']) if form else None
            if key and key not in covered:
                covered.add(key)
                if rows not in selected:selected.append(rows)
    # At most two forms per each of two opposing slots: four extra cases.
    return selected[:limit+4]
