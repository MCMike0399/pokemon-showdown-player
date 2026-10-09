"""Read-only per-match set/stat provenance audit, separate from policy inputs."""
from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ml.features import Features
from ml.postgame import archived_log
from ml.teams import team_id

STATS=('hp','atk','def','spa','spd','spe')
SET_FIELDS=('species','nature','evs','ivs','ability','item','moves','level','gender','teraType')


def fact(value, provenance):
    known=value is not None and value!=''
    return {'value':copy.deepcopy(value) if known else None,'provenance':provenance if known else 'unknown'}


def disclosed_sheet(packed):
    """Only explicitly transmitted EV/IV fields are decoded; blanks stay unknown."""
    result=[]
    for text in packed.split(']'):
        fields=text.split('|')+['']*12
        if not fields[0]:continue
        row={'species':fact(fields[1] or fields[0],'public-team-sheet')}
        for name,index in (('item',2),('ability',3),('nature',5),('gender',7)):
            row[name]=fact(fields[index],'public-team-sheet')
        row['moves']=fact(fields[4].split(',') if fields[4] else None,'public-team-sheet')
        for name,index,default in (('evs',6,0),('ivs',8,31)):
            values=fields[index].split(',') if fields[index] else []
            row[name]=fact({key:int(values[i]) if i<len(values) and values[i] else default for i,key in enumerate(STATS)} if values else None,'public-team-sheet')
        row['level']=fact(int(fields[10]) if fields[10] else None,'public-team-sheet')
        misc=fields[11].split(',')
        row['teraType']=fact(misc[5] if len(misc)>5 else None,'public-team-sheet')
        result.append(row)
    return result


def audit_episode(episode, rules, log, fixture_plan=None):
    if episode.get('status')!='complete':raise ValueError('postgame audit requires a terminal episode')
    champions=rules.get('champions')
    mechanics={'ev_field_units':'stat_points' if champions else 'traditional_evs' if champions is False else 'unknown',
               'ivs_affect_stats':False if champions else True if champions is False else None,
               'level_clause_mod':rules.get('level_clause_mod'),'rules_provenance':'pinned-format-dex' if rules else 'unavailable'}
    own=[{key:fact(row.get(key),'submitted-own-team') for key in SET_FIELDS} for row in episode.get('team_sets') or []]
    stats=[];seen=set();coverage=Counter();max_hp={}
    for decision,step in enumerate(episode.get('steps',[])):
        snap=step.get('snapshot')
        if not snap:continue
        coverage['captured_decisions']+=1
        req=snap.get('request',{});coverage['private_request_decisions']+=bool(req)
        pokemon=req.get('side',{}).get('pokemon',[])
        coverage['own_stat_decisions']+=any(mon.get('stats') for mon in pokemon)
        for index,mon in enumerate(pokemon):
            if not mon.get('stats'):continue
            identity=mon.get('ident',str(index));condition=mon.get('condition','')
            try:max_hp[identity]=float(condition.split()[0].split('/')[1].rstrip('gry'))
            except (IndexError,ValueError):pass
            species=mon.get('details','').split(',')[0]
            signature=(identity,species,json.dumps(mon['stats'],sort_keys=True),max_hp.get(identity))
            if signature in seen:continue
            seen.add(signature);stats.append({'decision_index':decision,'request_party_index':index+1,'species':species,
                'stats_as_requested':copy.deepcopy(mon['stats']),'observed_max_hp':max_hp.get(identity),
                'provenance':'private-player-request','note':'Request stats are unboosted; effective battle stats also depend on recorded boosts, status, abilities, items and fields.'})
    foe='p2' if episode['side']=='p1' else 'p1';sheets=[];turn=0
    for line in log:
        if line.startswith('|turn|'):turn=int(line.split('|')[2])
        if line.startswith('|showteam|'+foe+'|'):
            sheets.append({'first_public_turn':turn,'pokemon':disclosed_sheet(line.split('|',3)[3])})
    result={'version':1,'episode_id':episode['id'],'format':episode['format'],'source':episode.get('source'),
        'team_fingerprint':episode.get('team'),'mechanics':mechanics,'own_submitted_sets':own,'own_observed_stats':stats,
        'opponent_public_sheets':sheets,'coverage':dict(coverage),'opponent_hidden_set_policy':'Unknown unless explicitly disclosed; speed/damage inference remains an estimate, not a recovered set.',
        'raw_evidence_retained':{'team_sets':bool(episode.get('team_sets')),'decision_snapshots':bool(coverage),'terminal_log':bool(log)}}
    # Synthetic fixture truth is available only as a terminal, separately labelled
    # diagnostic. A live opponent can never inherit a research/fixture spread.
    if fixture_plan and episode.get('source')=='local' and fixture_plan.get('format')==episode['format']:
        fingerprint=episode.get('simulation',{}).get('opponent_team')
        matches={team_id(episode['format'],sets):sets for sets in fixture_plan.get('opponents',[])}
        if fingerprint in matches:
            result['offline_fixture_diagnostics']={'provenance':'frozen-simulator-fixture','scope':'postgame-diagnostics-only',
                'opponent_declared_sets':copy.deepcopy(matches[fingerprint]),'opponent_team_fingerprint':fingerprint}
    return result


def run(root,campaign,start,cutoff,output,fixture_plan=None):
    ledger=[json.loads(line) for line in (campaign/'games.jsonl').read_text().splitlines() if line.strip()]
    if not 1<=start<=cutoff or len(ledger)<cutoff:raise ValueError('requested exact completed cutoff is unavailable')
    selected=ledger[start-1:cutoff];rooms=[row['room'] for row in selected]
    if len(set(rooms))!=len(rooms):raise ValueError('duplicate terminal rooms')
    fmt=json.loads((campaign/'manifest.json').read_text())['format'];rules=Features.cached(fmt).dex.get('rules',{})
    db=sqlite3.connect((root/'experience.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);db.execute('BEGIN')
    try:
        index=defaultdict(list)
        for key,room in db.execute("SELECT id,json_extract(data,'$.room') FROM episodes WHERE format=? AND source='ladder' AND status='complete'",(fmt,)):index[room].append(key)
        reports=[]
        for ordinal,room in enumerate(rooms,start):
            if len(index[room])!=1:raise ValueError('ambiguous terminal episode')
            episode=json.loads(db.execute('SELECT data FROM episodes WHERE id=?',(index[room][0],)).fetchone()[0])
            reports.append({'ordinal':ordinal,**audit_episode(episode,rules,archived_log(episode,db),fixture_plan)})
    finally:db.close()
    summary={'start':start,'cutoff':cutoff,'games':len(reports),'own_full_set_games':sum(len(r['own_submitted_sets'])==6 for r in reports),
        'own_nature_complete_games':sum(bool(r['own_submitted_sets']) and all(s['nature']['value'] for s in r['own_submitted_sets']) for r in reports),
        'own_evs_complete_games':sum(bool(r['own_submitted_sets']) and all(s['evs']['value'] is not None for s in r['own_submitted_sets']) for r in reports),
        'own_explicit_iv_games':sum(any(s['ivs']['value'] is not None for s in r['own_submitted_sets']) for r in reports),
        'opponent_public_sheet_games':sum(bool(r['opponent_public_sheets']) for r in reports),
        'opponent_public_nature_games':sum(any(s['nature']['value'] for sheet in r['opponent_public_sheets'] for s in sheet['pokemon']) for r in reports),
        'opponent_public_evs_games':sum(any(s['evs']['value'] is not None for sheet in r['opponent_public_sheets'] for s in sheet['pokemon']) for r in reports)}
    output.mkdir(parents=True,exist_ok=True)
    (output/'matches.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in reports))
    (output/'summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--cutoff',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.root,a.campaign,a.start,a.cutoff,a.output),indent=2))
