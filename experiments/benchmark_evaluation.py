"""Compare CPU evaluator concurrency on fixed production cases, without live writes."""
import argparse
import asyncio
import copy
import json
import shutil
import time
from pathlib import Path
import numpy as np
from ml.resources import ResourcePolicy
from ml.storage import Store
from ml.worker import parallel_games
from ml.reload import source_generation

async def main(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    plans=sorted((args.source/'data/ml/candidates').glob('*/evaluation-plan.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    selected=None
    for path in plans:
        plan=json.loads(path.read_text())
        if plan.get('incumbent') and Path(plan['incumbent'][0]['inference_root']).joinpath('experience.sqlite3').is_file():
            selected=(path,plan);break
    if selected is None:raise ValueError('need a readable saved production evaluation plan')
    path,plan=selected
    parent=output/'models';parent.mkdir()
    fmt=plan['incumbent'][0]['format']
    shutil.copy2(args.checkpoint,parent/(fmt+'.pt'))
    inputs=output/'inputs';shutil.copytree(plan['incumbent'][0]['inference_root'],inputs)
    Store(output/'records').close()
    tasks=copy.deepcopy(plan['incumbent'][:args.games])
    for task in tasks:
        task.update(root=str(output/'records'),checkpoint_root=str(output),inference_root=str(inputs),
                    source_generation=source_generation(),simulator_slots_root=str(output/'records'))
        if task.get('opponent_checkpoint_root'):task['opponent_checkpoint_root']=str(output)
    rows=[];expected=None
    for repeat in range(2):
        for workers in (args.workers if repeat==0 else reversed(args.workers)):
            policy=ResourcePolicy(max_workers=workers,training_threads=1,backend='cpu',min_available_gb=args.min_available_gb,
                                  max_system_cpu_percent=args.max_cpu_percent)
            policy.apply_background(enable_mps=False)
            start=time.perf_counter()
            result=await parallel_games(copy.deepcopy(tasks),policy,time.monotonic()+300)
            seconds=time.perf_counter()-start
            games=result['games']
            valid=(len(games)==len(tasks) and not result.get('deferred') and all(not g.get('unfinished') and not g.get('rejected_actions') for g in games))
            outcomes=[(g.get('seed'),g.get('learner_side'),g.get('winner'),g.get('tie',False)) for g in games]
            if valid:
                if expected is None:expected=outcomes
                elif expected!=outcomes:raise ValueError('concurrency changed fixed-case outcomes')
            rows.append({'workers':workers,'repeat':repeat,'seconds':seconds,'games':len(games),
                         'complete':valid,'games_per_minute':60*len(games)/seconds,'results':games,
                         'resources':result.get('resources'),'reason':result.get('reason')})
            (output/'results.json').write_text(json.dumps({'source_plan':str(path),'same_cases':True,'runs':rows},indent=2)+'\n')
            print(json.dumps({k:v for k,v in rows[-1].items() if k!='results'}),flush=True)
    measured={}
    for workers in args.workers:
        completed=[r['games_per_minute'] for r in rows if r['workers']==workers and r['complete']]
        measured[str(workers)]=float(np.median(completed)) if completed else None
    print(json.dumps({'median_games_per_minute':measured}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games',type=int,default=24)
    p.add_argument('--workers',type=int,nargs='+',default=[2,3])
    p.add_argument('--min-available-gb',type=float,default=2)
    p.add_argument('--max-cpu-percent',type=float,default=75)
    asyncio.run(main(p.parse_args()))
