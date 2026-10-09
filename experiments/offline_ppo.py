"""Notebook-friendly PPO experiments using the repository's real offline simulator."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import shutil
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
import psutil
import torch
from ml.model import Model
from ml.resources import memory_pressure, ResourceDeferred
from ml.reload import source_generation
from ml.scout import Scout, ingest_public
from ml.simulator import load_dex, validate_team
from ml.storage import Store
from ml.worker import rollout

REPO = Path(__file__).resolve().parents[1]
FMT = 'gen9championsvgc2026regmc'
CORPUS = REPO/'data/offline-corpus/divemac-20261009'
EXPERIMENT_SOURCE = Path(__file__).read_bytes()
EXPERIMENT_SHA256 = hashlib.sha256(EXPERIMENT_SOURCE).hexdigest()

@dataclass(frozen=True)
class Arm:
    name: str
    learning_rate: float = 1e-3
    entropy_start: float = .01
    entropy_end: float = .01
    target_steps: int = 480
    epochs: int = 3
    minibatch_size: int = 32  # zero means approximately four minibatches

ARMS = (
    Arm('high_lr_reference'),
    Arm('low_entropy', entropy_start=1e-4, entropy_end=1e-4),
    Arm('lower_lr', learning_rate=2.5e-4),
    Arm('larger_collection', learning_rate=2.5e-4, target_steps=1024, epochs=4, minibatch_size=0),
    Arm('combined', learning_rate=2.5e-4, entropy_start=1e-4, entropy_end=1e-4, target_steps=1024, epochs=4, minibatch_size=0),
    Arm('entropy_decay', learning_rate=2.5e-4, entropy_end=1e-4, target_steps=1024, epochs=4, minibatch_size=0),
    Arm('repository_control', learning_rate=3e-4, epochs=4),
)

@dataclass
class Compute:
    workers: int = 8
    training_threads: int = 4
    backend: str = 'cpu'
    gpu_fraction: float = .50
    min_available_gb: float = 2
    min_disk_gb: float = 10
    max_root_gb: float = 100

    def __post_init__(self):
        if not 1 <= self.workers <= max(1, (os.cpu_count() or 2)-1):
            raise ValueError('reserve one CPU core; workers must be positive')
        if not 1 <= self.training_threads <= 4 or self.backend not in ('cpu','mps'):
            raise ValueError('threads must be 1..4 and backend cpu or mps')
        if not .01 <= self.gpu_fraction <= .7 or self.min_available_gb < 1:
            raise ValueError('keep bounded Metal allocation and at least 1 GiB available')
        if self.min_disk_gb < 2 or self.max_root_gb < 1:
            raise ValueError('keep at least 2 GiB disk headroom and a positive run budget')

    def check(self):
        free = psutil.virtual_memory().available/2**30
        pressure = memory_pressure()
        if free < self.min_available_gb or pressure is not None and pressure & 4:
            raise ResourceDeferred(f'memory headroom exhausted ({free:.2f} GiB); rerun in a fresh output after checking artifacts')
        if psutil.disk_usage(REPO).free/2**30 < self.min_disk_gb:
            raise ResourceDeferred('disk headroom exhausted')

    def apply(self):
        self.check()
        torch.set_num_threads(self.training_threads)
        if self.backend == 'mps':
            if not torch.backends.mps.is_available():
                raise ValueError('Metal unavailable; choose CPU or execute outside the restricted sandbox')
            torch.mps.set_per_process_memory_fraction(self.gpu_fraction)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def inventory():
    return {'host':platform.node(),'chip':subprocess.run(['sysctl','-n','machdep.cpu.brand_string'], capture_output=True,text=True).stdout.strip(),
        'cpu_cores':os.cpu_count(),'memory_gib':psutil.virtual_memory().total/2**30,
        'available_gib':psutil.virtual_memory().available/2**30,'mps_available':torch.backends.mps.is_available(),
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'node':subprocess.check_output(['node','--version'],text=True).strip(),'source_generation':source_generation(),
        'experiment_sha256':EXPERIMENT_SHA256}


async def prepare_teams(corpus=CORPUS, max_teams=4):
    """Select complete local teams; public partial spreads are never invented."""
    await load_dex(FMT)
    candidates = [('Self-built rain fixture',json.loads((REPO/'examples/champions-rain.json').read_text())['sets'])]
    path = Path(corpus)/'teams.json'
    if path.exists():
        content = json.loads(path.read_text())
        content = content.get('teams',content)
        for name, item in content.items():
            if isinstance(item,dict) and item.get('format') == FMT and item.get('sets'):
                candidates.append((name,item['sets']))
    result, skipped, seen = [], [], set()
    for name, sets in candidates:
        key = json.dumps(sets, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        validation = await validate_team(FMT,sets)
        if validation.get('valid') is False or validation.get('errors'):
            skipped.append({'name':name,'validation':validation})
        else:
            result.append({'name':name,'sets':sets})
        if len(result) >= max_teams:
            break
    if not result:
        raise ValueError('no simulator-valid teams')
    return {'format':FMT,'teams':result,'skipped':skipped,'provenance':'copied local team store and self-built fixture; see corpus manifest'}


def task(root, checkpoint, inputs, teams, index, seed, training, opponent='random'):
    return {'root':str(root),'checkpoint_root':str(checkpoint),'inference_root':str(inputs),
        'format':FMT,'source_generation':source_generation(),
        'team1':teams[index%len(teams)]['sets'],'team2':teams[(index+1)%len(teams)]['sets'],
        'seed':seed,'side':'p2' if (index+index//len(teams))%2 else 'p1',
        'training':training,'sample_actions':True,'opponent':opponent,
        'open_team_sheets':False,'seed_encoding':'full-v1'}


def games(pool, tasks, compute):
    results = []
    # Initialize evaluation stores once before children query the same schema.
    # Training writers each have a shard, removing per-step SQLite contention.
    originals = {}
    prepared = []
    for original in tasks:
        item = dict(original)
        if item['training']:
            shard = Path(item['root'])/'rollout-shards'/str(item['seed'])
            originals[str(shard)] = item['root']
            item['root'] = str(shard)
        else:
            path = Path(item['root'])/'experience.sqlite3'
            if not path.exists():
                Store(path.parent).close()
        prepared.append(item)
    tasks = prepared
    offset = 0
    while offset < len(tasks):
        compute.check()
        headroom = psutil.virtual_memory().available/2**30-compute.min_available_gb
        resident = getattr(pool, '_offline_resident', 0)
        capacity = min(compute.workers, int(headroom/.15), resident+int(headroom/.55))
        if capacity < 1:
            raise ResourceDeferred('not enough memory headroom for another simulator wave')
        calls = [pool.submit(rollout,t) for t in tasks[offset:offset+capacity]]
        offset += len(calls)
        pool._offline_resident = max(resident,len(calls))
        pool._offline_peak = max(getattr(pool, '_offline_peak', 0),len(calls))
        for future in calls:
            result = future.result()
            results.append(result)
    for shard, destination in originals.items():
        source, target = Store.read_only(Path(shard)), Store(Path(destination))
        try:
            for row in source.db.execute('SELECT data FROM episodes'):
                target.save_episode(json.loads(row[0]))
        finally:
            source.close()
            target.close()
    return results


def scores(results):
    completed = [r for r in results if not r.get('unfinished') and not r.get('ongoing') and ('winner' in r or r.get('tie'))]
    values = [0 if r.get('tie') else 1 if r.get('winner') == 'LocalBrain' else -1 for r in completed]
    wins = sum(v == 1 for v in values)
    n = len(values)
    if n:
        p, z = wins/n, 1.959963984540054
        center = (p+z*z/(2*n))/(1+z*z/n)
        radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        interval = [center-radius,center+radius]
    else:
        interval = [None,None]
    return {'completed':n,'unfinished':len(results)-n,'wins':wins,'losses':sum(v == -1 for v in values),
        'draws':sum(v == 0 for v in values),'mean_episode_return':float(np.mean(values)) if values else None,
        'win_rate':wins/n if n else None,'win_rate_wilson95':interval,
        'rejected_actions':sum(r.get('rejected_actions',0) for r in results)}


def ensure_clean(results):
    summary = scores(results)
    if summary['unfinished'] or summary['rejected_actions']:
        raise RuntimeError('incomplete or invalid simulator games; retained for diagnosis, no learning claim')


def evaluate(pool, root, inputs, teams, compute, count, seed, label):
    rows = []
    for opponent in ('random','tactical'):
        cases = [task(root/'evaluation-records',root,inputs,teams,i,seed+i,False,opponent) for i in range(count)]
        results = games(pool,cases,compute)
        ensure_clean(results)
        record = {'label':label,'opponent':opponent,**scores(results),'games':results,'policy_mode':'sampled'}
        rows.append(record)
    return rows


def run_arm(output, arm, seed, teams, compute, updates=10, eval_games=32, eval_every=2,
            initial_checkpoint=None, inference_root=None):
    """One seed/arm. Frozen full-game collection; each fresh batch used exactly once."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a fresh run directory; saved runs are never overwritten')
    if updates < 1 or eval_games < 1 or eval_every < 1 or arm.target_steps < 1:
        raise ValueError('updates, evaluation counts and step target must be positive')
    if not 0 <= seed < 1000000:
        raise ValueError('training seed must be 0..999999')
    compute.apply()
    output.mkdir(parents=True)
    store = Store(output)
    inputs = Path(inference_root).resolve() if inference_root else output/'empty-inputs'
    if inference_root is None:
        Store(inputs).close()
    if initial_checkpoint:
        (output/'models').mkdir()
        shutil.copy2(initial_checkpoint,output/'models'/(FMT+'.pt'))
    model = Model(output,FMT,seed=seed)
    model.set_device('cpu')
    model.save()
    manifest = {'arm':asdict(arm),'seed':seed,'compute':asdict(compute),'hardware':inventory(),
        'teams':teams,'updates':updates,'evaluation_games_per_opponent':eval_games,'evaluation_every':eval_every,
        'initial_checkpoint':str(initial_checkpoint) if initial_checkpoint else 'fresh weights with existing heuristic prior',
        'initial_sha256':model.checkpoint_sha256,'inputs':str(inputs),
        'reward':'terminal +1/-1/0; complete episodes only','collection':'minimum fresh completed decisions; overshoot retained',
        'training_seed_base':seed*10000000,'development_seed_base':1000000000000+seed*10000,
        'final_seed_base':2000000000000+seed*10000,'started':time.time()}
    write_json(output/'manifest.json',manifest)
    shutil.copy2(model.path,output/'initial.pt')
    reports, evaluations, cursor, total_steps = [], [], 0, 0
    start = time.perf_counter()
    context = multiprocessing.get_context('spawn')
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=compute.workers,mp_context=context) as pool:
            evaluations.extend(evaluate(pool,output,inputs,teams,compute,eval_games,manifest['development_seed_base'],'initial'))
            write_json(output/'evaluations.json',evaluations)
            for update in range(updates):
                revision = model.revision
                collected = []
                while True:
                    episodes = store.episodes(FMT,revision=revision)
                    steps = sum(len(e['steps']) for e in episodes)
                    if steps >= arm.target_steps:
                        break
                    cases = [task(output,output,inputs,teams,cursor+i,manifest['training_seed_base']+cursor+i,True)
                             for i in range(compute.workers)]
                    collected.extend(games(pool,cases,compute))
                    cursor += len(cases)
                    ensure_clean(collected)
                    # Disk growth is bounded between complete waves, without removing data.
                    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())/2**30
                    if size > compute.max_root_gb:
                        raise ResourceDeferred('experiment storage budget reached; data retained')
                fraction = update/max(1,updates-1)
                entropy = arm.entropy_start + fraction*(arm.entropy_end-arm.entropy_start)
                minibatch = arm.minibatch_size or math.ceil(steps/4)
                torch.manual_seed(seed+update)
                model.set_device(compute.backend)
                try:
                    training = model.train(episodes,epochs=arm.epochs,learning_rate=arm.learning_rate,
                        entropy_coef=entropy,minibatch_size=minibatch,shuffle_seed=seed+update,
                        checkpoint=compute.check)
                finally:
                    model.set_device('cpu')
                if not training['trained'] or training['excluded_episodes']:
                    raise RuntimeError('training batch failed collecting-policy checks; see saved episodes')
                store.mark_trained(training['consumed'])
                total_steps += training['steps']
                row = {'update':update+1,'seed':seed,'arm':arm.name,'training_steps':total_steps,
                    'elapsed_seconds':time.perf_counter()-start,'collection':scores(collected),
                    'episodes':[{'id':e['id'],'return':e['outcome'],'steps':len(e['steps'])} for e in episodes],
                    'training':training}
                reports.append(row)
                write_json(output/'updates.json',reports)
                print(json.dumps({'arm':arm.name,'seed':seed,'update':update+1,'steps':total_steps,
                    'return':row['collection']['mean_episode_return'],'device':training['device'],'kl':training['kl_history'][-1]}),flush=True)
                if (update+1)%eval_every == 0 or update+1 == updates:
                    evaluations.extend(evaluate(pool,output,inputs,teams,compute,eval_games,manifest['development_seed_base'],f'update-{update+1}'))
                    write_json(output/'evaluations.json',evaluations)
            # Separate final seeds, kept out of the development curve.
            evaluations.extend(evaluate(pool,output,inputs,teams,compute,eval_games,manifest['final_seed_base'],'final'))
            write_json(output/'evaluations.json',evaluations)
        report = {'status':'complete','arm':arm.name,'seed':seed,'steps':total_steps,'updates':reports,
                  'evaluations':evaluations,'elapsed_seconds':time.perf_counter()-start}
        write_json(output/'report.json',report)
        return report
    except BaseException as error:
        write_json(output/'interrupted.json',{'type':type(error).__name__,'message':str(error),
                   'completed_updates':len(reports),'steps':total_steps})
        raise
    finally:
        store.close()


def run_matrix(output, teams, compute, arms=ARMS, seeds=(0,1,2), **kwargs):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError('matrix output already exists; use a fresh output directory')
    root.mkdir(parents=True)
    (root/'executed-offline-ppo.py').write_bytes(EXPERIMENT_SOURCE)
    write_json(root/'matrix.json',{'arms':[asdict(a) for a in arms],'seeds':list(seeds),
        'compute':asdict(compute),'settings':{k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()},'hardware':inventory()})
    reports = []
    for seed in seeds:
        for arm in arms:
            reports.append(run_arm(root/f'{arm.name}-seed{seed}',arm,seed,teams,compute,**kwargs))
    write_json(root/'summary.json',[{'arm':r['arm'],'seed':r['seed'],'steps':r['steps'],
        'elapsed_seconds':r['elapsed_seconds'],'final':[e for e in r['evaluations'] if e['label']=='final']} for r in reports])
    return reports


def benchmark_workers(output, teams, counts=(1,4,8,12,14), count_games=28):
    root = Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    inputs = root/'inputs'
    Store(inputs).close()
    model = Model(root,FMT)
    rows = []
    for workers in counts:
        if workers >= (os.cpu_count() or 2):
            continue
        compute = Compute(workers=workers)
        compute.check()
        cases = [task(root/'records',root,inputs,teams,i,3000000000000+i,False) for i in range(count_games)]
        started = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results = games(pool,cases,compute)
            effective = getattr(pool, '_offline_peak', 0)
        seconds = time.perf_counter()-started
        ensure_clean(results)
        rows.append({'workers':workers,'peak_wave_workers':effective,'games':count_games,'seconds':seconds,'games_per_minute':60*count_games/seconds,
                     'startup_included':True,**scores(results)})
        write_json(root/'workers.json',rows)
    return rows


def benchmark_training(output, episode_root, minibatches=(32,128,256), repeats=2, min_steps=1024):
    """Time actual compatible complete-game PPO batches, including transfers/checks/save."""
    output, episode_root = Path(output).resolve(), Path(episode_root).resolve()
    output.mkdir(parents=True,exist_ok=False)
    source = Store.read_only(episode_root)
    episodes = source.episodes(FMT,untrained=False)
    source.close()
    # initial.pt corresponds to update 1 collecting revision, unlike the latest model.
    checkpoint = episode_root/'initial.pt'
    revision = torch.load(checkpoint,map_location='cpu',weights_only=True)['revision']
    episodes = [e for e in episodes if e['revision'] == revision]
    if not episodes:
        raise ValueError('need completed first-update rollouts from an experiment run')
    original_steps = sum(len(e['steps']) for e in episodes)
    original_episodes = episodes
    episodes = []
    # Repeat measured rollout shapes ONLY in isolated throughput trials, not in
    # the learning matrix; no policy-strength claim uses these fits.
    for repeat in range(max(1, math.ceil(min_steps/original_steps))):
        episodes.extend([{**e, 'id':f'benchmark-{repeat}-{e["id"]}'} for e in original_episodes])
    rows = []
    for device in ('cpu','mps'):
        if device == 'mps' and not torch.backends.mps.is_available():
            continue
        for minibatch in minibatches:
            durations, errors = [], []
            for repeat in range(repeats+1):
                root = output/f'{device}-batch{minibatch}-{repeat}'
                (root/'models').mkdir(parents=True)
                shutil.copy2(checkpoint,root/'models'/(FMT+'.pt'))
                model = Model(root,FMT)
                compute = Compute(backend=device)
                compute.apply()
                if device == 'mps':
                    torch.mps.synchronize()
                start = time.perf_counter()
                try:
                    model.set_device(device)
                    trained = model.train(episodes,epochs=3,minibatch_size=minibatch)
                    if not trained['trained'] or trained['excluded_episodes']:
                        raise RuntimeError('likelihood check failed on '+device)
                    if device == 'mps':
                        torch.mps.synchronize()
                    if repeat:
                        durations.append(time.perf_counter()-start)
                except (RuntimeError,NotImplementedError) as error:
                    errors.append(str(error))
                finally:
                    model.set_device('cpu')
                    if device == 'mps':
                        torch.mps.empty_cache()
            rows.append({'backend':device,'minibatch_size':minibatch,'seconds':float(np.median(durations)) if durations else None,
                         'steps':sum(len(e['steps']) for e in episodes),'unique_rollout_steps':original_steps,
                         'shape_replication_for_timing_only':len(episodes)>len(original_episodes),
                         'errors':errors,'warmup_runs':1,'timed_runs':len(durations)})
            write_json(output/'training.json',rows)
    return rows


async def prepare_scout(corpus, output, epochs=3, backend='cpu'):
    """Re-extract conservative pre-turn labels from real logs, split by complete battle."""
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    source = Store.read_only(Path(corpus)/'inputs')
    store = Store(output)
    reports = []
    try:
        for fmt in ('gen9championsvgc2026regmb',FMT):
            features = await load_dex(fmt)
            imported = []
            for row in source.db.execute('SELECT * FROM public_battles WHERE format=? ORDER BY id',(fmt,)):
                imported.append(ingest_public(store,fmt,json.loads(row['log']),row['source'],row['id'],features))
            if backend == 'mps':
                torch.mps.set_per_process_memory_fraction(.5)
            scout = Scout(store,fmt,features)
            training = scout.train(epochs=epochs,device=backend,checkpoint=Compute(backend=backend).check)
            reports.append({'format':fmt,'imported_battles':sum(r['imported'] for r in imported),
                            'labels':sum(r.get('samples',0) for r in imported),'training':training})
            write_json(output/'scout-report.json',reports)
    finally:
        source.close()
        store.close()
    return reports


def compare_scout(output, teams, compute, checkpoint, scout_root, count=100, seed=4000000000000):
    """Matched teams/seeds/sides/checkpoint, changing only replay-trained scout inputs."""
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    empty = output/'empty'
    Store(empty).close()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=compute.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for label, inputs in (('without_scout',empty),('with_scout',Path(scout_root))):
            root = output/label
            (root/'models').mkdir(parents=True)
            shutil.copy2(checkpoint,root/'models'/(FMT+'.pt'))
            rows.extend([{**r,'condition':label} for r in evaluate(pool,root,inputs,teams,compute,count,seed,'scout-ablation')])
            write_json(output/'comparison.json',rows)
    write_json(output/'paired.json',paired_scout_scores(rows))
    return rows


def paired_scout_scores(rows):
    result = []
    for opponent in ('random','tactical'):
        baseline = next(r for r in rows if r['condition']=='without_scout' and r['opponent']==opponent)
        candidate = next(r for r in rows if r['condition']=='with_scout' and r['opponent']==opponent)
        key = lambda g: (g['seed'],g['learner_side'],g['learner_team'],g['opponent_team'],g['revision'])
        if [key(g) for g in baseline['games']] != [key(g) for g in candidate['games']]:
            raise ValueError('scout cases are not paired')
        reward = lambda g: 0 if g.get('tie') else 1 if g.get('winner')=='LocalBrain' else -1
        delta = [reward(b)-reward(a) for a,b in zip(baseline['games'],candidate['games'])]
        better, worse = sum(d>0 for d in delta), sum(d<0 for d in delta)
        discordant = better+worse
        p = sum(math.comb(discordant,k) for k in range(better,discordant+1))/2**discordant if discordant else 1.0
        result.append({'opponent':opponent,'paired_games':len(delta),'improved':better,'worsened':worse,
            'unchanged':len(delta)-discordant,'mean_return_delta':float(np.mean(delta)),
            'one_sided_exact_sign_p':p,'interpretation':'exploratory finite scripted comparison; no deployment gate'})
    return result


def tables(root):
    """Read saved results without rerunning simulations."""
    import pandas as pd
    training, episodes, evaluation = [], [], []
    for path in sorted(Path(root).glob('*/updates.json')):
        for row in json.loads(path.read_text()):
            training.append({k:row[k] for k in ('arm','seed','update','training_steps','elapsed_seconds')} |
                {k:row['training'][k] for k in ('loss','mean_entropy','entropy_coef','training_seconds','epochs_completed','device')} |
                {'approx_kl':row['training']['kl_history'][-1],'clip_fraction':row['training']['clip_fraction_history'][-1]})
            running = row['training_steps']-sum(e['steps'] for e in row['episodes'])
            for e in row['episodes']:
                running += e['steps']
                episodes.append({'arm':row['arm'],'seed':row['seed'],'training_steps':running,**e})
    for path in sorted(Path(root).glob('*/evaluations.json')):
        manifest = json.loads((path.parent/'manifest.json').read_text())
        for row in json.loads(path.read_text()):
            evaluation.append({'arm':manifest['arm']['name'],'seed':manifest['seed'],**{k:v for k,v in row.items() if k!='games'}})
    return pd.DataFrame(training),pd.DataFrame(episodes),pd.DataFrame(evaluation)


def main():
    import asyncio
    path = REPO/'data/offline-corpus/macintosh-compute.json'
    tuned = json.loads(path.read_text()) if path.exists() else {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('smoke','pilot','full'),default='pilot')
    p.add_argument('--workers',type=int,default=tuned.get('workers',min(8,max(1,(os.cpu_count() or 2)-1))))
    p.add_argument('--backend',choices=('cpu','mps'),default=tuned.get('backend','cpu'))
    a = p.parse_args()
    prepared = asyncio.run(prepare_teams())
    settings = {'smoke':(1,32,4,(0,)), 'pilot':(3,128,8,(0,)), 'full':(30,None,100,(0,1,2))}
    updates,steps,evaluation,seeds = settings[a.mode]
    arms = [replace(arm,target_steps=steps) if steps else arm for arm in ARMS]
    run_matrix(a.output,prepared['teams'],Compute(workers=a.workers,backend=a.backend),arms=arms,seeds=seeds,
               updates=updates,eval_games=evaluation,eval_every=1 if a.mode!='full' else 5)

if __name__ == '__main__':
    main()
