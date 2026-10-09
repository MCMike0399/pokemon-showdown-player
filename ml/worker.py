"""Bounded background feeds, parallel rollouts and evaluated checkpoint promotion."""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from ml.reload import source_generation, SourceGuard, SourceChanged, pin_runtime
LOADED_GENERATION = os.environ.get('PS_SOURCE_GENERATION') or source_generation()

from ml.continuous import LearningConfig, queue_daily
from ml.resources import ResourcePolicy, ResourceDeferred, BackgroundGuard, backend_for
from ml.storage import Store, WriterBusy, DEFAULT_ROOT, fingerprint, now
from ml.data_budget import storage_status
from ml.input_snapshot import freeze_inputs


def rollout(task: dict):
    """Spawned CPU process: one simulator game, frozen collecting policy."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PS_TORCH_THREADS"] = "1"
    expected = task.get('source_generation', LOADED_GENERATION)
    try:
        SourceGuard(expected)(force=True)
        if expected != LOADED_GENERATION:
            raise SourceChanged('collector process belongs to another source generation')
        pin_runtime(expected)
    except SourceChanged:
        return {'unfinished': True, 'source_changed': True}
    import torch
    from ml.brain import Brain
    from ml.model import Model
    from ml.features import Features
    from ml.simulator import play_local
    torch.set_num_threads(1)
    store = Store(Path(task["root"]))
    inputs = Store.read_only(Path(task['inference_root'])) if task.get('inference_root') else None
    brain = Brain(store, Features.cached(task["format"]), inference_store=inputs)
    model = Model(Path(task["checkpoint_root"]), task["format"])
    brain.models[task["format"]] = model
    opponent_model = Model(Path(task['opponent_checkpoint_root']), task['format']) if task.get('opponent_checkpoint_root') else None
    # These actors sample only on CPU. Seed its generator directly instead of
    # touching accelerator generators through the global manual_seed API.
    torch.random.default_generator.manual_seed(task["seed"])
    try:
        return asyncio.run(play_local(brain, task["format"], task["team1"], task["team2"],
                                     opponent=task.get("opponent", "heuristic"), seed=task["seed"],
                                     training=task["training"], learner_side=task["side"],
                                     sample_actions=task.get("sample_actions", False), opponent_model=opponent_model,
                                     open_team_sheets=task.get('open_team_sheets', True),
                                     seed_encoding=task.get('seed_encoding', 'legacy')))
    finally:
        store.close()
        if inputs:
            inputs.close()


async def parallel_games(tasks: list[dict], policy: ResourcePolicy, deadline: float):
    results = []
    # Recheck host headroom between small waves. No unlimited worker pool.
    cursor = 0
    initial = policy.sample()
    capacity = initial["allowed_workers"]
    capacity = min(capacity, getattr(policy, '_simulator_cap', capacity))
    if not capacity:
        return {"games": results, "deferred": True, "resources": initial}
    context = multiprocessing.get_context("spawn")
    resident = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=capacity, mp_context=context) as pool:
        while cursor < len(tasks) and time.monotonic() < deadline:
            guard = getattr(policy, '_source_guard', None)
            if guard:
                try:
                    guard(force=True)
                except SourceChanged:
                    return {'games': results, 'deferred': True, 'source_changed': True}
            if hasattr(policy, '_simulator_cap') and not LearningConfig.load(Path(tasks[cursor]['root'])).enabled:
                return {'games': results, 'deferred': True, 'reason': 'background learning disabled'}
            disk = storage_status(Path(tasks[cursor]['root']), policy.max_disk_gb)
            if not disk['background_allowed']:
                return {'games': results, 'deferred': True, 'reason': 'live storage reserve protected', 'storage': disk}
            resources = policy.sample(resident_workers=resident)
            workers = min(capacity, resources["allowed_workers"])
            if workers == 0:
                return {"games": results, "deferred": True, "resources": resources}
            from ml.pipeline import simulator_slots
            slot_root = Path(tasks[cursor].get('simulator_slots_root', tasks[cursor]['root']))
            with simulator_slots(slot_root, policy.max_workers, workers) as slots:
                if not slots:
                    return {'games': results, 'deferred': True, 'reason': 'shared simulator slots in use'}
                wave = tasks[cursor:cursor+slots]
                cursor += len(wave)
                loop = asyncio.get_running_loop()
                calls = [loop.run_in_executor(pool, rollout, task) for task in wave]
                # Each simulator has its own bounded read/decision limits. Pool
                # jobs finish before releasing the wave's shared CPU slots.
                outputs = await asyncio.gather(*calls, return_exceptions=True)
                workers = slots
            resident = max(resident, workers)
            for output in outputs:
                if isinstance(output, dict) and output.get('source_changed'):
                    return {'games': results, 'deferred': True, 'source_changed': True}
                if isinstance(output, BaseException):
                    results.append({"unfinished": True, "error": str(output)[:300], "rejected_actions": 1})
                else:
                    results.append(output)
    return {"games": results, "deferred": cursor < len(tasks)}


def snapshot(source: Path, target_root: Path, fmt: str):
    target = target_root/"models"/(fmt+".pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def game_tasks(root, checkpoint_root, fmt, teams, games, training, seed_base, opponent_root,
               learner_team=None, curriculum='legacy', open_team_sheet_probability=.08):
    """Fixed opponent identity remains independent of the candidate checkpoint."""
    generation = source_generation()
    tasks = [{'root': str(root), 'checkpoint_root': str(checkpoint_root), 'format': fmt,
             'source_generation': generation,
             'team1': learner_team if learner_team is not None else teams[i % len(teams)],
             'team2': teams[(i + 1) % len(teams)],
             'seed': seed_base + i, 'side': 'p2' if (i + i // len(teams)) % 2 else 'p1',
             'training': training, 'opponent': 'random' if i % 4 == 0 else 'self' if i % 4 == 1 else 'heuristic',
             'opponent_checkpoint_root': str(opponent_root) if i % 4 == 1 else None}
            for i in range(games)]
    if curriculum == 'ladder-v1':
        import random
        cycle = ('self', 'tactical', 'heuristic', 'self', 'tactical', 'random', 'heuristic', 'tactical')
        for i, task in enumerate(tasks):
            # Complete a team sweep before changing opponent policy. The old
            # i%4 schedule confounded opponent identity with team index when
            # the pool size was a multiple of four.
            opponent = cycle[(i // len(teams) + seed_base) % len(cycle)]
            task.update(opponent=opponent,
                opponent_checkpoint_root=str(opponent_root) if opponent == 'self' else None,
                seed_encoding='full-v1',
                open_team_sheets=random.Random((seed_base + i) ^ 0x50A7).random() < open_team_sheet_probability)
    elif curriculum != 'legacy':
        raise ValueError('unknown practice curriculum')
    return tasks


def candidate_teams(store: Store, fmt: str):
    from harness import TeamStore
    from ml.teams import generated_spreads, team_id
    result = []
    teams = TeamStore()
    for name in teams.list():
        team = teams.get(name)
        if team["format"] == fmt:
            result.append(team["sets"])
    for row in store.db.execute("SELECT sets FROM research_teams WHERE format=? ORDER BY fetched DESC LIMIT 12", (fmt,)):
        sets, _ = generated_spreads(json.loads(row["sets"]), fmt)
        result.append(sets)
    # A simulator fixture is a last-resort training seed, never a tournament claim.
    if not result and fmt == "gen9championsvgc2026regmc":
        fixture = Path(__file__).resolve().parents[1]/"examples/champions-rain.json"
        result.append(json.loads(fixture.read_text())["sets"])
    dedup = {team_id(fmt, sets): sets for sets in result}
    return list(dedup.values())[:8]


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


async def training_teams(store, fmt, config):
    """Validate one opponent pool and the exact focused learner team."""
    from ml.simulator import validate_team
    teams = []
    for sets in candidate_teams(store, fmt):
        validation = await validate_team(fmt, sets)
        if not validation['errors']:
            teams.append(sets)
    focus, metadata = None, None
    name = config.training_teams.get(fmt)
    if name:
        from harness import TeamStore
        from ml.teams import team_id
        selected = TeamStore().get(name)
        if selected['format'] != fmt:
            raise ValueError('focused training team has a different format')
        validation = await validate_team(fmt, selected['sets'])
        if validation['errors']:
            raise ValueError('focused training team is illegal: ' + '; '.join(validation['errors']))
        focus = selected['sets']
        metadata = {'name': name, 'fingerprint': team_id(fmt, focus)}
    return teams, focus, metadata


async def collect(store, job, config, policy, deadline):
    """CPU-only producer: archive a bounded fresh batch, then hand off learning.

Partial production is completed, not replayed with the same random seed on a
retry. Another distinct job collects a new batch. Interrupted games stay pending
without rewards. Only candidate evaluation resumes previously declared cases.
"""
    from ml.features import Features
    from ml.model import Model
    from ml.simulator import load_dex
    from ml.pipeline import available_experience
    fmt = job['format']
    if (store.root / 'ready' / (fmt + '.json')).exists():
        return {'collected': False, 'reason': 'candidate waiting for a matchmaking boundary'}
    backlog = available_experience(store, fmt, config)
    if backlog['steps'] >= config.max_training_backlog_steps:
        return {'collected': False, 'reason': 'compatible experience backlog full', 'backlog': backlog}
    if not config.simulation_games:
        return {'collected': False, 'reason': 'local production disabled'}
    if not Features.cached(fmt).dex:
        await load_dex(fmt)
    teams, focus, focus_metadata = await training_teams(store, fmt, config)
    if not teams:
        return {'collected': False, 'reason': 'no legal local training teams'}
    work = store.root / 'collections' / fingerprint([job['id'], job.get('attempts', 0)])
    before = work / 'policy'
    # Brief writer lock binds this snapshot to the incumbent revision; collection
    # never holds it during simulation or blocks live recording.
    with store.writer():
        current = Model(store.root, fmt)
        snapshot(current.path, before, fmt)
    inputs = work / 'inputs'
    freeze_inputs(store, inputs, fmt)
    seed = int(fingerprint([job['id'], job.get('attempts', 0)])[:12], 16)
    tasks = game_tasks(store.root, before, fmt, teams, config.simulation_games, True, seed, before,
                       learner_team=focus, curriculum=config.curriculum,
                       open_team_sheet_probability=config.open_team_sheet_probability)
    for task in tasks:
        task['inference_root'] = str(inputs)
    atomic_json(work / 'plan.json', {'tasks': tasks, 'revision': current.revision, 'training_team': focus_metadata})
    output = await parallel_games(tasks, policy, deadline)
    result = {'collected': True, 'collection': output, 'revision': current.revision,
              'training_team': focus_metadata, 'partial': output.get('deferred', False)}
    atomic_json(work / 'report.json', result)
    # Idempotent handoff even if the producer is restarted after writing report.
    store.enqueue('collected-' + fingerprint(str(work)), 'learn', fmt)
    return result


def scout_cycle(root, fmt, backend, policy, deadline):
    """One GPU lane, separate SQLite connection from the evaluator thread."""
    from ml.scout import Scout
    from ml.features import Features
    private = Store(root)
    resources_checkpoint = BackgroundGuard(ResourcePolicy(**asdict(policy)), deadline)
    source_guard = SourceGuard(LOADED_GENERATION)
    def checkpoint():
        source_guard(force=True)
        resources_checkpoint()
    try:
        return Scout(private, fmt, Features.cached(fmt)).train(device=backend,
            duty_fraction=policy.gpu_duty_fraction if backend == 'mps' else 1,
            checkpoint=checkpoint)
    except (ResourceDeferred, TimeoutError):
        return {'deferred': True, 'reason': 'scout yielded to host pressure/deadline; checkpoint unchanged'}
    except (RuntimeError, ValueError) as error:
        return {'trained': False, 'error': type(error).__name__, 'promoted': False}
    finally:
        private.close()


async def evaluate_saved(store, job, config, policy, deadline, work):
    """Use CPU evaluation time for one bounded scout update on the GPU."""
    task = None
    scout_result = None
    report_path = work / 'scout-report.json'
    if isinstance(policy, ResourcePolicy) and getattr(policy, '_pipeline_lane', None) != 'evaluator' and not report_path.exists():
        state = json.loads((work / 'training-state.json').read_text())
        task = asyncio.create_task(asyncio.to_thread(scout_cycle, store.root, job['format'],
                                                     state['backend'], policy, deadline))
    try:
        result = await _evaluate_saved(store, job, config, policy, deadline, work)
    finally:
        if task:
            try:
                scout_result = await task
            except Exception as error:
                scout_result = {'error': type(error).__name__, 'promoted': False}
            if not scout_result.get('deferred'):
                atomic_json(report_path, scout_result)
    if report_path.exists():
        result['scout'] = json.loads(report_path.read_text())
    elif task:
        result['scout'] = scout_result
    atomic_json(work / 'report.json', result)
    return result


async def _evaluate_saved(store, job, config, policy, deadline, work):
    """Resume one durable candidate and declared cases; never retrain on retry."""
    from ml.model import Model
    from ml.promotion import paired_gate, stage
    import hashlib
    state = json.loads((work / 'training-state.json').read_text())
    plan = json.loads((work / 'evaluation-plan.json').read_text())
    generation = plan.get('source_generation') or source_generation()
    if generation != source_generation():
        return {'trained': False, 'reason': 'evaluation source changed; candidate retained for audit, not staged'}
    # Legacy plans have no code fingerprint. Retain their old results, but start
    # a fresh suite in a generation-specific directory rather than mixing them.
    evaluation_root = work / 'evaluations' / generation
    evaluation_root.mkdir(parents=True, exist_ok=True)
    fmt = job['format']
    current = Model(store.root, fmt)
    if current.revision != state['training']['previous_revision']:
        return {'trained': False, 'reason': 'candidate parent advanced; retained for audit'}
    candidate = Model(work / 'candidate', fmt)
    if hashlib.sha256(candidate.path.read_bytes()).hexdigest() != state['candidate_sha256']:
        raise ValueError('saved candidate changed; refusing evaluation resume')
    # Idempotent even if a process died immediately after the candidate save.
    with store.writer():
        store.mark_trained(state['training'].get('consumed', []))
    results = {}
    for label in ('incumbent', 'candidate'):
        path = evaluation_root / (label + '-evaluation.json')
        prior = json.loads(path.read_text()) if path.exists() else {'games': []}
        games = prior['games']
        tasks = plan[label]
        for task in tasks:
            task['source_generation'] = generation
        if len(games) < len(tasks) and time.monotonic() < deadline:
            output = await parallel_games(tasks[len(games):], policy, deadline)
            games.extend(output['games'])
            prior.update(games=games, deferred=output.get('deferred', False), resources=output.get('resources'))
            atomic_json(path, prior)
        results[label] = games
    training = {k: v for k, v in state['training'].items() if k != 'consumed'}
    if any(len(results[label]) != len(plan[label]) for label in results):
        report = {'deferred': True, 'reason': 'candidate evaluation paused; checkpoint and cases retained',
                  'training': training, 'evaluation_progress': {label: len(rows) for label, rows in results.items()},
                  'games_per_policy': plan['games'], 'candidate_checkpoint': str(candidate.path)}
        atomic_json(work / 'report.json', report)
        return report
    if source_generation() != generation:
        return {'trained': False, 'reason': 'evaluation source changed; candidate retained for audit, not staged'}
    evaluation = paired_gate(results['incumbent'], results['candidate'], plan['games'], plan['margin'])
    evaluation['source_generation'] = generation
    evaluation.update(games_per_policy=plan['games'], policy_mode='sampled',
                      same_starting_scout_research=True, team_pool_size=plan['team_pool_size'],
                      training_team=plan.get('training_team'))
    with store.writer():
        current = Model(store.root, fmt)
        staged = source_generation() == generation and current.revision == state['training']['previous_revision'] and stage(store, fmt, current.revision, candidate.path, evaluation)
    backend = state['backend']
    report = {'format': fmt, 'job': job['id'], 'training': training, 'backend': backend,
              'collection': state['collection'], 'evaluation': evaluation, 'promoted': False,
              'staged_for_between_game_promotion': staged, 'candidate_checkpoint': str(candidate.path)}
    atomic_json(work / 'report.json', report)
    return report


def enqueue_evaluation(store, job, work):
    """Recoverable handoff after immutable cases/checkpoint are durable."""
    state = json.loads((work / 'training-state.json').read_text())
    key = 'evaluate-' + job['id']
    with store.writer():
        store.mark_trained(state['training'].get('consumed', []))
        store.enqueue(key, 'evaluate', job['format'], {'training_job': job['id'],
                      'parent_revision': state['training']['previous_revision']})
    training = {k: v for k, v in state['training'].items() if k != 'consumed'}
    return {'training': training, 'backend': state['backend'], 'evaluation_job': key,
            'evaluation_pending': True, 'promoted': False, 'candidate_checkpoint': str(work / 'candidate' / 'models' / (job['format'] + '.pt'))}


async def learn(store: Store, job: dict, config: LearningConfig, policy: ResourcePolicy, deadline: float,
                progress=None):
    from ml.features import Features
    from ml.model import Model
    from ml.simulator import load_dex
    fmt = job["format"]
    split = getattr(policy, '_pipeline_lane', None) == 'learner'
    if (store.root / 'ready' / (fmt + '.json')).exists():
        return {"deferred": True, "reason": "evaluated candidate waiting for a matchmaking boundary; experience retained"}
    if not Features.cached(fmt).dex:
        await load_dex(fmt)
    # Load only after checking headroom. Foreground model keeps its own CPU lane.
    with store.writer():
        incumbent = Model(store.root, fmt)
    work = store.root/"candidates"/fingerprint(job["id"])
    before_root, candidate_root = work/"incumbent", work/"candidate"
    if (work / "training-state.json").exists():
        if split:
            return enqueue_evaluation(store, job, work)
        if progress:
            progress('evaluating', candidate_resume=True)
        return await evaluate_saved(store, job, config, policy, deadline, work)
    if split:
        in_flight = store.db.execute("SELECT COUNT(*) FROM jobs WHERE kind='evaluate' AND format=? AND status IN ('queued','running') AND json_extract(payload,'$.parent_revision')=?",
                                    (fmt, incumbent.revision)).fetchone()[0]
        if in_flight >= config.max_inflight_candidates:
            return {'deferred': True, 'reason': 'candidate evaluation backlog full; fresh experience retained'}
    snapshot(incumbent.path, before_root, fmt)
    teams, focus, focus_metadata = await training_teams(store, fmt, config)
    if not teams:
        return {"trained": False, "reason": "no legal local training teams"}
    seed_base = int(fingerprint(job["id"])[:8], 16) % 50000
    def tasks(checkpoint_root, games, training, seed_offset):
        return game_tasks(store.root, checkpoint_root, fmt, teams, games, training,
                          seed_base + seed_offset, before_root, learner_team=focus,
                          curriculum=config.curriculum, open_team_sheet_probability=config.open_team_sheet_probability)
    collected = None
    if job["kind"] == "practice" and config.simulation_games:
        collected = await parallel_games(tasks(before_root, config.simulation_games, True, 0), policy, deadline)
    episodes = store.episodes(fmt, None if job["payload"].get("imitation") else incumbent.revision)
    if focus_metadata:
        episodes = [e for e in episodes if e.get('team') == focus_metadata['fingerprint']]
    if not episodes:
        return {"trained": False, "reason": "no compatible complete episodes", "collection": collected,
                "training_team": focus_metadata}
    steps = sum(len(e.get('steps', [])) for e in episodes if e.get('on_policy') or job['payload'].get('imitation'))
    if steps < config.min_training_steps:
        return {"trained": False, "reason": "accumulating compatible experience", "steps": steps,
                "required_steps": config.min_training_steps, "collection": collected,
                "training_team": focus_metadata}
    headroom = policy.sample()
    if time.monotonic() >= deadline or not headroom.get('training_allowed', not headroom['deferred']):
        return {"deferred": True, "reason": "resource/time budget reached; experience retained"}
    snapshot(incumbent.path, candidate_root, fmt)
    candidate = Model(candidate_root, fmt)
    backend = backend_for(store.root, policy)
    candidate.set_device(backend)
    if progress:
        progress('training', backend=backend, compatible_steps=steps)
    def report_resources(resources):
        if progress:
            metrics = {}
            if backend == 'mps':
                import torch
                metrics = {'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                           'mps_driver_bytes': torch.mps.driver_allocated_memory()}
            progress('training', backend=backend, compatible_steps=steps, resources=resources, **metrics)
    resources_checkpoint = BackgroundGuard(policy, deadline, report_resources) if isinstance(policy, ResourcePolicy) else None
    source_guard = getattr(policy, '_source_guard', None)
    def checkpoint():
        if source_guard:
            source_guard(force=True)
        if resources_checkpoint:
            resources_checkpoint()
    try:
        training = candidate.train(episodes, epochs=job["payload"].get("epochs", config.training_epochs),
                                   imitation=job["payload"].get("imitation", False),
                                   duty_fraction=policy.gpu_duty_fraction if backend == "mps" else 1,
                                   deadline=deadline, checkpoint=checkpoint,
                                   learning_rate=config.learning_rate, entropy_coef=config.entropy_coef,
                                   minibatch_size=config.minibatch_size, target_kl=config.target_kl)
    except RuntimeError as error:
        if backend != "mps":
            raise
        # Restart from the same incumbent if a Metal kernel/budget is unavailable.
        snapshot(incumbent.path, candidate_root, fmt)
        candidate = Model(candidate_root, fmt)
        training = candidate.train(episodes, epochs=job['payload'].get('epochs', config.training_epochs),
                                   imitation=job['payload'].get('imitation', False), deadline=deadline,
                                   checkpoint=checkpoint, learning_rate=config.learning_rate,
                                   entropy_coef=config.entropy_coef, minibatch_size=config.minibatch_size,
                                   target_kl=config.target_kl)
        training["mps_fallback"] = str(error)[:200]
        backend = "cpu"
    if not training.get("trained"):
        return {"training": training, "collection": collected}
    candidate.set_device("cpu")
    if candidate.feature_profile.startswith('strategic-'):
        from ml.postgame import refresh_knowledge
        training['behavior_feedback'] = refresh_knowledge(store, candidate)
        candidate.save()
    if source_guard:
        source_guard(force=True)
    # Saved candidate uses CPU-compatible load; active checkpoint is untouched.
    eval_tasks = tasks(before_root, config.evaluation_games, False, 100000)
    candidate_tasks = tasks(candidate_root, config.evaluation_games, False, 100000)
    inputs = work / 'evaluation-inputs'
    freeze_inputs(store, inputs, fmt)
    for task in eval_tasks + candidate_tasks:
        task.update(inference_root=str(inputs), sample_actions=True)
    if source_guard:
        source_guard(force=True)
    plan_generation = source_guard.generation if source_guard else source_generation()
    for task in eval_tasks + candidate_tasks:
        task['source_generation'] = plan_generation
    import hashlib
    atomic_json(work / 'evaluation-plan.json', {'incumbent': eval_tasks, 'candidate': candidate_tasks,
                'source_generation': plan_generation,
                'games': config.evaluation_games, 'margin': config.promotion_margin, 'team_pool_size': len(teams),
                'training_team': focus_metadata})
    atomic_json(work / 'training-state.json', {'training': training, 'backend': backend, 'collection': collected,
                'candidate_sha256': hashlib.sha256(candidate.path.read_bytes()).hexdigest()})
    if split:
        result = enqueue_evaluation(store, job, work)
        if progress:
            progress('scout_training', backend=backend, candidate_revision=candidate.revision)
        scout_result = await asyncio.to_thread(scout_cycle, store.root, fmt, backend, policy, deadline)
        if not scout_result.get('deferred'):
            atomic_json(work / 'scout-report.json', scout_result)
        result['scout'] = scout_result
        atomic_json(work / 'handoff-report.json', result)
        return result
    if progress:
        progress('evaluating', backend=backend, candidate_revision=candidate.revision)
    return await evaluate_saved(store, job, config, policy, deadline, work)


async def run(root: Path, once: bool = True, schedule: bool = False, lane: str = 'all'):
    from ml.pipeline import pipeline_running
    if lane not in ('all', 'collector', 'learner', 'evaluator'):
        raise ValueError('lane must be all, collector, learner or evaluator')
    if lane == 'all' and pipeline_running(root):
        return {'skipped': True, 'reason': 'retained pipeline owns background scheduling'}
    config = LearningConfig.load(root)
    policy = ResourcePolicy(**config.resource)
    store = Store(root)
    lock = (root / (lane + '.lock' if lane in ('collector', 'evaluator') else 'worker.lock')).open('a')
    owned = False
    state = {}
    def progress(phase, **values):
        if lane != 'all' and owned:
            if 'last_result' in values:
                from ml.pipeline import job_summary
                values['last_result'] = job_summary(values['last_result'])
            state.update(pid=os.getpid(), phase=phase, updated_at=now(), source_generation=LOADED_GENERATION, **values)
            atomic_json(root / ('pipeline-' + lane + '.json'),
                        state)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"skipped": True, "reason": "learning worker already running"}
        owned = True
        policy._source_guard = SourceGuard(LOADED_GENERATION)
        policy._storage_root = root
        try:
            pin_runtime(LOADED_GENERATION)
        except SourceChanged:
            return {'reload_requested': True}
        if not config.enabled:
            return {"skipped": True, "reason": "continuous learning disabled"}
        if lane != 'all':
            # Two pools together retain at most max_workers processes; their
            # active waves also obey the shared slot cap and host headroom.
            policy._simulator_cap = policy.simulator_workers(lane)
            policy._pipeline_lane = lane
        if schedule:
            queue_daily(store, config)
        status = policy.sample()
        if (status["deferred"] and lane != 'learner') or not status['training_allowed']:
            progress('deferred', resources=status)
            return {"deferred": True, "resources": status}
        disk = storage_status(root, policy.max_disk_gb)
        if not disk['background_allowed']:
            progress('deferred', reason='learning data root exceeds storage budget')
            return {"deferred": True, "reason": "retained model data reached background storage limit; live reserve protected", "storage": disk}
        policy.apply_background(enable_mps=lane in ('all', 'learner'))
        started = time.monotonic()
        deadline = started + config.max_seconds
        reports = []
        while time.monotonic() < deadline:
            if source_generation() != LOADED_GENERATION:
                progress('source_reload_requested')
                break
            if not LearningConfig.load(root).enabled:
                break
            kinds = {'collector': ('practice',), 'learner': ('learn', 'feed'), 'evaluator': ('evaluate',)}.get(lane)
            job = store.claim(config.max_seconds+60, kinds=kinds)
            if not job:
                break
            try:
                progress('collecting' if lane == 'collector' else 'preparing', job=job['id'])
                if job["kind"] == "feed":
                    from ml.feeds import refresh_feeds
                    result = await asyncio.wait_for(refresh_feeds(store, job["format"], config.max_tournament_teams, config.max_replays),
                                                    timeout=max(1, deadline-time.monotonic()))
                    if config.archives:
                        from ml.feeds import archive_feed, ARCHIVES
                        from ml.simulator import load_dex
                        result["archives"] = []
                        for key in config.archives:
                            previous = store.db.execute("SELECT result FROM feed_state WHERE id=?", ("archive-"+key,)).fetchone()
                            if previous:
                                result["archives"].append({"source": key, "cached": True, **json.loads(previous["result"])})
                                continue
                            if time.monotonic() >= deadline or policy.sample()["deferred"]:
                                result["archives"].append({"source": key, "deferred": True})
                                continue
                            try:
                                features = await load_dex(ARCHIVES[key]["format"])
                                archived = await archive_feed(store, key, deadline=deadline)
                                from ml.scout import Scout
                                archived["scout"] = Scout(store, ARCHIVES[key]["format"], features).train(device=backend_for(root,policy), duty_fraction=policy.gpu_duty_fraction, checkpoint=BackgroundGuard(policy, deadline)) if time.monotonic()<deadline else {"deferred": True}
                                if not archived["partial"]:
                                    with store.db:
                                        store.db.execute("INSERT OR REPLACE INTO feed_state VALUES (?,?,?)", ("archive-"+key, now(), json.dumps(archived)))
                                result["archives"].append(archived)
                            except Exception as error:
                                result["archives"].append({"source": key, "error": str(error)[:300]})
                    from ml.scout import Scout
                    from ml.features import Features
                    result["scout"] = Scout(store, job["format"], Features.cached(job["format"])).train(device=backend_for(root,policy), duty_fraction=policy.gpu_duty_fraction, checkpoint=BackgroundGuard(policy, deadline)) if time.monotonic() < deadline else {"deferred": True}
                elif lane == 'collector':
                    result = await collect(store, job, config, policy, deadline)
                elif job['kind'] == 'evaluate':
                    progress('evaluating', job=job['id'])
                    work = store.root / 'candidates' / fingerprint(job['payload']['training_job'])
                    result = await evaluate_saved(store, job, config, policy, deadline, work)
                else:
                    result = await learn(store, job, config, policy, deadline, progress=progress)
                store.complete_job(job["id"], result, retry=result.get("deferred", False))
                progress('job_complete', last_job=job['id'], last_result=result)
                reports.append({"id": job["id"], "result": result})
                if result.get("deferred"):
                    break
            except SourceChanged:
                result = {'deferred': True, 'reason': 'source changed; job retained for a fresh worker'}
                store.complete_job(job['id'], result, retry=True)
                reports.append({'id': job['id'], 'result': result})
                progress('source_reload_requested', last_result=result)
                break
            except ResourceDeferred:
                result = {'deferred': True, 'reason': 'host pressure changed during training; experience retained'}
                store.complete_job(job['id'], result, retry=True)
                reports.append({'id': job['id'], 'result': result})
                progress('deferred', last_job=job['id'], last_result=result)
                break
            except TimeoutError as error:
                result = {"deferred": True, "reason": "bounded cycle time expired; work retained"}
                store.complete_job(job["id"], result, retry=True)
                reports.append({"id": job["id"], "result": result})
                progress('error', job=job['id'], error=result)
                break
            except WriterBusy:
                result = {'deferred': True, 'reason': 'learning writer busy; job retained'}
                store.complete_job(job['id'], result, retry=True)
                reports.append({'id': job['id'], 'result': result})
                progress('deferred', last_job=job['id'], last_result=result)
                break
            except Exception as error:
                result = {"error": str(error)[:500], "type": type(error).__name__}
                store.complete_job(job["id"], result, retry=False)
                reports.append({"id": job["id"], "result": result})
                progress('error', last_job=job['id'], last_result=result)
            if once and not schedule:
                break
        return {"resources": status, "elapsed_seconds": round(time.monotonic()-started, 2), "jobs": reports}
    finally:
        progress('idle')
        store.close()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument('--lane', choices=('all', 'collector', 'learner', 'evaluator'), default='all')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.root, args.once, args.schedule, args.lane)), indent=2))
