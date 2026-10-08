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
from pathlib import Path

from ml.continuous import LearningConfig, queue_daily
from ml.resources import ResourcePolicy, backend_for
from ml.storage import Store, WriterBusy, DEFAULT_ROOT, fingerprint, now


def rollout(task: dict):
    """Spawned CPU process: one simulator game, frozen collecting policy."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PS_TORCH_THREADS"] = "1"
    import torch
    from ml.brain import Brain
    from ml.model import Model
    from ml.features import Features
    from ml.simulator import play_local
    torch.set_num_threads(1)
    store = Store(Path(task.get("inference_root", task["root"])))
    brain = Brain(store, Features.cached(task["format"]))
    model = Model(Path(task["checkpoint_root"]), task["format"])
    brain.models[task["format"]] = model
    opponent_model = Model(Path(task['opponent_checkpoint_root']), task['format']) if task.get('opponent_checkpoint_root') else None
    torch.manual_seed(task["seed"])
    try:
        return asyncio.run(play_local(brain, task["format"], task["team1"], task["team2"],
                                     opponent=task.get("opponent", "heuristic"), seed=task["seed"],
                                     training=task["training"], learner_side=task["side"],
                                     sample_actions=task.get("sample_actions", False), opponent_model=opponent_model,
                                     open_team_sheets=task.get('open_team_sheets', True),
                                     seed_encoding=task.get('seed_encoding', 'legacy')))
    finally:
        store.close()


async def parallel_games(tasks: list[dict], policy: ResourcePolicy, deadline: float):
    results = []
    # Recheck host headroom between small waves. No unlimited worker pool.
    cursor = 0
    initial = policy.sample()
    capacity = initial["allowed_workers"]
    if not capacity:
        return {"games": results, "deferred": True, "resources": initial}
    context = multiprocessing.get_context("spawn")
    resident = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=capacity, mp_context=context) as pool:
        while cursor < len(tasks) and time.monotonic() < deadline:
            resources = policy.sample(resident_workers=resident)
            workers = min(capacity, resources["allowed_workers"])
            if workers == 0:
                return {"games": results, "deferred": True, "resources": resources}
            wave = tasks[cursor:cursor+workers]
            cursor += len(wave)
            loop = asyncio.get_running_loop()
            calls = [loop.run_in_executor(pool, rollout, task) for task in wave]
            # Each simulator has its own bounded read/decision limits. Pool jobs
            # finish before closing the wave; host headroom is sampled again next.
            outputs = await asyncio.gather(*calls, return_exceptions=True)
            resident = max(resident, workers)
            for output in outputs:
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
    tasks = [{'root': str(root), 'checkpoint_root': str(checkpoint_root), 'format': fmt,
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


def freeze_inputs(store: Store, destination: Path, fmt: str):
    """Both evaluations see the same research/scout, isolated from live feeds."""
    frozen = Store(destination)
    try:
        with frozen.db:
            for table in ('research_teams', 'documents', 'public_battles', 'scout_samples'):
                rows = store.db.execute(f"SELECT * FROM {table} WHERE format=?", (fmt,)).fetchall()
                frozen.db.execute(f"DELETE FROM {table}")
                if rows:
                    placeholders = ','.join('?' for _ in rows[0])
                    frozen.db.executemany(f"INSERT INTO {table} VALUES ({placeholders})", [tuple(row) for row in rows])
        source = store.root / 'scouts' / (fmt + '.pt')
        if source.exists():
            target = destination / 'scouts' / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    finally:
        frozen.close()


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


async def evaluate_saved(store, job, config, policy, deadline, work):
    """Resume one durable candidate and declared cases; never retrain on retry."""
    from ml.model import Model
    from ml.promotion import paired_gate, stage
    from ml.scout import Scout
    from ml.features import Features
    import hashlib
    state = json.loads((work / 'training-state.json').read_text())
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
    plan = json.loads((work / 'evaluation-plan.json').read_text())
    results = {}
    for label in ('incumbent', 'candidate'):
        path = work / (label + '-evaluation.json')
        prior = json.loads(path.read_text()) if path.exists() else {'games': []}
        games = prior['games']
        tasks = plan[label]
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
    evaluation = paired_gate(results['incumbent'], results['candidate'], plan['games'], plan['margin'])
    evaluation.update(games_per_policy=plan['games'], policy_mode='sampled',
                      same_starting_scout_research=True, team_pool_size=plan['team_pool_size'],
                      training_team=plan.get('training_team'))
    with store.writer():
        current = Model(store.root, fmt)
        staged = current.revision == state['training']['previous_revision'] and stage(store, fmt, current.revision, candidate.path, evaluation)
    backend = state['backend']
    scout = Scout(store, fmt, Features.cached(fmt))
    scout_report = scout.train(device=backend, duty_fraction=policy.gpu_duty_fraction if backend == 'mps' else 1) if time.monotonic() < deadline else {'deferred': True}
    report = {'format': fmt, 'job': job['id'], 'training': training, 'backend': backend,
              'collection': state['collection'], 'evaluation': evaluation, 'promoted': False,
              'staged_for_between_game_promotion': staged, 'candidate_checkpoint': str(candidate.path), 'scout': scout_report}
    atomic_json(work / 'report.json', report)
    return report


async def learn(store: Store, job: dict, config: LearningConfig, policy: ResourcePolicy, deadline: float):
    from ml.features import Features
    from ml.model import Model
    from ml.simulator import load_dex, validate_team
    fmt = job["format"]
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
        return await evaluate_saved(store, job, config, policy, deadline, work)
    snapshot(incumbent.path, before_root, fmt)
    teams = []
    for sets in candidate_teams(store, fmt):
        validation = await validate_team(fmt, sets)
        if not validation["errors"]:
            teams.append(sets)
    if not teams:
        return {"trained": False, "reason": "no legal local training teams"}
    focus, focus_metadata = None, None
    focus_name = config.training_teams.get(fmt)
    if focus_name:
        from harness import TeamStore
        from ml.teams import team_id
        focused = TeamStore().get(focus_name)
        if focused['format'] != fmt:
            raise ValueError('focused training team has a different format')
        validation = await validate_team(fmt, focused['sets'])
        if validation['errors']:
            raise ValueError('focused training team is illegal: ' + '; '.join(validation['errors']))
        focus = focused['sets']
        focus_metadata = {'name': focus_name, 'fingerprint': team_id(fmt, focus)}
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
    if time.monotonic() >= deadline or policy.sample()["deferred"]:
        return {"deferred": True, "reason": "resource/time budget reached; experience retained"}
    snapshot(incumbent.path, candidate_root, fmt)
    candidate = Model(candidate_root, fmt)
    backend = backend_for(store.root, policy)
    candidate.set_device(backend)
    try:
        training = candidate.train(episodes, epochs=job["payload"].get("epochs", 4),
                                   imitation=job["payload"].get("imitation", False),
                                   duty_fraction=policy.gpu_duty_fraction if backend == "mps" else 1,
                                   deadline=deadline)
    except RuntimeError as error:
        if backend != "mps":
            raise
        # Restart from the same incumbent if a Metal kernel/budget is unavailable.
        snapshot(incumbent.path, candidate_root, fmt)
        candidate = Model(candidate_root, fmt)
        training = candidate.train(episodes, epochs=job['payload'].get('epochs', 4),
                                   imitation=job['payload'].get('imitation', False), deadline=deadline)
        training["mps_fallback"] = str(error)[:200]
        backend = "cpu"
    if not training.get("trained"):
        return {"training": training, "collection": collected}
    candidate.set_device("cpu")
    # Saved candidate uses CPU-compatible load; active checkpoint is untouched.
    eval_tasks = tasks(before_root, config.evaluation_games, False, 100000)
    candidate_tasks = tasks(candidate_root, config.evaluation_games, False, 100000)
    inputs = work / 'evaluation-inputs'
    freeze_inputs(store, inputs, fmt)
    for task in eval_tasks + candidate_tasks:
        task.update(inference_root=str(inputs), sample_actions=True)
    import hashlib
    atomic_json(work / 'evaluation-plan.json', {'incumbent': eval_tasks, 'candidate': candidate_tasks,
                'games': config.evaluation_games, 'margin': config.promotion_margin, 'team_pool_size': len(teams),
                'training_team': focus_metadata})
    atomic_json(work / 'training-state.json', {'training': training, 'backend': backend, 'collection': collected,
                'candidate_sha256': hashlib.sha256(candidate.path.read_bytes()).hexdigest()})
    return await evaluate_saved(store, job, config, policy, deadline, work)


async def run(root: Path, once: bool = True, schedule: bool = False):
    config = LearningConfig.load(root)
    policy = ResourcePolicy(**config.resource)
    store = Store(root)
    lock = (root/"worker.lock").open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"skipped": True, "reason": "learning worker already running"}
        if not config.enabled:
            return {"skipped": True, "reason": "continuous learning disabled"}
        if schedule:
            queue_daily(store, config)
        status = policy.sample()
        if status["deferred"]:
            return {"deferred": True, "resources": status}
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())/2**30
        if size > policy.max_disk_gb:
            return {"deferred": True, "reason": "learning data root exceeds disk budget; preserve or prune old candidate artifacts"}
        policy.apply_background()
        started = time.monotonic()
        deadline = started + config.max_seconds
        reports = []
        while time.monotonic() < deadline:
            job = store.claim(config.max_seconds+60)
            if not job:
                break
            try:
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
                                archived["scout"] = Scout(store, ARCHIVES[key]["format"], features).train(device=backend_for(root,policy), duty_fraction=policy.gpu_duty_fraction) if time.monotonic()<deadline else {"deferred": True}
                                if not archived["partial"]:
                                    with store.db:
                                        store.db.execute("INSERT OR REPLACE INTO feed_state VALUES (?,?,?)", ("archive-"+key, now(), json.dumps(archived)))
                                result["archives"].append(archived)
                            except Exception as error:
                                result["archives"].append({"source": key, "error": str(error)[:300]})
                    from ml.scout import Scout
                    from ml.features import Features
                    result["scout"] = Scout(store, job["format"], Features.cached(job["format"])).train(device=backend_for(root,policy), duty_fraction=policy.gpu_duty_fraction) if time.monotonic() < deadline else {"deferred": True}
                else:
                    result = await learn(store, job, config, policy, deadline)
                store.complete_job(job["id"], result, retry=result.get("deferred", False))
                reports.append({"id": job["id"], "result": result})
                if result.get("deferred"):
                    break
            except TimeoutError as error:
                result = {"deferred": True, "reason": "bounded cycle time expired; work retained"}
                store.complete_job(job["id"], result, retry=True)
                reports.append({"id": job["id"], "result": result})
                break
            except WriterBusy:
                result = {'deferred': True, 'reason': 'learning writer busy; job retained'}
                store.complete_job(job['id'], result, retry=True)
                reports.append({'id': job['id'], 'result': result})
                break
            except Exception as error:
                result = {"error": str(error)[:500], "type": type(error).__name__}
                store.complete_job(job["id"], result, retry=False)
                reports.append({"id": job["id"], "result": result})
            if once and not schedule:
                break
        return {"resources": status, "elapsed_seconds": round(time.monotonic()-started, 2), "jobs": reports}
    finally:
        store.close()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.root, args.once, args.schedule)), indent=2))
