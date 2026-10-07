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
from ml.storage import Store, DEFAULT_ROOT, fingerprint, now


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
    store = Store(Path(task["root"]))
    brain = Brain(store, Features.cached(task["format"]))
    model = Model(Path(task["checkpoint_root"]), task["format"])
    brain.models[task["format"]] = model
    torch.manual_seed(task["seed"])
    try:
        return asyncio.run(play_local(brain, task["format"], task["team1"], task["team2"],
                                     opponent=task.get("opponent", "heuristic"), seed=task["seed"],
                                     training=task["training"], learner_side=task["side"]))
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


async def learn(store: Store, job: dict, config: LearningConfig, policy: ResourcePolicy, deadline: float):
    from ml.features import Features
    from ml.model import Model
    from ml.simulator import load_dex, validate_team
    from ml.scout import Scout
    fmt = job["format"]
    if not Features.cached(fmt).dex:
        await load_dex(fmt)
    # Load only after checking headroom. Foreground model keeps its own CPU lane.
    with store.writer():
        incumbent = Model(store.root, fmt)
    work = store.root/"candidates"/fingerprint(job["id"])
    before_root, candidate_root = work/"incumbent", work/"candidate"
    snapshot(incumbent.path, before_root, fmt)
    teams = []
    for sets in candidate_teams(store, fmt):
        validation = await validate_team(fmt, sets)
        if not validation["errors"]:
            teams.append(sets)
    if not teams:
        return {"trained": False, "reason": "no legal local training teams"}
    seed_base = int(fingerprint(job["id"])[:8], 16) % 50000
    def tasks(checkpoint_root, games, training, seed_offset):
        return [{"root": str(store.root), "checkpoint_root": str(checkpoint_root), "format": fmt,
                 "team1": teams[i % len(teams)], "team2": teams[(i+1) % len(teams)],
                 "seed": seed_base+seed_offset+i, "side": "p2" if i%2 else "p1", "training": training,
                 "opponent": "random" if training and i%4 == 0 else "heuristic"}
                for i in range(games)]
    collected = None
    if job["kind"] == "practice" and config.simulation_games:
        collected = await parallel_games(tasks(before_root, config.simulation_games, True, 0), policy, deadline)
    episodes = store.episodes(fmt, None if job["payload"].get("imitation") else incumbent.revision)
    if not episodes:
        return {"trained": False, "reason": "no compatible complete episodes", "collection": collected}
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
        training = candidate.train(episodes, deadline=deadline)
        training["mps_fallback"] = str(error)[:200]
        backend = "cpu"
    if not training.get("trained"):
        return {"training": training, "collection": collected}
    candidate.set_device("cpu")
    # Saved candidate uses CPU-compatible load; active checkpoint is untouched.
    eval_tasks = tasks(before_root, config.evaluation_games, False, 100000)
    candidate_tasks = tasks(candidate_root, config.evaluation_games, False, 100000)
    base_eval = await parallel_games(eval_tasks, policy, deadline)
    new_eval = await parallel_games(candidate_tasks, policy, deadline)
    base_games, new_games = base_eval["games"], new_eval["games"]
    complete = len(base_games) == len(new_games) == config.evaluation_games
    clean = complete and all(not g.get("unfinished") and not g.get("rejected_actions") for g in base_games+new_games)
    base_wins = sum(g.get("winner") == "LocalBrain" for g in base_games)
    new_wins = sum(g.get("winner") == "LocalBrain" for g in new_games)
    passed = clean and new_wins-base_wins >= max(1, int(config.evaluation_games*config.promotion_margin+0.999))
    with store.writer():
        current = Model(store.root, fmt)
        active_live = store.db.execute("SELECT 1 FROM episodes WHERE format=? AND source='ladder' AND status='pending' AND created>? LIMIT 1",
                                      (fmt, datetime_cutoff())).fetchone()
        promoted = passed and current.revision == incumbent.revision and not active_live
        if promoted:
            temporary = current.path.with_suffix(".promote.tmp")
            shutil.copy2(candidate.path, temporary)
            temporary.replace(current.path)
        # Consumption is tied to the durable candidate, even if it is rejected.
        store.mark_trained(training.pop("consumed"))
    scout = Scout(store, fmt, Features.cached(fmt))
    scout_report = scout.train(device=backend, duty_fraction=policy.gpu_duty_fraction if backend == "mps" else 1) if time.monotonic() < deadline else {"deferred": True}
    report = {"format": fmt, "job": job["id"], "training": training, "backend": backend,
              "collection": collected, "evaluation": {"games_per_policy": config.evaluation_games,
              "incumbent_wins": base_wins, "candidate_wins": new_wins, "complete": complete,
              "clean": clean, "same_seeds_and_sides": True, "team_pool_size": len(teams)},
              "promoted": promoted, "promotion_blocked_by_live_game": bool(active_live),
              "candidate_checkpoint": str(candidate.path), "scout": scout_report}
    (work/"report.json").write_text(json.dumps(report, indent=2))
    return report


def datetime_cutoff():
    from datetime import datetime, timezone, timedelta
    return (datetime.now(timezone.utc)-timedelta(minutes=30)).isoformat()


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
