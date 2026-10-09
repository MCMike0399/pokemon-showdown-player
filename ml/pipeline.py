"""Retained supervision for CPU collection and a single bounded learning lane.

Live matchmaking remains owned by the existing campaign/MCP process. This module
never logs in, chooses battle actions, or changes a playing checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from ml.reload import source_generation
LOADED_GENERATION = source_generation()

from ml.continuous import LearningConfig, queue_daily
from ml.resources import ResourcePolicy
from ml.storage import DEFAULT_ROOT, Store, fingerprint, now
from ml.data_budget import storage_status


def atomic_status(path: Path, value: dict):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def lock_held(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def pipeline_running(root: Path) -> bool:
    return lock_held(Path(root) / 'pipeline.lock')


def job_summary(result: dict) -> dict:
    """Keep monitoring bounded; full cases and rollouts stay in job reports."""
    summary = {key: result[key] for key in ('reason', 'error', 'type', 'deferred', 'backend',
        'evaluation_pending', 'evaluation_job', 'evaluation_progress', 'promoted',
        'staged_for_between_game_promotion', 'partial', 'steps', 'required_steps') if key in result}
    if result.get('training'):
        summary['training'] = {key: result['training'][key] for key in ('trained', 'episodes', 'steps', 'revision',
            'previous_revision', 'device', 'optimizer_steps', 'training_seconds', 'maximum_collecting_logprob_error',
            'learning_rate', 'entropy_coef', 'minibatch_size', 'mean_entropy', 'epochs_completed',
            'target_kl', 'kl_history', 'clip_fraction_history')
            if key in result['training']}
    if result.get('evaluation'):
        summary['evaluation'] = result['evaluation']
    if result.get('collection'):
        collection = result['collection']
        games = collection.get('games', [])
        summary['collection'] = {'games': len(games) if isinstance(games, list) else games,
                                 'deferred': collection.get('deferred', False)}
    if result.get('scout'):
        summary['scout'] = {key: result['scout'][key] for key in ('trained', 'promoted', 'samples', 'error', 'deferred') if key in result['scout']}
    return summary


@contextmanager
def simulator_slots(root: Path, limit: int, requested: int):
    """Share the simulator cap across collector and evaluator processes.

OS locks release on process exit; no stale PID files or expiring live leases.
Hold only the selected wave's slots, and release before the next host sample.
"""
    if not 1 <= limit <= 8 or requested < 1:
        raise ValueError('simulator limit must be 1..8 and request must be positive')
    directory = Path(root) / 'simulator-slots'
    directory.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for index in range(limit):
            handle = (directory / str(index)).open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            handles.append(handle)
            if len(handles) == requested:
                break
        yield len(handles)
    finally:
        for handle in handles:
            handle.close()


def available_experience(store: Store, fmt: str, config: LearningConfig) -> dict:
    """Read only unused compatible rows, without materializing feature arrays."""
    path = store.root / 'models' / (fmt + '.pt')
    if not path.exists():
        return {'episodes': 0, 'steps': 0, 'revision': None}
    import torch
    revision = torch.load(path, map_location='cpu', weights_only=True)['revision']
    query = "SELECT COUNT(*),COALESCE(SUM(json_array_length(data,'$.steps')),0) FROM episodes WHERE format=? AND status='complete' AND trained=0 AND revision=? AND json_extract(data,'$.on_policy')=1"
    args = [fmt, revision]
    if fmt in config.training_teams:
        from harness import TeamStore
        from ml.teams import team_id
        team = TeamStore().get(config.training_teams[fmt])
        if team['format'] != fmt:
            raise ValueError('focused collection team has a different format')
        query += ' AND team=?'
        args.append(team_id(fmt, team['sets']))
    row = store.db.execute(query, args).fetchone()
    return {'episodes': row[0], 'steps': row[1], 'revision': revision}


def queue_collection(store: Store, config: LearningConfig):
    queued, backlog = [], {}
    for fmt in config.formats:
        experience = available_experience(store, fmt, config)
        backlog[fmt] = experience
        if not config.enabled or not config.simulation_games or (store.root / 'ready' / (fmt + '.json')).exists():
            continue
        # Enough experience for several batches is useful; unlimited production
        # during slow evaluation is redundant and can exhaust unified memory/disk.
        if experience['steps'] >= config.max_training_backlog_steps:
            continue
        pending = store.db.execute("SELECT 1 FROM jobs WHERE format=? AND kind='practice' AND status IN ('queued','running') AND attempts<3 LIMIT 1", (fmt,)).fetchone()
        if not pending:
            key = 'pipeline-collect-' + fmt + '-' + fingerprint(now())
            queued.append(store.enqueue(key, 'practice', fmt))
    return {'queued': queued, 'backlog': backlog}


def status(root: Path = DEFAULT_ROOT) -> dict:
    """Read-only monitoring for OpenClaw, CLI and MCP; never start a player."""
    root = Path(root)
    result = {'supervisor_running': pipeline_running(root), 'lanes': {}}
    for lane in ('collector', 'learner', 'evaluator'):
        path = root / ('pipeline-' + lane + '.json')
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        if saved.get('last_result'):
            saved['last_result'] = job_summary(saved['last_result'])
        result['lanes'][lane] = {**saved, 'lock_held': lock_held(root / ('worker.lock' if lane == 'learner' else lane + '.lock'))}
    path = root / 'pipeline-status.json'
    if path.exists():
        try:
            result['supervisor'] = json.loads(path.read_text())
        except (OSError, ValueError):
            result['supervisor'] = {'unreadable': True}
    database = root / 'experience.sqlite3'
    if database.exists():
        db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            result['queue'] = [dict(r) for r in db.execute('SELECT kind,status,COUNT(*) n FROM jobs GROUP BY kind,status')]
            result['recent_jobs'] = []
            for row in db.execute('SELECT id,kind,status,updated,result FROM jobs ORDER BY updated DESC LIMIT 5'):
                entry = dict(row)
                entry['result'] = job_summary(json.loads(entry['result'])) if entry['result'] else None
                result['recent_jobs'].append(entry)
        finally:
            db.close()
    result['ready'] = sorted(p.name for p in (root / 'ready').glob('*.json') if not p.name.endswith('-last-promotion.json'))
    latest = root / 'campaigns' / 'latest.json'
    if latest.exists():
        try:
            pointer = json.loads(latest.read_text())
            campaign = Path(pointer['directory'])
            saved = json.loads((campaign / 'status.json').read_text())
            result['live_campaign'] = {key: saved.get(key) for key in ('phase', 'updated_at', 'completed', 'w', 'l', 't',
                'active_room', 'actor_revision', 'stop_mode', 'errors', 'source_generation')}
        except (OSError, ValueError, KeyError):
            result['live_campaign'] = {'unreadable': True}
    return result


async def supervise(root: Path = DEFAULT_ROOT, interval: float = 15, max_seconds: float | None = None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {'skipped': True, 'reason': 'pipeline supervisor already running'}
    children = {}
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    started = time.monotonic()
    log_dir = root / 'logs'
    log_dir.mkdir(exist_ok=True)
    store = Store(root)
    policy = None
    reload_requested = False
    disk_checked, root_gb, disk = 0, 0, {}
    try:
        while not stop.is_set() and (max_seconds is None or time.monotonic() - started < max_seconds):
            if source_generation() != LOADED_GENERATION:
                reload_requested = True
                break
            config = LearningConfig.load(root)
            if policy is None or policy.backend != config.resource.get('backend', 'auto') or policy.__dict__.get('_config') != config.resource:
                policy = ResourcePolicy(**config.resource)
                policy._config = config.resource.copy()
            resources = policy.sample()
            if time.monotonic() - disk_checked >= 60:
                disk = storage_status(root, policy.max_disk_gb)
                root_gb = disk['retained_bytes'] / 2**30
                disk_checked = time.monotonic()
            exited = {}
            for lane, process in list(children.items()):
                if process.poll() is not None:
                    exited[lane] = process.returncode
                    del children[lane]
            queue_daily(store, config, practice=False)
            production = {'queued': [], 'backlog': {}}
            if config.enabled and resources['training_allowed'] and disk.get('background_allowed', False):
                production = queue_collection(store, config)
                for lane, kinds in (('collector', ('practice',)), ('learner', ('learn', 'feed')), ('evaluator', ('evaluate',))):
                    if lane in children:
                        continue
                    if lane != 'learner' and resources['deferred']:
                        continue
                    placeholders = ','.join('?' for _ in kinds)
                    pending = store.db.execute("SELECT 1 FROM jobs WHERE kind IN (" + placeholders + ") AND (status='queued' OR (status='running' AND lease_until<?)) AND attempts<3 LIMIT 1", (*kinds, time.time())).fetchone()
                    if pending:
                        with (log_dir / ('pipeline-' + lane + '.log')).open('a') as log:
                            children[lane] = subprocess.Popen([sys.executable, '-m', 'ml.worker', '--root', str(root), '--lane', lane],
                                env={**os.environ, 'PS_SOURCE_GENERATION': LOADED_GENERATION},
                                cwd=str(Path(__file__).resolve().parents[1]), stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            atomic_status(root / 'pipeline-status.json', {'pid': os.getpid(), 'updated_at': now(),
                'source_generation': LOADED_GENERATION,
                'enabled': config.enabled, 'resources': resources, 'children': {lane: p.pid for lane, p in children.items()},
                'data_root_gb': round(root_gb, 3), 'disk_budget_exceeded': disk.get('budget_exceeded', False), 'storage': disk,
                'exited': exited, 'production': production, 'live_player': 'owned by retained campaign, independent of this supervisor'})
            timeout = interval if max_seconds is None else min(interval, max(0, max_seconds - (time.monotonic() - started)))
            try:
                await asyncio.wait_for(stop.wait(), timeout=timeout)
            except TimeoutError:
                pass
        # Workers already have bounded deadlines. Let games/candidate saves finish;
        # do not kill a simulator or discard a terminal trajectory on shutdown.
        while any(p.poll() is None for p in children.values()):
            atomic_status(root / 'pipeline-status.json', {'pid': os.getpid(), 'updated_at': now(),
                'source_generation': LOADED_GENERATION, 'draining': True, 'reload_requested': reload_requested,
                'children': {lane: p.pid for lane, p in children.items() if p.poll() is None}})
            await asyncio.sleep(1)
        atomic_status(root / 'pipeline-status.json', {'pid': os.getpid(), 'updated_at': now(),
            'source_generation': LOADED_GENERATION, 'stopped': True, 'reload_requested': reload_requested})
        return {'stopped': True, 'reload_requested': reload_requested}
    finally:
        store.close()
        lock.close()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--max-seconds', type=float)
    args = parser.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error('--max-seconds must be positive')
    result = status(args.root) if args.status else asyncio.run(supervise(args.root, max_seconds=args.max_seconds))
    print(json.dumps(result, indent=2))
    if result.get('reload_requested'):
        raise SystemExit(75)  # launchd restarts after all old children have drained.
