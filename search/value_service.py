"""Always-on learning loop for the search's value function.

    .venv/bin/python -m search.value_service            # supervise (launchd)
    .venv/bin/python -m search.value_service --status   # read-only status

Every tick (default 20 s) it does whatever the host can afford:

1. Ingest: finished live search games -> labelled positions
   (`search/ledger_positions.py`), so every live game becomes data.
2. Produce (CPU): self-play shards from `search/selfplay.cjs`, the deployed team
   vs the real-team pool, at most one new producer per tick and only while
   `ml.resources.ResourcePolicy` admits another worker. Producers and gates
   take locks from the SAME simulator slot pool as the PPO collector and
   evaluator (`data/ml/simulator-slots`), so the two pipelines share cores
   instead of oversubscribing them. Self-play uses the promoted value, so data
   improves as the evaluation improves.
3. Train (MPS): when enough new rows or live games arrived, train a candidate
   while holding the PPO learner's `worker.lock`, so the GPU is used by one
   learner at a time.
4. Gate (CPU): a candidate that beats the calibrated hand evaluation on held-out
   self-play AND live games is gated by paired real-team games, in two stages:
   development, then confirmation on fresh seeds. Both must pass.
5. Promote: atomic pointer `data/ml/value/current.json`; the live runner reads
   it at the next matchmaking boundary. At most `max_promotions_per_day`.

Child processes inherit their lock file descriptors, so a slot stays held for
exactly the life of the child, even if this supervisor restarts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search.value_store import (BACKGROUND_PGIDS, VALUE_ROOT, POINTER, _command, atomic_json,  # noqa: E402
                                current_value, dir_bytes, pack_shard, promote, resume_if_stale)

DATA = ROOT / 'data' / 'ml'
PY = sys.executable
TASKPOLICY = '/usr/sbin/taskpolicy'
DEFAULTS = {
    'enabled': True,
    'interval_seconds': 20,
    'producers': 4,                 # max concurrent self-play processes
    'games_per_shard': 20,          # ~20 min per shard on efficiency cores: yields slots promptly
    'focus_prob': 0.6,
    'selfplay_uses_current': True,  # self-play with the promoted value (expert iteration)
    'max_selfplay_gb': 8.0,         # stop producing above this (data kept, never deleted)
    'slot_limit_extra': 2,          # value lanes may use PPO max_workers + this many slots
    'ingest_seconds': 120,
    'train_every_rows': 60000,
    'train_every_live_games': 15,
    'train_max_rows': 1500000,
    'train_args': [],
    'train_retry_seconds': 1800,
    'gate_pairs': 80,
    'gate_alpha': 0.05,
    'gate_slots': 3,                # concurrent gate games
    'gate_min_slots': 2,
    'gate_engines': 1,
    'max_promotions_per_day': 1,
    # Normal QoS uses every core; the live runner pauses these process groups while
    # it decides (search.value_store.LivePriority). 'background' confines a role to
    # the efficiency cores; 'utility' was measured to starve (priority 4, ~17% core).
    'qos': {'producer': None, 'gate': None},
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads((VALUE_ROOT / 'config.json').read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def _alive(pid: int | None, marker: str = 'search/') -> bool:
    """Running, not a zombie, and still our program (pids are reused)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = Path('/proc') / str(pid) / 'stat'
    if stat.exists():
        try:
            if stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                return False
        except (OSError, IndexError):
            return False
    else:
        try:
            out = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
            if not out or out.startswith('Z'):
                return False
        except OSError:
            pass
    return marker in _command(pid)


class Slots:
    """Non-blocking fcntl locks in the shared simulator-slot directory."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def acquire(self, limit: int, n: int) -> list:
        handles = []
        # Highest index first: PPO lanes only use 0..max_workers-1, so the value
        # lanes take the extra slots before competing for PPO's.
        for index in reversed(range(limit)):
            if len(handles) == n:
                break
            handle = (self.dir / str(index)).open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            handles.append(handle)
        return handles


class Service:
    def __init__(self, data: Path = DATA):
        self.data = data
        self.vroot = VALUE_ROOT
        for sub in ('selfplay', 'live', 'candidates', 'gates', 'logs'):
            (self.vroot / sub).mkdir(parents=True, exist_ok=True)
        self.state_path = self.vroot / 'service-state.json'
        try:
            self.state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            self.state = {}
        self.state.setdefault('producers', {})
        self.state.setdefault('candidates', [])
        self.state.setdefault('trained_rows', 0)
        self.state.setdefault('trained_live_games', 0)
        self.state.setdefault('shard_seq', int(time.time()))
        self.children: dict[str, subprocess.Popen] = {}
        self.slots = Slots(data / 'simulator-slots')
        self.policy = None
        self.last_ingest = 0.0
        self.events: list[str] = []

    # ------------------------------------------------------------ helpers
    def save(self):
        atomic_json(self.state_path, self.state)

    def note(self, text: str):
        line = time.strftime('%Y-%m-%dT%H:%M:%S') + ' ' + text
        self.events = (self.events + [line])[-30:]
        print(line, flush=True)

    def resources(self) -> dict:
        from ml.continuous import LearningConfig
        from ml.resources import ResourcePolicy
        config = LearningConfig.load(self.data)
        if self.policy is None or getattr(self.policy, '_config', None) != config.resource:
            self.policy = ResourcePolicy(**config.resource)
            self.policy._config = dict(config.resource)
        status = self.policy.sample()
        status['slot_pool'] = self.policy.max_workers
        return status

    def spawn(self, name: str, argv: list[str], fds: list, log: Path, nice: int = 10,
              qos: str | None = 'background') -> subprocess.Popen:
        # QoS, not nice, decides who gets the performance cores on Apple silicon:
        # background work at default QoS pushed live decisions from 2.3 s to 10 s
        # median. 'background' = efficiency cores only; 'utility' may use idle
        # performance cores but yields to the live runner's default-QoS engines.
        if qos and os.path.exists(TASKPOLICY):
            argv = [TASKPOLICY, '-b', *argv] if qos == 'background' else [TASKPOLICY, '-c', qos, *argv]
        with log.open('a') as fh:
            proc = subprocess.Popen(argv, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=fh, stderr=fh,
                                    pass_fds=[h.fileno() for h in fds], preexec_fn=lambda: os.nice(nice),
                                    start_new_session=True,  # launchd kills the job's process group on restart
                                    env={**os.environ, 'OMP_NUM_THREADS': '2'})
        for h in fds:
            h.close()  # the child's inherited descriptor now owns the lock
        self.children[name] = proc
        return proc

    def selfplay_rows(self) -> int:
        """Rows in finished shards plus complete lines of shards still being written
(the trainer reads both; a partial last line is skipped)."""
        rows = 0
        known = self.state.get('shard_rows', {})
        for p in (self.vroot / 'selfplay').glob('*.jsonl'):
            if p.name in known:
                rows += known[p.name] or 0
            else:
                with p.open('rb') as fh:
                    rows += sum(chunk.count(b'\n') for chunk in iter(lambda: fh.read(1 << 20), b''))
        return rows

    def live_games(self) -> int:
        return len(list((self.vroot / 'live').glob('live-*.jsonl')))

    # ------------------------------------------------------------ 1. ingest
    def ingest(self):
        if time.time() - self.last_ingest < load_config()['ingest_seconds'] or 'ingest' in self.children:
            return
        runs = self.data / 'browser-runs'
        newest = max((p.stat().st_mtime for p in runs.glob('*/events.jsonl')), default=0)
        if newest <= self.state.get('ingested_mtime', 0):
            self.last_ingest = time.time()
            return
        self.state['ingested_mtime'] = newest
        self.last_ingest = time.time()
        self.spawn('ingest', [PY, str(ROOT / 'search' / 'ledger_positions.py'), '--worlds', '4'], [],
                   self.vroot / 'logs' / 'ingest.log', nice=5)

    # ------------------------------------------------------------ 2. produce
    def produce(self, cfg: dict, res: dict):
        running = [s for s, i in self.state['producers'].items()
                   if f'prod-{s}' in self.children or _alive(i.get('pid'))]
        if len(running) >= cfg['producers'] or not res['training_allowed'] or res['allowed_workers'] < 1:
            return
        if self.state.get('gate') and 'gate' not in self.children and not self.state['gate'].get('pid'):
            return  # a gate is waiting for slots: let running shards finish and free them
        if dir_bytes(self.vroot / 'selfplay') > cfg['max_selfplay_gb'] * 2**30:
            return
        handles = self.slots.acquire(res['slot_pool'] + cfg['slot_limit_extra'], 1)
        if not handles:
            return
        self.state['shard_seq'] += 1
        seq = self.state['shard_seq']
        shard = self.vroot / 'selfplay' / f'sp-{seq}.jsonl'
        argv = ['node', '--max-old-space-size=768', str(ROOT / 'search' / 'selfplay.cjs'),
                str(self.vroot / 'pool-teams.json'), str(self.vroot / 'focus-team.json'),
                str(cfg['games_per_shard']), str(seq % 2_000_000_000), str(shard),
                '--focus-prob', str(cfg['focus_prob']), '--tag', f'sp{seq}']
        value = current_value() if cfg['selfplay_uses_current'] else {'path': None}
        if value.get('path'):
            argv += [value['path'], '--value-beta', str(value['beta'])]  # optional 6th positional
        proc = self.spawn(f'prod-{seq}', argv, handles, self.vroot / 'logs' / 'selfplay.log', nice=10,
                          qos=cfg['qos']['producer'])
        self.state['producers'][str(seq)] = {'pid': proc.pid, 'shard': shard.name, 'started': time.time(),
                                             'value': value.get('sha256')}

    def finish_producer(self, name: str, code: int):
        seq = name.split('-', 1)[1]
        info = self.state['producers'].pop(seq, {})
        shard = self.vroot / 'selfplay' / info.get('shard', f'sp-{seq}.jsonl')
        if shard.exists():
            cache = pack_shard(shard)
            rows = sum(1 for _ in shard.open())
            self.state.setdefault('shard_rows', {})[shard.name] = rows
            self.note(f'shard {shard.name} done code={code} rows={rows} cached={bool(cache)}')

    # ------------------------------------------------------------ 3. train
    def maybe_train(self, cfg: dict, res: dict):
        if 'train' in self.children or not res['training_allowed']:
            return
        if time.time() - self.state.get('train_failed_at', 0) < cfg['train_retry_seconds']:
            return  # a failed run reloads up to train_max_rows: never retry in a tight loop
        rows, live = self.selfplay_rows(), self.live_games()
        new_rows = rows - self.state['trained_rows']
        new_live = live - self.state['trained_live_games']
        if new_rows < cfg['train_every_rows'] and not (new_live >= cfg['train_every_live_games'] and rows >= 20000):
            return
        lock = (self.data / 'worker.lock').open('a')  # the PPO learner's lock: one GPU learner at a time
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return
        self.state['pending_train'] = {'rows': rows, 'live': live, 'started': time.time()}
        out = self.vroot / 'logs' / f'train-{int(time.time())}.json'
        argv = [PY, str(ROOT / 'search' / 'train_value.py'), '--selfplay', str(self.vroot / 'selfplay' / '*.jsonl'),
                '--live', str(self.vroot / 'live'), '--out', str(self.vroot / 'candidates'),
                '--max-rows', str(cfg['train_max_rows']), *cfg['train_args']]
        self.state['pending_train']['report'] = str(out)
        self.spawn('train', ['/bin/sh', '-c', 'exec ' + ' '.join(_q(a) for a in argv) + ' > ' + _q(str(out))], [lock],
                   self.vroot / 'logs' / 'train.log', nice=5, qos=None)
        self.note(f'training on {rows} self-play rows, {live} live games')

    def finish_train(self, code: int):
        pending = self.state.pop('pending_train', {})
        try:
            report = json.loads(Path(pending['report']).read_text().strip().splitlines()[-1])
        except (OSError, ValueError, KeyError, IndexError):
            self.note(f'training failed code={code}; retry after backoff')
            self.state['train_failed_at'] = time.time()
            return
        if 'path' not in report:
            self.note(f"training produced no candidate: {str(report)[:200]}")
            self.state['train_failed_at'] = time.time()
            return
        self.state['trained_rows'] = pending['rows']
        self.state['trained_live_games'] = pending['live']
        cand = {'path': report['path'], 'sha256': report['sha256'], 'useful': report['useful'],
                'beta': report.get('recommended_beta', 1.0), 'val': report.get('val'), 'live_test': report.get('live_test'),
                'rows': report.get('rows'), 'trained_at': time.time(), 'status': 'trained'}
        if not cand['useful'] or not cand['beta']:
            cand['status'] = 'rejected: does not beat calibrated hand evaluation on held-out games'
        self.state['candidates'].append(cand)
        self.state['candidates'] = self.state['candidates'][-50:]
        self.note(f"candidate {cand['sha256'][:12]} {cand['status']} beta={cand['beta']} "
                  f"live_test={(cand['live_test'] or {}).get('learned')}")

    # ------------------------------------------------------------ 4. gate / 5. promote
    def maybe_gate(self, cfg: dict, res: dict):
        if 'gate' in self.children:
            return
        gate = self.state.get('gate')
        if gate and gate.get('pid'):
            if _alive(gate['pid']):
                return  # a gate started by a previous supervisor still owns its slots and results file
            gate.pop('pid')
            self.finish_gate(-1, cfg)
            gate = self.state.get('gate')
        if gate is None:
            waiting = [c for c in self.state['candidates'] if c['status'] == 'trained']
            if not waiting:
                return
            day = time.time() - 86400
            promos = [c for c in self.state['candidates'] if c.get('promoted_at', 0) > day]
            if len(promos) >= cfg['max_promotions_per_day']:
                return
            cand = waiting[-1]  # newest data wins; older trained candidates are superseded
            for c in waiting[:-1]:
                c['status'] = 'superseded'
            gate = {'sha256': cand['sha256'], 'stage': 'dev', 'incumbent': current_value()}
            self.state['gate'] = gate
        if not res['training_allowed'] or res['allowed_workers'] < 1:
            return
        cand = next(c for c in self.state['candidates'] if c['sha256'] == gate['sha256'])
        handles = self.slots.acquire(res['slot_pool'] + cfg['slot_limit_extra'], cfg['gate_slots'])
        if len(handles) < cfg['gate_min_slots']:
            for h in handles:
                h.close()
            return
        inc = gate['incumbent']
        out = self.vroot / 'gates' / f"{gate['sha256'][:16]}-{gate['stage']}.jsonl"
        seed = int(gate['sha256'][:8], 16) + (0 if gate['stage'] == 'dev' else 7_777_777)
        argv = [PY, str(ROOT / 'search' / 'value_gate.py'), '--candidate', cand['path'], '--beta', str(cand['beta']),
                '--pairs', str(cfg['gate_pairs']), '--concurrency', str(len(handles)), '--engines', str(cfg['gate_engines']),
                '--alpha', str(cfg['gate_alpha']), '--seed', str(seed), '--out', str(out)]
        if inc.get('path'):
            argv += ['--incumbent', inc['path'], '--incumbent-beta', str(inc['beta'])]
        report = self.vroot / 'logs' / f"gate-{gate['sha256'][:16]}-{gate['stage']}.json"
        gate['report'] = str(report)
        cand['status'] = 'gating:' + gate['stage']
        proc = self.spawn('gate', ['/bin/sh', '-c', 'exec ' + ' '.join(_q(a) for a in argv) + ' > ' + _q(str(report))],
                          handles, self.vroot / 'logs' / 'gate.log', nice=5, qos=cfg['qos']['gate'])
        gate['pid'] = proc.pid
        self.note(f"gate {gate['stage']} {gate['sha256'][:12]} with {len(handles)} slots vs "
                  f"{(inc.get('sha256') or 'hand')[:12]}")

    def finish_gate(self, code: int, cfg: dict):
        gate = self.state.get('gate') or {}
        gate.pop('pid', None)
        cand = next((c for c in self.state['candidates'] if c['sha256'] == gate.get('sha256')), None)
        try:
            result = json.loads(Path(gate['report']).read_text().strip().splitlines()[-1])
        except (OSError, ValueError, KeyError, IndexError):
            self.note(f'gate exited code={code} without a result; will resume')
            return  # results JSONL is resumable; the next tick restarts the same stage
        if not result.get('complete'):
            self.note('gate incomplete; will resume')
            return
        if cand is None:
            self.state.pop('gate', None)
            return
        cand.setdefault('gates', {})[gate['stage']] = result
        if not result['passed']:
            cand['status'] = f"rejected at {gate['stage']} gate"
            self.state.pop('gate', None)
            self.note(f"candidate {cand['sha256'][:12]} rejected at {gate['stage']}: "
                      f"{result['gains']} gains / {result['losses']} losses, p={result['p_one_sided']}")
            return
        if gate['stage'] == 'dev':
            gate['stage'] = 'confirm'
            cand['status'] = 'trained'  # keeps it eligible; the gate record carries the stage
            self.note(f"candidate {cand['sha256'][:12]} passed development; confirming on fresh seeds")
            return
        if current_value().get('sha256') != (gate['incumbent'] or {}).get('sha256'):
            cand['status'] = 'stale: incumbent changed during gating'
            self.state.pop('gate', None)
            return
        record = promote(Path(cand['path']), cand['beta'], {'gates': cand['gates'], 'val': cand['val'],
                                                            'live_test': cand['live_test']})
        cand['status'] = 'promoted'
        cand['promoted_at'] = record['promoted_at']
        self.state.pop('gate', None)
        self.note(f"PROMOTED {cand['sha256'][:12]} beta={cand['beta']}")

    # ------------------------------------------------------------ loop
    def reap(self, cfg: dict):
        for name, proc in list(self.children.items()):
            code = proc.poll()
            if code is None:
                continue
            del self.children[name]
            if name.startswith('prod-'):
                self.finish_producer(name, code)
            elif name == 'train':
                self.finish_train(code)
            elif name == 'gate':
                self.finish_gate(code, cfg)
            elif name == 'ingest':
                self.note(f'ingest done code={code}')

    def shed(self, res: dict):
        """Critical memory pressure: stop our own background load so macOS never has to
kill the live runner. Producers lose at most one self-play game; a gate resumes
from its results file. New work starts again once admission allows it."""
        pressure = res.get('memory_pressure')
        if not pressure or not pressure & 4:
            return
        pids = [p.pid for n, p in self.children.items() if n.startswith('prod-') or n == 'gate']
        pids += [i['pid'] for i in self.state['producers'].values() if _alive(i.get('pid'))]
        gate = self.state.get('gate') or {}
        if gate.get('pid') and _alive(gate['pid']):
            pids.append(gate['pid'])
        for pid in set(pids):
            try:
                os.killpg(pid, signal.SIGCONT)  # a paused group must be able to exit
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                pass
        if pids:
            self.note(f'critical memory pressure: stopped {len(set(pids))} background groups')

    def publish_pgids(self):
        """Process groups the live runner may pause: producers and gates (own session each)."""
        pids = [p.pid for n, p in self.children.items() if n.startswith('prod-') or n == 'gate']
        pids += [i['pid'] for s, i in self.state['producers'].items() if f'prod-{s}' not in self.children and _alive(i.get('pid'))]
        gate = self.state.get('gate') or {}
        if gate.get('pid') and 'gate' not in self.children and _alive(gate['pid']):
            pids.append(gate['pid'])
        try:  # ad-hoc jobs (e.g. measure_noise.py) started in their own session
            pids += [int(x) for x in json.loads((self.vroot / 'extra-pgids.json').read_text()) if _alive(int(x))]
        except (OSError, ValueError, TypeError):
            pass
        atomic_json(BACKGROUND_PGIDS, sorted(set(pids)))

    def adopt_orphans(self):
        """After a restart: producers still running keep their shard; record finished ones."""
        for seq, info in list(self.state['producers'].items()):
            if not _alive(info.get('pid')):
                self.finish_producer('prod-' + seq, -1)
        self.state.pop('pending_train', None)
        self.scan_shards()

    def scan_shards(self):
        """Count and cache shards written outside this supervisor (or before a restart)."""
        active = {i.get('shard') for i in self.state['producers'].values()}
        known = self.state.setdefault('shard_rows', {})
        for shard in sorted((self.vroot / 'selfplay').glob('*.jsonl')):
            if shard.name in known or shard.name in active:
                continue
            if time.time() - shard.stat().st_mtime < 120:
                continue  # possibly still being written by a manual producer
            pack_shard(shard)
            known[shard.name] = sum(1 for _ in shard.open())

    def status(self, res: dict, cfg: dict) -> dict:
        return {'pid': os.getpid(), 'updated_at': time.time(), 'enabled': cfg['enabled'],
                'children': {n: p.pid for n, p in self.children.items()},
                'orphan_producers': [s for s, i in self.state['producers'].items() if f'prod-{s}' not in self.children],
                'selfplay_rows': self.selfplay_rows(), 'selfplay_gb': round(dir_bytes(self.vroot / 'selfplay') / 2**30, 3),
                'live_games': self.live_games(), 'trained_rows': self.state['trained_rows'],
                'gate': self.state.get('gate'), 'current': current_value(),
                'candidates': [{k: c.get(k) for k in ('sha256', 'status', 'beta', 'trained_at')} for c in self.state['candidates'][-8:]],
                'resources': {k: res.get(k) for k in ('available_gb', 'system_cpu_percent', 'allowed_workers',
                                                      'memory_pressure', 'training_allowed', 'slot_pool')},
                'events': self.events[-12:]}

    def run(self, once: bool = False):
        lock = (self.vroot / 'service.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'skipped': 'value service already running'}))
            return
        stop = {'flag': False}
        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
        signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
        self.adopt_orphans()
        self.note('value service started')
        while not stop['flag']:
            cfg = load_config()
            res = self.resources()
            self.reap(cfg)
            # Orphaned producers from a previous supervisor: finish when they exit.
            for seq, info in list(self.state['producers'].items()):
                if f'prod-{seq}' not in self.children and not _alive(info.get('pid')):
                    self.finish_producer('prod-' + seq, -1)
            self.shed(res)
            self.publish_pgids()
            resume_if_stale()
            self.ingest()
            if int(time.time()) % 300 < cfg['interval_seconds']:
                self.scan_shards()
            if cfg['enabled']:
                self.maybe_gate(cfg, res)
                self.maybe_train(cfg, res)
                self.produce(cfg, res)
            self.save()
            atomic_json(self.vroot / 'service-status.json', self.status(res, cfg))
            if once:
                break
            for _ in range(int(cfg['interval_seconds'])):
                if stop['flag']:
                    break
                time.sleep(1)
        # Children are bounded (shards, gates resume); leave them to finish.
        self.save()
        self.note('value service stopping; children keep running and are adopted on restart')


def _q(text: str) -> str:
    import shlex
    return shlex.quote(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    if args.status:
        try:
            print(json.dumps(json.loads((VALUE_ROOT / 'service-status.json').read_text()), indent=2))
        except OSError:
            print(json.dumps({'running': False}))
        return
    Service().run(once=args.once)


if __name__ == '__main__':
    main()
