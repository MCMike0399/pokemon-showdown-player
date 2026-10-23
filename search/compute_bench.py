"""Fixed-world 4/6/8 engine benchmark, with live priority and recoverable c10 pause.

prepare freezes public positions; run temporarily parks c10 (not the live player),
measures each width twice, and restores registrations even if the parent dies.
No experiment settings, model pointers or live options are changed.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search.value_store import PAUSE_MARKER  # noqa: E402

VROOT = ROOT / 'data/ml/value'


def atomic_json(path, value):
    # The ingest service can publish the shared registry simultaneously.
    # Its .tmp pathname must not collide with this diagnostic's temporary file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def command(pid):
    return subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()


def registry(add=(), remove=()):
    # c10's publisher takes this same lock; the value service reads extra-pgids.
    with (VROOT / 'extra-pgids.lock').open('a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        for name in ('extra-pgids.json', 'background-pgids.json'):
            path = VROOT / name
            pids = (set(read(path, [])) - set(remove)) | set(add)
            atomic_json(path, sorted(p for p in pids if 'search/' in command(p)))


def restore(out):
    lease = read(out / 'lease.json', {})
    if lease.get('restored'):
        return
    for p in lease.get('benchmark_pids', []):
        if 'search/compute_bench.py worker' in command(p):
            os.killpg(p, signal.SIGCONT)
            os.killpg(p, signal.SIGTERM)
    groups = [p for p, cmd in lease.get('groups', []) if command(p) == cmd]
    registry(add=groups, remove=lease.get('benchmark_pids', []))
    if not PAUSE_MARKER.exists():
        for p in groups:
            os.killpg(p, signal.SIGCONT)
    supervisor = lease.get('supervisor')
    if supervisor and command(supervisor[0]) == supervisor[1]:
        os.kill(supervisor[0], signal.SIGCONT)
    atomic_json(out / 'lease.json', {**lease, 'restored': time.time()})


def prepare(out, source, count):
    from search.agent import SearchAgent
    from search.elo_monitor import _as_list
    from search.ledger_positions import _prefix
    from search.tracker import Tracker
    from search.config import agent_kwargs
    if (out / 'workload.json').exists():
        raise ValueError('workload already frozen; use another directory')
    decisions = read(source / 'decisions.json')
    terminals = read(source / 'terminals.json')
    sets = read(ROOT / 'teams.json')['Rain-Recife-special-stat-fix']['sets']
    candidates = [d for d in decisions if d['room'] in terminals and
                  (d.get('decision', {}).get('search') or {}).get('kind') == 'move']
    random.Random(20261023).shuffle(candidates)
    workload, used = [], set()
    for d in candidates:
        if d['room'] in used:
            continue
        agent = SearchAgent(**agent_kwargs(worlds=6, engines=1, adaptive_worlds=0, screen_log=0), seed=20261023)
        tracker = Tracker(d['request']['side']['id']).feed(_prefix(_as_list(terminals[d['room']]['log']), d['state']['turn']))
        ctx = {'request': d['request'], 'team_sets': sets}
        ours = agent._our_side(ctx, tracker)
        worlds = [{'p1': ours, 'p2': agent._their_side(tracker, agent.rng)} for _ in range(6)]
        if any(m['state'].get('status') in ('slp', 'frz') or m['state'].get('volatiles', {}).get('confusion')
               for w in worlds for s in ('p1', 'p2') for m in w[s]):
            continue  # rebuilt sleep/confusion durations currently use Math.random
        workload.append({'type': 'search', 'format': agent.fmt, 'field': agent._field(tracker),
                         'worlds': worlds, 'crn': True, 'screen': agent.screen, 'alpha': agent.alpha,
                         'leaf': agent.leaf, 'protect_d2': agent.protect_d2, 'world_offset': len(workload)*17})
        used.add(d['room'])
        if len(workload) == count:
            break
    if len(workload) != count:
        raise ValueError('insufficient deterministic positions')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'workload.json', workload)
    atomic_json(out / 'protocol.json', {
        'created': time.time(), 'seed': 20261023, 'positions': count,
        'workload_sha256': hashlib.sha256((out / 'workload.json').read_bytes()).hexdigest(),
        'widths': [4, 8, 6, 6, 8, 4], 'baseline_seconds': 120,
        'selection': 'smallest width within 5% of best throughput, subject to live latency review; no automatic change',
        'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'note': 'Position-search throughput, not full-game pairs/hour. Parked gates retain resident memory.'})


async def worker(out, width, phase):
    from search.agent import Engine
    workload = read(out / 'workload.json')
    engines = [Engine() for _ in range(width)]
    started = time.time()
    rows = []

    async def lane(i):
        for index in range(i, len(workload), width):
            t = time.time()
            result = await engines[i].call(workload[index], timeout=300)
            worlds = result['results']
            if any('error' in w for w in worlds):
                raise RuntimeError(str(worlds)[:500])
            rows.append({'index': index, 'seconds': time.time()-t,
                         'sims': sum(w['sims'] for w in worlds),
                         'values_sha256': hashlib.sha256(json.dumps([w['values'] for w in worlds],
                                                                    sort_keys=True).encode()).hexdigest()})
    try:
        await asyncio.gather(*(lane(i) for i in range(width)))
    finally:
        await asyncio.gather(*(e.close() for e in engines))
    atomic_json(out / f'phase-{phase}.json', {'width': width, 'started': started, 'ended': time.time(), 'rows': rows})


def run(out):
    protocol = read(out / 'protocol.json')
    if hashlib.sha256((out / 'workload.json').read_bytes()).hexdigest() != protocol['workload_sha256']:
        raise ValueError('workload hash changed')
    lock = (VROOT / 'compute-benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'lease.json').exists():
        raise ValueError('this benchmark has already run; retain evidence')
    state = read(VROOT / 'campaign-c10/state.json')
    supervisor = state['pid']
    groups = [state['jobs'][k]['pid'] for k in ('preview', 'endgame')
              if 'search/campaign.py worker' in command(state['jobs'][k]['pid'])]
    lease = {'owner': os.getpid(), 'owner_command': command(os.getpid()),
             'supervisor': [supervisor, command(supervisor)],
             'groups': [[p, command(p)] for p in groups], 'benchmark_pids': [], 'created': time.time()}
    if 'search/campaign.py run' not in lease['supervisor'][1]:
        raise ValueError('campaign supervisor not running; inspect before benchmarking')
    atomic_json(out / 'lease.json', lease)
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'guard', '--out', str(out)],
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        os.kill(supervisor, signal.SIGSTOP)
        registry(remove=groups)
        for p in groups:
            os.killpg(p, signal.SIGSTOP)
        # The ingest service stays running. Verify its next publication still
        # excludes parked gates before measurements begin.
        time.sleep(5)
        registry(remove=groups)
        for p in groups:
            os.killpg(p, signal.SIGSTOP)
        baseline = []
        for phase, width in enumerate([0] + protocol['widths'] + [0]):
            start = time.time()
            if width == 0:
                time.sleep(protocol['baseline_seconds'])
                baseline.append({'started': start, 'ended': time.time()})
                atomic_json(out / 'baselines.json', baseline)
            else:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                                          '--out', str(out), '--width', str(width), '--phase', str(phase)],
                                         start_new_session=True)
                lease['benchmark_pids'].append(child.pid)
                atomic_json(out / 'lease.json', lease)
                registry(add=[child.pid])
                code = child.wait(timeout=900)
                registry(remove=[child.pid])
                if code:
                    raise RuntimeError(f'phase {phase} failed: {code}')
            print(json.dumps({'phase': phase, 'width': width, 'seconds': round(time.time()-start, 1)}), flush=True)
    finally:
        restore(out)
        lock.close()


def summarize(out):
    phases = [read(p) for p in sorted(out.glob('phase-*.json'))]
    if len(phases) != 6 or len(read(out/'baselines.json', [])) != 2:
        raise ValueError('benchmark has not completed all preregistered phases')
    signatures = [{r['index']: (r['sims'], r['values_sha256']) for r in p['rows']} for p in phases]
    if any(s != signatures[0] for s in signatures):
        raise ValueError('worker widths did not evaluate identical workloads/results')
    intervals = [(p['started'], p['ended'], p['width']) for p in phases]
    intervals += [(p['started'], p['ended'], 0) for p in read(out/'baselines.json')]
    samples = {n: [] for n in (0, 4, 6, 8)}
    pause_overlap = {n: 0. for n in (4, 6, 8)}
    for path in (ROOT/'data/ml/browser-runs').glob('*/events.jsonl'):
        if path.stat().st_mtime < min(s for s,e,w in intervals):
            continue
        for line in path.open():
            if '"decision"' not in line:
                continue
            r = json.loads(line)
            d = r.get('decision') or {}; search = d.get('search') or {}
            if r.get('kind') != 'decision' or d.get('inference_ms') is None:
                continue
            t = r['time']; seconds = d['inference_ms']/1000
            if d.get('decider') == 'search_decider':
                for start, end, width in intervals:
                    if width:
                        pause_overlap[width] += max(0., min(t, end)-max(t-seconds, start))
            if search.get('kind') != 'move':
                continue
            for start, end, width in intervals:
                if start <= t-seconds and t <= end:
                    samples[width].append({'seconds': seconds, 'extra': bool(search.get('extra_worlds')),
                                           'turn': search.get('turn'), 'time': t})
    result = {'identical_results': True, 'widths': {}, 'live': {},
              'note': 'Fixed-position throughput; short observational live windows, not Elo evidence.'}
    for width in (4, 6, 8):
        group = [p for p in phases if p['width'] == width]
        seconds = sum(p['ended']-p['started'] for p in group)
        count = sum(len(p['rows']) for p in group)
        result['widths'][width] = {'seconds': seconds, 'positions_per_second': count/seconds,
            'estimated_live_pause_seconds': pause_overlap[width],
            'pause_adjusted_positions_per_second': count/(seconds-pause_overlap[width])}
    for width, rows in samples.items():
        seconds = sorted(r['seconds'] for r in rows)
        result['live'][width] = {'n': len(rows), 'median': seconds[len(seconds)//2] if rows else None,
                                 'p95': seconds[round(.95*(len(seconds)-1))] if rows else None,
                                 'extra_fraction': sum(r['extra'] for r in rows)/len(rows) if rows else None}
    atomic_json(out/'live-samples.json', samples)
    atomic_json(out/'summary.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['prepare', 'run', 'worker', 'guard', 'restore', 'summarize'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--positions', type=int, default=48)
    ap.add_argument('--width', type=int)
    ap.add_argument('--phase', type=int)
    args = ap.parse_args()
    out = args.out.resolve()
    if args.cmd == 'prepare':
        prepare(out, args.source, args.positions)
    elif args.cmd == 'worker':
        asyncio.run(worker(out, args.width, args.phase))
    elif args.cmd == 'run':
        run(out)
    elif args.cmd == 'restore':
        restore(out)
    elif args.cmd == 'summarize':
        print(json.dumps(summarize(out), indent=1))
    else:
        while True:
            lease = read(out / 'lease.json')
            if lease.get('restored'):
                break
            if command(lease['owner']) != lease['owner_command']:
                restore(out)
                break
            time.sleep(2)


if __name__ == '__main__':
    main()
