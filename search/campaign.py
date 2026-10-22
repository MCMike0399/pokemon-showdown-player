"""Persistent c10 development campaign, isolated from the active ladder experiment.

MPS training -> two fixed pooled offline gates -> wait for c9 -> one fresh live
A/B of the union of offline winners. Only that live test may promote. A launchd
supervisor resumes workers by their result ledgers; it never owns a player login.
"""
from __future__ import annotations

import argparse
import asyncio
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
from search import ab_test as ab, elo_monitor as em
from search.preview_guard import DEFAULT_MODEL
from search.value_store import atomic_json

TEAM = 'Rain-Recife-special-stat-fix'
DEFAULT_DIR = ROOT / 'data/ml/value/campaign-c10'


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_hash(spec):
    fixed = {k: v for k, v in spec.items() if k not in ('active', 'slots', 'poll_seconds', 'protocol_hash')}
    return hashlib.sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()


def verify_protocol(out, spec):
    if spec.get('protocol_hash') != protocol_hash(spec):
        raise ValueError('fixed campaign settings changed; start a new protocol instead')
    if not 2 <= int(spec['slots']) <= 8 or int(spec['slots']) % 2:
        raise ValueError('slots must be an even number from 2 to 8')
    if digest(out / 'PROTOCOL.md') != spec['protocol_document_sha256']:
        raise ValueError('campaign protocol document changed')
    if digest(out / 'inputs.json') != spec['inputs_sha256']:
        raise ValueError('training manifest changed')
    for path, sha in spec['assets'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('campaign input asset changed: ' + path)


def initialize(out, main):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'spec.json').exists():
        raise ValueError('campaign already exists; resume it instead of resetting evidence')
    active = ab.load_spec(main / 'data/ml/value/ab/current.json')
    spec = {'id': 'c10', 'active': True, 'created': time.time(), 'main': str(main.resolve()),
            'after': active['id'] if active else None, 'team': TEAM, 'slots': 8, 'pairs': 800,
            'alpha': .05, 'seed': 20261101, 'training_games': 2000, 'poll_seconds': 20,
            'preview': 'pvalue:' + DEFAULT_MODEL,
            'opponents': ['imitation:data/ml/human/models-v4/human-449e6bd0e8869deb.json:argmax:'
                          'data/ml/human/models-preview/preview-1455f4a55d18112a.json', 'search'],
            'live': {'elo0': 0., 'elo1': 40., 'alpha': .05, 'beta': .1, 'min_games': 300, 'max_games': 3000},
            'candidates': {'preview': {'preview_guard': {'path': DEFAULT_MODEL, 'max_drop': .025, 'penalty': .04}},
                           'endgame': {'endgame_d2': {'alive': 2, 'top_ours': 4, 'top_opp': 4, 'keep': 8, 'probes': 2}}}}
    games = em.load_games(main / 'data/ml/browser-runs')
    spec['training_policy_key'] = games[-1]['policy_key'].split('+ab-')[0]
    # Freeze the data selection before a single training/evaluation result exists.
    arms = {a['room']: a for a in ab.read_arms(active, main / 'data/ml/value/ab/current.json')} if active else {}
    files = []
    for g in games:
        if g['policy_key'].split('+ab-')[0] != spec['training_policy_key']:
            continue
        if '+ab-' in g['policy_key'] and (g['room'] not in arms or any(arms[g['room']]['cell'].values())):
            continue
        path = main / 'data/ml/value/teams/rain-recife-special-stat-fix/screen' / ('live-' + g['room'].removeprefix('battle-') + '.npz')
        if path.exists():
            files.append(path)
    files = files[-spec['training_games']:]
    if len(files) < 500:
        raise ValueError('need at least 500 completed baseline games with screening labels')
    inputs = out / 'inputs'
    inputs.mkdir()
    manifest = []
    for path in files:
        (inputs / path.name).symlink_to(path.resolve())
        manifest.append({'path': str(path.resolve()), 'sha256': digest(path)})
    atomic_json(out / 'inputs.json', manifest)
    spec['inputs_sha256'] = digest(out / 'inputs.json')
    protocol = (main / 'docs/c10-m5-campaign.md').read_text()
    (out / 'PROTOCOL.md').write_text(protocol)
    spec['protocol_document_sha256'] = digest(out / 'PROTOCOL.md')
    paths = [DEFAULT_MODEL, 'data/ml/human/models-v4/human-449e6bd0e8869deb.json',
             'data/ml/human/models-preview/preview-1455f4a55d18112a.json',
             'data/ml/value/pool-build/pool-teams-new.json', 'teams.json']
    paths += [p.relative_to(main).as_posix() for p in (main / 'cache/usage').glob('*regmc*') if p.is_file()]
    spec['assets'] = {p: digest(main / p) for p in paths}
    spec['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=main, text=True).strip()
    spec['protocol_hash'] = protocol_hash(spec)
    atomic_json(out / 'spec.json', spec)
    atomic_json(out / 'state.json', {'phase': 'training', 'jobs': {}, 'created': time.time()})
    return spec


def verify_inputs(out):
    for row in read(out / 'inputs.json', []):
        if digest(row['path']) != row['sha256']:
            raise ValueError('training input changed: ' + row['path'])


def options_for(name, spec, train):
    options = dict(spec['candidates'][name])
    if name == 'endgame':
        if not train or not train.get('useful') or train.get('device') != 'mps':
            return None
        from search.screen_data import weights_sha
        path = Path(train['path'])
        if weights_sha(path) != train['sha256']:
            raise ValueError('trained model checksum changed')
        # Relative paths work in pinned live, gate and promotion checkouts.
        relative = path.resolve().relative_to(Path(spec['main']).resolve()).as_posix()
        pre = train['prefilter']
        options['screen'] = {'probe': 4, 'keep_ours': 24, 'keep_opp': 20,
                             'learned': {'path': relative, 'pre_ours': pre['ours'], 'pre_opp': pre['opp']}}
    return options


def worker(out, name):
    spec = read(out / 'spec.json')
    verify_protocol(out, spec)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if code != spec['source_commit']:
        raise ValueError('worker source differs from the pinned campaign commit')
    if name == 'train':
        verify_inputs(out)
        lock = (ROOT / 'data/ml/worker.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX)
        from search.train_screen import main
        result = main(['--data', str(out / 'inputs/*.npz'), '--out', str(out / 'models'),
                       '--max-rows', '0', '--device', 'mps', '--mps-memory-fraction', '.5',
                       '--threads', '2', '--strict-holdout', '--seed', str(spec['seed'])])
    elif name == 'preflight':
        from scripts.auto_promote import preflight
        options = {}
        for candidate in spec['candidates']:
            options.update(options_for(candidate, spec, read(out / 'train-result.json')) or {})
        directory = out / 'preflight'
        directory.mkdir(exist_ok=True)
        result = preflight(directory, options, 'campaign c10')
    else:
        from search.team_data import load_teams
        from search.value_gate import run_gate
        opts = options_for(name, spec, read(out / 'train-result.json'))
        if opts is None:
            result = {'complete': True, 'passed': False, 'reason': 'screening prescreen failed'}
        else:
            pool = read(ROOT / 'data/ml/value/pool-build/pool-teams-new.json')
            pool = [t['sets'] if isinstance(t, dict) else t for t in pool]
            candidate = {'search': opts, 'preview': spec['preview']}
            incumbent = {'preview': spec['preview']}
            result = asyncio.run(run_gate(candidate, incumbent, out / f'{name}.jsonl', spec['pairs'],
                                          spec['slots'] // 2, spec['seed'] + (name == 'endgame'),
                                          load_teams()[spec['team']]['sets'], pool, 1, spec['alpha'],
                                          rule='pooled', opponent=spec['opponents']))
            result['options'] = opts
    atomic_json(out / f'{name}-result.json', result)


def command(pid):
    if not pid:
        return ''
    try:
        return subprocess.run(['ps', '-ww', '-o', 'command=', '-p', str(pid)], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except subprocess.SubprocessError:
        return ''


def job_alive(job, out):
    cmd = command(job.get('pid'))
    return bool(cmd and str(out) in cmd and ('search/campaign.py worker' in cmd or 'scripts/ab_watch.py' in cmd))


def publish_workers(out, state):
    """Join the existing pause registry without replacing another campaign's jobs."""
    path = ROOT / 'data/ml/value/extra-pgids.json'
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read(path, [])
        alive = [p for p in previous if 'search/' in command(p)]
        for name, job in state['jobs'].items():
            if name in ('preview', 'endgame') and job_alive(job, out):
                alive.append(job['pid'])
        atomic_json(path, sorted(set(alive)))


def start_job(name, out, spec, state):
    job = state['jobs'].setdefault(name, {'attempts': 0})
    if job_alive(job, out):
        return
    if job['attempts'] >= 3:
        raise RuntimeError(f'{name} failed three starts; see {out / (name + ".log")}')
    if name != 'watch':
        from ml.data_budget import storage_status
        if not storage_status(Path(spec['main']) / 'data/ml', 150_000_000_000 / 2**30)['background_allowed']:
            raise RuntimeError('150 GB data budget: preserve the live reserve before starting more background work')
    py = str(Path(spec['main']) / '.venv/bin/python')
    if name == 'watch':
        args = [py, '-u', str(Path(spec['main']) / 'scripts/ab_watch.py'), '--dir', str(out / 'live'),
                '--label', 'campaign c10', '--poll', '300']
        cwd = spec['main']
    else:
        args = [py, '-u', str(ROOT / 'search/campaign.py'), 'worker', '--dir', str(out), '--job', name]
        cwd = ROOT
    with (out / f'{name}.log').open('a') as log:
        proc = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job.update(pid=proc.pid, attempts=job['attempts'] + 1, started=time.time())
    atomic_json(out / 'state.json', state)


def handoff(out, spec, state, live_spec):
    """One fresh live test after the previous experiment; never replace an active spec."""
    current = read(live_spec, {})
    if current.get('id') == spec['id']:
        if state.get('live_options') and current.get('factors') != {'upgrade': state['live_options']}:
            raise ValueError('live experiment options no longer match the campaign')
        return 'live' if current.get('active') else 'done'
    if current.get('active') or current.get('id') not in (spec.get('after'), None):
        return 'waiting_for_previous_campaign'
    options = {}
    for name in spec['candidates']:
        result = read(out / f'{name}-result.json', {})
        if result.get('passed'):
            options.update(options_for(name, spec, read(out / 'train-result.json')) or {})
    if not options:
        return 'done_no_winner'
    # Capable code must already be deployed before any arm can be assigned.
    from scripts import live_deploy
    status = live_deploy.status()
    if status['dirty'] or status['missing_links']:
        raise RuntimeError('live worktree is not ready')
    source = Path(status['live']) / 'search/agent.py'
    if 'preview_guard' not in source.read_text():
        return 'waiting_for_code_deploy'
    if subprocess.run(['git', 'merge-base', '--is-ancestor', spec['source_commit'], status['head']],
                      cwd=spec['main'], capture_output=True).returncode:
        return 'waiting_for_code_deploy'
    (out / 'live').mkdir(exist_ok=True)
    atomic_json(out / 'previous-live-spec.json', current)
    # The union is fixed before its first game; there is one hypothesis test,
    # rather than selecting the best of several live tests after their outcomes.
    next_spec = ab.validate({'id': spec['id'], 'active': True, 'seed': spec['seed'] + 100,
                             'created': time.time(), 'protocol': str(out / 'PROTOCOL.md'),
                             'factors': {'upgrade': options}, **spec['live']})
    ab.write_spec(next_spec, live_spec)
    state['live_options'] = options
    return 'live'


def tick(out, spec, state):
    verify_protocol(out, spec)
    if not spec.get('active'):
        state['phase'] = 'paused'
        return
    if not (out / 'train-result.json').exists():
        state['phase'] = 'training'
        start_job('train', out, spec, state)
        return
    preflight = read(out / 'preflight-result.json')
    if preflight is None:
        state['phase'] = 'preflight'
        start_job('preflight', out, spec, state)
        return
    if not preflight.get('ok'):
        raise RuntimeError('promotion preflight failed; inspect preflight/promote.log before spending gate compute')
    pending = []
    for name in spec['candidates']:
        if not (out / f'{name}-result.json').exists():
            if options_for(name, spec, read(out / 'train-result.json')) is None:
                atomic_json(out / f'{name}-result.json', {'complete': True, 'passed': False, 'reason': 'prescreen failed'})
            else:
                start_job(name, out, spec, state)
                pending.append(name)
    if pending:
        state['phase'] = 'offline_gates'
        return
    if not any(read(out / f'{name}-result.json', {}).get('passed') for name in spec['candidates']):
        state['phase'] = 'done_no_winner'
        return
    state['phase'] = handoff(out, spec, state, Path(spec['main']) / 'data/ml/value/ab/current.json')
    if state['phase'] == 'live':
        start_job('watch', out, spec, state)


def supervise(out):
    lock = (out / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = read(out / 'state.json', {'jobs': {}})
    while not stopping:
        spec = read(out / 'spec.json')
        try:
            tick(out, spec, state)
            state.pop('error', None)
        except Exception as e:
            state.update(phase='error', error=repr(e))
        publish_workers(out, state)
        state.update(pid=os.getpid(), updated=time.time())
        atomic_json(out / 'state.json', state)
        for _ in range(spec['poll_seconds']):
            if stopping:
                break
            time.sleep(1)
    # Children keep their process groups and ledgers; a relaunched supervisor
    # adopts them, avoiding duplicate simulations after a restart.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['init', 'run', 'worker', 'status'])
    ap.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    ap.add_argument('--main', type=Path, default=ROOT)
    ap.add_argument('--job', choices=['train', 'preflight', 'preview', 'endgame'])
    args = ap.parse_args()
    out = args.dir.resolve()
    if args.command == 'init':
        print(json.dumps(initialize(out, args.main), indent=1))
    elif args.command == 'worker':
        worker(out, args.job)
    elif args.command == 'run':
        supervise(out)
    else:
        state = read(out / 'state.json', {})
        spec = read(out / 'spec.json', {})
        for name in ('train', 'preview', 'endgame'):
            result = read(out / f'{name}-result.json')
            if result:
                state[name] = {k: result.get(k) for k in ('complete', 'passed', 'useful', 'device', 'path', 'reason',
                                                           'pairs_complete', 'gains', 'losses', 'prefilter')}
            elif (out / f'{name}.jsonl').exists():
                from search.value_gate import summarize
                rows = []
                for line in (out / f'{name}.jsonl').read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
                state[name] = summarize(rows, spec['pairs'], spec['alpha'], rule='pooled')
        print(json.dumps(state, indent=1))


if __name__ == '__main__':
    main()
