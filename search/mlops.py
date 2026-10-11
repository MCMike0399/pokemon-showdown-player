"""Batch training campaigns on Macintosh, live play and deployment on DiveMac.

    .venv/bin/python -m search.mlops status                  # live record, campaign readiness, deploys
    .venv/bin/python -m search.mlops bootstrap               # one time: set up the trainer on Macintosh
    .venv/bin/python -m search.mlops campaign start [--selfplay-games 5000] [--force]
    .venv/bin/python -m search.mlops campaign status
    .venv/bin/python -m search.mlops campaign finish [--no-deploy] [--force]
    .venv/bin/python -m search.mlops deploy [--dry-run]      # inbox bundles -> live pointers
    .venv/bin/python -m search.mlops rollback --kind value|screen [--to none]
    .venv/bin/python -m search.mlops role live|full          # DiveMac's own value service

Nothing moves between the Macs on its own. DiveMac plays and collects; its value
service sends one notification when `campaign_every_games` (1000) new live games
arrived since the last campaign. You then start a campaign:

1. `campaign start` copies the code and a snapshot of DiveMac's data (live
   positions, screening logs, self-play, candidates, both pointers) to the trainer
   checkout on Macintosh (separate from the dev copy) and starts its value service
   in role `trainer` with that campaign: a big self-play simulation, one training
   run for the value net and one for the screening ranker on all data, the
   two-stage paired gate for each, then idle. A model that passes both stages is
   exported as an immutable bundle (weights + manifest with gate results, the
   incumbents it beat and the play-code fingerprint that was measured).
2. `campaign finish` (when status says done) pulls the bundles, the new self-play
   shards and gate logs back (DiveMac keeps every byte), stops the trainer and
   deploys each bundle that still holds: weights hash, both gates passed, same
   team, same incumbents as now, same play code as now, daily promotion limit.
   The live runner adopts the pointer at its next matchmaking boundary.

Keep the play code unchanged during a campaign: a model gated under other code is
stale and isn't deployed. `rollback` points live back at the previous promotion
(or the hand evaluation / probe screening with `--to none`). See docs/mlops.md.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search.value_store import atomic_json  # noqa: E402

VALUE_ROOT = ROOT / 'data' / 'ml' / 'value'
CONFIG = VALUE_ROOT / 'mlops.json'
DEFAULTS = {
    'host': 'Macintosh',  # ssh alias; 'local' = remote_root on this machine (tests, dry runs)
    'remote_root': str(Path.home() / 'Developer' / 'pokemon-showdown-trainer'),
    'remote_rsync': '/opt/homebrew/bin/rsync',  # macOS /usr/bin/rsync is openrsync
    'campaign_selfplay_games': 5000,  # per campaign; ~20 games per shard, one shard per producer
    'campaign_train_max_rows': 4000000,
    'connect_timeout': 12,
    'settle_seconds': 180,  # a data file untouched this long is complete
}
SHARD_PREFIX = 'sp-trainer'
LOCAL_RSYNC = '/opt/homebrew/bin/rsync' if os.path.exists('/opt/homebrew/bin/rsync') else 'rsync'
# Everything that decides how a position is searched or valued. A model gated
# under other play code proves nothing about this one (search/config.py).
PLAY_FILES = ('search/*.cjs', 'search/config.py', 'search/agent.py', 'search/priors.py', 'search/tracker.py',
              'battle_state.py', 'node_modules/pokemon-showdown/package.json')
INPUT_DIRS = ('cache/usage', 'cache/dex', 'artifacts/search-breakthrough/frozen', 'node_modules')
SKIP_SOURCE = ('notebooks/', 'results/')
SOURCE_LIST = '.mlops-source-files'
TRAINER_LABEL = 'dev.pokemon-showdown.value-trainer'
CAMPAIGNS = VALUE_ROOT / 'campaigns.jsonl'


def load_config(root: Path = VALUE_ROOT) -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads((root / 'mlops.json').read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def play_fingerprint(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    files = sorted({p for pattern in PLAY_FILES for p in Path(root).glob(pattern) if p.is_file()})
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b'\0' + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _slug(team: str) -> str:
    from search.team_data import slug
    return slug(team)


def _weights_sha(kind: str, path: Path) -> str:
    if kind == 'screen':
        from search.screen_data import weights_sha
    else:
        from search.value_store import weights_sha
    return weights_sha(path)


def _pointer_sha(kind: str, pointer: Path) -> str | None:
    if kind == 'screen':
        from search.screen_data import current_screen
        return current_screen(pointer).get('sha256')
    from search.value_store import current_value
    return current_value(pointer).get('sha256')


# ------------------------------------------------------------ trainer side
def export_bundle(kind: str, cand: dict, gate: dict, team: str, value_root: Path = VALUE_ROOT,
                  root: Path = ROOT, box: str = 'outbox') -> Path:
    """A candidate that passed both gate stages -> immutable bundle. The trainer
writes to its outbox (`campaign finish` brings it to DiveMac); a live host writes straight
to its own inbox, so every promotion goes through deploy_inbox's checks."""
    src = Path(cand['path'])
    sha = _weights_sha(kind, src)
    if sha != cand['sha256']:
        raise ValueError(f'candidate {src} hash changed since training')
    outbox = value_root / box
    final = outbox / f'{kind}-{sha[:16]}'
    if final.exists():
        return final
    outbox.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.tmp-', dir=outbox))
    shutil.copy2(src, tmp / src.name)
    manifest = {
        'kind': kind, 'team': team, 'sha256': sha, 'file': src.name,
        'beta': cand.get('beta'), 'prefilter': cand.get('prefilter'),
        'estimated_sim_saving': cand.get('estimated_sim_saving'),
        'val': cand.get('val'), 'live_test': cand.get('live_test'), 'rows': cand.get('rows'),
        'records': cand.get('records'), 'trained_at': cand.get('trained_at'),
        'gates': cand.get('gates') or {},
        'incumbent': (gate.get('incumbent') or {}).get('sha256'),
        'screen_incumbent': (gate.get('screen_incumbent') or {}).get('sha256'),
        'code': gate.get('code') or play_fingerprint(root),
        'host': os.uname().nodename, 'exported_at': time.time(),
    }
    atomic_json(tmp / 'manifest.json', manifest)
    os.replace(tmp, final)
    return final


def export_list(value_root: Path = VALUE_ROOT, settle: float = DEFAULTS['settle_seconds']) -> list[str]:
    """Trainer data DiveMac should keep: finished own shards and gate screening logs.
Paths relative to the repo root; nothing still being written."""
    out = []
    now = time.time()
    try:
        state = json.loads((value_root / 'service-state.json').read_text())
    except (OSError, ValueError):
        state = {}
    active = {Path(i.get('shard') or '').name for i in state.get('producers', {}).values()}
    active |= {n.replace('.jsonl', '.npz') for n in active}
    for team in (value_root / 'teams').glob('*/'):
        paths = list((team / 'selfplay').glob(SHARD_PREFIX + '-*')) + list((team / 'screen').glob('gate-*.npz'))
        for p in paths:
            if p.name in active or '.tmp' in p.name or not p.is_file():
                continue
            if now - p.stat().st_mtime < settle:
                continue
            out.append(p.relative_to(value_root.parents[2]).as_posix())
    return sorted(out)


# ------------------------------------------------------------ DiveMac side: deploy
def verify_bundle(bundle: Path, team: str, teams_root: Path, code: str, max_per_day: int,
                  now: float | None = None) -> tuple[bool, str, dict]:
    """(deployable, reason, manifest). Stale = would have been fine against older state."""
    try:
        manifest = json.loads((bundle / 'manifest.json').read_text())
    except (OSError, ValueError):
        return False, 'wait: unreadable manifest (incomplete transfer?)', {}
    kind = manifest.get('kind')
    if kind not in ('value', 'screen'):
        return False, f'rejected: unknown kind {kind!r}', manifest
    if _slug(manifest.get('team') or '') != _slug(team):
        return False, f"rejected: team {manifest.get('team')!r} is not the deployed {team!r}", manifest
    weights = bundle / str(manifest.get('file') or '')
    try:
        if _weights_sha(kind, weights) != manifest.get('sha256'):
            return False, 'wait: weights hash mismatch (incomplete transfer?)', manifest
    except (OSError, ValueError, KeyError):
        # A copy interrupted mid-bundle: the next pull completes it. A truly bad
        # bundle just waits, visibly; it can never pass this check.
        return False, 'wait: weights unreadable (incomplete transfer?)', manifest
    for stage in ('dev', 'confirm'):
        result = (manifest.get('gates') or {}).get(stage) or {}
        if not (result.get('complete') and result.get('passed')):
            return False, f'rejected: {stage} gate not passed', manifest
    if kind == 'value' and not manifest.get('beta'):
        return False, 'rejected: zero blend weight', manifest
    tdir = teams_root / _slug(team)
    if manifest.get('code') != code:
        return False, 'stale: play code changed since the gate', manifest
    if manifest.get('incumbent') != _pointer_sha('value', tdir / 'current.json'):
        return False, 'stale: value incumbent changed since the gate', manifest
    if manifest.get('screen_incumbent') != _pointer_sha('screen', tdir / 'screen-current.json'):
        return False, 'stale: screening incumbent changed since the gate', manifest
    log = tdir / ('screen-promotions.jsonl' if kind == 'screen' else 'promotions.jsonl')
    day = (now or time.time()) - 86400
    recent = 0
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get('promoted_at', 0) > day and not rec.get('rollback'):
                recent += 1
    if recent >= max_per_day:
        return False, 'wait: daily promotion limit reached', manifest
    return True, 'ok', manifest


def deploy_bundle(bundle: Path, manifest: dict, teams_root: Path) -> dict:
    """Copy the weights next to locally trained candidates and move the pointer."""
    from search.screen_data import promote_screen
    from search.value_store import promote
    kind = manifest['kind']
    tdir = teams_root / _slug(manifest['team'])
    dest_dir = tdir / ('screen-candidates' if kind == 'screen' else 'candidates')
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / manifest['file']
    if not dest.exists() or _weights_sha(kind, dest) != manifest['sha256']:
        tmp = dest.with_name(dest.name + '.tmp')
        shutil.copy2(bundle / manifest['file'], tmp)
        os.replace(tmp, dest)
    evidence = {'team': manifest['team'], 'gates': manifest['gates'], 'source': manifest.get('host'),
                'code': manifest.get('code'), 'bundle': bundle.name}
    if kind == 'screen':
        evidence.update(prefilter=manifest.get('prefilter'), estimated_sim_saving=manifest.get('estimated_sim_saving'))
        return promote_screen(dest, evidence, tdir / 'screen-current.json')
    evidence.update(val=manifest.get('val'), live_test=manifest.get('live_test'))
    return promote(dest, manifest['beta'], evidence, tdir / 'current.json')


def deploy_inbox(value_root: Path = VALUE_ROOT, root: Path = ROOT, dry_run: bool = False,
                 only: str | None = None) -> list[dict]:
    """Deploy every new inbox bundle that passes verify_bundle. Serialized by a lock:
`campaign finish` and the value service both call it, and the pointer check and
move must be one step."""
    import fcntl
    value_root.mkdir(parents=True, exist_ok=True)
    with (value_root / 'deploy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _deploy_inbox(value_root, root, dry_run, only)


def _deploy_inbox(value_root: Path, root: Path, dry_run: bool, only: str | None) -> list[dict]:
    from search.value_service import load_config as service_config
    scfg = service_config()
    team = scfg['team']
    teams_root = value_root / 'teams'
    inbox = value_root / 'inbox'
    state_path = value_root / 'mlops-state.json'
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    done = state.setdefault('bundles', {})
    code = play_fingerprint(root)
    results = []
    # Oldest first: a newer bundle gated against the older one's promotion then still matches.
    bundles = sorted((b for b in inbox.glob('*-*') if b.is_dir() and not b.name.startswith('.')),
                     key=lambda b: _exported_at(b))
    for bundle in bundles:
        if only and bundle.name != only:
            continue
        prev = done.get(bundle.name, {})
        if prev.get('status', '').split(':')[0] in ('deployed', 'rejected', 'stale'):
            continue
        ok, reason, manifest = verify_bundle(bundle, team, teams_root, code, scfg['max_promotions_per_day'])
        entry = {'bundle': bundle.name, 'at': time.time(), 'status': reason}
        if ok and not dry_run:
            record = deploy_bundle(bundle, manifest, teams_root)
            entry.update(status='deployed', pointer_sha=record['sha256'], kind=manifest['kind'])
        elif ok:
            entry['status'] = 'would deploy'
        results.append(entry)
        if dry_run or reason.startswith('wait'):
            continue  # retried on the next deploy
        done[bundle.name] = entry
        with (value_root / 'mlops-deploys.jsonl').open('a') as fh:
            fh.write(json.dumps(entry) + '\n')
    if not dry_run:
        atomic_json(state_path, state)
    return results


def _exported_at(bundle: Path) -> float:
    try:
        return float(json.loads((bundle / 'manifest.json').read_text()).get('exported_at') or 0)
    except (OSError, ValueError):
        return 0.0


def rollback(kind: str, to: str | None = None, value_root: Path = VALUE_ROOT) -> dict:
    """Point live back at the promotion before the current one, or at none."""
    import fcntl
    with (value_root / 'deploy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _rollback(kind, to, value_root)


def _rollback(kind: str, to: str | None, value_root: Path) -> dict:
    from search.value_service import load_config as service_config
    tdir = value_root / 'teams' / _slug(service_config()['team'])
    pointer = tdir / ('screen-current.json' if kind == 'screen' else 'current.json')
    log = tdir / ('screen-promotions.jsonl' if kind == 'screen' else 'promotions.jsonl')
    current = _pointer_sha(kind, pointer)
    history = []
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except ValueError:
                pass
    target = None
    if to != 'none':
        shas = [r.get('sha256') for r in history]
        idx = max((i for i, s in enumerate(shas) if s == current), default=None)
        if idx is not None:
            target = next((r for r in reversed(history[:idx]) if r.get('path') and r.get('sha256') != current), None)
    if target is None:
        record = {'path': None, 'sha256': None, 'beta': 0.0} if kind == 'value' else {'path': None, 'sha256': None}
    else:
        record = {k: target[k] for k in ('path', 'sha256', 'beta') if k in target}
    record.update(promoted_at=time.time(), rollback=True, rolled_back=current)
    atomic_json(pointer, record)
    with log.open('a') as fh:
        fh.write(json.dumps(record) + '\n')
    return record


# ------------------------------------------------------------ transport
class Remote:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.host = cfg['host']
        self.root = cfg['remote_root']
        self.local = self.host == 'local'

    def run(self, command: str, timeout: float = 600, check: bool = True) -> subprocess.CompletedProcess:
        if self.local:
            argv = ['/bin/sh', '-c', command]
        else:
            argv = ['ssh', '-o', 'BatchMode=yes', '-o', f"ConnectTimeout={self.cfg['connect_timeout']}", self.host, command]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if check and proc.returncode:
            raise RuntimeError(f'remote command failed ({proc.returncode}): {command}\n{proc.stderr[-2000:]}')
        return proc

    def reachable(self) -> bool:
        try:
            return self.run('true', timeout=self.cfg['connect_timeout'] + 20, check=False).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def path(self, rel: str = '') -> str:
        full = self.root.rstrip('/') + ('/' + rel if rel else '')
        return full if self.local else f'{self.host}:{full}'

    def rsync(self, src: str, dst: str, *flags: str, dry_run: bool = False) -> int:
        # No --partial: rsync writes a temp file and renames it when complete, so an
        # interrupted transfer never leaves a truncated file that --ignore-existing keeps.
        argv = [LOCAL_RSYNC, '-a', *flags]
        if not self.local:
            argv += ['-e', f"ssh -o BatchMode=yes -o ConnectTimeout={self.cfg['connect_timeout']}",
                     f"--rsync-path={self.cfg['remote_rsync']}"]
        if dry_run:
            argv += ['--dry-run', '--itemize-changes']
        argv += [src, dst]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=7200)
        if proc.returncode not in (0, 24):  # 24: a source file vanished mid-transfer (rotated log)
            raise RuntimeError(f"rsync failed ({proc.returncode}): {' '.join(argv[-2:])}\n{proc.stderr[-2000:]}")
        if dry_run and proc.stdout.strip():
            print(proc.stdout.strip()[:4000])
        return proc.returncode

    def python(self, args: str, **kw) -> subprocess.CompletedProcess:
        return self.run(f'cd {shlex.quote(self.root)} && .venv/bin/python {args}', **kw)


def _files_from(paths: list[str]) -> str:
    fh = tempfile.NamedTemporaryFile('w', delete=False, suffix='.list')
    fh.write('\n'.join(paths) + '\n')
    fh.close()
    return fh.name


def source_files(root: Path = ROOT) -> list[str]:
    if not (Path(root) / '.git').exists():  # the trainer checkout: the list push_code shipped
        try:
            return [p for p in (Path(root) / SOURCE_LIST).read_text().splitlines() if p and (Path(root) / p).is_file()]
        except OSError:
            return []
    out = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=root, capture_output=True,
                         text=True, check=True).stdout.split('\n')
    keep = [p for p in out if p and not p.startswith(SKIP_SOURCE) and not p.endswith('.log') and (root / p).is_file()]
    return keep + ['teams.json']


def push_code(remote: Remote, root: Path = ROOT, dry_run: bool = False):
    files = source_files(root)
    listing = _files_from(files)
    try:
        remote.rsync(f'{root}/', remote.path() + '/', f'--files-from={listing}', dry_run=dry_run)
        remote.rsync(listing, remote.path(SOURCE_LIST), dry_run=dry_run)  # last: the trainer's file list
    finally:
        os.unlink(listing)
    for rel in INPUT_DIRS:
        if (root / rel).exists():
            remote.run(f'mkdir -p {shlex.quote(remote.root + "/" + rel)}')
            remote.rsync(f'{root / rel}/', remote.path(rel) + '/', '--delete', dry_run=dry_run)


def push_data(remote: Remote, team: str, value_root: Path = VALUE_ROOT, settle: float = 180, dry_run: bool = False):
    """DiveMac-authoritative files overwrite; append-only data is a union of settled files."""
    rel_team = f'data/ml/value/teams/{_slug(team)}'
    tdir = value_root / 'teams' / _slug(team)
    remote.run(f'mkdir -p {shlex.quote(remote.root + "/" + rel_team)}')
    remote.rsync(str(value_root / 'pool-teams.json'), remote.path('data/ml/value/pool-teams.json'), dry_run=dry_run)
    # rsync follows this order: candidates land before the pointers that name them.
    owned = [p for p in ('team.json', 'games.jsonl', 'live', 'candidates', 'screen-candidates', 'promotions.jsonl',
                         'screen-promotions.jsonl', 'current.json', 'screen-current.json') if (tdir / p).exists()]
    listing = _files_from(owned)
    try:
        remote.rsync(f'{tdir}/', remote.path(rel_team) + '/', f'--files-from={listing}', '-r', dry_run=dry_run)
    finally:
        os.unlink(listing)
    now = time.time()
    union = [p.relative_to(tdir).as_posix() for sub in ('selfplay', 'screen') for p in (tdir / sub).glob('*')
             if p.is_file() and '.tmp' not in p.name and now - p.stat().st_mtime >= settle]
    if not dry_run:  # trainer pointer must not name a missing pointer file it never had
        for name in ('current.json', 'screen-current.json'):
            if not (tdir / name).exists():
                remote.run(f'rm -f {shlex.quote(remote.root + "/" + rel_team + "/" + name)}')
    listing = _files_from(union)
    try:
        remote.rsync(f'{tdir}/', remote.path(rel_team) + '/', f'--files-from={listing}', '--ignore-existing',
                     dry_run=dry_run)
    finally:
        os.unlink(listing)


def pull_results(remote: Remote, value_root: Path = VALUE_ROOT, root: Path = ROOT, dry_run: bool = False):
    proc = remote.python('-m search.mlops export-list', timeout=120)
    files = json.loads(proc.stdout.strip().splitlines()[-1])['files']
    if files:
        listing = _files_from(files)
        try:
            remote.rsync(remote.path() + '/', f'{root}/', f'--files-from={listing}', '--ignore-existing', dry_run=dry_run)
        finally:
            os.unlink(listing)
    inbox = value_root / 'inbox'
    inbox.mkdir(parents=True, exist_ok=True)
    if remote.run(f'test -d {shlex.quote(remote.root + "/data/ml/value/outbox")}', check=False).returncode == 0:
        remote.rsync(remote.path('data/ml/value/outbox') + '/', f'{inbox}/', '--exclude=.tmp-*', '--delay-updates',
                     dry_run=dry_run)
    audit = value_root / 'remote' / _host_dir(remote)
    audit.mkdir(parents=True, exist_ok=True)
    for name in ('service-status.json', 'service-state.json'):  # absent until the trainer's first tick
        remote.rsync(remote.path(f'data/ml/value/{name}'), str(audit / name), '--ignore-missing-args', dry_run=dry_run)
    remote.run(f'mkdir -p {shlex.quote(remote.root + "/data/ml/value/teams")}')
    remote.rsync(remote.path('data/ml/value/teams') + '/', f'{audit}/teams/', '--include=*/', '--include=gates/**',
                 '--exclude=*', '--prune-empty-dirs', dry_run=dry_run)
    return len(files)


def _host_dir(remote: Remote) -> str:
    return remote.host.lower()


def live_games(team: str, value_root: Path = VALUE_ROOT) -> int:
    return len(list((value_root / 'teams' / _slug(team) / 'live').glob('live-*.jsonl')))


def campaigns(value_root: Path = VALUE_ROOT) -> list[dict]:
    try:
        return [json.loads(x) for x in (value_root / 'campaigns.jsonl').read_text().splitlines() if x.strip()]
    except OSError:
        return []


def _log_campaign(record: dict, value_root: Path = VALUE_ROOT):
    with (value_root / 'campaigns.jsonl').open('a') as fh:
        fh.write(json.dumps(record) + '\n')


def campaign_ready(team: str, every: int = 1000, value_root: Path = VALUE_ROOT) -> dict:
    """Live games since the last campaign started (its snapshot), and whether a new one is due."""
    started = [c for c in campaigns(value_root) if c.get('event') == 'start' and _slug(c.get('team', '')) == _slug(team)]
    last = started[-1] if started else None
    games = live_games(team, value_root)
    since = games - (last['live_games'] if last else 0)
    return {'live_games': games, 'since_last': since, 'every': every, 'ready': since >= every,
            'last_campaign': last['id'] if last else None}


def _remote_json(remote: Remote, rel: str) -> dict | None:
    proc = remote.run(f'cat {shlex.quote(remote.root + "/" + rel)}', check=False)
    try:
        return json.loads(proc.stdout) if proc.returncode == 0 else None
    except ValueError:
        return None


def _write_remote_json(remote: Remote, rel: str, value: dict):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'f.json').write_text(json.dumps(value, indent=2))
        remote.rsync(f'{tmp}/f.json', remote.path(rel))


def _trainer_agent(remote: Remote, start: bool):
    if remote.local:
        return  # rehearsals: no LaunchAgent for a stand-in directory
    uid = 'gui/$(id -u)'
    if start:
        remote.run(f'launchctl bootout {uid}/{TRAINER_LABEL} 2>/dev/null; '
                   f'launchctl bootstrap {uid} $HOME/Library/LaunchAgents/{TRAINER_LABEL}.plist')
    else:
        remote.run(f'launchctl bootout {uid}/{TRAINER_LABEL}', check=False)


def _ready_remote(remote: Remote):
    if not remote.reachable():
        raise SystemExit(f'{remote.host} is not reachable (asleep?): wake it, plug it in and retry')
    if remote.run(f'test -x {shlex.quote(remote.root + "/.venv/bin/python")}', check=False).returncode:
        raise SystemExit('the trainer is not set up: run .venv/bin/python -m search.mlops bootstrap')


def campaign_start(selfplay_games: int | None = None, force: bool = False) -> dict:
    from search.value_service import load_config as service_config
    cfg = load_config()
    scfg = service_config()
    team = scfg['team']
    ready = campaign_ready(team, scfg['campaign_every_games'])
    if not ready['ready'] and not force:
        raise SystemExit(f"only {ready['since_last']} new live games since the last campaign "
                         f"(campaigns run every {ready['every']}); pass --force to start anyway")
    remote = Remote(cfg)
    _ready_remote(remote)
    status = _remote_json(remote, 'data/ml/value/service-status.json') or {}
    running = (status.get('campaign') or {})
    if running and running.get('phase') != 'done' and not force:
        raise SystemExit(f"campaign {running.get('id')} is still {running.get('phase')} on {remote.host}; "
                         'finish it first (or --force to replace it)')
    print('copying code and a data snapshot to the trainer...', flush=True)
    push_code(remote)
    push_data(remote, team, settle=cfg['settle_seconds'])
    games = int(selfplay_games or cfg['campaign_selfplay_games'])
    camp_id = time.strftime('%Y%m%dT%H%M%S')
    tcfg = _remote_json(remote, 'data/ml/value/config.json') or dict(TRAINER_SERVICE_CONFIG)
    tcfg.update(role='trainer', team=team, enabled=True, train_max_rows=int(cfg['campaign_train_max_rows']),
                campaign={'id': camp_id, 'selfplay_games': games})
    _write_remote_json(remote, 'data/ml/value/config.json', tcfg)
    _trainer_agent(remote, start=True)
    record = {'event': 'start', 'id': camp_id, 'team': team, 'at': time.time(), 'live_games': ready['live_games'],
              'selfplay_games': games, 'code': play_fingerprint(), 'host': remote.host}
    _log_campaign(record)
    return record


def campaign_status() -> dict:
    from search.value_service import load_config as service_config
    cfg = load_config()
    scfg = service_config()
    out = {'ready': campaign_ready(scfg['team'], scfg['campaign_every_games'])}
    history = campaigns()
    out['last'] = history[-1] if history else None
    starts = [c for c in history if c.get('event') == 'start']
    if starts and starts[-1].get('code') != play_fingerprint():
        out['warning'] = 'play code changed since the campaign started: its models will be stale'
    remote = Remote(cfg)
    if remote.reachable():
        status = _remote_json(remote, 'data/ml/value/service-status.json') or {}
        out['trainer'] = {k: status.get(k) for k in ('updated_at', 'campaign', 'selfplay_rows', 'gate')}
        out['trainer']['candidates'] = (status.get('candidates') or [])[-3:]
        out['trainer']['screen'] = [c for c in ((status.get('screen') or {}).get('candidates') or [])][-2:]
        out['trainer']['events'] = (status.get('events') or [])[-6:]
        camp = status.get('campaign') or {}
        if camp.get('phase') == 'selfplay':
            out['trainer']['progress'] = f"{camp.get('scheduled_games')}/{camp.get('selfplay_games')} self-play games started"
    else:
        out['trainer'] = f'{remote.host} unreachable (asleep?)'
    return out


def campaign_finish(deploy: bool = True, force: bool = False) -> dict:
    cfg = load_config()
    remote = Remote(cfg)
    _ready_remote(remote)
    status = _remote_json(remote, 'data/ml/value/service-status.json') or {}
    camp = status.get('campaign') or {}
    if camp.get('phase') != 'done' and not force:
        raise SystemExit(f"campaign {camp.get('id')} is {camp.get('phase')}; wait for done "
                         '(or --force to collect what is finished and stop it)')
    pulled = pull_results(remote)
    _trainer_agent(remote, start=False)  # Macintosh is free again until the next campaign
    out = {'event': 'finish', 'id': camp.get('id'), 'at': time.time(), 'phase': camp.get('phase'),
           'exported': camp.get('exported'), 'pulled_files': pulled}
    out['deploys'] = deploy_inbox() if deploy else 'skipped: run .venv/bin/python -m search.mlops deploy'
    _log_campaign(out)
    return out


# ------------------------------------------------------------ setup
def _plist(label: str, argv: list[str], cwd: str, log: str, extra: str, path_env: str) -> str:
    args = ''.join(f'\n\t\t<string>{a}</string>' for a in argv)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>{label}</string>
	<key>ProgramArguments</key>
	<array>{args}
	</array>
	<key>WorkingDirectory</key>
	<string>{cwd}</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>PATH</key>
		<string>{path_env}</string>
		<key>OMP_NUM_THREADS</key>
		<string>2</string>
	</dict>
	<key>StandardOutPath</key>
	<string>{log}.log</string>
	<key>StandardErrorPath</key>
	<string>{log}-errors.log</string>
{extra}</dict>
</plist>
'''


TRAINER_SERVICE_CONFIG = {
    # Trainer: no live ledgers to ingest; passed candidates are exported, not promoted.
    'role': 'trainer', 'producers': 4, 'gate_slots': 6, 'gate_min_slots': 3, 'slot_limit_extra': 2,
    'max_selfplay_gb': 40.0, 'max_promotions_per_day': 10, 'campaign': None,
}


def bootstrap(no_start: bool = False, skip_env: bool = False, keep_divemac_training: bool = False) -> dict:
    """Idempotent: trainer checkout, environment, config and its (unloaded) LaunchAgent."""
    from search.value_service import load_config as service_config
    cfg = load_config()
    remote = Remote(cfg)
    if not remote.reachable():
        raise SystemExit(f'{remote.host} is not reachable (asleep?); wake it and retry')
    r = remote.root
    remote.run(f'mkdir -p {shlex.quote(r)}/data/ml/logs {shlex.quote(r)}/data/ml/value')
    if not remote.local and remote.run(f'test -x {shlex.quote(cfg["remote_rsync"])}', check=False).returncode:
        raise SystemExit(f'{cfg["remote_rsync"]} missing on {remote.host}: brew install rsync there, '
                         'or set remote_rsync in data/ml/value/mlops.json')
    node = remote.run('command -v node || ls /opt/homebrew/bin/node', check=False).stdout.strip().splitlines()
    if not node:
        raise SystemExit(f'node is missing on {remote.host} (brew install node)')
    print('pushing code and inputs (first time ~1 GB)...', flush=True)
    push_code(remote)
    if not skip_env:
        if remote.run('export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH; command -v uv', check=False).returncode:
            raise SystemExit(f'uv is missing on {remote.host} (curl -LsSf https://astral.sh/uv/install.sh | sh)')
        print('building the Python environment...', flush=True)
        remote.run(f'cd {shlex.quote(r)} && export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH && '
                   'uv venv --allow-existing --python 3.14 .venv && uv pip install --python .venv/bin/python '
                   '-r requirements-ml.txt', timeout=3600)
        remote.python('-c "import torch; assert torch.backends.mps.is_available(), \'no MPS\'; print(torch.__version__)"')
    # Resource policy: the local autopilot.json minus PPO, sized for the trainer.
    from ml.continuous import LearningConfig
    auto = json.loads((ROOT / 'data' / 'ml' / 'autopilot.json').read_text())
    ncpu = int(remote.run('sysctl -n hw.ncpu').stdout.strip() or 8)
    auto['enabled'] = False  # no PPO autopilot on the trainer
    auto['resource'].update(reserve_cores=2, max_workers=max(2, ncpu - 6), max_system_cpu_percent=90,
                            max_disk_gb=80.0, backend='mps')
    LearningConfig(**auto)  # validate before shipping
    scfg = {**TRAINER_SERVICE_CONFIG, 'team': service_config()['team'],
            'gate_slots': max(3, min(8, ncpu - 8))}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'autopilot.json').write_text(json.dumps(auto, indent=2))
        existing = remote.run(f'cat {shlex.quote(r)}/data/ml/value/config.json', check=False).stdout
        try:
            scfg = {**scfg, **json.loads(existing), 'role': 'trainer'}  # keep hand edits
        except ValueError:
            pass
        (Path(tmp) / 'config.json').write_text(json.dumps(scfg, indent=2))
        remote.rsync(f'{tmp}/autopilot.json', remote.path('data/ml/autopilot.json'))
        remote.rsync(f'{tmp}/config.json', remote.path('data/ml/value/config.json'))
        team = service_config()['team']
        push_data(remote, team, settle=cfg['settle_seconds'])
        node_dir = str(Path(node[0]).parent)
        path_env = f'{node_dir}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
        trainer = _plist(TRAINER_LABEL, ['/usr/bin/caffeinate', '-i', f'{r}/.venv/bin/python', '-u', '-m',
                                         'search.value_service'], r, f'{r}/data/ml/logs/value-trainer',
                         '\t<key>RunAtLoad</key>\n\t<true/>\n\t<key>KeepAlive</key>\n\t<true/>\n'
                         '\t<key>ThrottleInterval</key>\n\t<integer>60</integer>\n'
                         '\t<key>ProcessType</key>\n\t<string>Standard</string>\n', path_env)
        (Path(tmp) / f'{TRAINER_LABEL}.plist').write_text(trainer)
        agents = '$HOME/Library/LaunchAgents' if not remote.local else str(Path(r) / 'LaunchAgents')
        remote.run(f'mkdir -p {agents}')
        dest = (f'{remote.host}:Library/LaunchAgents/' if not remote.local else agents + '/')
        remote.rsync(f'{tmp}/{TRAINER_LABEL}.plist', dest)
    out = {'trainer': remote.host, 'root': r, 'team': team, 'gate_slots': scfg['gate_slots'],
           'max_workers': auto['resource']['max_workers']}
    # The trainer agent is installed, not loaded: `campaign start` loads it, `campaign finish` unloads it.
    if not no_start and not remote.local:
        if not keep_divemac_training:
            # One trainer: DiveMac keeps live play and ingest, its CPU goes to live turns.
            out['divemac_role'] = set_role('live')
    return out


def set_role(role: str) -> dict:
    """DiveMac's own value service: 'full' trains+gates here too, 'live' only ingests."""
    if role not in ('full', 'live'):
        raise SystemExit("role must be 'full' or 'live' (the trainer role belongs on Macintosh)")
    path = VALUE_ROOT / 'config.json'
    try:
        cfg = json.loads(path.read_text())
    except (OSError, ValueError):
        cfg = {}
    before = cfg.get('role', 'full')
    cfg['role'] = role
    atomic_json(path, cfg)
    return {'role': role, 'before': before, 'note': 'read every tick; running gates/shards finish on their own'}


def status() -> dict:
    from search.value_service import load_config as service_config
    team = service_config()['team']
    tdir = VALUE_ROOT / 'teams' / _slug(team)
    out = {'team': team, 'divemac_role': service_config().get('role', 'full')}
    try:
        report = json.loads(subprocess.run([sys.executable, str(ROOT / 'search' / 'learning_report.py'), '--json',
                                            '--hours', '24'], capture_output=True, text=True, timeout=120).stdout)
        out['live_24h'] = {k: report['live'].get(k) for k in ('games', 'wins', 'losses', 'win_rate',
                                                              'rating_first_last', 'decision_ms_median', 'incidents')}
    except (ValueError, KeyError, subprocess.SubprocessError):
        out['live_24h'] = 'unavailable'
    from search.screen_data import current_screen
    from search.value_store import current_value
    out['live_uses'] = {'value': current_value(tdir / 'current.json').get('sha256') or 'hand evaluation',
                        'screen': current_screen(tdir / 'screen-current.json').get('sha256') or 'probe screening'}
    out['campaign'] = campaign_ready(team, service_config()['campaign_every_games'])
    history = campaigns()
    out['last_campaign_event'] = history[-1] if history else None
    try:
        out['deploys'] = [json.loads(x) for x in (VALUE_ROOT / 'mlops-deploys.jsonl').read_text().splitlines()[-5:]]
    except OSError:
        out['deploys'] = []
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    p = sub.add_parser('campaign'); p.add_argument('action', choices=('start', 'status', 'finish'))
    p.add_argument('--selfplay-games', type=int); p.add_argument('--force', action='store_true')
    p.add_argument('--no-deploy', action='store_true')
    p = sub.add_parser('deploy'); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('rollback'); p.add_argument('--kind', choices=('value', 'screen'), required=True)
    p.add_argument('--to', choices=('previous', 'none'), default='previous')
    p = sub.add_parser('bootstrap'); p.add_argument('--no-start', action='store_true'); p.add_argument('--skip-env', action='store_true')
    p.add_argument('--keep-divemac-training', action='store_true', help="leave DiveMac's value service in role full")
    p = sub.add_parser('role'); p.add_argument('role')
    sub.add_parser('export-list')  # trainer side, used by campaign finish
    sub.add_parser('fingerprint')
    args = ap.parse_args()
    if args.cmd == 'status':
        out = status()
    elif args.cmd == 'campaign':
        if args.action == 'start':
            out = campaign_start(args.selfplay_games, force=args.force)
        elif args.action == 'status':
            out = campaign_status()
        else:
            out = campaign_finish(deploy=not args.no_deploy, force=args.force)
    elif args.cmd == 'deploy':
        out = deploy_inbox(dry_run=args.dry_run)
    elif args.cmd == 'rollback':
        out = rollback(args.kind, args.to)
    elif args.cmd == 'bootstrap':
        out = bootstrap(no_start=args.no_start, skip_env=args.skip_env, keep_divemac_training=args.keep_divemac_training)
    elif args.cmd == 'role':
        out = set_role(args.role)
    elif args.cmd == 'export-list':
        print(json.dumps({'files': export_list()}))  # one line: campaign finish parses the last line
        return
    else:
        print(play_fingerprint())
        return
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
