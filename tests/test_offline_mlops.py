"""Train on Macintosh, deploy on DiveMac: bundles, verification, deployment, rollback."""
import json
import os
import time
from pathlib import Path

import pytest

from search import mlops, value_service, value_store
from search.screen_data import current_screen

TEAM = 'Rain-Recife-special-stat-fix'
PASS = {'complete': True, 'passed': True, 'gains': 12, 'losses': 2, 'p_one_sided': 0.006}


def _value(path: Path, seed: float = 0.1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'layers': [{'W': [[seed]], 'b': [0]}], 'mean': [0], 'std': [1], 'hand': {'a': 1, 'b': 0}}))
    return path


def _screen(path: Path, seed: float = 0.1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'layers': [{'W': [[seed]], 'b': [0]}], 'mean': [0], 'std': [1],
                                'prefilter': {'opp': 8, 'ours': 10}, 'version': 1}))
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A trainer value root and a live value root with the same team."""
    live = tmp_path / 'live' / 'data' / 'ml' / 'value'
    trainer = tmp_path / 'trainer' / 'data' / 'ml' / 'value'
    for root in (live, trainer):
        (root / 'teams' / 'rain-recife-special-stat-fix').mkdir(parents=True)
    monkeypatch.setattr(value_service, 'load_config', lambda: {**value_service.DEFAULTS, 'team': TEAM})
    monkeypatch.setattr(mlops, 'play_fingerprint', lambda root=None: 'code-1')
    return live, trainer


def _export(trainer: Path, kind: str = 'value', seed: float = 0.1, incumbent=None, screen_incumbent=None,
            gates=None) -> Path:
    tdir = trainer / 'teams' / 'rain-recife-special-stat-fix'
    if kind == 'value':
        path = _value(tdir / 'candidates' / f'value-{seed}.json', seed)
        sha = value_store.weights_sha(path)
    else:
        path = _screen(tdir / 'screen-candidates' / f'screen-{seed}.json', seed)
        from search.screen_data import weights_sha
        sha = weights_sha(path)
    cand = {'path': str(path), 'sha256': sha, 'beta': 0.5, 'gates': gates or {'dev': PASS, 'confirm': PASS},
            'prefilter': {'opp': 8, 'ours': 10}}
    gate = {'incumbent': {'sha256': incumbent}, 'screen_incumbent': {'sha256': screen_incumbent}, 'code': 'code-1'}
    return mlops.export_bundle(kind, cand, gate, TEAM, value_root=trainer)


def _deliver(bundle: Path, live: Path) -> Path:
    import shutil
    dest = live / 'inbox' / bundle.name
    shutil.copytree(bundle, dest)
    return dest


def test_passed_value_bundle_deploys_and_live_pointer_resolves(env):
    live, trainer = env
    bundle = _deliver(_export(trainer), live)
    [result] = mlops.deploy_inbox(value_root=live)
    assert result['status'] == 'deployed'
    pointer = live / 'teams' / 'rain-recife-special-stat-fix' / 'current.json'
    current = value_store.current_value(pointer)
    assert current['sha256'] == json.loads((bundle / 'manifest.json').read_text())['sha256']
    assert Path(current['path']).parent.name == 'candidates' and str(live) in current['path']
    assert current['beta'] == 0.5
    # Processed once: a later sync never redeploys or re-logs it.
    assert mlops.deploy_inbox(value_root=live) == []


def test_screen_bundle_deploys_to_its_own_pointer(env):
    live, trainer = env
    _deliver(_export(trainer, kind='screen'), live)
    [result] = mlops.deploy_inbox(value_root=live)
    assert result['status'] == 'deployed'
    screen = current_screen(live / 'teams' / 'rain-recife-special-stat-fix' / 'screen-current.json')
    assert screen['pre_opp'] == 8 and screen['pre_ours'] == 10
    assert value_store.current_value(live / 'teams' / 'rain-recife-special-stat-fix' / 'current.json')['path'] is None


@pytest.mark.parametrize('damage,reason', [
    ('gate', 'rejected: confirm gate not passed'),
    ('code', 'stale: play code changed since the gate'),
    ('incumbent', 'stale: value incumbent changed since the gate'),
    ('team', "rejected: team 'Other' is not the deployed"),
])
def test_unsafe_bundles_never_reach_the_pointer(env, damage, reason):
    live, trainer = env
    gates = {'dev': PASS, 'confirm': {**PASS, 'passed': False}} if damage == 'gate' else None
    bundle = _deliver(_export(trainer, gates=gates), live)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if damage == 'tamper':
        (bundle / manifest['file']).write_text((bundle / manifest['file']).read_text().replace('0.1', '0.2'))
    if damage == 'code':
        manifest['code'] = 'code-0'
    if damage == 'team':
        manifest['team'] = 'Other'
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    if damage == 'incumbent':  # live promoted something else meanwhile
        other = _value(live / 'teams' / 'rain-recife-special-stat-fix' / 'candidates' / 'other.json', 0.9)
        value_store.promote(other, 1.0, {}, live / 'teams' / 'rain-recife-special-stat-fix' / 'current.json')
    [result] = mlops.deploy_inbox(value_root=live)
    assert result['status'].startswith(reason)
    pointer = live / 'teams' / 'rain-recife-special-stat-fix' / 'current.json'
    assert value_store.current_value(pointer).get('sha256') != manifest['sha256']
    assert mlops.deploy_inbox(value_root=live) == []  # never retried


def test_chained_bundles_deploy_in_export_order_within_the_daily_limit(env, monkeypatch):
    live, trainer = env
    first = _export(trainer, seed=0.1)
    sha1 = json.loads((first / 'manifest.json').read_text())['sha256']
    time.sleep(0.01)
    second = _export(trainer, seed=0.2, incumbent=sha1)  # gated against the first
    _deliver(second, live); _deliver(first, live)
    results = mlops.deploy_inbox(value_root=live)
    assert [r['status'] for r in results] == ['deployed', 'wait: daily promotion limit reached']
    # Next day the waiting bundle (gated vs the now-live first one) deploys.
    monkeypatch.setattr(value_service, 'load_config',
                        lambda: {**value_service.DEFAULTS, 'team': TEAM, 'max_promotions_per_day': 2})
    assert [r['status'] for r in mlops.deploy_inbox(value_root=live)] == ['deployed']


def test_rollback_returns_to_previous_then_to_hand(env):
    live, trainer = env
    tdir = live / 'teams' / 'rain-recife-special-stat-fix'
    a = _value(tdir / 'candidates' / 'a.json', 0.1)
    b = _value(tdir / 'candidates' / 'b.json', 0.2)
    value_store.promote(a, 0.5, {}, tdir / 'current.json')
    value_store.promote(b, 0.7, {}, tdir / 'current.json')
    mlops.rollback('value', value_root=live)
    assert value_store.current_value(tdir / 'current.json')['sha256'] == value_store.weights_sha(a)
    mlops.rollback('value', to='none', value_root=live)
    assert value_store.current_value(tdir / 'current.json')['path'] is None
    # Rollbacks do not count against the daily promotion limit.
    ok, reason, _ = mlops.verify_bundle(_export(trainer, seed=0.3), TEAM, live / 'teams', 'code-1', 3)
    assert ok, reason


def test_pointer_from_another_checkout_resolves_locally(tmp_path, monkeypatch):
    root = tmp_path / 'trainer'
    cand = _value(root / 'data' / 'ml' / 'value' / 'teams' / 't' / 'candidates' / 'v.json')
    monkeypatch.setattr(value_store, 'ROOT', root)
    pointer = tmp_path / 'current.json'
    pointer.write_text(json.dumps({'path': '/srv/other-checkout/data/ml/value/teams/t/candidates/v.json',
                                   'sha256': value_store.weights_sha(cand), 'beta': 0.5}))
    assert value_store.current_value(pointer)['path'] == str(cand)


def test_export_list_skips_active_and_unsettled_files(tmp_path):
    vroot = tmp_path / 'data' / 'ml' / 'value'
    tdir = vroot / 'teams' / 'x'
    for name in ('selfplay/sp-trainer-1.jsonl', 'selfplay/sp-trainer-2.jsonl', 'selfplay/sp-9.jsonl',
                 'screen/gate-a-dev-1-cand-p1.npz', 'screen/live-room.npz', 'screen/gate-b.tmp.npz'):
        (tdir / name).parent.mkdir(parents=True, exist_ok=True)
        (tdir / name).write_text('x')
        os.utime(tdir / name, (time.time() - 600,) * 2)
    os.utime(tdir / 'selfplay' / 'sp-trainer-2.jsonl')  # just written
    (vroot / 'service-state.json').write_text(json.dumps({'producers': {'3': {'shard': str(tdir / 'selfplay' / 'sp-trainer-1.jsonl')}}}))
    assert mlops.export_list(vroot) == ['data/ml/value/teams/x/screen/gate-a-dev-1-cand-p1.npz']


def test_trainer_role_exports_instead_of_promoting(env, monkeypatch):
    live, trainer = env
    svc = value_service.Service.__new__(value_service.Service)
    tdir = trainer / 'teams' / 'rain-recife-special-stat-fix'
    cand_path = _value(tdir / 'candidates' / 'v.json')
    sha = value_store.weights_sha(cand_path)
    report = trainer / 'gate.json'
    report.write_text(json.dumps(PASS))
    svc.team, svc.tdir, svc.events = TEAM, tdir, []
    svc.vroot = tdir.parents[1]
    svc.state = {'candidates': [{'team': TEAM, 'path': str(cand_path), 'sha256': sha, 'beta': 0.5, 'status': 'gating:confirm',
                                 'gates': {'dev': PASS}}], 'screen_candidates': [],
                 'gate': {'kind': 'value', 'sha256': sha, 'stage': 'confirm', 'report': str(report), 'code': 'code-1',
                          'incumbent': {'sha256': None}, 'screen_incumbent': {'sha256': None}}}
    monkeypatch.setattr(svc, 'pointer', lambda: tdir / 'current.json', raising=False)
    svc.finish_gate(0, {**value_service.DEFAULTS, 'role': 'trainer'})
    assert svc.state['candidates'][0]['status'] == 'exported'
    # Its own pointer now names the exported model (the next gate's incumbent); live never reads it.
    assert value_store.current_value(tdir / 'current.json')['sha256'] == sha
    assert (trainer / 'outbox' / f'value-{sha[:16]}' / 'manifest.json').exists()


def test_tampered_or_half_transferred_bundle_waits_and_deploys_once_complete(env):
    live, trainer = env
    source = _export(trainer)
    bundle = _deliver(source, live)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    weights = bundle / manifest['file']
    good = weights.read_text()
    weights.write_text(good[:10])  # a sync interrupted mid-file
    pointer = live / 'teams' / 'rain-recife-special-stat-fix' / 'current.json'
    for _ in range(2):  # retried every sync, never deployed while damaged
        [result] = mlops.deploy_inbox(value_root=live)
        assert result['status'].startswith('wait: weights')
        assert value_store.current_value(pointer)['sha256'] is None
    weights.write_text(good)  # the next sync completed it
    assert [r['status'] for r in mlops.deploy_inbox(value_root=live)] == ['deployed']


def test_full_role_promotes_only_through_the_inbox_checks(env, monkeypatch):
    live, _ = env
    svc = value_service.Service.__new__(value_service.Service)
    tdir = live / 'teams' / 'rain-recife-special-stat-fix'
    cand_path = _value(tdir / 'candidates' / 'v.json')
    sha = value_store.weights_sha(cand_path)
    report = live / 'gate.json'
    report.write_text(json.dumps(PASS))
    svc.team, svc.tdir, svc.events = TEAM, tdir, []
    svc.vroot = tdir.parents[1]
    svc.state = {'candidates': [{'team': TEAM, 'path': str(cand_path), 'sha256': sha, 'beta': 0.5, 'status': 'gating:confirm',
                                 'gates': {'dev': PASS}}], 'screen_candidates': [],
                 'gate': {'kind': 'value', 'sha256': sha, 'stage': 'confirm', 'report': str(report), 'code': 'code-1',
                          'incumbent': {'sha256': None}, 'screen_incumbent': {'sha256': None}}}
    monkeypatch.setattr(svc, 'pointer', lambda: tdir / 'current.json', raising=False)
    svc.finish_gate(0, {**value_service.DEFAULTS, 'role': 'full'})
    assert svc.state['candidates'][0]['status'] == 'promoted'
    assert value_store.current_value(tdir / 'current.json')['sha256'] == sha
    assert json.loads((live / 'mlops-deploys.jsonl').read_text().splitlines()[-1])['status'] == 'deployed'


def test_trainer_runs_one_bounded_campaign_then_idles(tmp_path, monkeypatch):
    from tests.test_offline_learned_screening import _service
    svc, spawned, _ = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(value_service, 'export_bundle', lambda *a, **k: mlops.export_bundle(*a, **{**k, 'value_root': svc.vroot}))
    res = {'training_allowed': True, 'allowed_workers': 8, 'slot_pool': 8}
    cfg = {**value_service.load_config(), 'role': 'trainer', 'producers': 8, 'games_per_shard': 20,
           'campaign': {'id': 'c1', 'selfplay_games': 50}}
    alive = set()
    monkeypatch.setattr(value_service, '_alive', lambda pid, marker='search/': pid in alive)

    def tick():
        camp = svc.campaign(cfg)
        svc.maybe_gate(cfg, res); svc.maybe_train(cfg, res); svc.maybe_train_screen(cfg, res); svc.produce(cfg, res)
        svc.advance_campaign(camp, cfg)
        return camp

    camp = tick()
    for _ in range(5):
        tick()
    shards = [n for n, _ in spawned if n.startswith('prod-')]
    assert len(shards) == 3 and camp['scheduled_games'] == 60  # 3 x 20 >= 50, then no more
    assert all('sp-trainer-' in ' '.join(a) for n, a in spawned if n.startswith('prod-'))
    assert not any(n == 'train' for n, _ in spawned) and camp['phase'] == 'selfplay'  # no training mid-simulation
    for name in shards:  # the simulation finishes
        svc.children.pop(name)
        svc.state['producers'].pop(name.split('-', 1)[1])
    tick()
    assert camp['phase'] == 'train'
    tick()
    assert [n for n, _ in spawned].count('train') == 1  # no row threshold in a campaign
    # Training finishes with a useful candidate; too few screening worlds -> ranker skipped.
    cand = _value(svc.tdir / 'candidates' / 'v.json')
    report = Path(svc.state['pending_train']['report'])
    report.write_text(json.dumps({'path': str(cand), 'sha256': value_store.weights_sha(cand), 'useful': True,
                                  'recommended_beta': 0.5}))
    svc.children.pop('train'); svc.finish_train(0)
    tick()
    assert camp['value_trained'] and camp['screen_trained'] and camp['phase'] == 'gate'
    tick()  # dev gate starts
    gate_report = Path(svc.state['gate']['report'])
    for stage in ('dev', 'confirm'):
        gate_report.write_text(json.dumps(PASS))
        svc.children.pop('gate', None); svc.state['gate'].pop('pid', None)
        svc.finish_gate(0, cfg)
        if stage == 'dev':
            tick()
            gate_report = Path(svc.state['gate']['report'])
    assert svc.state['candidates'][-1]['status'] == 'exported'
    tick()
    assert camp['phase'] == 'done' and camp['exported'] == [value_store.weights_sha(cand)[:16]]
    n = len(spawned)
    for _ in range(3):  # done: nothing more until the next campaign id
        if camp['phase'] != 'done':
            tick()
    assert len(spawned) == n
    assert (svc.vroot / 'outbox' / f"value-{value_store.weights_sha(cand)[:16]}" / 'manifest.json').exists()


def test_campaign_ready_counts_live_games_since_last_start(tmp_path):
    vroot = tmp_path / 'value'
    live = vroot / 'teams' / 'rain-recife-special-stat-fix' / 'live'
    live.mkdir(parents=True)
    for i in range(12):
        (live / f'live-{i}.jsonl').write_text('{}')
    assert mlops.campaign_ready(TEAM, 10, vroot)['ready']
    mlops._log_campaign({'event': 'start', 'id': 'c1', 'team': TEAM, 'live_games': 12}, vroot)
    (live / 'live-99.jsonl').write_text('{}')
    ready = mlops.campaign_ready(TEAM, 10, vroot)
    assert ready == {'live_games': 13, 'since_last': 1, 'every': 10, 'ready': False, 'last_campaign': 'c1'}


def test_one_campaign_deploys_value_and_screening_in_export_order(env, monkeypatch):
    live, trainer = env
    monkeypatch.setattr(value_service, 'load_config',
                        lambda: {**value_service.DEFAULTS, 'team': TEAM, 'max_promotions_per_day': 1})
    first = _export(trainer, kind='value')
    sha_v = json.loads((first / 'manifest.json').read_text())['sha256']
    time.sleep(0.01)
    second = _export(trainer, kind='screen', incumbent=sha_v)  # gated with the campaign's new value
    _deliver(second, live); _deliver(first, live)
    assert [r['status'] for r in mlops.deploy_inbox(value_root=live)] == ['deployed', 'deployed']
    tdir = live / 'teams' / 'rain-recife-special-stat-fix'
    assert value_store.current_value(tdir / 'current.json')['sha256'] == sha_v
    assert current_screen(tdir / 'screen-current.json')['sha256'] == json.loads((second / 'manifest.json').read_text())['sha256']
