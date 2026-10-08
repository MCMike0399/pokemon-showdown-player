import asyncio
import fcntl
import json
import time

import pytest

from ml.brain import Brain
from ml.continuous import LearningConfig
from ml.features import Features
from ml.model import Model
from ml.pipeline import pipeline_running, simulator_slots, queue_collection, status
from ml.resources import ResourcePolicy, BackgroundGuard, ResourceDeferred
from ml.storage import Store, now
from tests.test_offline_learning import context

FMT = 'gen9championsvgc2026regmc'


def test_claims_partition_jobs_and_reclaim_only_their_own_lane(tmp_path):
    store = Store(tmp_path)
    try:
        for kind in ('practice', 'learn', 'feed'):
            store.enqueue(kind, kind, FMT)
        assert store.claim(kinds=('learn', 'feed'))['id'] == 'learn'
        assert store.claim(kinds=('practice',))['id'] == 'practice'
        assert store.claim(kinds=('learn', 'feed'))['id'] == 'feed'
        assert store.claim(kinds=('practice',)) is None
        with store.db:
            store.db.execute("UPDATE jobs SET lease_until=0 WHERE id='learn'")
        assert store.claim(kinds=('practice',)) is None
        assert store.claim(kinds=('learn',))['id'] == 'learn'
        assert store.claim(kinds=()) is None
    finally:
        store.close()


def test_shared_slots_bound_two_pools_and_release_after_failure(tmp_path):
    with simulator_slots(tmp_path, 4, 3) as first:
        assert first == 3
        with simulator_slots(tmp_path, 4, 3) as second:
            assert second == 1
            with simulator_slots(tmp_path, 4, 1) as third:
                assert third == 0
        with simulator_slots(tmp_path, 4, 2) as recovered:
            assert recovered == 1
    with pytest.raises(ValueError):
        with simulator_slots(tmp_path, 4, 4) as all_slots:
            assert all_slots == 4
            raise ValueError('wave failed')
    with simulator_slots(tmp_path, 4, 4) as recovered:
        assert recovered == 4


def test_supervisor_lock_blocks_legacy_kicks_without_mutating_status(tmp_path, monkeypatch):
    from ml.worker import run
    from ml.continuous import kick_worker
    assert not pipeline_running(tmp_path)
    with (tmp_path / 'pipeline.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert pipeline_running(tmp_path)
        assert status(tmp_path)['supervisor_running']
        assert asyncio.run(run(tmp_path))['skipped']
        monkeypatch.delenv('PS_DISABLE_WORKER_KICK')
        monkeypatch.setattr('ml.continuous.subprocess.Popen', lambda *a, **k: pytest.fail('duplicate kick'))
        kick_worker(tmp_path)
    assert not pipeline_running(tmp_path)
    assert not (tmp_path / 'experience.sqlite3').exists()


def test_backpressure_dedupes_and_keeps_live_collection_independent(tmp_path):
    store = Store(tmp_path)
    model = Model(tmp_path, FMT)
    config = LearningConfig(min_training_steps=8, max_training_backlog_steps=16)
    try:
        assert len(queue_collection(store, config)['queued']) == 1
        assert not queue_collection(store, config)['queued']
        job = store.claim(kinds=('practice',))
        store.complete_job(job['id'], {})
        store.save_episode({'id': 'full', 'format': FMT, 'revision': model.revision,
            'source': 'local', 'status': 'complete', 'on_policy': True, 'steps': [{}] * 16})
        assert not queue_collection(store, config)['queued']
        store.mark_trained(['full'])
        (tmp_path / 'ready').mkdir()
        (tmp_path / 'ready' / (FMT + '.json')).write_text('{}')
        assert not queue_collection(store, config)['queued']
        assert store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1
    finally:
        store.close()


def test_frozen_knowledge_records_into_the_shared_experience_store(tmp_path):
    production, inputs = Store(tmp_path / 'production'), Store(tmp_path / 'inputs')
    try:
        with inputs.db:
            inputs.db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)',
                ('usage', 'https://example.com', FMT, 'Showdown usage test', now(), now(),
                 json.dumps({'species': [{'species': 'Pelipper', 'usage': .3}]}), 'digest'))
        brain = Brain(production, Features(), inference_store=inputs)
        brain.decide(context(), explore=True, record=True)
        brain.finish('local-test', {'winner': 'Player'}, 'Player')
        recorded = production.episodes(FMT)[0]
        assert recorded['steps'][0]['snapshot']['knowledge']['pelipper'] == .3
        assert recorded['steps'][0]['inference_ms'] >= 0
        assert not inputs.episodes(FMT)
    finally:
        production.close()
        inputs.close()


def test_pressure_guard_aborts_unsaved_training_without_consuming_data(tmp_path):
    store = Store(tmp_path)
    brain = Brain(store, Features())
    try:
        for index in range(40):
            ctx = context('pressure-' + str(index))
            brain.decide(ctx, explore=True, record=True)
            brain.finish(ctx['room'], {'winner': 'Player'}, 'Player')
        model = brain.model(FMT)
        before = model.path.read_bytes()
        calls = 0
        def checkpoint():
            nonlocal calls
            calls += 1
            # First optimizer step executes; next minibatch must yield.
            if calls == 5:
                raise ResourceDeferred('foreground pressure')
        with pytest.raises(ResourceDeferred):
            model.train(store.episodes(FMT), checkpoint=checkpoint)
        assert model.path.read_bytes() == before
        assert len(store.episodes(FMT)) == 40
    finally:
        store.close()


def test_yellow_pressure_allows_work_but_red_and_swap_rate_defer(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('ml.resources.psutil.virtual_memory', lambda: SimpleNamespace(available=8 * 2**30))
    monkeypatch.setattr('ml.resources.psutil.cpu_percent', lambda interval: 10)
    swap = SimpleNamespace(used=12 * 2**30, sout=0)
    monkeypatch.setattr('ml.resources.psutil.swap_memory', lambda: swap)
    monkeypatch.setattr('ml.resources.memory_pressure', lambda: 2)
    policy = ResourcePolicy()
    assert policy.sample()['training_allowed']
    monkeypatch.setattr('ml.resources.memory_pressure', lambda: 4)
    assert not policy.sample()['training_allowed']
    monkeypatch.setattr('ml.resources.memory_pressure', lambda: 6)
    assert not policy.sample()['training_allowed']
    monkeypatch.setattr('ml.resources.memory_pressure', lambda: 1)
    assert policy.sample()['training_allowed']  # Old swap usage alone is not pressure.
    policy._swap_sample = (time.monotonic() - 1, 0)
    swap.sout = 128 * 2**20
    assert not policy.sample()['training_allowed']
    guard = BackgroundGuard(policy)
    monkeypatch.setattr(policy, 'sample', lambda: {'training_allowed': False})
    with pytest.raises(ResourceDeferred):
        guard()


def test_scout_training_overlaps_cpu_evaluation_and_is_not_repeated(tmp_path, monkeypatch):
    import threading
    from ml.worker import evaluate_saved
    store = Store(tmp_path)
    work = tmp_path / 'candidate'
    work.mkdir()
    (work / 'training-state.json').write_text(json.dumps({'backend': 'mps'}))
    scout_started, evaluation_started = threading.Event(), threading.Event()
    def scout(*args):
        scout_started.set()
        assert evaluation_started.wait(3)
        return {'trained': True, 'promoted': False}
    async def evaluate(*args):
        assert await asyncio.to_thread(scout_started.wait, 3)
        evaluation_started.set()
        return {'evaluation': {'passed': False}}
    monkeypatch.setattr('ml.worker.scout_cycle', scout)
    monkeypatch.setattr('ml.worker._evaluate_saved', evaluate)
    try:
        result = asyncio.run(evaluate_saved(store, {'format': FMT}, LearningConfig(),
                                            ResourcePolicy(), time.monotonic() + 10, work))
        assert result['scout']['trained']
        assert not result['scout']['promoted']
        monkeypatch.setattr('ml.worker.scout_cycle', lambda *a: pytest.fail('duplicate GPU work'))
        assert asyncio.run(evaluate_saved(store, {'format': FMT}, LearningConfig(),
                                         ResourcePolicy(), time.monotonic() + 10, work))['scout']['trained']
    finally:
        store.close()


def test_durable_evaluation_handoff_consumes_once_without_promoting(tmp_path):
    from ml.worker import enqueue_evaluation
    store = Store(tmp_path)
    model = Model(tmp_path, FMT)
    work = tmp_path / 'work'
    work.mkdir()
    store.save_episode({'id': 'one', 'source': 'local', 'format': FMT, 'team': 'test',
                        'revision': model.revision, 'status': 'complete', 'steps': []})
    (work / 'training-state.json').write_text(json.dumps({'backend': 'mps', 'training': {
        'trained': True, 'previous_revision': model.revision, 'revision': 'new', 'consumed': ['one']}}))
    job = {'id': 'training-one', 'format': FMT}
    try:
        result = enqueue_evaluation(store, job, work)
        assert result['evaluation_pending'] and not result['promoted']
        assert store.db.execute("SELECT trained FROM episodes WHERE id='one'").fetchone()[0] == 1
        enqueue_evaluation(store, job, work)
        assert store.db.execute("SELECT COUNT(*) FROM jobs WHERE kind='evaluate'").fetchone()[0] == 1
        assert Model(tmp_path, FMT).revision == model.revision
        claimed = store.claim(kinds=('evaluate',))
        assert claimed['payload']['parent_revision'] == model.revision
    finally:
        store.close()


def test_candidate_backpressure_preserves_fresh_batch(tmp_path):
    from ml.worker import learn
    store = Store(tmp_path)
    model = Model(tmp_path, FMT)
    try:
        for i in range(2):
            store.enqueue('evaluate-' + str(i), 'evaluate', FMT, {'parent_revision': model.revision})
        policy = ResourcePolicy()
        policy._pipeline_lane = 'learner'
        result = asyncio.run(learn(store, {'id': 'third', 'format': FMT, 'kind': 'learn', 'payload': {}},
                                   LearningConfig(), policy, time.monotonic() + 10))
        assert result['deferred'] and 'backlog full' in result['reason']
        assert Model(tmp_path, FMT).revision == model.revision
        assert not (tmp_path / 'candidates' / 'third' / 'training-state.json').exists()
    finally:
        store.close()


def test_monitoring_keeps_large_rollouts_out_of_status_and_is_idempotent():
    from ml.pipeline import job_summary
    original = {'collection': {'games': [{'private_features': [1, 2, 3]}] * 2000},
                'training': {'trained': True, 'steps': 256, 'consumed': ['id'] * 2000},
                'backend': 'mps'}
    compact = job_summary(original)
    assert compact['collection']['games'] == 2000
    assert 'private_features' not in json.dumps(compact)
    assert 'consumed' not in compact['training']
    assert job_summary(compact) == compact
