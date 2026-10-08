import asyncio
import hashlib
import json

import pytest

from ml.continuous import LearningConfig
from ml.model import Model
from ml.promotion import stage, promote_ready
from ml.reload import SourceChanged, SourceGuard
from ml.storage import Store

FMT = 'gen9championsvgc2026regmc'


def test_source_change_invalidates_a_ready_candidate_without_replacing_actor(tmp_path, monkeypatch):
    store = Store(tmp_path)
    actor = Model(tmp_path, FMT)
    candidate = Model(tmp_path / 'candidate', FMT)
    original = hashlib.sha256(actor.path.read_bytes()).hexdigest()
    generation = ['old']
    monkeypatch.setattr('ml.reload.source_generation', lambda: generation[0])
    try:
        assert stage(store, FMT, actor.revision, candidate.path, {'passed': True})
        generation[0] = 'new'
        output = promote_ready(store, FMT)
        assert not output['promoted'] and 'source changed' in output['reason']
        assert hashlib.sha256(actor.path.read_bytes()).hexdigest() == original
        assert candidate.path.exists()
        assert not (tmp_path / 'ready' / (FMT + '.json')).exists()
    finally:
        store.close()


def test_guard_stops_updates_when_source_changes(monkeypatch):
    generation = ['old']
    monkeypatch.setattr('ml.reload.source_generation', lambda: generation[0])
    guard = SourceGuard('old')
    guard(force=True)
    generation[0] = 'new'
    with pytest.raises(SourceChanged):
        guard(force=True)


def test_partial_deployment_refuses_runtime_initialization(monkeypatch):
    from ml import reload
    monkeypatch.setattr(reload, 'RUNTIME_GENERATION', 'previous')
    monkeypatch.setattr(reload, 'source_generation', lambda: 'new')
    with pytest.raises(SourceChanged):
        reload.pin_runtime('old')
    assert reload.RUNTIME_GENERATION == 'previous'


def test_recording_keeps_running_generation_when_disk_source_changes(tmp_path, monkeypatch):
    from ml.brain import Brain
    from ml.features import Features
    from tests.test_offline_learning import context
    monkeypatch.setattr('ml.reload.RUNTIME_GENERATION', 'running-battle')
    monkeypatch.setattr('ml.reload.source_generation', lambda: 'new-deployment')
    store = Store(tmp_path)
    try:
        brain = Brain(store, Features())
        brain.decide(context(), explore=True, record=True)
        episode = next(iter(brain.pending.values()))
        assert episode['collecting_policy']['source_generation'] == 'running-battle'
    finally:
        store.close()


def test_stale_evaluation_cannot_run_cases_or_stage(tmp_path, monkeypatch):
    from ml.worker import _evaluate_saved
    store = Store(tmp_path)
    work = tmp_path / 'candidate'
    work.mkdir()
    (work / 'training-state.json').write_text('{}')
    (work / 'evaluation-plan.json').write_text(json.dumps({'source_generation': 'old'}))
    monkeypatch.setattr('ml.worker.source_generation', lambda: 'new')
    async def unexpected(*args):
        pytest.fail('must not evaluate with another code generation')
    monkeypatch.setattr('ml.worker.parallel_games', unexpected)
    try:
        result = asyncio.run(_evaluate_saved(store, {'format': FMT}, None, None, 0, work))
        assert not result['trained'] and 'source changed' in result['reason']
        assert not (tmp_path / 'ready').exists()
    finally:
        store.close()


def test_supervisor_drains_owned_child_and_requests_restart_without_killing(tmp_path, monkeypatch):
    from ml import pipeline
    store = Store(tmp_path)
    store.enqueue('one', 'learn', FMT)
    store.close()
    LearningConfig(feed_daily=False).save(tmp_path)
    generations = iter(['old', 'new'])
    monkeypatch.setattr(pipeline, 'LOADED_GENERATION', 'old')
    monkeypatch.setattr(pipeline, 'source_generation', lambda: next(generations))
    monkeypatch.setattr(pipeline.ResourcePolicy, 'sample', lambda self: {'training_allowed': True, 'deferred': False})
    monkeypatch.setattr(pipeline, 'queue_collection', lambda *a: {'queued': [], 'backlog': {}})
    class Child:
        pid = 123
        def __init__(self):
            self.checks = 0
        def poll(self):
            self.checks += 1
            return 0 if self.checks > 3 else None
        def terminate(self):
            pytest.fail('must finish child naturally')
        kill = terminate
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **k: Child())
    writes = []
    original = pipeline.atomic_status
    def capture(path, value):
        writes.append(value)
        original(path, value)
    monkeypatch.setattr(pipeline, 'atomic_status', capture)
    result = asyncio.run(pipeline.supervise(tmp_path, interval=.001, max_seconds=5))
    assert result['reload_requested']
    assert any(v.get('draining') for v in writes)
    assert writes[-1]['stopped']
    assert not pipeline.pipeline_running(tmp_path)
    store = Store(tmp_path)
    assert store.claim(kinds=('learn',))['id'] == 'one'
    store.close()
