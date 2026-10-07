import asyncio
import json

import pytest

from ml.brain import Brain
from ml.continuous import LearningConfig, learning_report
from ml.features import Features
from ml.model import Model
from ml.promotion import paired_gate, promote_ready, stage
from ml.storage import Store, now
from ml.worker import freeze_inputs, learn

FMT = 'gen9championsvgc2026regmc'


def games(winners):
    return [{'winner': 'LocalBrain' if win else 'LocalOpponent', 'seed': i,
             'learner_side': 'p1' if i % 2 else 'p2', 'opponent': 'heuristic',
             'rejected_actions': 0} for i, win in enumerate(winners)]


def test_gate_requires_paired_clean_and_convincing_gain():
    base = games([False] * 20)
    assert not paired_gate(base, games([True] * 2 + [False] * 18), 20, .1)['passed']
    better = games([True] * 6 + [False] * 14)
    assert paired_gate(base, better, 20, .1)['passed']
    better[0]['seed'] = 999
    assert not paired_gate(base, better, 20, .1)['passed']
    better[0]['seed'] = 0
    better[0]['rejected_actions'] = 1
    assert not paired_gate(base, better, 20, .1)['passed']
    assert not paired_gate(base, better[:10], 20, .1)['passed']


def test_staged_candidate_survives_live_game_and_reloads_at_boundary(tmp_path):
    store = Store(tmp_path)
    brain = Brain(store, Features())
    incumbent = brain.model(FMT)
    candidate = Model(tmp_path / 'candidate', FMT)
    gate = paired_gate(games([False] * 20), games([True] * 20), 20, .1)
    assert not stage(store, FMT, incumbent.revision, candidate.path, {'passed': False})
    assert stage(store, FMT, incumbent.revision, candidate.path, gate)
    episode = {'id': 'live', 'room': 'battle-test', 'format': FMT, 'team': 'fixture',
               'source': 'ladder', 'revision': incumbent.revision, 'status': 'pending', 'steps': []}
    store.save_episode(episode)
    assert not promote_ready(store, FMT)['promoted']
    assert Model(tmp_path, FMT).revision == incumbent.revision
    # Even an old pending episode protects its collecting checkpoint.
    with store.db:
        store.db.execute("UPDATE episodes SET created='2000-01-01' WHERE id='live'")
    assert not promote_ready(store, FMT)['promoted']
    episode.update(status='complete', outcome=1)
    store.save_episode(episode)
    assert promote_ready(store, FMT)['revision'] == candidate.revision
    assert brain.model(FMT).revision == candidate.revision
    assert not promote_ready(store, FMT)['promoted']
    store.close()


def test_staging_rejects_corruption_and_changed_parent(tmp_path):
    store = Store(tmp_path)
    incumbent = Model(tmp_path, FMT)
    candidate = Model(tmp_path / 'candidate', FMT)
    assert stage(store, FMT, incumbent.revision, candidate.path, {'passed': True})
    candidate.path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        promote_ready(store, FMT)
    assert Model(tmp_path, FMT).revision == incumbent.revision
    incumbent.revision = 'advanced'
    incumbent.save()
    assert promote_ready(store, FMT)['reason'] == 'stale parent revision'
    store.close()


def test_learning_report_requires_actual_job():
    assert not learning_report({'background_learning': {'job': None}})['queued']
    assert learning_report({'background_learning': {'job': 'real'}})['queued']


def test_small_training_batch_remains_unconsumed(tmp_path):
    store = Store(tmp_path)
    model = Model(tmp_path, FMT)
    episode = {'id': 'small', 'format': FMT, 'source': 'ladder', 'team': 'fixture',
               'revision': model.revision, 'status': 'complete', 'on_policy': True,
               'steps': [{}], 'outcome': 1}
    store.save_episode(episode)
    class Policy:
        def sample(self):
            return {'deferred': False}
    # Legal team validation still uses the real pinned simulator.
    result = asyncio.run(learn(store, {'id': 'tiny', 'format': FMT, 'kind': 'learn', 'payload': {}},
                              LearningConfig(), Policy(), float('inf')))
    assert result['reason'] == 'accumulating compatible experience'
    assert len(store.episodes(FMT)) == 1
    assert Model(tmp_path, FMT).revision == model.revision
    store.close()


def test_frozen_inputs_keep_scout_support_and_research_identical(tmp_path):
    store = Store(tmp_path / 'live')
    with store.db:
        store.db.execute('INSERT INTO research_teams VALUES (?,?,?,?,?,?,?)',
                         ('team', FMT, 'https://example.com', '', '', now(), json.dumps([{'species': 'Pelipper'}])))
        store.db.execute('INSERT INTO scout_samples VALUES (?,?,?,?,?,?)',
                         ('sample', FMT, 'pelipper', 'protect', '[]', 'battle'))
        store.db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)',
                         ('usage', 'https://example.com', FMT, 'Showdown usage test', now(), now(),
                          json.dumps({'species': [{'species': 'Pelipper', 'usage': .3}]}), 'digest'))
    frozen_path = tmp_path / 'frozen'
    freeze_inputs(store, frozen_path, FMT)
    with store.db:
        store.db.execute('DELETE FROM research_teams')
        store.db.execute('DELETE FROM documents')
        store.db.execute('DELETE FROM scout_samples')
    frozen = Store(frozen_path)
    assert frozen.db.execute('SELECT COUNT(*) FROM scout_samples').fetchone()[0] == 1
    from ml.research import species_prior
    assert species_prior(frozen, FMT)['pelipper'] == 1.0
    assert frozen.db.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 1
    assert not species_prior(store, FMT)
    frozen.close()
    store.close()


def test_promotion_contention_defers_without_blocking_matchmaking(tmp_path):
    store = Store(tmp_path)
    incumbent = Model(tmp_path, FMT)
    candidate = Model(tmp_path / 'candidate', FMT)
    with store.writer():
        assert promote_ready(store, FMT)['reason'] == 'no staged candidate'
    assert stage(store, FMT, incumbent.revision, candidate.path, {'passed': True})
    with store.writer():
        assert promote_ready(store, FMT)['reason'] == 'learning writer busy; promotion deferred'
    assert promote_ready(store, FMT)['promoted']
    store.close()


def test_partial_evaluation_resumes_same_candidate_cases_without_retraining(tmp_path, monkeypatch):
    import hashlib
    import time
    from ml.worker import atomic_json, evaluate_saved
    store = Store(tmp_path)
    incumbent = Model(tmp_path, FMT)
    work = tmp_path / 'experiment'
    candidate = Model(work / 'candidate', FMT)
    episode = {'id': 'consumed', 'format': FMT, 'source': 'ladder', 'team': 'fixture',
               'revision': incumbent.revision, 'status': 'complete', 'steps': [], 'outcome': 1}
    store.save_episode(episode)
    atomic_json(work / 'training-state.json', {'training': {'trained': True, 'previous_revision': incumbent.revision,
                 'revision': candidate.revision, 'consumed': ['consumed']}, 'backend': 'cpu', 'collection': None,
                 'candidate_sha256': hashlib.sha256(candidate.path.read_bytes()).hexdigest()})
    cases = [{'seed': i, 'side': 'p1' if i % 2 else 'p2', 'opponent': 'heuristic'} for i in range(4)]
    atomic_json(work / 'evaluation-plan.json', {'incumbent': cases, 'candidate': cases,
                                              'games': 4, 'margin': .1, 'team_pool_size': 1})
    calls = []
    async def partial(tasks, policy, deadline):
        calls.append([task['seed'] for task in tasks])
        return {'games': [{'seed': task['seed'], 'learner_side': task['side'], 'opponent': task['opponent'],
                           'winner': 'LocalBrain', 'rejected_actions': 0} for task in tasks[:2]], 'deferred': True}
    monkeypatch.setattr('ml.worker.parallel_games', partial)
    monkeypatch.setattr('ml.model.Model.train', lambda *args, **kwargs: pytest.fail('must not retrain a saved candidate'))
    job = {'id': 'resume', 'format': FMT}
    config = LearningConfig()
    class Policy:
        gpu_duty_fraction = .5
    first = asyncio.run(evaluate_saved(store, job, config, Policy(), time.monotonic() + 30, work))
    assert first['deferred'] and first['evaluation_progress'] == {'incumbent': 2, 'candidate': 2}
    assert store.db.execute("SELECT trained FROM episodes WHERE id='consumed'").fetchone()[0] == 1
    second = asyncio.run(evaluate_saved(store, job, config, Policy(), time.monotonic() + 30, work))
    assert second['evaluation']['complete']
    assert second['evaluation']['clean']
    assert calls == [[0, 1, 2, 3], [0, 1, 2, 3], [2, 3], [2, 3]]
    assert not second['staged_for_between_game_promotion']
    assert Model(work / 'candidate', FMT).revision == candidate.revision
    assert Model(tmp_path, FMT).revision == incumbent.revision
    store.close()
