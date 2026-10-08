import copy
import hashlib
import json
import math

import numpy as np
import pytest

from ml.brain import Brain
from ml.data_quality import trajectory_issues
from ml.features import ACTION_DIM, STATE_DIM, Features
from ml.recording import encoded_input
from ml.storage import Store
from test_offline_learning import context, FMT


def recorded(tmp_path):
    store = Store(tmp_path)
    brain = Brain(store, Features())
    brain.decide(context(), explore=True, record=True)
    brain.finish('local-test', {'winner': 'Player'}, 'Player')
    return store, brain, store.episodes(FMT)[0]


def test_quality_accepts_captured_and_legacy_inputs_and_retains_provenance(tmp_path):
    store, brain, episode = recorded(tmp_path)
    assert not trajectory_issues(episode)
    assert episode['recorder_version'] == 3
    model = brain.model(FMT)
    assert episode['collecting_policy']['checkpoint_sha256'] == hashlib.sha256(model.path.read_bytes()).hexdigest()
    archive = store.root / episode['collecting_policy']['checkpoint_archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == model.checkpoint_sha256
    assert episode['collecting_policy']['dex_sha256'] == Features().signature()
    episode['steps'][0].pop('snapshot')
    assert not trajectory_issues(episode)
    store.close()


def test_collected_checkpoint_survives_an_active_checkpoint_replacement(tmp_path):
    store, brain, episode = recorded(tmp_path)
    archive = store.root / episode['collecting_policy']['checkpoint_archive']
    original = archive.read_bytes()
    model = brain.model(FMT)
    model.preview_temperature = .5
    model.revision = 'new-policy'
    model.save()
    assert model.path.read_bytes() != original
    assert archive.read_bytes() == original
    assert not trajectory_issues(episode)
    store.close()


@pytest.mark.parametrize('defect,expected', [
    ('unsent', 'unsubmitted-proposal'), ('index', 'invalid-feature-shape-or-index'),
    ('nonfinite', 'nonfinite-features'), ('mask', 'request-mask-mismatch'),
    ('probability', 'snapshot-likelihood-mismatch'), ('prefix', 'invalid-snapshot-or-log-prefix'),
    ('future', 'future-or-out-of-order-observation'), ('duplicate', 'duplicate-request-credit')])
def test_quality_rejects_broken_training_inputs(tmp_path, defect, expected):
    store, brain, episode = recorded(tmp_path)
    step = episode['steps'][0]
    if defect == 'unsent': step['submitted'] = False
    elif defect == 'index': step['index'] = len(step['actions'])
    elif defect == 'nonfinite': step['state'][0] = float('nan')
    elif defect == 'mask': step['snapshot']['request']['active'][0]['moves'][0]['disabled'] = True
    elif defect == 'probability': step['snapshot']['legal_choices'][0]['probability'] = .9
    elif defect == 'prefix': step['snapshot']['public_log']['sha256'] = 'changed'
    elif defect == 'future': step['snapshot']['turn'] = -2
    elif defect == 'duplicate': episode['steps'].append(copy.deepcopy(step))
    assert expected in trajectory_issues(episode)
    before = hashlib.sha256(brain.model(FMT).path.read_bytes()).hexdigest()
    result = brain.model(FMT).train([episode])
    assert not result['trained'] and result['excluded_episodes']
    assert hashlib.sha256(brain.model(FMT).path.read_bytes()).hexdigest() == before
    store.close()


def test_ppo_rejects_forged_collecting_probability_without_consumption(tmp_path):
    store, brain, episode = recorded(tmp_path)
    episode['steps'][0].pop('snapshot')
    episode['steps'][0]['logprob'] -= .1
    result = brain.model(FMT).train([episode])
    assert not result['trained']
    assert result['excluded_episodes'][0]['issues'] == ['collecting-checkpoint-likelihood-mismatch']
    assert result['maximum_collecting_logprob_error'] > .09
    assert len(store.episodes(FMT)) == 1
    store.close()


def test_demonstration_records_likelihood_of_demonstrated_command(tmp_path):
    store = Store(tmp_path)
    brain = Brain(store, Features())
    ctx = context()
    demonstration = ctx['choices'][-1]  # Low-pressure Protect, not the argmax attack.
    decision = brain.decide(ctx, demonstration=demonstration)
    brain.finish(ctx['room'], {'winner': 'Player'}, 'Player')
    episode = store.episodes(FMT)[0]
    step = episode['steps'][0]
    assert step['choice'] == demonstration
    assert math.exp(step['logprob']) == pytest.approx(decision['probability'])
    assert not trajectory_issues(episode)
    result = brain.train(FMT, imitation=True)
    assert result['trained'] and result['algorithm'] == 'behavior-cloning'
    assert result['source_steps'] == {episode['source']: 1}
    store.close()


def test_dense_json_compaction_is_lossless_and_stays_readable():
    rng = np.random.default_rng(27)
    actions = np.zeros((360, ACTION_DIM), np.float32)
    actions[:, :9] = rng.normal(size=(360, 9))
    actions[:, -1] = rng.normal(size=360)
    state = np.zeros(STATE_DIM, np.float32)
    state[:9] = rng.normal(size=9)
    for original in (actions, state):
        encoded = json.dumps(encoded_input(original))
        assert np.array_equal(np.asarray(json.loads(encoded), np.float32), original)
        assert len(encoded) < len(json.dumps(original.tolist())) * .75


def test_full_simulator_seeds_do_not_repeat_every_65536_games():
    from ml.simulator import simulator_seed
    assert simulator_seed(1) == simulator_seed(65537)  # Historical bridge limitation.
    assert simulator_seed(1, 'full-v1') != simulator_seed(65537, 'full-v1')
    assert simulator_seed(65537, 'full-v1') == [1, 6, 13, 19]
    assert len({tuple(simulator_seed(i << 16, 'full-v1')) for i in range(100)}) == 100
    with pytest.raises(ValueError):
        simulator_seed(-1, 'full-v1')


def test_collecting_temperatures_cannot_change_mid_episode(tmp_path):
    store = Store(tmp_path)
    brain = Brain(store, Features())
    brain.decide(context(), explore=True, record=True)
    brain.model(FMT).policy_temperature = .5
    with pytest.raises(ValueError, match='temperatures'):
        brain.decide(context(), explore=True, record=True)
    store.close()


def test_canonical_argument_free_tie_closes_and_remains_usable_training_data(tmp_path):
    from types import SimpleNamespace
    from harness import Player
    from ml.recording import battle_lines, log_prefix
    from ml.scout import public_lines, ingest_public
    terminal = ['|start', '|turn|1', '|tie']
    assert battle_lines(terminal) == public_lines(terminal) == terminal
    player = Player(SimpleNamespace(battles={'room': {'log': terminal}}))
    assert player.finished('room') and player.result('room')['tie']
    store = Store(tmp_path)
    brain = Brain(store, Features())
    brain.decide(context(), explore=True, record=True)
    brain.finish('local-test', player.result('room'), 'Player', public_log=terminal)
    episode = store.episodes(FMT)[0]
    assert episode['outcome'] == 0
    assert log_prefix(episode, episode['terminal_log']) == terminal
    assert not trajectory_issues(episode)
    assert ingest_public(store, FMT, terminal, 'test', 'tie')['imported']
    store.close()


def test_scout_identity_conflict_does_not_append_unlinked_labels(tmp_path):
    from ml.scout import ingest_public
    store = Store(tmp_path)
    f = Features({'moves': {'hurricane': {'basePower': 110}}})
    prefix = ['|switch|p1a: One|Pelipper|100/100', '|switch|p2a: Two|Pelipper|100/100',
              '|turn|1', '|move|p1a: One|Hurricane|p2a: Two']
    assert ingest_public(store, FMT, prefix + ['|win|One'], 'test', 'same-battle', f)['imported']
    count = store.db.execute('SELECT COUNT(*) FROM scout_samples').fetchone()[0]
    conflict = ingest_public(store, FMT, prefix + ['|turn|2', '|move|p2a: Two|Hurricane|p1a: One', '|win|One'], 'test', 'same-battle', f)
    assert not conflict['imported'] and conflict['prefix_conflict']
    assert store.db.execute('SELECT COUNT(*) FROM scout_samples').fetchone()[0] == count
    store.close()


def test_scout_training_preserves_but_excludes_orphaned_labels(tmp_path):
    from ml.scout import Scout
    store = Store(tmp_path)
    with store.db:
        for i in range(10):
            store.db.execute('INSERT INTO scout_samples VALUES (?,?,?,?,?,?)',
                             (str(i), FMT, 'pelipper', 'hurricane', json.dumps([0] * STATE_DIM), 'unlinked'))
    scout = Scout(store, FMT, Features({'moves': {'hurricane': {}}}))
    result = scout.train()
    assert not result['trained'] and result['orphaned_samples_excluded'] == 10
    assert store.db.execute('SELECT COUNT(*) FROM scout_samples').fetchone()[0] == 10
    store.close()


def test_curriculum_separates_team_and_policy_and_paired_sheet_visibility(tmp_path):
    from ml.worker import game_tasks
    teams = [[{'species': 'Team' + str(i)}] for i in range(8)]
    first = game_tasks(tmp_path, tmp_path / 'before', FMT, teams, 128, False, 123, tmp_path / 'fixed', curriculum='ladder-v1')
    second = game_tasks(tmp_path, tmp_path / 'after', FMT, teams, 128, False, 123, tmp_path / 'fixed', curriculum='ladder-v1')
    for team in teams:
        assert {t['opponent'] for t in first if t['team2'] == team} == {'random', 'self', 'heuristic', 'tactical'}
    assert 0 < sum(t['open_team_sheets'] for t in first) < len(first) // 4
    for a, b in zip(first, second):
        assert {k: v for k, v in a.items() if k != 'checkpoint_root'} == {k: v for k, v in b.items() if k != 'checkpoint_root'}


@pytest.mark.parametrize('field', ['learner_team', 'opponent_team', 'open_team_sheets'])
def test_gate_refuses_mismatched_pair_conditions(field):
    from ml.promotion import paired_gate
    from test_offline_promotion import games
    base = games([False] * 20)
    candidate = games([True] * 20)
    for game in base + candidate:
        game.update(learner_team='ours', opponent_team='theirs', open_team_sheets=False)
    assert paired_gate(base, candidate, 20, .1)['passed']
    candidate[0][field] = True if field == 'open_team_sheets' else 'different'
    assert not paired_gate(base, candidate, 20, .1)['passed']
