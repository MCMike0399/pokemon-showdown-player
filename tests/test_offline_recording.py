import asyncio
import copy
import json

import pytest

from ml.brain import Brain
from ml.features import Features
from ml.recording import digest, log_prefix, named_choice, record_log
from ml.storage import Store
from test_offline_learning import context, FMT


def test_capture_exact_inputs_names_and_distribution_without_future_leak(tmp_path):
    store = Store(tmp_path)
    brain = Brain(store, Features())
    ctx = context()
    ctx['public_log'] = ['|player|p1|PrivateName', '|turn|1', '|c|Other|excluded chat',
                         '|request|excluded request', '|challstr|excluded auth']
    original_request, original_state = copy.deepcopy(ctx['request']), copy.deepcopy(ctx['state'])
    result = brain.decide(ctx, explore=True, record=True)
    episode = next(iter(brain.pending.values()))
    step = episode['steps'][0]
    audit = step['snapshot']
    assert audit['request'] == original_request and audit['observation'] == original_state
    assert audit['provenance'] == 'captured-at-decision'
    assert [a['choice'] for a in audit['legal_choices']] == ctx['choices']
    assert audit['legal_choices'][audit['selected_index']]['choice'] == result['choice']
    assert sum(a['probability'] for a in audit['legal_choices']) == pytest.approx(1)
    assert {a['components'][0]['move'] for a in audit['legal_choices']} == {'Hurricane', 'Protect'}
    prefix = log_prefix(episode, audit['public_log'])
    assert prefix == ['|player|p1|Player-p1', '|turn|1']
    ctx['request']['rqid'] = 2
    ctx['request']['active'][0]['moves'][0]['pp'] = 2
    ctx['state']['weather'] = 'RainDance'
    ctx['public_log'].extend(['|move|p2a: Two|Protect|p2a: Two', '|turn|2'])
    brain.decide(ctx, explore=True, record=True)
    assert audit['request'] == original_request and audit['observation'] == original_state
    assert log_prefix(episode, audit['public_log']) == prefix
    assert len(episode['observation_logs']) == 1
    store.close()


def test_log_segments_handle_reconnect_prefix_changes_and_detect_corruption():
    episode = {}
    first = record_log(episode, ['|turn|1'])
    second = record_log(episode, ['|turn|1', '|turn|2'])
    other = record_log(episode, ['|start|', '|turn|1'])
    assert len(episode['observation_logs']) == 2
    assert log_prefix(episode, first) == ['|turn|1']
    assert log_prefix(episode, second) == ['|turn|1', '|turn|2']
    assert log_prefix(episode, other) == ['|start|', '|turn|1']
    episode['observation_logs'][0][0] = '|turn|99'
    with pytest.raises(ValueError, match='digest'):
        log_prefix(episode, first)


def test_named_preview_switch_and_unknown_target_are_not_guessed():
    ctx = context()
    ctx['request']['side']['pokemon'].append({'details': 'Swampert, L50', 'condition': '100/100'})
    assert named_choice(ctx, 'team 2,1')['pokemon'] == ['Swampert', 'Pelipper']
    assert named_choice(ctx, 'switch 2')['components'][0]['pokemon'] == 'Swampert'
    move = named_choice(ctx, 'move 1 2 mega')['components'][0]
    assert move['move'] == 'Hurricane' and move['modifiers'] == ['mega']
    assert move['target_pokemon'] is None and move['target_side'] == 'opponent'


def test_rejection_archives_proposal_without_training_credit(tmp_path):
    store = Store(tmp_path)
    brain = Brain(store, Features())
    brain.decide(context(), explore=True, record=True)
    episode = next(iter(brain.pending.values()))
    proposal = copy.deepcopy(episode['steps'][0])
    brain.reject('local-test', reason='socket-submission-failed')
    assert episode['steps'] == []
    assert episode['discarded_proposals'][0]['step'] == proposal
    assert episode['discarded_proposals'][0]['reason'] == 'socket-submission-failed'
    brain.finish('local-test', {'winner': 'Player'}, 'Player')
    assert not brain.train(FMT)['trained']
    store.close()


def test_restart_retains_original_snapshot_and_terminal_log(tmp_path):
    stores = [Store(tmp_path), Store(tmp_path)]
    original = Brain(stores[0], Features())
    ctx = context()
    ctx['public_log'] = ['|turn|1']
    original.decide(ctx, explore=True, record=True)
    first = copy.deepcopy(next(iter(original.pending.values()))['steps'][0])
    restarted = Brain(stores[1], Features())
    restored = restarted.resume('local-test', 'p1')
    assert restored['steps'][0] == first
    end = ['|turn|1', '|move|p1a: Pelipper|Protect|p1a: Pelipper', '|win|Player']
    restarted.finish('local-test', {'winner': 'Player'}, 'Player', public_log=end)
    completed = stores[1].episodes(FMT)[0]
    assert log_prefix(completed, completed['steps'][0]['snapshot']['public_log']) == ['|turn|1']
    assert log_prefix(completed, completed['terminal_log'])[-1] == '|win|redacted'
    assert completed['steps'][0] == first
    for store in stores:
        store.close()


def test_preview_calibration_preserves_move_policy_and_training_likelihoods(tmp_path):
    import numpy as np
    import torch
    from ml.model import Model
    from ml.features import ACTION_DIM, STATE_DIM
    model = Model(tmp_path, FMT)
    state = np.zeros(STATE_DIM, np.float32)
    actions = np.zeros((3, ACTION_DIM), np.float32)
    actions[:, -1] = [.1, .2, .3]
    before = model.predict(state, actions)['probabilities']
    model.preview_temperature = .25
    model.save()
    assert model.predict(state, actions)['probabilities'] == before
    preview = model.predict(state, actions, preview=True)
    assert preview['probabilities'][-1] > before[-1]
    assert Model(tmp_path, FMT).preview_temperature == .25
    samples = [{'choice': c, 'state': state.tolist(), 'actions': actions.tolist(), 'index': 2}
               for c in ('team 1,2,3', 'move 1')]
    states, encoded, mask, indices = model.batch(samples)
    logits, _ = model.net(states, encoded, mask)
    distribution = torch.distributions.Categorical(logits=model.training_logits(logits, samples))
    assert float(distribution.probs[0, 2].detach()) == pytest.approx(preview['probabilities'][2])
    assert distribution.probs[1].detach().tolist() == pytest.approx(before)


def test_preview_snapshot_probability_matches_collecting_distribution(tmp_path):
    from battle_state import legal_choices, observe
    import torch
    store = Store(tmp_path)
    brain = Brain(store, Features())
    ctx = context()
    ctx['request']['side']['pokemon'].append({'details': 'Swampert, L50', 'condition': '100/100'})
    ctx['request'].update(teamPreview=True, maxChosenTeamSize=2)
    ctx['choices'] = legal_choices(ctx['request'])
    ctx['state'] = observe(ctx['request'], [])
    model = brain.model(FMT)
    model.preview_temperature = .25
    brain.decide(ctx, explore=True, record=True)
    step = next(iter(brain.pending.values()))['steps'][0]
    states, actions, mask, indices = model.batch([step])
    logits, _ = model.net(states, actions, mask)
    distribution = torch.distributions.Categorical(logits=model.training_logits(logits, [step]))
    assert float(distribution.log_prob(indices)[0].detach()) == pytest.approx(step['logprob'])
    assert step['snapshot']['sampling_temperature'] == .25
    store.close()


def test_readable_inspection_is_bounded_private_and_marks_legacy_gaps(tmp_path):
    from ml.recording import read_recording
    store = Store(tmp_path)
    brain = Brain(store, Features())
    ctx = context()
    ctx['public_log'] = ['|turn|1']
    brain.decide(ctx, explore=True, record=True)
    brain.finish('local-test', {'winner': 'Player'}, 'Player', public_log=['|turn|1', '|turn|2', '|win|Player'])
    before = store.db.execute('SELECT data,trained FROM episodes').fetchone()
    view = read_recording(store, 'local-test', limit=1)
    audit = view['steps'][0]['snapshot']
    assert audit['public_log'] == ['|turn|1']
    assert 'legal_choices' not in audit and audit['legal_choices_count'] == len(ctx['choices'])
    assert audit['selected']['choice'] == view['steps'][0]['choice']
    assert len(read_recording(store, 'local-test', alternatives=True)['steps'][0]['snapshot']['legal_choices']) == len(ctx['choices'])
    after = store.db.execute('SELECT data,trained FROM episodes').fetchone()
    assert tuple(before) == tuple(after)
    legacy = json.loads(before[0])
    legacy['steps'][0].pop('snapshot')
    with store.db:
        store.db.execute('UPDATE episodes SET data=?', (json.dumps(legacy),))
    view = read_recording(store, 'local-test')
    assert view['steps'][0]['snapshot'] is None
    assert view['steps'][0]['provenance'] == 'legacy-encoded-only'
    assert not read_recording(store, 'unknown')['found']
    with pytest.raises(ValueError):
        read_recording(store, 'local-test', limit=100)
    store.close()
