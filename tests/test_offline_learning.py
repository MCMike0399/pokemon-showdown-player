import asyncio
import copy

import numpy as np
import pytest
import torch

from battle_state import legal_choices, observe
from ml.brain import Brain
from ml.features import ACTION_DIM, Features, STATE_DIM
from ml.model import Model
from ml.storage import Store
from ml.teams import rank_teams, team_id
from harness import Player, TeamStore


FMT = "gen9championsvgc2026regmc"


def context(room="local-test", side="p1"):
    req = {"rqid": 1, "side": {"id": side, "pokemon": [{"ident": side + ": Pelipper", "details": "Pelipper, L50", "condition": "100/100", "active": True}]},
           "active": [{"moves": [{"move": "Hurricane", "target": "normal", "pp": 10}, {"move": "Protect", "target": "self", "pp": 10}]}]}
    return {"room": room, "format": FMT, "request": req, "choices": legal_choices(req), "state": observe(req, [])}


@pytest.fixture
def brain(tmp_path):
    b = Brain(Store(tmp_path), Features())
    yield b
    b.store.close()


def test_record_terminal_reward_checkpoint_and_no_reuse(brain):
    ctx = context()
    result = brain.decide(ctx, explore=True, record=True)
    assert result["choice"] in ctx["choices"]
    assert not brain.finish(ctx["room"], {"ongoing": True}, "Player")["recorded"]
    finish = brain.finish(ctx["room"], {"winner": "Player"}, "Player")
    assert finish["outcome"] == 1
    model = brain.model(FMT)
    before = copy.deepcopy(model.net.state_dict())
    training = brain.train(FMT)
    assert training["trained"] and training["algorithm"] == "PPO"
    assert any(not torch.equal(before[k], model.net.state_dict()[k]) for k in before)
    loaded = Model(brain.store.root, FMT)
    assert loaded.revision == model.revision
    assert all(torch.equal(loaded.net.state_dict()[k], model.net.state_dict()[k]) for k in before)
    assert not brain.train(FMT)["trained"]
    assert not brain.finish(ctx["room"], {"winner": "Player"}, "Player")["recorded"]


def test_recommendations_and_deterministic_choices_not_ppo(brain):
    brain.decide(context())
    assert not brain.pending
    with pytest.raises(ValueError, match="sampled"):
        brain.decide(context(), record=True)


def test_demonstrations_increase_label_probability_and_are_not_ppo(brain):
    ctx = context()
    before = brain.decide(ctx)["top_choices"]
    prior = next(c["probability"] for c in before if c["choice"] == "move 2")
    for i in range(24):
        ctx = context("demo-" + str(i))
        brain.decide(ctx, demonstration="move 2")
        brain.finish(ctx["room"], {"tie": True}, "Teacher")
    assert not brain.train(FMT)["trained"]
    training = brain.train(FMT, epochs=15, imitation=True)
    assert training["trained"] and training["steps"] == 24
    after = brain.decide(context())["top_choices"]
    assert next(c["probability"] for c in after if c["choice"] == "move 2") > prior + 0.001


def test_retry_replaces_rejected_decision_and_training_blocks_active_game(brain):
    ctx = context()
    brain.decide(ctx, explore=True, record=True)
    brain.decide(ctx, explore=True, record=True)
    assert len(next(iter(brain.pending.values()))["steps"]) == 1
    with pytest.raises(ValueError, match="active"):
        brain.train(FMT)
    brain.reject(ctx["room"])
    assert not next(iter(brain.pending.values()))["steps"]


def test_masked_padding_has_zero_probability(brain):
    model = brain.model(FMT)
    samples = [{"state": np.zeros(STATE_DIM).tolist(), "actions": np.zeros((n, ACTION_DIM)).tolist(), "index": 0} for n in (1, 3)]
    states, actions, mask, _ = model.batch(samples)
    logits, _ = model.net(states, actions, mask)
    probabilities = logits.softmax(dim=-1)
    assert probabilities[0, 1:].sum() == 0
    assert probabilities[0, 0] == 1


def test_external_checkpoint_update_reloads_between_games(brain):
    other = Brain(Store(brain.store.root), Features())
    other.model(FMT)
    ctx = context()
    brain.decide(ctx, explore=True, record=True)
    brain.finish(ctx["room"], {"winner": "Player"}, "Player")
    assert brain.train(FMT)["trained"]
    assert not other.train(FMT)["trained"]
    assert other.model(FMT).revision == brain.model(FMT).revision
    other.store.close()


def test_team_version_and_source_isolation(brain, tmp_path):
    teams = TeamStore(tmp_path / "teams.json")
    sets = [{"species": "Pelipper", "moves": ["Protect"]}]
    teams.create("Rain", sets, FMT)
    for source, outcome in (("ladder", 1), ("local", -1)):
        episode = {"id": source, "format": FMT, "team": team_id(FMT, sets), "source": source,
                   "revision": "fixture", "outcome": outcome, "status": "complete", "steps": []}
        brain.store.save_episode(episode)
    assert rank_teams(brain.store, teams, FMT)[0]["posterior_win_rate"] > 0.5
    assert rank_teams(brain.store, teams, FMT, "local")[0]["posterior_win_rate"] < 0.5
    teams.update("Rain", sets=[{"species": "Pelipper", "moves": ["Hurricane"]}])
    assert rank_teams(brain.store, teams, FMT)[0]["games"] == 0


def test_live_submission_is_idempotent_and_rejection_not_rewarded(brain, tmp_path):
    from ml.live import LiveSession
    from ps_client import PSClient
    c = PSClient(username="configured", password="unused")
    c.logged_in, c.user = True, "Player"
    room = "battle-" + FMT + "-1"
    req = context(room)["request"]
    import json
    c.battles[room] = {"log": ["|request|" + json.dumps(req)]}
    p = Player(c, TeamStore(tmp_path / "live-teams.json"))
    sent = []
    async def choose(room, choice):
        sent.append(choice)
    p.choose = choose
    live = LiveSession(brain, p)
    asyncio.run(live.choose(room, FMT))
    repeat = asyncio.run(live.choose(room, FMT))
    assert repeat["already_submitted"] and len(sent) == 1
    c.battles[room]["log"].append("|error|[Invalid choice] bad target")
    c.battles[room]["log"].append("|win|Player")
    finish = live.finish(room)
    assert finish["experience"]["steps"] == 0
    assert not brain.train(FMT)["trained"]


def test_temperature_is_checkpointed_and_collecting_logprobs_match(brain):
    model = brain.model(FMT)
    model.policy_temperature = .5
    model.save()
    decision = brain.decide(context(), explore=True, record=True)
    step = next(iter(brain.pending.values()))['steps'][0]
    states, actions, mask, indices = model.batch([step])
    logits, _ = model.net(states, actions, mask)
    actual = torch.distributions.Categorical(logits=logits / .5).log_prob(indices)
    assert float(actual[0].detach()) == pytest.approx(step['logprob'])
    assert Model(brain.store.root, FMT).policy_temperature == .5
    brain.finish('local-test', {'winner': 'Player'}, 'Player')
    report = brain.train(FMT)
    assert report['trained'] and report['kl_history']


def test_ppo_stops_epochs_when_policy_drift_exceeds_budget(brain):
    for i in range(16):
        ctx = context('drift-' + str(i))
        brain.decide(ctx, explore=True, record=True)
        brain.finish(ctx['room'], {'winner': 'Player'}, 'Player')
    model = brain.model(FMT)
    for group in model.optimizer.param_groups:
        group['lr'] = 1.0
    result = model.train(brain.store.episodes(FMT), epochs=10, target_kl=.001)
    assert result['early_stopped']
    assert result['epochs_completed'] < 10
    assert result['kl_history'][-1] > .001


def test_lazy_model_and_scout_initialization_preserve_sampling_rng(brain):
    from ml.scout import Scout
    torch.manual_seed(101)
    before = torch.random.get_rng_state().clone()
    brain.model(FMT)
    assert torch.equal(torch.random.get_rng_state(), before)
    Scout(brain.store, FMT, Features())
    assert torch.equal(torch.random.get_rng_state(), before)


def test_loading_existing_checkpoint_does_not_contend_with_learner(brain):
    original = Model(brain.store.root, FMT)
    with brain.store.writer():
        assert brain.model(FMT).revision == original.revision
