import copy

import numpy as np
import pytest
import torch

from ml.batching import RolloutBatcher
from ml.features import ACTION_DIM, STATE_DIM
from ml.model import ActorCritic, Model


def samples():
    rng = np.random.default_rng(18)
    return [{'state': rng.normal(size=STATE_DIM).astype(np.float32).tolist(),
             'actions': rng.normal(size=(n, ACTION_DIM)).astype(np.float32).tolist(),
             'index': n - 1} for n in (1, 3, 9, 2)]


@pytest.mark.parametrize('budget', [128, .01, .0001])
def test_cache_preserves_variable_masks_indices_and_duplicate_selections(budget):
    rows = samples()
    cache = RolloutBatcher(rows, max_cached_mb=budget)
    for positions in ([2, 0, 2], [1, 3], [3, 0], [0, 1, 2, 3]):
        expected = Model.batch([rows[i] for i in positions])
        actual = cache.batch(positions)
        for a, b in zip(expected, actual):
            assert torch.equal(a, b)
        assert cache.cache_bytes <= cache.limit
    if budget == 128:
        assert cache.stats()['mode'] == 'resident'
    else:
        assert cache.stats()['mode'] == 'bounded-cpu'


@pytest.mark.parametrize('budget', [128, .01])
def test_cached_inputs_preserve_policy_likelihood_and_gradients(budget):
    rows = samples()
    first = ActorCritic()
    torch.nn.init.normal_(first.actor[-1].weight, std=.03)
    second = copy.deepcopy(first)
    positions = [2, 0, 3]
    results = []
    for net, batch in [(first, Model.batch([rows[i] for i in positions])),
                       (second, RolloutBatcher(rows, max_cached_mb=budget).batch(positions))]:
        states, actions, mask, indices = batch
        logits, value = net(states, actions, mask)
        likelihood = torch.distributions.Categorical(logits=logits).log_prob(indices)
        (likelihood.mean() + value.square().mean()).backward()
        results.append(likelihood)
    assert torch.equal(*results)
    for old, cached in zip(first.parameters(), second.parameters()):
        assert torch.equal(old.grad, cached.grad)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple MPS unavailable')
def test_resident_mps_cache_preserves_inputs_and_completed_optimizer_step():
    rows = samples()
    cached = RolloutBatcher(rows, device='mps')
    expected = Model.batch([rows[2], rows[0]])
    actual = cached.batch([2, 0])
    for old, new in zip(expected, actual):
        assert torch.equal(old, new.cpu())
    net = ActorCritic().to('mps')
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    states, actions, mask, indices = actual
    logits, values = net(states, actions, mask)
    loss = -torch.distributions.Categorical(logits=logits).log_prob(indices).mean() + values.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    assert torch.isfinite(loss).item()
