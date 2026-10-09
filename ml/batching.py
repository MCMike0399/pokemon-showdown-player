"""Bounded reuse of immutable rollout features for CPU/MPS candidate training."""
from __future__ import annotations

from collections import OrderedDict

import torch

from ml.features import ACTION_DIM, STATE_DIM


class RolloutBatcher:
    def __init__(self, samples, device='cpu', max_cached_mb=128):
        if not samples or max_cached_mb <= 0:
            raise ValueError('nonempty samples and positive cache budget required')
        self.samples = samples
        self.device = device
        self.lengths = [len(s['actions']) for s in samples]
        self.limit = int(max_cached_mb * 2**20)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.hits = self.misses = 0
        self.resident = None
        columns = max(self.lengths)
        estimated = len(samples) * (STATE_DIM * 4 + columns * (ACTION_DIM * 4 + 1) + 8)
        if estimated <= self.limit:
            # Common 256-step batches fit in a small resident allocation. The
            # slice in batch() preserves each minibatch's original mask width.
            states = torch.tensor([s['state'] for s in samples], dtype=torch.float32)
            actions = torch.zeros((len(samples), columns, ACTION_DIM))
            mask = torch.zeros((len(samples), columns), dtype=torch.bool)
            for i, sample in enumerate(samples):
                n = self.lengths[i]
                actions[i, :n] = torch.tensor(sample['actions'], dtype=torch.float32)
                mask[i, :n] = True
            indices = torch.tensor([s['index'] for s in samples])
            self.resident = tuple(t.to(device) for t in (states, actions, mask, indices))
            self.cache_bytes = sum(t.numel() * t.element_size() for t in self.resident)

    def batch(self, positions):
        positions = list(positions)
        if not positions:
            raise ValueError('minibatch must not be empty')
        columns = max(self.lengths[i] for i in positions)
        if self.resident is not None:
            indices = torch.tensor(positions, device=self.device)
            self.hits += len(positions)
            states, actions, mask, choices = self.resident
            # Slice the immutable source before gathering. A battle minibatch
            # need not copy the wide team-preview padding into a new tensor.
            return (states.index_select(0, indices),
                    actions[:, :columns].index_select(0, indices),
                    mask[:, :columns].index_select(0, indices),
                    choices.index_select(0, indices))
        # Large histories avoid a global padded GPU allocation. Cache packed
        # per-transition CPU tensors with eviction, then transfer one minibatch.
        encoded = []
        for i in positions:
            if i in self.cache:
                self.hits += 1
                state, actions = self.cache.pop(i)
                self.cache[i] = (state, actions)
            else:
                self.misses += 1
                sample = self.samples[i]
                state = torch.tensor(sample['state'], dtype=torch.float32)
                actions = torch.tensor(sample['actions'], dtype=torch.float32)
                size = (state.numel() + actions.numel()) * 4
                while self.cache and self.cache_bytes + size > self.limit:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= sum(t.numel() * t.element_size() for t in old)
                if size <= self.limit:
                    self.cache[i] = (state, actions)
                    self.cache_bytes += size
            encoded.append((state, actions))
        states = torch.stack([s for s, _ in encoded])
        actions = torch.nn.utils.rnn.pad_sequence([a for _, a in encoded], batch_first=True)
        lengths = torch.tensor([self.lengths[i] for i in positions])
        mask = torch.arange(columns)[None] < lengths[:, None]
        indices = torch.tensor([self.samples[i]['index'] for i in positions])
        return tuple(t.to(self.device) for t in (states, actions, mask, indices))

    def stats(self):
        return {'mode': 'resident' if self.resident is not None else 'bounded-cpu',
                'device': str(self.device), 'feature_bytes': self.cache_bytes,
                'limit_bytes': self.limit, 'hits': self.hits, 'misses': self.misses}
