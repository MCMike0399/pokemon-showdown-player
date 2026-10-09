"""Action-conditioned actor-critic and PPO with variable legal joint actions."""
from __future__ import annotations

import uuid
import tempfile
import time
import os
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from battle_state import to_id
from ml.features import ACTION_DIM, SCHEMA, STATE_DIM, FEATURE_PROFILES

# Small CPU batches are faster without a large BLAS worker pool, particularly on
# Apple silicon. The small model also works on Linux without accelerator setup.
torch.set_num_threads(max(1, min(4, int(os.environ.get("PS_TORCH_THREADS", "1")))))


class ActorCritic(nn.Module):
    def __init__(self, architecture='concat-v1'):
        super().__init__()
        if architecture not in ('concat-v1', 'matchup-v1'):
            raise ValueError('unsupported actor architecture')
        self.architecture = architecture
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.Tanh())
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 96), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(224, 96), nn.Tanh(), nn.Linear(96, 1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        if architecture == 'matchup-v1':
            self.matchup_state = nn.Linear(128, 96, bias=False)
            self.matchup_score = nn.Linear(96, 1, bias=False)
            nn.init.zeros_(self.matchup_score.weight)

    def forward(self, states, actions, mask=None):
        s = self.state_encoder(states)
        a = self.action_encoder(actions)
        repeated = s[:, None, :].expand(-1, actions.shape[1], -1)
        logits = self.actor(torch.cat((repeated, a), dim=-1)).squeeze(-1) + actions[:, :, -1]
        if self.architecture == 'matchup-v1':
            logits = logits + self.matchup_score(a * self.matchup_state(s)[:, None, :]).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.critic(s).squeeze(-1)


class Model:
    def __init__(self, root: Path, fmt: str, seed: int = 0):
        if not to_id(fmt) or fmt != to_id(fmt):
            raise ValueError("format must be a Showdown format id")
        self.path = Path(root) / "models" / (fmt + ".pt")
        self.fmt = fmt
        checkpoint = torch.load(self.path, map_location='cpu', weights_only=True) if self.path.exists() else None
        self.architecture = checkpoint.get('architecture', 'concat-v1') if checkpoint else 'concat-v1'
        self.preview_training_weight = float(checkpoint.get('preview_training_weight', 1)) if checkpoint else 1.0
        if not 1 <= self.preview_training_weight <= 16:
            raise ValueError('preview training weight must be 1..16')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net = ActorCritic(self.architecture)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.revision = uuid.uuid4().hex
        self.updates = 0
        self.policy_temperature = 1.0
        self.preview_temperature = None
        self.feature_profile = 'legacy'
        self.strategy_knowledge = checkpoint.get('strategy_knowledge', {}) if checkpoint else {}
        if not isinstance(self.strategy_knowledge, dict):
            raise ValueError('strategy knowledge must be a checkpointed dictionary')
        if self.path.exists():
            if checkpoint["schema"] != SCHEMA or checkpoint["format"] != fmt:
                raise ValueError("checkpoint schema/format mismatch")
            self.net.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.revision = checkpoint["revision"]
            self.updates = checkpoint["updates"]
            self.policy_temperature = float(checkpoint.get('policy_temperature', 1.0))
            if not .25 <= self.policy_temperature <= 2:
                raise ValueError('checkpoint policy temperature must be .25..2')
            self.preview_temperature = checkpoint.get('preview_temperature')
            if self.preview_temperature is not None and not .25 <= self.preview_temperature <= 2:
                raise ValueError('checkpoint preview temperature must be .25..2')
            self.feature_profile = checkpoint.get('feature_profile', 'legacy')
            if self.feature_profile not in FEATURE_PROFILES:
                raise ValueError('unsupported checkpoint feature profile')
        else:
            self.save()
        self.loaded_mtime = self.path.stat().st_mtime_ns
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
        try:
            torch.save({"schema": SCHEMA, "format": self.fmt, "revision": self.revision,
                        'architecture': self.architecture, 'preview_training_weight': self.preview_training_weight,
                        "updates": self.updates, "model": self.net.state_dict(),
                        "policy_temperature": self.policy_temperature,
                        "preview_temperature": self.preview_temperature,
                        "feature_profile": self.feature_profile,
                        "strategy_knowledge": self.strategy_knowledge,
                        "optimizer": self.optimizer.state_dict()}, tmp)
            tmp.replace(self.path)
            self.loaded_mtime = self.path.stat().st_mtime_ns
        finally:
            tmp.unlink(missing_ok=True)
        self.checkpoint_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def enable_matchups(self):
        """Explicit candidate migration; zero residual preserves existing logits.

        An additional state/action product provides a direct learning signal for
        matchup-conditioned preferences. Old artifacts remain concat-v1.
        """
        if self.architecture == 'matchup-v1':
            return
        device = next(self.net.parameters()).device
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            net = ActorCritic('matchup-v1').to(device)
        missing, unexpected = net.load_state_dict(self.net.state_dict(), strict=False)
        if set(missing) != {'matchup_state.weight', 'matchup_score.weight'} or unexpected:
            raise ValueError('unsupported matchup migration')
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
        self.architecture = 'matchup-v1'
        self.revision = uuid.uuid4().hex
        self.save()

    def set_device(self, device: str = "cpu"):
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable on this host")
        self.net.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        return self

    def temperature(self, preview: bool = False) -> float:
        return self.preview_temperature if preview and self.preview_temperature is not None else self.policy_temperature

    def training_logits(self, logits, samples):
        temperatures = torch.tensor([self.temperature(s.get('choice', '').startswith('team ')) for s in samples],
                                    device=logits.device, dtype=logits.dtype)
        return logits / temperatures[:, None]

    def predict(self, state, actions, explore: bool = False, preview: bool = False, selected_index: int | None = None):
        device = next(self.net.parameters()).device
        with torch.no_grad():
            logits, value = self.net(torch.as_tensor(state, device=device)[None], torch.as_tensor(actions, device=device)[None])
            logits = logits / self.temperature(preview)
            dist = Categorical(logits=logits[0])
            index = selected_index if selected_index is not None else int(dist.sample()) if explore else int(logits[0].argmax())
            return {"index": index, "logprob": float(dist.log_prob(torch.tensor(index, device=device))),
                    "value": float(value[0]), "probabilities": dist.probs.tolist()}

    @staticmethod
    def batch(samples):
        size = max(len(s["actions"]) for s in samples)
        actions = torch.zeros((len(samples), size, ACTION_DIM))
        mask = torch.zeros((len(samples), size), dtype=torch.bool)
        for i, sample in enumerate(samples):
            n = len(sample["actions"])
            actions[i, :n] = torch.tensor(sample["actions"], dtype=torch.float32)
            mask[i, :n] = True
        states = torch.tensor([s["state"] for s in samples], dtype=torch.float32)
        indices = torch.tensor([s["index"] for s in samples])
        return states, actions, mask, indices

    def train(self, episodes: list[dict], epochs: int = 4, imitation: bool = False,
              duty_fraction: float = 1.0, deadline: float | None = None,
              target_kl: float = 0.03, checkpoint=None, *,
              learning_rate: float | None = None, entropy_coef: float = 0.01,
              minibatch_size: int = 32, shuffle_seed: int = 0) -> dict:
        if not 1 <= epochs <= 30:
            raise ValueError("epochs must be between 1 and 30")
        if not 0 < target_kl <= 1:
            raise ValueError("target KL must be positive and at most 1")
        if learning_rate is not None and not 0 < learning_rate <= 1:
            raise ValueError('learning rate must be positive and at most 1')
        if not np.isfinite(entropy_coef) or not 0 <= entropy_coef <= 1:
            raise ValueError('entropy coefficient must be finite and between 0 and 1')
        if not isinstance(minibatch_size, int) or not 1 <= minibatch_size <= 4096:
            raise ValueError('minibatch size must be an integer between 1 and 4096')
        training_started = time.perf_counter()
        samples = []
        if checkpoint:
            checkpoint()
        ids = []
        excluded = []
        from ml.data_quality import trajectory_issues
        for episode in episodes:
            if episode["format"] != self.fmt or episode.get("status") != "complete":
                continue
            if not imitation and (episode["revision"] != self.revision or not episode.get("on_policy")):
                continue
            if episode.get('feature_profile', 'legacy') != self.feature_profile:
                continue
            if imitation and not episode.get("demonstration"):
                continue
            trajectory = episode.get("steps", [])
            if not trajectory or episode.get("schema") != SCHEMA:
                continue
            issues = trajectory_issues(episode)
            if issues:
                excluded.append({'id': episode['id'], 'issues': issues})
                continue
            ids.append(episode["id"])
            advantage = 0.0
            next_value = 0.0
            prepared = []
            for i in reversed(range(len(trajectory))):
                step = trajectory[i]
                reward = float(episode["outcome"] or 0) if i == len(trajectory) - 1 else 0.0
                delta = reward + 0.99 * next_value - step.get("value", 0.0)
                advantage = delta + 0.99 * 0.95 * advantage
                prepared.append({**step, "advantage": advantage, "return": advantage + step.get("value", 0.0),
                                 '_episode_id': episode['id'], '_source': episode.get('source', 'unknown')})
                next_value = step.get("value", 0.0)
            samples.extend(reversed(prepared))
        if not samples:
            return {"trained": False, "reason": "no compatible unconsumed demonstrations" if imitation else "no complete stochastic rollouts from the current model revision",
                    'excluded_episodes': excluded}
        maximum_logprob_error = 0.0
        from ml.batching import RolloutBatcher
        device = next(self.net.parameters()).device
        batcher = RolloutBatcher(samples, device)
        if not imitation:
            invalid = set()
            with torch.no_grad():
                for start in range(0, len(samples), 32):
                    if checkpoint:
                        checkpoint()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('training budget expired during rollout validation')
                    batch = samples[start:start + 32]
                    states, actions, mask, indices = batcher.batch(range(start, start + len(batch)))
                    logits, _ = self.net(states, actions, mask)
                    likelihood = Categorical(logits=self.training_logits(logits, batch)).log_prob(indices)
                    errors = (likelihood - torch.tensor([s['logprob'] for s in batch], device=device)).abs().tolist()
                    maximum_logprob_error = max(maximum_logprob_error, max(errors))
                    invalid.update(s['_episode_id'] for s, error in zip(batch, errors) if error > 1e-4)
            if invalid:
                excluded.extend({'id': key, 'issues': ['collecting-checkpoint-likelihood-mismatch']} for key in sorted(invalid))
                ids = [key for key in ids if key not in invalid]
                samples = [s for s in samples if s['_episode_id'] not in invalid]
            if not samples:
                return {'trained': False, 'reason': 'no rollouts match the collecting checkpoint likelihood',
                        'excluded_episodes': excluded, 'maximum_collecting_logprob_error': maximum_logprob_error}
            if invalid:
                del batcher
                batcher = RolloutBatcher(samples, device)
        advantages = np.array([s["advantage"] for s in samples], dtype=np.float32)
        # One tiny batch must not lose all its policy gradient when variance is 0.
        if len(samples) > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for sample, advantage in zip(samples, advantages):
            sample["advantage"] = float(advantage)
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = learning_rate
        rng = np.random.default_rng(shuffle_seed)
        losses = []
        entropies = []
        kl_history = []
        clip_fractions = []
        self.net.train()
        device = next(self.net.parameters()).device
        for _ in range(epochs):
            order = rng.permutation(len(samples))
            for start in range(0, len(samples), minibatch_size):
                if checkpoint:
                    checkpoint()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("training budget expired before saving candidate")
                started = time.monotonic()
                batch = [samples[i] for i in order[start:start + minibatch_size]]
                states, actions, mask, indices = batcher.batch(order[start:start + minibatch_size])
                logits, values = self.net(states, actions, mask)
                logits = self.training_logits(logits, batch)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(indices)
                phase_weights = torch.tensor([self.preview_training_weight if s.get('choice', '').startswith('team ') else 1
                                              for s in batch], device=device, dtype=logits.dtype)
                phase_weights = phase_weights / phase_weights.mean()
                if imitation:
                    loss = -(logprobs * phase_weights).mean()
                else:
                    old = torch.tensor([s["logprob"] for s in batch], device=device)
                    adv = torch.tensor([s["advantage"] for s in batch], device=device)
                    returns = torch.tensor([s["return"] for s in batch], device=device)
                    ratio = (logprobs - old).exp()
                    policy = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
                    loss = ((policy + 0.5 * (values - returns).square() - entropy_coef * dist.entropy()) * phase_weights).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss; checkpoint has not been saved")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                # Keep diagnostics on device until reporting; scalar reads
                # otherwise synchronize Metal after every optimizer step.
                losses.append(loss.detach())
                entropies.append(dist.entropy().detach().mean())
                if device.type == "mps" and duty_fraction < 1:
                    torch.mps.synchronize()
                    time.sleep((time.monotonic() - started) * (1 / duty_fraction - 1))
            if not imitation:
                # Assess the whole variable-action batch, rather than letting a
                # noisy minibatch drive the stopping decision.
                divergences, clipped = [], []
                with torch.no_grad():
                    for start in range(0, len(samples), 32):
                        if checkpoint:
                            checkpoint()
                        batch = samples[start:start + 32]
                        states, actions, mask, indices = batcher.batch(range(start, start + len(batch)))
                        logits, _ = self.net(states, actions, mask)
                        logits = self.training_logits(logits, batch)
                        old = torch.tensor([s['logprob'] for s in batch], device=device)
                        change = Categorical(logits=logits).log_prob(indices) - old
                        ratio = change.exp()
                        divergences.append(((ratio - 1) - change).detach())
                        clipped.append(((ratio - 1).abs() > .2).float())
                # Preserve the old per-sample order and Python double-precision
                # summation, including the exact whole-batch KL stop rule.
                divergences = torch.cat(divergences).cpu().tolist()
                clipped = torch.cat(clipped).cpu().tolist()
                kl_history.append(sum(divergences) / len(divergences))
                clip_fractions.append(sum(clipped) / len(clipped))
                if kl_history[-1] > target_kl:
                    break
        old_revision = self.revision
        if checkpoint:
            checkpoint()
        diagnostics = torch.stack((torch.stack(losses), torch.stack(entropies)), dim=1).cpu().tolist()
        losses = [row[0] for row in diagnostics]
        entropies = [row[1] for row in diagnostics]
        self.revision = uuid.uuid4().hex
        self.updates += 1
        self.save()
        return {"trained": True, "algorithm": "behavior-cloning" if imitation else "PPO", "episodes": len(ids),
                'architecture': self.architecture, 'preview_training_weight': self.preview_training_weight,
                "steps": len(samples), "loss": sum(losses) / len(losses),
                "epochs_completed": len(kl_history) if not imitation else epochs,
                "target_kl": target_kl, "kl_history": kl_history,
                "clip_fraction_history": clip_fractions,
                "early_stopped": not imitation and len(kl_history) < epochs,
                "previous_revision": old_revision, "revision": self.revision, "consumed": ids,
                'excluded_episodes': excluded, 'maximum_collecting_logprob_error': maximum_logprob_error,
                'batch_cache': batcher.stats(),
                'optimizer_steps': len(losses), 'training_seconds': round(time.perf_counter() - training_started, 3),
                'learning_rate': self.optimizer.param_groups[0]['lr'], 'entropy_coef': entropy_coef,
                'minibatch_size': minibatch_size, 'mean_entropy': sum(entropies) / len(entropies),
                'device': device.type, 'duty_fraction': duty_fraction,
                'source_steps': {source: sum(s['_source'] == source for s in samples)
                                 for source in sorted({s['_source'] for s in samples})}}
