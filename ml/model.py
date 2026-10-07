"""Action-conditioned actor-critic and PPO with variable legal joint actions."""
from __future__ import annotations

import uuid
import tempfile
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from battle_state import to_id
from ml.features import ACTION_DIM, SCHEMA, STATE_DIM

# Small CPU batches are faster without a large BLAS worker pool, particularly on
# Apple silicon. The small model also works on Linux without accelerator setup.
torch.set_num_threads(max(1, min(4, int(os.environ.get("PS_TORCH_THREADS", "1")))))


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.Tanh())
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 96), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(224, 96), nn.Tanh(), nn.Linear(96, 1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, states, actions, mask=None):
        s = self.state_encoder(states)
        a = self.action_encoder(actions)
        repeated = s[:, None, :].expand(-1, actions.shape[1], -1)
        logits = self.actor(torch.cat((repeated, a), dim=-1)).squeeze(-1) + actions[:, :, -1]
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.critic(s).squeeze(-1)


class Model:
    def __init__(self, root: Path, fmt: str, seed: int = 0):
        if not to_id(fmt) or fmt != to_id(fmt):
            raise ValueError("format must be a Showdown format id")
        self.path = Path(root) / "models" / (fmt + ".pt")
        self.fmt = fmt
        torch.manual_seed(seed)
        self.net = ActorCritic()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.revision = uuid.uuid4().hex
        self.updates = 0
        if self.path.exists():
            checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
            if checkpoint["schema"] != SCHEMA or checkpoint["format"] != fmt:
                raise ValueError("checkpoint schema/format mismatch")
            self.net.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.revision = checkpoint["revision"]
            self.updates = checkpoint["updates"]
        else:
            self.save()
        self.loaded_mtime = self.path.stat().st_mtime_ns

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
        try:
            torch.save({"schema": SCHEMA, "format": self.fmt, "revision": self.revision,
                        "updates": self.updates, "model": self.net.state_dict(),
                        "optimizer": self.optimizer.state_dict()}, tmp)
            tmp.replace(self.path)
            self.loaded_mtime = self.path.stat().st_mtime_ns
        finally:
            tmp.unlink(missing_ok=True)

    def set_device(self, device: str = "cpu"):
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable on this host")
        self.net.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        return self

    def predict(self, state, actions, explore: bool = False):
        device = next(self.net.parameters()).device
        with torch.no_grad():
            logits, value = self.net(torch.as_tensor(state, device=device)[None], torch.as_tensor(actions, device=device)[None])
            dist = Categorical(logits=logits[0])
            index = int(dist.sample()) if explore else int(logits[0].argmax())
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
              duty_fraction: float = 1.0, deadline: float | None = None) -> dict:
        if not 1 <= epochs <= 30:
            raise ValueError("epochs must be between 1 and 30")
        samples = []
        ids = []
        for episode in episodes:
            if episode["format"] != self.fmt or episode.get("status") != "complete":
                continue
            if not imitation and (episode["revision"] != self.revision or not episode.get("on_policy")):
                continue
            if imitation and not episode.get("demonstration"):
                continue
            trajectory = episode.get("steps", [])
            if not trajectory or episode.get("schema") != SCHEMA:
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
                prepared.append({**step, "advantage": advantage, "return": advantage + step.get("value", 0.0)})
                next_value = step.get("value", 0.0)
            samples.extend(reversed(prepared))
        if not samples:
            return {"trained": False, "reason": "no compatible unconsumed demonstrations" if imitation else "no complete stochastic rollouts from the current model revision"}
        advantages = np.array([s["advantage"] for s in samples], dtype=np.float32)
        # One tiny batch must not lose all its policy gradient when variance is 0.
        if len(samples) > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for sample, advantage in zip(samples, advantages):
            sample["advantage"] = float(advantage)
        rng = np.random.default_rng(0)
        losses = []
        self.net.train()
        device = next(self.net.parameters()).device
        for _ in range(epochs):
            order = rng.permutation(len(samples))
            for start in range(0, len(samples), 32):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("training budget expired before saving candidate")
                started = time.monotonic()
                batch = [samples[i] for i in order[start:start + 32]]
                states, actions, mask, indices = self.batch(batch)
                states, actions, mask, indices = (t.to(device) for t in (states, actions, mask, indices))
                logits, values = self.net(states, actions, mask)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(indices)
                if imitation:
                    loss = -logprobs.mean()
                else:
                    old = torch.tensor([s["logprob"] for s in batch], device=device)
                    adv = torch.tensor([s["advantage"] for s in batch], device=device)
                    returns = torch.tensor([s["return"] for s in batch], device=device)
                    ratio = (logprobs - old).exp()
                    policy = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                    loss = policy + 0.5 * (values - returns).square().mean() - 0.01 * dist.entropy().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss; checkpoint has not been saved")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                losses.append(float(loss.detach()))
                if device.type == "mps" and duty_fraction < 1:
                    torch.mps.synchronize()
                    time.sleep(min(0.1, (time.monotonic() - started) * (1 / duty_fraction - 1)))
        old_revision = self.revision
        self.revision = uuid.uuid4().hex
        self.updates += 1
        self.save()
        return {"trained": True, "algorithm": "behavior-cloning" if imitation else "PPO", "episodes": len(ids),
                "steps": len(samples), "loss": sum(losses) / len(losses),
                "previous_revision": old_revision, "revision": self.revision, "consumed": ids}
