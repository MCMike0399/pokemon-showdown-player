"""Harness adapter: inference, durable trajectories, terminal rewards and learning."""
from __future__ import annotations

import uuid
import time

from ml.features import Features, SCHEMA
from ml.model import Model
from ml.storage import Store, fingerprint, now


class Brain:
    def __init__(self, store: Store | None = None, features: Features | None = None,
                 inference_store: Store | None = None):
        self.store = store or Store()
        self.inference_store = inference_store or self.store
        self.features = features or Features.cached()
        self.explicit_features = features is not None
        self.format_features = {}
        self.models = {}
        self.pending = {}
        self.scouts = {}

    def resume(self, room: str, side: str):
        """Restore one durable recording; ambiguous fragments require reconciliation."""
        key = (room, side)
        if key in self.pending:
            return self.pending[key]
        rows = self.store.room_episodes(room, side)
        if any(row["status"] == "complete" for row in rows):
            return None
        pending = [row for row in rows if row["status"] == "pending"]
        if len(pending) > 1:
            raise ValueError("multiple pending episode segments; reconcile this room before resuming")
        if not pending:
            return None
        episode = pending[0]
        if episode.get("schema") != SCHEMA:
            raise ValueError("pending episode schema does not match this recorder")
        episode["recoveries"] = episode.get("recoveries", 0) + 1
        self.pending[key] = episode
        return episode

    def room_side(self, room: str):
        sides = {side for (recorded_room, side) in self.pending if recorded_room == room}
        sides.update(row.get("side") for row in self.store.room_episodes(room))
        sides.discard(None)
        if len(sides) > 1:
            raise ValueError("ambiguous recording perspective for this room")
        return next(iter(sides), "p1")

    def model(self, fmt):
        # Daily workers can promote a checkpoint while this MCP process stays
        # alive. Reload only between recorded games; active games keep their brain.
        if fmt in self.models and not any(e["format"] == fmt for e in self.pending.values()):
            import torch
            model = self.models[fmt]
            if model.path.exists() and model.path.stat().st_mtime_ns != model.loaded_mtime:
                disk = torch.load(model.path, map_location="cpu", weights_only=True)
                if disk["revision"] != model.revision:
                    self.models[fmt] = Model(self.store.root, fmt)
        if fmt not in self.models:
            path = self.store.root / 'models' / (fmt + '.pt')
            if path.exists():
                # Checkpoints are atomically replaced. Loading an existing one
                # is a reader operation and must not contend with learning.
                self.models[fmt] = Model(self.store.root, fmt)
            else:
                with self.store.writer():
                    self.models[fmt] = Model(self.store.root, fmt)
        return self.models[fmt]

    def decide(self, ctx: dict, explore: bool = False, record: bool = False, demonstration: str | None = None) -> dict:
        started = time.perf_counter()
        if not ctx.get("choices"):
            raise ValueError("no actionable request")
        fmt = ctx.get("format")
        if not fmt:
            raise ValueError("explicit battle format is required")
        if record and not explore and demonstration is None:
            raise ValueError("PPO recording requires sampled actions (explore=True)")
        if record or demonstration is not None:
            from ml.recording import RECORDER_VERSION, encoded_input
            side = (ctx["request"].get("side") or {}).get("id", "p1")
            restored = self.resume(ctx["room"], side)
            if restored is None and any(row["status"] == "complete" for row in self.store.room_episodes(ctx["room"], side)):
                raise ValueError("this room already has a completed recording")
        model = self.model(fmt)
        collecting = self.pending.get((ctx['room'], ctx['request'].get('side', {}).get('id', 'p1')))
        if collecting and collecting.get('feature_profile', 'legacy') != model.feature_profile:
            raise ValueError('cannot change feature profile during an episode')
        policy = (collecting or {}).get('collecting_policy')
        if policy and (policy['checkpoint_sha256'] != model.checkpoint_sha256 or
                       policy['policy_temperature'] != model.policy_temperature or
                       policy['preview_temperature'] != model.preview_temperature):
            raise ValueError('cannot change the collecting checkpoint or temperatures during an episode')
        ctx = {**ctx, 'feature_profile': model.feature_profile,
               'strategy_knowledge': model.strategy_knowledge}
        # Research affects bounded species-frequency features, never raw text.
        from ml.research import species_prior
        knowledge = species_prior(self.inference_store, fmt)
        if fmt not in self.format_features:
            self.format_features[fmt] = self.features if self.explicit_features else Features.cached(fmt)
        features = self.format_features[fmt]
        state, actions = features.encode(ctx, knowledge)
        scout_predictions = []
        if ctx.get("public_log"):
            from ml.scout import Scout
            if fmt not in self.scouts:
                self.scouts[fmt] = Scout(self.inference_store, fmt, features)
            side = ctx["request"].get("side", {}).get("id", "p1")
            scout_predictions = self.scouts[fmt].predict(ctx["public_log"], side)
            for opponent in scout_predictions:
                for move in opponent["moves"]:
                    features.add(state, "scout/" + opponent["slot"][-1] + "/" + move["move"], move["probability"])
        preview = bool(ctx['request'].get('teamPreview'))
        demonstrated_index = None
        if demonstration is not None:
            if demonstration not in ctx["choices"]:
                raise ValueError("demonstrated action is not in the legal mask")
            demonstrated_index = ctx["choices"].index(demonstration)
        prediction = model.predict(state, actions, explore, preview=preview, selected_index=demonstrated_index)
        inference_ms = round((time.perf_counter() - started) * 1000, 3)
        choice = ctx["choices"][prediction["index"]]
        if record or demonstration is not None:
            room = ctx["room"]
            key = (room, (ctx["request"].get("side") or {}).get("id", "p1"))
            episode = self.pending.get(key)
            if episode is None:
                from ml.reload import runtime_generation
                episode = {"id": uuid.uuid4().hex, "room": room, "format": fmt, "team": ctx.get("team_id", ""),
                           "source": ctx.get("source", "ladder"), "revision": model.revision,
                           "schema": SCHEMA, "created": now(), "status": "pending", "steps": [],
                           "side": key[1],
                           "feature_profile": model.feature_profile,
                           "recorder_version": RECORDER_VERSION,
                           'collecting_policy': {'checkpoint_sha256': model.checkpoint_sha256,
                               'source_generation': runtime_generation(),
                               'checkpoint_archive': self.store.retain_checkpoint(model),
                               'policy_temperature': model.policy_temperature,
                               'preview_temperature': model.preview_temperature,
                               'architecture': model.architecture,
                               'feature_profile': model.feature_profile, 'dex_sha256': features.signature()},
                           'team_sets': ctx.get('team_sets'),
                           'simulation': ctx.get('simulation'),
                           "on_policy": demonstration is None, "demonstration": demonstration is not None}
                self.pending[key] = episode
            if episode["revision"] != model.revision or episode["format"] != fmt or episode["team"] != ctx.get("team_id", ""):
                raise ValueError("cannot change model, format or team during an episode")
            if episode["demonstration"] != (demonstration is not None):
                raise ValueError("cannot mix demonstrations and stochastic rollouts")
            # Rejects and retries overwrite the same decision, rather than giving
            # unexecuted proposals credit for the eventual game outcome.
            request_id = ctx["request"].get("rqid", fingerprint(ctx["request"]))
            step = {"request_id": request_id, "state": encoded_input(state), "actions": encoded_input(actions),
                    "index": prediction["index"], "choice": choice, "logprob": prediction["logprob"], "value": prediction["value"],
                    'inference_ms': inference_ms}
            from ml.recording import snapshot
            step['snapshot'] = snapshot(episode, ctx, prediction, knowledge, scout_predictions, model.temperature(preview))
            if ctx.get("source") == "ladder":
                step["submitted"] = False
            if episode["steps"] and episode["steps"][-1]["request_id"] == request_id:
                episode["steps"][-1] = step
            else:
                episode["steps"].append(step)
            self.store.save_episode(episode)
        ranked = sorted(zip(ctx["choices"], prediction["probabilities"]), key=lambda p: p[1], reverse=True)
        return {"choice": choice, "probability": prediction["probabilities"][prediction["index"]],
                "value": prediction["value"], "revision": model.revision, "updates": model.updates,
                "sampled": explore, "recorded": record or demonstration is not None,
                'inference_ms': inference_ms,
                "opponent_predictions": scout_predictions,
                "top_choices": [{"choice": c, "probability": p} for c, p in ranked[:5]]}

    def reject(self, room: str, side: str = "p1", reason: str = 'rejected-or-interrupted-proposal'):
        episode = self.pending.get((room, side))
        if episode and episode["steps"]:
            removed = episode["steps"].pop()
            episode.setdefault('discarded_proposals', []).append({'reason': reason, 'discarded_at': now(), 'step': removed})
            self.store.save_episode(episode)

    def finish(self, room: str, result: dict, user: str, side: str = "p1", public_log: list[str] | None = None) -> dict:
        key = (room, side)
        episode = self.pending.get(key)
        if result.get("ongoing") or result.get("unfinished") or result.get("stalled") or (not result.get("tie") and not result.get("winner")):
            return {"recorded": False, "reason": "battle is not terminal; no reward assigned"}
        if not episode:
            completed = [row for row in self.store.room_episodes(room, side) if row["status"] == "complete"]
            if completed:
                row = completed[-1]
                return {"recorded": False, "already_recorded": True, "id": row["id"], "format": row["format"],
                        "steps": len(row["steps"]), "outcome": row["outcome"]}
            episode = self.resume(room, side)
        if not episode:
            return {"recorded": False, "reason": "no pending episode"}
        if not user:
            raise ValueError("player identity is required to attribute the outcome")
        episode["outcome"] = 0.0 if result.get("tie") else 1.0 if result["winner"].lower().replace(" ", "") == user.lower().replace(" ", "") else -1.0
        episode["status"] = "complete"
        if public_log is not None:
            from ml.recording import record_log
            episode['terminal_log'] = record_log(episode, public_log)
        # Completed records are immutable. Attach guarded feedback before the
        # single terminal write; review failures cannot block attribution.
        if public_log is not None:
            from ml.postgame import review
            try:
                episode['postgame'] = review(episode, public_log)
            except Exception as error:
                episode['postgame'] = {'version': 1, 'error': type(error).__name__}
        self.store.save_episode(episode)
        del self.pending[key]
        return {"recorded": True, "id": episode["id"], "format": episode["format"], "steps": len(episode["steps"]), "outcome": episode["outcome"]}

    def train(self, fmt: str, epochs: int = 4, imitation: bool = False):
        model = self.model(fmt)
        with self.store.writer():
            # An external CLI may have advanced the disk checkpoint since this
            # MCP process loaded it. Never overwrite those learned weights.
            import torch
            checkpoint = torch.load(model.path, map_location="cpu", weights_only=True)
            if checkpoint["revision"] != model.revision:
                raise ValueError("checkpoint changed in another process; restart/reload before training")
            return self._train(model, fmt, epochs, imitation)

    def _train(self, model, fmt, epochs, imitation):
        episodes = self.store.episodes(fmt, None if imitation else model.revision)
        if any(e["format"] == fmt for e in self.pending.values()):
            raise ValueError("finish active episodes before updating this format's model")
        result = model.train(episodes, epochs, imitation)
        if result.get("trained"):
            self.store.mark_trained(result.pop("consumed"))
        return result
