"""Learn executed opponent moves from strictly pre-turn public observations.

Public replays cannot supply exact private request masks for PPO. This auxiliary
classifier uses real executed-move labels instead; no guessed private requests,
future reveals, immobilized intents or called moves become training labels.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from battle_state import observe, to_id
from ml.features import Features, STATE_DIM
from ml.storage import Store, fingerprint, now

PUBLIC_KINDS = {"player", "gametype", "gen", "tier", "rule", "clearpoke", "poke", "showteam",
                "start", "turn", "move", "switch", "drag", "replace", "detailschange", "-formechange",
                "faint", "win", "tie", "-damage", "-heal", "-sethp", "-status", "-curestatus",
                "-item", "-enditem", "-ability", "-terastallize", "-boost", "-unboost", "-setboost",
                "-clearboost", "-clearallboost", "-start", "-end", "-weather", "-fieldstart",
                "-fieldend", "-sidestart", "-sideend"}


def public_lines(log: list[str] | str):
    lines = log.splitlines() if isinstance(log, str) else log
    # Player names are replaced locally; chat, requests, HTML and auth never persist.
    result = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 3 or parts[1] not in PUBLIC_KINDS:
            continue
        if parts[1] == "player":
            line = f"|player|{parts[2]}|Player-{parts[2]}"
        elif parts[1] == "win":
            line = "|win|redacted"
        result.append(line)
    return result


def public_context(log: list[str], perspective: str, fmt: str):
    request = {"side": {"id": perspective, "pokemon": []}}
    state = observe(request, log)
    opposite = "p2" if perspective == "p1" else "p1"
    inverse = observe({"side": {"id": opposite, "pokemon": []}}, log)
    state["my_actives"] = inverse["opp_actives"]
    state["my_party"] = inverse["opp_actives"]
    state["request_type"] = "public-observation"
    return {"format": fmt, "state": state, "choices": []}


def scout_vector(features: Features, ctx: dict, actor: dict):
    vector = features.state(ctx)
    features.add(vector, "scout/actor/" + to_id(actor["species"]))
    features.add(vector, "scout/position/" + actor["slot"][-1])
    return vector


def ingest_public(store: Store, fmt: str, log: list[str] | str, source: str, battle_id: str,
                  features: Features | None = None):
    if fmt != to_id(fmt) or not fmt:
        raise ValueError("exact format id required")
    lines = public_lines(log)
    if not any(line.startswith(("|win|", "|tie|")) for line in lines):
        return {"imported": False, "reason": "public game is not complete"}
    digest = fingerprint(lines)
    if store.db.execute("SELECT 1 FROM public_battles WHERE format=? AND digest=?", (fmt, digest)).fetchone():
        return {"imported": False, "reason": "already ingested"}
    features = features or Features.cached(fmt)
    before_turn = {}
    turn = 0
    samples = []
    prefix = []
    for line in lines:
        parts = line.split("|")
        if parts[1] == "turn":
            turn = int(parts[2])
            before_turn = {side: public_context(prefix + [line], side, fmt) for side in ("p1", "p2")}
        elif parts[1] == "move" and turn and len(parts) >= 4 and "[from]" not in line and before_turn:
            actor_id, move = parts[2], to_id(parts[3])
            actor_side = actor_id[:2]
            perspective = "p2" if actor_side == "p1" else "p1"
            ctx = before_turn[perspective]
            actor = next((m for m in ctx["state"]["opp_actives"] if m.get("ident") == actor_id), None)
            # A Pokémon switched in during the turn was not this actor before
            # joint decisions were submitted. Exclude those ambiguous examples.
            current = public_context(prefix, perspective, fmt)
            current_actor = next((m for m in current["state"]["opp_actives"] if m.get("slot") == actor_id.split(":")[0]), None)
            if actor and current_actor and current_actor.get("ident") == actor_id and move in features.dex.get("moves", {}):
                key = fingerprint({"battle": digest, "turn": turn, "actor": actor_id, "move": move})
                samples.append((key, fmt, to_id(actor["species"]), move,
                                json.dumps(scout_vector(features, ctx, actor).tolist()), digest))
        prefix.append(line)
    with store.db:
        store.db.execute("INSERT OR IGNORE INTO public_battles VALUES (?,?,?,?,?,?)",
                         (battle_id, fmt, source, now(), digest, json.dumps(lines)))
        store.db.executemany("INSERT OR IGNORE INTO scout_samples VALUES (?,?,?,?,?,?)", samples)
    return {"imported": True, "public_battle": battle_id, "samples": len(samples), "format": fmt}


class Scout:
    def __init__(self, store: Store, fmt: str, features: Features):
        self.store, self.fmt, self.features = store, fmt, features
        self.path = store.root / "scouts" / (fmt + ".pt")
        self.moves = sorted(features.dex.get("moves", {}))
        self.index = {move: i for i, move in enumerate(self.moves)}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            self.net = nn.Sequential(nn.Linear(STATE_DIM, 96), nn.Tanh(), nn.Linear(96, max(1, len(self.moves))))
        self.trained_samples = 0
        self.loaded_mtime = None
        self.transfer_source = None
        self.reload()
        if not self.path.exists() and "champions" in fmt:
            donor = store.root/"scouts"/"gen9championsvgc2026regmb.pt"
            if donor.exists() and fmt != "gen9championsvgc2026regmb":
                data = torch.load(donor, map_location="cpu", weights_only=True)
                old = data["model"]
                current = self.net.state_dict()
                for name in ("0.weight", "0.bias"):
                    current[name] = old[name]
                donor_index = {move:i for i,move in enumerate(data["moves"])}
                for move, index in self.index.items():
                    if move in donor_index:
                        current["2.weight"][index] = old["2.weight"][donor_index[move]]
                        current["2.bias"][index] = old["2.bias"][donor_index[move]]
                self.net.load_state_dict(current)
                self.transfer_source = "gen9championsvgc2026regmb (historical warm start; current-format validation required)"

    def reload(self):
        if self.path.exists() and self.loaded_mtime != self.path.stat().st_mtime_ns:
            data = torch.load(self.path, map_location="cpu", weights_only=True)
            if data["moves"] != self.moves:
                return
            self.net.load_state_dict(data["model"])
            self.trained_samples = data["samples"]
            self.loaded_mtime = self.path.stat().st_mtime_ns

    def predict(self, log: list[str], perspective: str):
        self.reload()
        if not self.trained_samples:
            return []
        ctx = public_context(public_lines(log), perspective, self.fmt)
        results = []
        for actor in ctx["state"]["opp_actives"]:
            species = to_id(actor["species"])
            sheet = next((s for s in ctx["state"]["opp_team_sheet"] if to_id(s["species"]) == species), None)
            if sheet:
                allowed = [to_id(m) for m in sheet["moves"]]
            else:
                allowed = [r[0] for r in self.store.db.execute("SELECT DISTINCT move FROM scout_samples WHERE format=? AND species=?", (self.fmt, species))]
            allowed = [m for m in allowed if m in self.index]
            if not allowed:
                continue
            with torch.no_grad():
                logits = self.net(torch.as_tensor(scout_vector(self.features, ctx, actor)))
                probs = logits[[self.index[m] for m in allowed]].softmax(0).tolist()
            ranked = sorted(zip(allowed, probs), key=lambda pair: pair[1], reverse=True)
            results.append({"slot": actor["slot"], "species": actor["species"],
                            "moves": [{"move": m, "probability": p} for m, p in ranked[:3]],
                            "trained_samples": self.trained_samples, "target": "executed public move"})
        return results

    def train(self, epochs: int = 3, device: str = "cpu", max_samples: int = 20000,
              duty_fraction: float = 1.0):
        rows = list(self.store.db.execute("SELECT * FROM scout_samples WHERE format=? ORDER BY id LIMIT ?", (self.fmt, max_samples)))
        if len(rows) < 8 or not self.moves:
            return {"trained": False, "reason": "need at least 8 usable public-move samples"}
        train, heldout = [], []
        for row in rows:
            if row["move"] not in self.index or not row["battle"]:
                continue
            # Split by battle rather than by turn, so one game doesn't leak into
            # both partitions. Rows contain only pre-turn observations.
            bucket = int(row["battle"][:8], 16) % 5
            (heldout if bucket == 0 else train).append(row)
        if not train:
            return {"trained": False, "reason": "no training partition"}
        def tensors(partition):
            return (torch.tensor([json.loads(r["vector"]) for r in partition], device=device),
                    torch.tensor([self.index[r["move"]] for r in partition], device=device))
        self.net.to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        validation = tensors(heldout) if heldout else None
        def validation_loss():
            with torch.no_grad():
                return float(nn.functional.cross_entropy(self.net(validation[0]), validation[1])) if validation else None
        before = validation_loss()
        for _ in range(epochs):
            for start in range(0, len(train), 128):
                import time
                started = time.monotonic()
                x, y = tensors(train[start:start+128])
                loss = nn.functional.cross_entropy(self.net(x), y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                optimizer.step()
                if device == "mps" and duty_fraction < 1:
                    torch.mps.synchronize()
                    time.sleep(min(0.1, (time.monotonic()-started) * (1/duty_fraction - 1)))
        after = validation_loss()
        self.net.cpu()
        promoted = len(heldout) >= 8 and after <= before
        result = {"trained": True, "samples": len(rows), "heldout": len(heldout),
                  "validation_loss_before": before, "validation_loss_after": after, "promoted": promoted, "device": device,
                  "transfer_source": self.transfer_source}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        target = self.path if promoted else self.path.with_suffix(".candidate.pt")
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save({"model": self.net.state_dict(), "samples": len(rows), "moves": self.moves}, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        if promoted:
            self.trained_samples = len(rows)
        return result
