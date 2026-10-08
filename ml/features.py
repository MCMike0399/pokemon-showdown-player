"""Stable, bounded state and action features; only observations available at choice time."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from battle_state import hp_fraction, legacy_hp_fraction, to_id

SCHEMA = 1
STATE_DIM = 384
ACTION_DIM = 192
FEATURE_PROFILES = {'legacy', 'weather-v1', 'tactics-v1', 'preview-v1', 'preview-v2', 'rain-v1', 'mechanics-v1', 'lineup-v1'}


class Features:
    def __init__(self, dex: dict | None = None):
        self.dex = dex or {}
        self._signature = None

    def signature(self):
        if self._signature is None:
            self._signature = hashlib.sha256(json.dumps(self.dex, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return self._signature

    @classmethod
    def cached(cls, fmt: str = ""):
        root = Path(__file__).resolve().parents[1] / "cache"
        if fmt and (root / "dex" / fmt).exists():
            root = root / "dex" / fmt
        return cls({name: json.loads((root / f"{name}.json").read_text())
                    for name in ("moves", "pokedex", "typechart") if (root / f"{name}.json").exists()})

    @staticmethod
    def add(vector, name, value=1.0):
        raw = hashlib.blake2b(name.encode(), digest_size=8).digest()
        index = int.from_bytes(raw[:4], "little") % (len(vector) - 16)
        vector[index] += float(value) * (1 if raw[4] % 2 else -1)

    def species(self, name: str) -> dict:
        return self.dex.get("pokedex", {}).get(to_id(name), {})

    def move_data(self, ctx: dict, name: str, data: dict, actor: dict | None = None, targets: list | None = None) -> dict:
        """Optional weather-aware encoding; old checkpoints keep exact legacy inputs."""
        profile = ctx.get('feature_profile', 'legacy')
        if profile not in FEATURE_PROFILES:
            raise ValueError('unsupported feature profile')
        if profile in ('legacy', 'preview-v1', 'preview-v2', 'lineup-v1'):
            return data
        state = ctx['state']
        active = state.get('my_actives', []) + state.get('opp_actives', [])
        suppressed = any(hp_fraction(mon.get('condition')) > 0 and
                         to_id(mon.get('ability') or mon.get('baseAbility') or '') in ('cloudnine', 'airlock')
                         for mon in active)
        weather = '' if suppressed else to_id(state.get('weather') or '')
        affected = actor or (state.get('my_actives') or [{}])[0]
        if name in ('hurricane', 'thunder') and targets and len(targets) == 1:
            affected = targets[0]
        if to_id(affected.get('item') or '') == 'utilityumbrella' and weather in ('raindance', 'primordialsea', 'sunnyday', 'desolateland'):
            weather = ''
        result = dict(data)
        types = {'raindance': 'Water', 'primordialsea': 'Water', 'sunnyday': 'Fire',
                 'desolateland': 'Fire', 'sandstorm': 'Rock', 'hail': 'Ice', 'snowscape': 'Ice'}
        if name == 'weatherball' and weather in types:
            result.update(type=types[weather], basePower=data.get('basePower', 50) * 2)
        elif name in ('hurricane', 'thunder'):
            if weather in ('raindance', 'primordialsea'):
                result['accuracy'] = True
            elif weather in ('sunnyday', 'desolateland'):
                result['accuracy'] = 50
        return result

    def effectiveness(self, attack: str, mon: dict) -> float:
        types = [mon["teraType"]] if mon.get("teraType") else self.species(mon.get("species", "")).get("types", [])
        multiplier = 1.0
        for target in types:
            code = self.dex.get("typechart", {}).get(target, {}).get("damageTaken", {}).get(attack, 0)
            multiplier *= {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}.get(code, 1.0)
        return multiplier

    def _mon(self, vector, prefix, mon, profile='legacy'):
        for key in ("species", "item", "ability", "baseAbility", "teraType"):
            if mon.get(key):
                self.add(vector, f"{prefix}/{key}/{to_id(mon[key])}")
        health = hp_fraction if profile in ('tactics-v1', 'rain-v1') else legacy_hp_fraction
        self.add(vector, prefix + "/hp", health(mon.get("condition")))
        for status in ("par", "brn", "slp", "psn", "tox", "frz", "fnt"):
            if status in (mon.get("condition") or "").split():
                self.add(vector, prefix + "/status/" + status)
        for stat, value in mon.get("boosts", {}).items():
            self.add(vector, prefix + "/boost/" + stat, value / 6)
        for stat, value in (mon.get("stats") or self.species(mon.get("species", "")).get("baseStats", {})).items():
            self.add(vector, prefix + "/stat/" + stat, value / 255)
        for t in self.species(mon.get("species", "")).get("types", []):
            self.add(vector, prefix + "/type/" + t)
        for move in mon.get("moves", []):
            if isinstance(move, str):
                self.add(vector, prefix + "/move/" + to_id(move))
            else:
                name = to_id(move.get("id", move.get("move", "")))
                self.add(vector, prefix + "/move/" + name, (move.get("pp", 1) / max(1, move.get("maxpp", 1))) if not move.get("disabled") else -1)
        for effect in mon.get("volatiles", []):
            self.add(vector, prefix + "/volatile/" + effect)

    def state(self, ctx: dict, knowledge: dict | None = None) -> np.ndarray:
        vector = np.zeros(STATE_DIM, dtype=np.float32)
        state = ctx["state"]
        self.add(vector, "format/" + ctx.get("format", ""))
        self.add(vector, "request/" + state["request_type"])
        for key in ("weather",):
            self.add(vector, key + "/" + str(state.get(key)))
        for effect in state.get("field", []):
            self.add(vector, "field/" + effect)
        for side, effects in state.get("hazards", {}).items():
            for effect in effects:
                self.add(vector, side + "/condition/" + effect)
        for group in ("my_actives", "my_party", "opp_actives", "opp_revealed", "opp_team_sheet"):
            for i, mon in enumerate(state.get(group, [])):
                self._mon(vector, f"{group}/{i}", mon, ctx.get('feature_profile', 'legacy'))
        for name in state.get("opp_preview", []):
            self.add(vector, "preview/" + to_id(name))
        for i, event in enumerate(reversed(state.get("history", []))):
            self.add(vector, f"history/{i}/{event['side']}/{event['slot']}/{to_id(event['move'])}", 1 / math.sqrt(i + 1))
        for species, frequency in (knowledge or {}).items():
            self.add(vector, "meta/" + species, frequency)
        vector[-16] = min(float(state.get("turn", 0)) / 50, 2)
        health = hp_fraction if ctx.get('feature_profile') in ('tactics-v1', 'rain-v1') else legacy_hp_fraction
        vector[-15] = sum(health(m.get("condition")) for m in state.get("my_party", [])) / 6
        vector[-14] = len(ctx.get("choices", [])) / 1000
        return np.clip(vector, -5, 5)

    def action(self, ctx: dict, choice: str) -> np.ndarray:
        vector = np.zeros(ACTION_DIM, dtype=np.float32)
        state = ctx["state"]
        health = hp_fraction if ctx.get('feature_profile') in ('tactics-v1', 'rain-v1') else legacy_hp_fraction
        party = state.get("my_party", [])
        prior = 0.0
        if choice.startswith("team "):
            indices = [int(i) - 1 for i in choice[5:].split(",")]
            for order, index in enumerate(indices):
                if index < len(party):
                    self._mon(vector, f"team/{order}", party[index], ctx.get('feature_profile', 'legacy'))
            self.add(vector, "team-preview")
        else:
            for i, component in enumerate(choice.split(",")):
                parts = component.strip().split()
                self.add(vector, f"slot/{i}/{parts[0]}")
                if parts[0] == "switch":
                    mon = party[int(parts[1]) - 1]
                    self._mon(vector, f"switch/{i}", mon, ctx.get('feature_profile', 'legacy'))
                    prior -= 0.4
                elif parts[0] == "move":
                    me = state.get("my_actives", [])[i]
                    move = me.get("moves", [])[int(parts[1]) - 1]
                    name = to_id(move.get("id", move.get("move", "")))
                    target = next((int(p) for p in parts[2:] if p.lstrip("-+").isdigit()), 0)
                    targets = state.get("opp_actives", [])
                    if target > 0:
                        targets = [m for m in targets if m.get("slot", "").endswith(chr(96 + target))]
                    elif target < 0:
                        targets = state.get("my_actives", [])[-target - 1:-target]
                    data = self.move_data(ctx, name, self.dex.get("moves", {}).get(name, {}), me, targets)
                    for key in ("type", "category", "target"):
                        self.add(vector, f"move/{i}/{key}/{data.get(key, move.get(key, ''))}")
                    self.add(vector, f"move/{i}/{name}")
                    self.add(vector, f"target/{i}/{target}")
                    for event in ("mega", "megax", "megay", "terastallize"):
                        if event in parts:
                            self.add(vector, f"event/{i}/{event}")
                    power = data.get("basePower", 0) / 100
                    accuracy = data.get("accuracy", 100)
                    accuracy = 1.0 if accuracy is True else float(accuracy or 100) / 100
                    stab = 1.5 if data.get("type") in self.species(me.get("species", "")).get("types", []) else 1.0
                    pressure = sum(power * accuracy * stab * self.effectiveness(data.get("type", ""), m)
                                   for m in targets if health(m.get("condition")) > 0)
                    if target < 0:
                        pressure *= -1
                    # A tactical initialization, not an exact damage calculator.
                    prior += pressure
                    if name in ("protect", "detect", "wideguard"):
                        prior += 0.15
                    vector[-16 + i * 4] = power
                    vector[-15 + i * 4] = data.get("priority", 0) / 7
                    vector[-14 + i * 4] = min(pressure, 8) / 4
                    vector[-13 + i * 4] = health(me.get("condition"))
            components = choice.split(",")
            if len(components) == 2:
                # Joint interaction allows learning focus fire, attack+support,
                # double Protect and switch+attack rather than independent picks.
                self.add(vector, "joint/" + "/".join(p.strip().split()[0] for p in components))
        if ctx.get('feature_profile') == 'tactics-v1':
            from ml.tactics import score
            prior = score(ctx, choice, self)
        elif ctx.get('feature_profile') == 'rain-v1':
            from ml.rain import score
            prior = score(ctx, choice, self)
        elif ctx.get('feature_profile') == 'mechanics-v1':
            from ml.mechanics import score
            prior = score(ctx, choice, self)
        elif ctx.get('feature_profile') in ('preview-v1', 'preview-v2') and choice.startswith('team '):
            from ml.preview import score
            prior = score(ctx, choice, self)
            if ctx['feature_profile'] == 'preview-v2':
                prior = 4 * prior / (4 + abs(prior))
        elif ctx.get('feature_profile') == 'lineup-v1' and choice.startswith('team '):
            from ml.preview import rain_lineup_score
            prior = rain_lineup_score(ctx, choice, self)
        vector[-1] = max(-4, min(4, prior))
        return np.clip(vector, -5, 5)

    def encode(self, ctx, knowledge=None):
        return self.state(ctx, knowledge), np.stack([self.action(ctx, c) for c in ctx["choices"]])
