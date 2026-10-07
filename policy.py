"""
policy.py - a better sample policy for the harness (still just an example).

Uses Pokemon Showdown data (typechart + pokedex + moves, cached under cache/)
to pick the most effective damaging move, switch when low, and avoid the
switch-loop / move-repeat pathologies.

Agents can replace this with their own callable: decide(ctx) -> choice string.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA = "https://play.pokemonshowdown.com/data"

TYPE_MULT = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}


def _id(name: str) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


async def load_data(names: tuple = ("typechart", "pokedex", "moves")) -> Dict[str, dict]:
    """Download + cache PS data. Some tables only ship as .js (exports.BattleX = {...})."""
    CACHE.mkdir(exist_ok=True)
    out: Dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=60) as c:
        for n in names:
            f = CACHE / (n + ".json")
            if not f.exists():
                body = None
                for ext in ("json", "js"):
                    r = await c.get(f"{DATA}/{n}.{ext}", headers={"User-Agent": "ps-player/1.0"})
                    if r.status_code != 200:
                        continue
                    text = r.text
                    if ext == "js":
                        # data tables ship as JS: exports.BattleX = {key: {...}}  (unquoted keys)
                        text = text[text.index("{"): text.rindex("}") + 1]
                        text = re.sub(r"([{,]\s*)([A-Za-z_$][\w$]*)\s*:", r'\1"\2":', text)
                        text = text.replace("'", '"')
                    body = json.loads(text)
                    break
                if body is None:
                    raise RuntimeError(f"could not fetch PS data table {n!r}")
                f.write_text(json.dumps(body))
            out[n] = json.loads(f.read_text())
    return out


def hp_pct(cond: Optional[str]) -> float:
    try:
        cur = (cond or "").split(" ")[0]
        a, b = cur.split("/")
        return 100.0 * float(a) / float(b)
    except Exception:
        return 100.0


class SmartPolicy:
    def __init__(self, typechart: dict, pokedex: dict, moves: dict) -> None:
        self.tc, self.px, self.mv = typechart, pokedex, moves
        self.history: List[str] = []
        self.last_action: str = ""

    def _types(self, species: str) -> List[str]:
        return (self.px.get(_id(species), {}) or {}).get("types", []) or []

    def _eff(self, atk_type: str, def_types: List[str]) -> float:
        m = 1.0
        for dt in def_types:
            row = (self.tc.get(dt, {}) or {}).get("damageTaken", {})
            m *= TYPE_MULT.get(row.get(atk_type, 0), 1.0)
        return m

    def _score_move(self, move_name: str, def_types: List[str]) -> float:
        d = self.mv.get(_id(move_name), {}) or {}
        bp = d.get("basePower") or 0
        if not bp:
            return 0.0
        acc = d.get("accuracy")
        acc = 100.0 if acc is True or acc is None else float(acc)
        return bp * self._eff(d.get("type", "normal"), def_types) * (acc / 100.0)

    def __call__(self, ctx: Dict[str, Any]) -> str:
        state = ctx.get("state") or {}
        choices = ctx.get("choices") or []
        rtype = state.get("request_type")
        moves = [c for c in choices if c.startswith("move")]
        switches = [c for c in choices if c.startswith("switch")]

        if rtype == "teamPreview" or (choices and choices[0].startswith("team ")):
            self.last_action = "preview"
            return choices[0]

        me = state.get("my_active") or {}
        party = state.get("my_party") or []
        opp_species = (state.get("opp_active") or {}).get("species") or ""
        opp_types = self._types(opp_species)
        hp = hp_pct(me.get("condition"))

        def party_cond(choice: str) -> str:
            """switch N refers to party slot N (1-based), NOT the bench list index."""
            try:
                idx = int(choice.split()[1]) - 1
            except Exception:
                return "100/100"
            if 0 <= idx < len(party):
                return party[idx].get("condition") or "100/100"
            return "100/100"

        healthy_bench = [s for s in switches
                         if not party_cond(s).endswith(" fnt") and hp_pct(party_cond(s)) > 40]

        # 1) low HP and a healthy bench and we didn't just switch -> switch
        if hp < 30 and healthy_bench and self.last_action != "switch":
            best = max(healthy_bench, key=lambda s: hp_pct(party_cond(s)))
            self.last_action = "switch"
            self._note(best)
            return best

        # 2) pick the best damaging move
        req_moves = (me.get("moves") or [])
        scored = []
        for i, c in enumerate(moves):
            name = req_moves[i]["move"] if i < len(req_moves) else c
            scored.append((self._score_move(name, opp_types), c))
        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            pick = scored[0][1]
            # 3) avoid the same move three turns running
            if self.history[-2:] == [pick, pick] and len(scored) > 1:
                pick = scored[1][1]
            self.last_action = "attack"
            self._note(pick)
            return pick

        # 4) fall back: first move, else first choice
        pick = moves[0] if moves else (choices[0] if choices else "default")
        self.last_action = "fallback"
        self._note(pick)
        return pick

    def _note(self, pick: str) -> None:
        self.history.append(pick)
        del self.history[:-8]


def _bench_cond(bench: List[dict], choice: str) -> str:
    try:
        idx = int(choice.split()[1]) - 1
    except Exception:
        return "100/100"
    # bench list is ordered as the in-battle party minus actives; approximate by index
    if 0 <= idx < len(bench):
        return bench[idx].get("condition") or "100/100"
    return "100/100"
