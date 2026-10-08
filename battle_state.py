"""Perspective-correct public battle observations and joint singles/doubles actions.

The request is the authority for private information; protocol lines are only
the public player stream. Do not feed an omniscient simulator log to this module.
"""
from __future__ import annotations

from itertools import permutations, product
import re
from typing import Any


def to_id(value: str) -> str:
    return "".join(c for c in str(value).lower() if c.isalnum())


def legacy_hp_fraction(condition: str | None) -> float:
    text = (condition or "").split(" ")[0]
    if text == "0" or "fnt" in (condition or ""):
        return 0.0
    try:
        current, maximum = text.split("/")
        return max(0.0, min(1.0, float(current) / float(maximum)))
    except (ValueError, ZeroDivisionError):
        return 1.0


def hp_fraction(condition: str | None) -> float:
    # Public HP can carry the display color immediately after the denominator.
    text = re.sub(r'^(\d+/\d+)[gry](?=\s|$)', r'\1', condition or '')
    return legacy_hp_fraction(text)


def unpack_team(packed: str) -> list[dict]:
    result = []
    for member in packed.split("]"):
        fields = member.split("|") + [""] * 12
        if not fields[0]:
            continue
        result.append({"name": fields[0], "species": fields[1] or fields[0],
                       "item": fields[2], "ability": fields[3],
                       "moves": fields[4].split(",") if fields[4] else [],
                       "nature": fields[5]})
    return result


def observe(request: dict, log: list[str], user: str = "", room: str = "") -> dict:
    mine = (request.get("side") or {}).get("id", "p1")
    foe = "p2" if mine == "p1" else "p1"
    slots: dict[str, dict] = {}
    revealed: dict[str, dict] = {}
    preview = {"p1": [], "p2": []}
    sheets = {"p1": [], "p2": []}
    conditions = {"p1": set(), "p2": set()}
    field: set[str] = set()
    weather = None
    turn = 0
    history = []
    for line in log:
        p = line.split("|")
        if len(p) < 3:
            continue
        kind = p[1]
        ident = p[2]
        slot = ident.split(":")[0]
        if kind == "turn":
            turn = int(ident)
        elif kind == "poke" and ident in preview and len(p) > 3:
            preview[ident].append(p[3].split(",")[0])
        elif kind == "showteam" and ident in sheets and len(p) > 3:
            # Packed teams themselves contain pipes; retain the whole payload.
            sheets[ident] = unpack_team(line.split("|", 3)[3])
        elif kind in ("switch", "drag", "replace") and len(p) > 4:
            old = revealed.get(ident, {})
            mon = {**old, "ident": ident, "slot": slot, "species": p[3].split(",")[0],
                   "condition": p[4], "boosts": {}, "volatiles": []}
            revealed[ident] = mon
            slots[slot] = mon
        elif kind in ("detailschange", "-formechange") and slot in slots and len(p) > 3:
            slots[slot]["species"] = p[3].split(",")[0]
        elif kind in ("-damage", "-heal", "-sethp") and slot in slots and len(p) > 3:
            slots[slot]["condition"] = p[3]
        elif kind == "faint" and slot in slots:
            slots[slot]["condition"] = "0 fnt"
        elif kind in ("-status", "-curestatus") and slot in slots and len(p) > 3:
            condition = slots[slot].get("condition", "100/100").split()[0]
            slots[slot]["condition"] = condition + (" " + p[3] if kind == "-status" else "")
        elif kind in ("-item", "-enditem", "-ability", "-terastallize") and slot in slots and len(p) > 3:
            key = {"-item": "item", "-enditem": "item", "-ability": "ability", "-terastallize": "teraType"}[kind]
            slots[slot][key] = "" if kind == "-enditem" else p[3]
        elif kind in ("-boost", "-unboost", "-setboost") and slot in slots and len(p) > 4:
            boosts = slots[slot]["boosts"]
            amount = int(p[4])
            boosts[p[3]] = amount if kind == "-setboost" else max(-6, min(6, boosts.get(p[3], 0) + amount * (-1 if kind == "-unboost" else 1)))
        elif kind == "-clearboost" and slot in slots:
            slots[slot]["boosts"] = {}
        elif kind == "-clearallboost":
            for mon in slots.values():
                mon["boosts"] = {}
        elif kind in ("-start", "-end") and slot in slots and len(p) > 3:
            effects = set(slots[slot]["volatiles"])
            effects.add(p[3]) if kind == "-start" else effects.discard(p[3])
            slots[slot]["volatiles"] = sorted(effects)
        elif kind == "-weather":
            weather = None if ident == "none" else ident
        elif kind in ("-fieldstart", "-fieldend"):
            field.add(ident) if kind == "-fieldstart" else field.discard(ident)
        elif kind in ("-sidestart", "-sideend") and len(p) > 3:
            side = ident[:2]
            if side in conditions:
                conditions[side].add(p[3]) if kind == "-sidestart" else conditions[side].discard(p[3])
        elif kind == "move" and len(p) > 3:
            history.append({"side": "mine" if ident.startswith(mine) else "theirs",
                            "slot": slot[-1:], "move": p[3], "turn": turn})
            if slot in slots:
                known = set(slots[slot].get("moves_known", []))
                known.add(p[3])
                slots[slot]["moves_known"] = sorted(known)
    party = (request.get("side") or {}).get("pokemon") or []
    active_req = request.get("active") or []
    count = len(active_req) or len(request.get("forceSwitch") or [])
    # The simulator orders side.pokemon with active positions first, including a
    # fainted/commanding position. Filtering on HP would change move slot indices.
    my_actives = []
    for i in range(count):
        mon = party[i] if i < len(party) else {}
        slot = mine + chr(97 + i)
        public = slots.get(slot, {})
        entry = {**public, **mon, "slot": slot,
                 "species": mon.get("details", "").split(",")[0] or public.get("species", ""),
                 "ability": mon.get("baseAbility", public.get("ability")),
                 "moves": active_req[i].get("moves", []) if i < len(active_req) else []}
        my_actives.append(entry)
    opponents = [slots[k] for k in sorted(slots) if k.startswith(foe)]
    return {"room": room, "turn": turn, "me": user, "side_id": mine,
            "weather": weather, "field": sorted(field),
            "hazards": {"mine": sorted(conditions[mine]), "theirs": sorted(conditions[foe])},
            "my_actives": my_actives, "my_active": next((m for m in my_actives if hp_fraction(m.get("condition")) > 0), None),
            "my_party": [{**m, "species": m.get("details", "").split(",")[0]} for m in party],
            "my_bench": [m for m in party if not m.get("active")],
            "opp_actives": opponents, "opp_active": next((m for m in opponents if hp_fraction(m.get("condition")) > 0), None),
            "opp_revealed": [m for ident, m in revealed.items() if ident.startswith(foe)],
            "opp_preview": preview[foe], "opp_team_sheet": sheets[foe], "history": history[-16:],
            "request_type": "wait" if request.get("wait") else "teamPreview" if request.get("teamPreview") else "forceSwitch" if "forceSwitch" in request else "move"}


def _targets(target: str, slot: int, count: int) -> list[int | None]:
    if count == 1:
        return [None]
    allies = [-(i + 1) for i in range(count) if i != slot]
    if target in ("normal", "any"):
        return list(range(1, count + 1)) + allies
    if target == "adjacentFoe":
        return list(range(1, count + 1))
    if target == "adjacentAlly":
        return allies
    if target == "adjacentAllyOrSelf":
        return [-(i + 1) for i in range(count)]
    return [None]


def legal_choices(request: dict | None) -> list[str]:
    """All choices visible as legal from a singles/doubles request, without pruning.

    Hidden trapping/disabling effects can still cause a server rejection; callers
    must retry using the updated request. Triples and custom multi formats aren't
    supported. A request mask is not a replacement for server team validation.
    """
    if not request or request.get("wait"):
        return []
    party = (request.get("side") or {}).get("pokemon") or []
    if request.get("teamPreview"):
        n = len(party)
        chosen = min(n, int(request.get("maxChosenTeamSize") or n))
        return ["team " + ",".join(map(str, p)) for p in permutations(range(1, n + 1), chosen)]
    active = request.get("active") or []
    forced = request.get("forceSwitch")
    count = len(forced) if forced is not None else len(active)
    if count > 2:
        raise ValueError("only singles and doubles are supported")
    switches = [f"switch {i + 1}" for i, m in enumerate(party)
                if not m.get("active") and hp_fraction(m.get("condition")) > 0]
    slots = []
    for i in range(count):
        if forced is not None:
            # Exactly min(available bench, forced slots) replacements must occur.
            slots.append(switches + (["pass"] if sum(forced) > len(switches) else []) if forced[i] else ["pass"])
            continue
        mon = party[i] if i < len(party) else {}
        a = active[i]
        if hp_fraction(mon.get("condition")) == 0 or a.get("commanding") or mon.get("commanding"):
            slots.append(["pass"])
            continue
        opts = []
        for j, move in enumerate(a.get("moves") or []):
            if move.get("disabled") or (move.get("pp") == 0 and to_id(move.get("id", move.get("move", ""))) != "struggle"):
                continue
            events = [""]
            for flag, event in (("canMegaEvo", "mega"), ("canMegaEvoX", "megax"), ("canMegaEvoY", "megay"), ("canTerastallize", "terastallize")):
                if a.get(flag):
                    events.append(event)
            # Locked two-turn moves and Recharge intentionally omit target in
            # requests: the previous turn's target is reused by the simulator.
            for target, event in product(_targets(move.get("target", "scripted"), i, count), events):
                suffix = (f" {target}" if target is not None else "") + (f" {event}" if event else "")
                opts.append(f"move {j + 1}" + suffix)
        if not a.get("trapped"):
            opts += switches
        slots.append(opts or ["default"])
    result = []
    for combo in product(*slots) if slots else []:
        switched = [s for s in combo if s.startswith("switch ")]
        if len(set(switched)) != len(switched):
            continue
        if forced is not None and len(switched) != min(sum(forced), len(switches)):
            continue
        if sum(any(event in s.split() for event in ("mega", "megax", "megay")) for s in combo) > 1:
            continue
        if sum("terastallize" in s.split() for s in combo) > 1:
            continue
        result.append(", ".join(combo))
    return result
