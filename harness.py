"""
harness.py - an easy harness to use Pokemon Showdown like a regular player.

Provides:
  * TeamStore  - CRUD teams as JSON, with PS packed-team conversion
  * Player     - ladder/accept battles, read state, decide, play a full game
  * CLI        - python harness.py teams list|add|rm|show ; python harness.py play <format>

Agent-friendly: every decision point is a function you supply.

Team JSON shape:
  {"name": "Rain VGC", "format": "gen9championsvgc2026regmc",
   "sets": [{"name":..,"species":..,"item":..,"ability":..,"moves":[4],"nature":..,
             "evs":{"hp":..,"atk":..},"level":50,"gender":..}]}
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ps_client import PSClient
from battle_state import legal_choices, observe

HERE = Path(__file__).resolve().parent
TEAMS_FILE = HERE / "teams.json"


def load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()


def _pn(name: str) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def pack_set(s: Dict[str, Any]) -> str:
    name = s.get("name") or s.get("species", "")
    species = s.get("species", "")
    moves = s.get("moves", [])
    evs = s.get("evs") or {}
    ev_str = ",".join(str(evs.get(k, "")) for k in ("hp", "atk", "def", "spa", "spd", "spe"))
    if ev_str.replace(",", "") == "":
        ev_str = ""
    parts = [
        name,
        "" if _pn(species) == _pn(name) else _pn(species),
        _pn(s.get("item", "")),
        _pn(s.get("ability", "")),
        ",".join(_pn(m) for m in moves),
        s.get("nature", ""),
        ev_str,
        s.get("gender", ""),
        ",".join("" if (s.get("ivs") or {}).get(k, 31) == 31 else str(s["ivs"][k])
                 for k in ("hp", "atk", "def", "spa", "spd", "spe")) if s.get("ivs") else "",
        "S" if s.get("shiny") else "",
        str(s.get("level", "")) if s.get("level") and s.get("level") != 100 else "",
    ]
    happiness = str(s["happiness"]) if s.get("happiness") is not None and s["happiness"] != 255 else ""
    if s.get("teraType") or s.get("pokeball") or s.get("hpType") or s.get("gigantamax") or s.get("dynamaxLevel", 10) != 10:
        happiness += "," + ",".join((s.get("hpType", ""), _pn(s.get("pokeball", "")),
                                     "G" if s.get("gigantamax") else "",
                                     str(s["dynamaxLevel"]) if s.get("dynamaxLevel", 10) != 10 else "",
                                     s.get("teraType", "")))
    return "|".join(parts + [happiness])


def pack_team(sets: List[Dict[str, Any]]) -> str:
    return "]".join(pack_set(s) for s in sets)


class TeamStore:
    def __init__(self, path: Path = TEAMS_FILE) -> None:
        self.path = path
        if not path.exists():
            path.write_text("{}")

    def _all(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text() or "{}")
        except json.JSONDecodeError:
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2))

    def create(self, name: str, sets: List[Dict[str, Any]], fmt: str = "gen9randombattle") -> Dict[str, Any]:
        data = self._all()
        if name in data:
            raise ValueError("team already exists: " + name)
        data[name] = {"format": fmt, "sets": sets}
        self._write(data)
        return data[name]

    def list(self) -> List[str]:
        return sorted(self._all().keys())

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self._all():
            raise KeyError(name)
        return self._all()[name]

    def update(self, name: str, sets: Optional[List[Dict[str, Any]]] = None,
               fmt: Optional[str] = None, new_name: Optional[str] = None) -> Dict[str, Any]:
        data = self._all()
        if name not in data:
            raise KeyError(name)
        t = data.pop(name)
        if sets is not None:
            t["sets"] = sets
        if fmt is not None:
            t["format"] = fmt
        data[new_name or name] = t
        self._write(data)
        return t

    def delete(self, name: str) -> bool:
        data = self._all()
        if name not in data:
            return False
        data.pop(name)
        self._write(data)
        return True

    def packed(self, name: str) -> str:
        return pack_team(self.get(name)["sets"])


class Player:
    """Plays games: search, read state, decide, choose. One per connection."""

    def __init__(self, client: Optional[PSClient] = None, teams: Optional[TeamStore] = None) -> None:
        self.c = client or PSClient()
        self.teams = teams or TeamStore()

    async def login(self) -> Dict[str, Any]:
        await self.c.connect()
        return await self.c.login()

    async def close(self) -> None:
        await self.c.close()

    async def ladder(self, fmt: str, packed_team: Optional[str] = None) -> str:
        if packed_team:
            await self.c.send("|/utm " + packed_team)
        await self.c.send("|/search " + fmt)
        return "searching: " + fmt

    async def cancel(self) -> str:
        return await self.c.send("|/cancelsearch")

    async def challenge(self, user: str, fmt: str, packed_team: Optional[str] = None) -> str:
        if packed_team:
            await self.c.send("|/utm " + packed_team)
        return await self.c.send("|/challenge " + user + ", " + fmt)

    def battles(self) -> List[str]:
        return sorted(self.c.battles.keys())

    def waiting(self) -> Dict[str, Any]:
        return self.c.current_requests()

    async def wait_for_request(self, timeout: float = 60.0, poll: float = 1.0,
                              exclude: Optional[List[str]] = None) -> Optional[str]:
        """Wait for a battle that needs a choice and is NOT already finished.
        exclude: room ids to ignore (e.g. battles already played this session)."""
        skip = set(exclude or [])
        end = time.time() + timeout
        while time.time() < end:
            for room in self.waiting():
                if room in skip or self.finished(room):
                    continue
                return room
            await asyncio.sleep(poll)
        return None

    def active_battles(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Battle rooms that are unfinished and not excluded."""
        skip = set(exclude or [])
        return [r for r in self.battles() if r not in skip and not self.finished(r)]

    async def forfeit(self, room: str) -> str:
        """Forfeit / leave a battle."""
        return await self.c.send(room + "|/forfeit")

    def _request(self, room: str) -> Optional[dict]:
        log = self.c.battles.get(room, {}).get("log", [])
        reqs = [l for l in log if l.startswith("|request|")]
        if not reqs:
            return None
        try:
            return json.loads(reqs[-1][len("|request|"):])
        except Exception:
            return None

    def legal_choices(self, room: str) -> List[str]:
        return legal_choices(self._request(room))

    async def choose(self, room: str, choice: str) -> str:
        return await self.c.choose(room, choice)

    def finished(self, room: str) -> bool:
        log = self.c.battles.get(room, {}).get("log", [])
        return any(l.startswith("|win|") or l.startswith("|tie|") for l in log)

    def result(self, room: str) -> Dict[str, Any]:
        log = self.c.battles.get(room, {}).get("log", [])
        for l in log:
            if l.startswith("|win|"):
                return {"room": room, "winner": l.split("|", 2)[2]}
            if l.startswith("|tie|"):
                return {"room": room, "winner": None, "tie": True}
        return {"room": room, "winner": None, "ongoing": True}

    async def play(self, room: str, decide: Callable[[dict], str], max_turns: int = 300,
                   verbose: bool = True, on_turn: Optional[Callable[[dict], None]] = None,
                   stall_timeout: float = 120.0, timer: bool = True,
                   forfeit_on_stall: bool = False) -> Dict[str, Any]:
        """Play one battle. decide(ctx) receives a dict with room, turn, choices, request, battle
        and returns a choice string (or None to default to the first legal choice).
        Only acts on a *new* rqid, so it never double-chooses for the same turn."""
        answered: set = set()
        if timer:
            try:
                await self.c.send(room + "|/timer on")
            except Exception:
                pass
        last = time.time()
        for _ in range(max_turns):
            if self.finished(room):
                break
            req = self._request(room)
            rqid = (req or {}).get("rqid")
            if not req or req.get("wait") or rqid in answered:
                if stall_timeout and time.time() - last > stall_timeout:
                    res = self.result(room)
                    res.update({"stalled": True, "waited": round(time.time() - last, 1)})
                    return res
                await asyncio.sleep(1)
                continue
            last = time.time()
            ctx = {
                "room": room,
                "turn": self.c.battle_summary(room).get("turn"),
                "choices": self.legal_choices(room),
                "request": self._request(room),
                "battle": self.c.battle_summary(room),
                "state": self.state_summary(room),
            }
            try:
                choice = decide(ctx) or ctx["choices"][0]
            except Exception as e:
                choice = (ctx["choices"][0] if ctx["choices"] else "default") or "default"
            if not choice:
                choice = "default"
                if verbose:
                    print("  decide error:", e, "; fallback", choice)
            if verbose:
                print("  turn", ctx["turn"], "rq", rqid, ":", choice, " (of", len(ctx["choices"]), "legal)", flush=True)
            answered.add(rqid)
            if on_turn:
                on_turn(ctx)
            await self.choose(room, choice)
            await asyncio.sleep(1.2)
        res = self.result(room)
        if not res.get("winner") and not res.get("tie"):
            res["unfinished"] = True          # battle never produced a winner
            if forfeit_on_stall:
                try:
                    await self.forfeit(room)
                    res["forfeited"] = True
                except Exception:
                    pass
        return res


    # --- agent-friendly snapshot + batch play ---
    def state_summary(self, room: str) -> Dict[str, Any]:
        """Compact, structured snapshot of a battle - what an agent needs to decide,
        without parsing protocol lines."""
        state = observe(self._request(room) or {}, self.c.battles.get(room, {}).get("log", []),
                        self.c.user or "", room)
        state["choices"] = self.legal_choices(room)
        return state

    async def play_batch(self, n: int, decide: Callable[[dict], str],
                         fmt: str = "gen9randombattle",
                         packed_team: Optional[str] = None,
                         verbose: bool = True,
                         forfeit_on_stall: bool = True) -> Dict[str, Any]:
        """Play n fresh ladder games and return a win/loss record."""
        played: List[str] = []
        record: Dict[str, Any] = {"format": fmt, "wins": 0, "losses": 0, "ties": 0,
                                   "unfinished": 0, "games": []}
        for i in range(n):
            await self.ladder(fmt, packed_team)
            room = await self.wait_for_request(timeout=120, exclude=played)
            if not room:
                record["games"].append({"game": i, "error": "no battle found"})
                break
            played.append(room)
            if verbose:
                print("game", i, "battle:", room, flush=True)
            res = await self.play(room, decide, verbose=verbose,
                                  forfeit_on_stall=forfeit_on_stall)
            res.update({"game": i, "turns": self.c.battle_summary(room).get("turn")})
            if res.get("winner") == self.c.user:
                record["wins"] += 1
            elif res.get("winner"):
                record["losses"] += 1
            elif res.get("unfinished"):
                record["unfinished"] += 1       # stalled/aborted, NOT a tie
            else:
                record["ties"] += 1
            record["games"].append(res)
            if verbose:
                print("game", i, "result:", res, flush=True)
            await asyncio.sleep(2)
        return record


def random_decide(ctx: dict) -> str:
    """Example policy: pick a random legal choice."""
    return random.choice(ctx["choices"]) if ctx["choices"] else "move 1"


async def _main(argv: List[str]) -> None:
    store = TeamStore()
    if not argv:
        print(__doc__)
        return
    cmd = argv[0]
    if cmd == "teams":
        sub = argv[1] if len(argv) > 1 else "list"
        if sub == "list":
            print(json.dumps(store.list(), indent=2))
        elif sub == "show":
            print(json.dumps(store.get(argv[2]), indent=2))
        elif sub == "packed":
            print(store.packed(argv[2]))
        elif sub == "rm":
            print("deleted" if store.delete(argv[2]) else "not found")
        elif sub == "add":
            sets = json.loads(Path(argv[3]).read_text()) if len(argv) > 3 else []
            print(json.dumps(store.create(argv[2], sets), indent=2))
    elif cmd == "play":
        fmt = argv[1]
        team = argv[2] if len(argv) > 2 else None
        p = Player()
        print(await p.login())
        print(await p.ladder(fmt, store.packed(team) if team else None))
        room = await p.wait_for_request(timeout=120)
        print("battle:", room)
        if room:
            print(await p.play(room, random_decide))
        await p.close()
    else:
        print(__doc__)


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
