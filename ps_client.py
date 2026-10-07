"""
ps_client.py - Minimal async Pokemon Showdown client over the raw SockJS WebSocket.

Endpoint: wss://sim3.psim.us/showdown/websocket
Client->server messages: "ROOMID|TEXT" (ROOMID may be blank for global commands)
Server->client: ">ROOMID\n|TYPE|DATA..." (">ROOMID" omitted for lobby)

Docs: https://github.com/smogon/pokemon-showdown/blob/master/PROTOCOL.md
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import httpx
import websockets

WS_URL = os.environ.get("PS_WS_URL", "wss://sim3.psim.us/showdown/websocket")
LOGIN_URL = os.environ.get("PS_LOGIN_URL", "https://play.pokemonshowdown.com/api/login")
SERVER = "https://play.pokemonshowdown.com"

HEADERS = {
    "Origin": SERVER,
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_LINES_PER_ROOM = 500


class PSClient:
    """A small, inspectable Showdown client. One instance == one connection."""

    def __init__(self, username: str | None = None, password: str | None = None) -> None:
        self.username = username or os.environ.get("PS_USERNAME")
        self.password = password or os.environ.get("PS_PASSWORD")
        self.ws: Optional[Any] = None
        self.task: Optional[asyncio.Task] = None
        self.challstr: Optional[str] = None
        self.user: Optional[str] = None
        self.logged_in = False
        self.rooms: Dict[str, Deque[str]] = {}
        self.global_log: Deque[str] = deque(maxlen=400)
        self.battles: Dict[str, Dict[str, Any]] = {}
        self.connected = False
        self.last_error: Optional[str] = None
        self._ready = asyncio.Event()

    # ---------------- connection ----------------
    async def connect(self, timeout: float = 15.0) -> str:
        if self.connected:
            return self.challstr or ""
        self._ready.clear()
        self.ws = await websockets.connect(
            WS_URL, additional_headers=HEADERS, ping_interval=25, max_size=2 ** 24
        )
        self.connected = True
        self.task = asyncio.create_task(self._read_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return self.challstr or ""

    async def _read_loop(self) -> None:
        room = ""
        try:
            async for raw in self.ws:  # type: ignore[union-attr]
                for line in str(raw).split("\n"):
                    if line.startswith(">"):
                        room = line[1:].strip()
                        if room and room not in self.rooms:
                            self.rooms[room] = deque(maxlen=MAX_LINES_PER_ROOM)
                        continue
                    if not line:
                        continue
                    self._store(room, line)
                    self._handle(room, line)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
        finally:
            self.connected = False
            self.logged_in = False

    def _store(self, room: str, line: str) -> None:
        self.global_log.append(f"{room or 'global'}: {line}")
        if room:
            self.rooms.setdefault(room, deque(maxlen=MAX_LINES_PER_ROOM)).append(line)

    def _handle(self, room: str, line: str) -> None:
        if line.startswith("|challstr|"):
            self.challstr = line[len("|challstr|"):]
            self._ready.set()
        elif line.startswith("|updateuser|"):
            parts = line.split("|")
            # |updateuser|name|named|avatar|settings
            name = parts[2] if len(parts) > 2 else ""
            if name and not name.startswith("Guest"):
                self.user = name.strip()
                self.logged_in = True
        elif line.startswith("|init|battle") and room:
            self.battles.setdefault(room, {})["started"] = True
        elif line.startswith("|title|") and room:
            self.battles.setdefault(room, {})["title"] = line[len("|title|"):]
        if room.startswith("battle-"):
            self.battles.setdefault(room, {}).setdefault("log", []).append(line)

    async def login(self, username: str | None = None, password: str | None = None) -> Dict[str, Any]:
        username = username or self.username
        password = password or self.password
        if not username or not password:
            raise RuntimeError("missing PS username/password (set PS_USERNAME / PS_PASSWORD)")
        if not self.connected:
            await self.connect()
        if not self.challstr:
            raise RuntimeError("no challstr received from server")
        async with httpx.AsyncClient(timeout=20) as c:
            resp = await c.post(
                LOGIN_URL,
                data={"name": username, "pass": password, "challstr": self.challstr},
            )
        body = resp.text
        if not body.startswith("]"):
            raise RuntimeError(f"login rejected: {body[:200]}")
        assertion = json.loads(body[1:]).get("assertion")
        if not assertion:
            raise RuntimeError(f"no assertion in login response: {body[:200]}")
        await self.send(f"|/trn {username},0,{assertion}")
        for _ in range(80):
            if self.user and not self.user.startswith("Guest"):
                break
            await asyncio.sleep(0.25)
        ok = bool(self.user and not self.user.startswith("Guest"))
        self.logged_in = ok
        return {"loggedIn": ok, "user": self.user, "challstr": bool(self.challstr)}

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            self.task = None
        if self.ws:
            await self.ws.close()
        self.connected = False
        self.logged_in = False

    # ---------------- io ----------------
    async def send(self, message: str) -> str:
        """Send a raw client->server message ('ROOMID|TEXT' or '|/cmd')."""
        if not self.connected or not self.ws:
            await self.connect()
        await self.ws.send(message)  # type: ignore[union-attr]
        return "sent"

    async def chat(self, room: str, text: str) -> str:
        return await self.send(f"{room}|{text}")

    async def join(self, room: str) -> str:
        return await self.send(f"|/join {room}")

    async def read(self, room: str, limit: int = 40) -> List[str]:
        if room in ("global", "", "lobby"):
            return list(self.global_log)[-limit:]
        return list(self.rooms.get(room, []))[-limit:]

    # ---------------- battles ----------------
    async def choose(self, room: str, choice: str) -> str:
        """choice examples: 'move 1', 'move 1, move 2', 'switch 2', 'team 123456'."""
        return await self.send(f"{room}|/choose {choice}")

    def battle_summary(self, room: str) -> Dict[str, Any]:
        log = self.battles.get(room, {}).get("log", [])
        players, hp, turn, weather, active = {}, {}, None, None, {}
        for line in log:
            if line.startswith("|player|"):
                p = line.split("|")
                if len(p) > 3:
                    players[p[2]] = p[3]
            elif line.startswith("|turn|"):
                turn = line.split("|")[2] if len(line.split("|")) > 2 else None
            elif line.startswith("|-weather|"):
                weather = line.split("|")[2] if len(line.split("|")) > 2 else None
            elif line.startswith("|switch|") or line.startswith("|drag|"):
                p = line.split("|")
                if len(p) > 3:
                    active[p[3].split(":")[0].strip()] = p[2]
            elif line.startswith("|request|"):
                pass
        requests = [l for l in log if l.startswith("|request|")]
        req = None
        if requests:
            try:
                req = json.loads(requests[-1][len("|request|"):])
            except Exception:  # noqa: BLE001
                req = None
        return {
            "room": room,
            "title": self.battles.get(room, {}).get("title"),
            "players": players,
            "turn": turn,
            "weather": weather,
            "active": active,
            "request": req,
            "lines": len(log),
        }

    def current_requests(self) -> Dict[str, Any]:
        """Which battles are waiting on us to choose."""
        waiting = {}
        for room in list(self.battles):
            reqs = [l for l in self.battles[room].get("log", []) if l.startswith("|request|")]
            if not reqs:
                continue
            try:
                r = json.loads(reqs[-1][len("|request|"):])
            except Exception:  # noqa: BLE001
                continue
            if r.get("wait"):
                continue
            if r.get("teamPreview") or r.get("active") or r.get("forceSwitch"):
                waiting[room] = {"rqid": r.get("rqid"), "preview": bool(r.get("teamPreview")),
                                 "forceSwitch": bool(r.get("forceSwitch")),
                                 "active": [a.get("moves") for a in (r.get("active") or [])]}
        return waiting
