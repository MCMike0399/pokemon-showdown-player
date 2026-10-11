"""The contract for anything that plays Showdown for our account, and its MCP implementation.

`IPlayer` is the transport: it logs in, builds the team, finds and enters
battles, reads a battle's state and submits choices. It never decides what to
play. Decisions always come from the local model (search agent / Brain), so a
transport can change without changing how the game is played.

Implementations:

- `ml.browser.PlaywrightPlayer`: the official client in Chrome, programmatic
  clicks. The default for everything.
- `McpPlayer` (here): OpenClaw's Pokemon Showdown MCP harness
  (`ps_mcp_server.py`, the same tools OpenClaw loads), driven as an MCP client
  over stdio. Its own websocket connection for the same account. The live runner
  keeps it as a fallback for the edge cases where Chrome can't get a choice to the
  server (docs/players.md).

A snapshot is a dict: room, user {name}, request (the server's latest request
for us, or None), log (battle lines), waiting (our choice for this request is in),
partial, missing (we are not in that room).
"""
from __future__ import annotations

import abc
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class IPlayer(abc.ABC):
    """What a Showdown transport must do for the runner. Choices come from the caller."""

    name = 'player'

    @abc.abstractmethod
    async def login(self, username: str, password: str) -> None:
        """Be logged in as exactly this registered account (never a guest, never another)."""

    @abc.abstractmethod
    async def select_team(self, fmt: str, team: dict) -> None:
        """Make this exact team (`teams.json` entry: name + sets) the one used for `fmt`."""

    @abc.abstractmethod
    async def find_battle(self, fmt: str, team: dict, timeout: float, stop: asyncio.Event | None = None) -> str | None:
        """Ladder search with the selected team; the new battle room, or None on timeout or stop (search cancelled)."""

    @abc.abstractmethod
    async def cancel_search(self) -> None: ...

    @abc.abstractmethod
    async def active_battles(self) -> list[str]:
        """Unfinished battles where we are a player."""

    @abc.abstractmethod
    async def enter_room(self, room: str) -> None:
        """Join a battle room and wait until its state (request or result) is readable."""

    @abc.abstractmethod
    async def snapshot(self, room: str) -> dict:
        """The battle as this transport sees it (module docstring)."""

    @abc.abstractmethod
    async def submit(self, snapshot: dict, choice: str) -> None:
        """Send `choice` (e.g. 'move 1 2, move 3', 'team 4,2,3,6') for the snapshot's request."""

    @abc.abstractmethod
    async def set_timer(self, room: str, on: bool = True) -> None: ...

    @abc.abstractmethod
    async def forfeit(self, room: str) -> None: ...

    @abc.abstractmethod
    async def leave(self, room: str) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


def _ended(log: list[str]) -> bool:
    return any(line.startswith('|win|') or line in ('|tie', '|tie|') for line in log)


class McpPlayer(IPlayer):
    """OpenClaw's MCP harness (`ps_mcp_server.py`) as an `IPlayer`.

The server process is started on first use and keeps one logged-in Showdown
connection. Every action is an MCP tool call: ps_login, ps_join, ps_read,
ps_battle, ps_battles, ps_result, ps_choose, ps_ladder, ps_send.
"""

    name = 'mcp'

    def __init__(self, server: Path = ROOT / 'ps_mcp_server.py', python: str = sys.executable,
                 call_timeout: float = 8.0):
        self.server, self.python, self.call_timeout = Path(server), python, call_timeout
        self.session = None
        self._stack: AsyncExitStack | None = None
        self.user: str | None = None
        self.team_name: str | None = None

    # ------------------------------------------------------------ MCP plumbing
    async def start(self):
        if self.session is not None:
            return
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        stack = AsyncExitStack()
        params = StdioServerParameters(command=self.python, args=[str(self.server)], cwd=str(self.server.parent),
                                       env={**os.environ})
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), self.call_timeout)
        except BaseException:
            await stack.aclose()  # never leave a half-started child behind
            raise
        self._stack, self.session = stack, session

    async def tools(self) -> list[str]:
        await self.start()
        return sorted(t.name for t in (await self.session.list_tools()).tools)

    async def call(self, tool: str, **arguments) -> str:
        await self.start()
        try:
            result = await asyncio.wait_for(self.session.call_tool(tool, arguments), self.call_timeout)
        except Exception:
            await self.close()  # a dead or hung harness is restarted on the next call
            raise
        text = ''.join(getattr(part, 'text', '') for part in result.content)
        if getattr(result, 'isError', False):
            raise RuntimeError(f'{tool} failed: {text[:300]}')
        return text

    async def call_json(self, tool: str, **arguments):
        return json.loads(await self.call(tool, **arguments))

    # ------------------------------------------------------------ IPlayer
    async def login(self, username: str, password: str) -> None:
        # The harness logs in with the account configured in .env; refuse anything else.
        status = await self.call_json('ps_status')
        if not (status.get('loggedIn') and status.get('connected')):
            await self.call('ps_login')
            status = await self.call_json('ps_status')
        user = status.get('user') or ''
        if not status.get('loggedIn') or _id(user) != _id(username):
            raise ValueError('MCP harness is not logged in as the configured account')
        self.user = user

    async def select_team(self, fmt: str, team: dict) -> None:
        # ps_ladder uploads the stored team (/utm) by name right before searching.
        self.team_name = team['name']

    async def find_battle(self, fmt: str, team: dict, timeout: float, stop: asyncio.Event | None = None) -> str | None:
        await self.select_team(fmt, team)
        before = set(await self.active_battles())
        await self.call('ps_ladder', format=fmt, team=self.team_name or '')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not (stop and stop.is_set()):
            new = [r for r in await self.active_battles() if r not in before]
            if new:
                return new[0]
            await asyncio.sleep(1)
        await self.cancel_search()
        return None

    async def cancel_search(self) -> None:
        await self.call('ps_send', room='', message='/cancelsearch')

    async def active_battles(self) -> list[str]:
        rooms = await self.call_json('ps_battles')
        out = []
        for room in rooms:
            if (await self.call_json('ps_result', room=room)).get('ongoing'):
                out.append(room)
        return out

    async def enter_room(self, room: str, timeout: float = 15.0) -> None:
        await self.call('ps_join', room=room)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = await self.snapshot(room)
            if snap.get('request') or _ended(snap['log']):
                return
            await asyncio.sleep(.5)
        raise TimeoutError(f'MCP harness saw no request for {room}')

    async def snapshot(self, room: str) -> dict:
        # The room's raw lines, request included (ps_battle truncates its JSON at
        # 6000 chars, and a team-preview request alone can be bigger).
        text = await self.call('ps_read', room=room, limit=1000)
        lines = [] if text.startswith('(no buffered lines') else text.split('\n')
        request = None
        for line in reversed(lines):
            if line.startswith('|request|'):
                try:
                    request = json.loads(line[len('|request|'):]) or None
                except ValueError:
                    request = None
                break
        log = [line for line in lines if not line.startswith('|request|')]
        return {'room': room, 'user': {'name': self.user, 'named': bool(self.user)}, 'request': request,
                'log': log, 'waiting': bool(request and request.get('wait')), 'partial': False,
                'missing': not log, 'ended': _ended(log), 'transport': self.name}

    async def submit(self, snapshot: dict, choice: str) -> None:
        # With the request id the server refuses the choice if the turn moved on.
        rqid = (snapshot.get('request') or {}).get('rqid')
        reply = await self.call('ps_choose', room=snapshot['room'], choice=f'{choice}|{rqid}' if rqid else choice)
        if reply.strip() != 'sent':
            raise RuntimeError(f'ps_choose did not send: {reply[:200]}')

    async def set_timer(self, room: str, on: bool = True) -> None:
        await self.call('ps_send', room=room, message='/timer ' + ('on' if on else 'off'))

    async def forfeit(self, room: str) -> None:
        await self.call('ps_send', room=room, message='/forfeit')

    async def leave(self, room: str) -> None:
        await self.call('ps_send', room='', message='/leave ' + room)

    async def close(self) -> None:
        stack, self._stack, self.session = self._stack, None, None
        if stack is not None:
            await stack.aclose()


def _id(text: str) -> str:
    return ''.join(ch for ch in str(text or '').lower() if ch.isalnum())
