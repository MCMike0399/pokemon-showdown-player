"""The IPlayer contract, the MCP harness player, and Chrome -> MCP fallback in the game loop."""
import asyncio
import json
from pathlib import Path

import pytest

import ml.browser as browser
from ml.browser import BrowserPlayer, PlaywrightPlayer
from ml.players import IPlayer, McpPlayer
from tests.test_offline_browser import FMT, ROOM, Page, _lost_choice_player, snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_both_transports_implement_the_whole_contract():
    abstract = set(IPlayer.__abstractmethods__)
    assert {'login', 'select_team', 'find_battle', 'enter_room', 'snapshot', 'submit', 'set_timer',
            'forfeit', 'leave', 'close', 'active_battles', 'cancel_search'} <= abstract
    assert BrowserPlayer is PlaywrightPlayer and issubclass(PlaywrightPlayer, IPlayer) and issubclass(McpPlayer, IPlayer)
    PlaywrightPlayer(Page(snapshot()))  # instantiable: nothing abstract left
    McpPlayer()


class FakeHarness(McpPlayer):
    """McpPlayer with the MCP tool calls answered locally."""

    def __init__(self, replies):
        super().__init__()
        self.replies, self.calls = replies, []

    async def call(self, tool, **arguments):
        self.calls.append((tool, arguments))
        reply = self.replies.get(tool, 'sent')
        return reply(arguments) if callable(reply) else reply


def test_mcp_player_sends_choices_with_the_request_id_and_refuses_other_accounts():
    harness = FakeHarness({'ps_status': json.dumps({'connected': True, 'loggedIn': True, 'user': 'Configured'})})
    asyncio.run(harness.login('configured', None))
    asyncio.run(harness.submit({'room': ROOM, 'request': {'rqid': 7}}, 'team 4,2,3,6'))
    assert harness.calls[-1] == ('ps_choose', {'room': ROOM, 'choice': 'team 4,2,3,6|7'})
    other = FakeHarness({'ps_status': json.dumps({'connected': True, 'loggedIn': True, 'user': 'Someone'})})
    with pytest.raises(ValueError, match='configured account'):
        asyncio.run(other.login('configured', None))


def test_mcp_player_snapshot_and_find_battle():
    battles = {'n': 0}

    def ps_battles(_):
        battles['n'] += 1
        return json.dumps([] if battles['n'] < 3 else [ROOM])
    # A team-preview request alone can exceed ps_battle's 6000-char cut: read the raw lines.
    big = {'rqid': 3, 'teamPreview': True, 'side': {'pokemon': [{'details': 'x' * 1500}] * 6}}
    assert len(json.dumps(big)) > 6000
    harness = FakeHarness({
        'ps_read': '|player|p1|Configured\n|request|' + json.dumps(big) + '\n|teampreview|4',
        'ps_battles': ps_battles,
        'ps_result': json.dumps({'room': ROOM, 'winner': None, 'ongoing': True}),
    })
    snap = asyncio.run(harness.snapshot(ROOM))
    assert snap['request'] == big and snap['log'] == ['|player|p1|Configured', '|teampreview|4'] and not snap['ended']
    assert not any(tool == 'ps_battle' for tool, _ in harness.calls)
    assert asyncio.run(harness.find_battle(FMT, {'name': 'Rain', 'sets': []}, timeout=5)) == ROOM
    assert ('ps_ladder', {'format': FMT, 'team': 'Rain'}) in harness.calls


def test_real_mcp_harness_exposes_every_tool_the_player_calls():
    pytest.importorskip('mcp')
    harness = McpPlayer()

    async def tools():
        try:
            return await harness.tools()
        finally:
            await harness.close()
    names = set(asyncio.run(tools()))
    assert {'ps_status', 'ps_login', 'ps_join', 'ps_read', 'ps_battle', 'ps_battles', 'ps_result',
            'ps_choose', 'ps_ladder', 'ps_send'} <= names


class RecordingFallback(IPlayer):
    name = 'fake-mcp'

    def __init__(self, server_rqid=4, on_submit=None):
        self.server_rqid, self.on_submit, self.sent, self.logins = server_rqid, on_submit, [], 0

    async def login(self, username, password): self.logins += 1
    async def select_team(self, fmt, team): pass
    async def find_battle(self, fmt, team, timeout): return None
    async def cancel_search(self): pass
    async def active_battles(self): return []
    async def enter_room(self, room): pass

    async def snapshot(self, room):
        return {'room': room, 'request': {'rqid': self.server_rqid}, 'log': [], 'ended': False}

    async def submit(self, snap, choice):
        self.sent.append((snap['request']['rqid'], choice))
        if self.on_submit:
            self.on_submit()
    async def set_timer(self, room, on=True): pass
    async def forfeit(self, room): pass
    async def leave(self, room): pass
    async def close(self): pass


def test_lost_choice_goes_through_the_fallback_when_chrome_resend_fails(tmp_path, monkeypatch):
    # Attempt 1: Chrome re-sends (lost again). Attempt 2: the MCP harness delivers it.
    store, page, player, calls, decided = _lost_choice_player(tmp_path, monkeypatch, snapshot(preview=True))
    try:
        fallback = RecordingFallback(on_submit=lambda: page.state['log'].append('|win|Configured'))
        player.fallback = fallback

        async def submit(state, choice):
            page.state['waiting'] = True
        player.submit = submit
        original = page.evaluate
        resends = []

        async def evaluate(script, room=None):
            if 'r.send(a.msg)' in script:  # Chrome's re-send never arrives
                resends.append(room)
                page.state['log'].append('|inactive|Configured has 20 seconds left.')
                return None
            if script == browser.SNAPSHOT and page.state['waiting'] and not any('left' in l for l in page.state['log']):
                page.state['log'].append('|inactive|Configured has 60 seconds left.')
            return await original(script, room)
        page.evaluate = evaluate
        asyncio.run(player.play(ROOM, FMT, record=False, timeout=5))
        assert fallback.sent == [(4, decided[0])] and fallback.logins == 1
        assert len(resends) == 1  # Chrome tried once first
    finally:
        store.close()


def test_repeated_chrome_submit_failures_use_the_fallback(tmp_path, monkeypatch):
    store, page, player, calls, decided = _lost_choice_player(tmp_path, monkeypatch, snapshot())
    try:
        fallback = RecordingFallback(on_submit=lambda: page.state['log'].append('|win|Configured'))
        player.fallback = fallback

        async def submit(state, choice):
            raise TimeoutError('Locator.click: Timeout 90000ms exceeded.')
        player.submit = submit
        asyncio.run(player.play(ROOM, FMT, record=False, timeout=5))
        assert len(decided) == 3 and fallback.sent == [(4, decided[-1])]  # third failure -> MCP
    finally:
        store.close()


def test_fallback_never_sends_for_a_request_the_server_moved_past(tmp_path, monkeypatch):
    store, page, player, calls, decided = _lost_choice_player(tmp_path, monkeypatch, snapshot())
    try:
        fallback = RecordingFallback(server_rqid=5)
        player.fallback = fallback
        assert asyncio.run(player.fallback_submit(ROOM, 4, 'move 1 1, move 1', 'test')) is False
        assert fallback.sent == []
    finally:
        store.close()


def test_drill_fallback_mode_blocks_chrome_resend_so_only_mcp_can_deliver(tmp_path, monkeypatch):
    drill = tmp_path / 'drill'
    drill.write_text('fallback')
    monkeypatch.setattr(browser, 'LOST_CHOICE_DRILL', drill)
    store, page, player, calls, decided = _lost_choice_player(tmp_path, monkeypatch, snapshot(preview=True))
    try:
        fallback = RecordingFallback(on_submit=lambda: page.state['log'].append('|win|Configured'))
        player.fallback = fallback
        original = page.evaluate
        warned = {'n': 0}

        async def evaluate(script, room=None):
            if 'r.choice = {waiting: true}' in script:
                page.state['waiting'] = True
            if script == browser.SNAPSHOT and page.state['waiting']:
                warned['n'] += 1
                page.state['log'].append(f"|inactive|Configured has {60 - warned['n']} seconds left.")
            return await original(script, room)
        page.evaluate = evaluate
        asyncio.run(player.play(ROOM, FMT, record=False, timeout=5))
        assert not drill.exists() and fallback.sent == [(4, decided[0])]
        assert not [s for s, _ in calls if 'r.send(a.msg)' in s]  # Chrome's re-send was suppressed
    finally:
        store.close()


def test_a_drill_applies_to_its_own_game_only(tmp_path, monkeypatch):
    store, page, player, calls, decided = _lost_choice_player(tmp_path, monkeypatch, snapshot())
    try:
        player._drill_mode, player._drill_rqid = 'fallback', 4  # left over from an earlier game

        async def submit(state, choice):
            page.state['log'].append('|win|Configured')
        player.submit = submit
        asyncio.run(player.play(ROOM, FMT, record=False, timeout=5))
        assert player._drill_mode is None and not player._drill_blocks_resend(4)
    finally:
        store.close()
