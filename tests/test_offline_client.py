"""Identity confirmation and packet routing without a network connection."""
import asyncio
from types import SimpleNamespace
import pytest
from websockets.exceptions import ConnectionClosedOK

from ps_client import PSClient


def test_guest_update_revokes_stale_logged_in_identity():
    client = PSClient(username="Player", password="unused")
    client._handle("", "|updateuser|Player|1|1|{}")
    assert client.logged_in
    client._handle("", "|updateuser|Guest 1234|0|1|{}")
    assert not client.logged_in
    assert client.user == "Guest 1234"


def test_global_packet_does_not_inherit_previous_battle_room():
    client = PSClient(username="Player", password="unused")
    room = "battle-gen9randombattle-1"
    class Socket:
        def __aiter__(self):
            async def frames():
                yield ">" + room + "\n|turn|1"
                yield "|error|A global command failed"
            return frames()
    client.ws = Socket()
    asyncio.run(client._read_loop())
    assert client.battles[room]["log"] == ["|turn|1"]
    assert client.global_log[-1] == "global: |error|A global command failed"


def test_login_waits_for_fresh_server_confirmation(monkeypatch):
    client = PSClient(username="Player", password="unused")
    client.connected, client.logged_in, client.user, client.challstr = True, True, "Player", "fixture"
    class Http:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, *args, **kwargs):
            return SimpleNamespace(text=']{"assertion":"fixture-assertion"}')
    monkeypatch.setattr("ps_client.httpx.AsyncClient", lambda **kwargs: Http())
    async def send(message):
        async def response():
            await asyncio.sleep(0.01)
            client._handle("", "|updateuser|Guest 1234|0|1|{}")
        asyncio.create_task(response())
    client.send = send
    result = asyncio.run(client.login())
    assert result["loggedIn"] is False
    assert result["user"] == "Guest 1234"


def test_terminal_null_request_is_not_waiting():
    client = PSClient(username='Player', password='unused')
    client.battles['battle-test'] = {'log': ['|request|null']}
    assert client.current_requests() == {}


def test_reconnect_requires_a_fresh_challenge(monkeypatch):
    client = PSClient(username='Player', password='unused')
    client.challstr, client.user, client.logged_in = 'old', 'Player', True
    class Socket:
        def __aiter__(self):
            async def frames():
                yield '|challstr|new'
                await asyncio.Event().wait()
            return frames()
        async def close(self):
            pass
    async def connect(*args, **kwargs):
        return Socket()
    monkeypatch.setattr('ps_client.websockets.connect', connect)
    async def run():
        assert await client.connect() == 'new'
        assert client.user is None and not client.logged_in
        reader = client.task
        await client.close()
        assert reader.done() and client.task is None
    asyncio.run(run())


def test_closed_action_socket_revokes_login_without_anonymous_reconnect():
    client = PSClient(username='Player', password='unused')
    client.connected = client.logged_in = True
    class Socket:
        async def send(self, message):
            raise ConnectionClosedOK(None, None)
    async def connect():
        pytest.fail('action submission must not reconnect anonymously')
    client.ws, client.connect = Socket(), connect
    with pytest.raises(ConnectionError):
        asyncio.run(client.choose('battle-test', 'move 1'))
    assert not client.connected and not client.logged_in
