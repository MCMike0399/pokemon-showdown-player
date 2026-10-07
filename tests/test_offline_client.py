"""Identity confirmation and packet routing without a network connection."""
import asyncio
from types import SimpleNamespace

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
