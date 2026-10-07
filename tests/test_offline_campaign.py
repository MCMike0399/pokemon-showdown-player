import asyncio
import json

import pytest

from ml.storage import Store
from scripts import campaign_ladder as runner

FMT = 'gen9championsvgc2026regmc'
ROOM = 'battle-' + FMT + '-test'


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'REPO', tmp_path)
    directory = tmp_path / 'campaign'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({'format': FMT, 'targets': {'ladder_games': 100}}))
    (directory / 'games.jsonl').write_text('')
    (directory / 'control.json').write_text(json.dumps({'target_ladder': 100}))
    (directory / 'status.json').write_text(json.dumps({'active_room': ROOM, 'actual_team': 'Exact'}))
    store = Store(tmp_path / 'data/ml')
    yield directory, store
    store.close()


def test_preflight_preserves_recovery_state_and_rejects_bad_targets(campaign):
    directory, _ = campaign
    controller = runner.Campaign(directory)
    assert controller.previous['active_room'] == ROOM
    assert controller.state['active_room'] == ROOM
    assert json.loads((directory / 'status.json').read_text())['active_room'] == ROOM
    (directory / 'control.json').write_text(json.dumps({'target_ladder': True}))
    with pytest.raises(runner.Blocked, match='integer'):
        controller.target()
    controller.db.close()


def test_nonterminal_attempts_preserve_active_room_without_counting(campaign):
    directory, _ = campaign
    controller = runner.Campaign(directory)
    calls = []
    async def call(session, tool, args, room=None):
        calls.append(tool)
        return {'result': {'unfinished': True}} if tool == 'ps_ml_play' else {'ongoing': True}
    controller.call = call
    with pytest.raises(runner.Blocked, match='unresolved'):
        asyncio.run(controller.play(None, ROOM, 'Exact'))
    assert calls.count('ps_ml_play') == 3
    assert controller.state['active_room'] == ROOM
    assert controller.state['completed'] == 0
    assert (directory / 'games.jsonl').read_text() == ''
    controller.db.close()


def test_count_uses_verified_episode_and_one_based_game_ordinal(campaign):
    directory, store = campaign
    store.save_episode({'id': 'verified', 'room': ROOM, 'format': FMT, 'team': 'exact',
                        'source': 'ladder', 'revision': 'collected', 'status': 'complete',
                        'outcome': 1, 'recorder_version': 2, 'steps': []})
    controller = runner.Campaign(directory)
    async def call(session, tool, args, room=None):
        return {'winner': 'Guest'} if tool == 'ps_result' else {'experience': {'id': 'verified'}}
    controller.call = call
    asyncio.run(controller.play(None, ROOM, 'Exact'))
    record = json.loads((directory / 'games.jsonl').read_text())
    assert record['game'] == 1
    assert record['episode_id'] == 'verified'
    assert controller.state['w'] == 1
    assert controller.state['completed'] == 1
    assert controller.state['active_room'] is None
    controller.db.close()


def test_blocker_reporting_exits_cleanly_and_preserves_room(campaign, monkeypatch):
    directory, _ = campaign
    async def blocked(self):
        raise runner.Blocked('test blocker')
    monkeypatch.setattr(runner.Campaign, 'run', blocked)
    asyncio.run(runner.main(directory))
    status = json.loads((directory / 'status.json').read_text())
    assert status['phase'] == 'blocked'
    assert status['active_room'] == ROOM
    assert json.loads((directory / 'events.jsonl').read_text())['reason'] == 'test blocker'


def test_source_reload_happens_before_search_at_empty_boundary(campaign, monkeypatch):
    from contextlib import asynccontextmanager
    directory, _ = campaign
    manifest = json.loads((directory / 'manifest.json').read_text())
    manifest['deadline_at'] = '2099-01-01T00:00:00Z'
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    (directory / 'status.json').write_text(json.dumps({'active_room': None}))
    controller = runner.Campaign(directory)
    generations = iter(['old', 'new'])
    monkeypatch.setattr(runner, 'source_generation', lambda: next(generations))
    @asynccontextmanager
    async def stdio(parameters):
        yield None, None
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def initialize(self):
            pass
    monkeypatch.setattr(runner, 'stdio_client', stdio)
    monkeypatch.setattr(runner, 'ClientSession', Session)
    calls = []
    async def call(session, tool, args, room=None):
        calls.append(tool)
        return {'loggedIn': True, 'user': 'Configured', 'configuredUser': 'Configured'}
    controller.call = call
    assert asyncio.run(controller.run()) == 75
    assert 'ps_ml_ladder' not in calls
    assert controller.state['active_room'] is None
    assert controller.state['phase'] == 'source_reload_requested'
    controller.db.close()
