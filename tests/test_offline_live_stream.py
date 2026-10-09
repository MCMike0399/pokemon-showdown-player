"""Read-only retained stream behavior with temporary campaign recordings."""
import http.client
import json
import sqlite3
import threading
from http.server import ThreadingHTTPServer

import pytest

from scripts import live_watch


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(live_watch, 'PROJECT', tmp_path)
    root = tmp_path / 'data/ml'
    directory = root / 'campaigns/test'
    directory.mkdir(parents=True)
    (root / 'campaigns/latest.json').write_text(json.dumps({
        'directory': str(directory), 'started_at': '2026-10-08T00:00:00Z'}))
    (directory / 'manifest.json').write_text(json.dumps({
        'format': 'test', 'targets': {'ladder_games': 100}}))
    (directory / 'control.json').write_text(json.dumps({'target_ladder': 500}))
    (directory / 'status.json').write_text(json.dumps({
        'phase': 'playing', 'active_room': 'battle-test-2'}))
    (directory / 'games.jsonl').write_text('{"room":"battle-test-1"}\n')
    database = sqlite3.connect(root / 'experience.sqlite3')
    database.execute('CREATE TABLE episodes (status TEXT, outcome REAL, revision TEXT, created TEXT, data TEXT, source TEXT, format TEXT)')
    database.execute("CREATE INDEX episodes_room_side ON episodes (json_extract(data,'$.room'),json_extract(data,'$.side'))")
    database.execute('CREATE TABLE public_battles (id TEXT, log TEXT)')
    for room, state, outcome in [('battle-test-1', 'complete', 1),
                                 ('battle-test-1', 'complete', 1),
                                 ('battle-test-2', 'pending', None),
                                 ('battle-test-3', 'complete', -1)]:
        database.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?,?)', (
            state, outcome, 'revision', '2026-10-08T01:00:00Z',
            json.dumps({'room': room, 'side': 'p2', 'steps': [1, 2]}), 'ladder', 'test'))
    database.commit()
    database.close()
    return directory


def test_status_uses_authorized_target_and_deduplicates_terminal_rooms(campaign):
    status = live_watch.status()
    assert status['target'] == 500
    assert status['completed'] == 1
    assert status['score'] == [1, 0, 0]
    assert status['room'] == 'battle-test-2'
    assert status['side'] == 'p2'
    assert status['choices'] == 2


def test_no_campaign_is_a_waiting_state(tmp_path, monkeypatch):
    monkeypatch.setattr(live_watch, 'PROJECT', tmp_path)
    assert live_watch.status()['phase'] == 'waiting'
    assert live_watch.battle_log()['log'] == []


def test_browser_run_supersedes_stopped_campaign_and_keeps_resumed_outcomes(campaign):
    folder = live_watch.PROJECT / 'data/ml/browser-runs/20261009T120000Z'
    folder.mkdir(parents=True)
    events = [
        {'kind': 'decision', 'time': 1791547200, 'room': 'battle-test-1'},
        {'kind': 'terminal', 'time': 1791547201, 'result': {'room': 'battle-test-1'}},
        {'kind': 'decision', 'time': 1791547202, 'room': 'battle-test-2'},
    ]
    (folder / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n{"partial":')
    info = live_watch.status()
    assert info['campaign'] == folder.name
    assert info['room'] == 'battle-test-2'
    assert info['score'] == [1, 0, 0]
    assert info['choices'] == 2
    assert json.loads((campaign / 'status.json').read_text())['active_room'] == 'battle-test-2'


def test_running_browser_recording_is_filtered_for_viewer(campaign):
    from ml.recording import record_log
    episode = {'room': 'battle-test-2', 'side': 'p2'}
    raw = ['|player|p1|RealOpponent|2', '|player|p2|RealAccount|3',
           '|start', '|turn|4', '|move|p2a: Politoed|Surf|p1a: Pikachu',
           '|c|RealAccount|private chat', '|request|secret-request', '|challstr|secret']
    episode['steps'] = [{'snapshot': {'public_log': record_log(episode, raw)}}]
    db = sqlite3.connect(live_watch.PROJECT / 'data/ml/experience.sqlite3')
    db.execute("UPDATE episodes SET data=? WHERE status='pending'", (json.dumps(episode),))
    db.commit()
    db.close()
    frame = live_watch.battle_log('battle-test-2')
    assert frame['live'] and frame['turn'] == 4 and frame['side'] == 'p2'
    assert '|player|p2|Your agent|1' in frame['log']
    exported = json.dumps(frame)
    for private in ['RealAccount', 'RealOpponent', 'private chat', 'secret-request', 'challstr']:
        assert private not in exported


def test_browser_snapshots_publish_live_and_terminal_frames_atomically(tmp_path):
    from ml.browser import publish_watch
    frame = {'room': 'battle-test-42', 'request': {'side': {'id': 'p2'}, 'private': 'secret'},
             'log': ['|player|p1|OpponentName|1', '|player|p2|AccountName|2', '|start', '|turn|6']}
    publish_watch(tmp_path, frame)
    path = tmp_path / 'live-watch/battle-test-42.json'
    first = json.loads(path.read_text())
    assert first['live'] and first['turn'] == 6
    assert 'secret' not in path.read_text()
    frame['log'].append('|win|AccountName')
    publish_watch(tmp_path, frame)
    terminal = json.loads(path.read_text())
    assert not terminal['live'] and terminal['log'][-1] == '|win|Your agent'
    assert json.loads((path.parent / 'current.json').read_text()) == terminal


def test_latest_state_is_retained_and_slow_consumers_skip_intermediate_states():
    hub = live_watch.StateHub()
    for turn in range(1, 5):
        hub.publish({'status': {'checked_at': str(turn)}, 'battle': {'turn': turn}, 'error': None})
    version, state = hub.snapshot(after=1, timeout=0)
    assert version == 4
    assert state['battle']['turn'] == 4
    hub.publish({**state, 'status': {'checked_at': 'later'}})
    assert hub.version == 4
    hub.publish({**state, 'error': 'unavailable'})
    assert hub.version == 5
    assert hub.snapshot(timeout=0)[1]['battle']['turn'] == 4


def read_event(response):
    lines = []
    while True:
        line = response.readline().decode().strip()
        if not line and lines:
            return '\n'.join(lines)
        if line:
            lines.append(line)


def test_http_stream_initial_reconnect_pin_and_room_validation(monkeypatch):
    hub = live_watch.StateHub()
    payload = {'status': {'score': [2, 3, 0]},
               'battle': {'room': 'battle-test-2', 'log': ['|turn|8'], 'live': True}, 'error': None}
    hub.publish(payload)
    monkeypatch.setattr(live_watch, 'battle_log', lambda room: {
        'room': room, 'log': ['|turn|1', '|win|Your agent'], 'live': False})
    server = ThreadingHTTPServer(('127.0.0.1', 0), live_watch.Handler)
    server.hub = hub
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    clients = []
    try:
        def open_stream(path, headers=None):
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
            clients.append(client)
            client.request('GET', path, headers=headers or {})
            return client.getresponse()
        response = open_stream('/api/events')
        assert response.status == 200
        assert response.getheader('Content-Type').startswith('text/event-stream')
        assert read_event(response) == 'retry: 1500'
        first = read_event(response)
        assert json.loads(first.split('data: ', 1)[1]) == payload
        # Even a matching Last-Event-ID gets a full snapshot after reconnection.
        reconnect = open_stream('/api/events', {'Last-Event-ID': f'{hub.instance}:1'})
        read_event(reconnect)
        assert read_event(reconnect) == first
        pinned = open_stream('/api/events?room=battle-test-1')
        read_event(pinned)
        assert json.loads(read_event(pinned).split('data: ', 1)[1])['battle']['room'] == 'battle-test-1'
        hub.publish({**payload, 'battle': {**payload['battle'], 'log': ['|turn|9']}})
        assert json.loads(read_event(response).split('data: ', 1)[1])['battle']['log'] == ['|turn|9']
        assert open_stream('/api/events?room=../../.env').status == 400
        assert open_stream('/.env').status == 404
    finally:
        hub.stopped.set()
        with hub.condition:
            hub.condition.notify_all()
        for client in clients:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_relay_turn_comes_from_log_when_older_publisher_omits_summary(campaign):
    folder = live_watch.PROJECT / 'data/ml/live-watch'
    folder.mkdir()
    (folder / 'battle-test-2.json').write_text(json.dumps({
        'room': 'battle-test-2', 'turn': None, 'side': 'p2', 'live': True,
        'log': ['|turn|2', '|move|p1a: Pikachu|Thunderbolt', '|turn|8']}))
    assert live_watch.battle_log('battle-test-2')['turn'] == 8
