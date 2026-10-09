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


def test_search_run_tracks_verified_games_across_restarts_without_ppo_episodes(campaign):
    root = live_watch.PROJECT / 'data/ml/browser-runs'
    for number, (stamp, room, winner) in enumerate([('20261009T120000Z', 'battle-test-41', 'Agent'),
                                ('20261009T120100Z', 'battle-test-42', 'Rival')]):
        folder = root / stamp
        folder.mkdir(parents=True)
        events = [
            {'kind': 'search_agent', 'time': 1791547200},
            {'kind': 'decision', 'time': 1791547201, 'room': room,
             'request': {'side': {'id': 'p2'}}},
            {'kind': 'terminal', 'time': 1791547202 + number, 'result': {'room': room, 'winner': winner},
             'log': ['|player|p1|Rival|1', '|player|p2|Agent|1', '|win|' + winner]},
        ]
        (folder / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n')
    folder = root / '20261009T120200Z'
    folder.mkdir()
    (folder / 'events.jsonl').write_text('\n'.join(map(json.dumps, [
        {'kind': 'search_agent', 'time': 1791547203},
        {'kind': 'decision', 'time': 1791547204, 'room': 'battle-test-43',
         'request': {'side': {'id': 'p2'}}},
        {'kind': 'decision', 'time': 1791547205, 'room': 'battle-test-43',
         'request': {'side': {'id': 'p2'}}},
    ])))
    info = live_watch.status()
    assert info['score'] == [1, 1, 0]
    assert info['completed'] == 2
    assert info['game'] == 3
    assert info['choices'] == 2 and info['side'] == 'p2'
    assert [row['result'] for row in info['recent']] == ['Loss', 'Win']
    replay = live_watch.battle_log('battle-test-42')
    assert not replay['live'] and replay['log'][-1] == '|win|Opponent'


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


def test_terminal_frame_keeps_our_perspective_after_client_clears_request():
    from ml.browser import watch_frame
    frame = watch_frame({'room': 'battle-test-42', 'user': {'name': 'Agent'},
                         'request': None, 'log': ['|player|p1|Rival|1',
                                                  '|player|p2|Agent|2', '|win|Agent']})
    assert frame['side'] == 'p2'
    assert not frame['live'] and frame['log'][-1] == '|win|Your agent'


def test_browser_observer_publishes_while_decider_is_waiting(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ml.browser import BrowserPlayer

    state = {'room': 'battle-test-42', 'side': 'p2', 'log': ['|start', '|turn|1']}

    class Page:
        async def evaluate(self, script, room):
            return state

    player = BrowserPlayer(Page(), SimpleNamespace(store=SimpleNamespace(root=tmp_path)))

    async def slow_decision(*args, **kwargs):
        # The normal decision loop has no opportunity to request a snapshot here.
        await asyncio.sleep(.05)
        state['log'].append('|turn|2')
        await asyncio.sleep(.25)
        assert json.loads((tmp_path / 'live-watch/battle-test-42.json').read_text())['turn'] == 2

    player._play = slow_decision
    asyncio.run(player.play(state['room'], 'test'))
    assert json.loads((tmp_path / 'live-watch/health.json').read_text())['room'] == state['room']


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


def test_stream_deltas_only_use_a_matching_prefix_and_reconnect_starts_full():
    before = {'status': {}, 'battle': {'room': 'battle-test-1', 'side': 'p2',
                                      'log': ['|start', '|turn|1']}}
    after = {**before, 'battle': {**before['battle'], 'log': before['battle']['log'] + ['|turn|2']}}
    assert live_watch.stream_payload(after, None) == after
    delta = live_watch.stream_payload(after, before)['battle']
    assert delta['log_start'] == 2 and delta['log'] == ['|turn|2']
    assert live_watch.stream_payload(after, after)['battle']['log'] == []
    rewrite = {**after, 'battle': {**after['battle'], 'log': ['|turn|3']}}
    assert live_watch.stream_payload(rewrite, after) == rewrite
    switched = {**after, 'battle': {**after['battle'], 'room': 'battle-test-2'}}
    assert live_watch.stream_payload(switched, before) == switched


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
        delta_client = open_stream('/api/events?deltas=1')
        read_event(delta_client)
        assert json.loads(read_event(delta_client).split('data: ', 1)[1]) == payload
        appended = {**payload, 'battle': {**payload['battle'], 'log': ['|turn|8', '|turn|9']}}
        hub.publish(appended)
        assert json.loads(read_event(response).split('data: ', 1)[1]) == appended
        delta = json.loads(read_event(delta_client).split('data: ', 1)[1])['battle']
        assert delta['log_start'] == 1 and delta['log'] == ['|turn|9']
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


def test_frame_log_classifies_appends_rewrites_and_new_rooms(tmp_path):
    hub = live_watch.StateHub()
    hub.frames = live_watch.TriageLog(tmp_path / 'frames.jsonl')
    battle = {'room': 'battle-test-1', 'log': ['|turn|1'], 'live': True, 'turn': 1}
    for frame in (battle, {**battle, 'log': ['|turn|1', '|turn|2']}, {**battle, 'log': ['|turn|1', '|turn|3']},
                  {**battle, 'room': 'battle-test-2'}):
        hub.publish({'status': {'score': [0, 0, 0]}, 'battle': frame, 'error': None})
    # Score-only changes do not log a frame.
    hub.publish({'status': {'score': [1, 0, 0]}, 'battle': {**battle, 'room': 'battle-test-2'}, 'error': None})
    records = [json.loads(line) for line in (tmp_path / 'frames.jsonl').read_text().splitlines()]
    assert [record['kind'] for record in records] == ['room', 'append', 'rewrite', 'room']
    assert records[1]['appended'] == 1
    assert records[2]['first_diff'] == 1 and records[2]['was'] == '|turn|2' and records[2]['now'] == '|turn|3'


def test_triage_log_rotates_one_generation(tmp_path):
    log = live_watch.TriageLog(tmp_path / 'client.jsonl', limit=200)
    for index in range(10):
        log.write({'event': 'frame', 'index': index})
    assert (tmp_path / 'client.1.jsonl').exists()
    assert (tmp_path / 'client.jsonl').stat().st_size <= 200


def test_viewer_diagnostics_are_accepted_and_debug_query_serves_the_page(tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), live_watch.Handler)
    server.hub = live_watch.StateHub()
    server.client_log = live_watch.TriageLog(tmp_path / 'client.jsonl')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def call(method, path, body=None):
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
            client.request(method, path, body=body, headers={'Content-Type': 'application/json'})
            response = client.getresponse()
            response.read()
            client.close()
            return response.status
        report = {'session': 'abc', 'events': [{'t': 5, 'event': 'seek', 'reason': 'backlog'}, 'junk']}
        assert call('POST', '/api/diag', json.dumps(report)) == 204
        assert call('POST', '/api/diag', 'not json') == 400
        assert call('POST', '/api/other', '{}') == 404
        assert call('POST', '/api/diag', 'x' * 70000) == 413
        assert call('GET', '/?debug=1') == 200
        records = [json.loads(line) for line in (tmp_path / 'client.jsonl').read_text().splitlines()]
        assert len(records) == 1 and records[0]['session'] == 'abc' and records[0]['reason'] == 'backlog'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
