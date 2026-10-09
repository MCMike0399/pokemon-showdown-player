#!/usr/bin/env python3
"""Replay a recorded browser game into the live viewer at its real turn pace.

Reproduces live playback (append deltas, turn timing, terminal frame) without a
player, account or Showdown connection. Serves the real index.html/viewer.js and
the same SSE code as scripts/live_watch.py, on a separate local port:

    .venv/bin/python scripts/live_watch_sim.py --events data/ml/browser-runs/<run>/events.jsonl
    open http://127.0.0.1:18490/?debug=1

Frame and client diagnostics go to --logs (default artifacts/live-watch-sim/).
"""
import argparse
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_watch  # noqa: E402
from ml.browser import watch_frame  # noqa: E402


def recorded_games(path):
    """Terminal browser logs (full client stepQueue) keyed by room, oldest first."""
    games = {}
    for line in Path(path).read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') == 'terminal' and event.get('log'):
            games[(event.get('result') or {}).get('room') or 'battle-sim-%d' % len(games)] = event
    return games


def segments(log):
    """Split a log into server updates; each update starts at a |t:| timestamp."""
    chunks, stamps = [[]], [None]
    for line in log:
        if line.startswith('|t:|') and chunks[-1]:
            chunks.append([])
            stamps.append(int(line[4:]))
        elif line.startswith('|t:|') and stamps[-1] is None:
            stamps[-1] = int(line[4:])
        chunks[-1].append(line)
    return list(zip(stamps, chunks))


class SimHub(live_watch.StateHub):
    def __init__(self, room, side, log, speed, max_gap, loop):
        super().__init__()
        self.room, self.side, self.log = room, side, log
        self.speed, self.max_gap, self.loop = speed, max_gap, loop

    def frame(self, lines, live):
        frame = watch_frame({'room': self.room, 'side': self.side, 'log': lines})
        frame['live'] = live
        return frame

    def status(self, live, game):
        return {'campaign': 'sim', 'room': self.room if live else None, 'phase': 'running',
                'feed_stale': False, 'score': [0, 0, 0], 'completed': 0, 'game': game,
                'choices': 0, 'side': self.side, 'updates': None, 'scout_samples': None,
                'recent': [], 'checked_at': time.time()}

    def run(self):
        game = 1
        while not self.stopped.is_set():
            shown, previous = [], None
            parts = segments(self.log)
            for index, (stamp, chunk) in enumerate(parts):
                if previous is not None and stamp is not None:
                    self.stopped.wait(min((stamp - previous) / self.speed, self.max_gap))
                previous = stamp if stamp is not None else previous
                shown += chunk
                live = index < len(parts) - 1
                self.publish({'status': self.status(live, game), 'battle': self.frame(shown, live),
                              'error': None})
                if self.stopped.is_set():
                    return
            if not self.loop:
                return
            self.stopped.wait(8)
            game += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--events', required=True, help='browser-runs/<run>/events.jsonl')
    parser.add_argument('--room', help='recorded room (default: longest game in the file)')
    parser.add_argument('--port', type=int, default=18490)
    parser.add_argument('--speed', type=float, default=1.0, help='divide recorded turn gaps by this')
    parser.add_argument('--max-gap', type=float, default=25.0, help='cap any single wait (seconds)')
    parser.add_argument('--loop', action='store_true', help='restart the game after it ends')
    parser.add_argument('--logs', type=Path, default=live_watch.PROJECT / 'artifacts/live-watch-sim')
    args = parser.parse_args()

    games = recorded_games(args.events)
    room = args.room or max(games, key=lambda key: len(games[key]['log']))
    event = games[room]
    side = event.get('side') or next((line.split('|')[2] for line in event['log']
                                      if line.startswith('|player|')), 'p1')
    sim_room = room if live_watch.ROOM.fullmatch(room or '') else 'battle-gen9sim-1'
    hub = SimHub(sim_room, side, event['log'], args.speed, args.max_gap, args.loop)
    hub.frames = live_watch.TriageLog(args.logs / 'frames.jsonl')
    threading.Thread(target=hub.run, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), live_watch.Handler)
    server.hub = hub
    server.client_log = live_watch.TriageLog(args.logs / 'client.jsonl')
    print(f'Replaying {sim_room} ({len(segments(event["log"]))} updates) on http://127.0.0.1:{args.port}/',
          flush=True)
    try:
        server.serve_forever()
    finally:
        hub.stopped.set()
        with hub.condition:
            hub.condition.notify_all()
        server.server_close()


if __name__ == '__main__':
    main()
