#!/usr/bin/env python3
"""Private read-only companion for watching a running campaign."""
import json
import re
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
HERE = PROJECT / "web/live-watch"
ROOM = re.compile(r"^battle-[a-z0-9]+-[0-9]+(?:-[a-z0-9]+)?$")


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def viewer_run():
    latest = read(PROJECT / 'data/ml/campaigns/latest.json')
    # Browser runs have their own ledger; never rewrite the stopped campaign.
    for events in sorted((PROJECT / 'data/ml/browser-runs').glob('*/events.jsonl'), reverse=True):
        records, room, first, phase = [], None, None, 'waiting'
        try:
            lines = events.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue  # A writer may still be appending the final line.
            if event.get('kind') == 'decision':
                room = event.get('room')
                if isinstance(room, str) and ROOM.fullmatch(room):
                    records.append({'room': room})
                    first = first or event['time']
                    phase = 'playing'
            elif event.get('kind') == 'terminal':
                room, phase = None, 'waiting'
            elif event.get('kind') == 'stopped':
                room, phase = None, 'stopped'
        if first is None:
            continue
        started = datetime.fromtimestamp(first, timezone.utc).isoformat()
        if latest.get('started_at', '') >= started:
            break
        fmt = records[0]['room'].split('-')[1]
        return ({'directory': str(events.parent), 'started_at': started, 'transport': 'browser'},
                {'active_room': room, 'phase': phase}, {'format': fmt}, {}, records)
    directory = latest.get('directory')
    if not directory:
        return latest, {}, {}, {}, []
    campaign = Path(directory)
    records = []
    try:
        for line in (campaign / 'games.jsonl').read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return latest, read(campaign / 'status.json'), read(campaign / 'manifest.json'), read(campaign / 'control.json'), records


def status():
    latest, state, manifest, control, records = viewer_run()
    directory = latest.get("directory")
    if not directory:
        return {"phase": "waiting", "score": [0, 0, 0], "recent": [], "room": None}
    campaign = Path(directory)
    room = state.get("active_room")
    room = room if isinstance(room, str) and ROOM.fullmatch(room) else None
    known = {row.get("room") for row in records}
    if room:
        known.add(room)
    dbpath = PROJECT / "data/ml/experience.sqlite3"
    database = sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)
    database.row_factory = sqlite3.Row
    try:
        rooms = sorted(known)
        placeholders = ','.join('?' for _ in rooms) or "NULL"
        rows = [dict(row) for row in database.execute(f"""
            SELECT status,outcome,created,json_extract(data,'$.room') room,
                   CASE WHEN json_extract(data,'$.room')=? AND status!='complete'
                        THEN json_array_length(json_extract(data,'$.steps')) ELSE 0 END choices,
                   json_extract(data,'$.side') side
            FROM episodes INDEXED BY episodes_room_side WHERE source='ladder' AND format=?
                AND json_extract(data,'$.room') IN ({placeholders})
                AND julianday(created)>=julianday(?) ORDER BY created
        """, (room, manifest.get("format"), *rooms,
              '1970-01-01T00:00:00Z' if latest.get('transport') == 'browser' else latest.get("started_at")))]
    finally:
        database.close()
    # Reconnects can leave several segments for one room. Count terminal rooms once.
    complete = {}
    for row in rows:
        if row["status"] == "complete" and row["outcome"] in (-1, 0, 1):
            complete[row["room"]] = row
    score = [sum(r["outcome"] == n for r in complete.values()) for n in (1, -1, 0)]
    live = [r for r in rows if r["room"] == room]
    if room in complete:
        room = None
    observer = read(PROJECT / "artifacts/openclaw-monitoring" / campaign.name / "latest.json")
    recent = []
    for index, row in enumerate(reversed(list(complete.values()))):
        recent.append({"game": len(complete) - index, "room": row["room"],
                       "result": "Win" if row["outcome"] == 1 else "Loss" if row["outcome"] == -1 else "Tie"})
        if len(recent) == 8:
            break
    return {"campaign": campaign.name, "room": room, "phase": state.get("phase", "waiting"),
            "score": score, "completed": len(complete),
            "target": control.get("target_ladder", state.get("target_ladder", manifest.get("targets", {}).get("ladder_games", 100))),
            "game": len(complete) + 1, "choices": live[-1]["choices"] if live else 0,
            "side": live[-1]["side"] if live else None,
            "updates": observer.get("actor", {}).get("updates"),
            "scout_samples": observer.get("scout", {}).get("samples"),
            "recent": recent, "checked_at": datetime.now(timezone.utc).isoformat()}


def battle_log(room=None):
    latest, state, _, _, _ = viewer_run()
    room = room or state.get("active_room")
    if room and ROOM.fullmatch(room):
        snapshot = read(PROJECT / "data/ml/live-watch" / (room + ".json"))
        if snapshot:
            return {**snapshot, "turn": next((int(line.split('|')[2]) for line in reversed(snapshot.get("log", []))
                                                if re.fullmatch(r"\|turn\|[0-9]+", line)), 0)}
    info = status()
    room = room or (info["recent"][0]["room"] if info["recent"] else None)
    if not room or not ROOM.fullmatch(room):
        return {"room": None, "log": [], "live": False, "source": "waiting"}
    database = sqlite3.connect((PROJECT / "data/ml/experience.sqlite3").resolve().as_uri() + "?mode=ro", uri=True, timeout=3)
    try:
        # Already running browser processes can be previewed from their durable
        # observations without reconnecting the player or opening another socket.
        recorded = database.execute("SELECT data,status,outcome FROM episodes INDEXED BY episodes_room_side WHERE json_extract(data,'$.room')=? ORDER BY created DESC LIMIT 1", (room,)).fetchone()
        if recorded:
            episode = json.loads(recorded[0])
            steps = episode.get('steps', [])
            captured = steps[-1].get('snapshot', {}) if steps else {}
            reference = episode.get('terminal_log') or captured.get('public_log')
            if reference:
                from ml.recording import log_prefix
                from ml.browser import watch_frame
                frame = watch_frame({'room': room, 'side': episode.get('side'),
                                     'log': log_prefix(episode, reference)})
                frame['live'] = recorded[1] != 'complete'
                if recorded[1] == 'complete':
                    frame['log'] = ['|win|' + ('Your agent' if recorded[2] == 1 else 'Opponent')
                                    if line.startswith('|win|') else line for line in frame['log']]
                    frame['source'] = 'completed-recording'
                return frame
        row = database.execute("SELECT log FROM public_battles WHERE id=?", ("own-" + room,)).fetchone()
        perspective = database.execute("SELECT json_extract(data,'$.side'),outcome FROM episodes WHERE json_extract(data,'$.room')=? AND status='complete' LIMIT 1", (room,)).fetchone()
    finally:
        database.close()
    if not row:
        return {"room": room, "log": [], "live": False, "source": "waiting-for-player-relay"}
    side = perspective[0] if perspective else "p1"
    names = {s: "Your agent" if s == side else "Opponent" for s in ("p1", "p2")}
    lines = []
    for line in json.loads(row[0]):
        parts = line.split("|")
        if len(parts) > 2 and parts[1] == "player":
            line = f"|player|{parts[2]}|{names.get(parts[2], 'Player')}|1"
        elif line.startswith("|win|") and perspective:
            line = "|win|" + ("Your agent" if perspective[1] == 1 else "Opponent")
        lines.append(line)
    return {"room": room, "side": side, "log": lines, "live": False, "source": "completed-recording"}


class StateHub:
    """One read-only publisher; every connection starts with the latest full state.

    Slow consumers skip superseded states rather than accumulating an event queue.
    The player's atomic relay files and SQLite recordings survive viewer restarts.
    """

    def __init__(self):
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.instance = uuid.uuid4().hex
        self.version = 0
        self.payload = {"status": {"phase": "waiting", "score": [0, 0, 0], "recent": []},
                        "battle": {"room": None, "log": [], "live": False}, "error": None}
        self.signature = None

    def publish(self, payload):
        # checked_at is a health timestamp, not a change in campaign state.
        stable = {**payload, "status": {key: value for key, value in payload["status"].items()
                                       if key != "checked_at"}}
        signature = json.dumps(stable, sort_keys=True)
        with self.condition:
            if signature == self.signature:
                return
            self.signature = signature
            self.payload = payload
            self.version += 1
            self.condition.notify_all()

    def snapshot(self, after=None, timeout=15):
        with self.condition:
            if after == self.version and not self.stopped.is_set():
                self.condition.wait_for(lambda: self.version != after or self.stopped.is_set(), timeout)
            return self.version, self.payload

    def run(self):
        next_status = 0
        info = self.payload["status"]
        while not self.stopped.is_set():
            try:
                if time.monotonic() >= next_status:
                    info = status()
                    next_status = time.monotonic() + 2
                # Retain the last game between matches, until the next room starts.
                previous = self.payload["battle"].get("room") if info.get("campaign") == self.payload["status"].get("campaign") else None
                room = info.get("room") or previous
                frame = battle_log(room)
                self.publish({"status": info, "battle": frame, "error": None})
            except (OSError, sqlite3.Error, ValueError, KeyError):
                self.publish({**self.payload, "error": "Campaign state temporarily unavailable"})
            self.stopped.wait(.25)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        route = urlsplit(self.path)
        room = parse_qs(route.query).get("room", [None])[0]
        if route.path in ("/api/battle", "/api/events") and room is not None and not ROOM.fullmatch(room):
            self.send_error(400, "Invalid battle room")
            return
        if route.path == "/api/events":
            return self.stream(room)
        if route.path in ("/api/status", "/api/battle"):
            try:
                payload = status() if route.path == "/api/status" else battle_log(room)
                body = json.dumps(payload).encode()
                content_type, code = "application/json", 200
            except (OSError, sqlite3.Error, ValueError):
                body = b'{"error":"Campaign status is temporarily unavailable"}'
                content_type, code = "application/json", 503
        elif self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            content_type, code = "text/html; charset=utf-8", 200
        elif self.path == "/viewer.js":
            body = (HERE / "viewer.js").read_bytes()
            content_type, code = "text/javascript; charset=utf-8", 200
        elif self.path == "/favicon.ico":
            body, content_type, code = b"", "image/x-icon", 204
        else:
            body, content_type, code = b"Not found", "text/plain", 404
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass  # A viewer may navigate away while an asset/API reply is sent.

    def stream(self, room):
        hub = self.server.hub
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.connection.settimeout(20)
        version = None
        pinned = None
        try:
            self.wfile.write(b"retry: 1500\n\n")
            self.wfile.flush()
            while not hub.stopped.is_set():
                current, payload = hub.snapshot(version)
                if current == version:
                    body = b": heartbeat\n\n"
                else:
                    if room and payload["battle"].get("room") != room:
                        # Completed replays are immutable; avoid repeated SQLite reads.
                        if not pinned or pinned.get("live") or not pinned.get("log"):
                            pinned = battle_log(room)
                        payload = {**payload, "battle": pinned}
                    body = (f"id: {hub.instance}:{current}\nevent: state\ndata: " +
                            json.dumps(payload) + "\n\n").encode()
                self.wfile.write(body)
                self.wfile.flush()
                version = current
        except (OSError, sqlite3.Error, ValueError, KeyError):
            pass  # Browser reconnects with a fresh complete snapshot.

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18489)
    args = parser.parse_args()
    hub = StateHub()
    publisher = threading.Thread(target=hub.run, daemon=True)
    publisher.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.hub = hub
    try:
        server.serve_forever()
    finally:
        hub.stopped.set()
        with hub.condition:
            hub.condition.notify_all()
        server.server_close()
