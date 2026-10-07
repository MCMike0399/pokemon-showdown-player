#!/usr/bin/env python3
"""Private read-only companion for watching a running campaign."""
import json
import re
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "web/live-watch"
ROOM = re.compile(r"^battle-[a-z0-9]+-[0-9]+(?:-[a-z0-9]+)?$")


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def status():
    latest = read(PROJECT / "data/ml/campaigns/latest.json")
    directory = latest.get("directory")
    if not directory:
        return {"phase": "waiting", "score": [0, 0, 0], "recent": [], "room": None}
    campaign = Path(directory)
    state = read(campaign / "status.json")
    manifest = read(campaign / "manifest.json")
    records = []
    try:
        for line in (campaign / "games.jsonl").read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    room = state.get("active_room")
    room = room if isinstance(room, str) and ROOM.fullmatch(room) else None
    known = {row.get("room") for row in records}
    if room:
        known.add(room)
    dbpath = PROJECT / "data/ml/experience.sqlite3"
    database = sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)
    database.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in database.execute("""
            SELECT status,outcome,revision,created,json_extract(data,'$.room') room,
                   json_array_length(json_extract(data,'$.steps')) choices,
                   json_extract(data,'$.side') side
            FROM episodes WHERE source='ladder' AND format=?
                AND julianday(created)>=julianday(?) ORDER BY created
        """, (manifest.get("format"), latest.get("started_at"))) if row["room"] in known]
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
            "target": manifest.get("targets", {}).get("ladder_games", 100),
            "game": len(complete) + 1, "choices": live[-1]["choices"] if live else 0,
            "side": live[-1]["side"] if live else None,
            "updates": observer.get("actor", {}).get("updates"),
            "scout_samples": observer.get("scout", {}).get("samples"),
            "recent": recent, "checked_at": datetime.now(timezone.utc).isoformat()}


def battle_log(room=None):
    latest = read(PROJECT / "data/ml/campaigns/latest.json")
    state = read(Path(latest["directory"]) / "status.json")
    room = room or state.get("active_room")
    if room and ROOM.fullmatch(room):
        snapshot = read(PROJECT / "data/ml/live-watch" / (room + ".json"))
        if snapshot:
            return snapshot
    info = status()
    room = room or (info["recent"][0]["room"] if info["recent"] else None)
    if not room or not ROOM.fullmatch(room):
        return {"room": None, "log": [], "live": False, "source": "waiting"}
    database = sqlite3.connect((PROJECT / "data/ml/experience.sqlite3").resolve().as_uri() + "?mode=ro", uri=True, timeout=3)
    try:
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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlsplit(self.path)
        if route.path in ("/api/status", "/api/battle"):
            try:
                payload = status() if route.path == "/api/status" else battle_log(parse_qs(route.query).get("room", [None])[0])
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
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18489)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
