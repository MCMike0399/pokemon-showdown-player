#!/usr/bin/env python3
"""Bounded campaign controller with verified outcomes and explicit recovery."""
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def atomic(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


def append(path, obj):
    with path.open("a") as handle:
        handle.write(json.dumps(obj) + "\n")


def terminal(result):
    return isinstance(result, dict) and bool(result.get("winner") or result.get("tie")) and not any(
        result.get(key) for key in ("ongoing", "unfinished", "unresolved", "stalled"))


def source_generation():
    paths = [REPO / name for name in ('ps_client.py', 'ps_mcp_server.py', 'harness.py', 'battle_state.py', 'scripts/campaign_ladder.py')]
    paths.extend(sorted((REPO / 'ml').glob('*.py')))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Blocked(Exception):
    pass


class Campaign:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        # Read before writing status; the previous room is recovery evidence.
        self.previous = read(self.directory / "status.json")
        self.manifest = read(self.directory / "manifest.json")
        self.format = self.manifest["format"]
        self.base = "Campaign-" + self.directory.name + "-base"
        self.records = []
        for line in (self.directory / "games.jsonl").read_text().splitlines():
            if line.strip():
                self.records.append(json.loads(line))
        seen = set()
        for row in self.records:
            if not terminal(row.get("result")):
                raise Blocked("games.jsonl contains a nonterminal record; reconcile it before continuing")
            if row["room"] in seen:
                raise Blocked("duplicate terminal room in games.jsonl")
            seen.add(row["room"])
        self.seen = seen
        self.state = {"phase": "preflight", "active_room": self.previous.get("active_room"),
                      "current_tool": None, "completed": len(seen), "errors": [],
                      "recorder_version": 2, "controller_version": 3}
        self.db = sqlite3.connect((REPO / "data/ml/experience.sqlite3").as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.recount()

    def recount(self):
        outcomes = []
        for record in self.records:
            rows = self.db.execute("SELECT id,outcome FROM episodes WHERE json_extract(data,'$.room')=? AND status='complete'",
                                   (record["room"],)).fetchall()
            if len(rows) != 1:
                raise Blocked("terminal room must have one complete episode: " + record["room"])
            outcomes.append(rows[0]["outcome"])
        score = [sum(value == outcome for value in outcomes) for outcome in (1, -1, 0)]
        self.state.update(w=score[0], l=score[1], t=score[2], w_l_t=score, completed=len(self.seen))

    def target(self):
        target = read(self.directory / "control.json").get("target_ladder", self.manifest["targets"]["ladder_games"])
        if type(target) is not int or not 1 <= target <= 1000:
            raise Blocked("campaign target must be an integer 1..1000")
        return target

    def status(self, next_step):
        self.state.update(updated_at=utc(), next_step=next_step, target_ladder=self.target())
        atomic(self.directory / "status.json", self.state)

    async def call(self, session, tool, args, room=None):
        self.state.update(current_tool=tool, tool_started_at=utc())
        self.status("Calling " + tool)
        append(self.directory / "events.jsonl", {"type": "tool_start", "ts": utc(), "tool": tool, "args": args,
                                      "room": room, "game": self.state["completed"]})
        started = time.monotonic()
        try:
            response = await session.call_tool(tool, args)
            text = "".join(getattr(item, "text", "") for item in response.content or [])
            if getattr(response, "isError", False) or getattr(response, "is_error", False) or getattr(response, "error", None):
                raise Blocked(tool + ": " + text[:300])
            append(self.directory / "events.jsonl", {"type": "tool_end", "ts": utc(), "tool": tool, "room": room,
                "duration_ms": round((time.monotonic()-started)*1000), "isError": False, "text": text[:2000]})
            try:
                return json.loads(text)
            except ValueError:
                return text
        except Exception as error:
            append(self.directory / "events.jsonl", {"type": "tool_error", "ts": utc(), "tool": tool,
                                          "room": room, "error": str(error)[:400]})
            raise Blocked(str(error)) from error

    async def identity(self, session):
        info = await self.call(session, "ps_status", {})
        normalize = lambda value: "".join(c for c in value.lower() if c.isalnum())
        if not info.get("loggedIn") or normalize(info.get("user") or "") != normalize(info.get("configuredUser") or ""):
            raise Blocked("configured account identity was lost; no further matchmaking or account changes")

    def exact_team(self, room):
        rows = self.db.execute("SELECT team,status FROM episodes WHERE json_extract(data,'$.room')=?", (room,)).fetchall()
        fingerprints = {row["team"] for row in rows if row["status"] == "pending"}
        if len(fingerprints) != 1:
            raise Blocked("cannot identify one exact collecting team for recovery")
        from harness import TeamStore
        from ml.teams import team_id
        teams = TeamStore()
        for name in teams.list():
            team = teams.get(name)
            if team["format"] == self.format and team_id(self.format, team["sets"]) in fingerprints:
                return name
        raise Blocked("exact collecting team is no longer in the team store")

    async def play(self, session, room, team, resumed=False):
        self.state.update(phase="playing", active_room=room, actual_team=team)
        self.status("Resolve the existing battle" if resumed else "Play current battle")
        started = time.monotonic()
        result = await self.call(session, "ps_result", {"room": room}, room)
        for attempt in range(3):
            if terminal(result):
                break
            output = await self.call(session, "ps_ml_play", {"room": room, "format": self.format, "team": team, "learn": True}, room)
            result = output.get("result") or await self.call(session, "ps_result", {"room": room}, room)
            if terminal(result):
                break
        if not terminal(result):
            raise Blocked("battle remains unresolved after bounded same-room attempts")
        finished = await self.call(session, "ps_ml_finish", {"room": room}, room)
        matches = self.db.execute("SELECT id,revision,team,data,outcome FROM episodes WHERE json_extract(data,'$.room')=? AND status='complete'", (room,)).fetchall()
        if len(matches) != 1:
            raise Blocked("terminal battle does not have exactly one verified completed recording")
        episode = matches[0]
        record = {"game": self.state["completed"] + 1, "room": room, "team_requested": self.base, "team_used": team,
                  "team_fingerprint": episode["team"], "result": result, "winner": result.get("winner"),
                  "finish": finished, "actor_revision": episode["revision"], "episode_id": episode["id"],
                  "recorder_version": json.loads(episode["data"]).get("recorder_version"),
                  "duration_s": round(time.monotonic()-started, 1), "ended_utc": utc(), "resumed": resumed}
        if room in self.seen:
            raise Blocked("this terminal room was already counted")
        append(self.directory / "games.jsonl", record)
        self.records.append(record); self.seen.add(room); self.recount()
        self.state.update(phase="idle", active_room=None, current_tool=None, actor_revision=episode["revision"])
        self.status("Verified terminal game; next search is safe")
        print("PROGRESS %d/%d W=%d L=%d T=%d room=%s" %
              (len(self.seen), self.target(), self.state["w"], self.state["l"], self.state["t"], room), flush=True)

    async def run(self):
        generation = source_generation()
        parameters = StdioServerParameters(command=str(REPO / ".venv/bin/python"), args=[str(REPO / "ps_mcp_server.py")])
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer, read_timeout_seconds=1200) as session:
                await session.initialize()
                login = await self.call(session, "ps_login", {})
                if not login.get("loggedIn"):
                    raise Blocked("configured account login was not confirmed")
                await self.identity(session)
                pending = self.previous.get("active_room")
                if pending and pending not in self.seen:
                    team = self.previous.get("actual_team") or self.exact_team(pending)
                    await self.call(session, "ps_join", {"room": pending}, pending)
                    await asyncio.sleep(1)
                    await self.play(session, pending, team, resumed=True)
                deadline = datetime.fromisoformat(self.manifest["deadline_at"].replace("Z", "+00:00")).timestamp()
                while len(self.seen) < self.target() and time.time() < deadline:
                    if source_generation() != generation:
                        self.state.update(phase='source_reload_requested', active_room=None, current_tool=None)
                        self.status('Source changed; close this connection and reload at the terminal boundary')
                        return 75
                    await self.identity(session)
                    model = await self.call(session, "ps_ml_status", {"format": self.format})
                    self.state["actor_revision"] = model["model"]["revision"]
                    room = None
                    for attempt in range(3):
                        if time.time() >= deadline:
                            break
                        await self.call(session, "ps_ml_ladder", {"format": self.format, "team": self.base, "explore_team": len(self.seen) >= 50})
                        waiting = await self.call(session, "ps_ml_wait", {"timeout": 120})
                        if waiting.get("room"):
                            room = waiting["room"]; team = waiting["team"]; break
                        await self.call(session, "ps_send", {"room": "", "message": "/cancelsearch"})
                        await asyncio.sleep((30, 60, 120)[attempt])
                    if not room:
                        if time.time() >= deadline:
                            break
                        raise Blocked("three matchmaking attempts timed out")
                    await self.play(session, room, team)
                    await asyncio.sleep(2)
                reached = len(self.seen) >= self.target()
                self.state.update(phase="ladder_target_complete" if reached else "deadline_reached", active_room=None, current_tool=None)
                self.status("Configured ladder target complete" if reached else "Campaign deadline reached; terminal games preserved")
                print("RUN_TARGET_COMPLETE", flush=True)


def describe_error(error):
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(describe_error(item) for item in error.exceptions)
    return str(error)


async def main(campaign_directory):
    campaign = None
    try:
        campaign = Campaign(campaign_directory)
        return await campaign.run() or 0
    except Exception as error:
        if campaign:
            campaign.state.update(phase="blocked", current_tool=None)
            campaign.state["errors"].append({"error": describe_error(error)[:500], "ts": utc()})
            campaign.status("Resolve the blocker before starting another battle")
            append(campaign.directory / "events.jsonl", {"type": "blocked", "ts": utc(), "reason": describe_error(error)[:500]})
        print("BLOCKED: " + describe_error(error), flush=True)
        return 0
    finally:
        if campaign:
            campaign.db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.campaign)))
