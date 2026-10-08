"""Reuse the harness connection for live inference, submissions and full games."""
from __future__ import annotations

import asyncio
import time
import math
import json
from contextlib import nullcontext

from ml.storage import fingerprint
from ml.teams import team_id


def context(player, room: str, fmt: str, team: str = "") -> dict:
    request = player._request(room)
    if player.finished(room) or not request or request.get("wait"):
        raise ValueError("battle is finished or has no actionable request")
    if room.startswith("battle-") and room.split("-")[1] != fmt:
        raise ValueError("format does not match the battle room")
    sets = player.teams.get(team)["sets"] if team else None
    if team and player.teams.get(team)["format"] != fmt:
        raise ValueError("stored team format does not match battle format")
    return {"room": room, "format": fmt, "team_id": team_id(fmt, sets) if sets else "", "source": "ladder",
            "turn": player.c.battle_summary(room).get("turn"), "request": request,
            "choices": player.legal_choices(room), "state": player.state_summary(room),
            "public_log": list(player.c.battles.get(room, {}).get("log", []))}


class LiveSession:
    def __init__(self, brain, player):
        self.brain, self.player = brain, player
        self.submitted = {}
        self.finished_rooms = set()
        self.finished_outputs = {}
        self.watch_room = None
        self.watch_digest = None
        self.reconnect_counts = {}

    async def recover_transport(self, room: str):
        """Reauthenticate this same client/account and rejoin only the collecting room."""
        client = self.player.c
        self.watch_room = None
        await client.close()
        login = await client.login()
        if not login.get('loggedIn'):
            raise ValueError('configured account identity could not be restored')
        normalize = lambda value: ''.join(c for c in (value or '').lower() if c.isalnum())
        if normalize(client.user) != normalize(client.username):
            raise ValueError('recovery identity does not match configured account')
        client.battles.pop(room, None)
        client.rooms.pop(room, None)
        self.submitted = {key: value for key, value in self.submitted.items() if key[0] != room}
        await client.join(room)
        stop = time.monotonic() + 20
        while time.monotonic() < stop:
            if self.player.finished(room) or self.player._request(room):
                self.watch_room = room
                return
            await asyncio.sleep(.2)
        raise TimeoutError('same-room rejoin did not produce a request or terminal result')

    def publish_watch(self, room: str):
        """Mirror only battle protocol, using the existing player's connection."""
        if not room.startswith("battle-") or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in room):
            return
        from ml.scout import PUBLIC_KINDS
        from ml.storage import now
        raw = self.player.c.battles.get(room, {}).get("log", [])
        req = self.player._request(room) or {}
        side = req.get("side", {}).get("id") or self.brain.room_side(room)
        summary = self.player.c.battle_summary(room)
        names = {key: "Your agent" if key == side else "Opponent" for key in ("p1", "p2")}
        visible = PUBLIC_KINDS | {"teampreview", "teamsize", "upkeep", "inactive", "inactiveoff", "cant", "swap"}
        log = []
        for line in raw:
            parts = line.split("|")
            if len(parts) < 3 or not (parts[1] in visible or parts[1].startswith("-")):
                continue
            if parts[1] == "player":
                avatar = parts[4] if len(parts) > 4 and parts[4].isdigit() else "1"
                line = f"|player|{parts[2]}|{names.get(parts[2], 'Player')}|{avatar}"
            elif parts[1] == "win":
                winner_side = next((key for key, value in summary.get("players", {}).items() if value == parts[2]), None)
                line = "|win|" + names.get(winner_side, "Winner")
            log.append(line)
        data = {"room": room, "side": side, "turn": summary.get("turn"), "log": log,
                "live": not self.player.finished(room), "source": "player-relay"}
        digest = fingerprint(data)
        if digest == self.watch_digest:
            return
        data["captured_at"] = now()
        folder = self.brain.store.root / "live-watch"
        folder.mkdir(exist_ok=True)
        text = json.dumps(data)
        for target in (folder / (room + ".json"), folder / "current.json"):
            temporary = target.with_suffix(".tmp")
            temporary.write_text(text)
            temporary.replace(target)
        self.watch_digest = digest

    async def choose(self, room: str, fmt: str, team: str = "", explore: bool = True, demonstration: str | None = None):
        if not self.player.c.logged_in:
            raise ValueError("call ps_login once before playing")
        ctx = context(self.player, room, fmt, team)
        self.watch_room = room
        req = ctx["request"]
        side = req.get("side", {}).get("id", "p1")
        episode = self.brain.resume(room, side)
        if episode:
            if episode["format"] != fmt or episode["team"] != ctx["team_id"]:
                raise ValueError("cannot change format or team while resuming an episode")
            if self.brain.model(fmt).revision != episode["revision"]:
                raise ValueError("collecting checkpoint revision changed; cannot resume this PPO recording")
            if self.brain.model(fmt).feature_profile != episode.get('feature_profile', 'legacy'):
                raise ValueError('collecting feature profile changed; cannot resume this recording')
        if episode and (episode["demonstration"] != (demonstration is not None) or
                        (episode["on_policy"] and not explore and demonstration is None)):
            raise ValueError("keep one learning mode for the whole recorded game")
        key = (room, req.get("rqid", fingerprint(req)))
        log = self.player.c.battles[room].get("log", [])
        latest = max(((k,v) for k,v in self.submitted.items() if k[0] == room),
                     key=lambda pair: pair[1]["log_index"], default=None)
        if latest and any(line.startswith("|error|") for line in log[latest[1]["log_index"]:]):
            if latest[1]["decision"].get("recorded"):
                self.brain.reject(room, req.get("side", {}).get("id", "p1"))
            del self.submitted[latest[0]]
        submitted = self.submitted.get(key)
        if submitted:
            return {**submitted["decision"], "already_submitted": True}
        saved = episode["steps"][-1] if episode and episode["steps"] else None
        if saved and saved["request_id"] == key[1]:
            # A crash may follow socket submission but precede the next request.
            # Resubmit the exact sampled action, preserving its collecting logprob.
            if saved["choice"] not in ctx["choices"]:
                raise ValueError("recovered choice is no longer legal; reconcile the pending recording")
            decision = {"choice": saved["choice"], "probability": math.exp(saved["logprob"]),
                        "value": saved["value"], "revision": episode["revision"],
                        "sampled": episode["on_policy"], "recorded": True, "recovered": True}
        else:
            decision = self.brain.decide(ctx, explore=explore, record=explore and demonstration is None, demonstration=demonstration)
        log_index = len(log)
        try:
            await self.player.choose(room, decision["choice"])
        except Exception:
            if not (decision.get('recovered') and saved and saved.get('submitted') is True):
                self.brain.reject(room, req.get("side", {}).get("id", "p1"), reason='socket-submission-failed')
            raise
        if decision.get("recorded"):
            episode = self.brain.pending[(room, side)]
            episode["steps"][-1]["submitted"] = True
            self.brain.store.save_episode(episode)
        self.submitted[key] = {"log_index": log_index, "decision": decision}
        return decision

    def finish(self, room: str):
        if room in self.finished_outputs:
            return self.finished_outputs[room]
        req = self.player._request(room) or {}
        side = req.get("side", {}).get("id") or self.brain.room_side(room)
        result = self.player.result(room)
        if not result.get("ongoing"):
            restored = self.brain.resume(room, side)
            if restored and restored["steps"] and restored["steps"][-1].get("submitted") is False:
                # Conservatively exclude a proposal whose socket submission was
                # interrupted. It cannot be given credit for a terminal result.
                self.brain.reject(room, side)
        latest = max((v for key,v in self.submitted.items() if key[0] == room),
                     key=lambda v: v["log_index"], default=None)
        log = self.player.c.battles.get(room, {}).get("log", [])
        if latest and latest["decision"].get("recorded") and any(line.startswith("|error|") for line in log[latest["log_index"]:]):
            self.brain.reject(room, side)
        # Some ladder rooms anonymize names. The private request establishes our
        # side; its public battle name identifies the winner in this room.
        own_name = self.player.c.battle_summary(room).get("players", {}).get(side) or self.player.c.user or ""
        record = self.brain.finish(room, result, own_name, side, public_log=log)
        learning = None
        if not result.get("ongoing") and room not in self.finished_rooms:
            self.finished_rooms.add(room)
            from ml.scout import ingest_public
            from ml.continuous import enqueue_battle, kick_worker
            fmt = record.get("format") or (room.split("-")[1] if room.startswith("battle-") else None)
            if fmt:
                public = ingest_public(self.brain.store, fmt,
                                       self.player.c.battles.get(room, {}).get("log", []), "own-live-game", "own-"+room)
                learning = {"public_experience": public,
                            "job": enqueue_battle(self.brain.store, fmt, record.get("id", room))}
                if learning["job"]:
                    kick_worker(self.brain.store.root)
        if not result.get("ongoing"):
            self.submitted = {key: value for key, value in self.submitted.items() if key[0] != room}
        output = {"result": result, "experience": record, "background_learning": learning}
        if not result.get("ongoing"):
            self.finished_outputs[room] = output
        self.publish_watch(room)
        return output

    async def play(self, room: str, fmt: str, team: str = "", learn: bool = True, timeout: float = 120,
                   decision_lock=None):
        if not 5 <= timeout <= 600:
            raise ValueError("stall timeout must be 5..600 seconds")
        self.watch_room = room
        req = self.player._request(room) or {}
        side = req.get("side", {}).get("id") or self.brain.room_side(room)
        episode = self.brain.resume(room, side)
        if not self.player.c.logged_in and not episode:
            raise ValueError("call ps_login once before playing")
        if self.player.c.logged_in:
            if room not in self.player.c.battles:
                await self.player.c.join(room)
            try:
                await self.player.c.send(room + "|/timer on")
            except ConnectionError:
                if self.player.c.connected and self.player.c.logged_in:
                    raise
        last_progress = time.monotonic()
        last_key = None
        ots_accepted = False
        retries = 0
        while not self.player.finished(room):
            if not self.player.c.connected or not self.player.c.logged_in:
                attempts = self.reconnect_counts.get(room, 0)
                if attempts >= 3:
                    return {'result': {'room': room, 'unfinished': True, 'connection_lost': True},
                            'experience': 'pending; no reward assigned', 'transport_recoveries': attempts}
                self.reconnect_counts[room] = attempts + 1
                try:
                    await self.recover_transport(room)
                    if not self.player.finished(room):
                        await self.player.c.send(room + '|/timer on')
                except Exception:
                    await self.player.c.close()
                    await asyncio.sleep(1)
                    continue
                last_key = None
                last_progress = time.monotonic()
                continue
            req = self.player._request(room) or {}
            log = self.player.c.battles.get(room, {}).get("log", [])
            if not ots_accepted and any("/acceptopenteamsheets" in line for line in log):
                try:
                    await self.player.c.send(room + "|/acceptopenteamsheets")
                except ConnectionError:
                    if self.player.c.connected and self.player.c.logged_in:
                        raise
                    continue
                ots_accepted = True
            key = (req.get("rqid", fingerprint(req)), fingerprint(req))
            if key != last_key:
                last_progress = time.monotonic()
                last_key = key
                retries = 0
            if req and not req.get("wait") and self.player.legal_choices(room):
                async with decision_lock or nullcontext():
                    try:
                        decision = await self.choose(room, fmt, team, explore=learn)
                    except Exception:
                        if not self.player.c.connected or not self.player.c.logged_in:
                            continue
                        raise
                if not decision.get("already_submitted"):
                    retries += 1
                    if retries > 5:
                        return {"result": {"room": room, "unfinished": True, "error": "repeated rejected choices"}}
            if time.monotonic() - last_progress > timeout:
                return {"result": {"room": room, "unfinished": True, "stalled": True}, "experience": "pending; no reward assigned"}
            await asyncio.sleep(0.4)
        async with decision_lock or nullcontext():
            output = self.finish(room)
        output['transport_recoveries'] = self.reconnect_counts.get(room, 0)
        if learn and output["experience"].get("recorded"):
            from ml.continuous import learning_report
            output["training"] = learning_report(output)
        return output
