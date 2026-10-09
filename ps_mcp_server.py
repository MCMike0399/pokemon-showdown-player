"""
ps_mcp_server.py - MCP server exposing live Pokemon Showdown control to agents.

Run:  .venv/bin/python ps_mcp_server.py          (stdio transport)

Tools:
  ps_status            connection / login / rooms / battles
  ps_login             log in with the configured account
  ps_join              join a chat room
  ps_read              read recent messages from a room (or global log)
  ps_send              send chat text / a slash command to a room
  ps_battles           list current battle rooms
  ps_battle            detailed battle state (players, turn, weather, request)
  ps_waiting           battles currently waiting on a choice
  ps_choose            submit a choice (move N / switch N / team ...)
  ps_dex               Pokemon / move / item / ability lookup (cached from PS data)

Credentials come from POKEMON_SHOWDOWN_PLAYER/.env (PS_USERNAME, PS_PASSWORD)
or the process environment.
"""
from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer

from ps_client import PSClient

HERE = Path(__file__).resolve().parent


def load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

mcp = MCPServer("pokemon-showdown-player")
client = PSClient()

DATA_BASE = "https://play.pokemonshowdown.com/data"
_cache: dict[str, dict] = {}


async def _data(name: str) -> dict:
    if name not in _cache:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{DATA_BASE}/{name}.json", headers={"User-Agent": "ps-player/1.0"})
            r.raise_for_status()
            _cache[name] = r.json()
    return _cache[name]


@mcp.tool()
async def ps_status() -> str:
    """Show the Pokemon Showdown connection status, logged-in user, rooms and battles."""
    info = {
        "connected": client.connected,
        "loggedIn": client.logged_in,
        "user": client.user,
        "rooms": sorted(client.rooms.keys())[:40],
        "battles": sorted(client.battles.keys()),
        "lastError": client.last_error,
        "configuredUser": client.username,
    }
    return json.dumps(info, indent=2)


@mcp.tool()
async def ps_login() -> str:
    """Connect to Showdown and log in with the configured account (from .env)."""
    await client.connect()
    res = await client.login()
    return json.dumps(res, indent=2)


@mcp.tool()
async def ps_join(room: str) -> str:
    """Join a Showdown room (e.g. 'lobby', 'vgc', 'help', or a battle id)."""
    if not client.connected:
        await client.connect()
    await client.join(room)
    return f"requested join: {room}"


@mcp.tool()
async def ps_read(room: str, limit: int = 40) -> str:
    """Read the most recent lines of a room. Use room='global' for the global log."""
    lines = await client.read(room, limit)
    return "\n".join(lines) if lines else f"(no buffered lines for {room!r})"


@mcp.tool()
async def ps_send(room: str, message: str) -> str:
    """Send text (or a slash command) to a room. Commands: '/join x', '/msg user, hi', '/challenge user, gen9championsvgc2026regmc'."""
    if not client.connected:
        await client.connect()
    await client.chat(room, message)
    return f"sent to {room!r}: {message}"


@mcp.tool()
async def ps_battles() -> str:
    """List the battle rooms this connection is in."""
    return json.dumps(sorted(client.battles.keys()), indent=2)


@mcp.tool()
async def ps_battle(room: str) -> str:
    """Detailed state of one battle room (players, turn, weather, active, last request)."""
    return json.dumps(client.battle_summary(room), indent=2)[:6000]


@mcp.tool()
async def ps_waiting() -> str:
    """List battles currently waiting on us to submit a choice."""
    return json.dumps(client.current_requests(), indent=2)[:4000]


@mcp.tool()
async def ps_choose(room: str, choice: str) -> str:
    """Submit a battle choice. Examples: 'move 1', 'move 1, move 2', 'switch 2', 'team 1,2,3,4'."""
    return await client.choose(room, choice)


@mcp.tool()
async def ps_dex(name: str, kind: str = "pokemon") -> str:
    """Look up PS data. kind: pokemon | move | item | ability | typechart. name is case-insensitive."""
    kind_map = {
        "pokemon": "pokedex",
        "move": "moves",
        "item": "items",
        "ability": "abilities",
        "typechart": "typechart",
    }
    key = kind_map.get(kind.lower())
    if not key:
        return f"unknown kind {kind!r}; use pokemon|move|item|ability|typechart"
    data = await _data(key)
    if kind.lower() == "typechart":
        return json.dumps(data, indent=2)[:3000]
    slug = name.lower().replace(" ", "").replace("-", "")
    for k, v in data.items():
        if k.lower() == slug:
            entry = {kk: vv for kk, vv in v.items() if kk not in ("learnset", "desc")}
            return json.dumps({"id": k, **entry}, indent=2)[:4000]
    matches = [k for k in data if slug in k.lower()][:10]
    return f"no exact match for {name!r}; similar: {matches}"



# ---------------- harness: teams + matchmaking ----------------
from harness import Player, TeamStore, pack_team  # noqa: E402

teams = TeamStore()
player = Player(client=client, teams=teams)


@mcp.tool()
async def ps_team_list() -> str:
    """List the locally stored teams (the harness team store)."""
    return json.dumps(teams.list(), indent=2)


@mcp.tool()
async def ps_team_get(name: str) -> str:
    """Show one stored team as JSON."""
    return json.dumps(teams.get(name), indent=2)


@mcp.tool()
async def ps_team_create(name: str, sets_json: str, format: str = "gen9randombattle") -> str:
    """Create a team. sets_json is a JSON array of sets.
    Set fields: name, species, item, ability, moves[4], nature, evs{hp,atk,def,spa,spd,spe}, level, gender."""
    sets = json.loads(sets_json)
    teams.create(name, sets, format)
    return "created " + name + " (" + str(len(sets)) + " sets)"


@mcp.tool()
async def ps_team_update(name: str, sets_json: str = "", format: str = "", new_name: str = "") -> str:
    """Update a team (any of: new sets, new format, rename)."""
    sets = json.loads(sets_json) if sets_json else None
    teams.update(name, sets=sets, fmt=(format or None), new_name=(new_name or None))
    return "updated " + name


@mcp.tool()
async def ps_team_delete(name: str) -> str:
    """Delete a stored team."""
    return "deleted" if teams.delete(name) else "not found"


@mcp.tool()
async def ps_team_packed(name: str) -> str:
    """Show the Showdown packed form of a stored team (what /utm sends)."""
    return teams.packed(name)


@mcp.tool()
async def ps_ladder(format: str, team: str = "") -> str:
    """Search a ladder battle. format e.g. gen9randombattle or gen9championsvgc2026regmc.
    team: optional stored team name (uploaded with /utm first)."""
    if not client.connected:
        await client.connect()
    packed = teams.packed(team) if team else None
    _match_context[format] = {"team": team, "sets": teams.get(team)["sets"] if team else None}
    return await player.ladder(format, packed)


@mcp.tool()
async def ps_choices(room: str) -> str:
    """List the legal choices for the current battle request (e.g. move 1, switch 2)."""
    ch = player.legal_choices(room)
    return json.dumps({"room": room, "count": len(ch), "choices": ch}, indent=2)


@mcp.tool()
async def ps_result(room: str) -> str:
    """Result of a battle (winner / tie / ongoing)."""
    return json.dumps(player.result(room), indent=2)


# Optional ML dependencies load only when a brain tool is invoked. Basic team /
# connection tools continue to work with the original requirements.txt.
_brain = None
_ml_session = None
_ml_lock = asyncio.Lock()
_observer = None
_play_locks = {}
_match_context = {}


def _ml():
    global _brain, _ml_session, _observer
    if _brain is None:
        from ml.brain import Brain
        from ml.live import LiveSession
        _brain = Brain()
        _ml_session = LiveSession(_brain, player)
        from ml.continuous import BattleObserver
        _observer = BattleObserver(_ml_session, _ml_lock)
        _observer.start()
    return _brain


@mcp.tool()
async def ps_ml_status(format: str = "") -> str:
    """Experience/research counts and model revision. Install requirements-ml.txt first."""
    async with _ml_lock:
        brain = _ml()
        info = brain.store.stats()
        if format:
            model = brain.model(format)
            info["model"] = {"format": format, "revision": model.revision, "updates": model.updates,
                             "policy_temperature": model.policy_temperature,
                             "preview_temperature": model.preview_temperature,
                             "feature_profile": model.feature_profile,
                             "checkpoint": str(model.path), "algorithm": "action-conditioned PPO actor-critic",
                             "strength": "unmeasured; zero updates uses a tactical initialization"}
            ready = brain.store.root / 'ready' / (format + '.json')
            staged = json.loads(ready.read_text()) if ready.exists() else None
            info['staged_candidate'] = ({'parent_revision': staged['parent_revision'],
                                         'staged_at': staged['staged_at'], 'evaluation': staged['evaluation']}
                                        if staged else None)
        return json.dumps(info, indent=2)


@mcp.tool()
async def ps_ml_recording(room: str, start: int = 0, limit: int = 3, alternatives: bool = False) -> str:
    """Inspect saved decision-time requests, observations and named choices.
    Read-only; no login, action submission or training. Legacy missing snapshots
    stay unknown. Default returns selected/top-five choices; alternatives=True
    includes the whole saved mask. Does not include future battle-log events."""
    async with _ml_lock:
        from ml.recording import read_archive
        from ml.storage import DEFAULT_ROOT
        root = _brain.store.root if _brain is not None else DEFAULT_ROOT
        return json.dumps(read_archive(root, room, start, limit, alternatives), indent=2)


@mcp.tool()
async def ps_ml_decide(room: str, format: str, team: str = "", explore: bool = False) -> str:
    """Recommend a legal joint action with probabilities; does not submit/record it.
    Use ps_ml_choose to submit and collect trainable experience."""
    async with _ml_lock:
        from ml.live import context
        team = team or _match_context.get(format, {}).get("team", "")
        return json.dumps(_ml().decide(context(player, room, format, team), explore=explore), indent=2)


@mcp.tool()
async def ps_ml_choose(room: str, format: str, team: str = "", explore: bool = True) -> str:
    """Read waiting state, select and submit an ML action on the existing connection.
    explore=True samples the policy and records PPO experience. Retries are idempotent.
    Call ps_ml_finish when the game ends. explore=False plays without PPO recording."""
    async with _ml_lock:
        _ml()
        team = team or _match_context.get(format, {}).get("team", "")
        return json.dumps(await _ml_session.choose(room, format, team, explore), indent=2)


@mcp.tool()
async def ps_ml_demonstrate(room: str, format: str, choice: str, team: str = "") -> str:
    """Submit an expert/agent-selected legal choice as an imitation label.
    Use for a whole game; do not mix with PPO decisions in that game. Finish it,
    then ps_ml_train(format, imitation=True). Web prose alone is never a label."""
    async with _ml_lock:
        _ml()
        return json.dumps(await _ml_session.choose(room, format, team, False, demonstration=choice), indent=2)


@mcp.tool()
async def ps_ml_finish(room: str, train: bool = True) -> str:
    """Record a verified terminal win/loss/tie and optionally update the brain.
    Ongoing games are left pending with no reward; repeated calls cannot double-credit."""
    async with _ml_lock:
        brain = _ml()
        output = _ml_session.finish(room)
        experience = output["experience"]
        if train and experience.get("recorded"):
            from ml.continuous import learning_report
            output["training"] = learning_report(output)
        return json.dumps(output, indent=2)


@mcp.tool()
async def ps_ml_train(format: str, epochs: int = 4, imitation: bool = False) -> str:
    """Queue bounded background learning from complete compatible unused episodes.
    Candidates must beat the incumbent on paired evaluations before promotion.
    imitation=True consumes explicitly recorded demonstration labels."""
    async with _ml_lock:
        if not 1 <= epochs <= 30:
            raise ValueError("epochs must be 1..30")
        from ml.continuous import kick_worker
        from ml.storage import fingerprint, now
        brain = _ml()
        key = "manual-"+fingerprint(now())
        brain.store.enqueue(key, "learn", format, {"epochs": epochs, "imitation": imitation})
        kick_worker(brain.store.root)
        return json.dumps({"queued": True, "job": key, "promotion": "paired evaluation required"})


@mcp.tool()
async def ps_ml_play(room: str, format: str, team: str = "", learn: bool = True) -> str:
    """Play an existing battle entirely with the brain; collect experience and learn
    after a completed game. Reuses ps_login's connection. No new matchmaking/account."""
    _ml()
    team = team or _match_context.get(format, {}).get("team", "")
    lock = _play_locks.setdefault(room, asyncio.Lock())
    if lock.locked():
        raise ValueError("this room already has an active automated player")
    async with lock:
        return json.dumps(await _ml_session.play(room, format, team, learn, decision_lock=_ml_lock), indent=2)


@mcp.tool()
async def ps_team_plan(format: str, base: str = "", explore: bool = False,
                       evidence_source: str = "ladder", save_as: str = "") -> str:
    """Let the learned team-outcome model select/validate a team or sourced variant.
    Returns exact selected sets, evidence, sources and whether stats were generated.
    save_as creates a named team without overwriting a different existing team."""
    async with _ml_lock:
        from ml.teams import TeamPlanner
        return json.dumps(await TeamPlanner(_ml().store, teams).plan(format, base, explore, evidence_source, save_as), indent=2)


@mcp.tool()
async def ps_ml_ladder(format: str, team: str = "", explore_team: bool = False,
                       max_team_candidates: int = 8) -> str:
    """One player operation: select a team with the team model, validate, upload,
    and enter matchmaking on ps_login's existing connection. Explicit team limits
    selection to that team and its researched variants. With an explicit team,
    max_team_candidates=1 pins its exact sets for controlled comparisons.
    Does not log into another account."""
    if not client.logged_in:
        raise ValueError("call ps_login once first")
    from ml.teams import TeamPlanner
    from ml.storage import fingerprint
    async with _ml_lock:
        brain = _ml()
        if any(not player.finished(room) for room in player.battles()):
            raise ValueError("finish the existing battle before another matchmaking operation")
        from ml.promotion import promote_ready
        promotion = promote_ready(brain.store, format)
        plan = await TeamPlanner(brain.store, teams).plan(format, team, explore_team,
                                                       max_candidates=max_team_candidates)
        selected = plan["selected"]
        name = "ML-"+selected["id"]
        if name not in teams.list():
            teams.create(name, selected["sets"], format)
        _match_context[format] = {"team": name, "sets": selected["sets"]}
        status = await player.ladder(format, selected["packed"])
    return json.dumps({"status": status, "team": name, "promotion": promotion,
                       "selection": {k:v for k,v in selected.items() if k not in ("packed", "sets")}}, indent=2)


@mcp.tool()
async def ps_ml_wait(timeout: int = 120) -> str:
    """Wait for the first unfinished battle request on this same connection.
    Returns room/format/team for ps_ml_play; does not log in or create a new client."""
    if not 1 <= timeout <= 180:
        raise ValueError("timeout must be 1..180 seconds")
    room = await player.wait_for_request(timeout=timeout)
    if not room:
        return json.dumps({"room": None, "timeout": True, "last_error": client.last_error})
    fmt = room.split("-")[1]
    return json.dumps({"room": room, "format": fmt, "team": _match_context.get(fmt, {}).get("team", "")})


@mcp.tool()
async def ps_learning_status() -> str:
    """Durable learning queue, feeds, shared-host headroom, backend and scout status."""
    async with _ml_lock:
        from ml.continuous import LearningConfig
        from ml.resources import ResourcePolicy
        from ml.pipeline import status as pipeline_status
        brain = _ml()
        config = LearningConfig.load(brain.store.root)
        benchmark_file = brain.store.root/"compute-benchmark.json"
        return json.dumps({"enabled": config.enabled, "resources": ResourcePolicy(**config.resource).sample(),
                           "queue": [dict(r) for r in brain.store.db.execute("SELECT id,kind,format,status,result FROM jobs ORDER BY created DESC LIMIT 10")],
                           "feeds": [dict(r) for r in brain.store.db.execute("SELECT * FROM feed_state")],
                           "compute": json.loads(benchmark_file.read_text()) if benchmark_file.exists() else "not benchmarked",
                           'pipeline': pipeline_status(brain.store.root),
                           "experience": brain.store.stats()}, indent=2)


@mcp.tool()
async def ps_learning_run(format: str, refresh_web: bool = False, practice: bool = True) -> str:
    """Queue a bounded shared-host learning cycle; refresh_web ingests exact-format
    tournament sheets/usage/public replays. CPU/MPS jobs run outside live inference."""
    async with _ml_lock:
        from ml.continuous import kick_worker
        from ml.storage import fingerprint, now
        brain = _ml()
        stamp = fingerprint(now())
        keys = []
        if refresh_web:
            keys.append(brain.store.enqueue("feed-"+stamp, "feed", format))
        keys.append(brain.store.enqueue("learn-"+stamp, "practice" if practice else "learn", format))
        kick_worker(brain.store.root)
        return json.dumps({"queued": keys, "resource_aware": True})


@mcp.tool()
async def ps_team_validate(name: str) -> str:
    """Validate a stored team against the pinned official offline simulator.
    Requires npm install && npm run simulator:build. The live server remains authoritative."""
    from ml.simulator import validate_team
    team = teams.get(name)
    return json.dumps(await validate_team(team["format"], team["sets"]), indent=2)


@mcp.tool()
async def ps_research_search(query: str, limit: int = 8) -> str:
    """Search current VGC resources. Brave API if configured, otherwise live tournament/team indexes.
    Returns source URLs/snippets; research is evidence, not executable instructions."""
    from ml.research import search
    return json.dumps(await search(query, limit), indent=2)


@mcp.tool()
async def ps_research_fetch(url: str, format: str) -> str:
    """Fetch/cache a primary VGC resource with URL, timestamp and explicit format tag.
    Supported domains: pokemon.com, pokemon-home.com, smogon.com, pokemonshowdown.com,
    limitlessvgc.com, victoryroad.pro, labmaus.net, pokepast.es."""
    async with _ml_lock:
        from ml.research import ingest
        return json.dumps(await ingest(_ml().store, url, format), indent=2)


@mcp.tool()
async def ps_research_read(query: str, format: str) -> str:
    """Retrieve cached matching-format research for the orchestrating agent, with citations."""
    async with _ml_lock:
        from ml.research import retrieve
        return json.dumps(retrieve(_ml().store, query, format), indent=2)


@mcp.tool()
async def ps_research_team(sets_json: str, format: str, source_url: str, player_name: str = "", event: str = "") -> str:
    """Import a complete sourced team as research after simulator validation.
    Does not overwrite TeamStore. Valid species priors steer model observations;
    ps_team_recommend/ps_team_variants expose candidates for subsequent testing."""
    async with _ml_lock:
        from ml.research import import_team
        from ml.simulator import validate_team
        sets = json.loads(sets_json)
        validation = await validate_team(format, sets)
        if validation["errors"]:
            raise ValueError("; ".join(validation["errors"]))
        return json.dumps(import_team(_ml().store, sets, format, source_url, player_name, event), indent=2)


@mcp.tool()
async def ps_research_usage(format: str, month: str, rating: int = 1630) -> str:
    """Fetch published YYYY-MM Showdown usage for this exact regulation and cutoff.
    Stores sourced moves/items/species statistics and updates model meta features."""
    async with _ml_lock:
        from ml.research import refresh_usage
        return json.dumps(await refresh_usage(_ml().store, format, month, rating), indent=2)


@mcp.tool()
async def ps_research_paste(url: str, format: str, player_name: str = "", event: str = "", save_as: str = "") -> str:
    """Fetch a credited PokePaste, parse and validate with official Showdown, import
    attributed research. Optionally create a new named harness team (never overwrite).
    The caller verifies the source's regulation and player attribution."""
    async with _ml_lock:
        from ml.research import import_paste
        result = await import_paste(_ml().store, url, format, player_name, event)
        if save_as:
            teams.create(save_as, result["sets"], format)
            result["saved_as"] = save_as
        return json.dumps(result, indent=2)


@mcp.tool()
async def ps_team_recommend(format: str, evidence_source: str = "ladder") -> str:
    """Rank stored/sourced teams using Bayesian win estimates and uncertainty.
    Exact team versions and local/ladder evidence stay separate. Unplayed teams
    have no demonstrated advantage; exploration_score favors testing uncertainty."""
    async with _ml_lock:
        from ml.teams import rank_teams
        return json.dumps(rank_teams(_ml().store, teams, format, evidence_source), indent=2)


@mcp.tool()
async def ps_team_variants(name: str) -> str:
    """Suggest same-species complete set replacements from same-format research.
    Each candidate includes attribution and needs validation and evaluation."""
    async with _ml_lock:
        from ml.teams import suggest_variants
        return json.dumps(suggest_variants(_ml().store, teams.get(name)), indent=2)

if __name__ == "__main__":
    from ml.reload import source_generation, pin_runtime
    pin_runtime(os.environ.get('PS_SOURCE_GENERATION') or source_generation())
    mcp.run()
