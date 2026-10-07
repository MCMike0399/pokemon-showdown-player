"""Incremental, attributed data feeds; exact formats and failures stay explicit."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta

import httpx
from bs4 import BeautifulSoup

from ml.research import fetch, ingest, import_team, refresh_usage
from ml.scout import ingest_public
from ml.storage import Store, fingerprint, now

# Owner-declared MIT public logs, pinned dataset revision. Historical regulations
# are never silently relabeled as the current one. Download is bounded and opt-in.
ARCHIVES = {
    "vgc-bench-champions-mb": {
        "format": "gen9championsvgc2026regmb", "license": "MIT (dataset owner declaration)",
        "card": "https://huggingface.co/datasets/cameronangliss/vgc-battle-logs",
        "url": "https://huggingface.co/datasets/cameronangliss/vgc-battle-logs/resolve/74f260510b1dbcaf4c42c8bf653f726b51d476ef/logs_gen9championsvgc2026regmb.json",
        "sha256": "7711711fc19ec36eb7f9c46b4f18e3621e33ef8e3ec2757983bf4589fbae87dc",
        "max_bytes": 8_000_000,
    }
}


def previous_month():
    return (datetime.now(timezone.utc).replace(day=1) - timedelta(days=1)).strftime("%Y-%m")


def parse_limitless(body: bytes, fmt: str):
    soup = BeautifulSoup(body, "html.parser")
    label = "Regulation Set M-" + fmt[-1].upper() if re.fullmatch(r"gen9championsvgc2026regm[a-z]", fmt) else None
    if not label or label not in soup.get_text(" ", strip=True):
        raise ValueError("tournament regulation does not match this feed's format")
    event = soup.title.get_text(" ", strip=True) if soup.title else "Limitless tournament"
    results = []
    for block in soup.select(".tournament-decklist"):
        sets = []
        for mon in block.select(".pkmn"):
            def text(selector):
                element = mon.select_one(selector)
                return element.get_text(" ", strip=True) if element else ""
            sets.append({"species": text(".name"), "item": text(".details .item"),
                         "ability": text(".ability").removeprefix("Ability: "),
                         "nature": text(".nature").removesuffix(" Nature"),
                         "moves": [m.get_text(" ", strip=True) for m in mon.select(".moves li")], "level": 50})
        if len(sets) != 6:
            continue
        header = block.select_one(".teamlist-toggle, .player, .tournament-teamlist-header, .decklist-header")
        # The sheet does not disclose spreads. Never attribute generated stats
        # to the tournament player or treat these as full expert battle requests.
        results.append({"sets": sets, "player": header.get_text(" ", strip=True) if header else "",
                        "event": event, "metadata": {"stat_points": "not_disclosed", "kind": "public_team_sheet"}})
    return results


async def tournament_feed(store: Store, fmt: str, max_teams: int = 12):
    from ml.research import search
    from ml.simulator import validate_team
    results = await search("Champions " + fmt[-3:].upper() + " winning teams", 8)
    imported, errors = [], []
    for resource in results["results"]:
        if not re.fullmatch(r"https://limitlessvgc\.com/tournaments/\d+", resource["url"]):
            continue
        url = resource["url"] + "/teams"
        try:
            body, _ = await fetch(url, max_bytes=6_000_000)
            sheets = parse_limitless(body, fmt)
            await ingest(store, url, fmt, max_bytes=6_000_000)
            for sheet in sheets[:max_teams-len(imported)]:
                validation = await validate_team(fmt, sheet["sets"])
                if validation["errors"]:
                    errors.append({"url": url, "error": "simulator rejected team sheet", "details": validation["errors"][:3]})
                    continue
                entry = import_team(store, sheet["sets"], fmt, url, sheet["player"], sheet["event"])
                with store.db:
                    store.db.execute("INSERT OR REPLACE INTO team_metadata VALUES (?,?)", (entry["id"], json.dumps(sheet["metadata"])))
                imported.append(entry["id"])
            if len(imported) >= max_teams:
                break
        except (ValueError, httpx.HTTPError) as error:
            errors.append({"url": url, "error": str(error)[:300]})
    return {"source": "limitless", "imported_teams": len(imported), "errors": errors,
            "note": "public team sheets; undisclosed spreads remain marked unknown"}


async def replay_feed(store: Store, fmt: str, limit: int = 24):
    url = "https://replay.pokemonshowdown.com/search.json"
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "pokemon-showdown-player/1.0"}) as client:
        response = await client.get(url, params={"format": fmt})
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError("replay index exceeds byte budget")
        index = response.json()
        if isinstance(index, dict):
            index = index.get("replays", [])
        imported = []
        for replay in index[:limit]:
            replay_id = replay.get("id", "")
            if not re.fullmatch(re.escape(fmt) + r"-\d+", replay_id):
                continue
            if store.db.execute("SELECT 1 FROM public_battles WHERE id=?", ("replay-"+replay_id,)).fetchone():
                continue
            replay_url = "https://replay.pokemonshowdown.com/" + replay_id + ".json"
            body, _ = await fetch(replay_url, max_bytes=2_000_000)
            data = json.loads(body)
            result = ingest_public(store, fmt, data.get("log", ""), replay_url, "replay-"+replay_id)
            if result.get("imported"):
                imported.append(result)
    return {"source": "showdown-replays", "imported_battles": len(imported),
            "samples": sum(r["samples"] for r in imported)}


async def archive_feed(store: Store, key: str, max_games: int = 500, deadline: float | None = None):
    spec = ARCHIVES[key]
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        async with client.stream("GET", spec["url"]) as response:
            response.raise_for_status()
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > spec["max_bytes"]:
                    raise ValueError("archive exceeds approved byte budget")
                chunks.append(chunk)
    import hashlib
    body = b"".join(chunks)
    if hashlib.sha256(body).hexdigest() != spec["sha256"]:
        raise ValueError("archive checksum does not match approved dataset revision")
    data = json.loads(body)
    count, samples = 0, 0
    partial = False
    for battle_id, entry in list(data.items())[:max_games]:
        import time
        if deadline is not None and time.monotonic() >= deadline:
            partial = True
            break
        log = entry[1] if isinstance(entry, list) else entry.get("log", "")
        result = ingest_public(store, spec["format"], log, spec["card"], key+"-"+battle_id)
        count += bool(result.get("imported"))
        samples += result.get("samples", 0)
    return {"source": key, "format": spec["format"], "license": spec["license"],
            "bytes": size, "imported_battles": count, "samples": samples, "partial": partial,
            "note": "historical format, separate scout model; never relabeled as current M-C"}


async def refresh_feeds(store: Store, fmt: str, max_teams: int = 12, max_replays: int = 24):
    report = {"format": fmt, "checked": now(), "sources": []}
    for name, operation in (
        ("usage", lambda: refresh_usage(store, fmt, previous_month())),
        ("tournaments", lambda: tournament_feed(store, fmt, max_teams)),
        ("replays", lambda: replay_feed(store, fmt, max_replays)),
    ):
        try:
            result = await operation()
            if name == "usage":
                result = {"source": "smogon", "month": result["month"], "species": len(result["species"])}
            report["sources"].append({"name": name, "ok": True, **result})
        except (ValueError, httpx.HTTPError, KeyError) as error:
            report["sources"].append({"name": name, "ok": False, "error": str(error)[:300]})
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO feed_state VALUES (?,?,?)", ("daily-"+fmt, now(), json.dumps(report)))
    return report
