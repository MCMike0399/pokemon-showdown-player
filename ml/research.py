"""Bounded web research with provenance, format filtering and structured team priors.

Search uses Brave when BRAVE_SEARCH_API_KEY exists; otherwise it searches live
primary-source tournament/team indexes. No API key is needed for those indexes.
Fetched prose is retrieved for the orchestrating agent, never executed or used
as a training label. The agent must explicitly extract/attribute team sets.
"""
from __future__ import annotations

import json
import os
import re
import asyncio
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from battle_state import to_id
from ml.storage import Store, fingerprint, now

SOURCE_DOMAINS = ("pokemon.com", "pokemon-home.com", "smogon.com", "pokemonshowdown.com",
                  "limitlessvgc.com", "victoryroad.pro", "labmaus.net", "pokepast.es")
HEADERS = {"User-Agent": "pokemon-showdown-player-research/1.0"}


def trusted(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return (parsed.scheme == "https" and parsed.port in (None, 443) and not parsed.username and
            any(host == domain or host.endswith("." + domain) for domain in SOURCE_DOMAINS))


async def fetch(url: str, max_bytes: int = 1_000_000) -> tuple[bytes, str]:
    if not trusted(url):
        raise ValueError("research URL must be HTTPS on a documented source domain")
    async with httpx.AsyncClient(timeout=20, headers=HEADERS, follow_redirects=False) as client:
        for _ in range(5):
            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    url = str(response.url.join(response.headers["location"]))
                    if not trusted(url):
                        raise ValueError("redirect left the research source domains")
                    continue
                response.raise_for_status()
                chunks = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("resource exceeds download size limit")
                    chunks.append(chunk)
                return b"".join(chunks), url
    raise ValueError("too many redirects")


async def search(query: str, limit: int = 8) -> dict:
    if not query.strip() or len(query) > 500 or not 1 <= limit <= 20:
        raise ValueError("provide a query under 500 characters and limit 1..20")
    key = os.environ.get("BRAVE_SEARCH_API_KEY")
    async with httpx.AsyncClient(timeout=20, headers=HEADERS) as client:
        if key:
            # Never persist/return API keys or authenticated request headers.
            response = await client.get("https://api.search.brave.com/res/v1/web/search",
                                        params={"q": query, "count": limit}, headers={"X-Subscription-Token": key})
            response.raise_for_status()
            results = [{"title": r.get("title", ""), "url": r["url"], "snippet": r.get("description", "")}
                       for r in response.json().get("web", {}).get("results", [])]
            provider = "brave"
        else:
            indexes = ("https://limitlessvgc.com/tournaments", "https://victoryroad.pro/champions-replica/",
                       "https://circuit.victoryroad.pro/")
            fetched = await asyncio.gather(*(fetch(url) for url in indexes), return_exceptions=True)
            results = []
            errors = []
            tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query.lower())) - {"pokemon", "pokémon", "teams", "team", "vgc", "the", "best", "latest", "winning"}
            seen = set()
            for url, resource in zip(indexes, fetched):
                if isinstance(resource, Exception):
                    errors.append({"url": url, "error": type(resource).__name__})
                    continue
                soup = BeautifulSoup(resource[0], "html.parser")
                for a in soup.find_all("a", href=True):
                    href = str(httpx.URL(url).join(a["href"]))
                    if not trusted(href) or href in seen:
                        continue
                    if not ("pokepast.es/" in href or re.search(r"/tournaments/\d+", href) or "/tournament/" in href):
                        continue
                    seen.add(href)
                    row = a.find_parent("tr") or a.parent
                    table = a.find_parent("table")
                    heading = table.find_previous(["h1", "h2", "h3"]) if table else None
                    snippet = ((heading.get_text(" ", strip=True) + ": ") if heading else "") + row.get_text(" ", strip=True)[:1000]
                    title = a.get_text(" ", strip=True) or snippet[:160] or "Credited tournament team"
                    words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (title + " " + snippet).lower()))
                    score = len(tokens & words)
                    results.append({"title": title, "url": href, "snippet": snippet,
                                    "discovered_on": url, "relevance": score})
            # Source order retains each operator's newest-first index for ties.
            results.sort(key=lambda r: r["relevance"], reverse=True)
            provider = "primary-source-indexes"
            if not results and errors:
                return {"provider": provider, "query": query, "searched_at": now(), "results": [], "errors": errors}
    return {"provider": provider, "query": query, "searched_at": now(),
            "results": [{**r, "supported_source": trusted(r["url"])} for r in results[:limit]]}


async def ingest(store: Store, url: str, fmt: str, max_bytes: int = 1_000_000) -> dict:
    if not fmt or fmt != to_id(fmt):
        raise ValueError("an explicit format id is required")
    body, final_url = await fetch(url, max_bytes=max_bytes)
    soup = BeautifulSoup(body, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else final_url
    published = soup.find("meta", attrs={"property": "article:published_time"})
    published_at = published.get("content") if published else None
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)[:100_000]
    doc_id = fingerprint({"url": final_url, "format": fmt})
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (doc_id, final_url, fmt, title, now(), published_at, text, fingerprint(text)))
    return {"id": doc_id, "url": final_url, "format": fmt, "title": title,
            "published_at": published_at, "characters": len(text),
            "note": "format is the caller's tag; inspect the source before importing sets"}


def retrieve(store: Store, query: str, fmt: str, limit: int = 5) -> list[dict]:
    tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    ranked = []
    for row in store.db.execute("SELECT * FROM documents WHERE format=?", (fmt,)):
        text = row["text"]
        words = Counter(re.findall(r"[a-z0-9]{3,}", text.lower()))
        score = sum(min(words[t], 5) for t in tokens)
        if tokens and score == 0:
            continue
        ranked.append({"url": row["url"], "title": row["title"], "published_at": row["published"],
                       "fetched_at": row["fetched"], "score": score, "excerpt": text[:4000]})
    return sorted(ranked, key=lambda d: (d["score"], d["fetched_at"]), reverse=True)[:limit]


def import_team(store: Store, sets: list[dict], fmt: str, url: str, player: str = "", event: str = "") -> dict:
    if not trusted(url):
        raise ValueError("team attribution needs a supported source URL")
    if not fmt or fmt != to_id(fmt) or not 1 <= len(sets) <= 6:
        raise ValueError("provide a format id and 1..6 complete sets")
    for mon in sets:
        if not mon.get("species") or not 1 <= len(mon.get("moves", [])) <= 4:
            raise ValueError("each set needs species and 1..4 moves")
    team_id = fingerprint({"format": fmt, "sets": sets})
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO research_teams VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (team_id, fmt, url, player, event, now(), json.dumps(sets)))
    return {"id": team_id, "format": fmt, "source": url, "player": player, "event": event,
            "note": "candidate only; simulator validation is required before play"}


def species_prior(store: Store, fmt: str, max_age_days: int = 90) -> dict[str, float]:
    counter = Counter()
    for row in store.db.execute("SELECT sets, fetched FROM research_teams WHERE format=?", (fmt,)):
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["fetched"])).days
        if age > max_age_days:
            continue
        for mon in json.loads(row["sets"]):
            counter[to_id(mon["species"])] += 1
    total = sum(counter.values()) or 1
    prior = {species: count / total for species, count in counter.most_common(32)}
    row = store.db.execute("SELECT text, fetched FROM documents WHERE format=? AND title LIKE 'Showdown usage %' ORDER BY published DESC LIMIT 1", (fmt,)).fetchone()
    if row and (datetime.now(timezone.utc) - datetime.fromisoformat(row["fetched"])).days <= max_age_days:
        for mon in json.loads(row["text"]).get("species", [])[:32]:
            prior[to_id(mon["species"])] = max(prior.get(to_id(mon["species"]), 0), float(mon["usage"]))
    return prior


async def refresh_usage(store: Store, fmt: str, month: str, rating: int = 1630) -> dict:
    if not re.fullmatch(r"\d{4}-\d{2}", month) or fmt != to_id(fmt) or rating not in (0, 1500, 1630, 1695, 1760):
        raise ValueError("provide YYYY-MM, a format id and a published rating cutoff")
    # User supplies month explicitly; never silently confuse current/previous
    # month or import stats from an unrelated regulation.
    url = f"https://www.smogon.com/stats/{month}/chaos/{fmt}-{rating}.json"
    # Gzip transfer size is much smaller than decoded JSON. Bound the decoded
    # resource, rather than trusting Content-Length from a compressed response.
    body, _ = await fetch(url, max_bytes=96_000_000)
    data = json.loads(body)
    title = f"Showdown usage {month} {fmt} {rating}"
    rows = []
    for species, stats in sorted(data.get("data", {}).items(), key=lambda p: p[1].get("usage", 0), reverse=True)[:50]:
        rows.append({"species": species, "usage": stats.get("usage", 0),
                     "moves": sorted(stats.get("Moves", {}), key=stats.get("Moves", {}).get, reverse=True)[:4],
                     "items": sorted(stats.get("Items", {}), key=stats.get("Items", {}).get, reverse=True)[:3]})
    text = json.dumps({"info": data.get("info", {}), "species": rows})
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (fingerprint(url), url, fmt, title, now(), month, text, fingerprint(text)))
    return {"url": url, "species": rows, "format": fmt, "month": month}


async def import_paste(store: Store, url: str, fmt: str, player: str = "", event: str = "") -> dict:
    """Import real complete sets with the official paste parser and validator."""
    if not re.fullmatch(r"https://pokepast\.es/[a-fA-F0-9]{16}/?", url):
        raise ValueError("provide a public pokepast.es paste URL")
    body, _ = await fetch(url.rstrip("/") + "/raw", max_bytes=100_000)
    from ml.simulator import bridge_query
    parsed = await bridge_query({"type": "import", "format": fmt, "text": body.decode()})
    if parsed["errors"]:
        raise ValueError("; ".join(parsed["errors"]))
    result = import_team(store, parsed["sets"], fmt, url, player, event)
    return {**result, "sets": parsed["sets"], "validated": True}
