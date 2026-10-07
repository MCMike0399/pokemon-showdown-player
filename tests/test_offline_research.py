import asyncio
import json

import pytest

from ml.research import import_team, ingest, retrieve, search, species_prior, trusted, import_paste
from ml.storage import Store, now


FMT = "gen9championsvgc2026regmc"


def test_source_domains_and_redirect_constraints():
    assert trusted("https://limitlessvgc.com/tournaments/428")
    assert trusted("https://champions-news.pokemon-home.com/en/page/816.html")
    assert not trusted("https://limitlessvgc.com.attacker.example/")
    assert not trusted("http://limitlessvgc.com/")
    assert not trusted("https://localhost/")
    assert not trusted("https://127.0.0.1/")
    assert not trusted("https://secret@smogon.com/")


def test_research_provenance_format_and_age_filters(tmp_path, monkeypatch):
    store = Store(tmp_path)
    async def fake_fetch(url, max_bytes=1_000_000):
        return b'<title>Rain VGC</title><meta property="article:published_time" content="2026-09-20"><script>untrusted script</script><article>Rain team Pelipper Tailwind</article>', url
    monkeypatch.setattr("ml.research.fetch", fake_fetch)
    doc = asyncio.run(ingest(store, "https://victoryroad.pro/team/", FMT))
    matches = retrieve(store, "Pelipper", FMT)
    assert matches[0]["url"] == doc["url"]
    assert matches[0]["published_at"] == "2026-09-20"
    assert "untrusted script" not in matches[0]["excerpt"]
    assert retrieve(store, "Pelipper", "gen9ou") == []
    sets = [{"species": "Pelipper", "moves": ["Hurricane", "Protect"]}]
    import_team(store, sets, FMT, "https://pokepast.es/example", "Trainer", "Event")
    assert species_prior(store, FMT) == {"pelipper": 1.0}
    assert species_prior(store, "gen9ou") == {}
    with store.db:
        store.db.execute("UPDATE research_teams SET fetched='2025-01-01T00:00:00+00:00'")
    assert species_prior(store, FMT) == {}
    store.close()


def test_index_search_returns_credited_resources_without_api_key(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    async def fake_fetch(url):
        return b'<table><tr><td>2026 Champions M-C Winner Alice</td><td><a href="https://pokepast.es/abc">Team</a></td></tr></table>', url
    monkeypatch.setattr("ml.research.fetch", fake_fetch)
    result = asyncio.run(search("Champions M-C Alice"))
    assert result["provider"] == "primary-source-indexes"
    assert result["results"][0]["url"] == "https://pokepast.es/abc"
    assert "Alice" in result["results"][0]["snippet"]
    assert result["results"][0]["supported_source"]


def test_paste_import_requires_official_validation_and_preserves_attribution(tmp_path, monkeypatch):
    store = Store(tmp_path)
    async def fake_fetch(url, max_bytes=0):
        return b"Pelipper @ Focus Sash\nAbility: Drizzle\n- Protect", url
    async def valid_parser(payload):
        return {"errors": [], "sets": [{"species": "Pelipper", "item": "Focus Sash", "moves": ["Protect"]}]}
    monkeypatch.setattr("ml.research.fetch", fake_fetch)
    monkeypatch.setattr("ml.simulator.bridge_query", valid_parser)
    url = "https://pokepast.es/1234567890abcdef"
    result = asyncio.run(import_paste(store, url, FMT, "Alice", "Event"))
    assert result["validated"] and result["source"] == url and result["player"] == "Alice"
    async def invalid_parser(payload):
        return {"errors": ["illegal move"], "sets": None}
    monkeypatch.setattr("ml.simulator.bridge_query", invalid_parser)
    with pytest.raises(ValueError, match="illegal move"):
        asyncio.run(import_paste(store, url, FMT))
    assert store.db.execute("SELECT COUNT(*) FROM research_teams").fetchone()[0] == 1
    store.close()
