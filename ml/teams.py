"""Team selection learns from outcomes; suggestions retain complete sourced sets."""
from __future__ import annotations

import json
import math
import random

from ml.storage import Store, fingerprint


def team_id(fmt: str, sets: list[dict]) -> str:
    return fingerprint({"format": fmt, "sets": sets})


def rank_teams(store: Store, teams, fmt: str, source: str = "ladder") -> list[dict]:
    candidates = {}
    for name in teams.list():
        team = teams.get(name)
        if team["format"] == fmt:
            key = team_id(fmt, team["sets"])
            candidates[key] = {"id": key, "name": name, "sets": team["sets"], "source_url": None}
    for row in store.db.execute("SELECT * FROM research_teams WHERE format=?", (fmt,)):
        candidates.setdefault(row["id"], {"id": row["id"], "name": row["player"] or row["event"] or row["id"],
                                          "sets": json.loads(row["sets"]), "source_url": row["url"]})
    evidence = {}
    for row in store.db.execute("SELECT team, outcome FROM episodes WHERE format=? AND source=? AND status='complete'", (fmt, source)):
        wins, losses, games = evidence.get(row["team"], (0.0, 0.0, 0))
        score = (row["outcome"] + 1) / 2
        evidence[row["team"]] = (wins + score, losses + 1 - score, games + 1)
    ranked = []
    for key, candidate in candidates.items():
        wins, losses, games = evidence.get(key, (0.0, 0.0, 0))
        a, b = 1 + wins, 1 + losses
        mean = a / (a + b)
        sigma = math.sqrt(a * b / ((a + b)**2 * (a + b + 1)))
        ranked.append({**candidate, "games": games, "posterior_win_rate": mean,
                       "uncertainty": sigma, "evidence_source": source,
                       "exploration_score": mean + sigma})
    return sorted(ranked, key=lambda t: (t["posterior_win_rate"], t["games"]), reverse=True)


def suggest_variants(store: Store, base: dict, max_suggestions: int = 10) -> list[dict]:
    fmt = base["format"]
    suggestions = []
    for row in store.db.execute("SELECT * FROM research_teams WHERE format=?", (fmt,)):
        sets = json.loads(row["sets"])
        for i, mon in enumerate(base["sets"]):
            for alternate in sets:
                if alternate["species"] != mon["species"] or alternate == mon:
                    continue
                candidate = [dict(m) for m in base["sets"]]
                candidate[i] = alternate
                items = [m.get("item", "") for m in candidate if m.get("item")]
                if len(items) != len(set(items)):
                    continue
                key = team_id(fmt, candidate)
                if any(s["id"] == key for s in suggestions):
                    continue
                suggestions.append({"id": key, "format": fmt, "sets": candidate,
                                    "changed_species": mon["species"], "source_url": row["url"],
                                    "reason": "alternative complete set from same-format research",
                                    "requires_validation": True})
                if len(suggestions) >= max_suggestions:
                    return suggestions
    return suggestions


def generated_spreads(sets: list[dict], fmt: str):
    """Materialize undisclosed spreads as explicitly generated experiment inputs."""
    from ml.features import Features
    features = Features.cached(fmt)
    result, generated = [], False
    for original in sets:
        mon = dict(original)
        if "evs" not in mon:
            generated = True
            points = 32 if "champions" in fmt else 252
            remaining = 2 if "champions" in fmt else 4
            stats = features.species(mon["species"]).get("baseStats", {})
            # Base species stats can be misleading for a special Mega set.
            # Undisclosed spreads follow the actual attacks, then nature/stats.
            from battle_state import to_id
            power = {"Physical": 0, "Special": 0}
            for move in mon.get("moves", []):
                data = features.dex.get("moves", {}).get(to_id(move), {})
                if data.get("category") in power:
                    power[data["category"]] += data.get("basePower", 0)
            if power["Physical"] != power["Special"]:
                main = "atk" if power["Physical"] > power["Special"] else "spa"
            elif mon.get("nature") in ("Modest", "Timid", "Quiet", "Mild", "Rash"):
                main = "spa"
            elif mon.get("nature") in ("Adamant", "Jolly", "Brave", "Lonely", "Naughty"):
                main = "atk"
            else:
                main = "atk" if stats.get("atk", 0) > stats.get("spa", 0) else "spa"
            if mon.get("nature") in ("Careful", "Calm", "Sassy", "Impish", "Bold", "Relaxed"):
                mon["evs"] = {"hp": points, "spd": points, "def": remaining}
            else:
                mon["evs"] = {main: points, "spe": points, "hp": remaining}
        result.append(mon)
    return result, generated


class TeamPlanner:
    """One harness-facing boundary for learned team choice, variants and validation."""
    def __init__(self, store: Store, teams):
        self.store, self.teams = store, teams

    async def plan(self, fmt: str, base: str = "", explore: bool = False,
                   source: str = "ladder", save_as: str = "", max_candidates: int = 8):
        from ml.simulator import validate_team
        if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
            raise ValueError("max_candidates must be an integer 1..32")
        ranked = rank_teams(self.store, self.teams, fmt, source)
        if base:
            original = self.teams.get(base)
            if original["format"] != fmt:
                raise ValueError("base team's format differs from requested format")
            ranked = [{"name": base, "sets": original["sets"], "source_url": None},
                      *suggest_variants(self.store, original, max_candidates-1)]
        valid, errors = [], []
        rng = random.Random()
        for candidate in ranked[:max_candidates]:
            sets, generated = generated_spreads(candidate["sets"], fmt)
            validation = await validate_team(fmt, sets)
            if validation["errors"]:
                errors.append({"name": candidate.get("name", candidate.get("id")), "errors": validation["errors"][:3]})
                continue
            key = team_id(fmt, sets)
            outcomes = [r[0] for r in self.store.db.execute("SELECT outcome FROM episodes WHERE format=? AND source=? AND team=? AND status='complete'", (fmt, source, key))]
            wins = sum((outcome+1)/2 for outcome in outcomes)
            a, b = 1+wins, 1+len(outcomes)-wins
            score = rng.betavariate(a, b) if explore else a/(a+b)
            metadata_row = self.store.db.execute("SELECT data FROM team_metadata WHERE id=?", (key,)).fetchone()
            provenance = json.loads(metadata_row[0]) if metadata_row else {}
            valid.append({"id": key, "name": candidate.get("name", key), "sets": sets, "format": fmt,
                          "source_url": candidate.get("source_url") or provenance.get("source_url"), "games": len(outcomes),
                          "posterior_win_rate": a/(a+b), "selection_score": score, "validated": True,
                          "stat_points": "generated_hypothesis" if generated else provenance.get("stat_points", "provided"),
                          "provenance": provenance,
                          "packed": validation["packed"]})
        if not valid:
            raise ValueError("no legal team candidates; create/import a team first: " + json.dumps(errors)[:500])
        valid.sort(key=lambda c: (c["selection_score"], c["games"]), reverse=True)
        selected = valid[0]
        if save_as:
            if save_as in self.teams.list():
                existing = self.teams.get(save_as)
                if team_id(existing["format"], existing["sets"]) != selected["id"]:
                    raise ValueError("save_as already names a different team")
            else:
                self.teams.create(save_as, selected["sets"], fmt)
            selected["saved_as"] = save_as
        return {"selected": selected, "alternatives": [{k:v for k,v in c.items() if k != "packed"} for c in valid[1:]],
                "rejected_candidates": errors, "evidence_source": source,
                "team_model": "Beta-Bernoulli outcome bandit", "exploration": explore,
                "note": "generated spreads are experimental harness choices, never attributed expert spreads"}
