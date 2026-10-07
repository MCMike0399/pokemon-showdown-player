#!/usr/bin/env python3
"""Read-only campaign/learning inspection; never connects to Showdown or creates a model."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {"unavailable": str(error)}


def checkpoint(path):
    if not path.exists():
        return {"available": False}
    # Load tensors only. Instantiating Model would initialize/write missing weights.
    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=True)
        return {"path": str(path), "revision": data.get("revision"),
                "updates": data.get("updates"), "samples": data.get("samples"),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    except Exception as error:
        return {"path": str(path), "unavailable": str(error)}


def snapshot(root, fmt, since=None):
    output = {"checked_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
              "format": fmt, "since": since,
              "actor": checkpoint(root / "models" / (fmt + ".pt")),
              "scout": checkpoint(root / "scouts" / (fmt + ".pt")),
              "config": read_json(root / "autopilot.json"),
              "compute": read_json(root / "compute-benchmark.json")}
    try:
        import psutil
        output["host"] = {"cpu_percent": psutil.cpu_percent(interval=0.1),
                          "available_gb": round(psutil.virtual_memory().available / 2**30, 2),
                          "disk_free_gb": round(psutil.disk_usage(PROJECT).free / 2**30, 2)}
        output["processes"] = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                argv = process.info["cmdline"] or []
                entries = [arg for arg in argv if arg in ("ml.worker", "ml.cli") or
                           (Path(arg).name == "ps_mcp_server.py" and len(arg) < 500)]
                if entries:
                    output["processes"].append({"pid": process.pid, "name": process.info["name"],
                                                "entrypoints": entries})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        output["host"] = {"unavailable": "psutil is not installed"}
    latest = root / "campaigns" / "latest.json"
    if latest.exists():
        output["campaign"] = read_json(latest)
        directory = output["campaign"].get("directory")
        if directory:
            output["campaign_status"] = read_json(Path(directory) / "status.json")
    path = root / "experience.sqlite3"
    if not path.exists():
        output["database"] = "not present; no database created"
        return output
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        # One consistent database view while the live player/worker writes WAL.
        db.execute("BEGIN")
        clause, args = "format=?", [fmt]
        if since:
            clause += " AND julianday(created)>=julianday(?)"
            args.append(since)
        output["episodes"] = [dict(r) for r in db.execute(f"""
            SELECT source,status,COUNT(*) games,SUM(outcome=1) wins,
                   SUM(outcome=-1) losses,SUM(outcome=0) ties,SUM(trained) consumed,
                   SUM(json_array_length(json_extract(data,'$.steps'))) decisions
            FROM episodes WHERE {clause} GROUP BY source,status
        """, args)]
        for row in output["episodes"]:
            if row["status"] == "complete":
                n, wins = row["games"], row["wins"] or 0
                rate, z = wins / n, 1.96
                center = (rate + z*z/(2*n)) / (1 + z*z/n)
                half = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
                row.update(win_rate=rate, win_rate_95_wilson=[center-half, center+half])
        output["recent_ladder"] = [dict(r) for r in db.execute(f"""
            SELECT id,revision,team,created,status,outcome,trained,
                   json_extract(data,'$.room') room,
                   json_array_length(json_extract(data,'$.steps')) decisions,
                   json_extract(data,'$.on_policy') on_policy,
                   json_extract(data,'$.demonstration') demonstration
            FROM episodes WHERE {clause} AND source='ladder' ORDER BY created DESC LIMIT 10
        """, args)]
        output["ladder_by_revision"] = [dict(r) for r in db.execute(f"""
            SELECT revision,team,COUNT(*) games,SUM(outcome=1) wins,
                   SUM(outcome=-1) losses,SUM(outcome=0) ties
            FROM episodes WHERE {clause} AND source='ladder' AND status='complete'
            GROUP BY revision,team
        """, args)]
        output["jobs"] = [dict(r) for r in db.execute(f"""
            SELECT status,COUNT(*) jobs,
                   SUM(json_extract(result,'$.error') IS NOT NULL) with_error
            FROM jobs WHERE {clause} GROUP BY status
        """, args)]
        output["recent_jobs"] = []
        for row in db.execute(f"""
            SELECT id,kind,status,attempts,created,updated,lease_until,result
            FROM jobs WHERE {clause} ORDER BY updated DESC LIMIT 10
        """, args):
            record = dict(row)
            result = json.loads(record.pop("result") or "{}")
            record["result"] = {key: result[key] for key in (
                "error", "reason", "deferred", "training", "evaluation", "promoted",
                "promotion_blocked_by_live_game", "backend", "scout") if key in result}
            if record["status"] == "running":
                record["lease_expired"] = record["lease_until"] < time.time()
            output["recent_jobs"].append(record)
        output["scout_data"] = dict(db.execute("""
            SELECT COUNT(*) samples,COUNT(DISTINCT battle) battles
            FROM scout_samples WHERE format=?
        """, (fmt,)).fetchone())
        output["feeds"] = [dict(r) for r in db.execute("SELECT id,checked FROM feed_state ORDER BY checked DESC")]
        output["limitations"] = [
            "Episode counts include existing experience unless --since is supplied; use campaign room IDs for exact attribution.",
            "Local training outcomes and actor update counts do not measure held-out playing strength.",
            "Rejected proposals, tool latency and agent liveness need campaign events; episodes retain accepted trajectories.",
            "A job marked complete can contain an error; consumed episodes do not imply a promoted candidate.",
            "Scout sample counts and configuration are current totals, not filtered by --since.",
        ]
        return output
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "ml")
    parser.add_argument("--format", default=FORMAT)
    parser.add_argument("--since", help="ISO timestamp; filter episode/job creation times")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="repeat until interrupted (minimum 5 seconds)")
    args = parser.parse_args()
    if args.since:
        try:
            stamp = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                parser.error("--since must include a timezone")
            args.since = stamp.astimezone(timezone.utc).isoformat()
        except ValueError:
            parser.error("--since must be an ISO timestamp")
    if args.watch is not None and args.watch < 5:
        parser.error("--watch must be at least 5 seconds")
    try:
        while True:
            print(json.dumps(snapshot(args.root.resolve(), args.format, args.since), indent=2), flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
