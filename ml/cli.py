"""CLI for local learning, evaluation, demonstrations, research and live play."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ml.storage import DEFAULT_ROOT, Store, fingerprint, now


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="experience/checkpoint directory")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("compute")
    s = sub.add_parser("feed")
    s.add_argument("--format", required=True)
    s.add_argument("--archive", choices=("vgc-bench-champions-mb",))
    s = sub.add_parser("continuous")
    s.add_argument("--disable", action="store_true")
    s.add_argument("--enable", action="store_true")
    s.add_argument("--run", action="store_true")
    for command in ("local", "evaluate", "bootstrap"):
        s = sub.add_parser(command)
        s.add_argument("--format", required=True)
        s.add_argument("--team1", type=Path)
        s.add_argument("--team2", type=Path)
        s.add_argument("--games", type=int, default=10)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--opponent", choices=("heuristic", "random", "self"), default="heuristic")
        s.add_argument("--alternate-sides", action="store_true")
    s = sub.add_parser("train")
    s.add_argument("--format", required=True)
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--imitation", action="store_true")
    s = sub.add_parser("import-demo")
    s.add_argument("path", type=Path, help="JSONL normalized decision contexts and demonstrated choices")
    s.add_argument("--format", required=True)
    s = sub.add_parser("validate")
    s.add_argument("path", type=Path)
    s.add_argument("--format", required=True)
    s = sub.add_parser("dex")
    s.add_argument("--format", required=True)
    s = sub.add_parser("research")
    s.add_argument("query")
    s.add_argument("--format", required=True)
    s.add_argument("--fetch", type=int, default=3)
    s = sub.add_parser("usage")
    s.add_argument("--format", required=True)
    s.add_argument("--month", required=True)
    s = sub.add_parser("import-team")
    s.add_argument("url")
    s.add_argument("--format", required=True)
    s.add_argument("--player", default="")
    s.add_argument("--event", default="")
    s.add_argument("--save-as", default="", help="also create a named harness team; never overwrite")
    s = sub.add_parser("play")
    s.add_argument("--format", required=True)
    s.add_argument("--team", required=True, help="stored TeamStore name")
    s.add_argument("--games", type=int, default=1)
    s.add_argument("--no-learn", action="store_true")
    return p


def read_team(path):
    data = json.loads(path.read_text()) if path else None
    return data.get("sets") if isinstance(data, dict) else data


async def main(args):
    store = Store(args.root)
    try:
        if args.command == "status":
            return store.stats()
        if args.command == "compute":
            from ml.resources import ResourcePolicy, benchmark
            from ml.continuous import LearningConfig
            policy = ResourcePolicy(**LearningConfig.load(store.root).resource)
            policy.apply_background()
            result = benchmark(policy)
            (store.root/"compute-benchmark.json").write_text(json.dumps(result, indent=2))
            return {"benchmark": result, "headroom": policy.sample()}
        if args.command == "feed":
            from ml.feeds import archive_feed, refresh_feeds, ARCHIVES
            from ml.simulator import load_dex
            if args.archive:
                await load_dex(ARCHIVES[args.archive]["format"])
                return await archive_feed(store, args.archive)
            return await refresh_feeds(store, args.format)
        if args.command == "continuous":
            from ml.continuous import LearningConfig
            from dataclasses import asdict
            config = LearningConfig.load(store.root)
            if args.enable and args.disable:
                raise ValueError("choose enable or disable")
            if args.enable or args.disable:
                config.enabled = args.enable
                config.save(store.root)
            if args.run:
                from ml.worker import run
                return await run(store.root, schedule=True)
            return asdict(config)
        from ml.brain import Brain
        from ml.simulator import load_dex, play_local, validate_team
        if args.command == "validate":
            return await validate_team(args.format, read_team(args.path))
        if args.command == "dex":
            features = await load_dex(args.format)
            return {"format": args.format, "tables": {k: len(v) for k, v in features.dex.items()}}
        brain = Brain(store)
        if args.command == "train":
            return brain.train(args.format, args.epochs, args.imitation)
        if args.command == "import-demo":
            brain.features = await load_dex(args.format)
            brain.explicit_features = True
            count = 0
            # Normalized decision contexts retain the legal request mask. A public
            # replay is not equivalent: it usually omits private requests.
            for line in args.path.read_text().splitlines():
                example = json.loads(line)
                if example["format"] != args.format:
                    raise ValueError("demonstration format mismatch")
                ctx = example["context"]
                ctx.update({"room": "demo-" + fingerprint(example), "format": args.format, "source": example.get("source", "human-demonstration")})
                from battle_state import legal_choices, observe
                ctx["choices"] = legal_choices(ctx["request"])
                ctx.setdefault("state", observe(ctx["request"], ctx.get("log", [])))
                brain.decide(ctx, demonstration=example["choice"])
                side = ctx["request"].get("side", {}).get("id", "p1")
                brain.finish(ctx["room"], {"tie": True}, "demonstrator", side)
                count += 1
            return {"imported": count, "note": "train --imitation to consume these labels"}
        if args.command in ("local", "evaluate", "bootstrap"):
            if not 1 <= args.games <= 10000:
                raise ValueError("games must be 1..10000")
            brain.features = await load_dex(args.format)
            brain.explicit_features = True
            teacher = args.command == "bootstrap"
            training = args.command == "local"
            team1, team2 = read_team(args.team1), read_team(args.team2)
            record = {"format": args.format, "mode": args.command, "opponent": args.opponent,
                      "initial_revision": brain.model(args.format).revision, "alternate_sides": args.alternate_sides,
                      "wins": 0, "losses": 0, "ties": 0, "unfinished": 0, "games": []}
            # Freeze an opponent snapshot for the whole run instead of moving
            # the evaluation target as the learning model updates.
            opponent_model = None
            if args.opponent == "self":
                import copy
                opponent_model = copy.deepcopy(brain.model(args.format))
            for i in range(args.games):
                result = await play_local(brain, args.format, team1, team2, args.opponent,
                                          args.seed + i, training=training, teacher=teacher, opponent_model=opponent_model,
                                          learner_side="p2" if args.alternate_sides and i % 2 else "p1")
                record["games"].append(result)
                bucket = "unfinished" if result.get("unfinished") else "ties" if result.get("tie") else "wins" if result.get("winner") == "LocalBrain" else "losses"
                record[bucket] += 1
                if training:
                    result["training"] = brain.train(args.format)
                print(json.dumps({"game": i + 1, "result": result}), flush=True)
            if teacher:
                record["training"] = brain.train(args.format, imitation=True)
            n = record["wins"] + record["losses"] + record["ties"]
            if n:
                import math
                rate = record["wins"] / n
                center = (rate + 1.96**2 / (2*n)) / (1 + 1.96**2 / n)
                half = 1.96 * math.sqrt(rate*(1-rate)/n + 1.96**2/(4*n*n)) / (1+1.96**2/n)
                record["win_rate"] = rate
                record["win_rate_95_interval"] = [max(0, center-half), min(1, center+half)]
            report = store.root / f"{args.command}-{args.format}-{fingerprint(now())}.json"
            report.write_text(json.dumps(record, indent=2))
            return {**record, "report": str(report)}
        if args.command == "research":
            from ml.research import ingest, search
            if not 0 <= args.fetch <= 10:
                raise ValueError("fetch must be 0..10")
            results = await search(args.query)
            resources = []
            for resource in [r for r in results["results"] if r["supported_source"]][:args.fetch]:
                try:
                    resources.append(await ingest(store, resource["url"], args.format))
                except (ValueError, OSError) as error:
                    resources.append({"url": resource["url"], "error": str(error)})
            return {**results, "resources": resources}
        if args.command == "usage":
            from ml.research import refresh_usage
            return await refresh_usage(store, args.format, args.month)
        if args.command == "import-team":
            from ml.research import import_paste
            result = await import_paste(store, args.url, args.format, args.player, args.event)
            if args.save_as:
                from harness import TeamStore
                TeamStore().create(args.save_as, result["sets"], args.format)
                result["saved_as"] = args.save_as
            return result
        if args.command == "play":
            from harness import Player
            from ml.live import LiveSession
            if not 1 <= args.games <= 100:
                raise ValueError("live games must be 1..100")
            p = Player()
            await load_dex(args.format)
            session = LiveSession(brain, p)
            played = []
            try:
                login = await p.login()
                if not login.get("loggedIn"):
                    raise ValueError("configured account login failed")
                sets = p.teams.get(args.team)["sets"]
                validation = await validate_team(args.format, sets)
                if validation["errors"]:
                    raise ValueError("; ".join(validation["errors"]))
                results = []
                for _ in range(args.games):
                    await p.ladder(args.format, validation["packed"])
                    room = await p.wait_for_request(timeout=120, exclude=played)
                    if not room:
                        await p.cancel()
                        raise TimeoutError("matchmaking timed out")
                    played.append(room)
                    result = await session.play(room, args.format, args.team, learn=not args.no_learn)
                    results.append(result)
                    if result["result"].get("unfinished"):
                        break
                return {"games": results}
            finally:
                await p.close()
        raise ValueError("unknown command")
    finally:
        store.close()


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(asyncio.run(main(args)), indent=2))
