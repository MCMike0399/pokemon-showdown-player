"""Real-mechanics offline training, evaluation and self-play without live accounts."""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from pathlib import Path

from battle_state import legal_choices, observe
from ml.brain import Brain
from ml.features import Features
from ml.teams import team_id

BRIDGE = Path(__file__).resolve().parents[1] / "simulator.cjs"


def simulator_seed(seed: int, encoding: str = 'legacy') -> list[int]:
    if encoding == 'legacy':
        return [seed % 65536, 7, 13, 19]
    if encoding != 'full-v1' or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError('full-v1 requires an unsigned 64-bit integer seed')
    return [seed & 65535, ((seed >> 16) & 65535) ^ 7,
            ((seed >> 32) & 65535) ^ 13, ((seed >> 48) & 65535) ^ 19]


async def bridge_query(payload: dict) -> dict:
    process = await asyncio.create_subprocess_exec("node", str(BRIDGE), stdin=asyncio.subprocess.PIPE,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, errors = await asyncio.wait_for(process.communicate((json.dumps(payload) + "\n").encode()), 30)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode or not output:
        raise RuntimeError(errors.decode()[:2000] or output.decode()[:2000] or "simulator returned no output; run npm install && npm run simulator:build")
    response = json.loads(output.splitlines()[0])
    if "error" in response:
        raise ValueError(response["error"])
    return response


async def validate_team(fmt: str, sets: list[dict]) -> dict:
    return await bridge_query({"type": "validate", "format": fmt, "sets": sets})


async def load_dex(fmt: str) -> Features:
    response = await bridge_query({"type": "dex", "format": fmt})
    root = BRIDGE.parent / "cache" / "dex" / fmt
    root.mkdir(parents=True, exist_ok=True)
    for name, data in response["dex"].items():
        (root / f"{name}.json").write_text(json.dumps(data))
    return Features(response["dex"])


async def play_local(brain: Brain, fmt: str, team1=None, team2=None, opponent: str = "heuristic",
                     seed: int = 0, training: bool = True, max_decisions: int = 1000,
                     teacher: bool = False, opponent_model=None, learner_side: str = "p1",
                     sample_actions: bool = False, open_team_sheets: bool = True, seed_encoding: str = 'legacy') -> dict:
    if opponent not in ("heuristic", "random", "self", "tactical"):
        raise ValueError("opponent must be heuristic, random, self or tactical")
    if not isinstance(open_team_sheets, bool):
        raise ValueError('open_team_sheets must be boolean')
    seed_words = simulator_seed(seed, seed_encoding)
    if learner_side not in ("p1", "p2"):
        raise ValueError("learner side must be p1 or p2")
    process = await asyncio.create_subprocess_exec("node", str(BRIDGE), stdin=asyncio.subprocess.PIPE,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    room = "local-" + uuid.uuid4().hex
    logs = {"p1": [], "p2": []}
    rng = random.Random(seed)
    result = {"room": room, "ongoing": True}
    decisions = 0
    rejected = 0
    unavailable = 0
    consecutive_unavailable = 0
    model = brain.model(fmt)
    opponent_brain = None
    if opponent == 'self':
        opponent_brain = Brain(brain.store, brain.features)
        opponent_brain.models[fmt] = opponent_model or model

    async def send(payload):
        process.stdin.write((json.dumps(payload) + "\n").encode())
        await process.stdin.drain()

    try:
        await send({"type": "start", "format": fmt, "team1": team1 if learner_side == "p1" else team2,
                    "team2": team2 if learner_side == "p1" else team1, "learnerSide": learner_side,
                    "seed": seed_words, "openTeamSheets": open_team_sheets})
        while decisions < max_decisions:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
            if not line:
                errors = (await process.stderr.read()).decode()
                raise RuntimeError("simulator terminated before battle end: " + errors[:1000])
            event = json.loads(line)
            if "error" in event:
                raise ValueError(event["error"])
            side = event["side"]
            logs[side].extend(event.get("lines", []))
            errors = [line for line in event.get('lines', []) if line.startswith('|error|')]
            if not errors:
                consecutive_unavailable = 0
            for public in event.get("lines", []):
                if public.startswith("|error|"):
                    brain.reject(room, side)
                    # Hidden trapping/disabling is disclosed by the server's
                    # corrected request. Remove the unexecuted proposal and
                    # retry that observable mask, just as the live player does.
                    if (public.startswith('|error|[Unavailable choice]') and
                            legal_choices(event.get('request')) and consecutive_unavailable < 5):
                        unavailable += 1
                        consecutive_unavailable += 1
                        continue
                    rejected += 1
                    raise ValueError("simulator rejected generated action: " + public)
                if public.startswith("|win|"):
                    result = {"room": room, "winner": public.split("|", 2)[2]}
                elif public in ('|tie', '|tie|'):
                    result = {"room": room, "winner": None, "tie": True}
            if not result.get("ongoing"):
                break
            request = event.get("request")
            choices = legal_choices(request)
            if not choices:
                continue
            state = observe(request, logs[side], "LocalBrain" if side == learner_side else "LocalOpponent", room)
            sets = team1 if side == learner_side else team2
            ctx = {"room": room, "format": fmt, "request": request, "state": state, "choices": choices,
                   "source": "local", "team_id": team_id(fmt, sets) if sets else "random-generated",
                   'team_sets': sets, 'simulation': {'seed': seed, 'learner_side': learner_side,
                       'seed_encoding': seed_encoding, 'simulator_seed': seed_words,
                       'opponent': opponent, 'open_team_sheets': open_team_sheets,
                       'opponent_team': team_id(fmt, team2) if team2 else 'random-generated',
                       'opponent_revision': opponent_model.revision if opponent_model is not None else None},
                   "public_log": logs[side]}
            if side == learner_side and teacher:
                choice = max(choices, key=lambda c: brain.features.action(ctx, c)[-1])
                brain.decide(ctx, demonstration=choice)
            elif side == learner_side:
                choice = brain.decide(ctx, explore=training or sample_actions, record=training)["choice"]
            elif opponent == "random":
                choice = rng.choice(choices)
            elif opponent == "self":
                choice = opponent_brain.decide(ctx, explore=training, record=False)['choice']
            elif opponent == 'tactical':
                from ml.preview import score as preview_score
                if request.get('teamPreview'):
                    choice = max(choices, key=lambda c: preview_score(ctx, c, brain.features))
                else:
                    tactical_ctx = {**ctx, 'feature_profile': 'mechanics-v1'}
                    choice = max(choices, key=lambda c: brain.features.action(tactical_ctx, c)[-1])
            else:
                # A separate frozen scripted opponent; never updated by training.
                choice = max(choices, key=lambda c: brain.features.action(ctx, c)[-1])
            await send({"type": "choose", "side": side, "choice": choice})
            decisions += 1
        if result.get("ongoing"):
            result = {"room": room, "unfinished": True}
        else:
            brain.finish(room, result, "LocalBrain", learner_side)
            if training or teacher:
                from ml.scout import ingest_public
                result["public_experience"] = ingest_public(brain.store, fmt, logs[learner_side], "local-simulation", room,
                                                            features=brain.features)
        result.update({"decisions": decisions, "rejected_actions": rejected, "seed": seed,
                       "opponent": opponent, "learner_side": learner_side, "revision": model.revision})
        result["policy_mode"] = "sampled" if training or sample_actions else "greedy"
        result["unavailable_choices"] = unavailable
        result.update(open_team_sheets=open_team_sheets,
                      seed_encoding=seed_encoding, simulator_seed=seed_words,
                      learner_team=team_id(fmt, team1) if team1 else 'random-generated',
                      opponent_team=team_id(fmt, team2) if team2 else 'random-generated')
        if opponent_model is not None:
            result['opponent_revision'] = opponent_model.revision
        return result
    finally:
        # Truncations/errors remain pending in SQLite with no reward. They must
        # not hold an in-memory episode open and block the next training batch.
        brain.pending.pop((room, learner_side), None)
        if process.returncode is None:
            process.terminate()
        await process.wait()
