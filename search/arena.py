"""Offline arena: any two agents over the pinned official simulator, no accounts.

Open team sheets are OFF by default (most live ladder opponents decline OTS), so
each side sees only its own request and its own public stream.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battle_state import legal_choices, observe  # noqa: E402
from ml.simulator import BRIDGE, simulator_seed  # noqa: E402

FMT = 'gen9championsvgc2026regmc'


class BrainAgent:
    """The incumbent neural policy, sampled at its checkpoint temperature like live play."""

    def __init__(self, root: Path, explore: bool = True, name: str = 'incumbent'):
        from ml.brain import Brain
        from ml.model import Model
        from ml.storage import Store
        store = Store.__new__(Store)
        import sqlite3
        store.root = Path(root)
        store.db = sqlite3.connect(f'file:{Path(root) / "experience.sqlite3"}?mode=ro', uri=True, check_same_thread=False)
        store.db.row_factory = sqlite3.Row
        self.brain = Brain(store)
        self.brain.models[FMT] = Model(Path(root), FMT)
        self.explore = explore
        self.name = name

    async def decide(self, ctx: dict) -> str:
        return self.brain.decide(ctx, explore=self.explore, record=False)['choice']

    def preview_fn(self):
        async def preview(ctx):
            return self.brain.decide(ctx, explore=self.explore, record=False)['choice']
        return preview

    async def close(self):
        pass


class RandomAgent:
    name = 'random'

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    async def decide(self, ctx):
        return self.rng.choice(ctx['choices'])

    async def close(self):
        pass


async def play_game(agents: dict, teams: dict, seed: int, ots: bool = False, max_decisions: int = 600,
                    timeout: float = 60) -> dict:
    """agents/teams keyed by 'p1'/'p2'. Returns winner side and diagnostics."""
    proc = await asyncio.create_subprocess_exec('node', str(BRIDGE), stdin=asyncio.subprocess.PIPE,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                limit=1 << 25)
    room = 'arena-' + uuid.uuid4().hex[:12]
    logs = {'p1': [], 'p2': []}
    names = {'p1': 'LocalBrain', 'p2': 'LocalOpponent'}
    result = {'room': room, 'seed': seed, 'winner': None, 'turns': 0, 'rejections': {'p1': 0, 'p2': 0},
              'decision_ms': {'p1': 0.0, 'p2': 0.0}, 'decisions': {'p1': 0, 'p2': 0}}
    started = time.time()

    async def send(payload):
        proc.stdin.write((json.dumps(payload) + '\n').encode())
        await proc.stdin.drain()

    pending_requests = {}
    try:
        await send({'type': 'start', 'format': FMT, 'team1': teams['p1'], 'team2': teams['p2'], 'learnerSide': 'p1',
                    'seed': simulator_seed(seed, 'full-v1'), 'openTeamSheets': ots})
        decisions = 0
        done = False
        while decisions < max_decisions and not done:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
            if not line:
                raise RuntimeError('simulator ended: ' + (await proc.stderr.read()).decode()[:500])
            event = json.loads(line)
            if 'error' in event:
                raise RuntimeError(event['error'])
            side = event['side']
            logs[side].extend(event.get('lines', []))
            rejected = False
            for public in event.get('lines', []):
                if public.startswith('|win|'):
                    winner_name = public.split('|', 2)[2]
                    result['winner'] = 'p1' if winner_name == names['p1'] else 'p2'
                    done = True
                elif public in ('|tie', '|tie|'):
                    result['winner'] = 'tie'
                    done = True
                elif public.startswith('|turn|'):
                    result['turns'] = int(public.split('|')[2])
                elif public.startswith('|error|'):
                    rejected = True
                    result['rejections'][side] += 1
            if done:
                break
            request = event.get('request')
            if request is not None:
                pending_requests[side] = request
            request = pending_requests.get(side)
            if not request or (event.get('request') is None and not rejected):
                continue
            choices = legal_choices(request)
            if not choices:
                continue
            ctx = {'room': room, 'format': FMT, 'request': request, 'choices': choices, 'source': 'local',
                   'team_sets': teams[side], 'public_log': list(logs[side]),
                   'state': observe(request, logs[side], names[side], room),
                   'team_id': 'arena'}
            t0 = time.perf_counter()
            if rejected and result['rejections'][side] > 3:
                choice = random.Random(seed + decisions).choice(choices)
            elif rejected:
                # Hidden disable/trap: retry with a different proposal.
                choice = random.Random(seed + decisions).choice(choices)
            else:
                choice = await agents[side].decide(ctx)
            result['decision_ms'][side] += (time.perf_counter() - t0) * 1000
            result['decisions'][side] += 1
            await send({'type': 'choose', 'side': side, 'choice': choice})
            decisions += 1
        if not done:
            result['winner'] = None
            result['unfinished'] = True
    finally:
        if proc.returncode is None:
            proc.terminate()
        await proc.wait()
    result['seconds'] = round(time.time() - started, 1)
    result['log_p1'] = logs['p1']
    return result
