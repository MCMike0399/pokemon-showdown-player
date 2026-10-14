"""Imitation opponent: plays like a ladder human, sampled from the learned policy.

An arena/gate opponent on the deployment distribution (docs/human-opponent.md):
the policy trained on public replays scores each legal joint action of a position
rebuilt as the player sees it (own exact sets from the request, the other side
sampled from priors), and the agent samples with a temperature. Team preview
comes from the supplied `preview` callable (the frozen incumbent, as in the
gate); forced switches use a one-world search, which the policy does not cover.
"""
from __future__ import annotations

import math
import random

from battle_state import to_id
from search.agent import SearchAgent
from search.human.compare_live import load_net, logits
from search.tracker import Tracker


class ImitationAgent:
    name = 'imitation'

    def __init__(self, model_path, fmt='gen9championsvgc2026regmc', preview=None, seed: int = 0, temp: float = 1.0,
                 sample: bool = True, preview_model=None, preview_temp: float = 1.0, search_topk: int = 0):
        self.net = load_net(model_path)
        self.model_path = str(model_path)
        self.fmt = fmt
        self.preview = preview
        # A human preview policy (search/human/preview_model.py) replaces `preview` when given.
        self.preview_policy = None
        if preview_model:
            from search.human.preview_model import PreviewPolicy
            self.preview_policy = PreviewPolicy(preview_model, temp=preview_temp, seed=seed)
        self.temp = temp
        self.sample = sample
        # search_topk > 0: among the policy's k likeliest actions, play the one the
        # one-world search values best ("a human who picks sensibly among their habits").
        self.search_topk = int(search_topk or 0)
        self.rng = random.Random(seed)
        # Worlds and forced switches reuse the search agent's builders (one world, one engine).
        self.helper = SearchAgent(fmt=fmt, worlds=1, engines=1, seed=seed, preview=preview, screen=None)
        self.stats = {'decisions': 0, 'fallbacks': 0, 'errors': [], 'sims': 0, 'ms': 0.0}

    def reseed(self, seed):
        self.rng = random.Random(seed)
        self.helper.reseed(seed)
        if self.preview_policy is not None:
            self.preview_policy.rng = random.Random(seed + 11)

    async def close(self):
        await self.helper.close()

    async def decide(self, ctx: dict) -> str:
        request = ctx['request']
        choices = ctx['choices']
        if len(choices) == 1:
            return choices[0]
        if request.get('teamPreview'):
            if self.preview_policy is not None:
                tracker = Tracker(request['side']['id']).feed(ctx.get('public_log') or [])
                return self.preview_policy.choose(request, tracker.preview.get(tracker.foe) or [], choices, sample=self.sample)
            return await self.preview(ctx) if self.preview is not None else choices[0]
        self.stats['decisions'] += 1
        try:
            if request.get('forceSwitch'):
                return await self.helper._force_switch(ctx)
            return await self._move(ctx)
        except Exception as error:
            self.stats['fallbacks'] += 1
            self.stats['errors'].append(repr(error)[:200])
            try:  # keep the failing position for diagnosis (private artifacts)
                import json
                import time
                from pathlib import Path
                d = Path(__file__).resolve().parents[2] / 'artifacts' / 'human-policy' / 'imitation-errors'
                d.mkdir(parents=True, exist_ok=True)
                (d / f'{int(time.time() * 1000)}.json').write_text(json.dumps({'error': repr(error)[:300], 'request': request,
                                                                             'choices': choices, 'public_log': ctx.get('public_log'),
                                                                             'team_sets': ctx.get('team_sets')}))
            except Exception:
                pass
            return choices[0]

    async def _move(self, ctx: dict) -> str:
        import numpy as np
        request = ctx['request']
        side = request['side']['id']
        tracker = Tracker(side).feed(ctx.get('public_log') or [])
        if any(a.get('canMegaEvo') for a in request.get('active') or []):
            tracker.mega_used[side] = False
        world = {'p1': self.helper._our_side(ctx, tracker), 'p2': self.helper._their_side(tracker, self.helper.rng)}
        field = self.helper._field(tracker)
        out = await self.helper.engines[0].call({'type': 'human', 'format': self.fmt, 'field': field, 'worlds': [world]}, timeout=60)
        legal = set(ctx['choices'])
        # The engine enumerates moves by its rebuilt move slots; the request lists only the
        # usable moves (one when locked), so translate each part by move id (as the search does).
        active = request.get('active') or []
        # Legal parts per slot: a locked two-turn move or Recharge is listed without a
        # target, so a translated part drops its target when that is the legal form.
        legal_parts = [set() for _ in range(max(1, len(active)))]
        for c in ctx['choices']:
            for slot, part in enumerate(x.strip() for x in c.split(',')):
                if slot < len(legal_parts):
                    legal_parts[slot].add(part)
        translated = []
        for c in out['choices']:
            parts = []
            for slot, part in enumerate(x.strip() for x in c.split(',')):
                words = part.split()
                if words and words[0] == 'move' and slot < len(active):
                    j = int(words[1]) - 1
                    ids = out['moves'][slot] if slot < len(out['moves']) else []
                    req_ids = [to_id(m.get('id') or m.get('move')) for m in active[slot].get('moves') or []]
                    if j >= len(ids) or ids[j] not in req_ids:
                        parts = None
                        break
                    head = ['move', str(req_ids.index(ids[j]) + 1)]
                    part = ' '.join(head + words[2:])
                    if part not in legal_parts[slot]:
                        untargeted = ' '.join(head + [w for w in words[2:] if not w.lstrip('-').isdigit()])
                        if untargeted in legal_parts[slot]:
                            part = untargeted
                parts.append(part)
            translated.append(', '.join(parts) if parts is not None else None)
        idx = [i for i, c in enumerate(translated) if c in legal]
        if not idx:
            raise ValueError('no enumerated action is legal')
        X = np.concatenate([np.repeat(np.asarray(out['sx'])[None, :], len(idx), 0), np.asarray(out['a'])[idx]], axis=1)
        z = logits(self.net, X) / self.temp
        z = z - z.max()
        p = np.exp(z); p /= p.sum()
        if self.search_topk:
            top = [idx[k] for k in np.argsort(-p)[:self.search_topk]]
            cands = [out['choices'][k] for k in top]
            if len(cands) == 1:
                return cands[0]
            sim_map = {}
            active = request.get('active') or []
            for c in cands:
                sim_map.setdefault(self.helper._to_sim(c, active, tracker), c)
            results = await self.helper._search(ctx, tracker, world['p1'], list(sim_map))
            totals = {}
            for r in results:
                if 'error' in r:
                    continue
                for c, v in r['values'].items():
                    totals.setdefault(sim_map.get(c, c), []).append(v)
            self.stats['sims'] += sum(r.get('sims') or 0 for r in results)
            if not totals:
                return cands[0]
            return translated[out['choices'].index(max(totals, key=lambda c: sum(totals[c]) / len(totals[c])))]
        if self.sample:
            r, acc = self.rng.random(), 0.0
            pick = len(idx) - 1
            for k, v in enumerate(p):
                acc += v
                if r <= acc:
                    pick = k
                    break
        else:
            pick = int(np.argmax(p))
        return translated[idx[pick]]
