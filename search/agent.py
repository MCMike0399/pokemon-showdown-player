"""Search agent: determinized simultaneous-move search with the official simulator.

`SearchAgent.decide(ctx)` takes the same ctx the harness gives the Brain (request,
public_log, team_sets, choices) and returns a choice string. Only the request and
our public stream are used; opponent sets come from ladder usage priors.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import random
import re
import time
from pathlib import Path

from battle_state import hp_fraction, to_id
from search.priors import sample_bench, sample_set
from search.tracker import PROTECT_MOVES, Tracker, _hp

ENGINE = Path(__file__).resolve().parent / 'engine.cjs'
_MOVES = None


def move_info(move_id: str) -> dict:
    global _MOVES
    if _MOVES is None:
        root = Path(__file__).resolve().parents[1] / 'cache' / 'dex' / 'gen9championsvgc2026regmc' / 'moves.json'
        _MOVES = json.loads(root.read_text())
    return _MOVES.get(to_id(move_id), {})


class Engine:
    """One long-lived node search process."""

    def __init__(self):
        self.proc = None
        self.counter = itertools.count()
        self.lock = asyncio.Lock()

    async def start(self):
        if self.proc is None or self.proc.returncode is not None:
            self.proc = await asyncio.create_subprocess_exec(
                'node', '--max-old-space-size=1024', str(ENGINE), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=1 << 26)

    async def call(self, msg: dict, timeout: float = 120) -> dict:
        async with self.lock:
            await self.start()
            msg = {**msg, 'id': next(self.counter)}
            self.proc.stdin.write((json.dumps(msg) + '\n').encode())
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
            if not line:
                err = (await self.proc.stderr.read()).decode()[:2000]
                self.proc = None
                raise RuntimeError('search engine died: ' + err)
            out = json.loads(line)
            if 'error' in out and 'results' not in out:
                raise RuntimeError(out['error'])
            return out

    async def close(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            await self.proc.wait()


def _base(species: str) -> str:
    return re.sub(r'-Mega(-[XYZ])?$', '', species.split(',')[0].strip())


def _our_set(team_sets: list[dict], species: str) -> dict | None:
    sid = to_id(_base(species))
    for s in team_sets or []:
        if to_id(s.get('species') or s.get('name')) == sid:
            return s
    for s in team_sets or []:
        if to_id(_base(s.get('species') or '')) == sid or sid.startswith(to_id(s.get('species') or '')):
            return s
    return None


def _volatiles(mon) -> dict:
    return {k: v for k, v in (mon.volatiles or {}).items() if v is not None} if mon else {}


class SearchAgent:
    name = 'search'

    def __init__(self, fmt: str = 'gen9championsvgc2026regmc', worlds: int = 4, engines: int = 1,
                 seeds: int = 1, alpha: float = 0.5, opp_tau: float = 1.0, max_opp: int | None = None,
                 preview=None, seed: int = 0, usage_cutoff: str = '0', opp_no_switch: bool = False,
                 keep_nonmega: bool = False, log=None, screen=None, value_path=None, value_beta=0.0, value_scale=8.0, preview_mode='incumbent', preview_worlds=6):
        self.fmt = fmt
        self.n_worlds = worlds
        self.engines = [Engine() for _ in range(max(1, engines))]
        self.seeds = seeds
        self.alpha = alpha
        self.opp_tau = opp_tau
        self.max_opp = max_opp
        self.preview = preview
        self.rng = random.Random(seed)
        self.cutoff = usage_cutoff
        self.opp_no_switch = opp_no_switch
        self.keep_nonmega = keep_nonmega
        self.log = log
        self.screen = screen
        self.preview_mode = preview_mode
        self.preview_worlds = preview_worlds
        self.value = {'value_path': value_path, 'value_beta': value_beta, 'value_scale': value_scale}
        self.stats = {'decisions': 0, 'ms': 0.0, 'fallbacks': 0, 'errors': []}

    async def close(self):
        for e in self.engines:
            await e.close()

    # ------------------------------------------------------------ world building
    def _our_side(self, ctx: dict, tracker: Tracker, order: list[int] | None = None) -> list[dict]:
        request = ctx['request']
        side = request['side']
        party = side['pokemon']
        active_req = request.get('active') or []
        out = []
        idxs = order if order is not None else list(range(len(party)))
        for pos, i in enumerate(idxs):
            mon = party[i]
            species = mon['details'].split(',')[0]
            base_set = _our_set(ctx.get('team_sets') or [], species)
            if base_set is None:
                raise ValueError('own set not found for ' + species)
            set_ = dict(base_set)
            set_['moves'] = [to_id(m) for m in mon.get('moves') or base_set['moves']]
            frac, status, cur, mx = _hp(mon.get('condition', '100/100'))
            name = mon['ident'].split(': ', 1)[1]
            tm = tracker.mons.get((side['id'], name))
            st = {
                'hp_abs': cur if cur is not None and not frac == 0 else None,
                'fainted': frac == 0 or 'fnt' in mon.get('condition', ''),
                'status': status,
                'item': mon.get('item', ''),
                'item_gone': (mon.get('item', '') == '' and bool(base_set.get('item'))),
                'ability': mon.get('ability') or mon.get('baseAbility'),
                'mega': '-Mega' in species,
                'boosts': dict(tm.boosts) if tm and tm.active_slot else {},
                'volatiles': _volatiles(tm) if tm and tm.active_slot else {},
                'actions': tm.actions if tm and tm.active_slot else 0,
                'active_turns': (tracker.turn - tm.switch_turn) if tm and tm.active_slot else 0,
                'protect_last_turn': bool(tm and tm.active_slot and tm.last_move in PROTECT_MOVES and
                                          tm.last_move_turn == tracker.turn - 1),
                'sleep_turns': tm.sleep_turns if tm else 0,
                'toxic_turns': tm.toxic_turns if tm else 0,
                'turn': tracker.turn,
            }
            if pos < len(active_req) and order is None:
                pp = {}
                for m in active_req[pos].get('moves', []):
                    if 'pp' in m:
                        pp[to_id(m.get('id') or m.get('move'))] = m['pp']
                st['pp'] = pp
            out.append({'set': set_, 'state': st})
        return out[:4]

    def _their_side(self, tracker: Tracker, rng: random.Random) -> list[dict]:
        foe = tracker.foe
        revealed = [m for m in tracker.side_mons(foe)]
        actives = [tracker.slots.get(foe + 'a'), tracker.slots.get(foe + 'b')]
        ordered = []
        for m in actives:
            if m is not None and m not in ordered:
                ordered.append(m)
        for m in revealed:
            if m not in ordered:
                ordered.append(m)
        preview = tracker.preview.get(foe) or []
        need = max(0, 4 - len(ordered))
        extra = sample_bench(rng, preview, [m.base_species for m in ordered], need, self.fmt, self.cutoff) if preview else []
        out = []
        mega_used = tracker.mega_used.get(foe)
        sheet = {to_id(_base(x.get('species') or x.get('name') or '')): x for x in (tracker.sheets.get(foe) or [])}

        def from_sheet(species, set_):
            # Open team sheet: exact species/item/ability/moves/nature; only spreads stay sampled.
            x = sheet.get(to_id(_base(species)))
            if not x:
                return set_
            out = dict(set_)
            out['moves'] = [to_id(mv) for mv in x.get('moves') or []] or set_['moves']
            if x.get('item'):
                out['item'] = to_id(x['item'])
            if x.get('ability'):
                out['ability'] = to_id(x['ability'])
            if x.get('nature'):
                out['nature'] = x['nature']
            return out

        for m in ordered[:4]:
            observed = {'moves': m.moves, 'item': None if not m.item_known else m.item, 'ability': m.ability,
                        'mega': m.mega, 'mega_species': m.species if m.mega else None,
                        'mega_ruled_out': mega_used and not m.mega}
            set_ = sample_set(rng, m.base_species, observed, self.fmt, self.cutoff)
            if m.item_known and m.item and not m.mega:
                set_['item'] = m.item
            set_ = from_sheet(m.base_species, set_)
            st = {
                'hp': m.hp, 'fainted': m.fainted, 'status': m.status,
                'boosts': dict(m.boosts) if m.active_slot else {},
                'volatiles': _volatiles(m) if m.active_slot else {},
                'item': '' if m.item_gone else None,
                'item_gone': m.item_gone,
                'ability': m.ability if not m.mega else None,
                'mega': m.mega,
                'actions': m.actions if m.active_slot else 0,
                'active_turns': (tracker.turn - m.switch_turn) if m.active_slot else 0,
                'protect_last_turn': bool(m.active_slot and m.last_move in PROTECT_MOVES and
                                          m.last_move_turn == tracker.turn - 1),
                'sleep_turns': m.sleep_turns, 'toxic_turns': m.toxic_turns, 'turn': tracker.turn,
                'locked_move': m.last_move if (m.active_slot and to_id(set_.get('item', '')).startswith('choice')
                                               and m.last_move and not m.item_gone) else None,
            }
            out.append({'set': set_, 'state': st})
        for species in extra:
            observed = {'mega_ruled_out': mega_used}
            set_ = from_sheet(species, sample_set(rng, species, observed, self.fmt, self.cutoff))
            out.append({'set': set_, 'state': {'hp': 1.0, 'turn': tracker.turn}})
        # Pad (unknown preview) with generic placeholders so the battle has 4 members.
        while len(out) < 4 and preview:
            break
        return out

    def _field(self, tracker: Tracker) -> dict:
        field = tracker.field_state()
        me, foe = tracker.me, tracker.foe
        return {'turn': max(1, tracker.turn), 'weather': field['weather'], 'terrain': field['terrain'],
                'pseudo': field['pseudo'], 'sides': {'p1': field['sides'].get(me, {}), 'p2': field['sides'].get(foe, {})},
                'mega_used': {'p1': tracker.mega_used.get(me), 'p2': tracker.mega_used.get(foe)}}

    # ------------------------------------------------------------ choices
    def _prune_ours(self, ctx: dict) -> list[str]:
        request = ctx['request']
        choices = list(ctx['choices'])
        active = request.get('active') or []

        def ok_part(part: str, slot: int) -> bool:
            words = part.split()
            if words[0] != 'move':
                return True
            j = int(words[1]) - 1
            moves = active[slot].get('moves', []) if slot < len(active) else []
            if j >= len(moves):
                return True
            info = move_info(moves[j].get('id') or moves[j].get('move'))
            target = int(words[2]) if len(words) > 2 and words[2].lstrip('-').isdigit() else None
            if target is not None and target < 0 and info.get('category') != 'Status':
                return False
            return True

        out = []
        for c in choices:
            parts = [p.strip() for p in c.split(',')]
            if all(ok_part(p, i) for i, p in enumerate(parts)):
                out.append(c)
        if not self.keep_nonmega:
            can_mega = [bool(a.get('canMegaEvo')) for a in active]
            if any(can_mega):
                with_mega = [c for c in out if ' mega' in c]
                if with_mega:
                    out = with_mega
        return out or choices

    # ------------------------------------------------------------ decide
    async def decide(self, ctx: dict) -> str:
        request = ctx['request']
        choices = ctx['choices']
        if len(choices) == 1:
            return choices[0]
        if request.get('teamPreview'):
            if self.preview_mode == 'value':
                try:
                    return await self._value_preview(ctx)
                except Exception as error:
                    self.stats['errors'].append('preview ' + repr(error)[:300])
            if self.preview is not None:
                return await self.preview(ctx)
            return choices[0]
        started = time.perf_counter()
        try:
            if request.get('forceSwitch'):
                choice = await self._force_switch(ctx)
            else:
                choice = await self._move(ctx)
        except Exception as error:  # never stall a game
            self.stats['fallbacks'] += 1
            self.stats['errors'].append(repr(error)[:300])
            choice = self._fallback(ctx)
        self.stats['decisions'] += 1
        self.stats['ms'] += (time.perf_counter() - started) * 1000
        return choice

    async def _value_preview(self, ctx: dict) -> str:
        from search.priors import species_weight
        request = ctx['request']
        side = request['side']['id']
        tracker = Tracker(side).feed(ctx.get('public_log') or [])
        ours = []
        for mon in request['side']['pokemon']:
            st = _our_set(ctx.get('team_sets') or [], mon['details'].split(',')[0])
            if st is None:
                raise ValueError('own set missing')
            ours.append(st)
        foe = tracker.preview.get(tracker.foe) or []
        if len(foe) < 4:
            raise ValueError('opponent preview missing')
        worlds = [[sample_set(self.rng, sp, {}, self.fmt, self.cutoff) for sp in foe] for _ in range(self.preview_worlds)]
        weights = [species_weight(sp, self.fmt, self.cutoff) for sp in foe]
        out = await self.engines[0].call({'type': 'preview', 'format': self.fmt, 'ours': ours, 'worlds': worlds,
                                          'opp_weights': weights, 'samples_per_world': 3,
                                          'seed': self.rng.randrange(1, 10**6), **self.value})
        legal = set(ctx['choices'])
        scored = {c: v for c, v in out['values'].items() if c in legal}
        if not scored:
            raise ValueError('no legal preview plan')
        best = max(scored, key=scored.get)
        if self.log is not None:
            self.log.append({'turn': 0, 'top': sorted(scored.items(), key=lambda kv: -kv[1])[:5]})
        return best

    def _fallback(self, ctx):
        pruned = self._prune_ours(ctx)
        return pruned[0]

    async def _search(self, ctx: dict, tracker: Tracker, our_side: list[dict], our_choices: list[str] | None):
        rng = self.rng
        worlds = []
        for _ in range(self.n_worlds):
            worlds.append({'p1': our_side, 'p2': self._their_side(tracker, rng)})
        field = self._field(tracker)
        base = {'type': 'search', 'format': self.fmt, 'field': field, 'our_choices': our_choices,
                'seeds': self.seeds, 'alpha': self.alpha, 'opp_tau': self.opp_tau,
                'max_opp': self.max_opp, 'opp_no_switch': self.opp_no_switch, 'screen': self.screen, **self.value}
        n = len(self.engines)
        chunks = [worlds[i::n] for i in range(n)]
        calls = []
        for k, (engine, chunk) in enumerate(zip(self.engines, chunks)):
            if chunk:
                calls.append(engine.call({**base, 'worlds': chunk, 'world_offset': k * 1000 + rng.randrange(1000)}))
        outs = await asyncio.gather(*calls)
        results = [r for out in outs for r in out['results']]
        return results

    async def _move(self, ctx: dict) -> str:
        side = ctx['request']['side']['id']
        tracker = Tracker(side).feed(ctx.get('public_log') or [])
        our_choices = self._prune_ours(ctx)
        active = ctx['request'].get('active') or []
        if any(a.get('canMegaEvo') for a in active):
            tracker.mega_used[side] = False
        sim_map = {}
        for c in our_choices:
            sim_map.setdefault(self._to_sim(c, active, tracker), c)
        our_side = self._our_side(ctx, tracker)
        results = await self._search(ctx, tracker, our_side, list(sim_map))
        for r in results:
            if 'values' in r:
                r['values'] = {sim_map.get(k, k): v for k, v in r['values'].items()}
        totals: dict[str, list[float]] = {}
        errors = []
        for r in results:
            if 'error' in r:
                errors.append(r['error'])
                continue
            for c, v in r['values'].items():
                totals.setdefault(c, []).append(v)
        if not totals:
            raise RuntimeError('all worlds failed: ' + '; '.join(errors)[:500])
        good = len([r for r in results if 'error' not in r])
        scored = {c: sum(v) / len(v) - (0.5 if len(v) < good else 0.0) for c, v in totals.items()}
        best = max(scored, key=scored.get)
        if self.log is not None:
            top = sorted(scored.items(), key=lambda kv: -kv[1])[:5]
            self.log.append({'turn': tracker.turn, 'top': top, 'errors': errors[:2],
                             'opp': [r.get('top_opp') for r in results[:1]],
                             'ms': [r.get('ms') for r in results], 'sims': [r.get('sims') for r in results]})
        return best

    @staticmethod
    def _to_sim(choice: str, active: list, tracker: Tracker) -> str:
        """Address moves by id: locked requests list one move, so indices differ in the rebuilt battle."""
        parts = []
        for slot, part in enumerate(p.strip() for p in choice.split(',')):
            words = part.split()
            if words and words[0] == 'move' and slot < len(active):
                moves = active[slot].get('moves') or []
                j = int(words[1]) - 1
                if 0 <= j < len(moves):
                    mid = to_id(moves[j].get('id') or moves[j].get('move'))
                    rest = words[2:]
                    has_target = any(w.lstrip('-').isdigit() for w in rest)
                    if not has_target and move_info(mid).get('target') in ('normal', 'any', 'adjacentFoe'):
                        foe = tracker.foe
                        alive = [i + 1 for i, k in enumerate('ab') if (m := tracker.slots.get(foe + k)) and not m.fainted]
                        rest = [str(alive[0] if alive else 1)] + rest
                    part = ' '.join(['move', mid] + rest)
            parts.append(part)
        return ', '.join(parts)

    async def _force_switch(self, ctx: dict) -> str:
        request = ctx['request']
        side = request['side']['id']
        tracker = Tracker(side).feed(ctx.get('public_log') or [])
        party = request['side']['pokemon']
        forced = request['forceSwitch']
        choices = ctx['choices']
        if len(choices) == 1:
            return choices[0]
        best, best_v = choices[0], -1e9
        for choice in choices:
            parts = [p.strip() for p in choice.split(',')]
            order = list(range(len(party)))
            for slot, part in enumerate(parts):
                if part.startswith('switch'):
                    k = int(part.split()[1]) - 1
                    order[slot], order[k] = order[k], order[slot]
            # Build a hypothetical request state where the replacement is active.
            fake_party = [dict(party[i]) for i in order]
            for i, mon in enumerate(fake_party):
                mon['active'] = i < len(forced)
            fake_ctx = {**ctx, 'request': {'side': {**request['side'], 'pokemon': fake_party}, 'active': []}}
            t2 = Tracker(side).feed(ctx.get('public_log') or [])
            # Newly switched mons have no boosts/volatiles.
            for slot, part in enumerate(parts):
                if part.startswith('switch'):
                    t2.slots.pop(side + 'ab'[slot], None)
            our_side = self._our_side(fake_ctx, t2, order=None)
            for i, part in enumerate(parts):
                if part.startswith('switch') and i < len(our_side):
                    our_side[i]['state'].update(boosts={}, volatiles={}, actions=0, active_turns=0)
            try:
                results = await self._search(ctx, t2, our_side, None)
            except Exception as error:
                self.stats['errors'].append(repr(error)[:300])
                continue
            vals = []
            for r in results:
                if 'error' in r or not r.get('values'):
                    continue
                vals.append(max(r['values'].values()))
            if vals:
                v = sum(vals) / len(vals)
                if v > best_v:
                    best, best_v = choice, v
        return best
