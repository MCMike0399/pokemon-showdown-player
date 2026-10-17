"""Scouting: what an opponent revealed in their earlier public games, as observations for worlds.

66% of live opponents have earlier public replays under the same name, and items revealed
there matched the current game's items in 98.8% of cases (docs/c7-live-review.md). At team
preview the agent starts a background lookup; facts arrive before turn 1 or a little later,
and every world treats them as observed (moves, item, ability; an earlier open team sheet as
the exact set), each with probability `trust`, since a player can change a set between games.
Facts the current game reveals always win.

Sources are injected, so gates never touch the network:
- `OnlineScout` (live runner): replay.pokemonshowdown.com search by user, newest first,
  a few logs per opponent, cached on disk;
- `StaticScout` (tests, gates with simulated coverage): a fixed facts table per player.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from battle_state import to_id

BASE = 'https://replay.pokemonshowdown.com'
UA = 'pokemon-showdown-player scouting (contact via github MCMike0399)'


def _sid(species: str) -> str:
    return to_id(re.sub(r'-Mega(-[XYZ])?$', '', species.split(',')[0].strip()))


def player_side(log: list[str], player: str) -> str | None:
    uid = to_id(player)
    for line in log:
        if line.startswith('|player|'):
            p = line.split('|')
            if len(p) > 3 and to_id(p[3]) == uid:
                return p[2]
    return None


def extract_facts(log: list[str], player: str) -> dict | None:
    """One game: the player's six and, per species, what it revealed (moves, item, ability)."""
    from search.tracker import Tracker
    side = player_side(log, player)
    if side not in ('p1', 'p2'):
        return None
    us = 'p2' if side == 'p1' else 'p1'
    t = Tracker(us).feed(log)
    six = [_sid(x) for x in t.preview.get(side) or []]
    mons = {}
    for m in t.side_mons(side):
        sid = _sid(m.base_species or m.species)
        item = to_id(m.item) if m.item_known and m.item else None
        mons[sid] = {'moves': [to_id(x) for x in m.moves if to_id(x) != 'struggle'], 'item': item,
                     'ability': to_id(m.ability) if m.ability and not m.mega else None, 'mega': bool(m.mega)}
    sheet = {}
    for x in t.sheets.get(side) or []:
        sid = _sid(x.get('species') or x.get('name') or '')
        if sid:
            sheet[sid] = {'moves': [to_id(v) for v in x.get('moves') or []], 'item': to_id(x.get('item') or '') or None,
                          'ability': to_id(x.get('ability') or '') or None, 'nature': x.get('nature') or None}
    return {'six': six, 'mons': mons, 'sheet': sheet}


def merge_facts(games: list[dict], six: list[str], min_overlap: int = 4) -> dict:
    """Facts per species of the current six from earlier games (newest first) whose six shares at
    least `min_overlap` species with it: moves by frequency then recency (at most 4), the newest
    revealed item and ability, the newest open team sheet."""
    current = {_sid(x) for x in six}
    moves: dict[str, Counter] = {}
    out: dict[str, dict] = {}
    for rank, g in enumerate(games):
        if not g or len(current & set(g.get('six') or [])) < min_overlap:
            continue
        for sid, f in (g.get('mons') or {}).items():
            if sid not in current:
                continue
            o = out.setdefault(sid, {'moves': [], 'item': None, 'ability': None, 'sheet': None, 'games': 0})
            o['games'] += 1
            c = moves.setdefault(sid, Counter())
            for k, mv in enumerate(f.get('moves') or []):
                c[mv] += 1 + 1e-3 / (1 + rank)  # ties: the newer game first
            o['item'] = o['item'] or f.get('item')
            o['ability'] = o['ability'] or f.get('ability')
        for sid, s in (g.get('sheet') or {}).items():
            if sid in current:
                o = out.setdefault(sid, {'moves': [], 'item': None, 'ability': None, 'sheet': None, 'games': 0})
                o['sheet'] = o['sheet'] or s
    for sid, c in moves.items():
        out[sid]['moves'] = [mv for mv, _ in c.most_common(4)]
    return out


def apply_facts(observed: dict, facts: dict | None, item_known: bool, item_gone: bool) -> dict:
    """Observed facts of the current game, completed with scouted ones it has not revealed."""
    if not facts:
        return observed
    out = dict(observed)
    revealed = [to_id(x) for x in observed.get('moves') or []]
    if len(revealed) < 4:
        out['moves'] = (revealed + [m for m in facts.get('moves') or [] if m not in revealed])[:4]
    if not item_known and not item_gone and not observed.get('mega') and facts.get('item'):
        out['item'] = facts['item']
    if not observed.get('ability') and facts.get('ability') and not observed.get('mega'):
        out['ability'] = facts['ability']
    return out


def sheet_fits(sheet: dict | None, revealed_moves: list[str]) -> bool:
    """An earlier open team sheet still describes the set if it has every move seen so far."""
    return bool(sheet and sheet.get('moves')) and set(to_id(m) for m in revealed_moves) <= set(sheet['moves'])


class StaticScout:
    """Facts per player name (tests; gates with simulated coverage)."""

    def __init__(self, table: dict[str, dict] | None = None):
        self.table = {to_id(k): v for k, v in (table or {}).items()}

    async def facts(self, player: str, six: list[str], room: str | None = None) -> dict:
        return self.table.get(to_id(player)) or {}


class SimScout:
    """Gate stand-in for an opponent's history, from their true sets: with probability `coverage`
    the player has earlier games; then each species' item is known with `p_item` (wrong with
    `p_wrong_item`), each move with `p_move`, the ability with `p_ability`. Defaults match live
    scouting measured on 1,220 games at min_overlap 2 (docs/c7-live-review.md): facts for 45% of
    opponents, ~2.5 items per covered six, item precision 96%, revealed-move recall 71%."""

    def __init__(self, sets: list[dict], seed: int, coverage: float = 0.45, p_item: float = 0.42,
                 p_wrong_item: float = 0.04, p_move: float = 0.6, p_ability: float = 0.4):
        import random
        rng = random.Random(seed)
        self.table = {}
        if rng.random() >= coverage:
            return
        items = [to_id(x.get('item') or '') for x in sets if x.get('item')]
        for x in sets:
            sid = _sid(x.get('species') or x.get('name') or '')
            moves = [to_id(m) for m in x.get('moves') or [] if rng.random() < p_move]
            item = to_id(x.get('item') or '') if rng.random() < p_item else None
            if item and rng.random() < p_wrong_item:
                item = rng.choice([i for i in items if i != item] or [item])
            ability = to_id(x.get('ability') or '') if rng.random() < p_ability else None
            if moves or item or ability:
                self.table[sid] = {'moves': moves, 'item': item, 'ability': ability, 'sheet': None, 'games': 1}

    async def facts(self, player: str, six: list[str], room: str | None = None) -> dict:
        return dict(self.table)


class OnlineScout:
    """The opponent's newest public replays in this format, a few logs each, cached on disk."""

    def __init__(self, fmt: str, cache: Path, max_games: int = 6, min_overlap: int = 4, timeout: float = 6.0,
                 spacing: float = 0.4, log=None):
        self.fmt, self.cache = fmt, Path(cache)
        self.max_games, self.min_overlap, self.timeout, self.spacing = max_games, min_overlap, timeout, spacing
        self.log = log
        self.cache.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str) -> bytes | None:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None

    def _replay_log(self, rid: str) -> list[str] | None:
        path = self.cache / (re.sub(r'[^a-z0-9-]', '', rid) + '.json')
        if path.exists():
            try:
                return json.loads(path.read_text())['log'].split('\n')
            except (ValueError, KeyError):
                pass
        body = self._get(f'{BASE}/{urllib.parse.quote(rid)}.json')
        time.sleep(self.spacing)
        if not body:
            return None
        try:
            rep = json.loads(body)
        except ValueError:
            return None
        path.write_text(json.dumps({'id': rep.get('id'), 'uploadtime': rep.get('uploadtime'), 'log': rep.get('log', '')}))
        return (rep.get('log') or '').split('\n')

    def _lookup(self, player: str, six: list[str], room: str | None) -> dict:
        t0 = time.time()
        body = self._get(f'{BASE}/search.json?user={urllib.parse.quote(to_id(player))}&format={self.fmt}')
        if not body:
            return {}
        try:
            listing = json.loads(body)
        except ValueError:
            return {}
        num = re.search(r'-(\d+)', room or '')
        ids = [r['id'] for r in sorted(listing, key=lambda r: -(r.get('uploadtime') or 0))
               if r.get('id') and not (num and f'-{num.group(1)}' in r['id'])]
        games = []
        for rid in ids[:self.max_games]:
            log = self._replay_log(rid)
            if log:
                games.append(extract_facts(log, player))
        facts = merge_facts(games, six, self.min_overlap)
        self.last = {'player': to_id(player), 'listed': len(listing), 'read': len(games), 'species': len(facts),
                     'items': sum(1 for f in facts.values() if f.get('item')),
                     'sheets': sum(1 for f in facts.values() if f.get('sheet')), 'seconds': round(time.time() - t0, 1)}
        return facts

    async def facts(self, player: str, six: list[str], room: str | None = None) -> dict:
        self.last = {}
        facts = await asyncio.to_thread(self._lookup, player, six, room)
        if self.log and self.last:
            self.log('scout', room=room, **self.last)  # on the event loop, not the worker thread
        return facts
