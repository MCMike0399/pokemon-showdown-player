"""Public-log battle tracker for search: per-Pokemon and field state with turn counters.

Only the player's own public stream is read. Our private facts (exact HP, item,
PP) come from the request; the opponent's hidden facts are left unknown here and
filled by `search.priors` determinization.
"""
from __future__ import annotations

import re

from battle_state import to_id

PROTECT_MOVES = {'protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'silktrap',
                 'burningbulwark', 'obstruct', 'maxguard', 'endure'}
# Default durations of timed effects; extended variants are handled at read time.
DURATIONS = {'weather': 5, 'terrain': 5, 'trickroom': 5, 'gravity': 5, 'wonderroom': 5, 'magicroom': 5,
             'tailwind': 4, 'reflect': 5, 'lightscreen': 5, 'auroraveil': 5, 'safeguard': 5, 'mist': 5}
EXTENDED = {'weather': 8, 'terrain': 8, 'reflect': 8, 'lightscreen': 8, 'auroraveil': 8}


def _effect_id(text: str) -> str:
    return to_id(text.split(':', 1)[1] if ':' in text else text)


def _hp(text: str) -> tuple[float, str | None, int | None, int | None]:
    """Return (fraction, status, current, max) from a public condition string."""
    text = (text or '').strip()
    if not text or text.startswith('0') and ('fnt' in text or text.split()[0] == '0'):
        return 0.0, None, 0, None
    parts = text.split()
    status = parts[1] if len(parts) > 1 and parts[1] in ('brn', 'par', 'slp', 'frz', 'psn', 'tox') else None
    hp = re.sub(r'[gry]$', '', parts[0])
    try:
        cur, mx = hp.split('/')
        cur_i, max_i = int(float(cur)), int(float(mx))
        return max(0.0, min(1.0, cur_i / max_i)), status, cur_i, max_i
    except (ValueError, ZeroDivisionError):
        return 1.0, status, None, None


class Mon:
    __slots__ = ('side', 'name', 'species', 'base_species', 'hp', 'hp_exact', 'maxhp', 'status', 'status_turn',
                 'sleep_turns', 'boosts', 'volatiles', 'item', 'item_known', 'item_gone', 'ability', 'moves',
                 'fainted', 'active_slot', 'switch_turn', 'actions', 'last_move', 'last_move_turn', 'mega',
                 'locked_move', 'revealed', 'toxic_turns')

    def __init__(self, side: str, name: str, species: str):
        self.side, self.name, self.species = side, name, species
        self.base_species = species
        self.hp, self.hp_exact, self.maxhp = 1.0, None, None
        self.status, self.status_turn, self.sleep_turns, self.toxic_turns = None, 0, 0, 0
        self.boosts: dict[str, int] = {}
        self.volatiles: dict[str, dict] = {}
        self.item, self.item_known, self.item_gone = None, False, False
        self.ability = None
        self.moves: list[str] = []
        self.fainted = False
        self.active_slot = None
        self.switch_turn = 0
        self.actions = 0
        self.last_move = None
        self.last_move_turn = -1
        self.mega = False
        self.locked_move = None
        self.revealed = False

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class Tracker:
    def __init__(self, me: str):
        self.me = me
        self.foe = 'p2' if me == 'p1' else 'p1'
        self.turn = 0
        self.preview = {'p1': [], 'p2': []}
        self.sheets = {'p1': [], 'p2': []}
        self.mons: dict[tuple[str, str], Mon] = {}
        self.slots: dict[str, Mon] = {}
        self.weather = None  # (id, start_turn, source_item_known)
        self.terrain = None
        self.pseudo: dict[str, int] = {}
        self.side_conditions = {'p1': {}, 'p2': {}}
        self.mega_used = {'p1': False, 'p2': False}
        self.winner = None

    # ---------------------------------------------------------------- parsing
    def _mon(self, ident: str, details: str | None = None) -> Mon | None:
        side = ident[:2]
        name = ident.split(': ', 1)[1].strip() if ': ' in ident else ident[3:].strip()
        key = (side, name)
        mon = self.mons.get(key)
        if mon is None:
            if details is None:
                # Mentioned before switch-in (rare: e.g. [of] references); attach via slot.
                return self.slots.get(ident.split(':')[0])
            species = details.split(',')[0].strip()
            mon = Mon(side, name, species)
            mon.base_species = re.sub(r'-Mega(-[XYZ])?$', '', species)
            self.mons[key] = mon
        return mon

    def _slot_mon(self, ident: str) -> Mon | None:
        slot = ident.split(':')[0].strip()
        if len(slot) == 3:
            mon = self.slots.get(slot)
            if mon is not None:
                return mon
        return self._mon(ident)

    def feed(self, log: list[str]) -> 'Tracker':
        for line in log:
            self.line(line)
        return self

    def line(self, line: str) -> None:
        p = line.split('|')
        if len(p) < 2:
            return
        kind = p[1]
        arg = p[2] if len(p) > 2 else ''
        if kind == 'turn':
            self.turn = int(arg)
        elif kind == 'poke' and arg in self.preview and len(p) > 3:
            self.preview[arg].append(p[3].split(',')[0].strip())
        elif kind == 'showteam' and arg in self.sheets:
            from battle_state import unpack_team
            self.sheets[arg] = unpack_team(line.split('|', 3)[3])
        elif kind in ('switch', 'drag', 'replace') and len(p) > 4:
            slot = arg.split(':')[0].strip()
            mon = self._mon(arg, p[3])
            if kind == 'replace':
                # Illusion broke: the slot holds a different Pokemon than shown.
                old = self.slots.get(slot)
                if old is not None and old is not mon:
                    old.active_slot = None
            prev = self.slots.get(slot)
            if prev is not None and prev is not mon and kind != 'replace':
                prev.active_slot = None
                prev.boosts = {}
                prev.volatiles = {}
                prev.locked_move = None
            mon.species = p[3].split(',')[0].strip()
            if '-Mega' in mon.species:
                mon.mega = True
            frac, status, cur, mx = _hp(p[4])
            mon.hp = frac
            mon.hp_exact = cur if mx and mx != 100 else None
            if mx and mx != 100:
                mon.maxhp = mx
            mon.status = status
            mon.active_slot = slot
            if kind != 'replace':
                mon.switch_turn = self.turn
                mon.actions = 0
                mon.boosts = {}
                mon.volatiles = {}
                mon.locked_move = None
            mon.revealed = True
            self.slots[slot] = mon
        elif kind in ('detailschange', '-formechange') and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                species = p[3].split(',')[0].strip()
                if '-Mega' in species and not mon.mega:
                    mon.mega = True
                    self.mega_used[mon.side] = True
                if kind == 'detailschange':
                    mon.species = species
        elif kind in ('-damage', '-heal', '-sethp') and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                frac, status, cur, mx = _hp(p[3])
                mon.hp = frac
                if mx and mx != 100:
                    mon.hp_exact, mon.maxhp = cur, mx
                if frac == 0:
                    mon.fainted = True
                rest = '|'.join(p[4:])
                m = re.search(r'\[from\] item: ([^|\]]+)', rest)
                if m and '[of]' not in rest:
                    self._item(mon, m.group(1))
                m = re.search(r'\[from\] ability: ([^|\]]+)', rest)
                if m and '[of]' not in rest:
                    mon.ability = to_id(m.group(1))
                if kind == '-damage' and 'psn' in rest and mon.status == 'tox':
                    mon.toxic_turns += 1
        elif kind == 'faint':
            mon = self._slot_mon(arg)
            if mon:
                mon.fainted, mon.hp, mon.hp_exact = True, 0.0, 0
        elif kind == '-status' and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                mon.status = p[3]
                mon.status_turn = self.turn
                mon.sleep_turns = 0
                mon.toxic_turns = 0
                rest = '|'.join(p[4:])
                m = re.search(r'\[from\] item: ([^|\]]+)', rest)
                if m:
                    self._item(mon, m.group(1))
        elif kind == '-curestatus' and len(p) > 3:
            mon = self._slot_mon(arg) if arg[:3] in self.slots or ': ' in arg else None
            if mon:
                mon.status = None
        elif kind == '-cureteam':
            for mon in self.mons.values():
                if mon.side == arg[:2]:
                    mon.status = None
        elif kind in ('-item',) and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                self._item(mon, p[3])
        elif kind == '-enditem' and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                rest = '|'.join(p[4:])
                mon.item = to_id(p[3])
                mon.item_known = True
                # Air Balloon popping, berries eaten, knocked off: the item is gone.
                mon.item_gone = True
        elif kind == '-ability' and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                rest = '|'.join(p[4:])
                m = re.search(r'\[from\] ability: ([^|\]]+)', rest)
                if m:
                    # Trace/Receiver/Power of Alchemy etc: the holder's own ability is the source.
                    if to_id(m.group(1)) in ('trace', 'receiver', 'powerofalchemy'):
                        mon.ability = to_id(m.group(1))
                elif '[from]' not in rest:
                    mon.ability = to_id(p[3])
        elif kind in ('-boost', '-unboost') and len(p) > 4:
            mon = self._slot_mon(arg)
            if mon:
                try:
                    amount = int(p[4])
                except ValueError:
                    amount = 0
                stat = p[3]
                mon.boosts[stat] = max(-6, min(6, mon.boosts.get(stat, 0) + (amount if kind == '-boost' else -amount)))
                rest = '|'.join(p[5:])
                m = re.search(r'\[from\] item: ([^|\]]+)', rest)
                if m:
                    self._item(mon, m.group(1), consumed=True)
        elif kind == '-setboost' and len(p) > 4:
            mon = self._slot_mon(arg)
            if mon:
                mon.boosts[p[3]] = int(p[4])
        elif kind in ('-clearboost', '-clearpositiveboost'):
            mon = self._slot_mon(arg)
            if mon:
                mon.boosts = {} if kind == '-clearboost' else {k: v for k, v in mon.boosts.items() if v < 0}
        elif kind == '-clearnegativeboost':
            mon = self._slot_mon(arg)
            if mon:
                mon.boosts = {k: v for k, v in mon.boosts.items() if v > 0}
        elif kind == '-clearallboost':
            for mon in self.slots.values():
                mon.boosts = {}
        elif kind == '-invertboost':
            mon = self._slot_mon(arg)
            if mon:
                mon.boosts = {k: -v for k, v in mon.boosts.items()}
        elif kind == '-copyboost' and len(p) > 3:
            src, dst = self._slot_mon(p[3]), self._slot_mon(arg)
            if src and dst:
                dst.boosts = dict(src.boosts)
        elif kind in ('-start', '-end') and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                eff = _effect_id(p[3])
                if eff.startswith('perish') and kind == '-start':
                    mon.volatiles['perishsong'] = {'count': int(eff[6:] or 3)}
                elif eff.startswith('stockpile'):
                    mon.volatiles['stockpile'] = {'layers': int(eff[9:] or 1)} if kind == '-start' else None
                    if kind == '-end':
                        mon.volatiles.pop('stockpile', None)
                elif kind == '-start':
                    info = {'turn': self.turn}
                    if eff in ('encore', 'disable') and len(p) > 4:
                        info['move'] = to_id(p[4])
                    if eff == 'substitute':
                        info['hp'] = 0.25
                    if eff == 'typechange' and len(p) > 4:
                        info['types'] = p[4]
                    mon.volatiles[eff] = info
                    rest = '|'.join(p[4:])
                    m = re.search(r'\[from\] item: ([^|\]]+)', rest)
                    if m:
                        self._item(mon, m.group(1))
                else:
                    mon.volatiles.pop(eff, None)
                    if eff == 'yawn':
                        pass
        elif kind == '-singleturn' or kind == '-singlemove':
            pass
        elif kind == '-weather':
            if arg == 'none':
                self.weather = None
            elif '[upkeep]' in line:
                pass
            else:
                rest = '|'.join(p[3:])
                self.weather = (to_id(arg), self.turn)
                m = re.search(r'\[from\] ability: ', rest)
                mon = self._slot_mon(p[-1].replace('[of] ', '')) if '[of]' in rest else None
                if m and mon:
                    mon.ability = to_id(re.search(r'ability: ([^|]+)', rest).group(1))
        elif kind == '-fieldstart' and len(arg) > 0:
            eff = _effect_id(arg)
            if eff.endswith('terrain'):
                self.terrain = (eff, self.turn)
                rest = '|'.join(p[3:])
                if '[from] ability:' in rest and '[of]' in rest:
                    mon = self._slot_mon(rest.split('[of] ')[1].split('|')[0])
                    if mon:
                        mon.ability = to_id(re.search(r'ability: ([^|]+)', rest).group(1))
            else:
                self.pseudo[eff] = self.turn
        elif kind == '-fieldend':
            eff = _effect_id(arg)
            if eff.endswith('terrain'):
                self.terrain = None
            else:
                self.pseudo.pop(eff, None)
        elif kind in ('-sidestart', '-sideend') and len(p) > 3:
            side = arg[:2]
            eff = _effect_id(p[3])
            if side in self.side_conditions:
                if kind == '-sidestart':
                    layers = self.side_conditions[side].get(eff, {}).get('layers', 0) + 1
                    self.side_conditions[side][eff] = {'turn': self.turn, 'layers': layers}
                else:
                    self.side_conditions[side].pop(eff, None)
        elif kind == 'move' and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                move = to_id(p[3])
                rest = '|'.join(p[4:])
                called = '[from]' in rest and ('[from] move:' in rest or 'lockedmove' in rest)
                if not called or 'lockedmove' in rest:
                    if move not in mon.moves and move != 'struggle' and '[from]' not in rest:
                        mon.moves.append(move)
                    mon.actions += 1
                    mon.last_move = move
                    mon.last_move_turn = self.turn
                    if '[still]' not in rest and '[notarget]' not in rest:
                        pass
                    if mon.item and to_id(mon.item).startswith('choice') and not mon.item_gone:
                        mon.locked_move = move
                    elif not mon.item_known:
                        mon.locked_move = mon.locked_move  # unknown; priors may add choice lock
        elif kind == 'cant' and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                if p[3] == 'slp':
                    mon.sleep_turns += 1
                mon.last_move = None
                mon.last_move_turn = self.turn
        elif kind == '-mega' and len(p) > 3:
            mon = self._slot_mon(arg)
            if mon:
                mon.mega = True
                self.mega_used[mon.side] = True
                if len(p) > 4 and p[4]:
                    self._item(mon, p[4])
        elif kind == 'win':
            self.winner = arg
        elif kind == '-transform':
            pass

    def _item(self, mon: Mon, name: str, consumed: bool = False) -> None:
        mon.item = to_id(name)
        mon.item_known = True
        if consumed:
            mon.item_gone = True

    # ---------------------------------------------------------------- queries
    def remaining(self, kind: str, start: int, eff: str | None = None) -> int:
        base = DURATIONS.get(eff or kind, 5)
        elapsed = self.turn - max(start, 1) if self.turn >= 1 else 0
        left = base - elapsed
        if left <= 0:
            ext = EXTENDED.get(eff or kind)
            left = ext - elapsed if ext else 1
        return max(1, left)

    def field_state(self) -> dict:
        out = {'turn': self.turn, 'weather': None, 'terrain': None, 'pseudo': {}, 'sides': {}}
        if self.weather and self.weather[0] not in ('none',):
            w = self.weather[0]
            out['weather'] = {'id': w, 'duration': 0 if w in ('desolateland', 'primordialsea', 'deltastream') else
                              self.remaining('weather', self.weather[1])}
        if self.terrain:
            out['terrain'] = {'id': self.terrain[0], 'duration': self.remaining('terrain', self.terrain[1])}
        for eff, start in self.pseudo.items():
            out['pseudo'][eff] = self.remaining(eff, start, eff)
        for side in ('p1', 'p2'):
            conds = {}
            for eff, info in self.side_conditions[side].items():
                if eff in DURATIONS:
                    conds[eff] = {'duration': self.remaining(eff, info['turn'], eff)}
                else:
                    conds[eff] = {'layers': info.get('layers', 1)}
            out['sides'][side] = conds
        return out

    def side_mons(self, side: str) -> list[Mon]:
        return [m for (s, _), m in self.mons.items() if s == side]

    def active(self, side: str) -> list[Mon | None]:
        return [self.slots.get(side + 'a'), self.slots.get(side + 'b')]
