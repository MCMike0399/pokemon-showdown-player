"""Speed inference from observed move order (determinization filter).

Opposing stat spreads are sampled from ladder usage (search/priors.py). The public
log also tells us who moved first: within a turn, at equal priority, the faster
Pokemon acts first (the slower one under Trick Room). Every such observation
between one of OUR Pokemon (whose effective Speed we know exactly) and an opposing
one bounds the opposing effective Speed. A sampled set that contradicts an
observation describes a Pokemon the opponent cannot have.

`observations()` walks the log turn by turn with the search tracker, takes the
state just before the turn's first move (after switches, Megas and switch-in
weather), and records ordered (ours, theirs) pairs of moves with equal, known
priority whose EARLIER move precedes the first event that can change a Speed
mid-turn (Gen 9 re-sorts the queue after every action, so that action was chosen
with the snapshot Speeds). Turns with order manipulation (Quick Claw, Custap,
Quick Draw, After You, Quash, Instruct) are skipped, as are pairs whose priority
depends on an opposing ability we cannot see (Prankster, Gale Wings, Triage) and
opposing species whose possible abilities change Speed in ways the log hides
(Protosynthesis, Quark Drive, Stall, Mycelium Might). Speed ties are random, so
every bound is non-strict.

`violations(set_, constraints)` counts the observations a sampled set breaks,
using the Champions level-50 stat formula (data/mods/champions/scripts.js) with
the sampled nature, stat points, item and ability.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from battle_state import to_id
from search.tracker import Tracker

DEX = Path(__file__).resolve().parents[1] / 'cache' / 'dex' / 'gen9championsvgc2026regmc'
STAGE = {-6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3, 0: 1.0,
         1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2}
WEATHER_ABILITY = {'swiftswim': ('raindance', 'primordialsea'), 'chlorophyll': ('sunnyday', 'desolateland'),
                   'sandrush': ('sandstorm',), 'slushrush': ('snow', 'hail', 'snowscape')}
# Opposing abilities whose effect on order the log does not show reliably.
HIDDEN_ORDER_ABILITIES = {'protosynthesis', 'quarkdrive', 'stall', 'myceliummight'}
PRIORITY_ABILITIES = {'prankster', 'galewings', 'triage'}
SKIP_TURN = ('item: Quick Claw', 'Custap Berry', 'ability: Quick Draw', 'move: After You', 'move: Quash',
             'move: Instruct', 'ability: Dancer')
# Lines after which Speeds may differ from the snapshot (end of usable pairs in the turn).
SPEED_EVENTS = ('-boost', '-unboost', '-setboost', '-clearallboost', '-clearboost', '-clearnegativeboost',
                '-invertboost', '-copyboost', '-swapboost', '-sidestart', '-sideend', '-fieldstart', '-fieldend',
                '-weather', '-status', '-curestatus', '-enditem', '-item', '-formechange', 'detailschange',
                '-ability', '-transform', '-mega')
SPEED_START = re.compile(r'protosynthesis|quarkdrive|slowstart|unburden|speed', re.I)


@lru_cache(maxsize=1)
def _dex():
    return (json.loads((DEX / 'pokedex.json').read_text()), json.loads((DEX / 'moves.json').read_text()),
            json.loads((DEX / 'natures.json').read_text()))


def raw_speed(species: str, nature: str, evs: dict) -> int | None:
    """Champions level-50 Speed stat (levelclausemod branch of the mod's statModify)."""
    dex, _, natures = _dex()
    entry = dex.get(to_id(species))
    if not entry:
        return None
    base = entry['baseStats']['spe']
    points = int((evs or {}).get('spe', 0) or 0)
    stat = int((2 * base + 31 + max(2 * points - 1, 0)) * 50 / 100) + 5
    nat = natures.get(to_id(nature or 'serious'), {})
    if nat.get('plus') == 'spe':
        stat = int(int(stat * 110) / 100)
    elif nat.get('minus') == 'spe':
        stat = int(int(stat * 90) / 100)
    return stat


def effective_speed(species: str, nature: str, evs: dict, item: str, ability: str, ctx: dict) -> float | None:
    """Turn-order Speed: stat x stage x Tailwind x paralysis x item x ability."""
    stat = raw_speed(species, nature, evs)
    if stat is None:
        return None
    v = stat * STAGE[max(-6, min(6, int(ctx.get('boost', 0))))]
    ability = to_id(ability or '')
    item = '' if ctx.get('item_gone') else to_id(item or '')
    if ctx.get('tailwind'):
        v *= 2
    if item == 'choicescarf':
        v *= 1.5
    elif item == 'ironball':
        v *= 0.5
    weathers = WEATHER_ABILITY.get(ability)
    if weathers and ctx.get('weather') in weathers:
        v *= 2
    if ability == 'surgesurfer' and ctx.get('terrain') == 'electricterrain':
        v *= 2
    if ability == 'unburden' and ctx.get('item_gone'):
        v *= 2
    if ability == 'quickfeet' and ctx.get('status'):
        v *= 1.5
    elif ctx.get('status') == 'par':
        v *= 0.5
    return v


def _possible_abilities(species: str) -> set[str]:
    dex, _, _ = _dex()
    entry = dex.get(to_id(species)) or {}
    out = {to_id(a) for a in (entry.get('abilities') or {}).values()}
    base = dex.get(to_id(entry.get('baseSpecies') or '')) or {}
    return out | {to_id(a) for a in (base.get('abilities') or {}).values()}


def _priority(move: str, ability: str | None, species: str, terrain: str | None, hp_full: bool) -> float | None:
    """Move priority; None when it may depend on an ability we cannot see."""
    _, moves, _ = _dex()
    info = moves.get(to_id(move))
    if not info:
        return None
    pri = float(info.get('priority', 0))
    if to_id(move) == 'grassyglide' and terrain == 'grassyterrain':
        pri += 1
    qualifies = {'prankster': info.get('category') == 'Status',
                 'galewings': info.get('type') == 'Flying' and hp_full,
                 'triage': bool((info.get('flags') or {}).get('heal'))}
    if ability is None:  # opposing: unknown ability
        if any(qualifies[a] for a in PRIORITY_ABILITIES & _possible_abilities(species)):
            return None
        return pri
    ability = to_id(ability)
    if ability in qualifies and qualifies[ability]:
        pri += {'prankster': 1, 'galewings': 1, 'triage': 3}[ability]
    return pri


def _our_set(team_sets: list[dict], species: str) -> dict | None:
    from search.agent import _our_set as lookup
    return lookup(team_sets, species)


def _snapshot(tracker: Tracker, mon, side: str) -> dict:
    field = tracker.field_state()
    weather = (field.get('weather') or {}).get('id')
    terrain = (field.get('terrain') or {}).get('id')
    return {'boost': mon.boosts.get('spe', 0), 'tailwind': 'tailwind' in field['sides'].get(side, {}),
            'status': mon.status, 'item_gone': mon.item_gone, 'weather': weather, 'terrain': terrain,
            'trickroom': 'trickroom' in field['pseudo'], 'species': mon.species, 'hp_full': mon.hp >= 0.999}


def observations(log: list[str], me: str, team_sets: list[dict]) -> dict[str, list[dict]]:
    """Opposing Pokemon name -> Speed bounds {'ctx', 'bound', 'ge'} from the public log.

`ge` True: their effective Speed >= bound (they moved first without Trick Room, or
after us under it); False: <= bound."""
    tracker = Tracker(me)
    foe = tracker.foe
    out: dict[str, list[dict]] = {}
    turns: list[list[str]] = [[]]
    for line in log:
        if line.startswith('|turn|'):
            turns.append([line])
        else:
            turns[-1].append(line)
    for chunk in turns:
        moves, snap, frozen, skip = [], {}, False, any(s in line for line in chunk for s in SKIP_TURN)
        for line in chunk:
            p = line.split('|')
            kind = p[1] if len(p) > 1 else ''
            if kind == 'move' and not skip and len(p) > 3:
                rest = '|'.join(p[4:])
                if '[from]' not in rest:
                    mon = tracker._slot_mon(p[2])
                    if mon is not None:
                        if not snap:
                            snap = {(m.side, m.name): _snapshot(tracker, m, m.side)
                                    for s in ('p1', 'p2') for m in tracker.active(s) if m}
                        if (mon.side, mon.name) in snap:
                            # A move after a Speed change can still be the LATER half of a pair.
                            moves.append((mon, to_id(p[3]), frozen))
            elif snap and (kind in SPEED_EVENTS or (kind == '-start' and SPEED_START.search(line))):
                frozen = True
            tracker.line(line)
        # Pairs (A before B): ours vs theirs, same priority, both snapshotted.
        for i, (a, ma, late) in enumerate(moves):
            if late:
                continue  # chosen to act after Speeds changed: the snapshot no longer applies
            for b, mb, _ in moves[i + 1:]:
                if a.side == b.side:
                    continue
                ours, theirs = (a, b) if a.side == me else (b, a)
                theirs_first = theirs is a
                s_o, s_t = snap[(ours.side, ours.name)], snap[(theirs.side, theirs.name)]
                if HIDDEN_ORDER_ABILITIES & _possible_abilities(s_t['species']):
                    continue
                oset = _our_set(team_sets, s_o['species'])
                if oset is None:
                    continue
                o_ability = oset.get('ability')
                if '-Mega' in s_o['species']:
                    o_ability = (_dex()[0].get(to_id(s_o['species'])) or {}).get('abilities', {}).get('0', o_ability)
                p_o = _priority(ma if a is ours else mb, o_ability, s_o['species'], s_o['terrain'], s_o['hp_full'])
                p_t = _priority(ma if a is theirs else mb, None, s_t['species'], s_t['terrain'], s_t['hp_full'])
                if p_o is None or p_t is None or p_o != p_t:
                    continue
                bound = effective_speed(s_o['species'], oset.get('nature'), oset.get('evs') or {},
                                        oset.get('item'), o_ability, s_o)
                if bound is None:
                    continue
                tr = s_o['trickroom']
                out.setdefault(theirs.name, []).append(
                    {'ctx': s_t, 'bound': bound, 'ge': theirs_first != tr, 'turn': tracker.turn})
    return out


def violations(set_: dict, constraints: list[dict], tol: float = 0.5) -> int:
    """Observations a sampled opposing set contradicts (0 = consistent)."""
    dex, _, _ = _dex()
    bad = 0
    for c in constraints:
        ctx = c['ctx']
        species = ctx['species']
        item, ability = set_.get('item'), set_.get('ability')
        if '-Mega' in species:  # Mega forme at that time: its own ability, the stone is no Speed item
            item = ''
            ability = (dex.get(to_id(species)) or {}).get('abilities', {}).get('0', ability)
        v = effective_speed(species, set_.get('nature'), set_.get('evs') or {}, item, ability, ctx)
        if v is None:
            continue
        if (c['ge'] and v < c['bound'] - tol) or (not c['ge'] and v > c['bound'] + tol):
            bad += 1
    return bad

