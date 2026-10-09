"""Opponent set priors from Smogon ladder usage stats (chaos JSON), conditioned on observations.

Sets are sampled per determinization ("world"). Observed facts (revealed moves,
item, ability, Mega Evolution) are always respected; everything else is drawn from
the ladder distribution for that species.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from battle_state import to_id

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / 'cache' / 'usage'
STAT_ORDER = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')


@lru_cache(maxsize=4)
def load_usage(fmt: str = 'gen9championsvgc2026regmc', cutoff: str = '0') -> dict:
    path = STATS / f'{fmt}-{cutoff}.json'
    data = json.loads(path.read_text())['data']
    dex = json.loads((ROOT / 'cache' / 'dex' / fmt / 'pokedex.json').read_text())
    by_base: dict[str, list] = {}
    for name, entry in data.items():
        info = dex.get(to_id(name), {})
        base = info.get('baseSpecies') or name
        # Keep regional/gender formes (Indeedee-F, Arcanine-Hisui) as their own base.
        if '-Mega' not in name:
            base = name
        prepared = {
            'name': name, 'count': float(entry.get('Raw count', 1)),
            'mega': '-Mega' in name,
            'moves': _norm(entry.get('Moves', {})),
            'items': _norm(entry.get('Items', {})),
            'abilities': _norm(entry.get('Abilities', {})),
            'spreads': _norm(entry.get('Spreads', {}), top=60),
            'teammates': entry.get('Teammates', {}),
        }
        by_base.setdefault(to_id(base), []).append(prepared)
    return {'by_base': by_base, 'dex': dex}


def _norm(table: dict, top: int | None = None) -> list[tuple[str, float]]:
    items = [(k, float(v)) for k, v in table.items() if k and k != 'nothing' and v > 0]
    items.sort(key=lambda kv: -kv[1])
    if top:
        items = items[:top]
    total = sum(v for _, v in items) or 1.0
    return [(k, v / total) for k, v in items]


def _base_id(species: str, dex: dict) -> str:
    sid = to_id(species)
    info = dex.get(sid, {})
    if '-Mega' in species and info.get('baseSpecies'):
        return to_id(info['baseSpecies'])
    return sid


def species_weight(species: str, fmt: str = 'gen9championsvgc2026regmc', cutoff: str = '0') -> float:
    usage = load_usage(fmt, cutoff)
    entries = usage['by_base'].get(_base_id(species, usage['dex']), [])
    return sum(e['count'] for e in entries) or 1.0


def _pick(rng: random.Random, table: list[tuple[str, float]]):
    if not table:
        return None
    r = rng.random()
    acc = 0.0
    for key, w in table:
        acc += w
        if r <= acc:
            return key
    return table[-1][0]


def sample_set(rng: random.Random, species: str, observed: dict | None = None,
               fmt: str = 'gen9championsvgc2026regmc', cutoff: str = '0') -> dict:
    """Draw one full set for a public species, consistent with `observed`.

    observed keys: moves (ids), item (id or None), item_gone (bool), ability (id),
    mega (bool: already Mega Evolved), mega_species (str|None), species (current).
    """
    observed = observed or {}
    usage = load_usage(fmt, cutoff)
    dex = usage['dex']
    base = _base_id(species, dex)
    entries = list(usage['by_base'].get(base, []))
    if not entries:
        # Cosmetic formes (Sinistcha-Masterpiece, Vivillon-Pokeball, Maushold-Four...).
        alt = to_id(dex.get(base, {}).get('baseSpecies') or '')
        if alt and alt in usage['by_base']:
            entries = list(usage['by_base'][alt])
    base_entry = next((e for e in usage['by_base'].get(base, []) if not e['mega']), None)
    mega_species = observed.get('mega_species')
    item = observed.get('item')
    if observed.get('mega'):
        filtered = [e for e in entries if e['mega'] and (not mega_species or to_id(e['name']) == to_id(mega_species))]
        entries = filtered or entries
    elif item:
        # A revealed non-stone item rules out the Mega entries.
        stone = dex.get(to_id(species), {}).get('requiredItem')
        if not stone or to_id(stone) != item:
            entries = [e for e in entries if not e['mega']] or entries
    elif observed.get('mega_ruled_out'):
        entries = [e for e in entries if not e['mega']] or entries
    revealed_moves = [to_id(m) for m in observed.get('moves', [])][:4]
    if entries and revealed_moves:
        # Weight formes by how well they explain the revealed moves.
        weights = []
        for e in entries:
            table = dict(e['moves'])
            like = 1.0
            for m in revealed_moves:
                like *= table.get(m, 0.002)
            weights.append(e['count'] * like)
    else:
        weights = [e['count'] for e in entries]
    if not entries:
        return {'species': species, 'item': item or '', 'ability': observed.get('ability') or '',
                'nature': 'Serious', 'evs': {'hp': 22, 'atk': 11, 'def': 11, 'spa': 11, 'spd': 11, 'spe': 0},
                'moves': revealed_moves, 'level': 50, 'unknown_species': True}
    total = sum(weights)
    r, acc, entry = rng.random() * total, 0.0, entries[-1]
    for e, w in zip(entries, weights):
        acc += w
        if r <= acc:
            entry = e
            break
    # Moves: keep revealed ones, fill with usage-weighted draws without replacement.
    moves = list(dict.fromkeys(revealed_moves))
    pool = [(m, w) for m, w in entry['moves'] if m not in moves]
    while len(moves) < 4 and pool:
        total = sum(w for _, w in pool)
        r, acc = rng.random() * total, 0.0
        for i, (m, w) in enumerate(pool):
            acc += w
            if r <= acc:
                moves.append(m)
                pool.pop(i)
                break
        else:
            moves.append(pool.pop()[0])
    if entry['mega']:
        chosen_item = (entry['items'][0][0] if entry['items'] else '')
    elif item:
        chosen_item = item
    else:
        chosen_item = _pick(rng, entry['items']) or ''
    ability = observed.get('ability')
    if not ability:
        if entry['mega']:
            # The set carries the base forme's ability; Mega Evolution swaps it in battle.
            ability = _pick(rng, base_entry['abilities']) if base_entry and base_entry['abilities'] else None
            if not ability:
                abil = dex.get(base, {}).get('abilities', {})
                ability = to_id(abil.get('0', ''))
        else:
            ability = _pick(rng, entry['abilities'])
    ability = ability or ''
    spread = _pick(rng, entry['spreads']) or 'Serious:2/32/0/0/0/32'
    nature, _, numbers = spread.partition(':')
    try:
        values = [int(x) for x in numbers.split('/')]
    except ValueError:
        values = [0] * 6
    evs = {k: v for k, v in zip(STAT_ORDER, values)}
    base_name = species if dex.get(to_id(species)) and '-Mega' not in species else (dex.get(base, {}).get('name') or species)
    if base_entry is None and not dex.get(base, {}).get('abilities', {}).get('0'):
        pass
    legal = {to_id(a) for a in dex.get(to_id(base_name), {}).get('abilities', {}).values()}
    if legal and to_id(ability) not in legal:
        ability = to_id(sorted(legal)[0]) if not entry['mega'] else to_id(dex.get(to_id(base_name), {}).get('abilities', {}).get('0', ''))
    return {'species': base_name, 'item': chosen_item, 'ability': ability, 'nature': nature, 'evs': evs,
            'moves': moves, 'level': 50, 'usage_entry': entry['name']}


def sample_bench(rng: random.Random, preview: list[str], seen: list[str], need: int,
                 fmt: str = 'gen9championsvgc2026regmc', cutoff: str = '0') -> list[str]:
    """Pick the unseen brought Pokemon from the preview, weighted by ladder usage."""
    seen_ids = {_base_id(s, load_usage(fmt, cutoff)['dex']) for s in seen}
    candidates = [s for s in preview if _base_id(s, load_usage(fmt, cutoff)['dex']) not in seen_ids]
    chosen = []
    while len(chosen) < need and candidates:
        weights = [species_weight(s, fmt, cutoff) for s in candidates]
        total = sum(weights)
        r, acc = rng.random() * total, 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                chosen.append(candidates.pop(i))
                break
        else:
            chosen.append(candidates.pop())
    return chosen
