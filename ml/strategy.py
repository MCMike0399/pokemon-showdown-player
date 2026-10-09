"""Observable one-turn planning with uncertain opponent hypotheses.

This is a bounded approximate evaluator, not an omniscient simulator. Opponent
spreads, unrevealed moves, switches and damage rolls remain uncertain. Complete
legal actions are scored jointly; the actor can learn corrections to this prior.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

from battle_state import hp_fraction, to_id
from ml.mechanics import charging, known_mon
from ml.opening import own_form, preview_score, _hit
from ml.rain import fake_out_available, weather
from ml.tactics import damage, stat
from ml.strategic_mechanics import SLEEP

PROTECT = {'protect', 'detect', 'spikyshield', 'banefulbunker', 'silktrap', 'burningbulwark'}


def effects(values):
    return {to_id(v).removeprefix('move') for v in values}


def grounded(mon, features):
    return ('Flying' not in features.species(mon.get('species', '')).get('types', []) and
            to_id(mon.get('ability') or mon.get('baseAbility')) != 'levitate' and
            to_id(mon.get('item')) != 'airballoon')


def speed(mon, side, state, features):
    value = stat(mon, 'spe', features)
    ability = to_id(mon.get('ability') or mon.get('baseAbility'))
    w = weather(state, mon)
    if 'tailwind' in effects(state.get('hazards', {}).get(side, [])):
        value *= 2
    if 'par' in mon.get('condition', '').split() and ability != 'quickfeet':
        value *= .5
    if ((ability == 'swiftswim' and w in ('raindance', 'primordialsea')) or
        (ability == 'chlorophyll' and w in ('sunnyday', 'desolateland')) or
        (ability == 'sandrush' and w == 'sandstorm') or
        (ability == 'slushrush' and w in ('hail', 'snowscape')) or
        (ability == 'surgesurfer' and 'electricterrain' in effects(state.get('field', [])))):
        value *= 2
    if to_id(mon.get('item')) == 'choicescarf':
        value *= 1.5
    return value


def hit(actor, target, name, data, state, features, side, spread=False):
    ctx = {'state': state, 'feature_profile': 'mechanics-v1'}
    data = features.move_data(ctx, name, data, actor, [target])
    if state.get('_strategic_v4'):
        from ml.continuation import variable_power
        data = variable_power(name, data, actor)
    ability = to_id(actor.get('ability') or actor.get('baseAbility'))
    kind = data.get('type', '')
    priority = move_priority(actor, name, data, state, features)
    defenders = state.get('opp_actives' if side == 'mine' else 'my_actives', [])
    if priority > 0 and (any(to_id(m.get('ability') or m.get('baseAbility')) in
        ('armortail', 'queenlymajesty', 'dazzling') and hp_fraction(m.get('condition')) > 0 for m in defenders) or
        ('psychicterrain' in effects(state.get('field', [])) and grounded(target, features))):
        return 0.0
    if state.get('_strategic_v2'):
        from ml.strategic_mechanics import estimate, move_data
        data = move_data(name, data, actor, state, grounded(actor, features))
        amount = estimate(actor, target, name, data, {**state, 'weather': weather(state, actor)}, features, spread)
    else:
        amount = damage(actor, target, data, {**state, 'weather': weather(state, actor)}, features, spread)
    if ability == 'toughclaws' and data.get('flags', {}).get('contact'):
        amount *= 5325 / 4096
    if ability == 'adaptability' and kind in features.species(actor.get('species', '')).get('types', []):
        amount *= 4 / 3
    if ability in ('hugepower', 'purepower') and data.get('category') == 'Physical':
        amount *= 2
    if ability == 'technician' and 0 < data.get('basePower', 0) <= 60:
        amount *= 1.5
    item = to_id(actor.get('item'))
    if item == 'lifeorb':
        amount *= 1.3
    if item == ('choiceband' if data.get('category') == 'Physical' else 'choicespecs'):
        amount *= 1.5
    screen_side = 'theirs' if side == 'mine' else 'mine'
    screens = effects(state.get('hazards', {}).get(screen_side, []))
    if 'auroraveil' in screens or ('reflect' if data.get('category') == 'Physical' else 'lightscreen') in screens:
        amount *= 2 / 3
    field = effects(state.get('field', []))
    if grounded(actor, features) and ((kind == 'Electric' and 'electricterrain' in field) or
        (kind == 'Grass' and 'grassyterrain' in field) or (kind == 'Psychic' and 'psychicterrain' in field)):
        amount *= 1.3
    if grounded(target, features):
        if kind == 'Dragon' and 'mistyterrain' in field:
            amount *= .5
        if name in ('earthquake', 'bulldoze', 'magnitude') and 'grassyterrain' in field:
            amount *= .5
    special_defense = data.get('category') == 'Special' and not (state.get('_strategic_v2') and name in ('psyshock', 'psystrike', 'secretsword'))
    if to_id(target.get('item')) == 'assaultvest' and special_defense:
        amount /= 1.5
    if weather(state, target) == 'sandstorm' and 'Rock' in features.species(target.get('species', '')).get('types', []) and special_defense:
        amount /= 1.5
    return amount


def move_priority(actor, name, data, state, features):
    value = data.get('priority', 0)
    if name == 'grassyglide' and 'grassyterrain' in effects(state.get('field', [])) and grounded(actor, features):
        value = 1
    if data.get('category') == 'Status' and to_id(actor.get('ability') or actor.get('baseAbility')) == 'prankster':
        value += 1
    return value


def move_names(mon):
    return [to_id(m if isinstance(m, str) else m.get('id', m.get('move', '')))
            for m in (mon.get('moves') or mon.get('moves_known') or [])]


def max_hp(mon, features):
    if mon.get('stats'):
        try:
            return float(mon['condition'].split()[0].split('/')[1].rstrip('gry'))
        except (KeyError, IndexError, ValueError):
            pass
    return features.species(mon.get('species', '')).get('baseStats', {}).get('hp', 80) + 90


def set_health(mon, fraction):
    original = mon.get('condition', '100/100')
    denominator = original.split()[0].split('/')[-1].rstrip('gry') if '/' in original else '100'
    mon['condition'] = '0 fnt' if fraction <= 0 else f'{max(.0001, fraction * float(denominator))}/{denominator}' + (' ' + original.split(' ', 1)[1] if ' ' in original else '')


def hypotheses(ctx, mon, targets, features):
    """Shrink historical move frequencies toward visible moves and STAB threats.

    Executed moves are evidence of behavior, not labels for optimal play. Known
    sheets constrain candidates; otherwise observed moves and corpus estimates
    are hypotheses, never injected into the observed state as revealed facts.
    """
    species = to_id(features.species(mon.get('species', '')).get('baseSpecies') or mon.get('species', ''))
    corpus = ctx.get('strategy_knowledge', {}).get('moves', {})
    counts = Counter(corpus.get(species, {}))
    if to_id(mon.get('species')) != species:
        counts.update(corpus.get(to_id(mon.get('species')), {}))
    names = move_names(mon)
    sheet = next((s for s in ctx['state'].get('opp_team_sheet', []) if to_id(s.get('species')) == species), None)
    if sheet:
        names = move_names(sheet)
    else:
        names = list(dict.fromkeys(names + [n for n, _ in sorted(counts.items(), key=lambda x: -x[1])[:5]]))
    recent = Counter(to_id(e.get('move')) for e in ctx['state'].get('history', [])
                     if e.get('side') == 'theirs' and e.get('slot') == mon.get('slot', '')[-1:]
                     and to_id(e.get('move')) in move_names(mon))
    candidates = []
    for name in names:
        data = features.dex.get('moves', {}).get(name, {})
        if not data or (name == 'fakeout' and not fake_out_available(ctx, mon)):
            continue
        extra_support = SLEEP if ctx['state'].get('_strategic_v2') else set()
        if ctx['state'].get('_strategic_v4'):
            from ml.continuation import SETUP, RECOVERY
            extra_support = extra_support | SETUP.keys() | RECOVERY
            if name in RECOVERY and hp_fraction(mon.get('condition')) >= .9:
                continue
        if not data.get('basePower') and name not in PROTECT | extra_support | {'tailwind', 'trickroom', 'helpinghand', 'wideguard', 'thunderwave', 'willowisp', 'spore', 'followme', 'ragepowder'}:
            continue
        weight = 2 + math.sqrt(counts.get(name, 0)) + min(3, recent[name])
        if ctx['state'].get('_strategic_v2') and name in SLEEP:
            weight *= 1.5
        if ctx['state'].get('_strategic_v4') and name in RECOVERY:
            weight *= 2.5 * (1 - hp_fraction(mon.get('condition')))
        if name in PROTECT:
            weight *= 1.5 if hp_fraction(mon.get('condition')) < .4 else .65
            if recent[name] and any(e.get('turn') == ctx['state'].get('turn', 0) - 1 and to_id(e.get('move')) in PROTECT and e.get('slot') == mon.get('slot', '')[-1:] and e.get('side') == 'theirs' for e in ctx['state'].get('history', [])):
                weight *= .3
        candidates.append((name, data, weight))
    if not sheet and not any(d.get('basePower') for _, d, _ in candidates):
        category = 'Physical' if stat(mon, 'atk', features) >= stat(mon, 'spa', features) else 'Special'
        for kind in features.species(mon.get('species', '')).get('types', ['Normal']):
            candidates.append(('estimated' + kind, {'type': kind, 'category': category,
                               'basePower': 90, 'accuracy': 100, 'target': 'normal'}, 3))
    attacks = sorted((c for c in candidates if c[1].get('basePower')), key=lambda c:
                     max((hit(mon, t, c[0], c[1], ctx['state'], features, 'theirs') for t in targets), default=0) * c[2], reverse=True)[:2]
    support = sorted((c for c in candidates if not c[1].get('basePower')), key=lambda c: -c[2])[:2 if ctx['state'].get('_strategic_v4') else 1]
    result = []
    for name, data, weight in attacks + support:
        if ctx['state'].get('_strategic_v2') and name in SLEEP:
            for index, target in enumerate(targets):
                from ml.strategic_mechanics import sleep_probability
                chance = sleep_probability(mon, target, name, data, ctx['state'], features)
                if chance:
                    result.append((name, data, index, weight * chance))
        elif data.get('target') in ('allAdjacent', 'allAdjacentFoes') or not data.get('basePower'):
            result.append((name, data, None, weight))
        else:
            ranked = sorted(range(len(targets)), key=lambda i: hit(mon, targets[i], name, data, ctx['state'], features, 'theirs') / max(.2, hp_fraction(targets[i].get('condition'))), reverse=True)
            for rank, i in enumerate(ranked):
                result.append((name, data, i, weight * (1 if rank == 0 else .45)))
    # An observed benched Pokemon can switch into the targeted position. No
    # unobserved set, stat spread or private selected-four is accessed.
    active_ids = {m.get('ident') for m in ctx['state'].get('opp_actives', [])}
    bench = [known_mon(m, ctx['state'], features) for m in ctx['state'].get('opp_revealed', [])
             if m.get('ident') not in active_ids and hp_fraction(m.get('condition')) > 0]
    if bench:
        incoming = max(bench, key=lambda m: -sum(hit(t, m, n, features.dex.get('moves', {}).get(n, {}), ctx['state'], features, 'mine') for t in targets for n in move_names(t)))
        result.append(('switch', {'hypothesis_switch': {**incoming, 'slot': mon.get('slot'), 'boosts': {}, 'volatiles': []}}, None, sum(r[3] for r in result) * .12))
    # A disclosed Mega stone licenses a prospective form hypothesis; raw species
    # alone does not disclose the opposing choice of stone or evolution timing.
    item = to_id(mon.get('item'))
    form = next((d for d in features.dex.get('pokedex', {}).values() if item and to_id(d.get('requiredItem')) == item and to_id(d.get('baseSpecies')) == to_id(mon.get('species'))), None)
    mine = ctx['state'].get('side_id', ctx.get('request', {}).get('side', {}).get('id', 'p1'))
    mega_used = any(features.species(m.get('species', '')).get('isMega') for m in ctx['state'].get('opp_revealed', []) + ctx['state'].get('opp_actives', [])) or any(line.startswith('|-mega|') and not line.split('|')[2].startswith(mine) for line in ctx.get('public_log', []))
    if form and not mega_used:
        prospective = {**mon, 'species': form['name'], 'ability': form.get('abilities', {}).get('0', ''),
                       'stats': {}}
        result = [(n, d, t, w * .5) for n, d, t, w in result] + [(n, {**d, 'hypothesis_form': prospective}, t, w * .5) for n, d, t, w in result if n != 'switch']
    total = sum(row[3] for row in result)
    return [(n, d, t, w / total) for n, d, t, w in result] or [('', {}, None, 1)]


def prepare(ctx, features):
    if '_strategy' in ctx:
        return ctx['_strategy']
    own = ctx['state'].get('my_actives', [])
    enemy = [known_mon(m, ctx['state'], features) for m in ctx['state'].get('opp_actives', []) if hp_fraction(m.get('condition')) > 0]
    options = [hypotheses(ctx, m, own, features) for m in enemy]
    def possible(rows):
        if sum(bool(r[1].get('hypothesis_form')) for r in rows) > 1:
            return False
        incoming = [r[1]['hypothesis_switch'].get('ident', r[1]['hypothesis_switch'].get('species')) for r in rows if r[1].get('hypothesis_switch')]
        return len(incoming) == len(set(incoming))
    joint = sorted((rows for rows in itertools.product(*options) if possible(rows)), key=lambda rows: -math.prod(r[3] for r in rows))[:6]
    total = sum(math.prod(r[3] for r in rows) for rows in joint)
    scenarios = [(rows, math.prod(r[3] for r in rows) / total) for rows in joint] if total else [((), 1)]
    ctx['_strategy'] = {'own': own, 'enemy': enemy, 'scenarios': scenarios, 'components': {}, 'scores': {}, 'roles': {}}
    return ctx['_strategy']


def component(ctx, slot, text, features, prepared):
    key = (slot, text)
    if key in prepared['components']:
        return prepared['components'][key]
    parts = text.strip().split()
    own = prepared['own']
    actor = own[slot] if slot < len(own) else {}
    name, data, target = '', {}, None
    if parts[0] == 'switch':
        actor = ctx['state']['my_party'][int(parts[1]) - 1]
    elif parts[0] == 'move' and slot < len(own):
        actor = own_form(ctx, actor, features, any(p in ('mega', 'megax', 'megay') for p in parts))
        move = actor.get('moves', [])[int(parts[1]) - 1]
        name = to_id(move.get('id', move.get('move', '')))
        data = dict(features.dex.get('moves', {}).get(name, {}))
        # Weather can change through the partner's switch before this executes.
        data['request_move'] = move
        target = next((int(p) for p in parts[2:] if p.lstrip('+-').isdigit()), None)
    row = (parts[0], actor, name, data, target)
    prepared['components'][key] = row
    return row


def role_value(ctx, mon, prepared, features):
    """Preserve a scarce answer to a plausible remaining threat, not raw count."""
    key = (mon.get('species'), tuple(sorted(mon.get('stats', {}).items())))
    if key in prepared['roles']:
        return prepared['roles'][key]
    revealed = ctx['state'].get('opp_revealed', prepared['enemy'])
    enemies = [m for m in revealed if hp_fraction(m.get('condition')) > 0]
    known = {to_id(features.species(m.get('species', '')).get('baseSpecies') or m.get('species')) for m in revealed}
    if len(revealed) < 4:
        enemies += [{'species': s, 'condition': '100/100'} for s in ctx['state'].get('opp_preview', []) if to_id(s) not in known]
    party = [m for m in ctx['state'].get('my_party', []) if hp_fraction(m.get('condition')) > 0]
    base = to_id(features.species(mon.get('species', '')).get('baseSpecies') or mon.get('species'))
    def coverage(actor, target):
        return max((hit(actor, target, name, features.dex.get('moves', {}).get(name, {}), ctx['state'], features, 'mine') for name in move_names(actor)), default=0)
    scarce = 0
    for enemy in enemies:
        best = coverage(mon, enemy)
        others = [m for m in party if to_id(features.species(m.get('species', '')).get('baseSpecies') or m.get('species')) != base]
        alternative = max((coverage(m, enemy) for m in others), default=0)
        scarce = max(scarce, min(.5, max(0, best - alternative - .25)))
    # A weather setter can also be the scarce enabler for a surviving partner.
    if to_id(mon.get('ability') or mon.get('baseAbility')) == 'drizzle' and any('electroshot' in move_names(m) or to_id(m.get('ability')) == 'swiftswim' for m in party):
        scarce = max(scarce, .25)
    prepared['roles'][key] = scarce
    return scarce


def initiative_value(ctx, ours, enemies, state, features, room):
    """Speed control is useful when it advances the remaining offensive plan."""
    party = [m for m in ctx['state'].get('my_party', []) if hp_fraction(m.get('condition')) > 0]
    def opportunities(team):
        numerator = denominator = 0.0
        for mon in team:
            for enemy in enemies:
                if hp_fraction(mon.get('condition')) <= 0 or hp_fraction(enemy.get('condition')) <= 0:
                    continue
                pressure = max((hit(mon, enemy, name, features.dex.get('moves', {}).get(name, {}), state, features, 'mine') for name in move_names(mon)), default=0)
                weight = min(1, pressure) * hp_fraction(mon.get('condition'))
                a, b = speed(mon, 'mine', state, features), speed(enemy, 'theirs', state, features)
                first = 0 if a == b else 1 if (a < b if room else a > b) else -1
                numerator += first * weight
                denominator += weight
        return numerator / denominator if denominator else 0
    return .6 * opportunities(ours) + .4 * opportunities(party or ours)


def simulate(ctx, ours, theirs, prepared, features, reverse_ties=False):
    """Resolve estimated actions in initiative order, including denial of actions."""
    mons = [dict(c[1]) for c in ours] + [dict(row[1].get('hypothesis_switch', m)) for row, m in zip(theirs, prepared['enemy'])]
    n = len(ours)
    initial = [hp_fraction(m.get('condition')) for m in mons]
    hp = initial[:]
    state = {**ctx['state'], 'my_actives': mons[:n], 'opp_actives': mons[n:]}
    value = 0.0
    room = 'trickroom' in effects(state.get('field', []))
    # Ordinary switch order belongs to outgoing actions, not incoming speed.
    our_switches = [i for i, c in enumerate(ours) if c[0] == 'switch']
    def switch_order(i):
        outgoing = prepared['own'][i] if i < n else prepared['enemy'][i - n]
        actor = mons[i] if ctx['state'].get('request_type') == 'forceSwitch' else outgoing
        action_speed = speed(actor, 'mine' if i < n else 'theirs', ctx['state'], features)
        return (action_speed if room else -action_speed, -i if reverse_ties else i)
    switches = sorted(our_switches + [n + j for j, row in enumerate(theirs) if row[0] == 'switch'], key=switch_order)
    for i in switches:
        ability = to_id(mons[i].get('ability') or mons[i].get('baseAbility'))
        if ability in ('drizzle', 'drought', 'sandstream', 'snowwarning'):
            state['weather'] = {'drizzle': 'RainDance', 'drought': 'SunnyDay', 'sandstream': 'Sandstorm', 'snowwarning': 'Snowscape'}[ability]
        terrains = {'psychicsurge': 'PsychicTerrain', 'grassysurge': 'GrassyTerrain', 'electricsurge': 'ElectricTerrain', 'mistysurge': 'MistyTerrain'}
        if ability in terrains:
            state['field'] = [v for v in state.get('field', []) if to_id(v).removeprefix('move') not in {'psychicterrain', 'grassyterrain', 'electricterrain', 'mistyterrain'}] + [terrains[ability]]
        if ability == 'intimidate':
            for j in (range(n, len(mons)) if i < n else range(n)):
                if to_id(mons[j].get('ability')) not in ('clearbody', 'whitesmoke', 'fullmetalbody', 'innerfocus', 'owntempo', 'oblivious', 'scrappy'):
                    delta = 1 if to_id(mons[j].get('ability')) == 'defiant' else 2 if to_id(mons[j].get('ability')) == 'competitive' else -1
                    key = 'spa' if to_id(mons[j].get('ability')) == 'competitive' else 'atk'
                    mons[j]['boosts'] = {**mons[j].get('boosts', {}), key: max(-6, min(6, mons[j].get('boosts', {}).get(key, 0) + delta))}
        if i < n and ctx['state'].get('request_type') != 'forceSwitch':
            value -= .10  # Lost attack opportunity, offset by survival/entry effects.
    # Evolution abilities activate after all switches; disclosed opposing Drought
    # can therefore overwrite a same-turn Politoed entry.
    for j, row in enumerate(theirs):
        if row[1].get('hypothesis_form'):
            form = row[1]['hypothesis_form']
            mons[n + j].update(species=form['species'], ability=form['ability'], stats=form['stats'])
            ability = to_id(mons[n + j].get('ability'))
            if ability in ('drought', 'drizzle'):
                state['weather'] = 'SunnyDay' if ability == 'drought' else 'RainDance'
            if ability == 'intimidate':
                for target in mons[:n]:
                    if to_id(target.get('ability') or target.get('baseAbility')) not in ('clearbody', 'whitesmoke', 'fullmetalbody', 'innerfocus', 'owntempo', 'oblivious', 'scrappy'):
                        target['boosts'] = {**target.get('boosts', {}), 'atk': max(-6, target.get('boosts', {}).get('atk', 0) - 1)}
    state.update(my_actives=mons[:n], opp_actives=mons[n:])
    events = []
    for i, (kind, actor, name, data, code) in enumerate(ours):
        if kind == 'move':
            targets = [n + j for j, m in enumerate(prepared['enemy']) if m.get('slot', '').endswith(chr(96 + code))] if code and code > 0 else [-code - 1] if code and code < 0 else list(range(n, len(mons)))
            events.append((i, name, data, targets))
    for j, (name, data, target, _) in enumerate(theirs):
        if name != 'switch':
            targets = [target] if target is not None else list(range(n))
            events.append((n + j, name, data, targets))
    protected, flinched, wide, helper, redirects = set(), set(), set(), {}, {}
    availability = [1.0] * len(mons)
    while events:
        def initiative(event):
            i, name, data, _ = event
            priority = move_priority(mons[i], name, data, state, features)
            if name in PROTECT or name == 'wideguard':
                priority = 4 if name in PROTECT else 3
            if name == 'helpinghand':
                priority = 5
            s = speed(mons[i], 'mine' if i < n else 'theirs', state, features)
            return (priority, -s if room else s, i if reverse_ties else -i)
        event = max(events, key=initiative)
        events.remove(event)
        i, name, data, targets = event
        if hp[i] <= 0 or i in flinched or 'slp' in mons[i].get('condition', '').split() or 'frz' in mons[i].get('condition', '').split():
            continue
        side = 'mine' if i < n else 'theirs'
        from ml.strategic_mechanics import SLEEP, move_data, sleep_probability
        if state.get('_strategic_v2'):
            data = move_data(name, data, mons[i], state, grounded(mons[i], features))
            if data.get('target') in ('allAdjacent', 'allAdjacentFoes'):
                targets = list(range(n, len(mons))) if i < n else list(range(n))
        sign = 1 if i < n else -1
        if name in PROTECT:
            previous = any(e.get('side') == side and e.get('slot') == mons[i].get('slot', '')[-1:] and e.get('turn') == state.get('turn', 0) - 1 and to_id(e.get('move')) in PROTECT for e in state.get('history', []))
            if not previous:
                protected.add(i)
            continue
        if state.get('_strategic_v4'):
            from ml.continuation import SETUP, RECOVERY, boosts, potential
            if name in SETUP:
                opposing = mons[n:] if i < n else mons[:n]
                before = potential(mons[i], opposing, state, features, side, hit, speed)
                mons[i]['boosts'] = boosts(mons[i], name)
                after = potential(mons[i], opposing, state, features, side, hit, speed)
                value += sign * .45 * (after - before) * availability[i]
                continue
            if name in RECOVERY:
                hp[i] = min(1, hp[i] + .5 * availability[i])
                set_health(mons[i], hp[i])
                continue
        if name == 'wideguard':
            wide.add(side)
            continue
        if name in ('followme', 'ragepowder'):
            redirects[side] = (i, name)
            continue
        if name == 'helpinghand':
            partner = next((j for j in (range(n) if i < n else range(n, len(mons))) if j != i), None)
            if partner is not None and hp[partner] > 0 and any(e[0] == partner and e[2].get('basePower') for e in events):
                helper[partner] = 1.5
            elif i < n:
                value -= .10
            continue
        if name in ('tailwind', 'trickroom'):
            before = initiative_value(ctx, mons[:n], mons[n:], state, features, room)
            if name == 'trickroom':
                room = not room
            elif 'tailwind' not in effects(state.get('hazards', {}).get(side, [])):
                state['hazards'] = {**state.get('hazards', {}), side: list(state.get('hazards', {}).get(side, [])) + ['Tailwind']}
            after = initiative_value(ctx, mons[:n], mons[n:], state, features, room)
            value += .4 * (after - before)
            continue
        if state.get('_strategic_v2') and name in SLEEP:
            for j in targets[:1]:
                if j in protected or hp[j] <= 0:
                    continue
                defending = 'mine' if j < n else 'theirs'
                if defending in redirects:
                    redirect, powder = redirects[defending]
                    immune = powder == 'ragepowder' and ('Grass' in features.species(mons[i].get('species', '')).get('types', []) or to_id(mons[i].get('item')) == 'safetygoggles' or to_id(mons[i].get('ability')) == 'overcoat')
                    if hp[redirect] > 0 and not immune:
                        j = redirect
                if j in protected or 'substitute' in effects(mons[j].get('volatiles', [])) or 'safeguard' in effects(state.get('hazards', {}).get(defending, [])):
                    continue
                chance = sleep_probability(mons[i], mons[j], name, data, state, features) * availability[i]
                value += sign * .25 * chance
                availability[j] *= 1 - chance
            continue
        if name in ('thunderwave', 'willowisp', 'spore'):
            for j in targets[:1]:
                if j in protected or hp[j] <= 0:
                    continue
                blocked = ('mistyterrain' in effects(state.get('field', [])) and grounded(mons[j], features)) or (name == 'spore' and ('Grass' in features.species(mons[j].get('species', '')).get('types', []) or to_id(mons[j].get('item')) == 'safetygoggles' or ('electricterrain' in effects(state.get('field', [])) and grounded(mons[j], features))))
                if not blocked:
                    value += sign * .25
            continue
        request_move = data.get('request_move', {'id': name, 'target': data.get('target', 'normal')})
        charge = name == 'electroshot' and charging({'state': state}, mons[i], request_move)
        if charge:
            value += sign * .06
            continue
        if name == 'fakeout' and not fake_out_available(ctx, mons[i]):
            continue
        spread = data.get('target') in ('allAdjacent', 'allAdjacentFoes')
        if not spread and data.get('basePower') and len(targets) > 1:
            targets = targets[:1]  # Locked target unknown: never invent a spread hit.
        # Standard doubles retargets a fainted foe to the other opposing slot;
        # a fainted ally target continues to fail.
        hostile = not targets or all((j < n) != (i < n) for j in targets)
        if not spread and data.get('basePower') and hostile and not any(hp[j] > 0 for j in targets):
            replacements = [j for j in (range(n, len(mons)) if i < n else range(n)) if hp[j] > 0]
            targets = replacements[:1]
        defending_side = 'theirs' if i < n else 'mine'
        if not spread and len(targets) == 1 and (targets[0] < n) != (i < n) and defending_side in redirects:
            j, redirection = redirects[defending_side]
            powder_immune = 'Grass' in features.species(mons[i].get('species', '')).get('types', []) or to_id(mons[i].get('item')) == 'safetygoggles' or to_id(mons[i].get('ability')) == 'overcoat'
            if hp[j] > 0 and not (redirection == 'ragepowder' and powder_immune):
                targets = [j]
        if data.get('target') == 'allAdjacent':
            targets = targets + [j for j in (range(n) if i < n else range(n, len(mons))) if j != i]
        if name == 'electroshot' and request_move.get('target') not in (None, 'scripted'):
            mons[i]['boosts'] = {**mons[i].get('boosts', {}), 'spa': min(6, mons[i].get('boosts', {}).get('spa', 0) + 1)}
        for j in targets:
            if j >= len(mons) or hp[j] <= 0 or j in protected:
                continue
            target_side = 'mine' if j < n else 'theirs'
            if spread and target_side in wide:
                continue
            amount = hit(mons[i], mons[j], name, data, state, features, side, spread and sum(hp[t] > 0 for t in targets) > 1) * helper.get(i, 1) * availability[i]
            if amount <= 0:
                continue
            before = hp[j]
            # Visible survival items/abilities prevent a full-health one-hit KO.
            if before == 1 and (to_id(mons[j].get('item')) == 'focussash' or to_id(mons[j].get('ability')) == 'sturdy'):
                amount = min(amount, .99)
            hp[j] = max(0, before - amount)
            if name == 'fakeout' and to_id(mons[j].get('ability')) not in ('innerfocus', 'shielddust') and to_id(mons[j].get('item')) != 'covertcloak':
                flinched.add(j)
            # Exact own maxHP must survive intermediate damage; damage() reads
            # the request denominator when exact stats exist.
            set_health(mons[j], hp[j])
            dealt = before - hp[j]
            if name in ('leechlife', 'gigadrain', 'drainpunch', 'hornleech', 'drainingkiss', 'oblivionwing', 'paraboliccharge') and dealt > 0:
                ratio = .75 if name in ('drainingkiss', 'oblivionwing') else .5
                hp[i] = min(1, hp[i] + dealt * max_hp(mons[j], features) * ratio / max(1, max_hp(mons[i], features)))
                set_health(mons[i], hp[i])
            if name in ('flareblitz', 'bravebird', 'doubleedge', 'woodhammer', 'headsmash') and to_id(mons[i].get('ability')) not in ('rockhead', 'magicguard'):
                ratio = .5 if name == 'headsmash' else 1 / 3
                hp[i] = max(0, hp[i] - dealt * max_hp(mons[j], features) * ratio / max(1, max_hp(mons[i], features)))
                set_health(mons[i], hp[i])
    value += sum((initial[j] - hp[j]) + .8 * (initial[j] > 0 and hp[j] == 0) for j in range(n, len(mons)))
    value -= sum((initial[j] - hp[j]) * (1 + .2 * role_value(ctx, ours[j][1], prepared, features)) +
                 (.9 + role_value(ctx, ours[j][1], prepared, features)) * (initial[j] > 0 and hp[j] == 0) for j in range(n))
    value -= .08 * sum(c[2] in PROTECT for c in ours)
    if ctx.get('feature_profile') in ('strategic-v2', 'strategic-v3', 'strategic-v4'):
        from ml.strategic_mechanics import unproductive_solo_protect
        value -= unproductive_solo_protect(ctx, ours, mons[n:])
    # Residual damage makes a lost tempo turn consequential. Known healing is
    # included; undisclosed recovery and field-expiry timers remain uncertain.
    for i, mon in enumerate(mons):
        if hp[i] <= 0:
            continue
        status = mon.get('condition', '').split()[1:]
        residual = 1 / 16 if 'brn' in status else 1 / 8 if set(status) & {'psn', 'tox'} else 0
        kind = weather(state, mon)
        if kind == 'sandstorm' and not set(features.species(mon.get('species', '')).get('types', [])) & {'Rock', 'Ground', 'Steel'} and to_id(mon.get('ability')) not in ('magicguard', 'overcoat', 'sandveil', 'sandrush', 'sandforce') and to_id(mon.get('item')) != 'safetygoggles':
            residual += 1 / 16
        if to_id(mon.get('ability')) == 'magicguard':
            residual = 0
        if to_id(mon.get('item')) == 'leftovers':
            residual -= 1 / 16
        sign = -1 if i < n else 1
        value += sign * (min(hp[i], residual) + .9 * (residual >= hp[i]) if residual > 0 else max(residual, hp[i] - 1))
    value -= .04 * sum(ours[i][1].get('species') != prepared['own'][i].get('species') for i in range(min(n, len(prepared['own']))) if ours[i][0] == 'move')
    # Value next-turn coverage for forced replacements and resistive pivots.
    for i in our_switches:
        if hp[i] > 0:
            best = max((hit(mons[i], m, name, features.dex.get('moves', {}).get(name, {}), state, features, 'mine') for name in move_names(mons[i]) for m in mons[n:] if hp_fraction(m.get('condition')) > 0), default=0)
            value += .25 * min(best, 1)
    return value


def turn_score(ctx, choice, features):
    prepared = prepare(ctx, features)
    ours = [component(ctx, i, text, features, prepared) for i, text in enumerate(choice.split(','))]
    key = tuple((c[0], c[1].get('species'), c[2], c[4]) for c in ours)
    if key not in prepared['scores']:
        all_mons = [c[1] for c in ours] + prepared['enemy']
        speeds = [speed(m, 'mine' if i < len(ours) else 'theirs', ctx['state'], features) for i, m in enumerate(all_mons)]
        tied = len(speeds) != len(set(speeds))
        values = [((simulate(ctx, ours, rows, prepared, features) + simulate(ctx, ours, rows, prepared, features, True)) / 2 if tied else simulate(ctx, ours, rows, prepared, features), p) for rows, p in prepared['scenarios']]
        # Modest pessimism protects against committing to one fragile prediction.
        expected = sum(v * p for v, p in values)
        our_resources = sum(hp_fraction(m.get('condition')) for m in ctx['state'].get('my_party', []))
        revealed = ctx['state'].get('opp_revealed', prepared['enemy'])
        foe_resources = sum(hp_fraction(m.get('condition')) for m in revealed) + max(0, 4 - len(revealed))
        pessimism = max(.05, min(.25, .15 + .05 * (our_resources - foe_resources)))
        prepared['scores'][key] = 3 * ((1 - pessimism) * expected + pessimism * min(v for v, _ in values))
    return prepared['scores'][key]


def opening_score(ctx, choice, features):
    value = preview_score(ctx, choice, features)
    # Electro Shot acquires +1 SpA before its hit. The legacy opening tables
    # omit it; correcting only this candidate avoids changing old likelihoods.
    cache = ctx['_opening_preview']
    if not ctx.get('_strategy_preview_boosted'):
        for rain, table in cache['tables'].items():
            simulated = {**ctx, 'state': {**ctx['state'], 'weather': 'RainDance' if rain else None}}
            for i, pair in enumerate(cache['variants']):
                for form, actor in enumerate(pair):
                    if 'electroshot' not in move_names(actor):
                        continue
                    boosted = {**actor, 'boosts': {**actor.get('boosts', {}), 'spa': min(6, actor.get('boosts', {}).get('spa', 0) + 1)}}
                    for j, enemy in enumerate(cache['enemies']):
                        amount = _hit(boosted, enemy, 'electroshot', features.dex.get('moves', {}).get('electroshot', {}), simulated, features)
                        table[i][form]['attack'][j] = max(table[i][form]['attack'][j], amount * (1 if rain else .45))
                        table[i][form]['immediate'][j] = max(table[i][form]['immediate'][j], amount * (1 if rain else .15))
        ctx['_strategy_preview_boosted'] = True
        value = preview_score(ctx, choice, features)
    picked = [ctx['state']['my_party'][int(x) - 1] for x in choice[5:].split(',')]
    leads = picked[:2]
    enemy = [known_mon({'species': s, 'condition': '100/100'}, ctx['state'], features) for s in ctx['state'].get('opp_preview', [])]
    if not enemy:
        return value
    speeds = [speed(m, 'mine', ctx['state'], features) for m in leads]
    foe_speeds = [speed(m, 'theirs', ctx['state'], features) for m in enemy]
    names = [set(move_names(m)) for m in leads]
    if any('trickroom' in moves for moves in names):
        slow = sum(s < t for s in speeds for t in foe_speeds) / (len(speeds) * len(foe_speeds))
        value += .65 * (2 * slow - 1)
    physical = sum(stat(m, 'atk', features) > stat(m, 'spa', features) for m in enemy) / len(enemy)
    if any(to_id(m.get('ability') or m.get('baseAbility')) == 'intimidate' for m in leads):
        value += .25 * physical
    blocked = any(to_id(m.get('ability') or m.get('baseAbility')) in ('armortail', 'dazzling', 'queenlymajesty') for m in enemy)
    value += .25 * any('fakeout' in moves for moves in names) * (not blocked)
    rain_lead = any(to_id(m.get('ability') or m.get('baseAbility')) == 'drizzle' for m in leads)
    sun_risk = any(to_id(m.get('species')).startswith(('charizard', 'torkoal', 'ninetales')) for m in enemy)
    if rain_lead and sun_risk and any('electroshot' in moves for moves in names):
        value -= .4  # A plausible weather overwrite makes the opening fragile.
    return 2 * value


def score(ctx, choice, features):
    if ctx.get('feature_profile') in ('strategic-v2', 'strategic-mechanics-v2', 'strategic-v3', 'strategic-v4'):
        if '_strategic_v2_context' not in ctx:
            ctx['_strategic_v2_context'] = {**ctx, 'state': {**ctx['state'], '_strategic_v2': True, '_strategic_v4': ctx.get('feature_profile') == 'strategic-v4'}}
        ctx = ctx['_strategic_v2_context']
    if choice.startswith('team '):
        return opening_score(ctx, choice, features)
    return turn_score(ctx, choice, features)
