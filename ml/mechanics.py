"""Opt-in joint pressure corrections using observable mechanics, not exact damage.

Preserves the incumbent's state and preview representation. Unknown opponent
commands, spreads and unrevealed abilities remain unknown.
"""
from __future__ import annotations

from battle_state import hp_fraction, to_id
from ml.rain import fake_out_available, mega_actor, weather


def charging(ctx, actor, move):
    # Locked releases omit a selectable target. The public stream need not
    # announce the private twoturnmove volatile.
    return (to_id(move.get('id', move.get('move', ''))) == 'electroshot' and
            move.get('target') not in (None, 'scripted') and
            weather(ctx['state'], actor) not in ('raindance', 'primordialsea') and
            to_id(actor.get('item', '')) != 'powerherb' and
            not {to_id(v) for v in actor.get('volatiles', [])} & {'electroshot', 'twoturnmove'})


def known_mon(mon, state, features):
    """Sheet facts fill unknowns; revealed changes/removals take precedence."""
    base = to_id(features.species(mon.get('species', '')).get('baseSpecies') or mon.get('species', ''))
    matches = [m for m in state.get('opp_team_sheet', []) if to_id(m.get('species')) == base]
    result = {**matches[0], **mon} if len(matches) == 1 else dict(mon)
    species = features.species(result.get('species', ''))
    if species.get('requiredItem') and not mon.get('ability'):
        result['ability'] = species.get('abilities', {}).get('0', '')
    return result


def multiplier(kind, actor, target, features):
    effective = features.effectiveness(kind, target)
    ability = to_id(target.get('ability') or target.get('baseAbility') or '')
    bypass = to_id(actor.get('ability') or actor.get('baseAbility') or '') in ('moldbreaker', 'teravolt', 'turboblaze')
    if not bypass and ((kind == 'Ground' and ability == 'levitate') or
                       (kind == 'Water' and ability in ('waterabsorb', 'stormdrain', 'dryskin')) or
                       (kind == 'Electric' and ability in ('voltabsorb', 'lightningrod', 'motordrive')) or
                       (kind == 'Fire' and ability == 'flashfire')):
        effective = 0
    if kind == 'Ground' and to_id(target.get('item', '')) == 'airballoon':
        effective = 0
    return effective


def score(ctx, choice, features):
    if choice.startswith('team '):
        return 0.0
    state = ctx['state']
    own = state.get('my_actives', [])
    enemies = [known_mon(m, state, features) for m in state.get('opp_actives', []) if hp_fraction(m.get('condition')) > 0]
    commands = [s.strip().split() for s in choice.split(',')]
    attacks, helpers, protected = {}, {}, set()
    prior = 0.0
    for i, parts in enumerate(commands):
        if parts[0] == 'move' and i < len(own):
            move = own[i].get('moves', [])[int(parts[1]) - 1]
            if to_id(move.get('id', move.get('move', ''))) in ('protect', 'detect'):
                protected.add(i)
    for i, parts in enumerate(commands):
        if parts[0] == 'switch':
            prior -= 0 if state.get('request_type') == 'forceSwitch' else .4
            continue
        if parts[0] != 'move' or i >= len(own):
            continue
        actor = mega_actor(own[i], parts, features)
        move = actor.get('moves', [])[int(parts[1]) - 1]
        name = to_id(move.get('id', move.get('move', '')))
        code = next((int(p) for p in parts[2:] if p.lstrip('+-').isdigit()), 0)
        targets = ([own[-code - 1]] if code < 0 and -code <= len(own) else
                   [m for m in enemies if m.get('slot', '').endswith(chr(96 + code))] if code > 0 else enemies)
        data = features.move_data(ctx, name, features.dex.get('moves', {}).get(name, {}), actor, targets)
        if name == 'helpinghand':
            if code < 0 and -code - 1 != i:
                helpers[-code - 1] = 1.5
            continue
        if name in ('protect', 'detect', 'wideguard'):
            repeated = any(e.get('side') == 'mine' and e.get('slot') == actor.get('slot', '')[-1:] and
                           to_id(e.get('move')) in ('protect', 'detect') and e.get('turn') == state.get('turn', 0) - 1
                           for e in state.get('history', []))
            prior += .05 if repeated else .15
            continue
        if name == 'fakeout' and not fake_out_available(ctx, actor):
            continue
        if charging(ctx, actor, move):
            prior += .15
            continue
        if code >= 0 and data.get('priority', 0) > 0 and any(
                to_id(m.get('ability') or m.get('baseAbility') or '') in ('armortail', 'queenlymajesty', 'dazzling') for m in enemies):
            continue
        power = data.get('basePower', 0) / 100
        accuracy = data.get('accuracy', 100)
        accuracy = 1 if accuracy is True else float(accuracy or 100) / 100
        kind = data.get('type', '')
        stab = 1.5 if kind in features.species(actor.get('species', '')).get('types', []) else 1
        kind_weather = weather(state, actor)
        scale = 1.0
        if kind_weather in ('raindance', 'primordialsea'):
            scale = 1.5 if kind == 'Water' else .5 if kind == 'Fire' else 1
        elif kind_weather in ('sunnyday', 'desolateland'):
            scale = 1.5 if kind == 'Fire' else .5 if kind == 'Water' else 1
        physical = data.get('category') == 'Physical'
        stat = 'atk' if physical else 'spa'
        stage = actor.get('boosts', {}).get(stat, 0)
        if name == 'electroshot' and move.get('target') not in (None, 'scripted'):
            stage = min(6, stage + 1)
        scale *= (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
        if physical and 'brn' in actor.get('condition', '').split() and to_id(actor.get('ability', '')) != 'guts':
            scale *= .5
        if actor is not own[i]:
            scale *= actor.get('stats', {}).get(stat, 1) / max(1, own[i].get('stats', {}).get(stat, 1))
        spread = data.get('target') in ('allAdjacent', 'allAdjacentFoes') and len(targets) > 1
        scale *= .75 if spread else 1
        rows = []
        for target in targets:
            if hp_fraction(target.get('condition')) <= 0:
                continue
            amount = power * accuracy * stab * scale * multiplier(kind, actor, target, features)
            friendly = code < 0
            if not friendly:
                effects = {to_id(e).removeprefix('move') for e in state.get('hazards', {}).get('theirs', [])}
                if 'auroraveil' in effects or ('reflect' if physical else 'lightscreen') in effects:
                    amount *= 2 / 3 if len(own) == 2 else .5
            if friendly and (-code - 1) in protected:
                amount = 0
            rows.append((amount, friendly))
        if data.get('target') == 'allAdjacent':
            for j, ally in enumerate(own):
                if i != j and j not in protected and hp_fraction(ally.get('condition')) > 0:
                    rows.append((power * accuracy * stab * scale * multiplier(kind, actor, ally, features), True))
        if code == 0 and data.get('target') not in ('allAdjacent', 'allAdjacentFoes') and len(rows) > 1:
            rows = [(sum(a for a, _ in rows) / len(rows), False)]
        if power:
            attacks[i] = rows
        if name == 'fakeout' and any(amount > 0 and not friendly for amount, friendly in rows):
            prior += .35
    for i, rows in attacks.items():
        prior += helpers.get(i, 1) * sum(-amount if friendly else amount for amount, friendly in rows)
    return prior
