"""Bounded observable-state tactical prior; hidden spreads use labelled estimates."""
from __future__ import annotations

from battle_state import hp_fraction, to_id


def stat(mon, name, features):
    base = mon.get('stats', {}).get(name)
    if base is None:
        # Neutral level-50 approximation, not a reconstructed opponent stat.
        base = features.species(mon.get('species', '')).get('baseStats', {}).get(name, 80) + 35
    stage = max(-6, min(6, mon.get('boosts', {}).get(name, 0)))
    return max(1, base * ((2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)))


def damage(actor, target, data, state, features, spread=False):
    if data.get('category') not in ('Physical', 'Special') or not data.get('basePower'):
        return 0.0
    physical = data['category'] == 'Physical'
    attack = stat(actor, 'atk' if physical else 'spa', features)
    defense = stat(target, 'def' if physical else 'spd', features)
    kind = data.get('type', '')
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
    stab = 1.5 if kind in features.species(actor.get('species', '')).get('types', []) else 1
    burn = .5 if physical and 'brn' in actor.get('condition', '').split() and to_id(actor.get('ability', '')) != 'guts' else 1
    weather = to_id(state.get('weather') or '')
    if any(to_id(m.get('ability') or m.get('baseAbility') or '') in ('cloudnine', 'airlock') and hp_fraction(m.get('condition')) > 0
           for m in state.get('my_actives', []) + state.get('opp_actives', [])):
        weather = ''
    weather_scale = 1
    if weather in ('raindance', 'primordialsea'):
        weather_scale = 1.5 if kind == 'Water' else .5 if kind == 'Fire' else 1
    elif weather in ('sunnyday', 'desolateland'):
        weather_scale = 1.5 if kind == 'Fire' else .5 if kind == 'Water' else 1
    max_hp = features.species(target.get('species', '')).get('baseStats', {}).get('hp', 80) + 90
    # Our own party supplies an exact max HP; public opponents only reveal ratios.
    if target.get('stats'):
        try:
            max_hp = float(target['condition'].split()[0].split('/')[1].rstrip('gry'))
        except (ValueError, KeyError, IndexError):
            pass
    accuracy = data.get('accuracy', 100)
    accuracy = 1 if accuracy is True else float(accuracy or 100) / 100
    return ((22 * data['basePower'] * attack / defense / 50 + 2) * stab * effective * burn * weather_scale *
            (.75 if spread else 1) * .925 * accuracy) / max(1, max_hp)


def score(ctx, choice, features):
    state = ctx['state']
    if choice.startswith('team '):
        # Preserve the existing preview behavior in this tactical experiment.
        return 0.0
    own = state.get('my_actives', [])
    enemies = [m for m in state.get('opp_actives', []) if hp_fraction(m.get('condition')) > 0]
    components = [text.strip().split() for text in choice.split(',')]
    protected = set()
    for i, parts in enumerate(components):
        if parts[0] == 'move' and i < len(own):
            move = own[i].get('moves', [])[int(parts[1]) - 1]
            if to_id(move.get('id', move.get('move', ''))) in ('protect', 'detect'):
                protected.add(i)
    inflicted = {}
    penalty, support = 0.0, 0.0
    for i, parts in enumerate(components):
        if parts[0] == 'switch':
            penalty += .12
            continue
        if parts[0] != 'move' or i >= len(own):
            continue
        actor = own[i]
        move = actor.get('moves', [])[int(parts[1]) - 1]
        name = to_id(move.get('id', move.get('move', '')))
        target_code = next((int(p) for p in parts[2:] if p.lstrip('+-').isdigit()), 0)
        targets = [m for m in enemies if m.get('slot', '').endswith(chr(96 + target_code))] if target_code > 0 else enemies
        data = features.move_data(ctx, name, features.dex.get('moves', {}).get(name, {}), actor, targets)
        if name in ('protect', 'detect'):
            repeated = any(event.get('side') == 'mine' and event.get('slot') == actor.get('slot', '')[-1:] and
                           to_id(event.get('move')) in ('protect', 'detect') and event.get('turn') == state.get('turn', 0) - 1
                           for event in state.get('history', []))
            threat = sum(damage(foe, actor, {'category': 'Physical' if stat(foe,'atk',features) >= stat(foe,'spa',features) else 'Special',
                                               'basePower':90,'accuracy':100,'type':(features.species(foe.get('species','')).get('types') or ['Normal'])[0]},
                                state, features) for foe in enemies)
            support += min(.7, threat * .5) * (.33 if repeated else 1)
            continue
        if target_code < 0:
            target_index = -target_code - 1
            if target_index < len(own):
                penalty += damage(actor, own[target_index], data, state, features)
            continue
        spread = data.get('target') in ('allAdjacent', 'allAdjacentFoes') and len(targets) > 1
        for target in targets:
            key = target.get('slot', target.get('species'))
            inflicted[key] = inflicted.get(key, 0) + damage(actor, target, data, state, features, spread)
        if data.get('target') == 'allAdjacent':
            for j, ally in enumerate(own):
                if j != i and j not in protected:
                    penalty += damage(actor, ally, data, state, features, True)
    value = 0.0
    for target in enemies:
        hit = inflicted.get(target.get('slot', target.get('species')), 0)
        health = hp_fraction(target.get('condition'))
        # Value a KO but stop rewarding additional damage after that target dies.
        value += min(hit, health) + .3 * min(1, hit / max(.01, health))
    return 2.5 * (value + support - penalty)
