"""Experimental rain/joint-support prior; never enabled on a legacy checkpoint.

Observable mechanics inform a bounded pressure estimate, not an exact damage or
opponent-action prediction. Unknown switches, protection and speed are uncertain.
"""
from __future__ import annotations

from battle_state import hp_fraction, to_id
from ml.tactics import damage


def weather(state, actor):
    active = state.get('my_actives', []) + state.get('opp_actives', [])
    if any(hp_fraction(m.get('condition')) > 0 and
           to_id(m.get('ability') or m.get('baseAbility') or '') in ('airlock', 'cloudnine') for m in active):
        return ''
    kind = to_id(state.get('weather') or '')
    if kind in ('raindance', 'primordialsea', 'sunnyday', 'desolateland') and to_id(actor.get('item', '')) == 'utilityumbrella':
        return ''
    return kind


def mega_actor(actor, parts, features):
    """Use the selected form's typing/ability; stats are labelled approximations."""
    if not any(p in ('mega', 'megax', 'megay') for p in parts):
        return actor
    if not hasattr(features, '_mega_forms'):
        features._mega_forms = {(to_id(d.get('baseSpecies', '')), to_id(d.get('requiredItem', ''))): d
                                for d in features.dex.get('pokedex', {}).values() if d.get('requiredItem')}
    form = features._mega_forms.get((to_id(actor.get('species', '')), to_id(actor.get('item', ''))))
    if not form:
        return actor
    original = features.species(actor.get('species', '')).get('baseStats', {})
    stats = {name: value + form['baseStats'].get(name, 0) - original.get(name, 0)
             for name, value in actor.get('stats', {}).items()}
    return {**actor, 'species': form['name'], 'stats': stats, 'ability': form.get('abilities', {}).get('0', '')}


def fake_out_available(ctx, actor):
    """A full observed entry prefix establishes eligibility; absent logs stay unknown."""
    entered = None
    turn = 0
    for line in ctx.get('public_log', []):
        parts = line.split('|')
        if len(parts) < 3:
            continue
        if parts[1] == 'turn':
            turn = int(parts[2])
        elif parts[1] in ('switch', 'drag') and parts[2].split(':')[0] == actor.get('slot'):
            entered = turn
    return entered is None or ctx['state'].get('turn', 0) <= max(1, entered + 1)


def score(ctx, choice, features):
    # Isolate turn policy changes; previous preview experiments did not pass.
    if choice.startswith('team '):
        return 0.0
    state = ctx['state']
    own = state.get('my_actives', [])
    enemies = [m for m in state.get('opp_actives', []) if hp_fraction(m.get('condition')) > 0]
    parts = [p.strip().split() for p in choice.split(',')]
    hits, support, protected = {}, 0.0, set()
    attacks = {}
    for i, command in enumerate(parts):
        if command[0] != 'move' or i >= len(own):
            continue
        actor = mega_actor(own[i], command, features)
        move = actor.get('moves', [])[int(command[1]) - 1]
        name = to_id(move.get('id', move.get('move', '')))
        code = next((int(p) for p in command[2:] if p.lstrip('+-').isdigit()), 0)
        targets = ([own[-code - 1]] if code < 0 and -code <= len(own) else
                   [m for m in enemies if m.get('slot', '').endswith(chr(96 + code))] if code > 0 else enemies)
        data = features.move_data(ctx, name, features.dex.get('moves', {}).get(name, {}), actor, targets)
        if name in ('protect', 'detect', 'wideguard'):
            protected.add(i)
            support += .15
            continue
        if name == 'fakeout' and not fake_out_available(ctx, actor):
            continue
        if data.get('priority', 0) > 0 and code >= 0 and any(
                to_id(m.get('ability') or m.get('baseAbility') or '') in ('armortail', 'queenlymajesty', 'dazzling') for m in enemies):
            continue
        # Charge-phase Electro Shot boosts SpA but does not hit this turn.
        charging = name == 'electroshot' and weather(state, actor) not in ('raindance', 'primordialsea') and not (
            {to_id(v) for v in actor.get('volatiles', [])} & {'electroshot', 'twoturnmove'}) and to_id(actor.get('item', '')) != 'powerherb'
        if charging:
            support += .10
            continue
        spread = data.get('target') in ('allAdjacent', 'allAdjacentFoes') and len(targets) > 1
        if data.get('basePower'):
            affected_state = {**state, 'weather': weather(state, actor)}
            # Electro Shot's observable pre-hit +1 is part of the same attack.
            if name == 'electroshot' and not ({to_id(v) for v in actor.get('volatiles', [])} & {'electroshot', 'twoturnmove'}):
                actor = {**actor, 'boosts': {**actor.get('boosts', {}), 'spa': min(6, actor.get('boosts', {}).get('spa', 0) + 1)}}
            attacks[i] = [(target, damage(actor, target, data, affected_state, features, spread), code < 0)
                          for target in targets]
            if data.get('target') == 'allAdjacent':
                attacks[i].extend((ally, damage(actor, ally, data, affected_state, features, True), True)
                                  for j, ally in enumerate(own) if j != i)
        if name == 'fakeout' and code > 0 and targets and any(features.effectiveness('Normal', t) for t in targets):
            support += .25
    # Helping Hand only benefits the joint partner's actual damaging command.
    multipliers = {}
    for i, command in enumerate(parts):
        if command[0] != 'move' or i >= len(own):
            continue
        move = own[i]['moves'][int(command[1]) - 1]
        if to_id(move.get('id', move.get('move', ''))) == 'helpinghand':
            code = next((int(p) for p in command[2:] if p.lstrip('+-').isdigit()), 0)
            if code < 0 and -code - 1 != i and -code - 1 in attacks:
                multipliers[-code - 1] = 1.5
    penalty = .12 * sum(p[0] == 'switch' for p in parts)
    for i, rows in attacks.items():
        for target, amount, friendly in rows:
            amount *= multipliers.get(i, 1.0)
            if friendly:
                if not any(j in protected and target is own[j] for j in range(len(own))):
                    penalty += amount
            else:
                key = target.get('slot', target.get('species'))
                hits[key] = hits.get(key, 0) + amount
    value = sum(min(hits.get(m.get('slot', m.get('species')), 0), hp_fraction(m.get('condition'))) +
                .3 * min(1, hits.get(m.get('slot', m.get('species')), 0) / max(.01, hp_fraction(m.get('condition')))) for m in enemies)
    return 2.5 * (value - penalty) + support
