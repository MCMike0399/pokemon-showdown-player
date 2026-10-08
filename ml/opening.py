"""Versioned matchup and prospective-form priors from choice-time observations.

Own forms use exact supplied sets. Opponent damage and speed are estimates from
visible species/moves/sheets; unknown spreads and commands are never reconstructed.
"""
from __future__ import annotations

import math
from battle_state import hp_fraction, to_id
from ml.mechanics import charging, known_mon
from ml.rain import fake_out_available, weather
from ml.tactics import damage, stat


def own_form(ctx, mon, features, evolve=False):
    if not evolve:
        return mon
    if not hasattr(features, '_opening_forms'):
        features._opening_forms = {(to_id(d.get('baseSpecies')), to_id(d.get('requiredItem'))): d
                                  for d in features.dex.get('pokedex', {}).values() if d.get('requiredItem')}
    form = features._opening_forms.get((to_id(mon.get('species')), to_id(mon.get('item'))))
    if form is None:
        return mon
    sets = [s for s in ctx.get('team_sets') or [] if to_id(s.get('species')) == to_id(mon.get('species'))]
    stats = dict(mon.get('stats', {}))
    exact = False
    if len(sets) == 1 and features.dex.get('natures') and features.dex.get('rules'):
        own_set = sets[0]
        nature = features.dex['natures'].get(to_id(own_set.get('nature', 'Serious')), {})
        rules = features.dex['rules']
        level = own_set.get('level', 50)
        for name, base in form['baseStats'].items():
            if name == 'hp':
                continue
            ev = (own_set.get('evs') or {}).get(name, 0)
            if rules['champions'] and not rules['level_clause_mod']:
                value = base + ev + 20
            elif rules['champions']:
                value = math.floor((2 * base + 31 + max(2 * ev - 1, 0)) * level / 100) + 5
            else:
                iv = (own_set.get('ivs') or {}).get(name, 31)
                value = math.floor((2 * base + iv + ev // 4) * level / 100) + 5
            factor = 110 if nature.get('plus') == name else 90 if nature.get('minus') == name else 100
            stats[name] = (value * factor) // 100
        exact = True
    # Missing own sets preserve the observed stats, rather than invent exact ones.
    return {**mon, 'species': form['name'], 'ability': form.get('abilities', {}).get('0', ''),
            'baseAbility': form.get('abilities', {}).get('0', ''), 'stats': stats,
            'prospective_stats_exact': exact}


def _moves(mon, features):
    names = mon.get('moves') or mon.get('moves_known') or []
    return [(to_id(m if isinstance(m, str) else m.get('id', m.get('move', ''))),
             m if isinstance(m, dict) else {}) for m in names]


def _hit(actor, target, name, data, ctx, features, spread=False):
    affected = features.move_data({**ctx, 'feature_profile': 'mechanics-v1'}, name, data, actor, [target])
    state = {**ctx['state'], 'weather': weather(ctx['state'], actor)}
    amount = damage(actor, target, affected, state, features, spread)
    if to_id(actor.get('ability')) == 'toughclaws' and data.get('flags', {}).get('contact'):
        amount *= 5325 / 4096
    return amount


def _threat(enemy, target, ctx, features):
    enemy = known_mon(enemy, ctx['state'], features)
    moves = [(name, features.dex.get('moves', {}).get(name, {})) for name, _ in _moves(enemy, features)]
    attacks = [(n, d) for n, d in moves if d.get('basePower') and d.get('category') in ('Physical', 'Special')]
    if not attacks:
        # Species STAB estimates are hypotheses, not discovered moves or spreads.
        category = 'Physical' if stat(enemy, 'atk', features) >= stat(enemy, 'spa', features) else 'Special'
        attacks = [('', {'type': t, 'category': category, 'basePower': 90, 'accuracy': 100})
                   for t in features.species(enemy.get('species', '')).get('types', [])]
    return max((_hit(enemy, target, n, d, ctx, features) for n, d in attacks), default=0)


def _best_attack(mon, enemy, ctx, features, immediate=False):
    scores = []
    for name, request_move in _moves(mon, features):
        data = features.dex.get('moves', {}).get(name, {})
        if not data.get('basePower'):
            continue
        scale = 1
        if name == 'electroshot' and weather(ctx['state'], mon) not in ('raindance', 'primordialsea'):
            scale = .15 if immediate else .45
        scores.append(scale * _hit(mon, enemy, name, data, ctx, features))
    return max(scores, default=0)


def preview_score(ctx, choice, features):
    cache = ctx.setdefault('_opening_preview', {})
    if not cache:
        state = ctx['state']
        party = state.get('my_party', [])
        enemies = [known_mon({'species': s, 'condition': '100/100'}, state, features)
                   for s in state.get('opp_preview', [])]
        variants = [(m, own_form(ctx, m, features, True)) for m in party]
        tables = {}
        for rain in (False, True):
            simulated = {**ctx, 'state': {**state, 'weather': 'RainDance' if rain else None}}
            tables[rain] = [[{
                'attack': [_best_attack(m, e, simulated, features) for e in enemies],
                'immediate': [_best_attack(m, e, simulated, features, True) for e in enemies],
                'threat': [_threat(e, m, simulated, features) for e in enemies],
                'speed': stat(m, 'spe', features),
            } for m in pair] for pair in variants]
        cache.update(party=party, enemies=enemies, variants=variants, tables=tables)
    party, enemies = cache['party'], cache['enemies']
    if not enemies:
        return 0.0
    order = [int(i) - 1 for i in choice[5:].split(',')]
    picked = [party[i] for i in order]
    setter = lambda m: to_id(m.get('ability') or m.get('baseAbility')) == 'drizzle'
    rain = any(setter(m) for m in picked)
    lead_rain = any(setter(m) for m in picked[:2])
    modes = [None] + [i for i in order if cache['variants'][i][1] is not party[i]]
    scores = []
    for mega in modes:
        rows = [cache['tables'][rain][i][int(i == mega)] for i in order]
        leads = [cache['tables'][lead_rain][i][int(i == mega)] for i in order[:2]]
        coverage = pressure = exposure = 0
        for j, enemy in enumerate(enemies):
            best = sorted((r['attack'][j] for r in rows), reverse=True)
            coverage += min(best[0], 1.5) + .2 * min(best[1], 1)
            pressure += sum(min(r['immediate'][j], 1) for r in leads)
            enemy_speed = stat(enemy, 'spe', features)
            exposure += sum(min(r['threat'][j], 1.5) * (1 if enemy_speed >= r['speed'] else .65) for r in leads)
        count = len(enemies)
        score = 1.4 * coverage / count + .9 * pressure / count - .75 * exposure / count
        names = [{n for n, _ in _moves(m, features)} for m in picked]
        dependent = sum('electroshot' in n or to_id(m.get('ability')) == 'swiftswim' for m, n in zip(picked, names))
        score += (.4 if rain else -.5) * dependent
        if lead_rain:
            score += .55 * sum(bool(n & {'electroshot', 'thunder', 'hurricane'}) for n in names[:2])
        elif rain and 'electroshot' in set().union(*names[:2]):
            score -= .3  # Benched weather needs an entry turn.
        if any(to_id(m.get('ability') or m.get('baseAbility')) == 'intimidate' for m in picked[:2]):
            physical = sum(stat(e, 'atk', features) >= stat(e, 'spa', features) for e in enemies) / count
            score += .25 * physical
        if any('fakeout' in n for n in names[:2]):
            score += .12
        scores.append(score)
    # Bench order is immaterial. One prospective Mega per plan, never both.
    return max(scores)


def turn_score(ctx, choice, features):
    state = ctx['state']
    own = state.get('my_actives', [])
    enemies = [known_mon(m, state, features) for m in state.get('opp_actives', []) if hp_fraction(m.get('condition')) > 0]
    commands = [p.strip().split() for p in choice.split(',')]
    actors = [own_form(ctx, m, features, i < len(commands) and any(x in commands[i] for x in ('mega', 'megax', 'megay')))
              for i, m in enumerate(own)]
    protected, helpers = set(), {}
    for i, parts in enumerate(commands):
        if parts[0] != 'move' or i >= len(actors):
            continue
        name = _moves(actors[i], features)[int(parts[1]) - 1][0]
        if name in ('protect', 'detect'):
            protected.add(i)
        if name == 'helpinghand':
            code = next((int(p) for p in parts[2:] if p.lstrip('+-').isdigit()), 0)
            if code < 0 and -code - 1 != i:
                helpers[-code - 1] = 1.5
    inflicted = {}; value = 0.0
    for i, parts in enumerate(commands):
        if parts[0] == 'pass' or i >= len(actors):
            continue
        if parts[0] == 'switch':
            if state.get('request_type') != 'forceSwitch':
                incoming = state.get('my_party', [])[int(parts[1]) - 1]
                value -= .12 + .35 * max((_threat(e, incoming, ctx, features) for e in enemies), default=0)
            continue
        if parts[0] != 'move':
            continue
        actor = actors[i]
        name, request_move = _moves(actor, features)[int(parts[1]) - 1]
        data = features.dex.get('moves', {}).get(name, {})
        threats = [_threat(e, actor, ctx, features) for e in enemies]
        danger = sum(sorted(threats, reverse=True)[:2])
        health = hp_fraction(actor.get('condition'))
        if i in protected:
            repeated = any(e.get('side') == 'mine' and e.get('slot') == actor.get('slot', '')[-1:]
                           and to_id(e.get('move')) in ('protect', 'detect') and e.get('turn') == state.get('turn', 0) - 1
                           for e in state.get('history', []))
            value += (.12 + .35 * min(danger, health)) * (.25 if repeated else 1)
            continue
        # Defensive cost considers the selected form, not its original typing.
        value -= .35 * min(danger, health + .3)
        code = next((int(p) for p in parts[2:] if p.lstrip('+-').isdigit()), 0)
        targets = [actors[-code - 1]] if code < 0 else [e for e in enemies if e.get('slot', '').endswith(chr(96 + code))] if code > 0 else enemies
        if name == 'helpinghand':
            continue
        if name == 'fakeout' and not fake_out_available(ctx, actor):
            continue
        if charging(ctx, actor, request_move):
            value += .08
            continue
        if data.get('priority', 0) > 0 and code >= 0 and any(to_id(e.get('ability')) in ('armortail', 'queenlymajesty', 'dazzling') for e in enemies):
            continue
        if name == 'electroshot' and request_move.get('target') not in (None, 'scripted'):
            actor = {**actor, 'boosts': {**actor.get('boosts', {}), 'spa': min(6, actor.get('boosts', {}).get('spa', 0) + 1)}}
        slow_threat = sum(t for e, t in zip(enemies, threats) if stat(e, 'spe', features) >= stat(actor, 'spe', features))
        trick_room = any(to_id(x).removeprefix('move') == 'trickroom' for x in state.get('field', []))
        if trick_room:
            slow_threat = sum(t for e, t in zip(enemies, threats) if stat(e, 'spe', features) <= stat(actor, 'spe', features))
        execution = max(.25, 1 - .4 * max(0, slow_threat - health)) if data.get('priority', 0) <= 0 else 1
        spread = data.get('target') in ('allAdjacent', 'allAdjacentFoes') and len(targets) > 1
        for target in targets:
            hit = _hit(actor, target, name, data, ctx, features, spread) * helpers.get(i, 1) * execution
            if code < 0:
                if -code - 1 not in protected:
                    value -= hit
            else:
                key = target.get('slot', target.get('species'))
                inflicted[key] = inflicted.get(key, 0) + hit
        if data.get('target') == 'allAdjacent':
            for j, ally in enumerate(actors):
                if j != i and j not in protected:
                    value -= _hit(actor, ally, name, data, ctx, features, True)
        if name == 'fakeout' and code > 0 and any(_hit(actor, t, name, data, ctx, features) > 0 for t in targets):
            value += .25
        if any(x in parts for x in ('mega', 'megax', 'megay')):
            # Small option cost for consuming the shared Mega, not a mandate.
            value -= .04
    for enemy in enemies:
        hit = inflicted.get(enemy.get('slot', enemy.get('species')), 0)
        hp = hp_fraction(enemy.get('condition'))
        value += min(hit, hp) + .25 * min(1, hit / max(.01, hp))
    return 2.5 * value
