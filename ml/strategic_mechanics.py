"""Opt-in strategic-v2 estimates from public mechanics and observed stats.

These remain level-50 expected-damage estimates, not hidden-set reconstruction.
The incumbent's encoder and tactical estimator deliberately remain unchanged.
"""
from battle_state import hp_fraction, to_id
from ml.tactics import damage, stat

SLEEP = {'spore', 'hypnosis', 'sleeppowder', 'sing', 'lovelykiss', 'grasswhistle', 'darkvoid'}


def accuracy(actor, target, data):
    value = data.get('accuracy', 100)
    if value is True or to_id(actor.get('ability')) == 'noguard' or to_id(target.get('ability')) == 'noguard':
        return 1.0
    stage = max(-6, min(6, actor.get('boosts', {}).get('accuracy', 0) - target.get('boosts', {}).get('evasion', 0)))
    factor = (3 + stage) / 3 if stage >= 0 else 3 / (3 - stage)
    ability = to_id(actor.get('ability') or actor.get('baseAbility'))
    if ability == 'compoundeyes':
        factor *= 1.3
    if to_id(actor.get('item')) == 'widelens':
        factor *= 1.1
    return min(1.0, max(0.0, float(value or 100) * factor / 100))


def move_data(name, data, actor, state, grounded):
    if data.get('_strategic_v2_resolved'):
        return data
    result = {**data, '_strategic_v2_resolved': True}
    if name in ('storedpower', 'powertrip'):
        result['basePower'] = 20 + 20 * sum(max(0, value) for value in actor.get('boosts', {}).values())
    field = {to_id(v).removeprefix('move') for v in state.get('field', [])}
    if name == 'expandingforce' and 'psychicterrain' in field and grounded:
        result.update(basePower=result.get('basePower', 80) * 1.5, target='allAdjacentFoes')
    return result


def estimate(actor, target, name, data, state, features, spread=False):
    """Correct exceptional offensive/defensive stats without changing category."""
    physical = data.get('category') == 'Physical'
    attack_key = 'atk' if physical else 'spa'
    defense_key = 'def' if physical else 'spd'
    source, source_key = (target, 'atk') if name == 'foulplay' else (actor, 'def' if name == 'bodypress' else attack_key)
    if name in ('bodypress', 'foulplay'):
        actor = {**actor, 'stats': {**actor.get('stats', {}), attack_key: stat(source, source_key, features)},
                 'boosts': {**actor.get('boosts', {}), attack_key: 0}}
    if name in ('psyshock', 'psystrike', 'secretsword'):
        target = {**target, 'stats': {**target.get('stats', {}), defense_key: stat(target, 'def', features)},
                  'boosts': {**target.get('boosts', {}), defense_key: 0}}
    if name in ('darkestlariat', 'chipaway'):
        target = {**target, 'boosts': {**target.get('boosts', {}), defense_key: 0}}
    data = {**data, 'accuracy': accuracy(actor, target, data) * 100}
    return damage(actor, target, data, state, features, spread)


def sleep_probability(actor, target, name, data, state, features):
    if name not in SLEEP or set(target.get('condition', '').split()[1:]) & {'par', 'brn', 'slp', 'frz', 'psn', 'tox'}:
        return 0.0
    ability = to_id(target.get('ability') or target.get('baseAbility'))
    powder = name in ('spore', 'sleeppowder')
    if ability in ('insomnia', 'vitalspirit', 'sweetveil', 'purifyingsalt', 'comatose'):
        return 0.0
    if powder and ('Grass' in features.species(target.get('species', '')).get('types', []) or
                   ability == 'overcoat' or to_id(target.get('item')) == 'safetygoggles'):
        return 0.0
    field = {to_id(v).removeprefix('move') for v in state.get('field', [])}
    grounded = ('Flying' not in features.species(target.get('species', '')).get('types', []) and
                ability != 'levitate' and to_id(target.get('item')) != 'airballoon')
    if grounded and field & {'electricterrain', 'mistyterrain'}:
        return 0.0
    if to_id(actor.get('ability')) == 'prankster' and 'Dark' in features.species(target.get('species', '')).get('types', []):
        return 0.0
    allies = state.get('my_actives', []) if target in state.get('my_actives', []) else state.get('opp_actives', [])
    if any(to_id(m.get('ability') or m.get('baseAbility')) == 'sweetveil' and hp_fraction(m.get('condition')) > 0 for m in allies):
        return 0.0
    return accuracy(actor, target, data)


def unproductive_solo_protect(ctx, ours, enemies):
    """A narrow tempo cost; recovery, residual and active timers preserve stalling."""
    party = ctx['state'].get('my_party', [])
    alive = [m for m in party if hp_fraction(m.get('condition')) > 0]
    if len(alive) != 1 or not any(c[2] in ('protect', 'detect') for c in ours):
        return 0.0
    if to_id(alive[0].get('item')) in ('leftovers', 'blacksludge'):
        return 0.0
    if ctx['state'].get('field') or any(ctx['state'].get('hazards', {}).values()):
        return 0.0  # Expiry can change initiative, targeting or damage next turn.
    if any(set(m.get('condition', '').split()[1:]) & {'brn', 'psn', 'tox'} for m in enemies):
        return 0.0
    if any(to_id(v).removeprefix('move') in ('leechseed', 'perishsong') for m in enemies for v in m.get('volatiles', [])):
        return 0.0
    return .45
