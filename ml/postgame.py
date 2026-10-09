"""Fact-based post-match feedback and reusable opponent behavior evidence.

Observed outcomes describe what happened, not whether an unplayed choice wins.
Only executed opponent moves become statistical labels. No legacy trajectories
are relabeled for PPO, and no winning action is invented as an expert target.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from battle_state import hp_fraction, to_id
from ml.recording import log_prefix


def review(episode, log):
    own = episode.get('side', 'p1')
    turn = 0
    executed = Counter()
    failures = Counter()
    weather_changes = 0
    deaths = []
    identities = {}
    enemy_moves = defaultdict(Counter)
    enemy_leads = []
    for line in log:
        p = line.split('|')
        if len(p) < 3:
            continue
        if p[1] == 'turn':
            turn = int(p[2])
        elif p[1] in ('switch', 'drag', 'replace') and len(p) > 4:
            identities[p[2].split(':')[0]] = p[3].split(',')[0]
            if turn == 0 and not p[2].startswith(own):
                enemy_leads.append(to_id(p[3].split(',')[0]))
        elif p[1] in ('detailschange', '-formechange') and len(p) > 3:
            identities[p[2].split(':')[0]] = p[3].split(',')[0]
        elif p[1] == 'move' and len(p) > 3 and '[from]' not in line:
            name = to_id(p[3])
            if p[2].startswith(own):
                executed[(turn, p[2].split(':')[0][-1:], name)] += 1
            else:
                species = to_id(identities.get(p[2].split(':')[0], ''))
                if species:
                    enemy_moves[species][name] += 1
        elif p[1] == 'faint' and p[2].startswith(own):
            deaths.append({'turn': turn, 'species': identities.get(p[2].split(':')[0], 'unknown')})
        elif p[1] == '-weather' and len(p) > 2:
            weather_changes += '[upkeep]' not in line
        elif p[1] in ('cant', '-fail', '-immune', '-miss', '-block') and p[2].startswith(own):
            failures[p[1]] += 1
    chosen = []
    denied = []
    dry_charges = 0
    for step in episode.get('steps', []):
        snap = step.get('snapshot')
        if not snap or step.get('submitted') is False or step['choice'].startswith('team '):
            continue
        observation = snap['observation']
        decision_turn = observation.get('turn', 0)
        for i, component in enumerate(step['choice'].split(',')):
            parts = component.strip().split()
            if parts[0] != 'move' or i >= len(observation.get('my_actives', [])):
                continue
            actor = observation['my_actives'][i]
            moves = actor.get('moves', [])
            move = moves[int(parts[1]) - 1]
            name = to_id(move.get('id', move.get('move', '')))
            key = (decision_turn, actor.get('slot', '')[-1:], name)
            chosen.append(key)
            if not executed[key]:
                denied.append({'turn': decision_turn, 'species': actor.get('species'), 'move': name})
            if name == 'electroshot' and move.get('target') not in (None, 'scripted') and to_id(observation.get('weather')) not in ('raindance', 'primordialsea'):
                dry_charges += 1
    return {'version': 1, 'outcome': episode.get('outcome'),
            'captured_move_components': len(chosen), 'components_without_matching_execution': denied,
            'own_faints': deaths, 'weather_changes': weather_changes,
            'dry_electro_shot_selections': dry_charges, 'observed_failure_events': dict(failures),
            'opponent_leads': enemy_leads, 'opponent_executed_moves': {s: dict(c) for s, c in enemy_moves.items()},
            'limitations': ['Missing execution includes KO, flinch, status, a terminal earlier action and ambiguous locked releases.',
                            'Dry selections can be followed by a partner weather switch; no causal blame assigned.',
                            'Executed frequencies estimate behavior, not optimal choices or hidden intent.']}


def knowledge_from_reviews(reviews, fmt):
    counts = defaultdict(Counter)
    leads = Counter()
    for item in reviews:
        for species, moves in item.get('opponent_executed_moves', {}).items():
            counts[species].update(moves)
        leads.update(item.get('opponent_leads', []))
    return {'version': 1, 'format': fmt, 'games': len(reviews),
            'executed_samples': sum(sum(c.values()) for c in counts.values()),
            'moves': {s: dict(c) for s, c in counts.items()}, 'leads': dict(leads)}


def archived_log(episode, db):
    if episode.get('terminal_log'):
        return log_prefix(episode, episode['terminal_log'])
    row = db.execute('SELECT log FROM public_battles WHERE id=?', ('own-' + episode['room'],)).fetchone()
    return __import__('json').loads(row[0]) if row else []


def refresh_knowledge(store, model):
    """Fold new terminal ladder feedback into an evaluated candidate only.

    Immutable checkpoint evidence preserves reproducibility. The playing actor
    is never edited here; this knowledge update shares its candidate's gate.
    """
    import copy
    import json
    current = copy.deepcopy(model.strategy_knowledge)
    seen = set(current.get('learned_episode_ids', []))
    moves = {s: Counter(c) for s, c in current.get('moves', {}).items()}
    leads = Counter(current.get('leads', {}))
    added = 0
    rows = store.db.execute("""SELECT id,json_extract(data,'$.postgame') FROM episodes
        WHERE format=? AND source='ladder' AND status='complete'
        AND json_extract(data,'$.postgame.version')=1""", (model.fmt,))
    for episode_id, raw in rows:
        if episode_id in seen:
            continue
        item = json.loads(raw)
        if item.get('error'):
            continue
        for species, counts in item.get('opponent_executed_moves', {}).items():
            moves.setdefault(species, Counter()).update(counts)
        leads.update(item.get('opponent_leads', []))
        seen.add(episode_id)
        added += 1
    current.update(moves={s: dict(c) for s, c in moves.items()}, leads=dict(leads),
                   games=current.get('games', 0) + added, learned_episode_ids=sorted(seen),
                   executed_samples=sum(sum(c.values()) for c in moves.values()))
    model.strategy_knowledge = current
    return {'new_ladder_games': added, 'total_games': current['games'], 'executed_samples': current['executed_samples']}
