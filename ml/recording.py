"""Readable private decision snapshots, separate from immutable PPO feature inputs."""
from __future__ import annotations

import copy
import hashlib
import json

from ml.scout import PUBLIC_KINDS
from ml.storage import now

SNAPSHOT_VERSION = 1
RECORDER_VERSION = 3


def encoded_input(array):
    """Compact dense JSON without losing a single nonzero float32 value.

Zero-heavy feature arrays dominate episode size. Integer zero round-trips to
the same tensor; the public list-shaped schema and old readers stay usable.
"""
    values = array.tolist()
    if array.ndim == 1:
        return [0 if value == 0 else value for value in values]
    return [[0 if value == 0 else value for value in row] for row in values]
# Additional observable battle events useful for diagnosis, beyond scout labels.
BATTLE_KINDS = PUBLIC_KINDS | {
    'teampreview', 'teamsize', 'upkeep', 'cant', 'swap', 'callback',
    '-immune', '-resisted', '-supereffective', '-crit', '-miss', '-fail', '-notarget',
    '-activate', '-block', '-singleturn', '-singlemove', '-clearpositiveboost',
    '-clearnegativeboost', '-invertboost', '-copyboost', '-swapboost', '-cureteam',
    '-prepare', '-anim', '-mustrecharge', '-mega', '-primal', '-burst', '-transform',
    '-combine', '-waiting', '-nothing', '-hitcount', '-message', '-hint',
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def battle_lines(raw: list[str]) -> list[str]:
    """Battle events only: requests, chat, HTML and authentication stay excluded."""
    result = []
    for line in raw:
        parts = line.split('|')
        if len(parts) < 2 or parts[1] not in BATTLE_KINDS:
            continue
        if parts[1] == 'player':
            line = f'|player|{parts[2]}|Player-{parts[2]}'
        elif parts[1] == 'win':
            line = '|win|redacted'
        result.append(line)
    return result


def record_log(episode: dict, raw: list[str]) -> dict:
    """Share growing prefixes; reconnect differences get their own exact segment."""
    lines = battle_lines(raw)
    segments = episode.setdefault('observation_logs', [])
    for index, segment in enumerate(segments):
        common = min(len(segment), len(lines))
        if segment[:common] == lines[:common]:
            if len(lines) > len(segment):
                segment.extend(lines[len(segment):])
            return {'segment': index, 'length': len(lines), 'sha256': digest(lines)}
    segments.append(lines)
    return {'segment': len(segments) - 1, 'length': len(lines), 'sha256': digest(lines)}


def log_prefix(episode: dict, reference: dict) -> list[str]:
    segment = episode['observation_logs'][reference['segment']]
    if not 0 <= reference['length'] <= len(segment):
        raise ValueError('recorded log prefix length is invalid')
    lines = segment[:reference['length']]
    if digest(lines) != reference['sha256']:
        raise ValueError('recorded log prefix digest changed')
    return list(lines)


def named_choice(ctx: dict, choice: str) -> dict:
    """Resolve command slots from the exact request/observation, never future data."""
    party = ctx['request'].get('side', {}).get('pokemon', [])
    def species(index):
        return party[index].get('details', '').split(',')[0] if 0 <= index < len(party) else None
    if choice.startswith('team '):
        order = [int(value) for value in choice[5:].split(',')]
        return {'choice': choice, 'kind': 'team', 'order': order,
                'pokemon': [species(index - 1) for index in order]}
    components = []
    active = ctx['request'].get('active', [])
    state = ctx['state']
    for slot, text in enumerate(choice.split(',')):
        parts = text.strip().split()
        component = {'slot': slot + 1, 'kind': parts[0], 'command': text.strip()}
        if parts[0] == 'switch':
            index = int(parts[1])
            component.update(party_index=index, pokemon=species(index - 1))
        elif parts[0] == 'move':
            index = int(parts[1])
            moves = active[slot].get('moves', []) if slot < len(active) else []
            move = moves[index - 1] if 0 < index <= len(moves) else {}
            component.update(move_index=index, move=move.get('move', move.get('id')),
                             target_type=move.get('target'),
                             actor=state.get('my_actives', [])[slot].get('species') if slot < len(state.get('my_actives', [])) else None,
                             modifiers=[part for part in parts[2:] if not part.lstrip('+-').isdigit()])
            target = next((int(part) for part in parts[2:] if part.lstrip('+-').isdigit()), None)
            if target is not None:
                component['target_slot'] = target
                if target > 0:
                    mon = next((m for m in state.get('opp_actives', []) if m.get('slot', '').endswith(chr(96 + target))), None)
                    component['target_pokemon'] = mon.get('species') if mon else None
                    component['target_side'] = 'opponent'
                else:
                    own = state.get('my_actives', [])
                    component['target_pokemon'] = own[-target - 1].get('species') if 0 < -target <= len(own) else None
                    component['target_side'] = 'own'
        components.append(component)
    return {'choice': choice, 'kind': 'joint', 'components': components}


def snapshot(episode: dict, ctx: dict, prediction: dict, knowledge: dict, scout: list, temperature: float) -> dict:
    alternatives = [named_choice(ctx, choice) for choice in ctx['choices']]
    for alternative, probability in zip(alternatives, prediction['probabilities']):
        alternative['probability'] = probability
    return {'version': SNAPSHOT_VERSION, 'provenance': 'captured-at-decision', 'captured_at': now(),
            'turn': ctx['state'].get('turn', ctx.get('turn')),
            'request': copy.deepcopy(ctx['request']), 'observation': copy.deepcopy(ctx['state']),
            'legal_choices': alternatives, 'selected_index': prediction['index'],
            'sampling_temperature': temperature, 'knowledge': copy.deepcopy(knowledge),
            'feature_profile': ctx.get('feature_profile', 'legacy'),
            'opponent_predictions': copy.deepcopy(scout),
            'public_log': record_log(episode, ctx.get('public_log', []))}


def read_recording(store, room: str, start: int = 0, limit: int = 3, alternatives: bool = False) -> dict:
    """Readable exact snapshots for local inspection; legacy gaps stay unknown."""
    if start < 0 or not 1 <= limit <= 20:
        raise ValueError('start must be nonnegative and limit must be 1..20')
    rows = store.room_episodes(room)
    complete = [row for row in rows if row['status'] == 'complete']
    pending = [row for row in rows if row['status'] == 'pending']
    candidates = complete or pending
    if not candidates:
        return {'found': False, 'room': room}
    if len(candidates) != 1:
        raise ValueError('ambiguous recording; inspect episode IDs before choosing a perspective')
    episode = candidates[0]
    steps = []
    for index, step in enumerate(episode['steps'][start:start + limit], start):
        captured = copy.deepcopy(step.get('snapshot'))
        if captured:
            captured['public_log'] = log_prefix(episode, captured['public_log'])
            if not alternatives:
                choices = captured['legal_choices']
                captured['selected'] = choices[captured['selected_index']]
                captured['top_choices'] = sorted(choices, key=lambda a: a['probability'], reverse=True)[:5]
                captured['legal_choices_count'] = len(choices)
                del captured['legal_choices']
        steps.append({'decision_index': index, 'request_id': step['request_id'], 'choice': step['choice'],
                      'logprob': step['logprob'], 'value': step['value'], 'submitted': step.get('submitted'),
                      'snapshot': captured, 'provenance': 'captured-at-decision' if captured else 'legacy-encoded-only'})
    return {'found': True, 'room': room, 'episode_id': episode['id'], 'status': episode['status'],
            'side': episode['side'], 'format': episode['format'], 'team_fingerprint': episode['team'],
            'revision': episode['revision'], 'outcome': episode.get('outcome'), 'decision_count': len(episode['steps']),
            'discarded_proposal_count': len(episode.get('discarded_proposals', [])),
            'trajectory_fragmented': len(rows) > 1, 'steps': steps}


def read_archive(root, room: str, start: int = 0, limit: int = 3, alternatives: bool = False) -> dict:
    """Open only a read-only database; never initialize a brain or observer."""
    import sqlite3
    from pathlib import Path
    from types import SimpleNamespace
    db = sqlite3.connect((Path(root) / 'experience.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
    def episodes(name):
        return [json.loads(row[0]) for row in db.execute("SELECT data FROM episodes WHERE json_extract(data,'$.room')=? ORDER BY created", (name,))]
    try:
        return read_recording(SimpleNamespace(room_episodes=episodes), room, start, limit, alternatives)
    finally:
        db.close()
