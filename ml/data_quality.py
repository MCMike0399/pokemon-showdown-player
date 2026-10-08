"""Training-data checks without relabeling, backfilling or changing consumption."""
from __future__ import annotations

import math

import numpy as np

from battle_state import legal_choices
from ml.features import ACTION_DIM, STATE_DIM, SCHEMA
from ml.recording import log_prefix


def trajectory_issues(episode: dict) -> list[str]:
    """Fail closed on malformed inputs or proposals that were never submitted.

Legacy encoded-only trajectories remain valid for their collecting profile;
missing historical snapshots are a coverage limit, not fabricated evidence.
"""
    issues = set()
    if episode.get('schema') != SCHEMA:
        issues.add('schema-mismatch')
    if episode.get('outcome') not in (-1, 0, 1):
        issues.add('invalid-outcome')
    policy = episode.get('collecting_policy')
    if policy and policy.get('feature_profile') != episode.get('feature_profile', 'legacy'):
        issues.add('collecting-profile-mismatch')
    if not episode.get('steps'):
        issues.add('empty-trajectory')
    ids = []
    previous_turn = -1
    for step in episode.get('steps', []):
        ids.append(step.get('request_id'))
        if step.get('submitted') is False:
            issues.add('unsubmitted-proposal')
        try:
            state = np.asarray(step['state'], dtype=np.float32)
            actions = np.asarray(step['actions'], dtype=np.float32)
            index = step['index']
            valid = (state.shape == (STATE_DIM,) and actions.ndim == 2 and actions.shape[1] == ACTION_DIM and
                     isinstance(index, int) and 0 <= index < len(actions))
            if not valid:
                issues.add('invalid-feature-shape-or-index')
            if not np.isfinite(state).all() or not np.isfinite(actions).all():
                issues.add('nonfinite-features')
            if not math.isfinite(step['value']) or not math.isfinite(step['logprob']) or step['logprob'] > 1e-6:
                issues.add('invalid-collecting-likelihood-or-value')
        except (KeyError, TypeError, ValueError, OverflowError):
            issues.add('malformed-encoded-step')
            continue
        snapshot = step.get('snapshot')
        if not snapshot:
            continue
        try:
            alternatives = snapshot['legal_choices']
            if len(alternatives) != len(actions) or snapshot['selected_index'] != index or alternatives[index]['choice'] != step['choice']:
                issues.add('snapshot-selection-mismatch')
            probabilities = [a['probability'] for a in alternatives]
            if (not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) or
                    not math.isclose(sum(probabilities), 1, abs_tol=1e-5) or
                    not math.isclose(probabilities[index], math.exp(step['logprob']), rel_tol=1e-5, abs_tol=1e-7)):
                issues.add('snapshot-likelihood-mismatch')
            if [a['choice'] for a in alternatives] != legal_choices(snapshot['request']):
                issues.add('request-mask-mismatch')
            prefix = log_prefix(episode, snapshot['public_log'])
            turns = [int(line.split('|')[2]) for line in prefix if line.startswith('|turn|')]
            turn = snapshot['turn']
            if turn < previous_turn or (turns and max(turns) > turn) or any(line.startswith('|win|') or line in ('|tie', '|tie|') for line in prefix):
                issues.add('future-or-out-of-order-observation')
            previous_turn = turn
            if snapshot.get('feature_profile', 'legacy') != episode.get('feature_profile', 'legacy'):
                issues.add('collecting-profile-mismatch')
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            issues.add('invalid-snapshot-or-log-prefix')
    if len(ids) != len(set(ids)):
        issues.add('duplicate-request-credit')
    if episode.get('terminal_log'):
        try:
            terminal = log_prefix(episode, episode['terminal_log'])
            if not any(line.startswith('|win|') or line in ('|tie', '|tie|') for line in terminal):
                issues.add('nonterminal-final-log')
        except (KeyError, ValueError, IndexError, TypeError):
            issues.add('invalid-terminal-log-prefix')
    return sorted(issues)
