"""Read-only battle-pattern audit; inferred historical context is never PPO data."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def median(values):
    return statistics.median(values) if values else None


def audit(campaign: Path, root: Path, team_store=None):
    import numpy as np
    from harness import TeamStore
    from ml.model import Model
    from ml.recording import log_prefix
    from ml.storage import now
    from ml.teams import team_id
    manifest = json.loads((campaign / 'manifest.json').read_text())
    fmt = manifest['format']
    checkpoint = root / 'models' / (fmt + '.pt')
    if not checkpoint.exists():
        raise ValueError('audit requires an existing checkpoint')
    model = Model(root, fmt)
    teams = team_store or TeamStore()
    available = {}
    for name in teams.list():
        team = teams.get(name)
        if team['format'] == fmt:
            available[team_id(fmt, team['sets'])] = team['sets']
    records = [json.loads(line) for line in (campaign / 'games.jsonl').read_text().splitlines() if line.strip()]
    if len({r['room'] for r in records}) != len(records):
        raise ValueError('duplicate terminal campaign rooms')
    db = sqlite3.connect((root / 'experience.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
    strata = {}
    lineups = {}
    previews, probabilities, prior_gaps = [], [], []
    examples, excluded = [], []
    score = Counter()
    coverage = Counter()
    patterns = Counter()
    def log_for(episode):
        if episode.get('terminal_log'):
            return log_prefix(episode, episode['terminal_log']), 'captured-terminal'
        path = root / 'live-watch' / (episode['room'] + '.json')
        if path.exists():
            return json.loads(path.read_text())['log'], 'player-relay'
        row = db.execute('SELECT log FROM public_battles WHERE id=?', ('own-' + episode['room'],)).fetchone()
        return (json.loads(row[0]), 'scout-filtered') if row else ([], 'unavailable')
    try:
        for ordinal, record in enumerate(records, 1):
            episodes = [json.loads(row[0]) for row in db.execute("SELECT data FROM episodes WHERE json_extract(data,'$.room')=?", (record['room'],))]
            complete = [episode for episode in episodes if episode['status'] == 'complete']
            if len(complete) != 1:
                raise ValueError('terminal room must have exactly one complete episode')
            episode = complete[0]
            outcome = episode['outcome']
            score[outcome] += 1
            key = episode['team'] + '/' + episode['revision']
            group = strata.setdefault(key, {'team_fingerprint': episode['team'], 'revision': episode['revision'],
                                           'games': 0, 'wins': 0, 'losses': 0, 'ties': 0, 'decision_steps': 0})
            group['games'] += 1
            group['wins'] += outcome == 1
            group['losses'] += outcome == -1
            group['ties'] += outcome == 0
            if len(episodes) != 1:
                excluded.append({'game': ordinal, 'reason': 'fragmented recording; outcome retained, decision analysis excluded'})
                continue
            log, log_source = log_for(episode)
            coverage[log_source] += 1
            own = episode['side']
            turn = 0
            previous_protect = {}
            protect_examples = []
            for line in log:
                parts = line.split('|')
                if len(parts) < 3:
                    continue
                if parts[1] == 'turn':
                    turn = int(parts[2])
                elif parts[1] == 'move' and len(parts) >= 4 and parts[2].startswith(own) and '[from]' not in line:
                    name = parts[3].lower().replace(' ', '')
                    patterns['own_executed_moves'] += 1
                    if name in ('protect', 'detect'):
                        patterns['own_protect_moves'] += 1
                        if previous_protect.get(parts[2]) == turn - 1:
                            patterns['consecutive_protect_moves'] += 1
                            protect_examples.append({'turn': turn, 'event': line})
                        previous_protect[parts[2]] = turn
            for step in episode['steps']:
                group['decision_steps'] += 1
                coverage['captured_decisions' if step.get('snapshot') else 'legacy_encoded_decisions'] += 1
                probability = math.exp(step['logprob'])
                probabilities.append(probability)
                actions = np.asarray(step['actions'], dtype=np.float32)
                prior_gaps.append(float(actions[:, -1].max() - actions[step['index'], -1]))
                patterns['switch_components'] += sum(c.strip().startswith('switch ') for c in step['choice'].split(','))
                if not step['choice'].startswith('team '):
                    continue
                indices = [int(value) for value in step['choice'][5:].split(',')]
                sets = available.get(episode['team'])
                species = [sets[i - 1]['species'] for i in indices] if sets and all(0 < i <= len(sets) for i in indices) else None
                lineup_key = key + '/bring-' + ','.join(map(str, sorted(indices))) + '/lead-' + ','.join(map(str, indices[:2]))
                lineup = lineups.setdefault(lineup_key, {'team_fingerprint': episode['team'], 'revision': episode['revision'],
                                                        'order_indices': indices, 'species_from_verified_team': species,
                                                        'games': 0, 'wins': 0, 'losses': 0})
                lineup['games'] += 1
                lineup['wins'] += outcome == 1
                lineup['losses'] += outcome == -1
                entry = {'game': ordinal, 'team_fingerprint': episode['team'], 'revision': episode['revision'],
                         'choice': step['choice'], 'chosen_probability': probability, 'outcome': outcome,
                         'species_from_verified_team': species, 'exact_private_snapshot': bool(step.get('snapshot'))}
                if episode['revision'] == model.revision:
                    pred = model.predict(np.asarray(step['state'], dtype=np.float32), actions, preview=True)
                    distribution = pred['probabilities']
                    entropy = -sum(p * math.log(p) for p in distribution if p > 0)
                    entry.update(max_probability=max(distribution), legal_actions=len(distribution),
                                 normalized_entropy=entropy / math.log(len(distribution)) if len(distribution) > 1 else 0)
                previews.append(entry)
            if protect_examples and len(examples) < 5:
                examples.append({'game': ordinal, 'room': episode['room'], 'team_fingerprint': episode['team'],
                                 'revision': episode['revision'], 'outcome': outcome,
                                 'provenance': 'observed-public-events', 'events': protect_examples,
                                 'interpretation': 'Review in context; stalling Perish Song or field effects can be intentional.'})
        entropies = [p['normalized_entropy'] for p in previews if 'normalized_entropy' in p]
        summary = {'games': len(records), 'wins': score[1], 'losses': score[-1], 'ties': score[0],
                   'usable_decision_episodes': len(records) - len(excluded),
                   'decision_steps': len(probabilities), 'chosen_probability_median': median(probabilities),
                   'chosen_probability_under_5_percent': sum(p < .05 for p in probabilities),
                   'tactical_prior_gap_over_one': sum(g > 1 for g in prior_gaps),
                   'preview_chosen_probability_median': median([p['chosen_probability'] for p in previews]),
                   'preview_normalized_entropy_median': median(entropies),
                   'preview_max_probability_median': median([p['max_probability'] for p in previews if 'max_probability' in p])}
        hypotheses = [
            {'rank': 1, 'hypothesis': 'Team-preview sampling is too diffuse.',
             'evidence': {'normalized_entropy_median': summary['preview_normalized_entropy_median'],
                          'chosen_probability_median': summary['preview_chosen_probability_median']},
             'next_test': 'Isolate a preview-only policy change; leave move sampling and weights fixed; use new development/final cases.'},
            {'rank': 2, 'hypothesis': 'Some sampled turn actions sacrifice useful tactical pressure.',
             'evidence': {'choices_under_5_percent': summary['chosen_probability_under_5_percent'],
                          'prior_gap_over_one': summary['tactical_prior_gap_over_one']},
             'next_test': 'Inspect captured exact alternatives/targets and their battle context. A tactical prior is not an optimal-action label.'},
            {'rank': 3, 'hypothesis': 'Repeated protection may sometimes waste turns.',
             'evidence': {'consecutive_protect_events': patterns['consecutive_protect_moves']},
             'next_test': 'Separate successful stalling/defense from failed repeated protection before proposing reward or feature changes.'},
        ]
        return {'checked_at': now(), 'format': fmt, 'incumbent_revision': model.revision,
                'summary': summary, 'coverage': dict(coverage), 'patterns': dict(patterns),
                'strata': list(strata.values()), 'lineups': list(lineups.values()),
                'preview_evidence': previews, 'review_examples': examples,
                'excluded': excluded, 'ranked_hypotheses': hypotheses,
                'limitations': ['Descriptive correlations; team changes and small lineup samples prevent causal conclusions.',
                                'Legacy private requests/alternative names are unknown; no historical snapshots are fabricated.',
                                'Executed-move observations do not identify every intent or prove optimal choices.',
                                'Audit is read-only; it does not change consumption flags or create training labels.']}
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'data/ml')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.campaign, args.root)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
        print(json.dumps(result['summary'], indent=2))
    else:
        print(text)
