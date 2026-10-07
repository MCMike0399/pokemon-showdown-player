"""Read-only campaign and policy diagnostics from verified terminal recordings."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def analyze(campaign: Path, root: Path) -> dict:
    from ml.model import Model
    from ml.storage import now
    import numpy as np
    import torch
    manifest = json.loads((campaign / 'manifest.json').read_text())
    fmt = manifest['format']
    checkpoint = root / 'models' / (fmt + '.pt')
    if not checkpoint.exists():
        raise ValueError('existing checkpoint required; analysis never initializes weights')
    model = Model(root, fmt)
    rows = [json.loads(line) for line in (campaign / 'games.jsonl').read_text().splitlines() if line.strip()]
    rooms = [row['room'] for row in rows]
    if len(set(rooms)) != len(rooms):
        raise ValueError('duplicate campaign rooms; reconcile before analysis')
    db = sqlite3.connect((root / 'experience.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
    outcomes = []
    usable = []
    fragmented = 0
    probabilities, gaps, previews, errors = [], [], [], []
    try:
        for room in rooms:
            episodes = [json.loads(r[0]) for r in db.execute("SELECT data FROM episodes WHERE json_extract(data,'$.room')=?", (room,))]
            terminal = [episode for episode in episodes if episode['status'] == 'complete']
            if len(terminal) != 1:
                raise ValueError('terminal room does not have exactly one complete episode')
            episode = terminal[0]
            outcomes.append(episode['outcome'])
            if len(episodes) != 1:
                fragmented += 1
                continue
            usable.append(episode)
        for episode in usable:
            for step in episode['steps']:
                probability = math.exp(step['logprob'])
                probabilities.append(probability)
                actions = np.asarray(step['actions'], dtype=np.float32)
                gaps.append(float(actions[:, -1].max() - actions[step['index'], -1]))
                if step['choice'].startswith('team '):
                    previews.append(probability)
                if episode['revision'] == model.revision and episode.get('on_policy'):
                    with torch.no_grad():
                        states, encoded, mask, indices = model.batch([step])
                        logits, _ = model.net(states, encoded, mask)
                        logprob = torch.distributions.Categorical(logits=logits / model.policy_temperature).log_prob(indices)
                        errors.append(abs(float(logprob[0]) - step['logprob']))
        trained, single, live_blocked = 0, 0, 0
        for row in db.execute('SELECT result FROM jobs WHERE format=? AND result IS NOT NULL', (fmt,)):
            result = json.loads(row[0])
            training = result.get('training') or {}
            trained += bool(training.get('trained'))
            single += training.get('trained', False) and training.get('episodes') == 1
            live_blocked += bool(result.get('promotion_blocked_by_live_game'))
        wins = sum(value == 1 for value in outcomes)
        n = len(outcomes)
        p = wins / n if n else 0
        z = 1.96
        center = (p + z*z/(2*n))/(1+z*z/n) if n else 0
        half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n) if n else 0
        return {'checked_at': now(), 'format': fmt, 'revision': model.revision,
                'policy_temperature': model.policy_temperature,
                'terminal_games': n, 'wins': wins, 'losses': sum(value == -1 for value in outcomes),
                'ties': sum(value == 0 for value in outcomes), 'win_rate': p,
                'win_rate_wilson_95': [center-half, center+half],
                'last_ten': {'games': len(outcomes[-10:]), 'wins': sum(value == 1 for value in outcomes[-10:])},
                'decision_episodes': len(usable), 'fragmented_excluded': fragmented,
                'steps': len(probabilities),
                'chosen_probability_median': statistics.median(probabilities) if probabilities else None,
                'choices_under_five_percent': sum(value < .05 for value in probabilities),
                'tactical_prior_gap_over_one': sum(value > 1 for value in gaps),
                'preview_chosen_probability_median': statistics.median(previews) if previews else None,
                'collecting_logprob_checks': len(errors), 'maximum_collecting_logprob_error': max(errors, default=None),
                'training_jobs': trained, 'single_episode_training_jobs': single,
                'historical_live_blocked_promotions': live_blocked,
                'limitations': ['Probabilities and tactical priors do not establish move correctness.',
                                'Ladder, scripted evaluations, scout loss and team evidence measure different outcomes.',
                                'Descriptive interval assumes independent games; repeated opponents and changing teams weaken that assumption.']}
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'data/ml')
    args = parser.parse_args()
    print(json.dumps(analyze(args.campaign, args.root), indent=2))
