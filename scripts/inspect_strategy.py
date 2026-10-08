"""Read-only decision sensitivity and CPU latency on frozen recorded observations."""
import argparse
import copy
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.features import Features
from ml.model import Model
from ml.recording import log_prefix, named_choice


def run(root, checkpoint_root, campaign, output, count):
    fmt = 'gen9championsvgc2026regmc'
    model = Model(checkpoint_root, fmt)
    features = Features.cached(fmt)
    ledger = [json.loads(line) for line in (campaign / 'games.jsonl').read_text().splitlines() if line.strip()][:500]
    db = sqlite3.connect((root / 'experience.sqlite3').as_uri() + '?mode=ro', uri=True)
    previews, moves, times = [], [], []
    try:
        for game in reversed(ledger):
            rows = db.execute("SELECT data FROM episodes WHERE json_extract(data,'$.room')=? AND status='complete'", (game['room'],)).fetchall()
            if len(rows) != 1:
                continue
            e = json.loads(rows[0][0])
            for step in e['steps']:
                snap = step.get('snapshot')
                if not snap or not e.get('team_sets'):
                    continue
                preview = step['choice'].startswith('team ')
                target = previews if preview else moves
                if len(target) >= count:
                    continue
                ctx = {'room': e['room'], 'format': fmt, 'request': snap['request'],
                       'state': copy.deepcopy(snap['observation']), 'choices': [a['choice'] for a in snap['legal_choices']],
                       'team_sets': e['team_sets'], 'public_log': log_prefix(e, snap['public_log']),
                       'feature_profile': model.feature_profile, 'strategy_knowledge': model.strategy_knowledge}
                started = time.perf_counter()
                state, actions = features.encode(ctx, snap.get('knowledge'))
                prediction = model.predict(state, actions, preview=preview)
                elapsed = (time.perf_counter() - started) * 1000
                selected = ctx['choices'][prediction['index']]
                item = {'turn': ctx['state'].get('turn'), 'actual': step['choice'], 'proposed': selected,
                        'named': named_choice(ctx, selected), 'maximum_probability': max(prediction['probabilities']),
                        'inference_ms': round(elapsed, 3), 'choices': len(ctx['choices'])}
                item['saturated_prior_choices'] = int((np.abs(actions[:, -1]) >= 3.9999).sum())
                item['distinct_prior_values'] = len(set(actions[:, -1].tolist()))
                times.append(elapsed)
                if preview:
                    changed = {**ctx, 'state': {**ctx['state'], 'opp_preview': ['Torkoal','Charizard','Venusaur','Farigiraf','Incineroar','Garchomp'], 'opp_team_sheet': []}}
                    changed.pop('_opening_preview', None);changed.pop('_strategy', None)
                    changed.pop('_strategy_preview_boosted', None)
                    state2, actions2 = features.encode(changed, snap.get('knowledge'))
                    other = model.predict(state2, actions2, preview=True)
                    item['substituted_sun_team_choice'] = changed['choices'][other['index']]
                    item['opponent_substitution_changes_choice'] = item['substituted_sun_team_choice'] != selected
                    item['distribution_total_variation'] = float(np.abs(np.asarray(prediction['probabilities']) - np.asarray(other['probabilities'])).sum() / 2)
                target.append(item)
            if len(previews) >= count and len(moves) >= count:
                break
    finally:
        db.close()
    result = {'feature_profile': model.feature_profile, 'checkpoint_sha256': model.checkpoint_sha256,
              'preview_decisions': previews, 'move_decisions': moves,
              'median_inference_ms': float(np.median(times)), 'maximum_inference_ms': max(times, default=0),
              'preview_substitution_changes': sum(p['opponent_substitution_changes_choice'] for p in previews),
              'limitations': ['Counterfactual observation sensitivity and latency do not establish that proposed actions win.',
                              'Historical own actions are not relabeled as demonstrations or fresh PPO trajectories.']}
    output.write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k not in ('preview_decisions', 'move_decisions')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True);p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=12)
    a = p.parse_args()
    print(json.dumps(run(a.root.resolve(), a.checkpoint_root.resolve(), a.campaign.resolve(), a.output.resolve(), a.count), indent=2))
