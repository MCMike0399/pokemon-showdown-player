"""Recompute offline PPO evidence without training, simulations, or checkpoint writes.

Run with ``.venv/bin/python -m experiments.analyze_results``. Reports are deliberately
written under gitignored artifacts/. All databases and checkpoints are read only.
This is exploratory analysis, not a checkpoint promotion gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import torch

from ml.features import Features
from ml.model import ActorCritic
from ml.recording import log_prefix
from ml.scout import Scout
from ml.storage import Store

REPO = Path(__file__).resolve().parents[1]
FMT = 'gen9championsvgc2026regmc'


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(wins, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = wins / n
    divisor = 1 + z * z / n
    center = (p + z * z / (2 * n)) / divisor
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / divisor
    return [center - radius, center + radius]


def score(game):
    return 0 if game.get('tie') else 1 if game['winner'] == 'LocalBrain' else -1


def case_key(game):
    return (game['seed'], tuple(game['simulator_seed']), game['opponent'],
            game['learner_side'], game['learner_team'], game['opponent_team'],
            game['open_team_sheets'], game['seed_encoding'], game['policy_mode'])


def paired(before, after):
    """Exact one-sided sign/McNemar statistic on discordant paired outcomes."""
    left, right = {case_key(g): g for g in before}, {case_key(g): g for g in after}
    if len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys():
        raise ValueError('evaluation cases are not complete, unique, identical pairs')
    delta = np.array([score(right[k]) - score(left[k]) for k in left], dtype=float)
    better, worse = int((delta > 0).sum()), int((delta < 0).sum())
    discordant = better + worse
    probability = sum(math.comb(discordant, k) for k in range(better, discordant + 1)) / 2 ** discordant
    # Conditional on these frozen policies/cases, not a training-seed population CI.
    rng = np.random.default_rng(20261009)
    bootstrap = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {'pairs': len(delta), 'improved': better, 'worsened': worse,
            'unchanged': len(delta) - discordant, 'mean_return_delta': float(delta.mean()),
            'win_rate_delta': float(delta.mean() / 2),
            'conditional_game_bootstrap_win_delta95': (np.quantile(bootstrap, [.025, .975]) / 2).tolist(),
            'one_sided_exact_sign_p': probability}


def holm(rows):
    previous = 0.0
    ordered = sorted(rows, key=lambda r: r['one_sided_exact_sign_p'])
    for index, row in enumerate(ordered):
        previous = max(previous, min(1.0, row['one_sided_exact_sign_p'] * (len(rows) - index)))
        row['holm_adjusted_p'] = previous


def aggregate(games):
    wins = sum(score(g) == 1 for g in games)
    return {'games': len(games), 'wins': wins, 'win_rate': wins / len(games),
            'wilson95': wilson(wins, len(games)),
            'rejected_actions': sum(g.get('rejected_actions', 0) for g in games),
            'unavailable_choices': sum(g.get('unavailable_choices', 0) for g in games)}


def analyze_matrix(root):
    matrix = read_json(root / 'matrix.json')
    reports = [read_json(path) for path in sorted(root.glob('*-seed*/report.json'))]
    if not reports:
        raise ValueError('no completed matrix reports')
    rows, tests = [], []
    for arm in matrix['arms']:
        runs = [r for r in reports if r['arm'] == arm['name']]
        row = {'arm': arm['name'], 'settings': arm, 'seeds': [r['seed'] for r in runs]}
        all_updates = [u for r in runs for u in r['updates']]
        trainings = [u['training'] for u in all_updates]
        row['training'] = {
            'steps': sum(r['steps'] for r in runs),
            'episodes': sum(u['training']['episodes'] for u in all_updates),
            'optimizer_steps_per_seed': [sum(u['training']['optimizer_steps'] for u in r['updates']) for r in runs],
            'training_seconds': sum(t['training_seconds'] for t in trainings),
            'elapsed_seconds': sum(r['elapsed_seconds'] for r in runs),
            'max_epoch_kl': max(max(t['kl_history']) for t in trainings),
            'max_epoch_clip_fraction': max(max(t['clip_fraction_history']) for t in trainings),
            'early_stopped_updates': sum(t['early_stopped'] for t in trainings),
            'mean_entropy_first_update': [r['updates'][0]['training']['mean_entropy'] for r in runs],
            'mean_entropy_last_update': [r['updates'][-1]['training']['mean_entropy'] for r in runs],
            'minibatch_sizes': sorted({t['minibatch_size'] for t in trainings}),
            'max_collecting_logprob_error': max(t['maximum_collecting_logprob_error'] for t in trainings),
            'excluded_episodes': sum(len(t['excluded_episodes']) for t in trainings),
            'loss_min': min(t['loss'] for t in trainings), 'loss_max': max(t['loss'] for t in trainings),
            'collection_returns_per_seed': [[u['collection']['mean_episode_return'] for u in r['updates']] for r in runs],
        }
        row['evaluation'] = {}
        row['evaluation_by_matchup'] = {}
        for opponent in ('random', 'tactical'):
            labels = sorted({e['label'] for r in runs for e in r['evaluations']})
            row['evaluation'][opponent] = {
                label: aggregate([g for r in runs for e in r['evaluations']
                                  if e['label'] == label and e['opponent'] == opponent for g in e['games']])
                for label in labels}
            row['evaluation_by_matchup'][opponent] = {}
            for label in ('initial', 'final'):
                groups = defaultdict(list)
                for run in runs:
                    for evaluation in run['evaluations']:
                        if evaluation['label'] == label and evaluation['opponent'] == opponent:
                            for game in evaluation['games']:
                                groups[(game['learner_team'], game['opponent_team'])].append(game)
                row['evaluation_by_matchup'][opponent][label] = [
                    {'learner_team': key[0], 'opponent_team': key[1], **aggregate(values)}
                    for key, values in sorted(groups.items())]
            latest = 'update-' + str(len(runs[0]['updates']))
            per_seed = []
            initial, latest_games = [], []
            for run in runs:
                before = next(e['games'] for e in run['evaluations'] if e['label'] == 'initial' and e['opponent'] == opponent)
                after = next(e['games'] for e in run['evaluations'] if e['label'] == latest and e['opponent'] == opponent)
                per_seed.append({'seed': run['seed'], **paired(before, after)})
                initial.extend(before)
                latest_games.extend(after)
            test = {'arm': arm['name'], 'opponent': opponent, **paired(initial, latest_games), 'per_seed': per_seed}
            tests.append(test)
        rows.append(row)
    holm(tests)
    control = next(r for r in rows if r['arm'] == 'repository_control')
    final_tests = []
    for arm in rows:
        if arm is control:
            continue
        for opponent in ('random', 'tactical'):
            def finals(name):
                return [g for r in reports if r['arm'] == name for e in r['evaluations']
                        if e['opponent'] == opponent and e['label'] == 'final' for g in e['games']]
            final_tests.append({'arm': arm['arm'], 'opponent': opponent,
                                **paired(finals(control['arm']), finals(arm['arm']))})
    holm(final_tests)
    return {'matrix': matrix, 'runs': len(reports), 'updates': sum(len(r['updates']) for r in reports),
            'training_steps': sum(r['steps'] for r in reports),
            'collection_games': sum(u['collection']['completed'] for r in reports for u in r['updates']),
            'evaluation_games': sum(e['completed'] for r in reports for e in r['evaluations']),
            'arms': rows, 'development_initial_vs_last': tests, 'final_vs_repository_control': final_tests,
            'caution': 'Conditional fixed-policy case statistics; only three training seeds. Initial evaluation is repeated across arms and cannot be counted as independent baseline replication. Final cases differ from development; no initial actor was evaluated on final seeds.'}


def episodes_from_db(path):
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for row in connection.execute("SELECT data FROM episodes WHERE status='complete' ORDER BY created"):
            yield json.loads(row[0])
    finally:
        connection.close()


def select_steps(episodes, count):
    """Spread deterministic samples across episodes; use two real turn decisions each."""
    result = []
    for episode in episodes:
        decisions = [s for s in episode['steps'] if not s['choice'].startswith('team ') and len(s['actions']) > 1]
        for step in decisions[:2]:
            result.append((episode, step))
        if len(result) >= count:
            break
    if not result:
        raise ValueError('no recorded turn decisions for probes')
    return result[:count]


def probe_checkpoint(path, samples, scout, features):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    architecture = checkpoint.get('architecture', 'concat-v1')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = ActorCritic(architecture)
    net.load_state_dict(checkpoint['model'])
    net.eval()
    temperature = checkpoint.get('policy_temperature', 1)
    rows = defaultdict(list)
    heuristic_distance, entropies, maximum_entropies = [], [], []
    for index, (episode, step) in enumerate(samples):
        state = np.array(step['state'], dtype=np.float32)
        alternatives = np.array(step['actions'], dtype=np.float32)
        other = np.array(samples[(index + 1) % len(samples)][1]['state'], dtype=np.float32)
        modified = state.copy()
        captured = step.get('snapshot', {})
        actual_predictions = []
        if scout and captured.get('public_log'):
            actual_predictions = scout.predict(log_prefix(episode, captured['public_log']), episode.get('side', 'p1'))
            for prediction in captured.get('opponent_predictions', []):
                for move in prediction['moves']:
                    features.add(modified, 'scout/' + prediction['slot'][-1] + '/' + move['move'], -move['probability'])
            for prediction in actual_predictions:
                for move in prediction['moves']:
                    features.add(modified, 'scout/' + prediction['slot'][-1] + '/' + move['move'], move['probability'])
        synthetic = state.copy()
        for slot, move, probability in [('a', 'protect', .8), ('a', 'earthquake', .2), ('b', 'fakeout', .6), ('b', 'tailwind', .4)]:
            features.add(synthetic, f'scout/{slot}/{move}', probability)
        variants = [state, np.zeros_like(state), other, synthetic, modified]
        with torch.no_grad():
            logits, values = net(torch.as_tensor(np.array(variants)), torch.as_tensor(alternatives)[None].expand(len(variants), -1, -1))
            probabilities = (logits / temperature).softmax(-1).numpy()
            original = probabilities[0]
            prior_probabilities = (torch.as_tensor(alternatives[:, -1]) / temperature).softmax(0).numpy()
            heuristic_distance.append(float(np.abs(original - prior_probabilities).sum() / 2))
            entropies.append(float(-(original * np.log(np.maximum(original, 1e-30))).sum()))
            maximum_entropies.append(math.log(len(original)))
        for variant_index, name in enumerate(('zero_state', 'other_recorded_state', 'synthetic_scout', 'actual_scout_replacement'), 1):
            changed = probabilities[variant_index]
            rows[name].append({'total_variation': float(np.abs(changed - original).sum() / 2),
                               'greedy_choice_changed': bool(original.argmax() != changed.argmax()),
                               'critic_absolute_change': float(abs(values[variant_index] - values[0])),
                               'input_l2_change': float(np.linalg.norm(variants[variant_index] - state)),
                               'has_scout_prediction': bool(actual_predictions)})
    stats = {}
    for name, values in rows.items():
        variations = [v['total_variation'] for v in values]
        stats[name] = {'mean_total_variation': float(np.mean(variations)),
                       'max_total_variation': max(variations),
                       'greedy_changes': sum(v['greedy_choice_changed'] for v in values),
                       'mean_critic_absolute_change': float(np.mean([v['critic_absolute_change'] for v in values])),
                       'nonzero_inputs': sum(v['input_l2_change'] > 1e-8 for v in values),
                       'samples_with_scout_predictions': sum(v['has_scout_prediction'] for v in values)}
    try:
        display_path = str(path.relative_to(REPO))
    except ValueError:
        display_path = str(path)
    return {'path': display_path, 'sha256': sha256(path), 'architecture': architecture,
            'updates': checkpoint['updates'], 'feature_profile': checkpoint.get('feature_profile', 'legacy'),
            'policy_temperature': temperature, 'samples': len(samples), 'variants': stats,
            'mean_tv_from_same_temperature_heuristic_prior': float(np.mean(heuristic_distance)),
            'mean_policy_entropy': float(np.mean(entropies)),
            'mean_fraction_maximum_entropy': float(np.mean(np.array(entropies) / np.array(maximum_entropies))),
            'caution': 'Holding action vectors fixed isolates neural state dependence. Cross-profile inputs and counterfactual states are sensitivity diagnostics, not calibrated battle-strength estimates or legal counterfactual battles.'}


def probes(root, corpus, count):
    local = select_steps(episodes_from_db(root / 'high_lr_reference-seed0' / 'experience.sqlite3'), count)
    with gzip.open(corpus / 'historic-ladder-episodes.jsonl.gz', 'rt') as handle:
        historic = select_steps((json.loads(line) for line in handle), count)
    store = Store.read_only(root.parent / 'offline-scout')
    features = Features({name: read_json(corpus / 'dex' / FMT / (name + '.json'))
                         for name in ('moves', 'pokedex', 'typechart', 'natures', 'rules')})
    scout = Scout(store, FMT, features)
    try:
        checkpoints = [root / 'high_lr_reference-seed0' / 'initial.pt']
        checkpoints.extend(p / 'models' / (FMT + '.pt') for p in sorted(root.glob('*-seed*')) if (p / 'report.json').exists())
        checkpoints.append(corpus / 'models' / (FMT + '.pt'))
        return {'local_simulator_features': [probe_checkpoint(path, local, scout, features) for path in checkpoints],
                'historic_ladder_features': [probe_checkpoint(path, historic, scout, features) for path in (checkpoints[0], root / 'high_lr_reference-seed0/models' / (FMT + '.pt'), checkpoints[-1])],
                'local_feature_profile': local[0][0].get('feature_profile', 'legacy'),
                'historic_feature_profile': historic[0][0].get('feature_profile', 'legacy')}
    finally:
        store.close()


def findings(stats):
    budget = stats['budget']
    lines = ['# Macintosh offline PPO evidence', '',
             'Reproduce: `.venv/bin/python -m experiments.analyze_results`. This analysis reads retained game reports, databases and checkpoints; it does not train, simulate, promote or write checkpoints.', '',
             f"The completed main matrix contains {budget['runs']} runs, {budget['updates']} updates, {budget['training_steps']:,} fresh accepted training decisions, {budget['collection_games']:,} collection games and {budget['evaluation_games']:,} evaluation games. Each arm has three training seeds and only five updates. This is an early learning experiment, not a completed convergence study. Source: `artifacts/offline-ppo-reddit-budget/matrix.json`, per-run `report.json` and `updates.json`.", '',
             '## Outcomes and uncertainty', '',
             '| Arm | Final random wins / 96 | Final tactical wins / 96 | Tactical development change | Paired gain/loss | Raw p | Holm p |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in budget['arms']:
        test = next(t for t in budget['development_initial_vs_last'] if t['arm'] == arm['arm'] and t['opponent'] == 'tactical')
        random = arm['evaluation']['random']['final']
        tactical = arm['evaluation']['tactical']['final']
        lines.append(f"| {arm['arm']} | {random['wins']} | {tactical['wins']} | {100 * test['win_rate_delta']:+.2f} pp | {test['improved']}/{test['worsened']} | {test['one_sided_exact_sign_p']:.4f} | {test['holm_adjusted_p']:.4f} |")
    baseline = budget['arms'][0]['evaluation']
    lines.extend(['', f"The same initial heuristic-prior policy wins {baseline['random']['initial']['wins']}/96 random and {baseline['tactical']['initial']['wins']}/96 tactical development games in every arm. Those repeated evaluations are not additional independent baseline evidence. The zero-initialized final actor layer contributes no learned preference: logits start at the action-vector heuristic prior (`ml/model.py: ActorCritic.forward` and `ml/features.py: Features.action`). Random wins therefore cannot be attributed entirely to PPO.", '',
                  'Game-level Wilson intervals and conditional paired bootstrap intervals are in `stats.json`. They describe these frozen policies and finite cases; only three training seeds are available, so general claims about retraining stability are weak. An exact sign test across three positive seed effects cannot reach 0.05 (best possible one-sided p = 0.125). Four fixed neighboring team matchups and two scripted opponents cover only a small fraction of competitive battle strategy.', '',
                  'Development initial/update pairs verify supplied and effective simulator seeds, sides, team fingerprints, sheet visibility and sampled action mode. Holm corrections cover 14 exploratory arm/opponent development hypotheses. Final seeds are disjoint from development and collection, but the initial policy was never measured on final seeds. Consequently final-vs-initial improvements cannot be inferred by subtracting different suites. The 12 final arm/opponent comparisons against repository control are separately corrected. Reusing these finals to choose future adjustments turns them into development data; deployment needs a new untouched suite. Source: `experiments/offline_ppo.py: evaluate/run_arm/task` and per-game report fields.', '',
                  '## Optimizer exposure and entropy', '',
                  '| Arm | Optimizer steps by seed | Maximum epoch KL | Max clip fraction | KL early stops | First entropy by seed | Last entropy by seed |',
                  '| --- | --- | ---: | ---: | ---: | --- | --- |'])
    for arm in budget['arms']:
        train = arm['training']
        values = lambda xs: ', '.join(f'{v:.3f}' for v in xs)
        lines.append(f"| {arm['arm']} | {train['optimizer_steps_per_seed']} | {train['max_epoch_kl']:.6f} | {train['max_epoch_clip_fraction']:.3f} | {train['early_stopped_updates']} | {values(train['mean_entropy_first_update'])} | {values(train['mean_entropy_last_update'])} |")
    lines.extend(['', 'All matrix collection games use random opponents: `run_arm` calls `task(..., training=True)` without an opponent override, and `task` defaults to `random`. Tactical opponents appear only in evaluation. Tactical generalization is therefore outside the training-opponent distribution; more random games may mostly reinforce the prior against an easy opponent. Retain the production mixed tactical/self-play curriculum rather than copying this isolated notebook schedule.', '',
                  'The larger-collection arms simultaneously double the fresh-decision target, reduce learning rate, change epoch count and increase minibatches to roughly one quarter of the collection. Their 80 optimizer steps per run cannot be compared as an isolated rollout-size effect with the hundreds of smaller-minibatch steps in other arms. All use complete terminal-reward episodes, not the truncated vector horizons described in the Reddit discussion. Mean entropy is averaged across minibatches and varying legal-action counts; changes in state mix affect it. It is not a directly controlled entropy target or evidence of convergence.', '',
                  'The pure low-entropy arm isolates an entropy-coefficient change against the high-LR reference, but its finite tactical improvements are small and not evidence to globally force entropy to 1e-4. Lower learning rate materially reduces measured KL; large minibatches give very small policy movement in five updates. The KL budget is checked after complete epochs, so an epoch may cross 0.03 before later epochs stop. These stable finite losses/likelihood checks support conservative candidates, not superiority. Source: `ml/model.py: Model.train`, `experiments/offline_ppo.py: ARMS/run_arm`, retained `updates.json`.', '',
                  'The pilot has one seed, three updates and eight evaluation games per opponent, with all collection targets shrunk to 128. It verifies the experiment path but cannot validate a larger-rollout claim. Its statistics are preserved separately under `pilot` in `stats.json`.', '',
                  '## State and scout sensitivity', ''])
    for probe in stats['probes']['local_simulator_features']:
        if probe['path'].endswith('initial.pt') or 'high_lr_reference-seed0/models' in probe['path'] or 'divemac-20261009/models' in probe['path']:
            zero = probe['variants']['zero_state']
            scout = probe['variants']['actual_scout_replacement']
            lines.append(f"- `{probe['path']}` ({probe['architecture']}, updates {probe['updates']}, temperature {probe['policy_temperature']}): zero-state mean action-distribution total variation {zero['mean_total_variation']:.8f}, greedy changes {zero['greedy_changes']}/{probe['samples']}; actual scout replacement mean variation {scout['mean_total_variation']:.8f}, greedy changes {scout['greedy_changes']}/{probe['samples']}; mean distribution variation from its same-temperature action prior {probe['mean_tv_from_same_temperature_heuristic_prior']:.5f}.")
    lines.extend(['', 'Probes keep recorded legal action vectors fixed and vary only the state vector: zero, another recorded state, synthetic hashed scout moves, and replacement with the actual replay-trained scout predictions computed from the recorded pre-turn public-log prefix. Full per-checkpoint statistics and separate historical ladder-feature probes are in `stats.json`. Incompatible feature profiles and counterfactual states are explicitly diagnostic; these probes do not establish battle strength. The concat actor can learn state-conditioned preferences through its nonlinear joint layer; it is not algebraically state independent once trained. Its initial learned actor is exactly state independent because the output layer is zero. The critic can be strongly state dependent even when the action distribution changes little.', '',
                  'The incumbent scout ablation has 200 exact matched outcomes (100 random, 100 tactical), zero improved and zero worsened cases. This fails to establish battle benefit; finite outcome equality does not prove distributions were identical. The replay scout did improve held-out executed-move prediction loss, with complete-battle splits. Historical M-B remains historical; the current M-C checkpoint uses current-format held-out validation after historical warm start. Better move prediction alone is insufficient for actor deployment. Source: `artifacts/offline-scout/scout-report.json`, `artifacts/offline-scout-comparison/{paired,comparison}.json`, `ml/brain.py: decide`, `ml/scout.py`.', '',
                  'The repository already supports `matchup-v1`, an explicit state/action multiplicative residual. `Model.enable_matchups()` initializes its score to zero to preserve old logits, then starts a new candidate revision and optimizer. Do not recreate that architecture or silently migrate a live checkpoint. An isolated incumbent-based matchup candidate is a defensible experiment when state/scout conditioning is weak, subject to new on-policy collection and matched independent evaluation. These fresh concat models are much weaker against tactical scripts than the copied incumbent scout comparison; that comparison uses a different suite/profile, so it is descriptive and not a paired promotion claim.', '',
                  '## Hardware findings', '',
                  'The short Macintosh worker sweep contains one 28-game timing per worker count, includes process startup and favors 14 workers (531.9 games/minute) over 8 (471.4), 4 (460.6), 12 (443.4) and 1 (198.5). This does not establish the optimum on the 10-core/16-GB shared DiveMac. Use persistent pools, one Torch thread per simulator, independent CPU evaluation capacity and actual host headroom. Oversubscribing or competing launchd learners can lose throughput and battle latency. Source: `artifacts/offline-worker-benchmark/workers.json`, `ml/resources.py`, `ml/pipeline.py`.', '',
                  'The final synchronized end-to-end training benchmark on Macintosh has 1,050 shape-replicated decisions from 150 unique decisions, one warmup and two timed runs per setting. CPU batch 256 is best at 2.073 s; MPS batch 128 is best at 2.186 s. GPU forcing is not an evidence-backed improvement for this small actor. Large variable action masks, validation passes, Python work, transfers and saves also matter. An earlier run showed a small MPS batch-32 advantage but lost at batch128/256; short runs vary and should not determine DiveMac settings. Benchmark the real candidate workload on DiveMac, preserve synchronization and end-to-end timing, and choose MPS only when it improves measured throughput or allows useful CPU/GPU overlap. Source: `artifacts/offline-training-benchmark{,-large,-final}/training.json`, `experiments/offline_ppo.py: benchmark_training`.', '',
                  '## Recommended adjustments', '',
                  '1. Retain the live incumbent. Do not promote a fresh notebook actor or scout based on random-opponent success or uncorrected development gains.',
                  '2. Test an incumbent-initialized `matchup-v1` candidate using existing migration, the exact focused team/profile, frozen scout/research inputs, complete accepted fresh on-policy collection, matched seeds/sides and a new final suite. Preserve revision boundaries and parent checksums.',
                  '3. Keep conservative learning rate/KL behavior. Expose explicit candidate entropy, minibatch and learning-rate settings if missing, retain safe defaults, and record optimizer exposure. Future ablations should hold optimizer steps and fresh decisions comparable; compare low entropy separately from rollout/batch changes.',
                  '4. Optimize host throughput rather than GPU occupancy. Keep live inference CPU-local; overlap bounded CPU collection and evaluation with candidate training, avoid simultaneous competing scheduled learners, benchmark full PPO shapes on DiveMac, and tune resource budgets from that shared host rather than copying Macintosh\'s 14-worker setting.',
                  '5. Retain reports, manifests, source fingerprints, initial/final weights, useful accepted trajectory databases and analysis on DiveMac with hashes before cleaning local experiment/corpus/cache/runtime data. Avoid copying duplicate per-game checkpoint/shard copies where the merged recordings and unique retained policies preserve provenance.', '',
                  'A concrete implementation change is to reduce diagnostic GPU readbacks: accumulate detached loss/entropy scalars and KL/clip tensors, transfer them together at the reporting/epoch boundary, then preserve their original ordered Python summation. Keep the finite-loss check, optimizer updates, action masking, temperatures, resource checks, explicit MPS duty pauses and KL stopping semantics. Separately slice resident action/mask width before index selection to avoid copying unused padding. This changes implementation cost while preserving the learned policy. [PyTorch 2.14 Tensor.tolist](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.tolist.html) documents the CPU transfer; [MPS scalar extraction source](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mps/operations/Scalar.mm) performs a blocking copy to a CPU destination, and [MPS synchronize](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html) defines explicit kernel completion.', '',
                  'The isolated equivalence helper is `experiments/benchmark_training_sync.py`; it loads explicit before/after source paths, clones frozen checkpoints, reads either compatible recording databases or a production JSONL snapshot, and compares parameters, Adam state, diagnostics, complete/invalid likelihood handling, strict KL stopping, reordered inputs and both resident/bounded caches. CPU comparisons are exact; MPS accepts tightly bounded numerical tolerance with identical stopping and exclusions. Default timing uses balanced alternating order, one warmup, four timed runs and excludes preparatory input deep copies. Local CPU results and archive selection audits are separate ignored JSON files beside this report. GPU performance must be measured in a host execution environment that exposes MPS; the current restricted local process cannot access it.', '',
                  'These are evidence-backed next candidates and service improvements; the experiments have not established a better live policy. Reports and stats remain gitignored.'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=REPO / 'artifacts/offline-ppo-reddit-budget')
    parser.add_argument('--corpus', type=Path, default=REPO / 'data/offline-corpus/divemac-20261009')
    parser.add_argument('--output', type=Path, default=REPO / 'artifacts/offline-analysis')
    parser.add_argument('--probe-samples', type=int, default=64)
    args = parser.parse_args()
    if not 2 <= args.probe_samples <= 256:
        parser.error('probe samples must be between 2 and 256')
    if REPO / 'artifacts' not in args.output.resolve().parents:
        parser.error('write analysis reports under gitignored artifacts/')
    artifact_root = args.matrix.resolve().parent
    stats = {'analysis_sha256': sha256(__file__),
             'source_sha256': {p: sha256(REPO / p) for p in ('ml/model.py', 'ml/features.py', 'ml/scout.py', 'ml/brain.py', 'experiments/offline_ppo.py')},
             'budget': analyze_matrix(args.matrix.resolve()),
             'pilot': analyze_matrix(artifact_root / 'offline-ppo-pilot'),
             'scout': read_json(artifact_root / 'offline-scout/scout-report.json'),
             'scout_paired': read_json(artifact_root / 'offline-scout-comparison/paired.json'),
             'workers': read_json(artifact_root / 'offline-worker-benchmark/workers.json'),
             'training_benchmarks': {p.parent.name: read_json(p) for p in sorted(artifact_root.glob('offline-training*/training.json'))},
             'probes': probes(args.matrix.resolve(), args.corpus.resolve(), args.probe_samples)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'stats.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')
    (args.output / 'findings.md').write_text(findings(stats))
    print(json.dumps({'output': str(args.output), 'runs': stats['budget']['runs'],
                      'training_steps': stats['budget']['training_steps'],
                      'paired_significant_after_holm': sum(t['holm_adjusted_p'] <= .05 for t in stats['budget']['development_initial_vs_last'])}))


if __name__ == '__main__':
    main()
