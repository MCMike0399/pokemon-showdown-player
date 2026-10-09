"""Compare isolated before/after trainers on genuine compatible frozen rollouts.

No live model or source file is modified. Checkpoints are cloned into temporary
directories below the report directory; only reports persist. MPS timings include
explicit synchronization, transfers, validation and saving. Alternate run order
to reduce drift. Use --episodes-file for a read-only production JSONL snapshot.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_stream(args, checkpoint):
    if args.episodes_file:
        opener = gzip.open if args.episodes_file.suffix == '.gz' else open
        with opener(args.episodes_file, 'rt') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        path = args.episodes_dir / 'experience.sqlite3'
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            for row in connection.execute(
                    "SELECT data FROM episodes WHERE status='complete' AND revision=? ORDER BY created", (checkpoint['revision'],)):
                yield json.loads(row[0])
        finally:
            connection.close()
def read_episodes(args, checkpoint):
    selected, steps = [], 0
    for episode in episode_stream(args, checkpoint):
        if (episode['format'] != checkpoint['format'] or episode['revision'] != checkpoint['revision']
                or episode.get('status') != 'complete' or not episode.get('on_policy')
                or episode.get('feature_profile', 'legacy') != checkpoint.get('feature_profile', 'legacy')):
            continue
        count = len(episode.get('steps', []))
        if count and steps + count <= args.max_steps:
            selected.append(episode)
            steps += count
    if not selected:
        raise ValueError('no complete compatible episodes within the configured decision bound')
    return selected


def tensors_equal(left, right, strict, atol=2e-7, rtol=2e-6):
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        left, right = left.detach().cpu(), right.detach().cpu()
        return bool(torch.equal(left, right) if strict or not left.is_floating_point()
                    else torch.allclose(left, right, atol=atol, rtol=rtol))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensors_equal(left[k], right[k], strict, atol, rtol) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tensors_equal(a, b, strict, atol, rtol) for a, b in zip(left, right))
    return left == right


def compare(left, right, device):
    exact_keys = ('trained', 'steps', 'episodes', 'epochs_completed', 'optimizer_steps', 'early_stopped',
                  'consumed', 'excluded_episodes', 'architecture', 'reason')
    float_keys = ('loss', 'mean_entropy', 'kl_history', 'clip_fraction_history', 'maximum_collecting_logprob_error')
    a, b = left['report'], right['report']
    details = {key: a.get(key) == b.get(key) for key in exact_keys}
    for key in float_keys:
        details[key] = (a.get(key) == b.get(key) if device == 'cpu' or a.get(key) is None or b.get(key) is None
                        else bool(np.allclose(a[key], b[key], atol=2e-7, rtol=2e-6)))
    return {'report_fields': details,
            'parameters_bitwise_equal': tensors_equal(left['parameters'], right['parameters'], True),
            'optimizer_bitwise_equal': tensors_equal(left['optimizer'], right['optimizer'], True),
            'parameters_within_tolerance': tensors_equal(left['parameters'], right['parameters'], device == 'cpu'),
            'optimizer_within_tolerance': tensors_equal(left['optimizer'], right['optimizer'], device == 'cpu')}


def fit(model_module, batching_module, checkpoint_path, episodes, args, device, target_kl):
    with tempfile.TemporaryDirectory(prefix='isolated-sync-', dir=args.output) as directory:
        directory = Path(directory)
        (directory / 'models').mkdir()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        shutil.copy2(checkpoint_path, directory / 'models' / (checkpoint['format'] + '.pt'))
        model = model_module.Model(directory, checkpoint['format']).set_device(device)
        # Model.train imports batching lazily. Override only during this one fit;
        # restore it even on validation or allocator errors.
        previous = sys.modules.get('ml.batching')
        sys.modules['ml.batching'] = batching_module
        try:
            fresh_episodes = copy.deepcopy(episodes)
            if device == 'mps':
                torch.mps.synchronize()
            started = time.perf_counter()
            report = model.train(fresh_episodes, epochs=args.epochs, minibatch_size=args.minibatch_size,
                                 shuffle_seed=17, duty_fraction=args.duty_fraction, target_kl=target_kl)
            if device == 'mps':
                torch.mps.synchronize()
            seconds = time.perf_counter() - started
            model.set_device('cpu')
            return {'seconds': seconds, 'report': report,
                    'parameters': copy.deepcopy(model.net.state_dict()),
                    'optimizer': copy.deepcopy(model.optimizer.state_dict())}
        finally:
            if previous is None:
                sys.modules.pop('ml.batching', None)
            else:
                sys.modules['ml.batching'] = previous


def verify_batch_inputs(before, after, episodes, device):
    rows = [s for episode in episodes for s in episode['steps']]
    turn_positions = [i for i, step in enumerate(rows) if not step.get('choice', '').startswith('team ')]
    orders = [list(range(min(32, len(rows)))), list(reversed(range(min(32, len(rows))))), turn_positions[:32]]
    results = []
    for budget in (128, .05):
        old = before.RolloutBatcher(rows, device=device, max_cached_mb=budget)
        new = after.RolloutBatcher(rows, device=device, max_cached_mb=budget)
        for order in orders:
            if not order:
                continue
            original, changed = old.batch(order), new.batch(order)
            results.append({'budget_mb': budget, 'positions': len(order),
                            'actions_width': original[1].shape[1],
                            'original_mode': old.stats()['mode'], 'new_mode': new.stats()['mode'],
                            'all_inputs_bitwise_equal': tensors_equal(original, changed, True)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-model', type=Path, required=True)
    parser.add_argument('--before-batching', type=Path, required=True)
    parser.add_argument('--after-model', type=Path, default=REPO / 'ml/model.py')
    parser.add_argument('--after-batching', type=Path, default=REPO / 'ml/batching.py')
    parser.add_argument('--episodes-dir', type=Path, default=REPO / 'artifacts/offline-ppo-pilot/high_lr_reference-seed0')
    parser.add_argument('--episodes-file', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('auto', 'cpu', 'mps'), default='auto')
    parser.add_argument('--max-steps', type=int, default=256)
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--duty-fraction', type=float, default=1)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 8192 or not 1 <= args.repeats <= 20 or not 0 < args.duty_fraction <= 1:
        parser.error('bounded max steps/repeats and duty fraction are required')
    checkpoint_path = args.checkpoint or args.episodes_dir / 'initial.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    episodes = read_episodes(args, checkpoint)
    before_model = load_module(args.before_model, '_sync_model_before')
    before_batching = load_module(args.before_batching, '_sync_batching_before')
    after_model = load_module(args.after_model, '_sync_model_after')
    after_batching = load_module(args.after_batching, '_sync_batching_after')
    args.output.mkdir(parents=True, exist_ok=False)
    devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []) if args.backend == 'auto' else [args.backend]
    if 'mps' in devices and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this execution environment')
    result = {'host': platform.node(), 'torch': torch.__version__, 'mps_available': torch.backends.mps.is_available(),
              'input_steps': sum(len(e['steps']) for e in episodes), 'input_episodes': len(episodes),
              'checkpoint_sha256': digest(checkpoint_path), 'checkpoint_revision': checkpoint['revision'],
              'input_episode_sha256': hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest(),
              'source_sha256': {name: digest(getattr(args, name)) for name in ('before_model', 'after_model', 'before_batching', 'after_batching')},
              'settings': {'epochs': args.epochs, 'minibatch_size': args.minibatch_size, 'duty_fraction': args.duty_fraction,
                           'warmup_runs': 1, 'timed_runs': args.repeats,
                           'timed_order_balanced': args.repeats % 2 == 0,
                           'input_cloning_timed': False}, 'devices': [],
              'caution': 'Timing includes validation and saving. This is implementation equivalence, not independent policy evaluation. Short timing differences can be noise.'}
    for device in devices:
        timings = {'before': [], 'after': []}
        checks = []
        batching = verify_batch_inputs(before_batching, after_batching, episodes, device)
        assert all(row['all_inputs_bitwise_equal'] for row in batching), 'batch input mismatch'
        for repeat in range(args.repeats + 1):
            fitted = {}
            order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
            for name in order:
                models, batches = (before_model, before_batching) if name == 'before' else (after_model, after_batching)
                fitted[name] = fit(models, batches, checkpoint_path, episodes, args, device, .03)
                if repeat:
                    timings[name].append(fitted[name]['seconds'])
            check = compare(fitted['before'], fitted['after'], device)
            assert all(check['report_fields'].values()) and check['parameters_within_tolerance'] and check['optimizer_within_tolerance'], check
            checks.append(check)
        # A deliberately impossible collecting likelihood must receive the same
        # exclusion and no training credit in both implementations.
        corrupted = copy.deepcopy(episodes)
        for episode in corrupted:
            episode['steps'][0]['logprob'] += 1
        exclusion = compare(fit(before_model, before_batching, checkpoint_path, corrupted, args, device, .03),
                            fit(after_model, after_batching, checkpoint_path, corrupted, args, device, .03), device)
        assert all(exclusion['report_fields'].values()), exclusion
        early_stop = compare(fit(before_model, before_batching, checkpoint_path, episodes, args, device, 1e-8),
                             fit(after_model, after_batching, checkpoint_path, episodes, args, device, 1e-8), device)
        assert all(early_stop['report_fields'].values()) and early_stop['parameters_within_tolerance'], early_stop
        medians = {name: float(np.median(values)) for name, values in timings.items()}
        result['devices'].append({'backend': device, 'timings': timings, 'median_seconds': medians,
                                  'speedup': medians['before'] / medians['after'], 'batch_input_checks': batching,
                                  'equivalence': checks, 'invalid_likelihood_equivalence': exclusion,
                                  'strict_kl_stop_equivalence': early_stop})
        (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'backend': device, 'steps': result['input_steps'], 'median_seconds': medians,
                          'speedup': medians['before'] / medians['after'], 'equivalent': True}), flush=True)


if __name__ == '__main__':
    main()
