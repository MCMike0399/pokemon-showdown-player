"""Leakage and intervention bounds for the experimental preview correction."""
import hashlib
import json

import numpy as np
import pytest

from search.preview_residual import TARGET, RISKS, features, bounded_choice, PreviewResidual
from search.human.preview_model import PLANS, NS
from search.train_preview_residual import split_games


def test_chronological_split_keeps_both_sides_and_duplicates_together():
    games = [f'battle-{n}' for n in range(100)]
    splits = split_games(games[::-1] + games)
    assert [splits[g] for g in games] == ['train']*70 + ['val']*15 + ['test']*15


def test_matchup_features_ignore_team_order_but_distinguish_leads():
    plan = PLANS[0]
    a = features(TARGET, RISKS, plan)
    perm = list(range(5, -1, -1))
    b = features([TARGET[i] for i in perm], RISKS[::-1], tuple(perm.index(i) for i in plan))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, features(TARGET, RISKS, (plan[2], plan[3], plan[0], plan[1])))


def test_correction_never_selects_illegal_or_far_below_incumbent_plan():
    scores = np.array([.70, .69, .60, .80])
    assert bounded_choice(scores, np.array([-.35, .35, .35, .35]), [0, 1, 2]) == 1
    assert bounded_choice(scores, np.array([0, -.35, .35, .35]), [0, 1, 2]) == 0
    assert bounded_choice(scores, np.zeros(4), []) is None


def model_files(tmp_path):
    baseline = tmp_path / 'base.json'
    baseline.write_text(json.dumps({'layers': [{'W': [[0.]*(4*NS)], 'b': [0.]}]}))
    path = tmp_path / 'residual.json'
    path.write_text(json.dumps({'version': 1, 'cap': .35,
        'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'layers': [{'W': [[0.]*len(features(TARGET, RISKS, PLANS[0]))], 'b': [0.]}]}))
    return {'path': str(path), 'baseline': str(baseline)}


def test_inference_is_opt_in_matchup_scoped_and_legality_preserving(tmp_path):
    model = PreviewResidual(model_files(tmp_path))
    ctx = {'request': {'side': {'id': 'p1', 'pokemon': [{'details': s} for s in TARGET]}},
           'public_log': ['|poke|p2|' + s + '|' for s in RISKS], 'choices': ['team 1,2,3,4']}
    assert model.choose(ctx) == 'team 1,2,3,4'
    ctx['public_log'] = ['|poke|p2|Pikachu|']*6
    assert model.choose(ctx) is None
    ctx['public_log'] = ['|poke|p2|' + s + '|' for s in RISKS]
    ctx['request']['side']['pokemon'][0]['details'] = 'Pikachu'
    assert model.choose(ctx) is None


def test_mismatched_baseline_and_widening_bound_refuse(tmp_path):
    config = model_files(tmp_path)
    with pytest.raises(ValueError, match='eligibility'):
        PreviewResidual({**config, 'max_drop': .1})
    from pathlib import Path
    Path(config['baseline']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        PreviewResidual(config)
