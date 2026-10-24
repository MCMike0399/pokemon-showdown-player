"""Campaign separation, frozen protocols, public-information preview and test isolation."""
import json
from pathlib import Path

import numpy as np
import pytest

from search import campaign, train_screen
from search.preview_guard import PLANS, rerank


def choice(p):
    return 'team ' + ','.join(str(i + 1) for i in p)


def test_guard_can_change_leads_without_banning_intimidate_from_the_bring():
    scores = np.zeros(len(PLANS))
    a = PLANS.index((0, 1, 2, 3))
    b = PLANS.index((1, 2, 0, 3))
    scores[a], scores[b] = .60, .59
    legal = [choice(PLANS[a]), choice(PLANS[b])]
    abilities = ['Intimidate', '', '', '', '', '']
    assert rerank(scores, abilities, ['Milotic'], legal) == legal[1]
    assert rerank(scores, abilities, ['Politoed'], legal) == legal[0]
    scores[b] = .56  # the candidate may not override a stronger matchup prediction
    assert rerank(scores, abilities, ['Kingambit', 'Milotic'], legal) == legal[0]
    assert rerank(scores, abilities, ['Milotic'], [legal[0]]) == legal[0]


def test_holdout_failure_rejects_without_widening_the_prefilter():
    pre = {'ours': 24, 'opp': 20}
    met = {'ours': {'groups': 100, 'by_pre': {'24': {'recall': .95, 'best': .99}}},
           'theirs': {'groups': 100, 'by_pre': {'20': {'recall': .5}, '24': {'recall': .99}}}}
    assert not train_screen.check_holdout(met, pre)
    assert pre == {'ours': 24, 'opp': 20}
    met['theirs']['by_pre']['20']['recall'] = .92
    assert train_screen.check_holdout(met, pre)


def test_protocol_allows_resource_changes_but_not_new_decision_rules():
    spec = {'pairs': 800, 'alpha': .05, 'slots': 8, 'active': True, 'poll_seconds': 20}
    original = campaign.protocol_hash(spec)
    assert campaign.protocol_hash({**spec, 'slots': 4, 'active': False}) == original
    assert campaign.protocol_hash({**spec, 'pairs': 400}) != original


def test_handoff_never_overwrites_c9_or_an_unrelated_active_campaign(tmp_path):
    p = tmp_path / 'current.json'
    for ident in ('c9', 'another'):
        p.write_text(json.dumps({'id': ident, 'active': True}))
        before = p.read_bytes()
        assert campaign.handoff(tmp_path, {'id': 'c10', 'after': 'c9'}, {}, p) == 'waiting_for_previous_campaign'
        assert p.read_bytes() == before


def test_handoff_is_idempotent_after_a_restart(tmp_path):
    p = tmp_path / 'current.json'
    p.write_text(json.dumps({'id': 'c10', 'active': True}))
    assert campaign.handoff(tmp_path, {'id': 'c10'}, {}, p) == 'live'
    p.write_text(json.dumps({'id': 'c10', 'active': False}))
    assert campaign.handoff(tmp_path, {'id': 'c10'}, {}, p) == 'done'


def test_belief_handoff_waits_for_c10_even_when_ladder_spec_is_inactive(tmp_path):
    p = tmp_path / 'current.json'
    p.write_text(json.dumps({'id': 'c9', 'active': False}))
    previous = tmp_path / 'c10'
    previous.mkdir()
    (previous / 'state.json').write_text(json.dumps({'phase': 'offline_gates'}))
    spec = {'id': 'c12', 'predecessors': ['c9', 'c10'], 'wait_for_campaigns': [str(previous)],
            'candidates': {}}
    before = p.read_bytes()
    assert campaign.handoff(tmp_path, spec, {}, p) == 'waiting_for_previous_campaign'
    assert p.read_bytes() == before
    (previous / 'state.json').write_text(json.dumps({'phase': 'done_no_winner'}))
    assert campaign.handoff(tmp_path, spec, {}, p) == 'done_no_winner'
    # Once C12 owns the live spec, C10's old supervisor status cannot block it.
    p.write_text(json.dumps({'id': 'c12', 'active': True}))
    (previous / 'state.json').write_text(json.dumps({'phase': 'waiting_for_previous_campaign'}))
    assert campaign.handoff(tmp_path, spec, {}, p) == 'live'


def test_nontraining_campaign_starts_preflight_without_fake_training_receipt(tmp_path, monkeypatch):
    jobs = []
    monkeypatch.setattr(campaign, 'verify_protocol', lambda *a: None)
    monkeypatch.setattr(campaign, 'start_job', lambda name, *a: jobs.append(name))
    state = {}
    campaign.tick(tmp_path, {'active': True, 'training_required': False}, state)
    assert state['phase'] == 'preflight' and jobs == ['preflight']
    jobs.clear()
    campaign.tick(tmp_path, {'active': True}, state)
    assert state['phase'] == 'training' and jobs == ['train']


def test_no_offline_winner_does_not_start_a_ladder_test(tmp_path):
    p = tmp_path / 'current.json'
    p.write_text(json.dumps({'id': 'c9', 'active': False}))
    spec = {'id': 'c10', 'after': 'c9', 'candidates': {'preview': {}, 'endgame': {}}}
    before = p.read_bytes()
    assert campaign.handoff(tmp_path, spec, {}, p) == 'done_no_winner'
    assert p.read_bytes() == before


def test_existing_worker_is_adopted_instead_of_duplicated(tmp_path, monkeypatch):
    state = {'jobs': {'train': {'pid': 123, 'attempts': 1}}}
    monkeypatch.setattr(campaign, 'command', lambda pid: f'python /pin/search/campaign.py worker --dir {tmp_path}')
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **kw: pytest.fail('duplicate worker'))
    campaign.start_job('train', tmp_path, {}, state)
    assert state['jobs']['train']['attempts'] == 1


def test_gpu_failure_or_failed_prescreen_cannot_enter_endgame_gate():
    spec = {'candidates': {'endgame': {'endgame_d2': {'alive': 2}}}}
    assert campaign.options_for('endgame', spec, {'useful': False, 'device': 'mps'}) is None
    assert campaign.options_for('endgame', spec, {'useful': True, 'device': 'cpu'}) is None


def test_handoff_creates_one_fresh_union_test_only_after_previous_finishes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import live_deploy
    p = tmp_path / 'current.json'
    p.write_text(json.dumps({'id': 'c9', 'active': False}))
    (tmp_path / 'preview-result.json').write_text(json.dumps({'passed': True}))
    (tmp_path / 'endgame-result.json').write_text(json.dumps({'passed': False}))
    source = tmp_path / 'pin/search'
    source.mkdir(parents=True)
    (source / 'agent.py').write_text('preview_guard')
    monkeypatch.setattr(live_deploy, 'status', lambda: {'dirty': False, 'missing_links': [], 'head': 'new', 'live': str(source.parent)})
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    options = {'preview_guard': {'max_drop': .025}}
    monkeypatch.setattr(campaign, 'options_for', lambda *a: options)
    spec = {'id': 'c10', 'after': 'c9', 'main': str(tmp_path), 'seed': 12, 'source_commit': 'old',
            'candidates': {'preview': options, 'endgame': {}}, 'live': {'max_games': 3000}}
    state = {}
    assert campaign.handoff(tmp_path, spec, state, p) == 'live'
    created = p.read_bytes()
    assert json.loads(created)['factors'] == {'upgrade': options}
    assert campaign.handoff(tmp_path, spec, state, p) == 'live' and p.read_bytes() == created
