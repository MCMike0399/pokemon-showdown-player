"""Live factorial A/B (search/ab_test.py, scripts/ab_watch.py): assignment, SPRT, freezing, promotion."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from search import ab_test as ab
from search import elo_monitor as em

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ab_watch  # noqa: E402

FACTORS = {'protect': {'protect_d2': {'behind': 99.0}}, 'boost': {'eval_terms': {'boost': 0.5}},
           'bench': {'bench_prior': 'back_rate'}}


def spec(**kw):
    return ab.validate({'id': 'cx', 'active': True, 'seed': 3, 'factors': FACTORS, **kw})


def test_validate_rejects_overlapping_options_and_bad_frozen():
    with pytest.raises(ValueError):
        ab.validate({'id': 'cx', 'factors': {'a': {'k': 1}, 'b': {'k': 2}}})
    with pytest.raises(ValueError):
        ab.validate({'id': 'cx', 'factors': {'a': {'k': 1}}, 'frozen': {'zz': True}})
    assert spec()['elo1'] == 40.0  # defaults filled


def test_load_spec_inactive_or_missing_is_none(tmp_path):
    p = tmp_path / 'current.json'
    assert ab.load_spec(p) is None
    p.write_text(json.dumps({'id': 'cx', 'active': False, 'factors': FACTORS}))
    assert ab.load_spec(p) is None
    p.write_text(json.dumps({'id': 'cx', 'active': True, 'factors': FACTORS}))
    assert ab.load_spec(p)['factors'] == FACTORS


def test_blocks_play_every_cell_once_and_frozen_factors_stay_fixed():
    s = spec()
    for block in range(5):
        cells = [tuple(ab.cell_at(s, block * 8 + i).values()) for i in range(8)]
        assert len(set(cells)) == 8
    s = spec(frozen={'boost': True})
    cells = [ab.cell_at(s, i) for i in range(40)]
    assert all(c['boost'] for c in cells)
    assert Counter((c['protect'], c['bench']) for c in cells) == {k: 10 for k in
                                                                  [(a, b) for a in (0, 1) for b in (0, 1)]}
    assert ab.overrides(s, {'protect': True, 'boost': True, 'bench': False}) == {
        'protect_d2': {'behind': 99.0}, 'eval_terms': {'boost': 0.5}}


def test_next_arm_counts_the_phase_and_a_resumed_room_keeps_its_arm(tmp_path):
    p = tmp_path / 'current.json'
    s = spec()
    seen = []
    for i in range(8):
        arm = ab.next_arm(s, p)
        assert arm['n'] == i
        ab.record_arm(s, f'room-{i}', arm, p)
        seen.append(tuple(arm['cell'].values()))
    assert len(set(seen)) == 8
    back = ab.arm_for_room(s, 'room-3', p)
    assert tuple(back['cell'].values()) == seen[3] and back['overrides'] == ab.overrides(s, back['cell'])
    assert ab.arm_for_room(s, 'room-x', p) is None
    # A new phase (one factor frozen) starts its own block count.
    assert ab.next_arm(spec(frozen={'bench': False}), p)['n'] == 0


def test_sprt_accepts_h1_for_a_real_gain_and_h0_for_none():
    rng = random.Random(1)
    on = [float(rng.random() < 0.58) for _ in range(1500)]
    off = [float(rng.random() < 0.50) for _ in range(1500)]
    lo, hi = ab.bounds(spec())
    assert ab.sprt(on, off, 0, 40)['llr'] >= hi
    same = [float(rng.random() < 0.5) for _ in range(1500)]
    assert ab.sprt(same, off, 0, 40)['llr'] <= lo or ab.sprt(same, off, 0, 40)['d'] < 0.03
    assert abs(ab.score_elo(ab.elo_score(40)) - 40) < 1e-6


def rows_for(s, n, p_on, rng):
    rows = []
    for i in range(n):
        cell = ab.cell_at(s, i)
        p = p_on if cell['protect'] else 0.5
        rows.append({'room': f'r{i}', 'score': float(rng.random() < p), 'opp_rating': 1300, 'cell': cell,
                     'random': ab.randomised(s), 'time': i})
    return rows


def test_analyse_decides_only_after_min_games():
    s = spec(min_games=300)
    st = ab.analyse(s, rows_for(s, 120, 0.9, random.Random(2)))
    assert st['factors']['protect']['decision'] is None
    st = ab.analyse(s, rows_for(s, 2000, 0.6, random.Random(2)))
    assert st['factors']['protect']['decision'] == 'H1'
    assert st['factors']['protect']['perf_on'] > st['factors']['protect']['perf_off']
    assert st['factors']['bench']['decision'] in ('H0', None)
    assert sum(c['games'] for c in st['cells'].values()) == 2000


def test_elo_monitor_puts_ab_games_in_their_own_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(em, 'my_name', lambda root=em.ROOT: 'Bot')
    log = ['|player|p1|Rival|1|1300', '|player|p2|Bot|2|1300', '|win|Bot']
    events = [{'kind': 'team', 'name': 'Rain', 'time': 1}, {'kind': 'search_agent', 'time': 1, 'worlds': 6},
              {'kind': 'terminal', 'time': 5, 'side': 'p2', 'result': {'room': 'a', 'winner': 'Bot'}, 'log': log},
              {'kind': 'ab_arm', 'room': 'b', 'id': 'cx', 'cell': {}, 'random': [], 'n': 0, 'overrides': {}},
              {'kind': 'terminal', 'time': 9, 'side': 'p2', 'result': {'room': 'b', 'winner': 'Bot'}, 'log': log}]
    d = tmp_path / 'runs' / 'r1'
    d.mkdir(parents=True)
    (d / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
    games = em.load_games(tmp_path / 'runs')
    assert games[0]['policy_key'] + '+ab-cx' == games[1]['policy_key']
    segs = em.segments(games, em.load_runs(tmp_path / 'runs'))
    assert len(segs) == 2 and 'live A/B cx' in segs[1]['label']


def test_watch_freezes_decided_factors_then_promotes_the_winners(tmp_path, monkeypatch):
    p = tmp_path / 'current.json'
    raw = {'id': 'cx', 'active': True, 'seed': 3, 'factors': FACTORS, 'min_games': 300, 'max_games': 2000}
    p.write_text(json.dumps(raw))
    out = tmp_path / 'camp'
    out.mkdir()
    rows = rows_for(ab.validate(raw), 2400, 0.65, random.Random(5))
    monkeypatch.setattr(ab, 'status', lambda path, runs=None: {
        **ab.analyse(ab.validate(json.loads(Path(path).read_text())), rows), 'active': True})
    calls = []

    def fake_promote(gates, options, label, evidence, dry, message=None):
        calls.append((options, message))
    assert ab_watch.step(p, out, 'campaign cx', promote=fake_promote) == 'promoted'
    spec_now = json.loads(p.read_text())
    # protect wins (H1); the null factors stop at H0 or at max_games and stay off.
    assert spec_now['frozen'] == {'protect': True, 'boost': False, 'bench': False}
    assert spec_now['active'] is False
    assert calls == [({'protect_d2': {'behind': 99.0}}, calls[0][1])] and 'protect' in calls[0][1]
    assert json.loads((out / 'promote.json').read_text())['winners'] == ['protect']
    assert ab_watch.step(p, out, 'campaign cx', promote=fake_promote) == 'waiting'


def test_watch_without_a_winner_only_deactivates(tmp_path, monkeypatch):
    p = tmp_path / 'current.json'
    raw = {'id': 'cx', 'active': True, 'seed': 3, 'factors': FACTORS, 'min_games': 300, 'max_games': 600}
    p.write_text(json.dumps(raw))
    out = tmp_path / 'camp'
    out.mkdir()
    rows = rows_for(ab.validate(raw), 700, 0.5, random.Random(6))
    monkeypatch.setattr(ab, 'status', lambda path, runs=None: {
        **ab.analyse(ab.validate(json.loads(Path(path).read_text())), rows), 'active': True})
    called = []
    assert ab_watch.step(p, out, 'campaign cx', promote=lambda *a, **k: called.append(a)) == 'finished'
    assert not called and json.loads(p.read_text())['active'] is False
