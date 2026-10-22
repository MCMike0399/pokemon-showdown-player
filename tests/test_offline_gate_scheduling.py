"""Exercise gate scheduling without spending simulator time on a decided panel."""
import asyncio
import json

from search import agent, value_gate


def fake_arena(monkeypatch):
    started, closed = [], []

    class FakeBrain:
        def __init__(self, *args, **kwargs):
            pass

        def preview_fn(self):
            return None

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.stats = {'fallbacks': 0, 'errors': [], 'decisions': 1, 'sims': 1, 'ms': 1}

        def reseed(self, seed):
            self.seed = seed

        async def close(self):
            closed.append(self.seed)

    async def play(agents, teams, seed, **kwargs):
        started.append(seed)
        await asyncio.sleep(0)  # let the whole old gather queue reach the semaphore
        return {'winner': 'p1', 'turns': 1}

    monkeypatch.setattr(value_gate, 'BrainAgent', FakeBrain)
    monkeypatch.setattr(agent, 'SearchAgent', FakeAgent)
    monkeypatch.setattr(value_gate, 'play_game', play)
    return started, closed


def run(path, concurrency=1, pairs=100, rule='superior'):
    return asyncio.run(value_gate.run_gate({}, {}, path, pairs, concurrency, 123, [], [[]], 1, .05,
                                           log=lambda _: None, rule=rule))


def test_futility_stops_waiting_games_and_resume_does_no_more_work(tmp_path, monkeypatch):
    started, closed = fake_arena(monkeypatch)
    path = tmp_path / 'gate.jsonl'
    result = run(path)
    assert len(started) == 80  # 40 concordant pairs decide futility, not all 200 games
    assert len(closed) == 160
    assert result['complete'] and result['futility_stop'] and not result['passed']
    before = path.read_bytes()
    result2 = run(path)
    assert len(started) == 80 and path.read_bytes() == before
    assert result2 == result


def test_only_already_running_games_can_finish_after_stop(tmp_path, monkeypatch):
    started, closed = fake_arena(monkeypatch)
    path = tmp_path / 'gate.jsonl'
    result = run(path, concurrency=4)
    assert 80 <= len(started) <= 83
    assert len(closed) == 2 * len(started)
    assert result['complete'] and result['futility_stop'] and not result['passed']
    assert len(path.read_text().splitlines()) == len(started)
    rows = list(map(json.loads, path.read_text().splitlines()))
    assert sum(not r.get('post_decision') for r in rows) == result['decision_rows']
    assert value_gate.summarize(rows, 100, .05)['pairs_complete'] == result['pairs_complete']


def test_small_panel_completes_both_arms(tmp_path, monkeypatch):
    started, closed = fake_arena(monkeypatch)
    path = tmp_path / 'gate.jsonl'
    result = run(path, concurrency=3, pairs=6)
    assert len(started) == 12 and len(closed) == 24
    assert result['pairs_complete'] == 6 and not result['futility_stop']
    assert {(r['pair'], r['arm']) for r in map(json.loads, path.read_text().splitlines())} == {
        (i, arm) for i in range(6) for arm in ('cand', 'inc')}


def test_late_results_cannot_reverse_a_futility_decision_on_resume(tmp_path, monkeypatch):
    started, _ = fake_arena(monkeypatch)
    path = tmp_path / 'gate.jsonl'
    rows = []
    for pair in range(42):
        gain = pair % 2 == 0 if pair < 40 else True
        rows.extend([{'pair': pair, 'arm': 'cand', 'score': float(gain)},
                     {'pair': pair, 'arm': 'inc', 'score': float(not gain)}])
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert not value_gate.summarize(rows, 100, .05)['complete']  # naive final tally would resume
    result = run(path, concurrency=4)
    assert not started
    assert result['futility_stop'] and result['pairs_complete'] == 40
    assert result['decision_rows'] == 80 and result['recorded_rows'] == 84


def test_summary_excludes_late_completions_for_promotion_consumers():
    rows = [{'pair': p, 'arm': a, 'score': float((p % 2 == 0) == (a == 'cand'))}
            for p in range(40) for a in ('cand', 'inc')]
    stop = value_gate.summarize(rows, 100, .05)
    rows += [{'pair': p, 'arm': a, 'score': float(a == 'cand'), 'post_decision': True}
             for p in range(40, 100) for a in ('cand', 'inc')]
    assert value_gate.summarize(rows, 100, .05) == stop


def test_pooled_rule_waits_until_planned_rejection_looks(tmp_path, monkeypatch):
    started, _ = fake_arena(monkeypatch)
    result = run(tmp_path / 'gate.jsonl', pairs=120, rule='pooled')
    assert len(started) == 120  # 60 pairs, not the legacy stop at 40
    assert result['pairs_complete'] == 60 and result['futility_stop']


def test_pooled_significance_only_at_the_final_panel():
    rows = [{'pair': p, 'arm': a, 'score': float(a == 'cand')}
            for p in range(80) for a in ('cand', 'inc')]
    for n in (20, 40, 60, 79):
        result = value_gate.summarize(rows[:n * 2], 80, .05, rule='pooled')
        assert result['p_one_sided'] < .05 and not result['complete'] and not result['passed']
    assert value_gate.summarize(rows, 80, .05, rule='pooled')['passed']


def test_pooled_rule_second_rejection_look_and_restart(tmp_path, monkeypatch):
    started, _ = fake_arena(monkeypatch)
    rows = [{'pair': p, 'arm': a, 'score': float((p < 30) == (a == 'cand'))}
            for p in range(60) for a in ('cand', 'inc')]
    assert not value_gate.summarize(rows[:100], 80, .05, rule='pooled')['complete']
    path = tmp_path / 'gate.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = run(path, pairs=80, rule='pooled')
    assert result['futility_stop'] and result['pairs_complete'] == 60 and not started


def test_small_pooled_panel_recovers_its_earlier_first_look(tmp_path, monkeypatch):
    started, _ = fake_arena(monkeypatch)
    path = tmp_path / 'gate.jsonl'
    first = run(path, pairs=16, rule='pooled')
    assert first['pairs_complete'] == 8 and len(started) == 16
    assert run(path, pairs=16, rule='pooled') == first and len(started) == 16


def test_errors_retry_the_same_seed_and_void_without_unbounded_work(tmp_path, monkeypatch):
    _, closed = fake_arena(monkeypatch)
    calls = []

    async def fail(agents, teams, seed, **kwargs):
        calls.append(seed)
        await asyncio.sleep(0)
        raise RuntimeError('simulator failure')

    monkeypatch.setattr(value_gate, 'play_game', fail)
    path = tmp_path / 'gate.jsonl'
    result = run(path, pairs=1)
    assert result['complete'] and result['pairs_void'] == 1 and not result['passed']
    assert len(set(calls)) == 1 and 3 <= len(calls) <= 6
    assert len(closed) == 2 * len(calls)
    before = len(calls)
    assert run(path, pairs=1) == result and len(calls) == before
