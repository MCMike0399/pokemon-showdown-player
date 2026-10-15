"""Performance-rating monitor: ledger parsing, dedupe, segments and the rating math."""
import json
import math

import pytest

from search import elo_monitor as em

ME = 'LadderBot'
CFG = {'worlds': 6, 'engines': 3, 'adaptive_max_ms': 9000.0}


def terminal(room, t, opp_rating, winner, my_rating=1200, side='p2', stringify=False, after=None, omit_rating=False):
    names = {'p1': 'Rival', 'p2': ME} if side == 'p2' else {'p1': ME, 'p2': 'Rival'}
    ratings = {'p1': opp_rating if side == 'p2' else my_rating, 'p2': my_rating if side == 'p2' else opp_rating}
    log = ['|j|☆' + ME, '|gametype|doubles',
           '|player|p1|%s|cynthia|%s' % (names['p1'], ratings['p1']),
           '|player|p2|%s|170|%s' % (names['p2'], ratings['p2']),
           '|player|p2|', '|player|p2|%s|170|' % names['p2']]  # reconnect lines carry no rating
    if omit_rating:
        log = [line for line in log if not line.startswith('|player|p') or line.count('|') < 5 or line.endswith('|')]
    if after:
        log.append("|raw|%s's rating: %d &rarr; <strong>%d</strong><br />(+9 for winning)" % (ME, my_rating, after))
    if winner:
        log.append('|win|' + winner)
    return {'kind': 'terminal', 'time': t, 'result': {'room': room, 'winner': winner},
            'side': side, 'log': str(log) if stringify else log}


def write_run(root, name, events, junk=True):
    path = root / name / 'events.jsonl'
    path.parent.mkdir(parents=True)
    lines = [json.dumps(e) for e in events]
    if junk:
        lines[1:1] = ['not json {', '', '{"kind": "terminal", "result": ']  # unparseable lines are skipped
    path.write_text('\n'.join(lines) + '\n')
    return path


def header(config=CFG, preview=None, team='Rain'):
    out = [{'kind': 'team', 'name': team, 'time': 1}]
    if preview:
        out.append({'kind': 'preview_policy', 'spec': preview, 'time': 1})
    out.append({'kind': 'search_agent', 'time': 1, **config})
    return out


@pytest.fixture(autouse=True)
def fixed_name(monkeypatch):
    monkeypatch.setattr(em, 'my_name', lambda root=em.ROOT: ME)


# ------------------------------------------------------------------ math

def games(opps, scores):
    return [{'opp_rating': o, 'score': s} for o, s in zip(opps, scores)]


def test_equal_ratings_half_score_gives_opponent_rating():
    perf = em.performance(games([1300] * 10, [1, 0] * 5))
    assert perf['rating'] == pytest.approx(1300, abs=0.01)
    assert perf['n'] == 10 and perf['win_rate'] == 50.0 and perf['clamped'] is None
    assert perf['expected'] == pytest.approx(perf['score'], abs=1e-6)


def test_known_asymmetric_case():
    # 76% against 1200s is +400*log10(0.76/0.24) = about +200.
    perf = em.performance(games([1200] * 100, [1] * 76 + [0] * 24))
    assert perf['rating'] == pytest.approx(1200 + 400 * math.log10(0.76 / 0.24), abs=0.01)
    # The same score against stronger opposition is a higher rating.
    assert em.performance(games([1500] * 100, [1] * 76 + [0] * 24))['rating'] == pytest.approx(
        perf['rating'] + 300, abs=0.01)


def test_mixed_opponents_solve_expected_equals_score():
    opps = [1000, 1100, 1250, 1400, 1500, 1600]
    scores = [1, 1, 0.5, 1, 0, 0]
    perf = em.performance(games(opps, scores))
    assert em.expected(perf['rating'], opps) == pytest.approx(sum(scores), abs=1e-6)


def test_interval_matches_formula():
    perf = em.performance(games([1200] * 100, [1] * 50 + [0] * 50))
    # p = 0.5 everywhere: variance 25, SE = 173.7/5 = 34.7, 95% half-width about 68.
    assert perf['ci'] == pytest.approx(1.96 * (400 / math.log(10)) / 5, abs=0.01)
    assert perf['ci'] == pytest.approx(68.1, abs=0.2)


def test_all_wins_and_all_losses_are_clamped_and_flagged():
    wins = em.performance(games([1200] * 5, [1] * 5))
    assert wins['rating'] == em.HIGH and wins['clamped'] == 'wins' and wins['ci'] is None
    losses = em.performance(games([1200] * 5, [0] * 5))
    assert losses['rating'] == em.LOW and losses['clamped'] == 'losses'


def test_empty_and_unrated_games():
    assert em.performance([])['rating'] is None
    assert em.performance([{'opp_rating': None, 'score': 1}])['n'] == 0


def test_rolling_series_downsamples_and_keeps_last_point():
    rated = games([1200 + (i % 7) * 20 for i in range(700)], [i % 2 for i in range(700)])
    series = em.rolling_series(rated)
    assert 2 <= len(series) <= em.MAX_SERIES_POINTS
    assert series[0][0] == 49 and series[-1][0] == 699
    assert em.rolling_series(rated[:49]) == []


# --------------------------------------------------------------- ledgers

def test_parsing_ratings_score_and_stringified_log(tmp_path):
    write_run(tmp_path, '20261001T000000Z', header() + [
        terminal('battle-a-1', 100, 1300, ME, my_rating=1250, after=1262),
        terminal('battle-a-2', 200, 1180, 'Rival', my_rating=1262, side='p1', stringify=True),
    ])
    games_ = em.load_games(tmp_path)
    assert [g['room'] for g in games_] == ['battle-a-1', 'battle-a-2']
    first, second = games_
    assert (first['my_rating'], first['opp_rating'], first['score'], first['my_rating_after']) == (1250, 1300, 1.0, 1262)
    assert (second['my_rating'], second['opp_rating'], second['score']) == (1262, 1180, 0.0)
    assert first['policy_key'] == second['policy_key'] and first['run'] == '20261001T000000Z'


def test_tie_and_no_winner_score_half_and_missing_rating_is_unrated(tmp_path):
    odd = terminal('battle-a-3', 300, 1200, None, omit_rating=True)
    write_run(tmp_path, '20261001T000000Z', header() + [terminal('battle-a-1', 100, 1200, None), odd,
                                                       {'kind': 'terminal', 'time': 5, 'result': {'room': 'x'}, 'log': 'garbage'}])
    games_ = em.load_games(tmp_path)
    assert [g['score'] for g in games_] == [0.5, 0.5]
    assert games_[1]['opp_rating'] is None
    assert em.performance(games_)['n'] == 1  # unrated games never enter the rating


def test_resumed_room_deduped_keeping_terminal_with_log(tmp_path):
    write_run(tmp_path, '20261001T000000Z', header() + [
        {'kind': 'terminal', 'time': 100, 'result': {'room': 'battle-r-1', 'winner': ME}, 'side': 'p2', 'log': []},
        terminal('battle-r-2', 150, 1200, 'Rival')])
    write_run(tmp_path, '20261001T010000Z', header() + [terminal('battle-r-1', 120, 1400, ME)])
    games_ = em.load_games(tmp_path)
    assert sorted(g['room'] for g in games_) == ['battle-r-1', 'battle-r-2']
    resumed = next(g for g in games_ if g['room'] == 'battle-r-1')
    assert resumed['opp_rating'] == 1400 and resumed['run'] == '20261001T010000Z'


def test_my_name_reads_env_without_password(tmp_path, monkeypatch):
    monkeypatch.undo()
    (tmp_path / '.env').write_text('PS_PASSWORD=secret\nPS_USERNAME="Someone"\n')
    assert em.my_name(tmp_path) == 'Someone'
    assert em.my_name(tmp_path / 'missing') == em.DEFAULT_NAME


# -------------------------------------------------------------- segments

def test_segments_by_policy_key(tmp_path):
    same = header(CFG)
    write_run(tmp_path, '20261001T000000Z', same + [terminal('b-1', 10, 1200, ME), terminal('b-2', 20, 1200, 'Rival')])
    write_run(tmp_path, '20261001T020000Z', header(CFG) + [terminal('b-3', 30, 1200, ME)])  # same key: same segment
    write_run(tmp_path, '20261002T000000Z', header(CFG, preview='pvalue:data/models/pvalue-abc.json') + [
        terminal('b-4', 40, 1300, ME), terminal('b-5', 50, 1300, ME)])
    write_run(tmp_path, '20261003T000000Z', header({**CFG, 'adaptive_max_ms': 6000.0}, preview='pvalue:data/models/pvalue-abc.json') + [
        terminal('b-6', 60, 1300, 'Rival')])
    runs = em.load_runs(tmp_path)
    segs = em.segments(em.games_from_runs(runs), runs)
    assert [len(s['games']) for s in segs] == [3, 2, 1]
    assert [s['runs'] for s in segs] == [['20261001T000000Z', '20261001T020000Z'], ['20261002T000000Z'], ['20261003T000000Z']]
    assert 'pvalue-abc' in segs[1]['label'] and 'preview' in segs[1]['changes'][0]
    assert 'adaptive_max_ms 9000.0 -> 6000.0' in segs[2]['changes']
    assert segs[0]['first'] == 10 and segs[0]['last'] == 30


def test_policy_key_ignores_volatile_fields_and_null_values():
    a = em.policy_key({'worlds': 6, 'time': 1, 'kind': 'search_agent', 'opp_model': None}, None, 'T')
    b = em.policy_key({'worlds': 6, 'time': 999, 'kind': 'search_agent'}, None, 'T')
    assert a == b
    assert a != em.policy_key({'worlds': 6}, 'pvalue:x', 'T')
    assert a != em.policy_key({'worlds': 6}, None, 'Other')


def test_report_and_windows(tmp_path):
    events = header()
    for i in range(120):
        events.append(terminal('r-%03d' % i, 1000 + i, 1250, ME if i % 2 else 'Rival', my_rating=1200 + i, after=1200 + i + 1))
    write_run(tmp_path, '20261001T000000Z', events)
    data = em.report(tmp_path)
    assert data['games'] == 120 and data['rating'] == 1320 and data['rating_change_50'] == 1320 - 1270
    assert data['windows']['50']['n'] == 50 and data['windows']['100']['n'] == 100 and data['windows']['200']['n'] == 120
    assert data['windows']['100']['rating'] == pytest.approx(1250, abs=0.01)
    assert len(data['segments']) == 1 and data['segments'][0]['current'] is True
    assert data['series'][-1][0] == 119
    json.dumps(data)
    assert 'Segments' in em.format_table(data)


def test_report_cached_follows_ledger_changes(tmp_path):
    path = write_run(tmp_path, '20261001T000000Z', header() + [terminal('c-1', 10, 1200, ME)])
    first = em.report_cached(tmp_path)
    assert em.report_cached(tmp_path) is first  # unchanged ledgers: memoised
    with path.open('a') as handle:
        handle.write(json.dumps(terminal('c-2', 20, 1200, 'Rival')) + '\n')
        handle.write('{"kind": "terminal", "res')  # half-written trailing line is left for later
    second = em.report_cached(tmp_path)
    assert second['games'] == 2
    with path.open('a') as handle:
        handle.write('\n')
    assert em.report_cached(tmp_path)['games'] == 2


def test_cli_json(tmp_path, capsys):
    write_run(tmp_path, '20261001T000000Z', header() + [terminal('c-1', 10, 1200, ME)])
    assert em.main(['--runs', str(tmp_path), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['games'] == 1
