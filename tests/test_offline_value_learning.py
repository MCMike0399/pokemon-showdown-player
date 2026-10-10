"""Offline checks for the search value-learning loop (search/value_*.py, train_value, selfplay, engine)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from search import value_gate, value_service, value_store
from search.ledger_positions import _prefix, load_games
from search.train_value import fit_hand, load_rows

ROOT = Path(__file__).resolve().parents[1]
FMT = 'gen9championsvgc2026regmc'
NODE = shutil.which('node') is not None and (ROOT / 'node_modules' / 'pokemon-showdown' / 'dist').exists()
needs_sim = pytest.mark.skipif(not NODE, reason='official simulator is not built')

SET = lambda species, item, ability, moves, nature='Modest', evs=None: {
    'species': species, 'item': item, 'ability': ability, 'moves': moves, 'nature': nature, 'level': 50,
    'evs': evs or {'hp': 2, 'spa': 32, 'spe': 32}}
OURS = [SET('Politoed', 'Mystic Water', 'Drizzle', ['muddywater', 'icebeam', 'weatherball', 'protect']),
        SET('Archaludon', 'Leftovers', 'Stamina', ['dragonpulse', 'electroshot', 'snarl', 'protect']),
        SET('Incineroar', 'Sitrus Berry', 'Intimidate', ['fakeout', 'flareblitz', 'partingshot', 'darkestlariat'],
            'Adamant', {'hp': 32, 'atk': 32, 'spd': 2}),
        SET('Farigiraf', 'Sitrus Berry', 'Armor Tail', ['psychic', 'trickroom', 'helpinghand', 'protect'])]
THEIRS = [SET('Garchomp', 'Life Orb', 'Rough Skin', ['earthquake', 'dragonclaw', 'rockslide', 'protect'],
              'Jolly', {'hp': 2, 'atk': 32, 'spe': 32}),
          SET('Sylveon', 'Leftovers', 'Pixilate', ['hypervoice', 'protect', 'moonblast', 'yawn']),
          SET('Rillaboom', 'Miracle Seed', 'Grassy Surge', ['fakeout', 'grassyglide', 'woodhammer', 'protect'],
              'Adamant', {'hp': 32, 'atk': 32, 'spd': 2}),
          SET('Kingambit', 'Black Glasses', 'Defiant', ['kowtowcleave', 'suckerpunch', 'ironhead', 'protect'],
              'Adamant', {'hp': 32, 'atk': 32, 'spd': 2})]
WORLD = {'p1': [{'set': s, 'state': {'turn': 3}} for s in OURS], 'p2': [{'set': s, 'state': {'turn': 3}} for s in THEIRS]}
FIELD = {'turn': 3, 'weather': {'id': 'raindance', 'duration': 2}, 'pseudo': {}, 'sides': {'p1': {}, 'p2': {}},
         'mega_used': {'p1': False, 'p2': False}}
CHOICES = ['move muddywater, move dragonpulse 1', 'move protect, move protect', 'move icebeam 1, move dragonpulse 1',
           'move weatherball 2, move snarl']


def engine(*messages):
    proc = subprocess.run(['node', str(ROOT / 'search' / 'engine.cjs')], capture_output=True, text=True, timeout=300,
                          input=''.join(json.dumps({'id': i, **m}) + '\n' for i, m in enumerate(messages)))
    return [json.loads(line) for line in proc.stdout.splitlines()]


@needs_sim
def test_crn_makes_action_values_independent_of_row_order():
    base = {'type': 'search', 'format': FMT, 'worlds': [WORLD], 'field': FIELD}
    on = engine({**base, 'our_choices': CHOICES, 'crn': True}, {**base, 'our_choices': CHOICES[::-1], 'crn': True})
    off = engine({**base, 'our_choices': CHOICES}, {**base, 'our_choices': CHOICES[::-1]})
    a, b = (r['results'][0]['values'] for r in on)
    assert a == pytest.approx(b)  # same reply -> same chance stream, whatever our row index
    c, d = (r['results'][0]['values'] for r in off)
    assert c != pytest.approx(d)  # legacy seeding depends on the row
    root = on[0]['results'][0]['root']
    assert np.isfinite(root['eq']) and np.isfinite(root['hand'])


@needs_sim
def test_features_and_hand_unit_value_net(tmp_path):
    (feat,) = engine({'type': 'features', 'format': FMT, 'worlds': [WORLD, WORLD], 'field': FIELD})
    x = feat['results'][0]['x']
    assert len(x) == 146 and feat['results'][0]['x'] == feat['results'][1]['x']
    # A constant net (logit 2.0) with hand calibration a=.5, b=1 is worth (2-1)/.5 = 2 in hand units.
    net = {'layers': [{'W': [[0.0] * 146], 'b': [2.0]}], 'mean': [0.0] * 146, 'std': [1.0] * 146,
           'hand': {'a': 0.5, 'b': 1.0}}
    path = tmp_path / 'const.json'
    path.write_text(json.dumps(net))
    (out,) = engine({'type': 'search', 'format': FMT, 'worlds': [WORLD], 'field': FIELD, 'our_choices': CHOICES[:2],
                     'value_path': str(path), 'value_beta': 1.0})
    values = out['results'][0]['values']
    assert all(v == pytest.approx(2.0) for v in values.values())


@needs_sim
def test_selfplay_rows_carry_game_side_and_targets(tmp_path):
    out = tmp_path / 'sp.jsonl'
    team = tmp_path / 'focus.json'
    team.write_text(json.dumps(OURS + [SET('Golisopod', 'Choice Band', 'Emergency Exit',
                                           ['firstimpression', 'leechlife', 'aquajet', 'knockoff'], 'Adamant')] +
                               [SET('Garchomp', 'Life Orb', 'Rough Skin', ['earthquake', 'dragonclaw', 'rockslide', 'protect'])]))
    pool = tmp_path / 'pool.json'
    pool.write_text(json.dumps([{'sets': THEIRS + [OURS[0], OURS[1]]}]))
    subprocess.run(['node', str(ROOT / 'search' / 'selfplay.cjs'), str(pool), str(team), '1', '5', str(out),
                    '--focus-prob', '1', '--tag', 't'], check=True, capture_output=True, timeout=300)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and len(rows) % 2 == 0
    assert {r['g'] for r in rows} == {'t-5-0'}
    assert sum(r['f'] for r in rows) == len(rows) // 2  # exactly one side is the focus team
    for a, b in zip(rows[::2], rows[1::2]):
        assert (a['s'], b['s']) == (0, 1) and a['t'] == b['t']
        assert a['y'] + b['y'] == pytest.approx(1) and a['h'] == pytest.approx(-b['h'])


def _write_rows(path, games, start=0, live=False, t0=0.0):
    rng = np.random.default_rng(start)
    with path.open('w') as fh:
        for k in range(games):
            y = float(rng.integers(0, 2))
            for t in range(6):
                h = (2 * y - 1) * 2 + rng.normal()
                row = {'x': [h, rng.normal(), 1.0], 'y': y, 'h': h, 't': t, 'g': f'{path.stem}-{start + k}'}
                if live:
                    row['time'] = t0 + k
                fh.write(json.dumps(row) + '\n')


def test_training_split_is_by_game_and_live_test_is_newest(tmp_path):
    sp = tmp_path / 'sp'
    sp.mkdir()
    _write_rows(sp / 'a.jsonl', 300)
    live = tmp_path / 'live'
    live.mkdir()
    for k in range(10):
        _write_rows(live / f'live-room{k}.jsonl', 1, start=1000 + k, live=True, t0=100.0 * k)
    data, manifest, info = load_rows([str(sp / '*.jsonl')], live, val_frac=0.2, live_test=0.3)
    train_g, val_g, test_g = (set(data[k][3]) for k in ('train', 'val', 'live_test'))
    assert train_g and val_g and not train_g & val_g
    assert info['live_test_games'] == 3
    assert test_g == {'live-room7-1007', 'live-room8-1008', 'live-room9-1009'}  # the newest games
    assert not test_g & train_g
    a, b = fit_hand(data['train'][2], data['train'][1])
    assert a > 0


def test_trained_candidate_is_immutable_and_pointer_verifies(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from search import train_value
    sp = tmp_path / 'sp'
    sp.mkdir()
    _write_rows(sp / 'a.jsonl', 400)
    out = tmp_path / 'cand'
    report = train_value.main(['--selfplay', str(sp / '*.jsonl'), '--live', str(tmp_path / 'none'), '--out', str(out),
                               '--device', 'cpu', '--epochs', '3', '--hidden', '16'])
    path = Path(report['path'])
    assert path.exists() and value_store.weights_sha(path) == report['sha256']
    assert report['recommended_beta'] in (0.0, 0.25, 0.5, 0.75, 1.0)
    pointer = tmp_path / 'current.json'
    assert value_store.current_value(pointer)['path'] is None  # no pointer: hand evaluation
    value_store.promote(path, 0.5, {'test': True}, pointer)
    assert value_store.current_value(pointer) == {'path': str(path.resolve()), 'beta': 0.5, 'sha256': report['sha256']}
    data = json.loads(path.read_text())
    data['layers'][0]['b'][0] += 1.0  # tamper: the pointer must refuse it
    path.write_text(json.dumps(data))
    assert value_store.current_value(pointer)['path'] is None


def test_shard_cache_round_trip(tmp_path):
    path = tmp_path / 's.jsonl'
    _write_rows(path, 5)
    x0, y0, h0, g0 = value_store.load_shard(path)
    assert value_store.pack_shard(path).exists()
    x1, y1, h1, g1 = value_store.load_shard(path)
    assert np.allclose(x0, x1) and np.array_equal(y0, y1) and g0 == g1
    with path.open('a') as fh:  # appended shard: the stale cache is ignored
        fh.write(json.dumps({'x': [0, 0, 0], 'y': 1, 'h': 0, 'g': 'new'}) + '\n')
    assert len(value_store.load_shard(path)[1]) == len(y0) + 1


def test_gate_summary_needs_all_pairs_and_a_significant_sign_test():
    rows = [{'pair': i, 'arm': 'cand', 'win': True} for i in range(12)]
    rows += [{'pair': i, 'arm': 'inc', 'win': i >= 11} for i in range(12)]
    s = value_gate.summarize(rows, 12, 0.05)
    assert (s['gains'], s['losses']) == (11, 0) and s['complete'] and s['passed']
    assert value_gate.summarize(rows, 13, 0.05)['passed'] is False  # unfinished panels never pass
    assert value_gate.sign_test(5, 5) > 0.5


def test_ledger_reader_keeps_search_turns_with_outcomes(tmp_path):
    run = tmp_path / 'runs' / 'r1'
    run.mkdir(parents=True)
    log = ['|player|p1|Me|1|1200', '|start', '|turn|1', '|move|p1a: X|Tackle|p2a: Y', '|turn|2', '|win|Me']
    events = [
        {'kind': 'decision', 'room': 'battle-1', 'time': 1, 'state': {'turn': 1},
         'request': {'rqid': 3, 'active': [{}], 'side': {'id': 'p1', 'name': 'Me'}},
         'decision': {'choice': 'move 1', 'decider': 'search_decider'}},
        {'kind': 'decision', 'room': 'battle-1', 'time': 2, 'state': {'turn': 2},
         'request': {'rqid': 4, 'active': [{}], 'side': {'id': 'p1', 'name': 'Me'}},
         'decision': {'choice': 'move 2'}},  # model decision: not a search turn
        {'kind': 'terminal', 'result': {'room': 'battle-1', 'winner': 'Me'}, 'log': repr(log)},
        {'kind': 'decision', 'room': 'battle-2', 'time': 3, 'state': {'turn': 1},
         'request': {'rqid': 9, 'active': [{}], 'side': {'id': 'p1', 'name': 'Me'}},
         'decision': {'choice': 'move 1', 'decider': 'search_decider'}},  # unfinished: no terminal
    ]
    (run / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    games = load_games(tmp_path / 'runs')
    assert list(games) == ['battle-1'] and len(games['battle-1']['decisions']) == 1
    assert _prefix(log, 1) == log[:3] and _prefix(log, 7) is None


def test_service_gate_state_machine_promotes_only_after_confirmation(tmp_path, monkeypatch):
    vroot = tmp_path / 'value'
    monkeypatch.setattr(value_service, 'VALUE_ROOT', vroot)
    monkeypatch.setattr(value_service, 'POINTER', vroot / 'current.json')
    pointer = vroot / 'current.json'
    monkeypatch.setattr(value_store, 'POINTER', pointer)
    monkeypatch.setattr(value_service, 'current_value', lambda: value_store.current_value(pointer))
    monkeypatch.setattr(value_service, 'promote', lambda c, b, e: value_store.promote(c, b, e, pointer))
    svc = value_service.Service(tmp_path)
    cand = vroot / 'candidates' / 'value-x.json'
    cand.write_text(json.dumps({'layers': [], 'mean': [], 'std': []}))
    sha = value_store.weights_sha(cand)
    svc.state['candidates'] = [{'path': str(cand), 'sha256': sha, 'beta': 0.5, 'status': 'gating:dev',
                                'val': {}, 'live_test': {}}]
    report = tmp_path / 'r.json'

    def finish(passed):
        report.write_text(json.dumps({'complete': True, 'passed': passed, 'gains': 9, 'losses': 1, 'p_one_sided': .01}))
        svc.state['gate']['report'] = str(report)
        svc.finish_gate(0, value_service.DEFAULTS)

    svc.state['gate'] = {'sha256': sha, 'stage': 'dev', 'incumbent': {'path': None, 'beta': 0.0, 'sha256': None}}
    finish(True)
    assert svc.state['gate']['stage'] == 'confirm' and not pointer.exists()
    finish(True)
    assert svc.state['candidates'][0]['status'] == 'promoted' and 'gate' not in svc.state
    assert value_store.current_value(pointer)['sha256'] == sha
    # A failed confirmation never promotes.
    svc.state['candidates'][0]['status'] = 'trained'
    svc.state['gate'] = {'sha256': sha, 'stage': 'confirm', 'incumbent': value_store.current_value(pointer)}
    finish(False)
    assert svc.state['candidates'][0]['status'] == 'rejected at confirm gate'


class _FakeEngine:
    """Returns fixed per-choice values; counts worlds requested."""

    def __init__(self, values):
        self.values, self.worlds = values, 0

    async def call(self, msg, timeout=120):
        self.worlds += len(msg['worlds'])
        return {'results': [{'values': dict(self.values), 'sims': 1, 'n_ours': 2, 'n_theirs': 1,
                             'root': {'eq': 0.0, 'hand': 0.0}} for _ in msg['worlds']]}

    async def close(self):
        pass


def _move_ctx():
    from battle_state import legal_choices
    from tests.test_offline_search import LOG
    mon = lambda name, species, active: {'ident': f'p1: {name}', 'details': f'{species}, L50, M',
                                          'condition': '167/167', 'active': active, 'moves': ['protect', 'muddywater'],
                                          'item': '', 'ability': 'drizzle'}
    request = {'rqid': 5, 'active': [{'moves': [{'move': 'Protect', 'id': 'protect', 'target': 'self'},
                                                {'move': 'Muddy Water', 'id': 'muddywater', 'target': 'allAdjacentFoes'}]}],
               'side': {'id': 'p1', 'name': 'Me', 'pokemon': [mon('Politoed', 'Politoed', True)]}}
    choices = legal_choices(request)
    return {'request': request, 'choices': choices, 'public_log': LOG,
            'team_sets': [{'species': 'Politoed', 'item': 'Mystic Water', 'ability': 'Drizzle',
                           'moves': ['protect', 'muddywater'], 'nature': 'Modest', 'level': 50}]}, choices


@pytest.mark.parametrize('margin,extra', [(0.1, 2), (5.0, 0)])
def test_close_decisions_get_extra_worlds(monkeypatch, margin, extra):
    import asyncio
    from search import agent as agent_mod
    monkeypatch.setattr(agent_mod, 'sample_bench', lambda *a, **k: [])
    monkeypatch.setattr(agent_mod, 'sample_set', lambda rng, species, observed, fmt, cutoff: {
        'species': species, 'item': '', 'ability': '', 'moves': ['tackle'], 'nature': 'Hardy', 'level': 50})
    ctx, choices = _move_ctx()
    agent = agent_mod.SearchAgent(worlds=3, engines=1, adaptive_margin=1.0, adaptive_worlds=2)
    sims = [agent._to_sim(c, ctx['request']['active'], agent_mod.Tracker('p1').feed(ctx['public_log'])) for c in choices]
    fake = _FakeEngine({sims[0]: 1.0, sims[1]: 1.0 - margin})
    agent.engines = [fake]
    choice = asyncio.run(agent.decide(ctx))
    assert choice == choices[0]
    assert fake.worlds == 3 + extra and agent.last_trace['extra_worlds'] == extra
    assert agent.last_trace['worlds'] == 3 + extra and agent.last_trace['ms'] >= 0


def test_gate_terminates_on_ties_unfinished_games_and_repeated_errors():
    rows = [{'pair': 0, 'arm': 'cand', 'score': 0.5, 'win': False}, {'pair': 0, 'arm': 'inc', 'score': 0.5, 'win': False},
            {'pair': 1, 'arm': 'cand', 'score': 1.0, 'win': True}, {'pair': 1, 'arm': 'inc', 'score': 0.5, 'win': False}]
    rows += [{'pair': 2, 'arm': 'cand', 'win': None, 'error': 'boom'}] * value_gate.MAX_ATTEMPTS
    rows += [{'pair': 2, 'arm': 'inc', 'score': 1.0, 'win': True}]
    s = value_gate.summarize(rows, 3, 0.05)
    assert s['complete'] and s['pairs_complete'] == 2 and s['pairs_void'] == 1
    assert (s['gains'], s['losses']) == (1, 0)  # a tie is concordant; a win over a tie is a gain
    assert not value_gate.summarize(rows[:-2], 3, 0.05)['complete']  # two errors: still retrying


def test_value_lanes_take_extra_slots_before_ppo_slots(tmp_path):
    slots = value_service.Slots(tmp_path)
    first = slots.acquire(6, 2)
    try:
        assert sorted(Path(h.name).name for h in first) == ['4', '5']
    finally:
        for h in first:
            h.close()


def test_live_priority_pauses_and_resumes_background_groups(tmp_path, monkeypatch):
    import asyncio
    import os
    import time as _time
    script = tmp_path / 'search' / 'spin.py'  # the leader's command must contain "search/"
    script.parent.mkdir()
    script.write_text('import time\nwhile True: time.sleep(0.05)\n')
    proc = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    monkeypatch.setattr(value_store, 'BACKGROUND_PGIDS', tmp_path / 'pgids.json')
    monkeypatch.setattr(value_store, 'PAUSE_MARKER', tmp_path / 'deciding')
    (tmp_path / 'pgids.json').write_text(json.dumps([proc.pid, other.pid]))
    def state(pid):
        stat = Path('/proc') / str(pid) / 'stat'
        if stat.exists():
            return stat.read_text().rsplit(')', 1)[1].split()[0]
        return subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    try:
        async def decide():
            async with value_store.LivePriority() as lp:
                _time.sleep(0.3)
                return lp.paused, state(proc.pid), state(other.pid), (tmp_path / 'deciding').exists()
        paused, during, unrelated, marker = asyncio.run(decide())
        assert paused == 1 and during.startswith('T') and not unrelated.startswith('T') and marker
        _time.sleep(0.2)
        assert not state(proc.pid).startswith('T') and not (tmp_path / 'deciding').exists()
        # Failsafe: a stale marker (runner died mid-decision) resumes everything.
        import signal
        os.killpg(proc.pid, signal.SIGSTOP)
        _time.sleep(0.2)
        assert state(proc.pid).startswith('T')
        (tmp_path / 'deciding').write_text(str(_time.time() - 600))
        assert value_store.resume_if_stale()
        _time.sleep(0.2)
        assert not state(proc.pid).startswith('T')
    finally:
        proc.kill(); other.kill()


@needs_sim
def test_engine_timeout_restarts_instead_of_reading_a_stale_answer():
    import asyncio
    from search.agent import Engine

    async def run():
        e = Engine()
        slow = {'type': 'search', 'format': FMT, 'worlds': [WORLD] * 3, 'field': FIELD, 'our_choices': CHOICES}
        with pytest.raises(TimeoutError):
            await e.call(slow, timeout=0.05)
        assert e.proc is None  # the busy engine was killed, not reused
        out = await e.call({'type': 'ping'})
        await e.close()
        return out
    assert asyncio.run(run())['ok'] is True


def test_critical_memory_pressure_stops_background_groups(tmp_path, monkeypatch):
    import time as _time
    script = tmp_path / 'search' / 'spin.py'
    script.parent.mkdir()
    script.write_text('import time\nwhile True: time.sleep(0.05)\n')
    proc = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    monkeypatch.setattr(value_service, 'VALUE_ROOT', tmp_path / 'value')
    svc = value_service.Service(tmp_path)
    svc.state['producers'] = {'1': {'pid': proc.pid, 'shard': 'sp-1.jsonl'}}
    try:
        svc.shed({'memory_pressure': 2})
        _time.sleep(0.2)
        assert proc.poll() is None  # warning level: keep working
        svc.shed({'memory_pressure': 4})
        proc.wait(timeout=5)
        assert proc.returncode is not None
    finally:
        proc.kill()


def test_gate_futility_stop_only_rejects():
    even = []
    for i in range(40):
        c = i % 2 == 0
        even += [{'pair': i, 'arm': 'cand', 'score': float(c)}, {'pair': i, 'arm': 'inc', 'score': float(not c)}]
    s = value_gate.summarize(even, 80, 0.05)
    assert s['complete'] and s['futility_stop'] and not s['passed']  # 20 gains / 20 losses at 40 pairs
    ahead = [dict(r, score=1.0) if r['arm'] == 'cand' else r for r in even]
    s = value_gate.summarize(ahead, 80, 0.05)
    assert not s['complete'] and not s['futility_stop']  # a leading candidate always plays the full panel
