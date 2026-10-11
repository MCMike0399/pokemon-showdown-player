"""Learned screening: probe logging, the MPS-trained ranker, its gate and its pointer."""
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from search import screen_data, value_gate, value_service, value_store
from tests.test_offline_value_learning import FIELD, FMT, WORLD, _FakeEngine, _move_ctx, engine, needs_sim

ROOT = Path(__file__).resolve().parents[1]
SCREEN = {'probe': 4, 'keep_opp': 20, 'keep_ours': 24}


def _dummy_net(path: Path, d: int, seed: int = 0):
    rnd = random.Random(seed)
    path.write_text(json.dumps({'layers': [{'W': [[rnd.gauss(0, .1) for _ in range(d)] for _ in range(8)], 'b': [0] * 8},
                                           {'W': [[rnd.gauss(0, .1) for _ in range(8)]], 'b': [0]}],
                                'mean': [0] * d, 'std': [1] * d, 'prefilter': {'opp': 20, 'ours': 24}, 'version': 1}))
    return path


@needs_sim
def test_logged_world_carries_probe_scores_for_every_candidate(tmp_path):
    base = {'type': 'search', 'format': FMT, 'worlds': [WORLD, WORLD], 'field': FIELD, 'screen': SCREEN, 'crn': True}
    plain, logged = engine(base, {**base, 'screen_log': 1})
    assert all('screen_record' not in r for r in plain['results'])
    first, second = logged['results']
    assert 'screen_record' not in second  # only the first `screen_log` worlds
    # Logging only reads: the decision and its cost are unchanged.
    assert first['values'] == pytest.approx(plain['results'][0]['values']) and first['sims'] == plain['results'][0]['sims']
    rec = first['screen_record']
    assert [len(x) for x in rec['sx']] == [146, 146]
    for side in ('ours', 'theirs'):
        assert len(rec[side]['a']) == len(rec[side]['s']) > SCREEN['keep_ours' if side == 'ours' else 'keep_opp']
        assert {len(a) for a in rec[side]['a']} == {70}
    assert 0 <= rec['ours']['best'] < len(rec['ours']['a'])


@needs_sim
def test_learned_prefilter_replaces_probe_simulations(tmp_path):
    net = _dummy_net(tmp_path / 'net.json', 146 + 70 + 1)
    base = {'type': 'search', 'format': FMT, 'worlds': [WORLD], 'field': FIELD, 'crn': True}
    probe, pure, hybrid, logged = engine(
        {**base, 'screen': SCREEN},
        {**base, 'screen': {**SCREEN, 'learned': {'path': str(net), 'pre_opp': 20, 'pre_ours': 24}}},
        {**base, 'screen': {**SCREEN, 'learned': {'path': str(net), 'pre_opp': 28, 'pre_ours': 32}}},
        {**base, 'screen': {**SCREEN, 'learned': {'path': str(net), 'pre_opp': 20, 'pre_ours': 24}}, 'screen_log': 1})
    p, l, h, g = (x['results'][0] for x in (probe, pure, hybrid, logged))
    assert p['screened'] == 'probe' and p['screen_sims'] > 0
    assert l['screened'] == 'learned' and l['screen_sims'] == 0 and l['sims'] <= 24 * 20 < p['sims']
    assert 0 < h['screen_sims'] <= 4 * (28 + 32) and h['sims'] < p['sims']
    assert l['n_ours'] <= 24 and l['n_theirs'] <= 20
    # A logged world runs the full probes, so its labels never depend on the ranker.
    assert g['screened'] == 'probe' and g['sims'] == p['sims'] and 'screen_record' in g


@needs_sim
def test_action_features_agree_for_move_ids_and_request_slots():
    script = r"""
const E = require('./search/engine.cjs');
const SF = require('./search/screenfeat.cjs');
const msg = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const b = E.buildBattle(msg.format, msg.world, msg.field);
const p = b.sides[0].active[0];
const ids = p.getMoveRequestData().moves.map(m => m.id);
const bySlot = SF.actionFeatures(b, 0, 'move 1 1, move 2 2');
const q = b.sides[0].active[1].getMoveRequestData().moves.map(m => m.id);
const byId = SF.actionFeatures(b, 0, `move ${ids[0]} 1, move ${q[1]} 2`);
const sw = SF.actionFeatures(b, 0, 'switch 3, move 1 1');
console.log(JSON.stringify({bySlot, byId, sw, dim: SF.ACTION_DIM}));
"""
    out = subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, timeout=60,
                         input=json.dumps({'format': FMT, 'world': WORLD, 'field': FIELD}))
    r = json.loads(out.stdout)
    assert r['bySlot'] == r['byId'] and len(r['bySlot']) == r['dim'] == 70
    assert r['sw'][1] == 1 and r['sw'][0] == 0  # slot 0 switches, does not move


def _synthetic_games(root: Path, n_games: int, seed: int = 0, prefix: str = 'gate'):
    """Probe scores that follow one action feature, so a ranker can learn them."""
    rng = np.random.default_rng(seed)
    for g in range(n_games):
        records = []
        for t in range(4):
            sx = rng.normal(size=(2, 146)).round(3).tolist()
            rec = {'sx': sx}
            for side, m in (('ours', 40), ('theirs', 36)):
                a = rng.normal(size=(m, 70))
                score = a[:, 7] * 2 + rng.normal(scale=.2, size=m)
                rec[side] = {'a': a.round(3).tolist(), 's': (score if side == 'ours' else -score).round(4).tolist()}
            rec['ours']['best'] = int(np.argmax(rec['ours']['s']))
            records.append({'turn': t + 1, 'record': rec})
        screen_data.write_records(records, root / f'{prefix}-{g}.npz', f'{prefix}-{g}')


def test_records_round_trip(tmp_path):
    _synthetic_games(tmp_path, 1)
    d = screen_data.load_records(tmp_path / 'gate-0.npz')
    assert d['S'].shape == (4, 2, 146) and d['A'].shape == (4 * 76, 70)
    assert set(np.unique(d['role'])) == {0, 1} and int(d['best'].sum()) == 4
    assert screen_data.write_records([{'turn': 1, 'record': {}}], tmp_path / 'x.npz', 'x') is None


def test_ranker_trains_picks_prefilter_and_verifies(tmp_path):
    pytest.importorskip('torch')
    from search import train_screen
    data = tmp_path / 'screen'
    _synthetic_games(data, 60)
    _synthetic_games(data, 10, seed=1, prefix='live')
    report = train_screen.main(['--data', str(data / '*.npz'), '--out', str(tmp_path / 'cands'), '--device', 'cpu',
                                '--epochs', '8', '--min-groups', '10'])
    assert report['useful'], report
    assert report['prefilter']['ours'] >= 24 and report['prefilter']['opp'] >= 20
    assert report['val']['ours']['by_pre'][str(report['prefilter']['ours'])]['recall'] >= .9
    assert report['estimated_sim_saving']['val'] >= .15
    path = Path(report['path'])
    assert screen_data.weights_sha(path) == report['sha256']
    # Immutable: a second identical run reuses the same file.
    pointer = tmp_path / 'screen-current.json'
    screen_data.promote_screen(path, {}, pointer)
    current = screen_data.current_screen(pointer)
    assert current['sha256'] == report['sha256'] and current['pre_ours'] == report['prefilter']['ours']
    model = json.loads(path.read_text())
    model['prefilter']['ours'] = 99  # an edited candidate never reaches live play
    path.write_text(json.dumps(model))
    assert screen_data.current_screen(pointer)['path'] is None


def _rows(pairs, gains, losses, cand_sims=50, inc_sims=100):
    rows = []
    for p in range(pairs):
        c, i = (1, 0) if p < gains else (0, 1) if p < gains + losses else (1, 1)
        rows += [{'pair': p, 'arm': 'cand', 'win': bool(c), 'score': float(c), 'sims': cand_sims * 10, 'decisions': 10},
                 {'pair': p, 'arm': 'inc', 'win': bool(i), 'score': float(i), 'sims': inc_sims * 10, 'decisions': 10}]
    return rows


def test_noninferior_gate_needs_equal_results_and_fewer_simulations():
    s = value_gate.summarize(_rows(80, 8, 8), 80, .05, 'noninferior', .2)
    assert s['complete'] and s['passed'] and s['sims_ratio'] == .5
    assert not value_gate.summarize(_rows(80, 8, 8, cand_sims=90), 80, .05, 'noninferior', .2)['passed']
    assert not value_gate.summarize(_rows(80, 7, 9), 80, .05, 'noninferior', .2)['passed']
    early = value_gate.summarize(_rows(40, 2, 7), 80, .05, 'noninferior', .2)
    assert early['futility_stop'] and early['complete'] and not early['passed']
    # The superior rule is unchanged: a tie never passes.
    assert not value_gate.summarize(_rows(80, 8, 8), 80, .05)['passed']


def _service(tmp_path, monkeypatch):
    from search import screen_data as sd
    from search import team_data
    vroot = tmp_path / 'value'
    monkeypatch.setattr(value_service, 'VALUE_ROOT', vroot)
    monkeypatch.setattr(team_data, 'TEAMS_ROOT', vroot / 'teams')
    svc = value_service.Service(tmp_path)
    spawned = []

    class Proc:
        pid = 4242

    def spawn(name, argv, fds, log, nice=10, qos=None):
        spawned.append((name, argv))
        svc.children[name] = Proc()
        return Proc()

    monkeypatch.setattr(svc, 'spawn', spawn)
    monkeypatch.setattr(svc.slots, 'acquire', lambda limit, n: [open(tmp_path / f's{i}', 'a') for i in range(n)])
    return svc, spawned, sd


def test_service_gates_screening_first_and_promotes_its_own_pointer(tmp_path, monkeypatch):
    svc, spawned, sd = _service(tmp_path, monkeypatch)
    res = {'training_allowed': True, 'allowed_workers': 4, 'slot_pool': 4}
    cfg = value_service.load_config()
    net = _dummy_net(tmp_path / 'screen-a.json', 10)
    sha = sd.weights_sha(net)
    value = svc.tdir / 'candidates' / 'value-x.json'
    value.write_text(json.dumps({'layers': [], 'mean': [], 'std': []}))
    svc.state['candidates'] = [{'path': str(value), 'sha256': value_store.weights_sha(value), 'beta': .5,
                                'status': 'trained', 'team': svc.team, 'val': {}, 'live_test': {}}]
    svc.state['screen_candidates'] = [{'kind': 'screen', 'path': str(net), 'sha256': sha, 'status': 'trained',
                                       'team': svc.team, 'prefilter': {'opp': 20, 'ours': 24}}]
    svc.maybe_gate(cfg, res)
    name, argv = spawned[-1]
    cmd = argv[-1]
    assert name == 'gate' and svc.state['gate']['kind'] == 'screen'
    assert '--rule noninferior' in cmd and '--cand-screen' in cmd and str(net) in cmd and '--screen-log-dir' in cmd
    report = tmp_path / 'r.json'

    def finish(passed):
        report.write_text(json.dumps({'complete': True, 'passed': passed, 'gains': 9, 'losses': 9, 'p_one_sided': .5}))
        svc.state['gate']['report'] = str(report)
        svc.children.pop('gate', None)
        svc.finish_gate(0, cfg)

    finish(True)
    assert svc.state['gate']['stage'] == 'confirm'
    finish(True)
    pointer = sd.screen_pointer(svc.team)
    assert sd.current_screen(pointer)['sha256'] == sha
    assert value_store.current_value(svc.pointer())['path'] is None  # the value pointer is untouched
    # The next gate is the value candidate, played with the deployed screening in every seat.
    svc.maybe_gate(cfg, res)
    cmd = spawned[-1][1][-1]
    deployed = sd.current_screen(pointer)['path']  # the verified copy beside the team's candidates
    assert svc.state['gate']['kind'] == 'value' and cmd.count(deployed) == 2
    assert '--rule' not in cmd


def test_service_trains_ranker_after_enough_logged_worlds(tmp_path, monkeypatch):
    svc, spawned, sd = _service(tmp_path, monkeypatch)
    _synthetic_games(sd.screen_root(svc.team), 3)
    cfg = {**value_service.load_config(), 'screen_min_records': 10, 'screen_train_every_records': 10}
    res = {'training_allowed': True, 'allowed_workers': 4, 'slot_pool': 4}
    svc.maybe_train_screen(cfg, res)
    assert spawned[-1][0] == 'train-screen' and svc.state['pending_screen_train']['records'] == 12
    out = Path(svc.state['pending_screen_train']['report'])
    out.write_text(json.dumps({'path': 'p.json', 'sha256': 'ab' * 32, 'useful': False, 'prefilter': {}}))
    svc.children.pop('train-screen')
    svc.finish_train_screen(0)
    assert svc.state['screen_trained_records'] == 12
    assert svc.state['screen_candidates'][-1]['status'].startswith('rejected')
    svc.maybe_train_screen(cfg, res)  # nothing new logged: no retrain
    assert [n for n, _ in spawned].count('train-screen') == 1


def test_agent_collects_records_and_sends_the_ranker(monkeypatch):
    import asyncio
    from search.agent import SearchAgent

    class Engine(_FakeEngine):
        def __init__(self, values):
            super().__init__(values)
            self.msgs = []

        async def call(self, msg, timeout=120):
            self.msgs.append(msg)
            out = await super().call(msg, timeout)
            for r in out['results'][:msg.get('screen_log', 0)]:
                r['screen_record'] = {'sx': [[0], [0]]}
            return out

    from search import agent as agent_mod
    monkeypatch.setattr(agent_mod, 'sample_bench', lambda *a, **k: [])
    monkeypatch.setattr(agent_mod, 'sample_set', lambda rng, species, observed, fmt, cutoff: {
        'species': species, 'item': '', 'ability': '', 'moves': ['tackle'], 'nature': 'Hardy', 'level': 50})
    ctx, choices = _move_ctx()
    agent = SearchAgent(worlds=2, engines=1, screen=SCREEN, screen_log=1.0,
                        screen_model={'path': '/x.json', 'pre_opp': 28, 'pre_ours': 32, 'sha256': 'abc'})
    track = agent_mod.Tracker('p1').feed(ctx['public_log'])
    fake = Engine({agent._to_sim(c, ctx['request']['active'], track): float(i) for i, c in enumerate(choices)})
    agent.engines = [fake]
    asyncio.run(agent.decide(ctx))
    assert fake.msgs[0]['screen']['learned'] == {'path': '/x.json', 'pre_opp': 28, 'pre_ours': 32}
    assert fake.msgs[0]['screen_log'] == 1
    records = agent.take_screen_records()
    assert len(records) == 1 and agent.take_screen_records() == []
    assert agent.last_trace['screen'] == 'abc'
    agent.set_screen(None)
    assert agent.screen_model is None


def test_split_keeps_both_arms_and_seats_of_a_gate_game_together():
    from search.train_screen import game_key
    names = [f'gate-abc-dev-7-{arm}-{who}.npz' for arm in ('cand', 'inc') for who in ('us', 'them')]
    assert {game_key(n) for n in names} == {'gate-abc-dev-7'}
    assert game_key('live-gen9x-123.npz') == 'live-gen9x-123.npz'


def test_value_candidates_are_not_starved_by_screening_retrains(tmp_path, monkeypatch):
    svc, spawned, sd = _service(tmp_path, monkeypatch)
    cfg = value_service.load_config()
    net = _dummy_net(tmp_path / 'screen-a.json', 10)
    value = svc.tdir / 'candidates' / 'value-x.json'
    value.write_text(json.dumps({'layers': [], 'mean': [], 'std': []}))
    svc.state['candidates'] = [{'path': str(value), 'sha256': value_store.weights_sha(value), 'beta': .5,
                                'status': 'trained', 'team': svc.team}]
    svc.state['screen_candidates'] = [{'kind': 'screen', 'path': str(net), 'sha256': sd.weights_sha(net),
                                       'status': 'trained', 'team': svc.team}]
    svc.state['last_gate_kind'] = 'screen'  # a screen gate just finished
    assert svc._next_gate(cfg)['kind'] == 'value'
    svc.state['last_gate_kind'] = 'value'
    assert svc._next_gate(cfg)['kind'] == 'screen'
