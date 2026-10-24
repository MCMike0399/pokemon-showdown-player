"""Information-set and mixed-strategy contracts, independent of Pokemon heuristics."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def solve(matrices, **options):
    proc = subprocess.run(['node', '-e', "const b=require('./search/belief.cjs');"
                           "const d=JSON.parse(require('fs').readFileSync(0,'utf8'));"
                           "console.log(JSON.stringify(b.solveBelief(d.matrices,d.options)));"],
                          cwd=ROOT, input=json.dumps({'matrices': matrices, 'options': options}),
                          text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def matrix(payoff, prior=None):
    return {'ours': [str(i) for i in range(len(payoff))],
            'theirs': [str(i) for i in range(len(payoff[0]))],
            'payoff': payoff, 'prior': prior or [1 / len(payoff[0])] * len(payoff[0])}


def test_mixed_strategy_is_preserved_not_argmaxed():
    out = solve([matrix([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])], risk=0)
    assert list(out['policy'].values()) == pytest.approx([1/3] * 3, abs=.002)
    assert out['security'] >= -.002
    assert out['gap'] <= .002


def test_one_policy_across_hidden_worlds_not_average_of_clairvoyant_policies():
    # World 1 optimum x(A)=1/11, world 2 optimum x(A)=0. Averaging those
    # strategies yields security ~.7045. The shared optimum is x(A)=1/11,
    # security 10/11, lower than the unattainable clairvoyant value 21/22.
    out = solve([matrix([[10, 0], [0, 1]]), matrix([[0, 0], [1, 1]])],
                risk=0, iterations=12000, tolerance=.003)
    assert out['policy']['0'] == pytest.approx(1/11, abs=.003)
    assert out['security'] == pytest.approx(10/11, abs=.003)
    assert out['gap'] < .006


def test_opponent_knows_own_world_and_can_choose_different_replies():
    # Averaging the matrices before minimizing would invent a value of +1.
    out = solve([matrix([[2, 0]]), matrix([[0, 2]])], risk=0)
    assert out['security'] == pytest.approx(0)


def test_exploitation_is_bounded_in_the_sampled_game():
    out = solve([matrix([[1, -1], [-1, 1]], [.9, .1])], risk=.1)
    assert out['policy']['0'] > .5
    assert out['security'] >= out['equilibrium_security'] - .100001
    assert out['model_value'] > out['equilibrium_model_value']


def test_dominance_and_action_labels_survive_reordering():
    a = matrix([[3, 2], [0, 1]])
    b = {'ours': ['1', '0'], 'theirs': ['x', 'y'], 'payoff': [[0, 1], [3, 2]], 'prior': [.5, .5]}
    out = solve([a, b], risk=0)
    assert out['policy']['0'] > .99
    assert out['security'] > 1.99


def test_missing_cells_fail_closed_instead_of_optimistic_conditional_mean():
    with pytest.raises(subprocess.CalledProcessError):
        solve([matrix([[1, 0], [0, 1]]), matrix([[3, 3]])])


@pytest.mark.parametrize('options', [{'risk': -1}, {'iterations': 0}, {'tolerance': -1}])
def test_invalid_solver_options_are_rejected(options):
    with pytest.raises(subprocess.CalledProcessError):
        solve([matrix([[1]])], **options)


@pytest.mark.parametrize('sleep', [False, True])
def test_completion_preserves_chance_streams_and_default_search(sleep):
    script = """
const E=require('./search/engine.cjs');
const mon=(species,ability,moves)=>({set:{species,ability,moves,level:50,
  nature:'Modest',evs:{hp:2,spa:32,spe:32}},state:{hp:1,turn:3}});
const world={p1:[mon('Politoed','Drizzle',['muddywater','protect']),
                 mon('Archaludon','Stamina',['dragonpulse','protect'])],
             p2:[mon('Garchomp','Rough Skin',['dragonclaw','protect']),
                 mon('Sylveon','Pixilate',['hypervoice','protect'])]};
const msg={type:'search',format:'gen9championsvgc2026regmc',worlds:[world],
  field:{turn:3,sides:{p1:{},p2:{}}},world_offset:42,crn:true,seeds:2,
  leaf:{a:.801,b:-.43},our_choices:['move muddywater, move dragonpulse 1',
    'move protect, move protect','move muddywater, move dragonpulse 2']};
if (SLEEP) { world.p1[0].state.status='slp'; world.p2[0].state.volatiles={confusion:{}}; }
Math.random=()=>0.01;
const plain=E.handle(msg).results[0];
const detailed=E.handle({...msg,matrix_detail:true}).results[0];
const m=detailed.matrix;
const partial={...m,ours:m.ours.slice(1),payoff:m.payoff.slice(1)};
Math.random=()=>0.99; // completion must not resample hidden duration counters
const completed=E.handle({type:'complete_matrix',jobs:[{source:{...msg,world},
  matrix:partial,wanted:[...m.ours,'move nonexistent, move protect']}]}).results[0];
console.log(JSON.stringify({plain,detailed,completed}));
"""
    proc = subprocess.run(['node', '-e', 'const SLEEP=' + json.dumps(sleep) + ';' + script], cwd=ROOT, text=True,
                          capture_output=True, check=True, timeout=60)
    result = json.loads(proc.stdout)
    detailed = result['detailed']
    original = detailed.pop('matrix')
    result['plain'].pop('ms')
    detailed.pop('ms')
    assert result['plain'] == detailed
    assert result['completed']['matrix'] == original
    assert result['completed']['sims'] == 2 * len(original['theirs'])


def test_policy_sampling_reseeds_without_perturbing_world_sampling():
    from search.agent import SearchAgent
    a = SearchAgent(seed=12, crn=True, root_policy={'risk': .1})
    first = [a.policy_rng.random() for _ in range(5)]
    worlds = [a.rng.random() for _ in range(5)]
    a.reseed(12)
    assert [a.rng.random() for _ in range(5)] == worlds
    assert [a.policy_rng.random() for _ in range(5)] == first


def test_belief_policy_requires_crn_and_known_options():
    from search.agent import SearchAgent
    with pytest.raises(ValueError, match='crn'):
        SearchAgent(root_policy={'risk': .1})
    with pytest.raises(ValueError, match='unknown'):
        SearchAgent(crn=True, root_policy={'riskk': .1})


@pytest.mark.parametrize('error', [RuntimeError, TimeoutError])
def test_root_failure_keeps_incumbent_choice_and_is_counted(monkeypatch, error):
    import asyncio
    from search.agent import SearchAgent
    agent = SearchAgent(crn=True, root_policy={'risk': .1})
    agent.stats.pop('root_fallbacks')  # live arm switching carries the old agent's counters
    ctx = {'choices': ['move 1, move 1', 'move 2, move 1'],
           'request': {'side': {'id': 'p1'}, 'active': []}}
    monkeypatch.setattr(agent, '_prune_ours', lambda ctx: ctx['choices'])
    monkeypatch.setattr(agent, '_our_side', lambda *a, **kw: [])

    async def search(*args, **kw):
        return [{'values': {'move 1, move 1': 1., 'move 2, move 1': 2.}}]

    async def broken(*args, **kw):
        raise error('incomplete test game')

    monkeypatch.setattr(agent, '_search', search)
    monkeypatch.setattr(agent, '_belief_choice', broken)
    choice = asyncio.run(agent.decide(ctx))
    assert choice == 'move 2, move 1'
    assert agent.stats['root_fallbacks'] == 1
    assert agent.stats['fallbacks'] == 0
    assert agent.last_trace['root_policy']['fallback'] == 'incumbent'


def test_certificate_brackets_exact_two_by_two_values():
    # Independent closed-form solutions; checks bounds rather than mirroring
    # the regret algorithm. Includes pure saddles and interior mixtures.
    import random
    rng = random.Random(17)
    cases = [[[rng.uniform(-2, 2) for _ in range(2)] for _ in range(2)] for _ in range(16)]
    for M in cases:
        a, b = M[0]
        c, d = M[1]
        xs = [0., 1.]
        denom = a - b - c + d
        if denom and 0 < (d-c)/denom < 1:
            xs.append((d-c)/denom)
        exact = max(min(x*a+(1-x)*c, x*b+(1-x)*d) for x in xs)
        out = solve([matrix(M)], risk=.1)
        assert out['equilibrium_security'] <= exact + 1e-9
        assert out['upper_bound'] >= exact - 1e-9
        assert out['gap'] < .02
        assert out['security'] >= out['equilibrium_security'] - .100001
