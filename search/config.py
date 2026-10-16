"""The deployed search configuration, shared by the live runner and every gate.

A gate that measures a different agent than the one that plays proves nothing
about live play (paper, limitations #1). Build both from `agent_kwargs()`.

`engines` only spreads the same worlds over processes: it changes which world
seeds are drawn, not the decision rule, so gates may use fewer engines to save
memory while measuring the same policy.
"""
from __future__ import annotations

import copy

LIVE = {
    'worlds': 6,
    'engines': 3,
    'screen': {'probe': 4, 'keep_opp': 20, 'keep_ours': 24},
    'alpha': 0.25,  # campaign c5 (docs/endgame-search.md); was 0.5
    'opp_tau': 1.0,
    'seeds': 1,
    # Common random numbers: unbiased variance reduction. On live ledger positions
    # the same worlds re-searched with new seeds flipped the decision 13/19 times
    # without CRN and 8/19 with it, at the same latency (measure_noise.py).
    'crn': True,
    # Close decisions (top-2 margin below 0.5 hand units) get 6 more fresh worlds:
    # world-sampling flips fell from 7/19 at 6 worlds to 5/19 at 12. Skipped when
    # the first batch took over half the 9 s budget (with a 6 s budget only 23% of
    # turns qualified overnight, first batches ran 3.7 s under memory pressure);
    # the ladder turn timer is 55 s with a 420 s bank.
    'adaptive_margin': 0.5,
    'adaptive_worlds': 6,
    'adaptive_max_ms': 9000.0,
    # A stalled search falls back to a legal move before the 55 s turn timer.
    'engine_timeout': 40.0,
    # Chance that a decision logs one world's full probe screening: training
    # labels for the learned screening ranker (docs/learned-screening.md). Free
    # under probe screening; under a learned ranker that world skips the
    # prefilter, so it is part of the deployed policy and gated with it.
    'screen_log': 0.5,
    # Learned human opponent policy (docs/human-opponent.md): None = the
    # equilibrium/greedy mix alone. Live adopts the team's opp-current.json pointer
    # between games; a gate's candidate arm sets it through --cand-search.
    'opp_model': None,
    # Campaign c5 full (docs/endgame-search.md), live 2026-10-16: win-probability
    # leaf, Protect re-scored a second turn when behind, speed inference from move
    # order. Gates vs the imitation: dev 44-23, confirm 52-38; vs search 29-17.
    'leaf': {'a': 0.801, 'b': -0.43},
    'protect_d2': {'behind': -1.0, 'top_opp': 3, 'keep': 6, 'probes': 2},
    'speed_infer': True,
}


def agent_kwargs(**overrides) -> dict:
    kwargs = copy.deepcopy(LIVE)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs
