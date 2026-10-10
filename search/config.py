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
    'alpha': 0.5,
    'opp_tau': 1.0,
    'seeds': 1,
    # Common random numbers: unbiased variance reduction. On live ledger positions
    # the same worlds re-searched with new seeds flipped the decision 13/19 times
    # without CRN and 8/19 with it, at the same latency (measure_noise.py).
    'crn': True,
    # Close decisions (top-2 margin below 0.5 hand units, about half of turns)
    # get 6 more fresh worlds: world-sampling flips fell from 7/19 at 6 worlds to
    # 5/19 at 12. Skipped when the first batch took over half the 6 s budget; the
    # ladder turn timer is 55 s with a 420 s bank.
    'adaptive_margin': 0.5,
    'adaptive_worlds': 6,
    'adaptive_max_ms': 6000.0,
    # A stalled search falls back to a legal move before the 55 s turn timer.
    'engine_timeout': 40.0,
}


def agent_kwargs(**overrides) -> dict:
    kwargs = copy.deepcopy(LIVE)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs
