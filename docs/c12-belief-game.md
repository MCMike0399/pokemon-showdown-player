# C12: shared belief-game decisions, followed by ladder confirmation

Protocol fixed before the new strength panel. This is an unproven candidate,
not a claim of an Elo breakthrough. Engineering evidence and the diagnosis are
in [the review](competitive-review-20261024.md); mathematical assumptions and
primary references are in [the research](game-theory-research.md).

## Candidate

Keep the C5 team, opponent-aware preview, leaf, world sampler, screening,
adaptive world budget and selective Protect continuation. Add exactly:

```json
{"root_policy":{"keep":24,"risk":0.1,"iterations":4000,"tolerance":0.01}}
```

Nominate at most 24 actions with the incumbent ranking. Complete every nominated
row against every retained opponent reply in every successful world, reusing
the original serialized battle and the original common chance streams. Exclude
an action if it is genuinely invalid in any completed world. Never impute a
missing payoff or average only favorable surviving worlds. Reconstruction
failures remain explicit and are not silently treated as zero-payoff worlds.

Run shared regret matching: one distribution over our joint actions, one
opponent response per sampled private type. The objective is the mean across
types of the minimum expected payoff against that type's replies. Stop at a
primal-dual gap of .01 or 4,000 iterations; always record the achieved gap.
The averaged strategies use nonnegative cumulative regrets and linear weights.

Allow exploitation by moving from that distribution toward the best pure
response to the existing opponent prior, but only as far as the resulting
sampled-game security stays within .1 leaf units of the solved distribution.
This is a line-segment optimization, not a claim to solve the full constrained
optimal-exploitation problem. Sample the entire doubles action with an RNG
independent of world sampling. Keep the chosen action through transport retries.

Existing selective deeper results replace their actual matrix cells before
the shared solve. Other cells remain one-turn estimates; completion does not
introduce additional deep searches. This changes the old scalar correction's
renormalization and is part of this fixed bundle. Forced replacements retain
the incumbent decision rule. Completion shares the existing engine timeout
budget; a failed root solve uses the incumbent's already-ranked action and is
recorded as an infrastructure failure for offline gates.

The certificate covers only the retained actions, sampled opponent types and
approximate leaf values. The opponent is modeled as knowing its own type and
our set. This is not a full information-set solution of Pokémon, a learned
long-horizon value function, or a guarantee against replies removed by screening.

## Engineering checks

Before launching the strength campaign, require the offline suite, real-engine
completion checks including sleep/confusion and fixed chance streams, fresh
recorded-context replay with legal choices and no unresolved root failures,
and a completed eight-pair engineering pilot. That pilot has no promotion
authority. Stop for unresolved infrastructure errors or a catastrophic pilot
deficit of at least four discordant losses beyond gains. Do not tune the
candidate on the pilot and reuse it as confirmation.

## Frozen strength panel

One 800-pair pooled panel, alpha .05; seed 202610250. Each pair compares the
candidate and C5 against the same opponent team, side and battle seed. Preview
is `pvalue-661448b6a320fce0` in both arms. Source, team, usage files, models and
team-pool hashes are frozen in `data/ml/value/campaign-c12/spec.json` before
the first panel game. Pilot seeds and diagnostic positions are excluded.

Opponents rotate in two-pair blocks so each receives both sides:

1. Incumbent search.
2. Frozen human imitation `human-449e6bd0e8869deb`, argmax.
3. The same imitation, sampled at temperature 1.

Both imitation variants use preview `preview-1455f4a55d18112a` and all opponents
use the frozen real-team pool. These are surrogate opponents; none is asserted
to reproduce competitive humans. Record their separate results as diagnostics.

No early success. Futility only at 400 and 600 completed pairs if gains do not
exceed losses. Final success requires gains > losses and the one-sided exact
paired sign test p <= .05. The first stopping decision is immutable. Failed
games retry their original seeds, at most three attempts, then void their pair
under the existing gate rules. Errors and voids remain visible. The second
400 pairs are fresh draws with no mid-panel retuning. No repeated restart with
a different seed after rejection.

Run three offline games concurrently (`slots:6` means three for this one
candidate under the existing supervisor). Register their process group with
live priority. The surviving C10 worker group keeps its existing three slots;
no C10 files, source, outcomes or acceptance rules change. Source is pinned in
`<checkout>.c12`; launchd adopts live workers or resumes the same ledger after
a restart. The 150 GB storage budget and live reserve remain in force.

## Fresh ladder experiment

An offline pass only permits a new live test. Wait for C10 to finish its whole
protocol, including its live experiment if it qualifies. Never replace any
active experiment or an unknown experiment ID. If C10 improves the baseline,
its promoted settings apply to both arms of C12's live comparison.

The capable source must already be deployed with the candidate **disabled**
before assigning an arm. Then create one fresh `c12` A/B specification, one
`upgrade` factor with the exact options above, seed 202610350, balanced two-game
blocks. Use the existing GSPRT of 0 versus +40 Elo, alpha .05, beta .10,
minimum 300 and maximum 3,000 games. No win-rate, peak-rating, subgroup or
offline-surrogate substitute for this decision. Rejection or inconclusive
retains the baseline. H1 permits the existing tested promotion path to update
main and gracefully deploy. Only subsequent ladder evidence supports an actual
strength claim. Report descriptive performance ratings and the randomized arm
effect separately; repeated-opponent and time dependence remain limitations.

## Operations

```bash
.venv/bin/python scripts/campaign_service.py install --campaign c12
.venv/bin/python -m search.campaign status --dir data/ml/value/campaign-c12
.venv/bin/python -m search.ab_test status
```

`state.json` reports phase/errors/jobs; `belief.jsonl` contains games and
`belief-result.json` the immutable stopping result. `live/` holds eventual
watcher/promotion receipts. `active:false` pauses new scheduling; existing
workers retain and finish their evidence. Do not edit the pinned checkout.
