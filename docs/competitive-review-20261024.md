# Competitive-performance review, 2026-10-24

The plateau is real. The strongest justified new candidate is a shared
belief-game root policy, with complete action coverage and bounded exploitation.
It has passed mathematical and recorded-position checks. **No new Elo gain has
yet been demonstrated.** The M5 can run it; strength must be established by
games against opponents, ultimately a randomized ladder comparison.

## What the results actually say

Frozen cutoff: 2026-10-24 17:48:21 UTC. The private evidence directory is
`data/ml/value/review-20261024/`; `live-inputs.json` freezes the rating records,
`games.json` the selected decision/terminal records, and the analysis scripts
reproduce the measurements. Raw names, room tokens and requests stay private.

| Segment / experiment | Evidence | Conclusion |
|---|---|---|
| Early search baseline | 1206 ±19 performance Elo, 1,325 games | Starting reference |
| Opponent-aware preview | 1236 ±28, 635 games | Modest chronological improvement; not a randomized comparison |
| C5 | 1288 ±21, 1,169 games | Strongest established long segment |
| C9 randomized experiment | 1,582 games; final boost effect +1 Elo, other factors froze off | No winning factor |
| Latest reconstruction deployment | 1218 ±37, 365 games | Current performance remains far below a competitive target |
| Immediately preceding 365 games | 1262 ±37 | Recent change −45, approximate difference interval −97 to +8; does not establish that the correctness fix caused harm |
| C10 preview | Futility at 400 pairs, 8 gains / 10 losses | Rejected |
| C10 endgame, frozen review | 613 pairs, 93 gains / 75 losses, p=.09475 | Pending, not a pass; no early promotion |

Intervals above are descriptive approximate 95% intervals. Deploy segments
confound policy, opponent population, time, and campaign load. The account's
instantaneous rating was 1312, but that does not overturn the 365-game
performance estimate. A ladder win rate near 50% is expected under matchmaking
and is not a measure of progress by itself.

The previous and current 365-game windows contain 2,607 and 2,614 ordinary
search decisions. Median latency was 5.93 vs 5.96 seconds; p95 8.52 vs 8.59.
Search fallbacks fell from three in one Illusion game to zero. Extra-world
search occurred in 66.2% vs 64.0% of decisions. There is no evidence here of a
new widespread transport failure explaining the plateau.

Of 181 current wins, 90 ended in an opponent forfeit and three in an opponent
timer loss. The remaining played-out games were 88/272 wins. **Do not treat
32.4% as an unbiased estimate of playing strength:** strong winning positions
also cause forfeits, so conditioning on how a game ends selects the sample.
All genuine wins remain in the campaign outcome.

The old Kingambit/Milotic/Annihilape deficit did not replicate at this cutoff:
50/104 current wins, performance1219 versus overall1218. Fairy-spreader previews
remain a concerning descriptive subgroup:28/76, performance1133±83. These
overlapping, selected subgroups do not justify another hard-coded team ban.

First-move value calibration also does not show gross universal optimism.
Among363 current games with a recorded move, the sigmoid-scaled first action
value averaged .508 and observed score .493 (Brier .244). This weak predictive
signal does not validate counterfactual rankings or multi-turn planning.

## Why hardware and more data have not solved it

The live policy is mainly simulator search with a hand-designed position
evaluation. C10's MPS model learns to preserve that search's screening choices;
its 1,186,801 rows are action-ranking labels from1,500 games, not1.19million
independent examples of superior human strategy. Its measured training took
8.1seconds. Running more GPU epochs would not change the decision objective.

Likewise, a public replay supplies the action somebody chose and the eventual
outcome, not outcomes for all the actions they could have chosen. Winner
prediction, imitation accuracy and simulation savings are useful diagnostics,
but are not causal evidence of a stronger policy. Previous leaf/preview
candidates repeatedly failed that transfer. C9's direct ladder experiment
is stronger evidence than surrogate-policy agreement.

The search already contains game theory, but its final policy is a best
response to an approximate opponent model. It independently solves each hidden
world, combines the opponent equilibrium with a greedy prior, and then chooses
one action. It does not execute our equilibrium mixture. Pure best response
can exploit a correct model, but becomes vulnerable when that model is wrong.
Changing `alpha` alone does not restore a security guarantee.

A second issue is incomplete action coverage. Each world screens its own
actions. An action's old final score averages only worlds retaining it, with
a constant .5 penalty for any missing worlds. That is a conditional average
after value-based selection. A rare favorable world can determine a choice
without evaluating its consequences in the other worlds.

The research document gives exact counterexamples and the appropriate shared
Bayesian objective. Root commitment itself is not strategy fusion; separately
optimized hidden-state continuations, including forced replacements, remain
an additional limitation. A new root solver does not resolve that entire game.

## Implemented candidate and measured behavior

`search/belief.cjs` solves one shared distribution over our complete joint
actions, while the opponent may react differently to its own hidden type.
`SearchAgent` completes missing nominated rows on the exact original simulator
state and chance streams. It uses a separate RNG to sample the resulting joint
action. Exploitation is allowed within a fixed sampled-game security budget.
The default live configuration is unchanged. Details and the fixed candidate
options are in [the C12 protocol](c12-belief-game.md).

On48 frozen positions, six worlds each:

- Common surviving actions before completion: median11, minimum0.
- Incumbent's chosen action missing from some worlds:3/48.
- Completed decision matrices change the greedy choice:11/48. This includes
  integration of existing deeper cells as well as completion, not an isolated
  causal estimate for screening bias.
- Candidate has no action above95% probability in33/48 positions.
- All solver gaps below .01 leaf units; median .00798.
- Median sampled-game security improvement over the incumbent pure choice:
  .118 leaf units. Median modeled payoff changes by −.016. Neither number is
  an observed win-probability or Elo effect.
- Additional simulations:13.4%. Solver median16ms, p95 49ms. Serial completion
  median1.11s. Measurements ran under changing background load and are not an
  isolated hardware speed comparison.

That diagnostic predates the cache hardening; its fixed positions exclude
sleep/confusion and retain the exact source hashes. The hardened implementation
reuses cached serialized roots, including sampled duration counters, across
adaptive batches and routes completion to the engine owning each root.

On12 fresh contexts from the latest deployment, using the actual three-engine
agent with its adaptive budget, both arms made only legal choices with zero
ordinary or root fallbacks. Four choices changed. Baseline median5.18s, candidate
6.26s; candidate maximum10.54s. This is a small engineering sample under load,
not a precise p99 estimate or a strength test.

Tests cover analytic mixed games, one common commitment across types, informed
opponent responses, exact2×2 security bounds, legal completion, preservation of
chance streams including sleep/confusion, default-path equality, separate RNGs,
timeout fallback, and campaign isolation. The first engineering pilot was stopped
when reconstruction of missing cells was found to resample sleep/confusion
durations; its evidence is retained. The corrected pilot reuses its seeds and
has no promotion authority.

## How a real improvement will be established

C12 freezes one candidate and source, runs an800-pair mixed-opponent panel with
both sides represented for each opponent, and allows only a fresh randomized
ladder experiment after an offline pass. C10 keeps its original protocol and
has priority for any qualified live test. The new campaign cannot overwrite
an active test. Only a live H1 result permits promotion; an inconclusive or
negative result leaves the incumbent in place.

The next scientific uncertainty is the leaf and belief quality, not solver
arithmetic. A small duality gap can coexist with bad Pokémon decisions when
the payoff table omits an important reply or misvalues future turns. If C12
fails, retain that negative result and investigate information-consistent
continuation policies and correlated opponent-set beliefs with fresh evidence;
do not claim that more epochs or another heuristic has solved competitive play.
