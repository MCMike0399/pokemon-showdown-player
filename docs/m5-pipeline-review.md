# M5 training and evaluation pipeline review

2026-10-22. The M5 can accelerate training, but training is already measured in
seconds. Evaluation takes hours or days, and the old evaluation rule rejects
many plausible improvements. The useful changes are to stop wasted simulation,
give modest improvements a fair offline test, and develop candidates alongside
the ongoing live experiment.

## Measured hardware and training

DiveMac has an Apple M5, 10 CPU cores and 16 GiB RAM. PyTorch 2.14.1 reports MPS
available. The user explicitly authorized aggressive RAM use and swap for this
work. The pilot therefore ran the full selected dataset, with a process-local
MPS memory allowance of half the recommended working set, without the usual
low-free-RAM deferral. No machine-wide memory policy was changed.

The existing value service is running with `role: live`, collecting data only;
its child list is empty. It has accumulated 61,533 screening worlds versus
15,034 at its last training. Its old gate metadata is not proof of an active
gate. Switching it blindly to `full` would revive old candidates and automatic
promotions during c9, so this pilot runs separately and writes no live pointers.

| Actual screening MLP workload | Two CPU threads | M5 GPU through MPS |
|---|---:|---:|
| Optimizer step, batch 4096 | 1.230 ms | 0.565 ms |
| Training examples per second | 3.33 million | 7.25 million |

This is **2.18× GPU throughput** for this model and batch, not a simulator or
whole-pipeline speedup. The benchmark uses real normalized screening inputs,
10 warmups and 120 timed optimizer steps per device, with MPS synchronization
around timing. [PyTorch documents the required synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html).
The Showdown simulator remains JavaScript on the CPU; this pipeline has no GPU
simulator implementation. More worlds and wider screening already failed c6.

## A completed GPU candidate pilot

One new screening ranker was trained on **777,435 action examples from 1,000
recent baseline-only games**. Experimental c9 settings were excluded. The newest
300 games were held out chronologically; training and validation use game-level
splits within the other 700. Training completed in **1.42 seconds, seven epochs**.

The ranker predicts which moves deserve expensive simulation. Its architecture
and fitting objective are unchanged; this pilot tests whether the newer,
deployment-matched data helps. Selection uses validation alone. The chosen
prefilter is then checked once on the chronological test, without increasing
its size in response to test results.

| Metric at chosen prefilter | Validation | Chronological test |
|---|---:|---:|
| Our probe-kept actions retained | 94.18% | 94.19% |
| Our world's final choice retained | 98.60% | 98.78% |
| Opposing probe-kept replies retained | 92.67% | 91.85% |
| Estimated simulation saving | 15.21% | 15.10% |

Chosen prefilters: 48 of our actions and 36 opposing actions, followed by the
existing probe screen and search matrix. The pilot passed its preregistered
accuracy and estimated-saving checks. **It has not passed a playing-strength
gate and is not deployed.** The existing screening promotion gate demands 20%
measured simulation saving, higher than this estimate; a 15% estimate alone
does not justify spending a full gate or reducing that threshold afterward.

These results support using MPS for candidate development, but do not show that
this candidate wins more games. Screening imitates the existing probe; better
probe prediction is an efficiency opportunity rather than a new strategy.

### Actual simulator check

A subsequent fixed 12-position diagnostic sampled one position per held-out
game (seed 20261022), used identical world seeds in both arms, and alternated
arm order. Both used six worlds, one engine and no adaptive extra worlds so
timing could not change the amount of search. Live decisions retained priority
between benchmark calls. The model and prefilter were not retuned.

| Twelve held-out positions | Probe screen | Learned prefilter |
|---|---:|---:|
| Total search time | 78.15 s | 60.29 s |
| Simulator calls | 67,174 | 54,890 |
| Fallbacks | 0 | 0 |

This is **22.9% less elapsed search time and 18.3% fewer simulations** in a small
sample on a shared machine. Nine of twelve choices were identical; the three
changes need strength evaluation. These timings do not establish a production
latency guarantee, and the simulation saving is still below the existing 20%
screening gate requirement. The artifact therefore remains an unpromoted
candidate, not a newly enabled playing policy.

The host was shared with the live player. During the simulator diagnostic,
five live move decisions had median 8.4 seconds versus 5.9 seconds across the
14 preceding decisions; no stalls were logged. The samples are too small and
the positions differ, so this is not a causal slowdown estimate. It does mean
we cannot claim background CPU work is free merely because RAM/swap is allowed.

## Fixed simulation waste

`search/value_gate.py` checked for futility before entering its concurrency
semaphore. Since every game's coroutine was queued immediately, all of them
passed that check before the first result arrived. A reproduced 100-pair panel
decided after 40 pairs but still ran all 200 games.

The fix checks again **after acquiring a simulator slot**, before constructing
agents. It latches the first decision, lets already-running games finish, and
marks their later results `post_decision`. Those results remain in the ledger
for audit but are excluded from decision summaries, including promotion
consumers. Resuming reconstructs the first decision and starts no extra games.

The same deterministic reproduction now runs **80 games instead of 200** at
concurrency one; at concurrency four it can finish at most three extra games.
This is a scheduling correction, not a change to c9 or to existing statistical
thresholds. Tests also cover resuming after late gains, full panels and bounded
same-seed retries after simulator errors.

## New optional offline gate

For a **new, preregistered experiment**, `--rule pooled --pairs 800` runs:

1. At 400 complete pairs, reject for futility if gains do not exceed losses.
2. At 600 complete pairs, apply the same rejection check.
3. At the full panel, apply one exact one-sided paired sign test at alpha .05.

There is no early promotion, repeated early significance test, or requirement
for two independently significant stages. Interim checks only reject, so they
cannot raise the false-positive rate above the final test under its usual
paired-outcome assumptions. The scheduler preserves the first stopping decision.
The old `superior` and `noninferior` rules remain available and unchanged.

### Power simulation

20,000 simulated campaigns per effect, seed 20261022, independent paired
outcomes with 20% discordant pairs. Gains and losses have probabilities
`(.20 + effect)/2` and `(.20 - effect)/2`. Exact sign-test cutoffs are used;
sample trajectories are checked against the implemented pooled rule.

| True score improvement | Intended legacy 400-pair dev + confirm pass rate | New pooled 800-pair pass rate |
|---|---:|---:|
| 0 percentage points | 0.06% | 4.30% |
| +2 points | 0.96% | 31.82% |
| +4 points | 9.50% | 78.68% |
| +5 points | 18.40% | 92.11% |
| +6 points | 30.08% | 98.12% |

The legacy simulation follows its intended stopping rule, excluding the queue
bug. It is conservative largely because it rejects very early and requires two
passes. The pooled rule spends more time on weak candidates: under a null
effect it averages 571 pairs, compared with 100 under the intended old rule.
That tradeoff buys sensitivity. The numbers depend on the assumed discordance
and independence; they are not a forecast of a real candidate's Elo gain.

**Opponent realism remains unresolved.** A better statistical rule cannot fix
the argmax imitation's action-distribution mismatch. A pass against that
opponent is development evidence, followed by confirmation on fresh live games;
it must not be silently turned into an automatic ladder promotion.

## Operating direction

Keep c9's player and watcher intact while developing the next candidate
offline. Use MPS for fitting on frozen data with chronological holdouts, then
measure actual simulator cost and tactical behavior before committing to a
long gate. Prioritize the focused preview/lead problem in
[the c9 review](c9-progress-review.md) over another broad value-model sweep.
Use the new pooled rule for a future protocol once its opponent population is
chosen, and reserve live A/B for evidence that transfers to the ladder.

Do not infer that data collection continuously improves the deployed player.
The pilot is complete; no recurring training service or new campaign was
started, and no model or gate is authorized to promote automatically by this
document. Source changes are local; the pinned player remains unchanged.

Artifacts are retained in `data/ml/value/pipeline-pilot-20261022/`:
`PROTOCOL.md`, `screen-files.json`, `pilot.py`, `benchmark.json`, `result.json`,
the immutable `screen-f5b60ce73d2c3178.json` candidate, and reproducible
`gate_power.py` / `gate-power.json`. The simulator check is reproducible from
`position_benchmark.py`, `position-manifest.json`, `positions.jsonl`, and
`position-result.json`. No source accounts or credentials are in
the public report. The full offline suite passed 463 tests; the final scheduler
suite passed all 10 tests, including the subsequently added error/retry case.
