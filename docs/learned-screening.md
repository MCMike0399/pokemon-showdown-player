# Learned screening: the GPU learns which actions are worth simulating

## Why

Almost all CPU in this project is the Showdown simulator: live search, self-play,
value gates and PPO collection. The GPU can't run it. All neural-network training
already runs on MPS and finishes in seconds, so the GPU is idle most of the time.

The GPU can still save simulator work. Each world of a search decision
(`docs/search-agent.md`, paper section 11) first runs a **probe screening**. Every
candidate action is simulated against four replies, only to rank the candidates
and keep the best 24 of ours and 20 of theirs. Then the 24×20 matrix is simulated.
On a live position with about 60 actions per side, that is

    probes 4·(|A| + |B|) ≈ 490   +   matrix 24·20 = 480   ≈ 970 simulations

so about half of every search exists only to decide what to search. A ranker
trained on the GPU from those same probe results orders the candidates without
simulating them. The CPU then spends its time on the matrix, which is the work
that decides the move.

## The loop

```
live game / gate game (CPU)                       data/ml/value/teams/<slug>/screen/*.npz
  probe screening runs anyway  ---- log 1 world ----> position, action features, probe score,
  (screen_log = 0.5 per decision)                      the world's final choice
                                                                  |
                                       search/train_screen.py (MPS, worker.lock)
                                                                  v
                                         screen-candidates/screen-<sha16>.json
                                         ranker + prefilter sizes chosen on held-out games
                                                                  |
                     paired gate x2 (CPU): same value, only the screening differs,
                     non-inferior results at >= 20% fewer simulations per decision
                                                                  v
                                             screen-current.json (atomic, SHA-256 verified)
                                                                  |
                     live runner adopts it between games; every later gate plays it
```

| Step | Device | Where |
|---|---|---|
| Log | CPU, no extra simulations | `engine.cjs` `screen_log`: the probe scores of every candidate in one world, plus the features below. Logging only reads; a test asserts identical values and simulation counts. |
| Features | CPU, microseconds | `search/screenfeat.cjs`: 70 per joint action (per slot: move/switch/pass, Mega, category, power, priority, accuracy, spread, ally target, Protect, Fake Out, field setter, setup, status, heal, pivot, redirection, estimated damage and KO on the target, speed order, HP; switch-in matchups; joint focus fire and combined KOs), plus the 146 position features of `valuefeat.cjs` from the choosing side. |
| Train | MPS | `search/train_screen.py`: 2×128 MLP. Its targets are probe scores standardized within each (world, side) group, because only the order inside a group matters. Splits are by game file, and the newest 30% of live games are an untouched test. |
| Gate | CPU | `search/value_gate.py --cand-screen … --rule noninferior --min-sim-saving 0.2` |
| Promote | - | `teams/<slug>/screen-current.json`, its own pointer next to the value's `current.json` |

`search/value_service.py` runs all of it. The first candidate is trained after
`screen_min_records` (3000) logged worlds, and later ones after another
`screen_train_every_records` (6000), which is more than one two-stage gate logs.
Training holds `data/ml/worker.lock`, so it never shares the GPU with the PPO or
value learner. Screening candidates are gated **before** value candidates,
because a promoted ranker makes every later gate, shard and live turn cheaper.
They are never gated twice in a row while a value candidate waits, so neither
loop starves the other. The one-promotion-per-day limit applies to each kind
separately.

## How the engine uses the ranker

`msg.screen.learned = {path, pre_opp, pre_ours}`:

1. Score every candidate with the ranker and keep the top `pre_opp` of theirs
   and `pre_ours` of ours.
2. Run the usual probe screening on the survivors only, if they still exceed the
   keeps (20 and 24). With `pre` equal to the keeps no probe runs at all.
3. Simulate the matrix as before.

The prefilter is chosen by the trainer, not by hand. On held-out games it takes
the smallest `pre` where:

- **theirs:** the prefilter keeps, on average, at least 92% of the 20 replies the
  probes would have kept (a dropped strong reply makes our values optimistic);
- **ours:** the world's final choice survives in at least 98% of worlds, and on
  average at least 90% of the probe-kept 24 survive.

Once the held-out live games hold at least 50 groups for a side, the chosen size
must also meet the thresholds there, within 0.03. Validation splits are by
played game: a gate pair's two arms replay the same seed and worlds, and its two
seats see the same game, so their four files always fall on the same side.

A candidate is `useful` only if both sizes exist and the estimated simulations
per world drop by at least 15%. The report compares the ranker's recall with two
baselines at every prefilter size: the engine's enumeration order and a random
order.

A logged world always runs the full probes, so its labels never depend on the
ranker in use. With a ranker deployed, `screen_log` = 0.5 therefore costs about
one world in twelve. The logged world is placed on a random engine, so one engine
does not always set the decision's wall time. It is part of the deployed configuration (`search/config.py`),
so gates measure it too.

## The gate rule

A ranker is meant to save CPU at equal strength, so a superiority test would be
the wrong question: a candidate that plays exactly as well would fail it half the
time. The screening gate pairs games exactly like the value gate (same seeds,
opponents, sides and worlds; same value in both arms). It passes a stage when:

- all 80 pairs complete with **gains ≥ losses**, and
- the candidate's **simulations per decision ≤ 80%** of the incumbent's, measured
  in the gate games themselves.

It stops early as a rejection when, after 40 pairs, losses exceed gains by 5 or
more. Both stages must pass: development, then confirmation on fresh seeds.

**Honest limit.** This is a point-estimate non-inferiority rule, not a test with
a stated error rate. With about 20 discordant pairs per stage, a candidate whose
true discordant loss rate is 60% still passes one stage with probability about
0.25, and both stages with about 0.06. A truly worse candidate is likely to be
caught, but a small loss of strength can slip through. The live record after
promotion is the next check. Rollback: delete `screen-current.json`; the runner
returns to probe screening at the next game.

## What the saving buys live

In the gate, both arms get the same worlds (the adaptive time bound is lifted).
Live, the 9 s adaptive budget gives 6 extra worlds only when the first batch took
under 4.5 s. Before this change, 58% of close turns missed them because the first
batch was slowed by background load. A cheaper first batch reinvests the saving
in more worlds on close decisions, where paper section 11 (Monte Carlo error)
says ranking is weakest.

## Operate

```
.venv/bin/python -m search.value_service --status     # "screen": current, logged/trained worlds, candidates
.venv/bin/python search/train_screen.py --team Rain-Recife-special-stat-fix   # manual candidate (MPS)
.venv/bin/python search/value_gate.py --cand-screen PATH --rule noninferior --min-sim-saving 0.2 \
    --pairs 80 --out runs/screen-gate.jsonl
cat data/ml/value/teams/<slug>/screen-current.json      # what live search uses
```

Settings in `data/ml/value/config.json`: `screen_enabled`, `screen_min_records`,
`screen_train_every_records`, `screen_max_gb` (logging stops above it; data is
never deleted) and `screen_min_sim_saving`.

## Limits

- Probe scores are themselves noisy: one chance sample per cell, four replies.
  The ranker learns to imitate the probes, not the true ranking. Only the gate
  says whether the imitation plays as well.
- Gate opponents are the search piloting real teams, and gate games are logged
  from both seats. Live positions are about a tenth of the data and serve as the
  distribution check (`live_test`).
- Self-play (`selfplay.cjs`) keeps its own cheaper screening and does not use the
  ranker yet. That is the next place the ranker would save CPU.
