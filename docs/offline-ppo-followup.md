# Offline PPO findings and M5 follow-up

The completed experiment compared seven settings across three training seeds,
with five updates per setting. Accepted complete trajectories and separate
initial, development and final evaluations are retained in the private archive.
The reproducible analysis is `python -m experiments.analyze_results`.

The initial heuristic policy was already strong against random opposition.
The training matrix used random opponents, and the final tactical scores did
not establish an improvement over the existing control. Paired development
comparisons and final comparisons against the control did not pass their
multiple-comparison corrections. Larger-collection arms also used substantially
fewer optimizer steps, confounding data size with optimization exposure.
Retain the live incumbent, mixed tactical/self-play curriculum, learning rate
`3e-4`, entropy coefficient `.01`, four epochs, 32-decision minibatches and KL
limit `.03`. Production candidates still need the existing paired promotion gate.

The replay scout improved held-out executed-move prediction, but its matched
battle comparison found no changed outcomes. Recorded-state probes showed weak
actor response to scout changes. A future `matchup-v1` experiment should start
from the incumbent, use the existing zero-residual migration, collect fresh
compatible trajectories, and pass an independent battle gate. Neither a new
architecture nor the notebook's fresh actors are automatically installed.

## Implementation improvements

Training now accumulates detached loss/entropy diagnostics and performs KL/clip
readback once per epoch. Ordered Python summation, the whole-batch KL gate,
finite-loss check, masks, optimizer updates and resource/duty checkpoints are
preserved. Resident action/mask tensors are narrowed before selecting minibatch
rows. CPU rollout processes seed the CPU generator, and CPU collection/evaluation
lanes skip Metal allocator setup. The learner retains its separate MPS lane.

`LearningConfig` exposes `learning_rate`, `entropy_coef`, `minibatch_size`,
`training_epochs` and `target_kl`, with existing defaults and validation. MPS
fallback receives the same settings. Status summaries retain those settings,
optimizer exposure, entropy, KL and clipping for diagnosis.

`ResourcePolicy.evaluation_workers` can reserve more of the existing shared
simulator budget for evaluation; collection receives the remainder. The sum
stays within `max_workers`. Default allocation is unchanged. Calibrate this on
the actual shared host rather than copying the M5 Pro's intensive worker count.

## DiveMac calibration

Bounded trials use captured compatible production trajectories and cloned
checkpoints; they never consume live experience or replace the actor. The
before/after trainer passed parameter, optimizer, diagnostic, exclusion and KL
checks on CPU and MPS. MPS trials use explicit synchronization. Full duty removed
the large measured scheduling penalty of 50% duty on this small cached workload;
diagnostic readback changes provided a smaller additional gain.

CPU evaluation trials use identical saved teams, seeds, sides, policy and frozen
inputs. Shared-host pressure initially deferred trials; those results remain
marked incomplete and are excluded from throughput selection. Completed trials
with a modestly higher CPU admission ceiling favored three evaluator processes
over two with unchanged outcomes. The calibrated local configuration retains
three reserved CPU cores, 2 GiB available RAM, the existing nonzero MPS allocator
cap, critical-pressure/swap guards, and the 150 GB retained-data budget. Detailed
timings and the exact deployed configuration remain private local artifacts.

These are cooperative limits, not OS quotas. Full training duty is useful when
an eligible batch exists; stale trajectories are not refitted to fill the GPU.
Live browser/search stays on CPU, and promotion remains before new matchmaking.

## Archive and cleanup

`python -m experiments.archive_results build OUTPUT` selects evidence only after
an exact canonical/shard audit. It retains canonical databases with consumption
flags, reports, manifests, original/final/collecting weights, sources, corpus
provenance, the failed smoke trace and rendered notebook. Database compression
is checked against the original SHA-256 and size. Collecting checkpoints are
placed at each referring run's recorded relative path; duplicate complete-run
shard databases and temporary benchmark fits are omitted.

`python -m experiments.archive_results verify OUTPUT --integrity` checks every
transferred file, byte-exact database decompression and SQLite integrity. Verify
on the destination before deleting the source. The retained archive is under
`data/research-archive/`; data, weights, reports, runtime caches and credentials
stay out of public source history. The notebook can discover the prepared
archive and display its saved reports without generating new games.
