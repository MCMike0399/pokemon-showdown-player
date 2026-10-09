# Always-on learning on a shared Mac

The foreground MCP process keeps latency-sensitive inference on CPU. A separate
bounded worker combines CPU simulator processes with CPU or Metal/MPS batch
training. The active player does not share the training process or GPU tensors.

For retained simultaneous collection, GPU learning and independent CPU candidate
evaluation, see [the play/training pipeline](play-training-pipeline.md). Install
with `scripts/learning_service.py install --pipeline`; the phased worker below
remains available for bounded standalone runs.

## Resource policy

Defaults reserve three CPU cores and at least 2 GB of available host memory,
cap simulator concurrency at four workers, cap measured host CPU use at 75%,
and defer when disk headroom falls below 2 GB. Available concurrency is recalculated
between waves; persistent pool processes amortize PyTorch startup. Background
processes run at nice +10 and the launchd task uses low-priority I/O.

The trainer benchmarks the actual actor-critic workload on CPU and MPS, explicitly
synchronizing GPU kernels. Auto mode selects MPS only for at least a 15% measured
speed advantage. CPU simulator workers collect games, then batch training can
use the GPU in the next phase. Foreground inference stays on CPU throughout.

Candidate training prepares immutable rollout features once instead of rebuilding
their Python lists for every optimizer/KL/likelihood pass. `RolloutBatcher` keeps
small batches resident on the selected device within a 128 MiB feature budget.
Larger histories use a bounded CPU cache with eviction and minibatch transfers.
Minibatches retain their original variable mask width, indices and probabilities;
tests compare inputs, likelihoods and gradients against the uncached path. The
feature cache budget excludes model, driver and temporary minibatch allocations,
which remain subject to the MPS allocator and host resource guards.

MPS allocations are capped at 12% of PyTorch's recommended unified-memory budget.
GPU training defaults to 50% duty through bounded pauses between minibatches.
This is an allocation/duty budget, not an OS guarantee that a fixed number of GPU
cores is reserved. The GPU is shared with macOS applications. Host-load and memory
checks defer work; explicit MPS failures restart candidate training on CPU.

```bash
.venv/bin/python -m ml.cli compute
.venv/bin/python -m ml.cli continuous
.venv/bin/python -m ml.cli continuous --run
```

Settings live in gitignored `data/ml/autopilot.json`; supported fields match
`LearningConfig` and `ResourcePolicy`. `backend` is `auto`, `cpu`, or `mps`.
Keep budgets below host capacity. The worker has a bounded wall-time budget and
an 8 GB data-root budget. It preserves experience when deferring; it never prunes
another service, changes Docker limits, kills unrelated processes or restarts
OpenClaw. Completed simulator waves can finish just beyond the cycle deadline.

## Daily feed and per-battle experience

Every recorded completed live battle finalizes its trajectory and a sanitized
public log, then queues a durable learning job. The background observer handles
the terminal event even if OpenClaw forgets a separate finish call. Duplicate
results do not produce duplicate episodes or feed entries.

Daily feeds retrieve exact-format Smogon usage, current Limitless tournament team
sheets, and public Showdown replays when the endpoint is accessible. Each source
keeps timestamps, regulation, URL and failure status. Public sheets retain player
attribution and mark undisclosed stat points. Generated spreads are separately
labelled experimental inputs and validated before simulations/matchmaking.

The small approved VGC-Bench M-B archive is checksum-pinned and MIT-labelled by
the dataset owner. Enable it by listing `vgc-bench-champions-mb` in `archives`.
It contains 324 completed public logs; they train a separate historical opponent
scout. A current Champions scout can initialize shared weights from that model,
but requires current-format held-out validation before deployment. Its games
are never relabeled as M-C, used as exact private requests or mixed into M-C PPO.

```bash
.venv/bin/python -m ml.cli feed --format gen9championsvgc2026regmc
.venv/bin/python -m ml.cli feed --format gen9championsvgc2026regmb \
  --archive vgc-bench-champions-mb
```

The scout learns executed-move distributions from the state at the start of the
turn. Labels from called moves, immobilized intents and ambiguous mid-turn
switch-ins are excluded. Later damage, moves/items and the result do not enter
earlier features. Its validation partition separates complete battles, and its
checkpoint is promoted only when held-out loss does not worsen. Reading prose
does not become a neural action label.

## Candidate evaluation and scheduling

Local practice cycles use validated team pools, random seeds, alternating player
sides and a mixture of random/tactical opponents. Each collecting revision stays
frozen. Candidates train only from compatible complete unused rollouts. Paired
evaluations compare the incumbent and candidate on identical seeds, sides and
teams. The active checkpoint changes only when the candidate beats the configured
margin and paired confidence gate, evaluations finish cleanly, and no live episode
is pending at the explicit matchmaking boundary.

The gate prevents observed regressions in that finite suite; it does not prove
general tournament strength. Jobs, checkpoints, reports and datasets stay local
and gitignored. Rejected candidates retain an audit report. Crashed job leases
can be reclaimed; a single worker lock prevents duplicate scheduled learners.

New plans default to `curriculum: "ladder-v1"`. Its repeating policy cycle includes
frozen self-play, a mechanical tactical script, the legacy heuristic, and random
play (2/8, 3/8, 2/8, 1/8 over a complete cycle). A small run may cover only part of
that cycle. Each policy completes a team sweep to avoid confounding opponent
policy with team index. `open_team_sheet_probability` defaults to 0.08 instead of
forcing sheets in every practice game. Explicit `curriculum: "legacy"` preserves
the previous schedule. Saved evaluation plans retain their original settings.

New curriculum tasks use `seed_encoding: "full-v1"`, mapping an unsigned 64-bit
integer into Showdown's four 16-bit seed words. Legacy tasks keep the original
16-bit mapping so old experiments remain reproducible. Pair validation checks
teams, sheet visibility, opponent revision and effective simulator seed as well
as the supplied seed and side. CLI practice can use `--opponent tactical` and
`--closed-team-sheets`; CLI seeds otherwise retain the legacy mapping.

To specialize the format's actor in one stored team, set `training_teams` in
`autopilot.json`, for example:

```json
{"training_teams": {"gen9championsvgc2026regmc": "My Politoed rain"}}
```

This fixes the learner's own team for new practice and paired evaluations while
rotating opponent teams and alternating player sides. PPO uses only compatible
unused episodes whose exact team fingerprint matches the focused build. Other
team experience is retained. The report records the focused name/fingerprint;
provided sets must match the format and pass simulator validation. Removing the
mapping restores general team-pool training. The actor remains one checkpoint per
format, initialized from existing weights; this is specialization rather than a
new pretrained expert. Already-declared candidates resume their original frozen
cases. Live matchmaking uses its own campaign team settings, and checkpoint
promotion still requires the normal gate and an empty recording boundary.

Install this project's own launchd task:

```bash
.venv/bin/python scripts/learning_service.py install
.venv/bin/python scripts/learning_service.py status
.venv/bin/python scripts/learning_service.py remove
.venv/bin/python -m ml.cli continuous --disable
```

The task wakes every 15 minutes, refreshes sources at most once per UTC day and
runs bounded local practice when headroom allows. Disabled learning stops new
worker processing while retaining data. No additional public Showdown accounts
or automatic public-ladder traffic are created by the scheduled worker.

Research, provenance and licenses: [continuous-learning research](continuous-learning-research.md).

## Online improvement boundaries

Per-battle jobs now accumulate at least `min_training_steps` (default 64) compatible
unused steps before PPO. Small batches remain unconsumed for a later job. PPO
reports approximate KL and clipping after each epoch and stops additional epochs
when KL exceeds its configured budget (default 0.03).

Training also validates trajectory structure and the likelihood of each original
encoded collecting action before updating weights. Invalid, unsubmitted,
out-of-order or checkpoint-incompatible episodes receive no training credit.
Reports include excluded episodes, maximum collecting likelihood error and step
counts by source. A completed recording or consumed batch does not mean that an
improved policy was promoted.

For an isolated pooled campaign experiment:

```bash
.venv/bin/python scripts/improve_campaign.py \
  --campaign data/ml/campaigns/<campaign-id> \
  --output artifacts/campaign-improvement
```

This intentionally reuses audited compatible campaign rollouts in an isolated
candidate, including previously consumed episodes, without resetting production
flags. It compares pooled PPO and a turn mechanics ablation against frozen
incumbent/research/scout inputs, first on declared development cases and then on
an independent final suite only for a development-passing candidate. It never
logs in, restarts a campaign, stages a candidate or promotes automatically.
New experiments record source hashes and refuse continued evaluation after
an inference/simulator implementation changes. Evaluation resume requires the
same saved source/checkpoints; interrupted initialization needs a fresh output
directory. Retain the interrupted artifacts rather than overwriting its plan.

Worker evaluations sample actions, matching `ps_ml_play(learn=True)`, and keep
starting research, scout weights and scout move support fixed for both policies.
They require complete paired games, the configured win margin, and a one-sided
exact paired sign-test p-value at most 0.05. These are finite scripted comparisons;
repeated candidate testing does not provide a campaign-wide false-positive guarantee.

Incomplete evaluations retain their candidate, declared cases, frozen inputs and
completed results. A later worker resumes only the remaining cases without
retraining. Sample consumption is tied to the durable candidate.

A passing candidate is staged under `data/ml/ready/`; it is not immediately installed
by a background worker. `ps_ml_ladder` checks it before another search, refuses to
replace a collecting checkpoint while any ladder recording is pending, verifies
the parent revision and candidate checksum, and promotes atomically. Write-lock
contention defers promotion while allowing matchmaking to continue. Status and
training reports distinguish a real queued job, a staged candidate and a promotion.
A manual `ps_ladder` call does not apply this ML promotion boundary.

Once a live terminal outcome has been persisted, optional public-experience,
enqueue or worker-start failures cannot make that result unfinished. The finish
response reports the failed phase and exception type without exposing arbitrary
exception text. Repeated finish calls retry the learning handoff at most three
times, preserving the same episode and durable job. A queued job remains queued
if its immediate worker launch fails; the scheduled worker can still claim it.
Core recording and outcome-attribution failures remain errors.

Hidden trapping or disabling can make a previously legal-looking choice unavailable.
Offline simulation retries up to five consecutive disclosures only when Showdown
provides an actionable corrected request. Rejected proposals receive no credit.
These recoveries are reported as `unavailable_choices`; unexpected invalid choices
still fail the evaluation gate.

Checkpoint `policy_temperature` defaults to 1.0 for older models. Sampling and
PPO likelihoods use the same temperature. A temperature change is a new policy:
assign a new revision and evaluate it on development and untouched final cases
before considering promotion. It is not a free runtime exploration knob.

New background jobs use a mixed curriculum: one quarter random opponents, one
quarter the collecting incumbent checkpoint, and one half tactical scripts.
The checkpoint opponent is sampled during collection and greedy during evaluation.
Both evaluated policies face that same frozen opponent, not a copy of themselves.
Saved evaluation plans retain their original cases across retries. This strengthens
future validation without changing completed campaign records or forcing promotion.
