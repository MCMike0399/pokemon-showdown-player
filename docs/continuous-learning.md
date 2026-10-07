# Always-on learning on a shared Mac

The foreground MCP process keeps latency-sensitive inference on CPU. A separate
bounded worker combines CPU simulator processes with CPU or Metal/MPS batch
training. The active player does not share the training process or GPU tensors.

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
margin, evaluations finish cleanly and no recent live episode is pending.

The gate prevents observed regressions in that finite suite; it does not prove
general tournament strength. Jobs, checkpoints, reports and datasets stay local
and gitignored. Rejected candidates retain an audit report. Crashed job leases
can be reclaimed; a single worker lock prevents duplicate scheduled learners.

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
