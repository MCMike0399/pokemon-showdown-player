# Match data, M5 learning, and OpenClaw orchestration

Primary-source review dated 2026-10-08. This document separates sourced facts from proposed experiments. It does not report a new strength improvement or change running services. Existing repository context is documented in [continuous learning](continuous-learning.md) and the [post-campaign review](post-campaign-learning.md): CPU battle inference, a separate bounded learner that can use MPS, candidate evaluation, and promotion between games already exist. Recent candidates did not establish a significant strength improvement.

## What collecting every match can accomplish

**Sourced fact:** PPO alternates rollout collection under a fixed old policy with several optimization epochs on that rollout batch, then advances the old policy. Algorithm 1 does not specify unrestricted replay of all historical policies. Its clipped probability ratio constrains an update; it does not make arbitrary historical data equivalent to current-policy rollouts. [Original PPO paper, sections 3–5 and Algorithm 1](https://arxiv.org/pdf/1707.06347).

**Implication for this project:** keep every match for audit, opponent modeling, outcome/value experiments, mechanics diagnosis, and future algorithms, but assign explicit eligibility for each learning objective. Preserve losses and ties as well as wins: a terminal result is evidence about a sequence, not proof that every action in a win was correct. Record pre-action observations, legal action mask, selected/submitted/accepted action, behavior probability, policy revision, reward, termination versus interruption, team hash, simulator/format revision, and seed provenance. Separate public information available at decision time from post-match reveals and simulator-private information.

**Proposed data contract:** immutable raw events plus versioned derived features and a rejection ledger. Partition own-policy PPO trajectories, demonstrations, public opponent scouting, interrupted/debug traces, and evaluation games. A reconstructible behavior probability validates what generated an action; it does not alone establish acceptable policy age. Prevent repeated ingestion with episode/action IDs. Keep an untouched chronological/opponent holdout; split by complete battle and opponent rather than individual turns. Evaluation data may be used for a later explicitly versioned training cycle, but then must be replaced as holdout data.

**Sourced alternative:** IMPALA deliberately decouples actors and learners and introduces V-trace correction for policy lag. Adopting its correction would be an algorithm change, not a configuration switch that legitimizes unrestricted historical PPO replay. [Original IMPALA paper](https://proceedings.mlr.press/v80/espeholt18a.html).

**Sourced alternative:** imitation learning must account for the state distribution induced by the learner; DAgger collects expert labels on states encountered by the learning policy. Merely treating one's own selected moves as expert demonstrations does not provide that supervision. [Original DAgger paper](https://proceedings.mlr.press/v15/ross11a.html).

**Proposed reuse modes:** old matches can train a separately evaluated opponent-action predictor, a carefully validated value/outcome predictor, or supervised policy pretraining when genuine expert/teacher action labels exist. Public replays lack private requests and our behavior likelihoods, so they cannot be fabricated into own-policy PPO episodes. Auxiliary models are hypotheses that need held-out validation and a downstream battle test.

## CPU and GPU overlap: what the official APIs establish

**Sourced facts:** Apple silicon CPU and GPU use the same physical memory. Shared resources avoid traditional separate-RAM copies, but synchronized access and sometimes multiple buffers are still necessary. Apple recommends submitting larger quantities of GPU work and profiling actual bottlenecks. Therefore CPU inference and GPU learning can overlap, while still competing for shared memory resources. [Apple, Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/). Apple's optimization guidance also identifies system memory bandwidth as a performance constraint. [Apple, Optimize Metal Performance](https://developer.apple.com/videos/play/wwdc2020/10632/).

**Sourced facts:** `torch.mps.synchronize()` waits for all kernels in all streams on the MPS device to complete. Use it before and after a wall-clock measurement of GPU work so command submission time is not confused with completed training time. [PyTorch synchronization API](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html). `Tensor.tolist()` moves tensors to CPU when necessary; MPS scalar extraction also performs a CPU copy before reading the value. Repeated Python scalar checks can therefore interrupt the queued workload. [PyTorch tolist API](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html), [PyTorch MPS scalar implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Scalar.mm).

**Proposed optimization:** batch/pad masked action scores and reuse precomputed features; avoid one GPU launch and `.item()` per action or transition. Aggregate metrics as tensors and transfer them once per reporting interval. Keep latency-sensitive single-battle inference on CPU unless its end-to-end benchmark demonstrates an advantage elsewhere. Synchronize before starting a duty-cycle sleep if the duty budget is intended to bound completed GPU work. Measure data loading, feature extraction, forward/backward, optimizer work, evaluation, and serialization separately before enlarging the model.

**Sourced memory controls:** `set_per_process_memory_fraction(f)` limits that process's MPS allocator to `f × recommendedMaxWorkingSetSize`; zero means unlimited and values above one exceed the recommendation. This is neither a GPU utilization percentage nor a whole-host memory reservation. [PyTorch allocation limit](https://docs.pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html). Tensor allocation excludes the cache; driver allocation includes cached pools and MPS/MPSGraph allocations. Track both. [Tensor allocation API](https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html), [Driver allocation API](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html).

**Proposed resource policy:** retain explicit host memory, CPU, and simulator-worker headroom while benchmarking one learner. Track memory pressure/swap, host CPU, MPS driver bytes, completed updates/second, local games/hour, and decision p50/p95/p99. Raising GPU duty is useful only if it improves experiment throughput without worsening battle deadlines or other services. A tiny policy can be CPU-faster because dispatch, synchronization, feature work, and simulator throughput dominate; that is an empirical question here.

## OpenClaw's suitable role and lifetime constraints

**Sourced facts:** session-scoped MCP runtimes can persist between turns, but reset/deletion, compaction rollover, explicit Stop, relevant configuration changes, or Gateway shutdown retire them. Detached one-shot runs retire their MCP runtimes at run end; retained transcripts do not extend that lifetime. [OpenClaw MCP registry lifecycle](https://docs.openclaw.ai/cli/mcp/registry). Isolated scheduled runs dispose their bundled MCP runtimes at completion. [OpenClaw automation execution lifecycle](https://docs.openclaw.ai/automation/cron-jobs/how-it-works).

**Sourced facts:** OpenClaw supports isolated scheduled jobs and custom sessions that retain conversation context; these are different from retaining a particular executing process. Script automations can call configured MCP tools without starting a conversational turn, subject to execution budgets. [OpenClaw automation payloads](https://docs.openclaw.ai/automation/cron-jobs/payloads). Subagents run as separate background sessions and return results through the parent lifecycle. [OpenClaw subagents](https://docs.openclaw.ai/tools/subagents).

**Proposed orchestration:** OpenClaw chooses experiments, inspects job results, reviews team hypotheses, and summarizes milestones. One retained controller owns the single battle login/socket and CPU actor. An independent deterministic Python worker owns learning/evaluation jobs. Cron reads status and dispatches durable jobs; it does not repeatedly recreate the battle MCP process. A named chat session is useful for history, but should not be treated as a socket-survival guarantee. Use heartbeat/owner leases, idempotent job IDs, one learner lock, recoverable queued/running/completed/failed records, immutable candidate checkpoints, and atomic compare-and-swap promotion against the evaluated incumbent revision. Freeze actor, scout, team, and research inputs within each battle and evaluation block. Never promote mid-game. OpenClaw reasoning need not run on every turn or gradient step.

## Controlled experiment: data quantity and concurrency

The following is a proposed protocol, not an executed result.

1. **Establish a frozen baseline.** Fix format/simulator commit, team, actor checkpoint, scout inputs, reward/features, opponent mixture, open-sheet visibility, and seed generator. Use independent local opponents as well as self-play; ladder observations remain a separate target-domain check. Archive every game, including collection failures, with eligibility reasons.
2. **Test data volume.** Collect a fresh common-policy reservoir and compare nested 32/128/512-game samples from that same revision, starting every candidate from the same checkpoint. Repeat with at least five independent collection/training seeds. Report valid decisions as well as games: game lengths vary. Historical audited batches may supply a retrospective warm-start experiment, labeled separately from fresh-policy PPO.
3. **Separate data and compute effects.** Run one comparison with fixed PPO epochs/hyperparameters and another with fixed optimizer-update or wall-clock budgets. Larger batches with fixed epochs also get more optimizer work; one experiment cannot otherwise attribute improvement solely to more data. Keep policy KL, clip fraction, entropy, critic error, and likelihood/eligibility rejection counts alongside battle results.
4. **Test resource overlap separately.** With fixed data and training configuration, randomize repeated bounded windows of CPU-only learner, MPS learner alone, CPU inference/simulation plus CPU learner, and CPU inference/simulation plus MPS learner. Warm up kernels; synchronize MPS timing; log p95/p99 inference latency and games/hour together with completed updates/second and host pressure. Increase simulator workers and GPU duty one variable at a time. Perform this initially in the local simulator.
5. **Evaluate independent seeds and opponents.** Use paired full simulator seeds and both sides for candidate/incumbent comparisons. Hold scout/team inputs constant. Separate development evaluations for selecting candidates from a locked final suite. Report per-opponent and aggregate effects with confidence intervals, draws, truncations, and errors. Repeatable mean improvement matters more than one fortunate candidate. Size the final suite for the predeclared effect worth detecting; 60-game tests cannot reliably establish every small improvement. Account for multiple hypotheses when choosing promotion evidence.
6. **Test teams independently.** Freeze the policy while comparing legal team variants, then retrain the winning variant under an explicit team revision. A simultaneous team/model change cannot identify the cause of an improvement. Current-format legality and mechanics validation precede any simulation.

**Decision rule:** adopt more collection, GPU work, or orchestration complexity when it improves validated strength per hour, coverage of important decisions, or experiment throughput under the latency/resource limits. Full GPU utilization and an ever-growing database are not themselves evidence of learning.

## Measurements and changes applied locally

The user subsequently requested resuming the completed 375-game campaign until
500 total. The existing supervised controller was resumed using recorder 3 and
an explicit `completion_only` control flag. Its original overnight deadline is
preserved as history; the new continuation ends at the target. A separate
read-only monitor uses the 375-game baseline. The playing actor remains on CPU;
its weights are frozen within each recorded battle and candidate promotion
remains gated at a matchmaking boundary.

A bounded test used repeated recorded features and synthetic optimization
targets, measuring completed work with MPS synchronization. These were short
sequential windows while the live campaign and other host workloads continued;
they are diagnostic timings, not randomized comparisons or strength results.
The original raw kernel benchmark favored MPS (1.057 ms versus 1.512 ms for CPU),
but rebuilding Python feature lists dominated the actual training pipeline.

| 64-decision batch, up to 360 choices | Before cache, ms/update | After cache, ms/update |
| --- | ---: | ---: |
| CPU | 43.08 | 14.08 |
| MPS, configured 50% duty | 121.61 | 34.99 |
| MPS, bounded full-duty benchmark | 44.06 | 7.45 |

Some 32-decision cached windows deferred because host headroom fell below the
configured threshold; those omitted cases are not fabricated. Different window
loads and MPS graph caches limit direct speedup attribution. The full-duty test
does not change the production GPU duty setting. Timing metadata stays in local
ignored `artifacts/m5-continuous-learning/` files.

Implemented a 128 MiB bounded rollout tensor cache, preserving input tensors,
legal masks, likelihoods and gradients. A real production candidate subsequently
trained on MPS concurrently with live CPU play, using 72,936,608 bytes of resident
feature tensors. Its training and evaluation records remain separate from the
playing checkpoint; faster training is not a claim of better decisions.

The data directory was measured at 7.35 GiB against its former 8 GiB processing
limit. The user explicitly increased the storage allowance to **50 GiB**. No data
was deleted, and CPU, available-memory and GPU limits were retained. This allows
continued collection/learning rather than forcing dataset deletion at the old
limit. Long-term archive compression and deduplicated frozen input bundles remain
useful storage work; a larger limit alone does not solve redundant copies.

All 156 offline tests passed, including an actual MPS optimizer step and exact
CPU gradient/mask equivalence checks. The cache source loaded through the normal
between-game controller reload. The live monitor reported healthy recordings
with no campaign errors at verification. Future GPU duty or minibatch changes
should use the controlled concurrency/strength protocol above rather than assume
that a fast diagnostic window proves an optimal production setting.
