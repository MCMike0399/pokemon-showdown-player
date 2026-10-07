# OpenClaw harness and model test campaign

Send the text below to OpenClaw on the machine holding this checkout. Run evidence
stays under gitignored `data/ml/campaigns/`. The observer can inspect it with
`.venv/bin/python scripts/monitor_campaign.py` without logging into Showdown.

---

Execute a measured Pokemon Showdown campaign in this repository. Your job is to
operate the harness and produce auditable evidence of how the agent, harness,
and local ML models perform. Proceed through execution, not just a plan.

## Scope and completion

- Target format: `gen9championsvgc2026regmc` (Champions M-C).
- Target: **100 terminal ladder games**, sequentially, in ten blocks of ten.
  First run three diagnostic games; they count toward the 100.
- Also run **600 local evaluation games**: 300 for the starting actor checkpoint
  and 300 for the ending actor checkpoint, on identical held-out matchups/seeds/sides.
- Campaign budget: 24 hours. Finish the current game before a planned session
  boundary. At the budget, cancel any outstanding search and report partial
  completion honestly. A stalled unresolved battle blocks new matchmaking.
- Use the existing account and local models. The battle actor chooses actions,
  the team bandit selects teams, and the existing background worker handles
  learning. Preserve its resource limits and promotion gate.
- Preserve existing source edits, credentials, experience and services. Write
  campaign scripts/artifacts only inside the campaign directory. Report a
  reproducible harness defect separately rather than silently patching the
  system during measurement. Keep account identifiers and raw evidence local.

## 1. Preflight and durable evidence

Locate `pokemon-showdown-player` (normally `~/Developer/pokemon-showdown-player`
on the host running the MCP server). Apply its AGENTS guidance.
Read `docs/agent-workflow.md`, `docs/ml.md`, `docs/continuous-learning.md`, and the
implementations of the tools used below. Use the installed native MCP server
`pokemonshowdown`; tool names normally start with `pokemonshowdown__ps_`.
Discover the actual schemas rather than inventing arguments or another bridge.

Verify ML dependencies and the pinned simulator are ready; run
`.venv/bin/python -m pytest -q`. Verify the server's request timeout is at least
1,200 seconds. Also verify the **agent-run timeout and tool-call budget** cover
the campaign: `openclaw agent --help` advertises a 600-second default or configured
value, independently of the MCP timeout. For a CLI launch, its `--timeout` flag
sets the run budget; arrange the long run before entering a game. A prompt cannot
extend the timeout of a process that already started. Check active battles before
starting: an existing unfinished
game belongs to its current operator and is a reason to defer this campaign.
Use one retained MCP runtime for each complete game. Log in once per MCP process
with `ps_login`; reuse that connection. A detached OpenClaw run closing its MCP
child mid-game loses in-memory recording state. Reopen only between games.
Runtime reference: https://docs.openclaw.ai/cli/mcp/registry.

Create a unique `data/ml/campaigns/<UTC-timestamp>/` directory. Write
`manifest.json` with the campaign ID, UTC start/deadline, targets, format,
source Git HEAD and dirty-file list, simulator pin, baseline counts, actor
revision/update count, scout checkpoint hash, learning config, compute backend,
and OpenClaw session/runtime identifiers when observable. Existing experience
is baseline evidence, not newly played campaign games.

Atomically write `data/ml/campaigns/latest.json` with
`{"campaign_id":"...","directory":"<absolute campaign path>","started_at":"<UTC ISO timestamp>"}`.
Keep these artifacts up to date:

- `events.jsonl`: append tool start/end/error and decision/result events with UTC
  timestamps, duration, game index, room, request ID when available, selected
  team fingerprint, actor revision, actual tool name, and compact returned fields.
  Save a start event **before** a long tool call so an observer sees work in progress.
- `games.jsonl`: one terminal record per unique room, with episode/job IDs,
  actual selected team, actor revision, outcome, decisions, duration, observed
  rejections/retries, rating change if actually returned, and evidence paths.
- `status.json`: atomic status update before/after every tool call and completed
  game: phase, updated_at, active room, current tool/tool_started_at, target and
  completed counts, W/L/T, errors, starting/current revisions, last job, next step.
- `evaluations/`, `snapshots/`, `report.md`: exact input suites, frozen inputs,
  raw per-game evaluations, checkpoints, and final conclusions.

Use null plus a reason for unavailable metrics. During a blocking `ps_ml_play`,
status may remain unchanged until it returns; tool_started_at and SQLite decision
growth distinguish a long call from an idle agent. Only claim heartbeats you
actually implement. Send progress in this conversation every ten completed games
and immediately on a blocker; evidence files are the primary monitoring channel.

## 2. Freeze the comparison inputs

Before the first campaign game, snapshot the active actor, scout, and research
database into the campaign directory. Use SQLite's backup API for a consistent
database copy including WAL changes, and atomic checkpoint copies/hash checks;
never copy just a live database file. Record revision and SHA-256 hashes.

Choose four distinct, simulator-valid M-C team matchups from the existing team
pool, including different compositions. Export their exact sets/fingerprints and
provenance. Record whether stat points were generated. If fewer compositions
are available, state the reduced coverage; a fixture mirror proves only a narrow
matchup. Reserve a new seed range and these matchups from campaign-directed
training. Record existing scheduled-training overlap rather than calling all
teams unseen. Do not choose the final suite after seeing candidate results.

Evaluate each actor on four matchups: **50 games per matchup versus heuristic**
and **25 per matchup versus random**, for 300 per actor. Use identical seed ranges
for the start/end actor, alternate learner sides, and preserve identical team
orientation. Use separate evaluator roots, isolated from `data/ml/` writers.
Both actor evaluations use the **same starting scout and research inputs**;
otherwise a scout/data change confounds the actor comparison. Record those hashes.

The existing CLI supports `--root <root> evaluate --format <format> --team1 <file>
--team2 <file> --games <n> --seed <seed> --opponent heuristic|random --alternate-sides`.
The global `--root` precedes the subcommand. Load the copied checkpoints, play
deterministically without recording PPO, and keep them unchanged throughout
evaluation. Verify hashes before/after. Start with a small smoke check of one
game per matchup, using seeds outside the measured suite; investigate a failed
smoke check before executing hundreds of games.

Run evaluations sequentially at low priority with one Torch thread; check the
existing resource headroom before each small batch and defer under pressure.
They must preserve at least the learning service's CPU/memory/disk reserves.
Write incremental per-game output. Existing CLI local/training commands update
weights immediately and bypass automatic promotion: use the worker for production
learning and reserve these roots for frozen evaluation. An evaluation exception
counts as a failed attempted game, not a loss or a vanished sample.

## 3. Play and audit the three diagnostic ladder games

Call `ps_ml_status(format)` and `ps_learning_status()`; save the actual results.
Ask `ps_team_plan(format, evidence_source="ladder", explore=False,
save_as="Campaign-<id>-base")` for a validated model-selected team. If none exists,
create the named fixture from `examples/champions-rain.json` and label it self-built.
Login if this runtime has not logged in. Respect Showdown's rules and play one game
at a time; perform no account creation or unsolicited chat.

For each game, use `ps_ml_ladder(format, team=<base>, explore_team=False)`, then
`ps_ml_wait(timeout=120)`. Capture the **returned selected team** and use the
team returned by `ps_ml_wait` in all battle tools. Even with explore_team=False,
the team planner can select a researched variant: the actual fingerprint, not
just the requested base name, defines the matchup.

For these first three games, inspect `ps_result(room)` and `ps_waiting()` before
acting; `ps_waiting` can retain a finished room's old request. Call
`ps_ml_choose(room, format, team=<actual team>, explore=True)` for each actionable
request. Poll with modest backoff when waiting on the opponent. Persist returned
choice, sampled/recorded flags, revision, policy probability, value, top choices
and opponent predictions. Record duplicate acknowledgements and server errors
separately; a submission acknowledgement alone does not prove server acceptance.
The tool owns legal joint actions, retries and recording. OpenClaw must not
substitute its own moves or turn recommendations into imitation labels.

After a verified terminal result, call `ps_ml_finish(room)` twice to check
idempotency on diagnostic games. Cross-check the episode ID/room in SQLite:
one complete episode, one terminal reward, a fixed actor revision/team per
episode, and no duplicate credit/job. Preserve observed PPO steps/log probabilities
and the associated sanitized public-game/scout evidence. Expected rewards are
win +1, loss -1, tie 0; unresolved games retain no terminal reward.

## 4. Complete the remaining ladder blocks

For subsequent games use the same search/wait flow, then
`ps_ml_play(room, format, team=<actual team>, learn=True)` for the full game,
followed by terminal result/finish verification. This delegates decisions to the
actor while keeping the connection alive. Record the returned episode and query
its stored trajectory for decision/revision evidence. Full-game tools do not
return every diagnostic metric; label unobserved latency/rejection counts as
unavailable, not zero.

Games 1–50 request the same base team with explore_team=False. For games 51–100,
allow the team model to explore with explore_team=True using that base and its
validated variants; record the exact selected team each time. Report phases,
team fingerprints and collecting actor revisions separately. Background promotion
can change the actor between games: these are adaptive ladder observations, not
a controlled proof that later play is stronger.

Let terminal recording enqueue its normal learning job. Between blocks inspect
`ps_learning_status`, jobs and `data/ml/candidates/*/report.json`. Log trained
episodes/steps, candidate revision, backend, incumbent/candidate paired wins,
evaluation completeness/rejections, promotion decision/blocker and scout held-out
loss/sample counts. A job's status="complete" can still contain an error.
The episode trained flag means consumed by candidate training, not successful
promotion. If jobs defer for resources, record deferral and back off rather than
flooding the queue. Preserve the existing daily feeds/scheduler; extra feed or
practice jobs are unnecessary for this campaign.

## 5. Recovery and stopping rules

- Search timeout: inspect status, global server errors and battle rooms; confirm
  no active game, cancel the existing search with
  `ps_send(room="", message="/cancelsearch")`, and retry with 30/60/120-second
  backoff. After three failed searches, checkpoint the campaign as blocked.
  Never start a second search while a game may already be assigned.
- Unfinished/stalled play: inspect the **same room**, result, waiting request and
  recent protocol errors. A healthy retained process may resume ps_ml_play in that
  room at most twice with the same team/mode. No new matchmaking until terminal.
- A tool timeout can leave the previous operation running. Inspect it before
  retrying. A per-room player lock error is a reason to wait, not spawn another
  connection. Transport failure, repeated rejected choices or a lost MCP process
  means save a blocker with active room and pending episode, then stop this lane.
  Disk evidence survives but this version cannot restore an interrupted recording
  session after process death. Never fabricate recovery or an outcome.
- Stop for inconsistent reward/episode attribution or duplicate terminal credit.
  Preserve the minimal evidence and concrete failing call rather than proceeding
  with contaminated measurements. Do not forfeit games just to satisfy the count.
- Resume a campaign only at a verified game boundary: reload manifest/events,
  reconcile existing room/episode IDs, count only terminal campaign games, and
  continue the remaining count using the configured account.

## 6. Evaluate the ending checkpoint and report

At completion or budget expiry, finish any healthy active game, freeze/hash the
then-active actor, and run the remaining 300 matched evaluation games using the
same start scout/research/suite. Keep promotion-gate evaluations separate from
this campaign's held-out suite. If the start/end revision is identical, report
no actor change; do not invent an improvement. Track scout learning separately.

Write report.md with three evidence-backed assessments:

1. **Agent operation:** real native MCP calls, connection/session behavior,
   decision delegation, observed overrides, errors/recovery, completion versus
   target and progress/liveness gaps.
2. **Harness reliability:** attempted searches/games, terminal and unfinished
   counts, observed invalid choices/duplicates/timeouts/disconnects, recording
   correctness, job outcomes/backlog, CPU/MPS use and resource deferrals.
3. **Model performance:** ladder W/L/T with 95% Wilson intervals, grouped by
   exact actor revision/team/phase; frozen start/end results per matchup/opponent
   with paired win-to-loss/loss-to-win changes and uncertainty; actor update and
   promotion history; scout held-out losses/sample coverage; team-bandit outcomes.

Use terminal games for win-rate denominators and report unsuccessful attempts
separately. Never combine local evaluation/training games with ladder results.
Policy probability is an action preference, value is a reward estimate, and lower
PPO/scout loss or more updates alone do not establish stronger battling. The
scout's repeatedly used validation split is not an untouched final test set.
Quote artifact paths for every material claim and disclose unavailable metrics,
training-team overlap, adaptive ladder opponents, and narrow scripted coverage.

Completion means the targets are actually met, or a clearly identified blocker/
budget limit is recorded with exact remaining work and a truthful partial report.
Return the absolute campaign directory and a compact result table in this chat.
