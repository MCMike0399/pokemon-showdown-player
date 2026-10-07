# OpenClaw player workflow

Use one retained OpenClaw session and one configured Showdown account. The native
server exposes tools as `pokemonshowdown__ps_*`. A detached agent run closes its
MCP child when the run ends, so complete the whole battle inside that run.

1. `ps_team_plan(format, evidence_source="ladder", save_as="Planned team")`
   asks the learned team-outcome model to select and validate sets. Check its
   citations and generated-spread flag. Use `explore=True` for team experiments.
2. `ps_login()` logs in once using the configured account.
3. `ps_ml_ladder(format, team="Planned team")` selects from that team and its
   researched variants, validates, uploads and searches on the same connection.
4. `ps_ml_wait(timeout=120)` returns the new room and the exact selected team.
5. `ps_ml_play(room, format, team, learn=True)` lets the neural model make every
   battle decision. The tool internally waits for requests, constructs complete
   legal joint actions, samples the policy and submits the selected action.
6. `ps_ml_finish(room)` returns the idempotent terminal record and learning job.
   An observer also finalizes completed recorded battles if this call is omitted.
7. `ps_learning_status()` reports the feed, jobs, headroom and compute backend.

For turn-by-turn control, use `ps_ml_decide` to inspect a recommendation and
`ps_ml_choose` to submit and record an actual model choice. Plain `ps_choose`
submissions are not automatically treated as PPO samples because the collecting
distribution is unknown. Expert overrides use `ps_ml_demonstrate`, then request
imitation training with `ps_ml_train(imitation=True)`.

## Integration contract

- The MCP harness owns credentials, the socket, requests, legal masks, team
  fingerprints, submission retries and terminal outcome verification.
- The battle actor-critic owns action scoring and selection. A public-log scout
  predicts executed opposing moves and contributes bounded belief features.
- The team model is a Beta-Bernoulli bandit over exact validated team versions.
  It learns which teams this policy pilots successfully. Generated stat spreads
  are harness hypotheses, not attributed tournament-player spreads.
- The background learner owns CPU/MPS training, data ingestion and evaluation.
  It saves a candidate separately and promotes only after a matched-seed/side
  evaluation improves win count by the configured margin without invalid choices.
  Fresh pending live episodes block active-checkpoint replacement.
- MCP reloads promoted checkpoints between games. Active recorded games keep the
  revision that generated their log probabilities. Historical off-policy games
  remain useful team/scout evidence but are not replayed as fresh PPO rollouts.

`ps_ml_play` holds a per-room lock. It acquires the model lock only around each
choice/finalization, so status and research calls stay available during a battle.
Background learning uses another low-priority process and a persistent job queue.

## Native OpenClaw setup

Install the Python/Node dependencies in [the setup guide](ml.md), then configure
the stdio server with the absolute interpreter and server paths from
`examples/openclaw-mcp.json`. No credentials belong in the MCP config. Use:

```bash
openclaw mcp add pokemonshowdown \
  --command /absolute/path/to/pokemon-showdown-player/.venv/bin/python \
  --arg /absolute/path/to/pokemon-showdown-player/ps_mcp_server.py
openclaw mcp configure pokemonshowdown --timeout 1200 --connect-timeout 30
openclaw mcp probe pokemonshowdown --json
```

A new retained session gets a fresh MCP child. `openclaw mcp reload` affects the
calling runtime, not every already-running gateway session. No gateway restart
is necessary for a fresh integration-test session. See the official
[MCP registry and runtime lifecycle](https://docs.openclaw.ai/cli/mcp/registry).
