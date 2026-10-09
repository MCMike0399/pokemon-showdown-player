# Browser play with the local model

`scripts/browser_play.py` replaces the protocol player's transport with
Playwright clicks in the official Showdown client. The existing local PyTorch
`Brain` chooses each complete legal action by default. With `--search`, the local
checkpoint chooses team preview and local official-simulator search chooses turns. OpenClaw, MCP and LLM inference are
not involved in the turn loop. This does not start the retained learning pipeline.

Install the optional browser dependencies and Chromium:

```sh
uv pip install --python .venv/bin/python -r requirements-browser.txt
.venv/bin/python -m playwright install chromium
```

Check the browser without logging in or starting games:

```sh
.venv/bin/python scripts/browser_play.py --inspect
```

Stop the existing campaign and any other account controller first. Both transports
share `data/ml/ladder-owner.lock`. Supply the existing registered account through
`.env` and an already selected checkpoint under `data/ml/models/`.

```sh
.venv/bin/python scripts/browser_play.py \
  --team 'My saved team' --format gen9championsvgc2026regmc --games 1 --headed
```

The team comes from the gitignored `teams.json`. The controller imports its exact
sets through the teambuilder, selects the format/team and clicks Battle. The
default is one game; `--games N` explicitly bounds a longer run.
`--continuous` keeps playing until signaled to stop or storage admission closes;
`--resume-active` resumes a single active account room after login. `--chrome` uses
installed Google Chrome; otherwise Playwright's Chromium is used. The persistent
profile is `data/browser-profile/`. A different logged-in account is refused.

For an interrupted game, pass `--room battle-...` with the original format/team.
Existing battles must resolve before another search. Ambiguous submissions,
partial UI choices and incompatible recording checkpoints require inspection;
the controller stops instead of silently choosing a fallback. SIGINT/SIGTERM
finishes the active game and prevents the next search. A stall retains the pending
recording and assigns no terminal reward.

The controller reads the browser's own request and public battle log. It uses
`battle_state.legal_choices` and `observe` to build the same model input, without
opening a second Showdown socket. JavaScript evaluation reads state; game actions
use visible buttons/checkboxes. Team preview tracks reordered button positions;
doubles use both move slots, positive opponent/negative ally targets, evolution
checkboxes and forced-switch position buttons. The official client fills pass
slots. Unsupported/default actions stop for inspection. This targets the official
classic client (`app.rooms`); client DOM/API changes may require adapter updates.
Offered open team sheets are accepted through the visible client button; only
actually disclosed opposing sets enter the observation.

Sampled decisions retain their collecting likelihoods, exact checkpoint and
private snapshots. Terminal games finalize through `Brain.finish` without
enqueuing training jobs. `--no-record` uses greedy inference and writes no PPO
episode. Local event logs, terminal protocol and browser inspection evidence live
under `data/ml/browser-runs/`; they are private and excluded from publication.
No new weights are trained or promoted by this command.

## Server pacing

Showdown's public [resource monitor](https://github.com/smogon/pokemon-showdown/blob/master/server/monitor.ts)
limits battle starts, validations and concurrent games; its
[user command handling](https://github.com/smogon/pokemon-showdown/blob/master/server/users.ts)
also throttles commands. The
[matchmaker](https://github.com/smogon/pokemon-showdown/blob/master/server/ladders.ts)
has restrictions for accounts marked as bots in suspect tests and ladder tours.
These sources do not establish a need for human-like random mouse/click delays,
or describe every control deployed on the live service.

The controller runs one game at a time, spaces model submissions at least one
second apart, and waits 30 seconds between games (`--game-delay`). It stops on
matchmaking rejection and does not automatically retry or disguise automation.
Normal DOM readiness and request freshness remain required after every delay.

## Live viewer

Browser snapshots publish filtered battle frames to `data/ml/live-watch/` on the
same observation loop, without a second Showdown connection. The retained viewer
automatically follows the newest browser run's event ledger instead of an older
campaign. It counts verified SQLite outcomes, including resumed games, and uses
recorded observations as a fallback for browser processes started before relay
support. Completed browser games remain replayable from their terminal logs.
Requests, chat, authentication and account names are excluded from viewer frames.

## Retained player and CPU/MPS learning

On macOS, start the continuous browser player independently of an agent session:

```sh
.venv/bin/python scripts/browser_service.py install --team 'My saved team'
.venv/bin/python scripts/browser_service.py status
.venv/bin/python scripts/browser_service.py remove
```

Use `--room battle-...` on installation to recover a known interrupted room.
The service owns the same account lock, uses installed Chrome, local checkpoint
preview and six-world/three-engine CPU search. It stops for server rejection,
ambiguous ownership or transport failure; it does not retry blindly. Removal
signals the player to finish its active game before unloading the service.
Logs are under `data/ml/logs/browser-player*.log`; the viewer follows its event ledger.

The independent `ml.pipeline` service retains CPU collection and evaluation plus
one MPS learner. `autopilot.json` selects `resource.backend: "mps"`; actual
training reports record device, optimizer steps and MPS allocation, so configuration
alone is not evidence of a GPU update. Inference remains on CPU. The configured
CPU/RAM/GPU duty and [storage limits](model-storage.md) remain in effect.
Search-game public logs supply scouting data; PPO uses eligible current-policy
local rollouts and affects the checkpoint used for preview. The search value net
is a separate experiment and is not trained by this PPO service.

An optional open-team-sheet offer can disappear while the client renders. Its
click has a one-second timeout and is attempted once per game. An expired offer
triggers a fresh request read; connection errors and ambiguous move submissions
still stop the player. Unavailable opposing sets remain unknown.

Before Battle is clicked, the controller waits for the exact selected team name,
format and nonempty saved payload in the official client. This prevents empty-team
validation requests while import or team selection is still settling.

Before each new search, the browser runner consumes a staged candidate through
the existing source/parent/checksum/evaluation gate. Resumed rooms keep their
current checkpoint. A successful promotion reloads the CPU model for the next
preview; workers never replace the playing checkpoint themselves.
