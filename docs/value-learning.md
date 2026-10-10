# Search value learning: every game feeds the evaluation the live search uses

## Why this loop exists

Live turns are chosen by simulator search (`docs/search-agent.md`). The PPO
actor-critic only chooses team preview live. Before this change:

- each live search game kept its choices, but threw away everything the search
  computed, and its positions were never labelled with the result;
- CPU self-play and MPS PPO trained an actor whose gate measured the sampled
  self-play policy, not the deployed search, and gradient updates rarely
  passed (paper, Section 14);
- the learned value for search existed but was untrained, ungated and off.

The search's leaf evaluation is the one learned component that acts on every
live turn. This loop trains it from every game, on CPU and MPS, and gates it
against the deployed configuration before it can reach live play.

## The loop

```
live game (search)            self-play (CPU, node)            ledgers + shards
     | ledger: request,              | Rain vs real-team pool          |
     | search trace, result          | rows {x,y,g,s,f,h,v}            v
     v                               v                       train (MPS, under
ledger_positions.py  ------->  data/ml/value/  ------------>  worker.lock)
 rebuilt worlds, outcome        live/ selfplay/                    |
                                                                   v
                     current.json  <---- promote <---- paired gate x2 (CPU)
                          |                          dev, then fresh-seed confirm
                          v
             live runner adopts it at the next matchmaking boundary;
             self-play adopts it for the next shard (expert iteration)
```

`search/value_service.py` (LaunchAgent `dev.pokemon-showdown.value-learning`)
runs the loop:

| Step | Device | What it does |
|---|---|---|
| Ingest | CPU | Every finished live search game becomes rows: the same tracker, priors and rebuild as the live search, labelled with the verified result (`search/ledger_positions.py`). |
| Produce | CPU (all cores) | `search/selfplay.cjs` shards of 20 games, the deployed team on one side with probability 0.6. Rows carry game id `g`, side, focus flag, hand evaluation `h` and the turn's one-turn equilibrium value `v`. Uses the promoted value. |
| Train | MPS | `search/train_value.py`: game-level splits; older live games calibrate (temperature, blend β), the newest 40% are an untouched test; the prescreen requires the deployed blend to beat the calibrated hand evaluation there and on self-play validation. Holds `data/ml/worker.lock`, so it never overlaps the PPO learner on the GPU. |
| Gate | CPU | `search/value_gate.py`: paired real-team games, same seeds/opponents/sides/worlds, deployed search config. Two stages (development, then confirmation on fresh seeds), one-sided exact sign test at 0.05 each, futility stop after 40 pairs, at most one promotion per day. |
| Promote | - | `data/ml/value/current.json` (atomic, SHA-256 verified on every read). A damaged or edited candidate falls back to the hand evaluation. |

Producers and gate games take locks from the same `data/ml/simulator-slots`
pool as the PPO collector and evaluator (the value lanes take the extra
`slot_limit_extra` slots first), and start only while
`ml.resources.ResourcePolicy` admits another worker (the shared CPU cap is
90%). One producer starts per tick, so load ramps up gradually. Children
inherit their lock descriptors: a slot is held exactly as long as the process
that uses it, and each child runs in its own session, so restarting the
supervisor never kills a shard or a gate.

## Live priority: every core for background work, all of them for live turns

macOS QoS was measured three ways on this M5 (4 performance + 6 efficiency
cores):

| Background work at | Live decision (median) | Background speed |
|---|---|---|
| default QoS (nice 10-15) | 10 s (from 2.3 s) | full |
| `taskpolicy -b` / `-c utility` | 2.3-3 s | starved: priority 4, ~17% of a core per process |
| default QoS, paused while live search runs | 2.4 s with adaptive worlds | full between decisions |

The last row is deployed. The service publishes its children's process groups
in `data/ml/value/background-pgids.json`; the live runner wraps each decision
in `search.value_store.LivePriority`, which sends SIGSTOP to those groups
(only if their leader is still a `search/` program) and SIGCONT afterwards. A
marker file records the pause; if a runner dies mid-decision, the service
resumes everything once the marker is 45 s old. The LaunchAgent runs as
`ProcessType Standard`, because a Background job's children inherit the
throttled role.

## Units: learned values on the hand-evaluation scale

The trainer fits the hand evaluation's calibration `p = sigmoid(a*h + b)` and
stores `(a, b)` with the weights. Omniscient self-play is more decisive than a
ladder game, so the net is overconfident live (first candidate: live AUC 0.83
vs the hand evaluation's 0.75, but worse log loss); a temperature `T` fitted
on the older live games fixes the scale. The engine converts the net's logit
`z` to hand units, `u = (z/T - b) / a`, clipped to ±20 (below the ±30
terminal score).
So `v = (1-β)·h + β·u` blends two estimates of the same quantity, and the
search's other constants (opponent softmax temperature, missing-world
penalty) keep their meaning.

## Decision noise (measured, not assumed)

`search/measure_noise.py` recomputes live ledger positions and counts how often
two repeats choose different actions:

- `chance_crn_off/on`: the same sampled worlds, different simulator seeds,
  without and with common random numbers (`crn`: a cell's seed depends on the
  opponent action, not on our row);
- `worlds_K/2K`: freshly sampled worlds, 6 vs 12.

Results are in `artifacts/value-learning/noise-60.json` (private).

## First night (2026-10-09/10): what each game bought

- Live: 121 games 63-58 overnight, rating 1285 -> 1368 against a median 1314
  opponent; each finished game added about 29 labelled positions and a search
  trace per turn.
- Learning: 12 hourly candidates (about 22 s each on MPS), all beating the
  calibrated hand evaluation on the newest held-out live games; the edge grew
  with self-play volume (log loss edge 0.012-0.015 at ~100k rows, 0.020-0.030 at
  130-160k). The first full gate still ended 10 gains / 9 losses: a better
  outcome predictor has not yet been a measurably better leaf evaluation. With
  about 24% discordant pairs, an 80-pair stage only detects large effects.
- Training on the older live games as well did not improve prediction on the
  newest ones (about 90 games against 160k self-play rows), so live games stay
  a calibration and test set (`--live-weight` keeps the option).
- The GPU is nearly idle by design (each value or PPO update takes seconds);
  simulation is the bottleneck, and the gate is its slowest consumer. Gates
  now stop early as a rejection once a candidate has played 40 pairs without
  more gains than losses; that rule can never cause a promotion.

`search/learning_report.py --hours 24` prints this picture at any time.

## Data organised by team

Everything is stored per team (`search/team_data.py`), so a new team never
mixes into an old team's data:

```
data/ml/value/teams/index.json        every team: games, wins, losses, positions, dates
data/ml/value/teams/<team-slug>/
    team.json                         name, format, team_id (fingerprint), exact sets
    games.jsonl                       one line per finished live game: room, time, result,
                                      our and the opponent's rating, turns, opponent preview,
                                      open team sheets, decisions, positions
    live/live-<room>.jsonl            labelled live positions (rows carry `team`)
    selfplay/sp-<n>.jsonl (+ .npz)    self-play with this team as the focus side
    candidates/value-<sha>.json       value nets trained for this team
    gates/<sha>-<stage>.jsonl         their paired gates
    current.json                      the gated value live search uses for this team
data/ml/browser-runs/<run>/events.jsonl   raw ledgers; each run logs a `team` event and
                                          every terminal event names its team
```

Older runs (before the `team` event existed) are identified exactly: several
saved teams share species and items and differ only in stat points, so a game
is matched by the exact stats in our requests against each saved team's
simulator stats. All 308 games through 2026-10-10 are
`Rain-Recife-special-stat-fix`. `python search/team_data.py` rebuilds the
catalog; the service does so after every ingest.

### Switching to another team

1. Save the team in `teams.json`.
2. Live: change `--team` in the `dev.pokemon-showdown.browser-player` plist and
   reload it between games (SIGTERM finishes the current game first).
3. Value learning: set `"team"` in `data/ml/value/config.json`. The service
   starts that team's own self-play, candidates, gates and pointer; the old
   team's data stays untouched.
4. PPO preview training: set `training_teams` in `data/ml/autopilot.json`.

For offline training elsewhere, copy `data/ml/value/teams/<slug>/` and
`data/ml/browser-runs/`.

## Operate

```
.venv/bin/python -m search.value_service --status        # what it is doing
cat data/ml/value/current.json                           # what live search uses
.venv/bin/python search/train_value.py                   # manual candidate
.venv/bin/python search/value_gate.py --candidate PATH --beta 0.75 --pairs 80 --out runs/g.jsonl
.venv/bin/python search/measure_noise.py --positions 60
```

`data/ml/value/config.json` overrides the defaults in `value_service.DEFAULTS`
(producers, shard size, train cadence, gate pairs/slots, promotions per day).
Rollback: delete or edit `current.json`; the runner returns to the hand
evaluation at the next game.

## Limits

- Live rows are about 10 positions and one result per game; they are a test set
  on the real distribution, not the training set.
- Self-play is omniscient search against search; the live opponent is a person.
  The gate's opponent is the search piloting teams real opponents brought.
- A better outcome predictor is not automatically a better leaf evaluation for
  one-turn search. Only the paired gate decides.
- With about 40% discordant pairs, an 80-pair stage detects effects of roughly
  +12 points or more. Smaller real gains will usually fail, by design.
