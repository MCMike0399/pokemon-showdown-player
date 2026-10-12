# MLOps: training campaigns on Macintosh, live play on DiveMac

DiveMac plays live games and collects data. Every 1000 new live games you run a
**training campaign** on Macintosh, the more powerful Mac: a big self-play
simulation, one training run, and the two-stage paired gate. You then bring back
whatever passed, and DiveMac deploys it to live play. Nothing moves between the
Macs on its own. Code: `search/mlops.py`.

Macintosh sessions are bursts of at most about 4 hours: self-play, MPS training
sweeps, a gate stage that fits. Anything longer (400-pair gates, continuous
collection, monitoring) runs on DiveMac, as pausable background groups next to
live play (`data/ml/value/extra-pgids.json`, picked up by the value service in
any role). After a burst, copy its results to DiveMac and remove Macintosh's
data copies.

```
DiveMac (live, always on)                    Macintosh (trainer, only during a campaign)
  browser-player plays, value service
  (role live) turns games into data
  1000 new games -> macOS notification
  campaign start ──── code + data snapshot ──> value service, role trainer, campaign:
                                                  1. self-play simulation (N games, all cores)
                                                  2. train value net + screening ranker (MPS), once
                                                  3. gate each: dev, then confirm on fresh seeds
                                                  4. export what passed, idle
  campaign status <── progress ──────────────
  campaign finish <── bundles, new self-play, gate logs; trainer stopped
  deploy (checks) -> teams/<slug>/current.json | screen-current.json
  browser-player adopts the new model at its next matchmaking boundary
```

## What is trained and deployed

The two models live search uses every turn, per team:

- value net (`current.json`), blended into the search's evaluation with beta;
- learned screening ranker (`screen-current.json`), which replaces probe simulations.

PPO (team preview) keeps its own pipeline on DiveMac and isn't part of this flow.

## A campaign, step by step (all commands on DiveMac)

```sh
.venv/bin/python -m search.mlops status                    # live record, live models, games since last campaign
.venv/bin/python -m search.mlops campaign start            # Macintosh awake + plugged in
.venv/bin/python -m search.mlops campaign status           # phase: selfplay -> train -> gate -> done
.venv/bin/python -m search.mlops campaign finish           # when done: pull, stop trainer, deploy
```

- `campaign start` refuses until 1000 new live games have arrived since the last
  campaign's snapshot (`--force` overrides). Its size is `--selfplay-games`
  (default 5000, in `data/ml/value/mlops.json` as `campaign_selfplay_games`).
  Training reads up to 4M rows (`campaign_train_max_rows`).
- A campaign takes hours: self-play, then about 80 paired gate games per stage,
  per model. Macintosh must stay awake. The trainer runs under `caffeinate -i`,
  so it won't idle-sleep while plugged in, but closing the lid on battery still
  sleeps it. Self-play and gate games survive a trainer restart and resume.
- `campaign finish` refuses until the phase is `done`. `--force` collects what
  is finished and stops the trainer. `--no-deploy` leaves the bundles in
  `data/ml/value/inbox/` for a later `.venv/bin/python -m search.mlops deploy`.
- **Keep the play code unchanged during a campaign** (`search/*.cjs`,
  `search/config.py`, `agent.py`, `priors.py`, `tracker.py`, `battle_state.py`,
  the simulator version). A model gated under other code is stale and isn't
  deployed. `campaign status` warns when the code has drifted.

## Deploy checks (all must hold, otherwise nothing changes)

1. The weights hash matches the manifest, so a damaged copy is rejected.
2. Both gate stages are complete and passed.
3. The bundle is for the deployed team.
4. The play-code fingerprint matches DiveMac's code now.
5. The incumbents it beat are still the live ones (value and screening).
6. No more than `max_promotions_per_day` promotions per kind (rollbacks don't count).

Rejected and stale bundles are never retried. A bundle whose copy was cut off
(unreadable or hash mismatch) waits for the next `deploy`, and a damaged bundle
can never pass. Every outcome goes to `data/ml/value/mlops-deploys.jsonl`.
Every promotion goes to the team's `promotions.jsonl` / `screen-promotions.jsonl`
with `source` = trainer host. Campaign starts and finishes go to
`data/ml/value/campaigns.jsonl`.

There is one promotion path. If DiveMac's own value service runs in role `full`,
it also exports its passed candidates into its own inbox and promotes them
through the same `deploy_inbox`, under a lock and with a single promotion ledger.

## Undo

```sh
.venv/bin/python -m search.mlops rollback --kind value            # previous promotion
.venv/bin/python -m search.mlops rollback --kind screen --to none # back to probe screening
```

## One-time setup

With Macintosh awake, plugged in and logged in:

```sh
.venv/bin/python -m search.mlops bootstrap
```

It's idempotent. It checks node, Homebrew rsync and uv on Macintosh, creates the
trainer checkout `~/Developer/pokemon-showdown-trainer` (separate from the dev
copy), copies the code and inputs (~1 GB: frozen preview model, usage stats,
node_modules), builds `.venv` (torch with MPS checked) and writes the trainer's
config. It installs `dev.pokemon-showdown.value-trainer` **without loading it**:
`campaign start` loads it and `campaign finish` unloads it. It also switches
DiveMac's value service to role `live`, so DiveMac only collects; pass
`--keep-divemac-training` to skip that. `.venv/bin/python -m search.mlops role full|live`
changes it later.

## Transfer details

- DiveMac-owned files overwrite the trainer's copy at `campaign start`:
  `team.json`, `games.jsonl`, `live/`, `candidates/`, `screen-candidates/` and
  both pointers, with the candidates copied before the pointers that name them.
  `value_store.localize` re-roots a pointer's absolute path into the trainer
  checkout, and the hash check still applies.
- Append-only data (self-play shards, screening logs) is a union of settled
  files in both directions, copied with `--ignore-existing`. Trainer shards are
  named `sp-trainer-*`.
- No `--partial`: an interrupted copy never leaves a truncated file under its
  final name.
