# Proposal: Macintosh plays shards of DiveMac's gates

Status: **proposal, not implemented.** It needs the user's go-ahead for each burst, and
the code changes listed in section 5. It extends [docs/mlops.md](mlops.md) and keeps its
rules.

## 1. Why

The gate is the slowest part of the improvement loop (docs/c7-live-review.md).
- A 400-pair stage is 800 games. On DiveMac that is about 15–17 hours at concurrency 3, while live play and a second stage share the 10 cores.
- One campaign (dev, confirm, standard) therefore takes about two days.
- Past promotions were worth +30 to +50 Elo each, so the pace of passed gates is the pace of the climb.

Macintosh is the more powerful Mac. Every gate pair is independent and deterministic, so
Macintosh can play some of a stage's pairs while DiveMac plays the rest. The stage then
settles sooner, without changing what it measures.

## 2. Rules that do not change

These come from the user's MLOps decisions (docs/mlops.md):
- **Live play stays on DiveMac, always.** DiveMac stays the only machine that promotes, deploys or restarts the runner.
- **Nothing moves between the Macs on its own.** There is no automatic sync, no daemon and no notification. Each burst is started by hand, and only after the user has said yes to that burst.
- **A burst lasts at most about 4 hours,** with Macintosh awake, plugged in and logged in, under `caffeinate -i`. Closing the lid on battery stops it, and that must be harmless (section 4).
- **After a burst,** results come back to DiveMac and Macintosh's data copies are removed.
- **The protocol is fixed before the first game.** Sharding is an execution detail, not an amendment: same seeds, same pairs, same code, same rule.

## 3. Why sharding is valid

For pair `p` of a stage with seed `S`, `search/value_gate.py` derives everything from
`S * 1_000_003 + p`: the game seed, both agents' seeds, the side we play and the opponent
team. Gates also remove every dependence on wall-clock time:
- `adaptive_max_ms = 1e9`, so adaptive worlds never depend on speed;
- `engine_timeout = 900`;
- `--engines 1`, so worlds are not spread over a machine-dependent number of processes.

So **a pair's result depends only on (code, inputs, seed, pair), not on the machine.** The
two arms of a pair must be played on the same machine, so that a pair is never split.

**Verify the claim before trusting it.** Every burst starts by replaying two pairs that DiveMac already completed in that stage. Winner, turns and decision count must match exactly. On any mismatch, the burst stops and nothing it played is used. Likely causes would be a different node or simulator version, or a different input file.

## 4. How a burst works

These commands are proposed, not yet built.

```sh
# DiveMac, with the user's OK for this burst:
.venv/bin/python -m search.mlops gate-burst start --stage data/ml/value/campaign-c7/gates/c7-play-dev \
    --pairs 300:399 --hours 4
.venv/bin/python -m search.mlops gate-burst status      # pairs done, ETA, determinism check
.venv/bin/python -m search.mlops gate-burst finish      # pull rows, verify, merge, clean Macintosh
```

**`start`** does six things:
1. It reads the stage's exact command line from its gate script: seeds, candidate/incumbent options, opponent, pool, previews.
2. It reads the pinned commit from the stage's worktree, for example `<checkout>.c7`.
3. It copies to Macintosh, under `~/Developer/pokemon-showdown-trainer/gates/<stage>/`:
   - a `git archive` of that commit;
   - `node_modules` (the pinned simulator) and `cache/` (usage statistics, dex);
   - `teams.json` (our team only);
   - the opponent pool;
   - the imitation and preview models the command names;
   - `artifacts/search-breakthrough/frozen` (the fallback preview actor).

   It writes a manifest with the sha256 of every input.
4. It reserves the pair range in `<stage>.shards.json` on DiveMac, so DiveMac's own process skips those pairs (section 5).
5. It runs the determinism check (section 3).
6. It starts `value_gate.py --pair-range 300:399 --out <stage>.mac.jsonl` under `caffeinate -i`, at the concurrency chosen by a short calibration (pick the fastest measured, not the largest).

**`finish`** works in four steps:
1. It rsyncs `<stage>.mac.jsonl` back, with no `--partial`.
2. It checks the manifest hashes and the commit.
3. It appends only complete rows whose `(pair, arm)` is in the reserved range and not already present. Pairs that Macintosh left unfinished are released back to DiveMac.
4. It deletes the burst directory on Macintosh. The merge goes to `data/ml/value/campaigns.jsonl` as `{"event": "gate_shard", "host": "Macintosh", ...}`.

**Stopping half-way is safe:**
- an unfinished pair simply has no complete rows;
- `finish --force` (or a sleeping Mac) releases the range;
- DiveMac plays those pairs later with the same seeds.

## 5. Code changes needed

1. **`search/value_gate.py`:**
   - `--pair-range A:B` plays only those pairs.
   - `--shards FILE` makes the DiveMac process re-read the reservations before each attempt and skip reserved pairs.
   - `--extra-rows GLOB` adds settled shard files to the rows that `summarize` sees.

   Today rows are read only at start. With these flags a running DiveMac stage does not need a restart to take a merged shard.
2. **The campaign scripts' `state()`** (each `auto-confirm.sh`, `scripts/auto_promote.py stage_state`): summarize over `<stage>.jsonl` plus merged shards. Simplest is to keep merging into `<stage>.jsonl` itself, under a lock, so nothing else changes.
3. **`search/mlops.py`:**
   - `gate-burst start|status|finish`;
   - the input manifest;
   - the two-pair determinism check;
   - cleanup.

   It reuses `Remote` (ssh `BatchMode`, Homebrew rsync, no `--partial`), `push_code` style file lists and `bootstrap`'s environment checks: node major version, `.venv`.
4. **Tests,** in the style of `tests/test_offline_mlops.py` with `host: local`:
   - shard reservation and skip;
   - merge dedupe;
   - a released range is replayed;
   - a determinism mismatch blocks the merge;
   - a reserved range is never played twice.

## 6. Expected gain and how to measure it

- **Calibration.** The first burst runs 30 minutes at concurrency 3, 4 and 6 and records games per hour. That number decides the range size for a 4-hour burst. Nobody has measured Macintosh on gate games yet.
- **The gain.** If Macintosh plays as many games per hour as DiveMac's two stages together, a 4-hour burst saves about 4 hours on one stage. Bursts on consecutive evenings would cut a campaign from about two days to about one.
- **The cost.** The bursts take the personal Mac's CPU for those hours. Live play on DiveMac gets a little faster, because DiveMac plays fewer gate games at the same time.

## 7. Order of work for an agent picking this up

1. Ask the user to approve the design, and later each burst.
2. Implement section 5, items 1 and 4. They are useful on their own: they make gate stages restartable by pair range.
3. Implement `gate-burst` with `host: local`, and rehearse it on a `/tmp` stand-in as `mlops` was.
4. Run the first real burst on a stage already running on DiveMac. Determinism check first, then calibration, then a range from the end of the stage (for example pairs 300–399), so that DiveMac's in-order play and the burst do not meet.
5. Record the measured games per hour here, and in `docs/mlops.md`.
