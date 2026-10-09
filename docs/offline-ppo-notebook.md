# Offline experiment notebook

Start JupyterLab from the repository:

```bash
./scripts/run_notebook.sh
```

Open `notebooks/01_offline_ppo.ipynb` with the **Pokemon Offline (M5 Pro)** kernel.
The launcher prints its authenticated local URL. It keeps Jupyter runtime and
plot caches under gitignored `data/jupyter/`. Run all cells to inspect existing
results. The run switches default to false; enable the desired experiment and
rerun its cell to launch new work. Each enabled matrix uses a fresh output
folder. Select `MODE = "full"` for the original 480/1,024 completed-decision
collection targets, 30 updates, three training seeds and 100 final games per
opponent. Pilot and smoke modes shrink these budgets for validation.

The notebook covers seven PPO arms, completed episode returns, raw outcomes,
policy entropy, approximate KL, clipping, optimizer losses, time, held-out
random/tactical scores, Wilson intervals, worker scaling, CPU/Metal training,
real-battle scout training, a paired scout-input comparison, and historical
ladder statistics. Final cases use seeds separate from collection and periodic
development evaluation. The initial actor has a heuristic prior: compare its
initial score before claiming an improvement from learning.

This adapts the Reddit discussion to the project's exact Champions M-C format,
terminal reward and complete-game collector. It does not implement the post's
truncated vector rollout horizon. The research note records retrieved comments,
inaccessible replies and primary sources:
[experiment research](offline-ppo-experiments-research.md).

## Rebuild the environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
npm ci --omit=optional
npm run simulator:build
.venv/bin/python -m ipykernel install --prefix .venv \
  --name pokemon-offline --display-name 'Pokemon Offline (M5 Pro)'
```

The tested local environment's exact versions are saved in
`data/offline-corpus/environment-freeze.txt`; versions and runtime source
fingerprints are also saved in each run manifest. Dependencies are native arm64
on this Apple-silicon host. Use CPU or explicitly select MPS after benchmarking.

## Data transfer

```bash
python3 scripts/sync_offline_data.py \
  --host DiveMac --source '~/Developer/pokemon-showdown-player' \
  --destination data/offline-corpus/divemac-NEW
```

The exporter reads a SQLite transaction snapshot on the source, excludes local
simulations from public logs, and selects source-attributed public battles,
completed real ladder episodes, research metadata, local teams, format dex
files and optional checkpoint/scout weights. The transfer validates file sizes
and SHA-256 checksums. Existing destinations cannot be overwritten. Temporary
exports remain under `/tmp` on the source for inspection. Neither the live WAL
database nor compressed duplicate candidate inputs need to be copied.

The prepared export has 1,258 real public logs (934 M-C and 324 historical M-B),
854 completed M-C ladder episodes, 12 research teams, six documents, and source
feed/license metadata. Its manifest is under
`data/offline-corpus/divemac-20261009/manifest.json`. Actual exported counts
and provenance always come from that manifest. Historical ladder trajectories
remain diagnostic data; the matrix never treats them as fresh on-policy PPO.
Public logs supply conservative pre-turn executed-move scout labels, with
whole-battle validation and regulations kept separate.

## Direct commands

```bash
.venv/bin/python -m experiments.offline_ppo \
  --output artifacts/offline-ppo-NEW --mode full --workers 8 --backend cpu
PYTHONPATH=. .venv/bin/python scripts/benchmark_offline.py workers \
  --output artifacts/offline-workers-NEW --workers 1 4 8 12 14 --games 28
PYTHONPATH=. .venv/bin/python scripts/benchmark_offline.py training \
  --output artifacts/offline-training-NEW \
  --episodes artifacts/offline-ppo-pilot/high_lr_reference-seed0
PYTHONPATH=. .venv/bin/python scripts/benchmark_offline.py scout \
  --output artifacts/offline-scout-NEW --backend mps
```

Only the simulator collector receives CPU worker concurrency. Training uses
one optimizer process, full duty, one to four CPU threads or Metal, and a bounded
MPS allocation fraction. The intensive experiment can request up to 14 workers
on the 15-core host. Wave size also reflects available unified-memory headroom;
new processes and simulator children are budgeted separately. At least 2 GiB
available RAM and 10 GiB free disk are reserved; critical memory pressure yields
work. The run stops between waves at its local storage budget, retaining all
recordings. Existing background service budgets are unchanged.

One owned process pool stays alive within an arm; collecting weights stay fixed
until all games finish. Worker recording databases avoid write contention, and
completed episodes are merged for one PPO update. On errors or interruption,
owned children are reaped, recordings and `interrupted.json` remain, and a new
run requires a fresh directory. Inspect artifacts before rerunning. This
notebook does not resume partial optimization or deploy experiment weights.

Source notebooks stay free of outputs. A verified executed notebook is saved
locally as `artifacts/offline-notebook-executed.ipynb`, so it can display prepared
results without copying local machine or corpus information into a public source
snapshot.
