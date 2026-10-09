# Lossless collection-input archives

Completed practice jobs retain frozen inference databases under
`data/ml/collections/*/inputs/`. These databases duplicate public research and
scout inputs. `scripts/archive_collection_inputs.py` compresses closed snapshots
without changing the main experience database, candidate evaluation inputs,
checkpoints, game recordings or training labels.

The live processing storage limit is 150 GiB (`resource.max_disk_gb`). Archiving preserves that limit;
it does not increase CPU, GPU, memory or simulator allowances. Each archive is
fully decompressed and checked against its original SHA-256 and byte count
before removing the uncompressed duplicate. Its `.archive.json` records the
job and restore hash. The verified archive and metadata are flushed before
removing the duplicate.

Preview eligible files:

```sh
.venv/bin/python scripts/archive_collection_inputs.py --limit 16 \
  --report artifacts/cold-archive-preview.json
```

Apply a bounded batch while respecting the live resource configuration:

```sh
.venv/bin/python scripts/archive_collection_inputs.py --apply --limit 16 \
  --respect-resource-budget --report data/ml/maintenance/cold-input-archive-latest.json
```

## Compaction under storage pressure

Finished candidate evaluations keep a frozen input database under
`data/ml/candidates/*/evaluation-inputs/` (about 300-600 MB each). With
`--when-over F`, when the data root exceeds `F x max_disk_gb` these snapshots are
also compacted (oldest first, same byte-exact verification), until use falls
below `--target x max_disk_gb` (default .75). A candidate is eligible only after
its `report.json` exists. Snapshots that still have a journal or an open reader
are skipped.

```sh
.venv/bin/python scripts/archive_collection_inputs.py --apply --limit 16 \
  --respect-resource-budget --when-over .85 --target .75 \
  --report data/ml/maintenance/cold-input-archive-latest.json
```

The `dev.burbuja.pokemon-cold-archive` LaunchAgent runs this every 900 s.

The command uses one low-priority process and an exclusive archive lock.
Running jobs, missing completion reports, symlinks, nonempty SQLite journals
and snapshots with active readers are excluded. Completed-job and unchanged-file
checks run again before removing a duplicate. On DiveMac, the verified Docker
file-sharing broker's cached handles are distinguished from actual readers.
The command requires `lsof`; failed handle inspection stops the batch.

Restore a snapshot for a historical review:

```sh
.venv/bin/python scripts/archive_collection_inputs.py \
  --restore data/ml/collections/JOB/inputs/experience.sqlite3.gz \
  --report artifacts/cold-archive-restore.json
```

Restore validates every byte, retains the archive and refuses to overwrite an
existing database. A restored original with an existing archive is retained on
subsequent archive runs.

DiveMac's local `dev.burbuja.pokemon-cold-archive` LaunchAgent runs the bounded
16-file command every 15 minutes. Its latest report and logs are private under
`data/ml/maintenance/`. It performs file maintenance; review and policy decisions
remain with the ongoing Codex session. The job can defer when host headroom is
insufficient and resumes on its next scheduled invocation.
