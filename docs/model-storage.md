# Model data storage

The deployment limit is **150,000,000,000 bytes (150 GB)**. The legacy
`resource.max_disk_gb` setting uses GiB, so set it to `150000000000 / 2**30`
(about 139.698). Accounting includes the repository's `data/`, `artifacts/`
and `cache/`, including SQLite WAL files and temporary saves. Dependencies and
external symlinked worktrees are excluded; hard links count once.

Optional background jobs, simulator waves, feed imports and new browser searches
stop at 90% (135 GB), reserving 15 GB for in-flight saves and live finalization.
An explicit battle resume and an active game's final recording remain allowed.
This is cooperative admission control, not an OS filesystem quota: external
tools can still write, and an already admitted operation can overshoot its
boundary. The supervisor rescans each minute; workers recheck between waves and
the learner checks at its regular resource checkpoints.

| Data | Retention and purpose |
|---|---|
| Canonical episodes, requests, action probabilities, terminal evidence and collecting checkpoints | Preserve for eligible on-policy learning, diagnostics and audit |
| Public replay logs and scout vectors | Preserve once in the canonical store; deduplicate by battle identity/digest; require terminal results and pre-turn executed-move labels |
| Historical formats | Keep separately labeled; never relabel M-B as M-C |
| Frozen inference inputs | Research, distinct species/move support and exact scout weights; share immutable content-addressed directories; canonical sample identities and bodies stay in the main store |
| Completed historical input snapshots | Losslessly gzip-compress after completion; retain exact-byte SHA256 and restore metadata |
| Candidate checkpoints, evaluation plans and reports | Preserve; paused evaluations and open input readers prevent archival |

New frozen inputs have `inference-only.json` and must be opened with
`Store.read_only`. They cannot train a scout. Their omitted bodies remain in the
canonical database. Both policies in a paired evaluation share the same frozen
inputs; a retry retains the original snapshot. Snapshot directories under `input-snapshots/` are referenced by symlinks:
do not delete them or garbage-collect them by age.

Collection remains bounded by current-policy backlog, candidate capacity, exact
format and the mixed opponent curriculum. Public/search games are scouting
evidence, not invented private PPO rollouts or expert actions. Evaluation cases
remain evaluation evidence. More retained bytes do not prove stronger play.

## Inspect and maintain

```bash
.venv/bin/python scripts/model_storage_report.py --output data/ml/maintenance/storage-quality.json
.venv/bin/python scripts/archive_collection_inputs.py --include-candidates --report /tmp/input-archive-plan.json
.venv/bin/python scripts/archive_collection_inputs.py --apply --include-candidates --respect-resource-budget --report data/ml/maintenance/cold-input-archive-latest.json
```

DiveMac schedules `scripts/maintain_model_storage.py` every 900 seconds with
a limit of 16 completed inputs per compression pass, background
priority and low-priority I/O. It respects the existing CPU/RAM guards and
skips a cycle when maintenance is already running.

The quality report includes all-corpus counts, source/format separation, scout
orphan counts and explicitly sampled structural validation across history plus
recent games. It records defects by episode ID without rewriting history. Exact
collecting likelihood and checkpoint eligibility remain the trainer's gate.

Restore a historical compressed input:

```bash
.venv/bin/python scripts/archive_collection_inputs.py --restore PATH/experience.sqlite3.gz --report /tmp/restore.json
```

Archives remain available after restoration. Verify the checksum and SQLite
integrity before any offline inspection; never point a live writer at frozen
inputs. Archival is serialized by `cold-input-archive.lock`, checks completion,
journals and open handles, verifies recovery and persists its replacement before
removing a duplicate. Default commands are dry runs unless `--apply` is given.
