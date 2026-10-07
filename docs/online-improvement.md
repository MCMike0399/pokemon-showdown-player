# Analyze and improve a running campaign

Keep one player connection and finish an existing room before searching again.
Read the current campaign's manifest, control and status files, then reconcile
terminal rooms against complete episodes. Counts alone do not establish recovery
or training quality.

```bash
.venv/bin/python scripts/monitor_campaign.py
.venv/bin/python scripts/analyze_decisions.py --campaign data/ml/campaigns/<campaign-id>
```

Decision diagnostics report the verified score, last ten results, fragmented
recordings excluded from policy analysis, collecting likelihood consistency,
and action-probability and tactical-prior distributions. Low probability does
not prove a bad move; a tactical prior is not a damage oracle. Compare identical
actor revisions and team fingerprints, and separate sampled ladder play from
greedy or sampled scripted evaluations. Do not retrain consumed corrected data
without an explicit isolated, audited experiment.

The portable campaign controller is `scripts/campaign_ladder.py`:

```bash
.venv/bin/python scripts/campaign_ladder.py --campaign data/ml/campaigns/<campaign-id>
```

It expects `manifest.json`, `control.json`, `status.json`, `games.jsonl` and the
campaign's named base team in the local team store. It preserves recovery state
before preflight, recovers the exact collecting team, and verifies one completed
episode per terminal room. The effective cap is `control.json`'s `target_ladder`;
changing it does not start an idle service. Never reset accumulated records or
use a restart to abandon a live battle.

Source changes are detected after terminal closeout, before the next search.
The controller closes its MCP connection and exits with code 75 to request a
supervised reload. Configure the campaign's own supervisor to relaunch that
exit; normal completion and reported blockers exit zero so they do not loop.
Do not restart an unrelated gateway or a live controller to load changes.

Learning jobs retain small compatible batches, evaluate candidates with frozen
inputs, and stage eligible checkpoints until the next `ps_ml_ladder` boundary.
See [continuous learning](continuous-learning.md) for the gate and resource
policy. A successful training job or a smaller loss is not evidence of better
play. Keep the incumbent when either final evaluation or integrity checks fail.
