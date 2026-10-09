# OpenClaw continuous-pipeline supervision

Review the existing continuous Showdown pipeline. The retained campaign owns the
only player connection; Python owns CPU collection, one learner and CPU candidate
evaluation. Keep these responsibilities intact.

From the configured repository workspace, run:

```bash
.venv/bin/python -m ml.pipeline --status
```

Read `data/ml/pipeline-status.json`, the latest campaign `status.json`, and the
specific recent job/candidate reports needed to explain anomalies. Compare with
the preceding review. Check fresh heartbeats, live terminal-game progress,
separate lane claims, evaluation backlog, stale parents, pressure deferrals,
storage growth and new errors. Separate recordings, unused steps, trained batches,
evaluations, rejections, staged candidates and actual promotions. GPU activity
does not prove stronger play; gate rejection is expected when evidence is weak.

This scheduled review is read-only. Keep conclusions in the automation's run
history. Do not send Discord/WhatsApp/email messages, log in, create another
client, submit actions, edit teams/models/configuration, restart services, prune
data or increase budgets from this check. State concrete evidence-backed
recommendations when intervention is needed. The interactive agent can queue
explicitly requested experiments through `ps_learning_run`; the harness owns
gated promotion. Preserve the exact format/team focus.

Read `docs/play-training-pipeline.md` for model/data, ownership and recovery limits.
