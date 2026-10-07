# Readable decision recordings

New recorded decisions retain both the original PPO feature tensors and a
versioned `snapshot` containing the exact private request, normalized observation,
all legal commands with Pokémon/move/target labels and policy probabilities, the
selected index, sampling temperature, research frequencies and scout predictions.
These fields are captured before submission. `submitted` distinguishes a proposal
from a completed socket write; it does not independently prove server execution.

The episode retains sanitized battle-log segments. Each snapshot references a
specific prefix by segment, length and SHA-256. Later events can extend a segment
without entering an earlier decision's context. A different reconnect prefix gets
its own segment. Terminal closeout saves its final observable log reference.
Chat, raw request protocol lines, HTML and authentication events are excluded from
these logs; the dedicated private-request field contains the battle request itself.

Popped/rejected proposals stay in `discarded_proposals` for diagnosis and receive
no PPO credit. Restart recovery preserves the original snapshot, action, feature
vectors, probability and episode identity. An older step without a snapshot remains
`legacy-encoded-only`; no private request or named alternative is invented from
later reveals. The existing feature schema and recorder version are preserved.

All recordings remain in the gitignored local data root. They include private team
information and are not copied into the public viewer or source publication.
The actor still uses its compact decision-time features; saving snapshots does not
make it a recurrent whole-match model. Model-interface or feature changes require
separate validation and appropriate checkpoint/collection compatibility.

## Inspect a recorded decision

OpenClaw can call `ps_ml_recording(room, start=0, limit=3, alternatives=False)` on
its existing MCP server. This is read-only and does not initialize a brain, start
an observer, log in, submit actions or enqueue training. By default the result
includes the selected and top-five named choices; `alternatives=True` returns all
saved legal choices. `start` is a zero-based decision index, not a battle turn.

```bash
.venv/bin/python scripts/inspect_decision.py --room <battle-room> --limit 1
.venv/bin/python scripts/inspect_decision.py --room <battle-room> --start 1 --alternatives
.venv/bin/python scripts/audit_campaign.py --campaign data/ml/campaigns/<campaign-id> \
  --output artifacts/campaign-audit.json
```

The audit groups outcomes by team fingerprint and actor revision, reports preview
entropy and lineup choices, counts switches and observed protection patterns, and
marks missing historical context. These are hypotheses to investigate, not labels
of optimal moves. Losing does not establish that every preceding action was wrong.

## Phase-specific policy experiments

A checkpoint may specify `preview_temperature` independently of its normal
`policy_temperature`. Omitting it preserves the previous behavior for all phases.
The actor, stored probabilities and PPO likelihoods use the same phase temperature.
Changing either temperature creates a different collecting policy: use a new
revision, isolate the candidate, declare development/final cases before evaluation,
and require the existing clean paired promotion gate. Do not adjust the active
policy during a recorded game or interpret a calibrated policy as new trained weights.

## Checkpoint feature profiles

`feature_profile` versions the meaning of encoded inputs independently of their
fixed dimensions. Missing metadata means `legacy`, which preserves the existing
encoding exactly. The experimental `weather-v1` profile uses observed weather
for Weather Ball's type/power and Hurricane/Thunder accuracy, with known weather
suppression and Utility Umbrella checks. It remains a partial tactical heuristic,
not a complete damage or weather-effect simulator.

A profile change is a new collecting policy: assign a new checkpoint revision and
validate an isolated candidate before promotion. The profile is captured in the
episode and each readable snapshot. PPO excludes recordings from another profile,
and both direct recording and live recovery refuse a change within a trajectory.
Re-encoding old readable observations is useful for diagnosis; those new vectors
must not be paired with old collecting log probabilities as fresh PPO data.
