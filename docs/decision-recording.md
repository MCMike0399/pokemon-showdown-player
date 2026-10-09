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
Argument-free events such as `|start` and the canonical `|tie` are retained. The
player and local simulator recognize both canonical and trailing-pipe tie forms,
so future draws finalize as zero rewards and remain available to the scout.

Popped/rejected proposals stay in `discarded_proposals` for diagnosis and receive
no PPO credit. Restart recovery preserves the original snapshot, action, feature
vectors, probability and episode identity. An older step without a snapshot remains
`legacy-encoded-only`; no private request or named alternative is invented from
later reveals. The feature schema remains 1. New episodes use recorder version 3;
old recordings are retained without migration or fabricated provenance.

Recorder 3 adds the collecting checkpoint SHA-256, phase temperatures, feature
profile, cached dex digest, and exact own sets when available. Local episodes also
retain the opponent policy/team, side, sheet visibility and effective simulator
seed. The collecting artifact is retained once by digest under
`models/collected/<format>/`, so a later promotion does not erase its weights.
Dense encoded arrays write numeric zero as `0`, preserving every float32
input and the existing list-shaped JSON schema while reducing new record size.
Historical digests/archives are not backfilled. A digest alone cannot reproduce
a missing historical model.

If the playing socket disconnects, `ps_ml_play` makes at most three attempts per
room to close the old transport, authenticate the configured account and rejoin
that exact room. A repeated request resubmits the saved action without sampling
again; a saved successful socket submission is retained if a resubmit fails.
An unsent new proposal is excluded from training credit. Exhausting recovery
returns an unfinished result and keeps the pending episode without a reward.
This bounded recovery does not establish that arbitrary crashes are lossless.

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
.venv/bin/python scripts/audit_training_data.py --campaign data/ml/campaigns/<campaign-id> \
  --output artifacts/training-data-audit.json
```

The audit groups outcomes by team fingerprint and actor revision, reports preview
entropy and lineup choices, counts switches and observed protection patterns, and
marks missing historical context. These are hypotheses to investigate, not labels
of optimal moves. Losing does not establish that every preceding action was wrong.

The training-data audit checks mask/selection alignment, feature dimensions and
finite values, collecting probabilities, duplicate requests, submitted flags,
log-prefix digests and temporal ordering. It reports corpus/source counts,
consumption and candidate-evaluation history. Fragmented rooms keep their verified
outcome but are excluded as trajectories. PPO validates original checkpoint
likelihoods before gradients and excludes whole incompatible episodes, retaining
their data and consumption flags. Demonstration likelihoods now refer to the
demonstrated command, even when that command differs from the actor's argmax.

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

Additional experimental profiles are opt-in. `tactics-v1` fixes colored public HP
and scores estimated damage using observed stat boosts/burn, target HP, joint
focus fire and friendly damage. Opponent stats/spreads remain estimates; this is
not an exact damage calculator. Legacy profiles retain their original encoded HP
values so stored collecting tensors keep their meaning.

`preview-v1` adds a visible-opponent coverage/lead prior. `preview-v2` maps that
prior monotonically into the bounded logit range; direct clipping can otherwise
collapse every legal lineup to the same score. Preview profiles preserve turn
encoding. A new profile/revision and complete candidate validation are required.

Offline evaluation can hold an explicit opponent checkpoint fixed through
`opponent_checkpoint_root`. Both policies then face the same model, including its
own feature profile, research and scout inputs; paired validation checks the
opponent revision as well as seeds/sides. This supports tougher comparisons than
random and tactical-script opponents alone. Passing a local suite still does not
establish human ladder strength.

For joint preview and prospective Mega scoring, isolated fresh-rollout training
and paired evaluation, see [Opening and Mega policy](opening-policy.md).

`mechanics-v1` preserves state/preview encoding and adds turn pressure corrections
for observed weather, boosts/burn, selected Mega form, known immunities/screens,
joint Helping Hand and friendly damage. Locked Electro Shot releases are identified
from the targetless request rather than guessed to be fresh charges. The prior
still estimates pressure; it does not model the opponent's complete response.

`lineup-v1` preserves turn encoding and adds a preview prior for bringing a rain
setter with rain-dependent partners and leading those partners together, alongside
visible coverage. Neither profile is enabled simply by installing source: they
require a new checkpoint revision and a passing isolated evaluation.
