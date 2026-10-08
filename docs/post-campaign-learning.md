# Campaign learning review, 2026-10-08

The completed campaign contains **375 verified terminal games: 146 wins, 229
losses, no ties (38.9%)**. Its 500-game continuation target ended at the deadline;
the original 100-game target was exceeded. Analysis and experiments used local
data and the offline simulator; the public-ladder controller was not restarted.

## What the recordings support

| Evidence | Observed |
| --- | ---: |
| Whole usable campaign trajectories | 374 |
| Encoded decision steps | 3,196 |
| Rich decision snapshots with verified log prefixes | 2,643 |
| Earlier encoded-only decisions | 553 |
| Terminal rich logs | 311 games |
| Current collecting-policy likelihood checks | 2,607 |
| Maximum likelihood error in the campaign audit | 0 |
| Campaign episodes marked consumed by candidate training | 372 |
| Structurally invalid whole campaign trajectories | 0 |
| Fragmented room excluded from trajectory analysis | 1 |

Every terminal room has one completed episode with a verified outcome. The
fragmented room retains its outcome; its completed suffix is not treated as a
whole trajectory. Consumption means that an episode trained a candidate, not
that its candidate improved or was promoted. No production flags were reset.
Earlier private requests and readable alternatives cannot be reconstructed
truthfully from public logs; their stored collecting tensors remain the evidence.

The corrected rain team and current actor account for 254 games, 104 wins and
150 losses (40.9%), with 2,148 steps. Across all teams, the current actor accounts
for 307 games and 124 wins (40.4%). The first 100 campaign games were 33–67;
the subsequent 275 were 113–162. Teams, opponents and scout inputs changed, so
those descriptive differences do not establish a causal actor improvement.

## Problems indicated by the data

Preview entropy was 97.9% of uniform over the full mask. Among 311 captured
previews, 102 omitted Politoed. Individual lineups were too sparsely represented
to turn their win rates into reliable expert labels. The actor sampled a median
preview probability of 0.00326 over 360 legal orders.

In captured turn decisions, rain Weather Ball was selected 17 times in 467
opportunities. Helping Hand was selected four times in 650 available decisions;
its best prior was below the overall best prior in 636 of those decisions.
Mega Evolution was available in 434 decisions and omitted in 211. These are
diagnostic frequencies, not assertions that using those commands was always best.
The prior ignores useful support interactions and some observable mechanics.
There were 12 selected explicitly friendly damaging components, while only three
consecutive Protect/Detect events appeared in the whole observed move history.
Protection spam is not supported as the main failure.

Electro Shot outside rain needs special care: 121 selected commands were charge
phases; 80 were locked releases or Power Herb exceptions. A public stream need
not announce its private two-turn volatile. Treating all 201 as fresh charge
mistakes would misdiagnose the policy.

Only 245/2,643 snapshots (9.3%) had opposing sheets; previous local practice always
revealed sheets. The old opponent schedule also correlated policy type with team
index in an eight-team pool. Finally, simulator mechanics used only the low
16 bits of a supplied seed, allowing distinct integers to repeat battle RNG.
These are training/evaluation distribution problems, even with intact episodes.

At the first corpus audit, the database occupied approximately 2.7 GiB. Local
episodes dominated its JSON payload (about 2.24 GB versus 0.56 GB for ladder
episodes). A later source audit found 11,916 current-format scout labels from
local simulation, 6,456 from own live games and 1,034 from public replays. These
are executed-move labels for scouting, not private PPO rollouts or expert actions.
The 6,923 historical M-B labels remain in their separate format. Background
learning stayed enabled, so corpus-wide counts are timestamped snapshots.

A provenance join also found 14 current-format scout labels from one digest
without an associated `public_battles` row. Their correctness cannot be checked
against that missing source. They are retained for audit and excluded from future
scout training. The ingestor previously allowed labels to be inserted after its
log insert was ignored on an identity conflict; it now rejects that conflict and
only appends labels after successfully inserting their source log. This reproduces
a concrete route to orphaned labels; it does not establish how that particular
historical digest became orphaned.

## Controlled candidate results

Weights, research, scout support, own team, opponent identities and player sides
were held fixed within each comparison. Both actors sampled actions; opponents
included frozen self-play, the legacy heuristic and a stronger mechanical script.
Promotion required at least a 10 percentage-point win gain and an exact paired
one-sided p-value at most 0.05, followed by an independent final suite.

| Candidate | Incumbent wins | Candidate wins | Paired gains/losses | p | Result |
| --- | ---: | ---: | --- | ---: | --- |
| Pooled ladder PPO, first 96-case panel | 49 | 49 | 4 / 4 | 0.637 | Rejected |
| Turn mechanics, first 96-case panel | 49 | 49 | 9 / 9 | 0.593 | Rejected |
| Rain lineup, separate 96-case panel | 49 | 50 | 8 / 7 | 0.500 | Rejected |
| Pooled PPO, full 64-bit seed panel, 96 cases | 47 | 49 | 6 / 4 | 0.377 | Rejected |
| Turn mechanics, full 64-bit seed panel, 96 cases | 47 | 46 | 7 / 8 | 0.696 | Rejected |

Pooled PPO used all 254 compatible corrected-team ladder episodes, including
already-consumed episodes only in isolation, with eight epochs and KL budget
0.03. Final KL was 0.01216; the stricter rerun checked collecting likelihoods
before any gradients, with maximum error 1.91e-6 and no excluded episodes.
The prior-based candidates changed inference features/calibration, not trained
weights. All 768 completed comparative games had zero invalid actions and zero
truncations. Independent final partitions were not opened because development
failed. Legacy seed panels retain their original mapping and are not claimed to
have globally unique mechanical RNG relative to every historical experiment.

An additional isolated initialization attempt failed before evaluation. A second
attempt stopped after its loader mistakenly dropped log segments while retaining
terminal references; the data guard correctly refused PPO. Those incomplete
attempts are excluded from the 768 comparative games. The loader was corrected
and the successful full-seed run above retained its terminal references.

The incumbent remains selected. Repeated training without demonstrated strength
gain is not a reason to install a new revision. Passing scripted opponents would
still require separate human-ladder validation.

## Implemented improvements

- Recorder 3 saves checkpoint/dex digests, phase temperatures, exact own sets and
  local simulation conditions. Collecting checkpoints are retained once by digest
  to survive promotion. It preserves feature schema 1 and all old rows.
- Dense zero encoding is lossless and retains the readable array schema. A
  representative rich live episode shrank from 1,325,800 to 1,069,108 JSON bytes
  (19.4%). Existing data was not rewritten; the database itself was not shrunk.
- PPO rejects malformed, unsubmitted, mismatched-mask, future, duplicate-request
  or invalid-prefix episodes, then checks original checkpoint likelihoods before
  gradients. Reports show exclusions and training steps by source.
- Demonstrations store the likelihood of the demonstrated command rather than
  retaining the likelihood of a different actor-selected command.
- Scout ingestion and source-log insertion are atomic. Orphaned labels stay
  stored but are excluded from training; frozen input bundles retain their source
  logs as well as the sampled vectors.
- Argument-free battle events and canonical ties survive recording and public
  ingestion; future draws terminate correctly with a zero outcome.
- New practice plans use a team-sweep curriculum with frozen self-play, tactical,
  heuristic and random opponents; sheet visibility defaults to 8%.
- New curriculum tasks use the complete 64-bit simulator seed. Saved legacy
  plans retain their mapping. Pair validation also checks teams, visibility and
  effective seed identity.
- The configured minimum batch increased from 64 to 256 compatible steps to
  reduce tiny noisy updates. CPU/GPU/time budgets and exact rain-team focus remain.
- Experimental `mechanics-v1` and `lineup-v1` profiles and a reusable isolated
  campaign experiment are available. Rejected profiles are not enabled on the
  production checkpoint.

The pressure corrections are partial estimates. Their mechanics were checked
against the [pinned simulator move definitions](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/moves.ts)
and [team-sheet implementation](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/sim/battle.ts),
rather than assuming hidden opponent spreads or actions.

## Reproduce and interpret

```bash
.venv/bin/python scripts/audit_training_data.py \
  --campaign data/ml/campaigns/<campaign-id> --output artifacts/data-audit.json
.venv/bin/python scripts/improve_campaign.py \
  --campaign data/ml/campaigns/<campaign-id> --output artifacts/improvement
.venv/bin/python -m pytest -q
```

Audits are read-only. Experiments use separate candidates and frozen inputs,
never restart a live campaign, and never stage/promote automatically. Keep local
datasets, checkpoints and reports out of source publication. Configuration changes
must be read by fresh workers before adding fields to persistent runtime settings.

The data now supports safer pooled training and better attribution. The next
strength work should target validated support/defense valuation and stronger
current-format demonstrations or opponents, rather than deriving "correct" moves
from a win alone or rerunning the same small PPO batches indefinitely.

Validation: all 149 offline tests passed. An additional eight-game smoke run
exercised all four opponent types and open/closed sheets using recorder 3 and
full seeds. All eight episodes passed data checks, retained their exact teams
and collecting checkpoints, and reached terminal outcomes with zero invalid
actions. Its 73 steps stayed unconsumed below the configured 256-step minimum;
the production actor remained unchanged. The clean source snapshot passed its
secret scan. No commit, publication, service restart or new ladder match was
performed for this review.
