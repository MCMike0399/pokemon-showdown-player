# Strategic planning validation

Verified October 8, 2026, for `gen9championsvgc2026regmc`, with the unchanged
corrected rain team. The selected policy is `strategic-v1`. It combines the
existing neural actor with a new joint planning prior and checkpointed behavior
evidence. This is a policy improvement; it is not a claim that all 500 archived
games were reused as fresh PPO or expert demonstrations.

## Data and research

All first 500 terminal campaign games had usable public logs: 196 wins and 304
losses. They supplied 4,696 executed opponent moves across 260 species/form
labels. The new checkpoint includes those aggregate behavior frequencies and
deduplication identities. Historical private snapshots support diagnostics;
unavailable old private observations remain unavailable.

A separate temporal diagnostic fitted frequencies on 400 games and tested 924
executed moves from the next 100. Species frequencies with shrinkage reduced
log loss from 5.102 to 3.170 versus a global-frequency baseline; top-choice
accuracy was 28.8% versus 9.2%. This diagnoses useful species conditioning, not
the planner's calibrated intention probabilities or battle strength.

Actual captions were retrieved for both requested Wolfe videos. Sourced
timestamped paraphrases and engineering hypotheses are in the
[new Mega study](vgc-strategic-research.md) and
[first Champions tournament study](vgc-wolfe-second-video.md).

## Complete-game comparisons

The official simulator, team pool, scout/research inputs, seeds, sides and
opponent policies were frozen. Both actors sampled actions using their declared
temperatures. Development selection and final cases were separate. Interrupted
and rejected evidence was preserved and never staged.

| Comparison | Incumbent wins | Candidate wins | Paired gains/losses | One-sided paired p | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial combined planner, development 64 | 41 | 44 | 11 / 8 | 0.324 | Rejected |
| Initial battle-only, development 64 | 41 | 43 | 9 / 7 | 0.402 | Rejected |
| Initial preview-only, development 64 | 41 | 38 | 5 / 8 | 0.867 | Rejected |
| Corrected combined planner, development 64 | 38 | 49 | 14 / 3 | 0.00636 | Passed |
| Corrected combined planner, independent final 160 | 106 | 125 | 37 / 18 | 0.00723 | Passed |

Corrections included execution-time weather, exact own Mega handling, impossible
opponent scenarios, switch order, terrain entry, retargeting, drain/recoil and
Electro Shot's pre-hit boost. A score cap had erased preview distinctions:
288 of 360 choices in one recorded case hit the same ceiling. Smooth scaling
preserved ordering, with no saturated choices in the later inspection.

A live integration check then exposed feedback being discarded by the store's
immutable-completion rule. Guarded feedback now enters the single terminal
write. The final panel was re-executed completely under that corrected source;
earlier partial results were not mixed into its counts.

The final win rates were **78.1% versus 66.3%**, a difference of 11.9 percentage
points. Finite local comparisons against this opponent pool do not establish
tournament strength or a particular real ladder win rate.

## Deployment and live observation

The native service promoted the passing candidate between games at 22:37:46 UTC.
Recordings verified the new checkpoint, feature profile and unchanged team.
No extra account or independent live player was started.

The first six complete new-policy games, campaign positions 533–538, finished
**4 wins and 2 losses**. One victory ended after its first move request, limiting
its competitive evidence. Leads varied across Politoed/Farigiraf,
Archaludon/Politoed, Golisopod/Politoed, Golisopod/Farigiraf, Garchomp/Golisopod,
and Farigiraf/Incineroar. Both Garchomp and Golisopod Mega Evolutions appeared.
All six persisted postgame feedback; fresh dry Electro Shot selections were
zero. Six games are a pilot, not a precise live improvement estimate.

Observation extended through position 542: the first ten new-policy games
finished **7 wins and 3 losses**, all with persisted feedback and zero fresh dry
Electro Shot selections. This remains a small live sample.

Live observation also found a scouting query scanning the complete sample table.
An additive covering index reduced the same species lookup from approximately
0.98 seconds to 14 microseconds. A regression exercises the actual Scout query
and verifies unchanged predictions after migration. Subsequent live decisions
were generally tens to hundreds of milliseconds rather than several seconds.
The actor weights and feature profile were preserved by this performance fix.

Actual routine MPS updates incorporated new ladder feedback into gated
candidates: one verified update included eight new games, 508 total behavior
games and 4,785 executed samples. The active policy remains frozen per battle;
candidate weights and evidence still require evaluation before promotion.
Routine learning stayed enabled, and the original four-simulator resource
budget was restored after the focused comparison.

214 offline tests passed in the isolated implementation. The shared live
checkout passed 220, including six concurrent viewer tests. Source publication
scanning passed; no dataset, checkpoint or private history was published.

Remaining planning priorities include stronger adversarial evaluation against
terrain spread attacks, sleep and defensive setup, explicit move-specific damage
rules, opponent stat uncertainty and field-expiry horizons. The two pilot losses
identify useful questions, not proof that an unplayed alternative would win.
