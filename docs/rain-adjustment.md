# Politoed rain: run audit and adjustments

Analysis date: October 7, 2026 in America/Mexico_City. Research and source links are in [Politoed rain research](politoed-rain-research.md). This report distinguishes real ladder recordings, controlled offline comparisons and mechanical validation.

## Decision

Keep the Recife runner-up's disclosed Politoed rain sets. Correct the generated special-attacking Garchomp spread and pin that exact corrected team for the next campaign searches. Retain the incumbent actor. The new `rain-v1` policy candidate and a bulkier Golisopod variant did not pass evaluation.

The current six Pokémon, items, abilities, natures and moves match Luciano Begot's runner-up team, but his stat-point spreads are undisclosed. Our original and corrected spreads are generated experiments, never attributed to him. There is no evidence establishing a globally best Politoed team. [Submitted tournament sheet](https://rk9.gg/teamlist/public/RE002-wN97aUf4Psz0ZJ/r9UAaJtg3GD2FYc6ZRl9), [event team database](https://limitlessvgc.com/tournaments/444/teams).

## Ladder evidence

The fixed audit cutoff contains 115 verified terminal games: **39 wins, 76 losses**, 33.9% wins. Each room has one completed episode. One interrupted/fragmented recording contributes its verified outcome but is excluded from decision analysis. The remaining 114 whole episodes contain 1,004 decisions, including 451 readable snapshots. Older missing private requests and alternative names were not reconstructed.

| Recorded team variant | Actor | Games | Wins | Losses |
| --- | --- | ---: | ---: | ---: |
| Original Mega Garchomp Z rain | Earlier policy | 16 | 4 | 12 |
| Choice Scarf Garchomp rain hybrid | Earlier policy | 52 | 18 | 34 |
| Original Mega Garchomp Z rain | Current policy | 41 | 14 | 27 |
| Choice Scarf Garchomp rain hybrid | Current policy | 6 | 3 | 3 |

The original Mega build's aggregate record is 18/57; the Scarf hybrid's is 21/58. These observational records mix policy, time, opponents and team exploration. They do not establish which build is better or how much of a loss is caused by the actor. The current actor's aggregate 17/47 also cannot be causally compared with the earlier actor's 22/68.

Team-preview probabilities remain diffuse: median normalized entropy 0.979 and median maximum choice probability 0.00641. The 360 ordered bring-four choices include strategically similar permutations, so entropy alone is not an optimality label. There were 34 observed own Protect/Detect moves and only three consecutive protection events in the audited logs; protection spam is not supported by this sample.

Among the 451 rich snapshots:

- Weather Ball was available in rain 71 times and selected twice. Both selected action encodings still represented its power as 50; the current legacy profile uses the static Normal-type move table.
- Helping Hand was available in 112 active-slot observations and selected twice. The legacy pressure prior does not credit its benefit to the joint partner's selected attack.
- Electro Shot was available without observed rain in 77 active-slot observations and selected 46 times. These counts include possible locked release requests; they do not prove 46 bad charge decisions.

These are actionable representation gaps and review hypotheses, rather than human expert labels or proof that every affected choice was wrong.

## Why the team affects the model

The actor receives compressed observable state and legal joint-action features, including species, moves, items, abilities, stats, boosts and weather. It has 384 state features and 192 action features, with a learned actor/critic and a bounded tactical prior. Team construction changes the moves and joint combinations available, damage thresholds, speed order, and viable plans. It also changes which states occur in later turns. Terminal PPO reward then credits decisions made under that policy/team pair.

A good rain team helps only if the policy can choose the right four, preserve or restore Drizzle, recognize immediate Electro Shot, use support and pick its Mega. A correct team cannot compensate for an input that treats Weather Ball as Normal in rain. Conversely, adding Perish Song or slower Trick Room pieces can increase the policy's planning burden. More legal or more popular sets do not automatically improve its win rate. This is why team and actor changes were compared separately.

## Construction repair

The generator previously chose Attack versus Special Attack using the un-Mega species' base stats. For a Modest Garchomp holding Garchompite Z with Dragon Pulse, Earth Power and Power Gem, that generated `atk:32, spe:32, hp:2` despite all attacks being special.

The generator now follows actual damaging move categories first, then nature and base stats for ties or missing move data. It leaves explicit provided spreads intact. The deployed correction is only `spa:32, spe:32, hp:2` on Garchomp, preserving every disclosed set component and the other five spreads.

The pinned official simulator verifies Mega Garchomp Z Special Attack **177 → 212**, HP **185 → 185**, Speed **203 → 203**. Ordinary Garchomp's Special Attack changes **110 → 145**. This is a clear allocation repair, independently of whether a particular battle crosses a knockout threshold. Baseline, corrected and optional Golisopod bulk builds all passed simulator legality.

## Controlled comparisons

All actors used frozen weights, temperatures, scout and research inputs. Each panel used the same eight generated opponent teams, simulator seeds and alternating player sides. Opponents comprised the frozen incumbent, a separately frozen tactical-profile actor and the scripted heuristic. Actions were sampled for the evaluated actor, matching its ladder mode. No production experience was consumed or relabeled.

The experiment declared 80 development and 160 independent final cases per team panel before play. Both original and corrected panels had to gain at least 10 percentage points with exact one-sided paired p ≤ 0.05 before advancing the policy candidate. The candidate kept existing preview behavior and changed only versioned turn-feature/prior semantics. No retraining was claimed.

| Development comparison | Incumbent wins | Alternative wins | Paired gain/loss | Decision |
| --- | ---: | ---: | ---: | --- |
| Original team: incumbent vs rain-v1 actor | 39/80 | 39/80 | 12/12 | Reject actor |
| Corrected team: incumbent vs rain-v1 actor | 44/80 | 41/80 | 10/13 | Reject actor |
| Same incumbent: original vs corrected Garchomp | 39/80 | 44/80 | 5/0 | Mechanical repair; 6.25-point local gain, below strength gate |
| Same incumbent: corrected vs bulkier Golisopod | 44/80 | 42/80 | 1/3 | Retain original Golisopod spread |

All 400 development games reached a terminal result with zero rejected actions. The policy candidate failed development, so its reserved final comparisons were not run and it was not staged or promoted. The Garchomp correction's development paired p is 0.03125, but its improvement falls below the predeclared 10-point margin. This does not prove stronger human ladder performance.

A separate 200-case team-only holdout was declared with fresh seeds after these development results. The frozen incumbent won **98/200 on the original team and 101/200 on the correction**: five paired gains and two losses, exact one-sided p = 0.2265625. All 400 games were terminal with zero rejected actions. This does not establish a reliable win-rate improvement. The correction remains justified by the confirmed allocation bug. Its opponent compositions overlap development; only seeds are held out. Across development and this holdout, 800 controlled games completed.

## Experimental model changes

`rain-v1` adds weather-aware Weather Ball and accuracy encoding, correct public colored HP, Water/Fire damage modifiers, observed Electro Shot charge/release handling and its immediate Special Attack boost, prospective Mega types/ability/estimated stats, joint Helping Hand benefit, Fake Out entry eligibility, known priority blockers, and a cap on overkill pressure. The simulator dex cache now retains Mega-form metadata needed by this profile.

This remains an approximate tactical prior. It is not an exact damage calculator, does not predict hidden opponent switches, and does not fully evaluate weather duration, speed control, protection, redirection, switching support or long-horizon endgames. Prospective Mega stats are estimates because exact nature information need not be present in a battle request. Correcting mechanics does not guarantee that old learned weights improve with a new input distribution; the comparison rejected that assumption here.

The profile is checkpointed and captured in recordings. Inference and PPO use the same phase temperatures and action likelihoods. Old encoded trajectories cannot train a different feature profile, and a profile cannot change inside an existing recorded episode. The production actor retains legacy semantics and its existing weights and temperatures.

## Applying the team safely

The campaign now reads optional `team_base`, `explore_team` and `max_team_candidates` settings before its next search. An explicit team with `max_team_candidates=1` pins its exact sets; this prevents an untried researched variant from silently replacing the controlled build. Team provenance retains the source sheet and labels the generated spreads. Existing teams and outcomes remain available for comparison.

The current campaign was configured with the corrected build, exploration disabled and one candidate. Its cap and overnight deadline were preserved. Source reloaded naturally between terminal games; no active game was stopped. The next game's stored request and fingerprint verified the corrected build, with ordinary Garchomp Special Attack 145. The actor revision, checksum and temperatures remained unchanged at deployment verification.

Local private evidence is under `artifacts/rain-adjustment/`: campaign/rain audits, stat proof, development plan and results, team-only holdout, deployment backup and live deployment audit. These include local operational artifacts and are excluded from public source export.

Validation: 117 offline tests passed after source and focused-training changes. Regression coverage includes special/physical/support generated spreads, explicit-spread preservation, exact-team selection/provenance, search-boundary configuration, rain versus charging Electro Shot, joint Helping Hand, Fake Out entry resets, prospective Mega metadata, checkpoint/profile isolation and exact collecting likelihoods. Focused-training tests verify varied opponents and sides with a fixed learner team, identical paired cases, configuration persistence, filtering by exact team fingerprint, and retention of other-team experience. A real simulator smoke game also completed with no rejected actions under `rain-v1`.

## Rain specialization

The user subsequently clarified that training should focus on the rain team. Previously the format's single actor practiced multiple own-team compositions. The new `training_teams` setting now pins the corrected Politoed rain build for new background practice and paired evaluation, while retaining varied opponents and alternating sides. PPO draws only compatible unused episodes from that exact build. Previous team data and declared experiments are preserved. This changes future training focus, without replacing the live actor or pretending existing weights were trained exclusively on rain.

The production learner keeps its existing compute budgets, 24-game practice cycle, 60-game evaluation and promotion gate. A separate bounded experiment collected **64 complete rain-team games and 484 on-policy decisions**, then trained four PPO epochs. All 64 recorded team fingerprints match the corrected build. It retained the incumbent legacy feature profile so its collecting likelihoods remained valid. Production consumption flags and checkpoint were untouched by that isolated experiment.

On the declared 100-case paired development comparison, the focused candidate won **71/100 versus the incumbent's 70/100**, with two paired gains, one loss and p = 0.5. All 200 evaluation games were clean and terminal. It failed the 10-point/confidence gate and was not staged or promoted. The 200 independent final cases per actor were left untouched. The isolated scout update is also confined to the experiment's root. Future background cycles retain the rain-team focus, but a larger training count does not by itself establish improvement.

During deployment, the new configuration field was applied before the running player process had loaded its new schema. Its older `LearningConfig` rejected `training_teams` during terminal finalization. The loss and all eight decisions had already been saved. The stopped controller was restarted with updated source; it reconciled the completed game exactly once and continued with the corrected team. A byte comparison verified that the completed episode was unchanged. This was a deployment sequencing error, not a lost battle recording. When adding configuration schema fields in a live session, wait for all readers to load the new schema before activating those fields.

## Finalized live settings and finalization resilience

The user explicitly requested finalizing and applying the adjustments after receiving an earlier OpenClaw blocked alert. Final choice: corrected Politoed rain team, exact-set selection with exploration disabled, incumbent actor retained, and new practice/PPO/evaluation focused on that build against varied opponents. The failed mechanics and focused-PPO candidates remain experimental. The incumbent uses turn temperature 0.25 and preview temperature 1.0. Its checkpoint has not been replaced merely to claim an update.

The live finalizer now separates the persisted terminal outcome from optional public-experience ingestion, learning enqueue and worker startup. A failure in those optional phases returns the completed result with an explicit stage/type error. Subsequent finish calls retry the handoff at most three times, retain a successfully queued job, and do not duplicate or rewrite the episode. Core recording failures still propagate. Regression tests reproduce the reported enqueue `TypeError` and a worker-start error through real recording objects, then verify terminal results, exact episode preservation, bounded retries and one durable job.

Campaign status now includes the loaded source generation, finalized team/model settings, the original target and whether it is complete. The expanded overnight target remains 500; the original 100 games are already complete. These additive fields give observers current deployment evidence rather than treating a historical blocked line as the latest state.

Verified after natural source reload: the real harness was playing at 126 completed games, 43 wins and 83 losses, with the corrected team fingerprint and unchanged incumbent. Three active recorded decisions matched their collecting log probabilities exactly. Source generation matched the finalized source. Recovered games 106 and 123 each have exactly one completed episode and one campaign record. Game 106's seven steps match its saved recovery-audit snapshot and all seven log-prefix hashes validate; the lack of a full pre-disconnect byte snapshot remains a limitation. Game 123's eight-step completed episode is byte-for-byte unchanged from before its controller restart. Duplicate recovery records would be a bug, not a validation requirement.

Final validation: **120 offline tests passed**. Current private deployment/recovery evidence is under `artifacts/rain-finalization/`. Counts are a timestamped verification, not a promise that the ongoing campaign stops at that count.

The production background worker also completed a focused PPO update from 28 compatible games and 212 decisions. Its candidate won 45/60 versus the incumbent's 42/60, with three paired gains and no losses (p = 0.125). It failed the configured margin/confidence gate and was not staged. Real per-battle worker reports name the corrected rain-team fingerprint and preserve small batches while accumulating experience. This verifies that the training focus is running in the production worker, beyond the isolated experiment. A subsequent status check showed 127 completed games, 43 wins and 84 losses, still playing with the selected incumbent and corrected team.
