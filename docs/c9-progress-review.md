# Campaign c9 progress review

Review on 2026-10-22, using a fixed cutoff at 15:48:40 UTC (09:48:40 Mexico City).
The campaign has run for about 73 hours, completing **1,223 games: 611 wins and
612 losses**. It has **no demonstrated Elo improvement** yet. Three factors have
already stopped unsuccessfully; boost evaluation is still inconclusive.

## Performance and decisions

| Sample | Games | Performance Elo, approximate 95% interval |
|---|---:|---:|
| Preceding c5 deployment segment | 1,169 | 1288 ± 21 |
| Whole c9 campaign | 1,223 | 1300 ± 20 |
| Last 200 c9 games | 200 | 1233 ± 50 |

The apparent whole-campaign gain is only 12 Elo, with a combined uncertainty of
about ±29. This comparison is descriptive: different dates, opponents, and a
changing mixture of experimental settings prevent a causal before/after claim.
The latest recorded account rating was 1155; account rating is more volatile
than performance estimated over hundreds of games.

| Factor | Randomised games | Score on / off | Estimated effect, descriptive 95% interval | Existing decision |
|---|---:|---:|---:|---|
| Wider Protect second-turn search | 431 | 47.7% / 53.9% | −44 Elo [−113, +22] | Frozen off, Oct 20 |
| Boost, speed and evasion evaluation | 1,223 | 51.3% / 48.6% | +19 Elo [−20, +58] | Continue testing |
| Back-rate bench prior | 397 | 47.5% / 55.3% | −55 Elo [−128, +14] | Frozen off, Oct 20 |
| Online scouting | 752 | 48.8% / 52.0% | −22 Elo [−73, +28] | Frozen off, Oct 21 |

These are the original randomised main effects, using only games in which each
factor was still being randomised. H0 means the preregistered test chose zero
over the +40 Elo alternative; it does **not** establish that a factor is harmful
or has exactly zero effect. The intervals are ordinary fixed-sample
approximations, not confidence sequences; repeated looks and stopping affect
their coverage. The existing SPRT remains the promotion rule.

**Decision: keep the three rejected factors off and continue the unchanged boost
test.** Its LLR is −0.117, well inside the −2.251 / +2.890 boundaries. Changing
its weights, selecting a favorable subset of games, or promoting now would
discard the experiment's protection against noise. No live configuration,
protocol, preview, team, or model pointer was changed during this review.

At approximately 382 games/day recently, the remaining 1,777 games to the
3,000-game cap would take about **4.6 more days**, unless the SPRT stops earlier.
The original 3–6 day expectation was not a completion guarantee. An effect near
+20 Elo sits between the two tested hypotheses and can remain inconclusive.

This campaign evaluates fixed alternatives; **it does not train a better model
after each game**. The move traces still show no learned value, screening, or
opponent-model pointer. More games provide evidence, not automatic learning.

## Why the early improvement disappeared

| Mexico City date | Games | Performance Elo |
|---|---:|---:|
| Oct 19 | 273 | 1362 ± 43 |
| Oct 20 | 405 | 1331 ± 35 |
| Oct 21 | 391 | 1243 ± 36 |
| Oct 22, partial day | 154 | 1248 ± 57 |

The early 1387 estimate from the first 189 games did not persist. In the latest
phase, where boost is the only remaining randomised factor, boost-on performs
at **1261 ± 46** (235 games), versus **1249 ± 46** off (236 games). Both are below
their earlier levels. This does not support blaming boost for the recent fall.
It also does not establish a benefit from restoring any frozen factor: the
phases occurred at different times and those factors failed their own tests.

The two recurring weak matchup groups below did not become more common between
the first 397 and last 471 games: Fairy-spreader teams were 20.7% versus 19.1%,
and Defiant/Competitive-species teams 28.0% versus 26.5%. They identify persistent
weaknesses, not an explanation for the entire recent decline. Its cause remains
unresolved; normal variation, opponent composition beyond these groups, and
changes in opponent strength not captured by rating remain possible.

## The candidate mechanisms changed behavior without proving better play

Wider Protect search reduced double-Protect choices from **5.29% to 1.68%** of
two-action move requests while that factor was randomised. Its win score still
fell. The earlier association between double Protect and losing was therefore
not sufficient evidence that suppressing it would improve play. Do not retune
the agent merely to make its action frequencies resemble humans.

Scouting and the improved bench likelihood likewise failed their live tests.
Better information or prediction accuracy can still fail to improve decisions
under an imperfect search and evaluation function.

## Persistent matchup weaknesses and the next experiment

The groups below were identified in the pre-c9 review, so these games provide
new evidence about the same weaknesses. They overlap and are not randomised
treatments. The deficit is actual score minus Elo-expected score using both
players' pregame ratings; it is not a forecast of an intervention's benefit.

| Opponent's preview contains | Games | Win rate | Deficit versus Elo expectation |
|---|---:|---:|---:|
| Kingambit, Milotic or Annihilape | 340 | 40.9% | −9.2 percentage points |
| Gardevoir, Floette-Eternal or Sylveon | 253 | 40.7% | −9.3 percentage points |

The existing preview selects Incineroar in **325 of 340** games against the
first group and leads it in **300 of 340**. Opposing Defiant/Competitive
activation immediately following our Intimidate appears in 170 games, with
46 wins (27.1%); 139 games have an opposing activation before turn 1. These
observations include other species with those abilities. Losing games can
create more opportunities for activation, so this is diagnostic evidence,
not an estimate of the effect of avoiding Incineroar.

Across 1,222 recorded previews, Farigiraf is selected only 22 times, while
Archaludon is selected 1,221 times. This leaves little live evidence for
alternative plans. The c7 preview ensemble already failed badly, so replacing
the entire preview with that model is not justified.

**Adjust the next research priority to a narrow preview/lead candidate**, with
the current team and turn search held fixed:

1. Audit the 139 opening ability-activation games. Compare alternative legal
   leads against the same opposing leads and plausible sets, using multiple
   fixed seeds. Measure whether the candidate actually avoids the opening
   disadvantage without sacrificing the rest of the matchup.
2. Make a bounded, matchup-specific candidate that re-ranks existing legal
   preview plans. Do not categorically ban Incineroar or force Farigiraf:
   Incineroar can still be useful against Kingambit, and the sparse Farigiraf
   results cannot justify a forced bring rule.
3. Check held-out later games and a broad matchup sample for regressions. Treat
   the retrospective score as candidate development, not promotion evidence.
4. After c9 settles, preregister a fresh live comparison with the same search on
   both sides. Report the full-population effect and the targeted-matchup effect
   separately. Do not reuse c9 outcomes to declare the new candidate successful.

This is the next development plan, **not a launched or queued campaign**. No
unvalidated preview change will start automatically from this document. The
Fairy-spreader weakness is a second candidate area; the old Indeedee and
Tailwind deficit claims did not replicate in c9 and are lower priority.

## Runtime and experiment integrity

- All 1,224 assigned rooms are accounted for: 1,223 complete and one active at
  capture. No duplicate assignments, missing opponent ratings, or deviations
  from deterministic balanced-block assignment were found.
- All 1,224 logged arm overrides match the assignment ledger and original spec.
  The deployed worktree is clean at `2c5dba7`. Player and watcher are running.
- The snapshot audit contains 11,797 decision events. Move decisions take
  roughly 4.8 seconds at the median and 9 seconds at p99; the maximum across
  all decision types is 21.3 seconds. No logged search failures or stalls were
  found in the campaign runs.
- All 22 submission timeouts were followed by a terminal result: 21 opponent
  forfeits and one normal knockout loss. They waste time but are not evidence
  of losses on the turn timer. No emergency transport change is warranted.
- Opponent forfeits/inactivity account for 317 wins. The other 906 games are
  294–612. Excluding forfeits selects on how a game ends and is not a valid
  alternative rating estimate; earned forfeit wins stay in the experiment.

## Reporting adjustment and reproducibility

Added `scripts/review_live_ab.py`, a read-only report with full-campaign,
recent, daily, and chronological-phase performance; descriptive 95% factor
intervals; assignment checks; and saved-input replay. This addresses the
misleading impression from a single blended campaign rating and the existing
status command's one-standard-error display. Neither the live viewer nor the
promotion rule is changed.

```bash
.venv/bin/python scripts/review_live_ab.py
.venv/bin/python scripts/review_live_ab.py --snapshot /tmp/c9-review.json
.venv/bin/python scripts/review_live_ab.py --from-snapshot /tmp/c9-review.json
```

This review's immutable inputs, summary, selected decisions, terminal records,
and matchup analysis are retained locally under
`data/ml/value/campaign-c9/review-20261022/`. Replay `inputs.json` with
`--from-snapshot` to recover the 1,223-game statistical report. The larger
decision/terminal extracts include the game active at capture; outcome analyses
exclude it by joining to the fixed completed-game snapshot.

Validation: 25 existing A/B and Elo-monitor tests, plus direct checks that the
saved report matches the original status totals, each phase/day partitions the
1,223 games, frozen factors exclude later games, and empty input renders safely.
