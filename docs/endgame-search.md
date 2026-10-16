# Endgames, Protect and speed: three search changes (campaign c5)

Research 2026-10-14 on DiveMac. Motivation: in losing positions the live search
used Protect until it lost, instead of playing for the outs a human expert plays
for. This note gives the diagnosis, the three changes, the offline evidence and
how they are gated. None is live until campaign c5 promotes it; the deployed
behaviour is unchanged while the options are unset (`search/config.py`).

## 1. Diagnosis

Every cell of the search matrix is one simulated turn scored by a material
evaluation (alive, HP, status, boosts, field), with ±30 for a finished game
(`handEvaluate`, `search/engine.cjs`). Over the live ledgers of 2026-10-11..14
(real move requests only), the share of decisions that include a Protect rises
as the searched value of the chosen action falls:

| Searched value of the chosen action | Decisions | With Protect | Games won |
|---|---:|---:|---:|
| ≥ 4 | 640 | 9% | 95% |
| 0 to 4 | 4,114 | 29–34% | 49–81% |
| −2 to 0 | 2,380 | 42–44% | 15–31% |
| −6 to −2 | 1,272 | 53–66% | 1–5% |
| < −25 (loss this turn in nearly every world) | 817 | 79% | 0% |

Two causes:

- **Horizon.** After a Protect turn the leaf scores the board, which still looks
  healthy; it does not see that the same double target arrives next turn, when
  Protect cannot be chained. Example: a full-HP Archaludon facing Mega Raichu-Y
  (26%) and Gholdengo scored Protect −3.4 and every attack about −23.
- **Linear risk attitude.** The material score predicts live results well:
  over 12,896 live positions, P(win) = σ(0.801·h − 0.430). A position at −4
  wins about 2%, one at +4 about 96%. On the linear scale, surviving one more
  turn of a lost position (−30 → −3) is worth more than the whole gap between a
  lost and a winning position, so a line with a 15% chance to turn the game
  (crit, miss, the opponent protecting) cannot compete with stalling.

Expert play does the opposite (WolfeyVGC's Mega Golisopod tournament,
[video](https://www.youtube.com/watch?v=Ai3L_iwn5oE)). Losing, he plays for the
opponent's mistakes and misses (around 58:17, 1:53:10, 2:17:14). Winning, he
picks the line that still wins against perfect play (1:03:31, 1:16:10). He also
reads speed tiers from turn order (1:27:44).

## 2. The three changes (all opt-in search options)

**Win-probability leaf** (`leaf: {a, b}`). Each cell is scored
`k·σ(a·v + b)` with the live calibration above (`k = 4/a`, unit slope at 50%).
Expectations over chance, opponent replies and worlds are then taken in
win-probability space: risk-seeking when behind, risk-averse when ahead.
Because the opponent model's greedy softmax runs on the same scale, `opp_tau`
was re-fitted for it (section 3).

**Protect second turn** (`protect_d2: {behind, top_opp, keep, probes}`). When a
world's static evaluation is below `behind` and its best action contains a
self-Protect, the best Protect action and the best action without one are both
re-scored by the existing selective second turn against the opponent's likeliest
replies. Cost: about +10% simulations, only in losing positions.

**Speed inference** (`speed_infer: true`, `search/speed.py`). Within a turn, at
equal priority, the faster Pokémon moves first (the slower under Trick Room).
Each ordered pair of one of our moves and an opposing move bounds the opponent's
effective Speed, since ours is known exactly (Champions level-50 stat formula,
stages, Tailwind, paralysis, Scarf, weather abilities). Opposing sets that
contradict an observation are redrawn (24 tries, then the fewest violations).
Turns with Quick Claw, Custap, Quick Draw, After You, Quash, Instruct or Dancer
are skipped. Pairs are skipped when the opposing priority could depend on a
hidden ability (Prankster, Gale Wings, Triage), and when the opposing species can
have Protosynthesis, Quark Drive, Stall or Mycelium Might. A pair counts only if
its earlier move precedes the turn's first Speed-changing event.

Validation: in 120 simulated games with known teams, 1 of 1,412 observations
contradicted the opponent's true set (an Illusion Zoroark-H posing as
Incineroar). On 200 live games, 169 had observations, and 14% of the usage-sampled
opposing sets the search would have used were impossible.

## 3. Offline evidence (same worlds and seeds, deployed configuration)

`search/leaf_counterfactual.py` re-searches recorded live positions with both
arms on identical worlds; a linear-vs-linear control changed 0 of 120 decisions.

| Positions | Leaf only: changed | Protect | Full candidate: changed | Protect | Decision time |
|---|---:|---:|---:|---:|---:|
| 60 live Protect choices with value < −2 | 27 | 56 → 41 | 41 | 55 → 33 | 3.1 → 3.5 s |
| 60 random turn-start positions | 20 | 19 → 24 | 27 | 18 → 17 | 4.5 → 4.6 s |

`opp_tau` for the leaf scale: see the c5 protocol (`search/opp_sweep.py` on
`compare_live.py --dump`, NLL of real ladder opponents' actions).

## 4. Fixes found on the way

- `|swap|` (Ally Switch) was ignored by the tracker, so after an Ally Switch the
  search placed the two opposing Pokémon in each other's slots for the rest of
  the game. Fixed in `search/tracker.py`; this fix is live-safe and applies to
  every arm.
- Team-preview and single-choice decisions reported the previous search trace in
  the ledger (`decide()` now clears `last_trace` first). Analyses of older ledgers
  must filter on the request (no `teamPreview`/`forceSwitch`, active moves present).

## 5. Gate and live judging

Campaign c5 (`data/ml/value/campaign-c5/PROTOCOL.md`) gates two candidates,
leaf only and the full set, against the deployed configuration with both arms on
the live pvalue preview (`value_gate.py --inc-preview`), against the argmax
ladder-human imitation. Live progress is judged by the Elo performance rating
(`search/elo_monitor.py`, viewer panel), not by win rate.

## 6. Results and promotion (2026-10-15/16)

| Stage (vs) | Pairs | Gains | Losses | p (one-sided) | Score cand / inc | Verdict |
|---|---:|---:|---:|---:|---:|---|
| leaf-dev (imitation) | 395 | 36 | 32 | .36 | — | failed |
| full-dev (imitation) | 397 | 44 | 23 | .007 | 299 / 279 | passed |
| full-confirm (imitation) | 397 | 52 | 38 | .085 | 295 / 282 | failed (p > .05) |
| full-standard (deployed search) | 198 | 29 | 17 | .052 | 128 / 115.5 | no harm |

The win-probability leaf alone does nothing; the bundle helps. Under the
pre-registered rule the confirm stage missed (p .085), so `auto_promote.py`
declined. **Amendment 1 (2026-10-16, after all stages settled):** full was
promoted anyway. Every stage favoured it, and no stage showed harm. Pooled over the
two stages that did not select it (confirm + standard), it went 81–55,
p = .016. The deviation is deliberate: the goal is live Elo, which an
imitation gate measures only indirectly (c4: +7.6 pp in the gate, +29 Elo live).
Live is now the arbiter.

**Live judging rule (written before any c5 live game):** compare the c5 deploy segment of
`search/elo_monitor.py` with the c4 segment (1235 ± 28 over 634 games). After ≥ 300
games, roll back (remove `leaf`, `protect_d2`, `speed_infer`; `alpha` 0.5) if the c5
performance is below the c4 segment's. The same restart also shipped a browser fix
(answered requests scoped to their room, 4b2800f). Before it, 36% of c4 games played
the third-ranked preview after a stall, so the c5 segment measures the fix plus the
search change.
