# Campaign c9: a live factorial A/B on the ladder

Review 2026-10-19 on DiveMac, after three days of unattended running (campaigns c7 and c8, live c5).

## 1. What the three days showed

**The live player has held its level.** The c5 segment reached 1,153 games, with performance 1289 ± 21 and a 49.3% win rate. That is −1.2 pp [−3.9, +1.6] against what Elo predicts: the bot plays exactly at its rating. The b12314c bug fixes went live 10-17 18:12Z and changed nothing detectable (1281 before, 1293 after).

**Every gated candidate failed:**

| Stage (vs the argmax imitation) | Gains | Losses | Verdict | Effect, 95% CI |
|---|---:|---:|---|---|
| c7 PLAY bundle (eval terms, Protect second turn, endgame d2, mega_avoid, back-rate prior) | 45 | 52 | failed (futility) | −1.5 pp (−6.4, +3.4) |
| c7 PREVIEW ensemble pvalue-2d43d11e | 61 | 88 | failed, **significant harm** | |
| c8 scouting (SimScout) | 17 | 14 | failed | +1.25 pp (−1.6, +4.1) |

## 2. Why the gate cannot see what we need

The figures in this section come from 46 logged gate games, all gate rows c4–c8, and the 1,153 live games.

**Power.**
- Only 17–24% of a search candidate's pairs are discordant.
- As implemented, the rule has three parts: the sign test, the futility check at every pair count from 40, and dev and confirm both passing. Together they detect about +11 pp with 80% power, and the actual false-positive rate is 2%.
- A real +30 to 50 Elo gain is +4 to 7 pp. A +5 pp candidate passes both stages 19% of the time. c5 (+46 Elo live) failed its confirm stage at 52–38.

**Alignment.**

| Opponent's behaviour | Argmax imitation (gate) | Ladder humans |
|---|---:|---:|
| Protect, per action | **29%** | 11.7% |
| Voluntary switch, per action | 2.8% | 8.4% |
| Our win rate against it | 75–77% | 49% |

Against the imitation our bot Protects 12.6% of the time; live it Protects 21%. A Protect or boost candidate is therefore judged in exactly the setting where it can't show its effect. Game length, Fake Out, speed control and targeting do match the ladder.

**Gate and live agreed in direction for the two promotions** (c4 +7.8 pp / +30 Elo, c5 +4.3 pp / +46 Elo). Nobody knows how many rejected candidates were real.

## 3. What the live games say (all 1,153 games of the segment)

**Protect is still overused.**
- We Protect on 25% of our actions; opponents do on 14%.
- On turns with two of our Pokemon active, we double-Protect 9.3% of the time; opponents 4.6%.
- After a double Protect, the search's own next-turn value falls by **0.117 [0.104, 0.131]** compared with turns of equal value where we neither Protect nor switch. After a single Protect it falls 0.025.
- After a voluntary switch it *rises* 0.041. The search picks Protect too often, and its own next look disagrees.
- At value ≥ .35, games with a double Protect lose 10.7 pp [5, 17].

**Opposing setup is underpriced.**
- When an opponent sets up unpunished, the result is −15 pp [−21, −9] (n 180).
- When our Intimidate triggers Defiant or Competitive (Kingambit, Milotic, Annihilape), we win 22% [16, 30] of played-out games. Otherwise we win 37%.
- The leaf prices an opposing +1 attack stage at 0.12. The live logistic fit says 0.5 to 0.8.

**Matchup holes.** These are mostly team weaknesses, so they are not search levers.

| Opponent's six has | Elo excess |
|---|---:|
| A Fairy spreader | −13.7 pp |
| Kingambit, Milotic or Annihilape | −10.3 pp |
| Indeedee | −8.5 pp |
| Tailwind setter, after the review | −8.6 pp |

**Claims of the c7 review that did not hold up on the 718 later games:**
- Golisopod Mega'd next to Garchomp: 62% won against 49% expected. It was noise.
- Re-setting rain after the opponent overwrites it: 46% against 44%. The effect is gone.
- Sneasler: about −1 pp once adjusted.
- Lead-pair differences are matchup selection; adjusted for the opponent's six they vanish.

**Not problems:**
- Forfeit wins are earned: at our last decision before a forfeit, the median value is .86.
- Decisions take a median 4.9 s (p99 10.4 s).
- Every stall and fallback happened before b12314c.

## 4. Campaign c9: test on the ladder itself

Each ladder game draws its own arm (`search/ab_test.py`). Both arms meet the same opponents at the same rating, and nothing else changes between them. So the comparison has no deploy-segment confounds and no opponent-model mismatch.

**Design.** A 2×2×2×2 factorial over four candidate factors.
- Games come in balanced blocks of 16: each block plays every combination once, in a seeded random order.
- A factor's effect compares the games where it was on with those where it was off. Every game informs all four factors, so four factors cost about as many games as one.
- Each factor has its own sequential test (normal-approximation SPRT), 0 against +40 Elo, α = .05, β = .10.
- A factor can decide from 300 games on. At 3,000 games without a decision it counts as not promoted. At about 370 games a day, a +40 Elo factor takes roughly 3–6 days.
- A decided factor is frozen at its level (on for H1, otherwise off), and the other factors continue.
- Once every factor is frozen, `scripts/ab_watch.py` promotes the factors frozen on through `auto_promote.promote()`: clean-checkout tests, secret scan, commit, push, pin and graceful restart. It then sets the spec inactive.

| Factor | Options when on | Evidence |
|---|---|---|
| `protect` | `protect_d2.behind = 99`: re-score the best Protect line and the best non-Protect line a turn deeper in *every* position, not only when losing | −0.117 value drift after a double Protect. On 40 live double-Protect positions it cut double Protects from 29 to 6, and on random positions it changed 1 of 22 decisions, at +13% decision time |
| `boost` | `eval_terms` {boost .5, foe_boost .15, speed .4, evasion .3, **weather 0**} | The live logistic fit of boost and Speed stages; unpunished setup −15 pp; Defiant/Competitive triggers. The weather term is dropped because its evidence vanished |
| `bench` | `bench_prior: 'back_rate'` | Back-pair log loss 1.65 against 2.47 for the deployed usage prior (uniform scores 1.79) |
| `scout` | `scout` {trust .85, max_games 6, min_overlap 2}: opponents' earlier public replays | Items revealed earlier matched 96–99% of the time. Facts exist for 45% of opponents. The gate could only simulate this coverage (+1.25 pp, CI −1.6 to +4.1); live is the only real test |

Every other setting is `search/config.py` LIVE at the deployed commit, with the live team preview pvalue-661448b6a320fce0.

**Caveats, written before the first game:**
- Main effects average over the other factors' levels. A strong interaction would blur them. The per-cell table in `search/ab_test.py status` shows interactions, but they don't decide anything.
- Four tests at α = .05 each: a false promotion of a null factor is possible, about 5% per factor. The SPRT almost never accepts H1 for a harmful factor.
- The protocol is fixed: `data/ml/value/campaign-c9/PROTOCOL.md`. Amendments are only allowed for documented bugs.

**Operation:**

    .venv/bin/python -m search.ab_test status           # per factor: games, score on/off, Elo, LLR, decision
    cat data/ml/value/ab/current.json                   # the spec (active, factors, frozen, history)
    tail data/ml/value/campaign-c9/promote.log          # the watcher's decisions

- The runner reads the spec before every game.
- To stop the experiment, set `"active": false` (no restart needed); every new game then plays LIVE.
- `search/elo_monitor.py` shows the A/B games as their own segment, `[live A/B c9]`.
- After a reboot, restart the watcher with the line in PROTOCOL.md. The runner needs nothing.

## 5. Gate changes for later campaigns (recommended, not built)

The gate is still the right tool for candidates too risky to try live. It should change in three ways:

1. **Pooled sequential stage.** Play one 800-pair stage with looks at 200, 400, 600 and 800 pairs (O'Brien-Fleming z ≥ 3.46 / 2.45 / 2.00 / 1.73). Stop for futility only at 400 and 600 pairs, when z < 0. Simulated: false-positive rate 4.9%, power 80% at +4 pp.
2. **Imitation with action-class biases.** Add Protect and switch logit biases, fitted to the live 11.7% and 8.4% rates.
3. **End-of-game HP margin as a secondary statistic.** The within-pair correlation is only 0.12. It can't become primary until it agrees with the win test over two campaigns.

The scripts behind these numbers are in `/tmp/gates2` and `/tmp/psa2`. They do not survive a reboot.
