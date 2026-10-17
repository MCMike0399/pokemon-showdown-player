# What the live games show (campaign c7)

Research 2026-10-17 on DiveMac, one day after the c5 endgame search went live and the
thinking-budget campaign c6 settled.

## 1. Where we are

| Segment | Performance | Games |
|---|---:|---:|
| baseline | 1206 ± 19 | 1,325 |
| c4 preview | 1236 ± 28 | 635 |
| c5 endgame search (2026-10-16 14:23Z on) | **1287 ± 34** | 435 |

c5 cleared its rollback rule (≥ 300 games, above c4). The ladder rating is 1326.

**c6 failed.** More thinking did not help:
- 12 worlds: 42–39, p .41.
- Wider screening: 37–38.

Sampling noise is not the bottleneck; the evaluation and the opponent model are.

**Forfeits flatter the win rate.** 83 of the segment's 212 wins are opponent forfeits. Played-out games are won 129 of 352 (37%).

**What 1800 means on this ladder.** These numbers come from 110,376 public replays (2026-09-09 to 10-13) and 164,612 rated human player-games.
- 1800 is the 99.9th percentile of player-games. 43 human accounts have ever reached it.
- At 1800 the opponents' median rating is 1738, so holding it needs a 58.8% win rate.
- K is 50 from 1100 to 1299, then 40.
- Rating-adjusted, the bot is −0.5 ± 1.9 pp against what its rating predicts. Humans with at least 4 of our six are +8.0 ± 1.8 pp, and with 5 of 6 +16.3 ± 5.3. The team is fine; the pilot is the gap.

Where the pilot falls short of humans with the same team, pooled by what the opponent's team contains:

| Opponent has | Gap (bot − humans, pp) |
|---|---:|
| Sneasler | −15.2 |
| Kingambit | −14.1 |
| Sun | −13.7 |
| Tailwind setter | −12.3 |
| Indeedee | −11.6 |
| Trick Room setter | −2.1 |

## 2. Why played-out games are lost

These counts come from detectors over all 435 games, with 46 losses read in full. "Expected" is the search's own calibrated P(win) at the first flagged decision.

- **Opposing boosts are underpriced.** A logistic fit over 20,114 live positions (2,385 games) prices each opposing +1 attack stage at 0.50–0.81 hand units beyond the leaf's 0.12, and each opposing +1 Speed stage at 0.31–0.49. Our own attack boosts come out at +0.39–0.64. In 68 games where an opponent set up freely, the bot won 22% against 40% expected.
- **Protect is overused.** We Protect on 26% of actions against 12% for the opponents, and double Protect on 10.8% of our two-Pokemon turns against 3.7%. `protect_d2` only re-checks Protect lines when the position is already losing (hand < −1).
- **No weather term.** The fit prices an opposing sun at −0.66. When the opponent overwrote rain with Politoed benched, bringing Politoed back won 50% (34 games) and not doing so won 9% (11).
- **The Mega goes on the wrong Pokemon.** Golisopod Mega'd beside an un-Mega'd Garchomp in 19 games, which won 16% against 38% expected. Mega Golisopod is 4x weak to Fire and loses Emergency Exit.
- **Bugs:**
  - **Double Mega.** When both opposing actives could Mega Evolve (two stones in a world), the engine listed only "one of them switches" options, so the opponent could never attack with both. About 1% of decisions were affected, mostly on turn 1.
  - **Parting Shot on our own ally.** 18 uses in 16 games.
  - **Struggle.** A Struggle request failed every world.
  - **Failed Protect.** A failed Protect still counted as chained, so the next, certain Protect was priced at 1/3. This affected 57 slot-decisions.

## 3. What the bot believes about opponents

These results come from the replays and 10,109 open team sheets (about 60k true sets).
- **Floette-Eternal was never a Mega.** Smogon files Floette-Mega under base Floette, so the prior gave Floettite about 0% (the true rate is 98%) and sampled Choice Scarf 64% of the time. It was on 7.8% of opposing teams.
- **The back pair was predicted worse than uniform.** Unseen members were weighted by ladder usage: Rillaboom outweighed Kommo-o about 18 to 1, yet every species is brought at a similar rate. Log loss for the back pair:
  - deployed prior 2.47
  - uniform 1.79
  - corpus bring odds with lead pairing 1.65
- **Items are sampled independently of revealed moves.** Corpus co-occurrence would lift the true item's log-likelihood from −1.78 to −1.27. This is not built yet.
- **Scouting.** 66% of live opponents have earlier public replays under the same name. Items revealed there matched the current game 98.8% of the time. This is not built yet.
- **The preview model drifted.** It was trained on 12.6k replays, including bot accounts on our rental team, and now brings Farigiraf in 2% of games and Incineroar in 96%. Humans with our team bring them 51% and 39%.

## 4. Changes

Shipped live (bugs, both gate arms): the double-Mega enumeration (`sideChoices`), ally-targeted
harmful status moves pruned (engine and agent), Struggle requests rebuilt with empty PP, a failed
Protect resets the chain (`tracker.py`), the Floette-Eternal Mega alias (`priors.py`).

Gated in campaign c7 (opt-in options, `search/config.py` unchanged until a gate passes):

| Option | What it does |
|---|---|
| `eval_terms` | evaluation terms: positive attack stages 0.5 per stage (+0.15 for the opponent), Speed stages 0.4, evasion 0.3, weather ownership ±0.4 (setter or abuser alive, scaled by turns left) |
| `protect_d2.behind = 99` | re-score Protect lines a turn deeper whenever the best action Protects, not only when losing |
| `endgame_d2` | with ≤ 2 Pokemon per side, re-score the top 4 actions a turn deeper |
| `mega_avoid: ['golisopod']` | keep the Mega for another stone holder while it lives |
| `bench_prior: 'back_rate'` | unseen back pair from corpus bring odds with lead pairing (`search/back_rates.json`, `search/human/build_back_rates.py`) |

On our own 285 live games with a known back pair, the new bench prior puts the true pair in
23.4% of worlds (17.6% before). Re-searching 25 random live positions with the whole bundle
changed 10 decisions; the median decision time rose from 6.0 to 7.2 s (max 16.7 s, against a
55 s turn).

The preview candidate is a 5-seed ensemble of the matchup-value model. It is trained on all 110,376 replays, excludes our account and 22 likely bot accounts, and has rating inputs set to 0 at inference (`PreviewValue` now reads ensembles).

Held-out plan-choice AUC measures whether the chosen plan beat the six's average plan:

| | Live model | Ensemble |
|---|---:|---:|
| All held-out sides | .519 | .556 |
| Rated ≥ 1400 | .509 | .572 |

Against our live opponents it brings Farigiraf in 24% of games and Incineroar in 37%.

**Gate.** The protocol was written before the first game. Stages:
- **PLAY** dev, confirm, then a no-harm stage against the deployed search.
- **PREVIEW** dev and confirm, both arms on the PLAY search.

Every stage is 400 pairs against the argmax ladder-human imitation. A stage passes at a one-sided sign test p ≤ .05 with more gains than losses. `value_gate.py --inc-search` lets the PREVIEW gate put the incumbent arm on the candidate search.

**Promotion.**
- PLAY goes live through `scripts/auto_promote.py`.
- PREVIEW goes live only after PLAY. If PLAY fails, PREVIEW is re-gated on the deployed search.

**Live judging** uses Elo segments.

## 5. Not built yet, by expected value

1. Scouting the opponent's earlier public replays: exact items, and team history.
2. Joint set sampling from open team sheets: item | revealed moves.
3. Bring rules for Farigiraf/Trick Room against fast and Defiant teams. The preview ensemble moves this way.
4. Stat spreads closer to what humans run on this team, such as bulkier Archaludon and Golisopod. This is a team change and needs its own arena gate.
