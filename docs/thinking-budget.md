# Thinking budget: more worlds or wider screening (campaign c6)

Research 2026-10-16 on DiveMac, after the c5 endgame search went live.

## 1. Live review (2026-10-14..16)

- **Elo per deploy segment** (`search/elo_monitor.py`):

  | Segment | Performance | Games |
  |---|---:|---:|
  | baseline | 1206 ± 19 | 1,325 |
  | c4 preview | 1236 ± 28 | 635 |
  | c5 endgame search, from 2026-10-16 14:23Z | 1255 ± 60 | 139 (too early to judge) |

- **A browser bug** cut into the c4 segment (fixed in 4b2800f).
  - The answered-request key had no room, so a new game's preview looked already answered.
  - The runner stalled until the server warned our clock, then sent the third-ranked preview through the fallback.
  - That hit 231 of 634 games (36%), which won 47.6% vs 52.1% for the others.
  - In the first 139 games after the fix: 0 fallbacks and 0 lost choices.
- **Game length decides results.** Over 630 live games, the win rate by length was:

  | Turns | Won |
  |---|---:|
  | 0–3 | 97% |
  | 4–6 | 60% |
  | 7–9 | 41% |
  | 10+ | 34% |

  Opponents forfeited 137 games, all of which we won. We never lost on the timer.
- **Search failures are rare live.** 6 of 1,398 decisions fell back: 4 engine timeouts (40 s) and 2 worlds with no legal actions.
- **Decisions are noise-bound.**
  - Re-searching 24 random live positions with the deployed configuration (6 worlds, fresh samples) chose a different action than live in 11.
  - With 12 worlds, the choice changed in 11 of 24 versus 6 worlds.
  - Live latency is a 4.8 s median (p90 6.7 s) against a 55 s turn timer with a 420 s bank.

## 2. Candidates

Measured with `search/leaf_counterfactual.py --cand` on live positions while live play competed for the CPU:

| Candidate | Change | Median | Max |
|---|---|---:|---:|
| deployed | 6 worlds, screen keep_opp 20 / keep_ours 24 | 4.4–4.6 s | 5.5 s |
| c6-worlds | 12 worlds | 6.6 s | 9.5 s |
| c6-screen | keep_opp 32 / keep_ours 36 | 7.2 s | 9.0 s |

Adaptive worlds (+6 when the top-2 margin is below 0.5) run only when the first batch took less than half of `adaptive_max_ms`. If c6-worlds is promoted, `adaptive_max_ms` rises from 9 s to 20 s with it. Gates lift that bound, so live keeps measuring the gated policy.

## 3. Gate

The protocol is `data/ml/value/campaign-c6/PROTOCOL.md` (private), written before the first game. Both candidates are gated separately against the deployed configuration (c5 + pvalue preview on both arms), against the argmax ladder-human imitation.

- **Stages:** 400 pairs per stage, dev then confirm on fresh seeds.
- **Promotion of c6-worlds:** automatic (`scripts/auto_promote.py`), once its confirm passes and the supplementary standard gate shows no harm.
- **c6-screen:** if it passes too, it needs a combined gate first, because worlds and screening together would be ungated.

Live judging: the Elo segment against the c5 segment, over ≥ 300 games.
