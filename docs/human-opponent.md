# Learning the ladder opponent: research, method and the first promoted improvement

Campaigns c3 and c4, 2026-10-13 to 2026-10-14, on DiveMac (Apple M5).

## Summary

For 1,308 live games the player sat at 50%. Every earlier candidate (61 learned
value nets, a selective second search turn, PPO policies) failed its paired gate.
This work found why: **the gates measured candidates against the search playing
itself, and the search's model of its opponent was wrong about real ladder
players.** Ladder players are not equilibrium players, and our team preview
ignored who we were facing.

We learned the ladder opponent from public replays and used it in three ways:

| Use | Result |
|---|---|
| A **gate opponent** that plays like a ladder human (imitation) | built and calibrated; it makes gates measure the deployment distribution |
| A **human opponent model inside the search** (campaign c3) | predicts real opponents much better (top-1 26% vs 18%), but **rejected** by its gate |
| An **opponent-aware team preview** from human results (campaign c4) | **passed both gate stages and is live since 2026-10-14 11:03 CST** |

The promoted change picks, among our 90 possible (lead pair, back pair) plans, the
plan ladder players won most with against the opponent's six. Against the
imitation opponent it won 81 more than it lost 52 in development (p = 0.007) and
87 against 62 in confirmation on fresh seeds (p = 0.024). It is the first
candidate in the project to pass both stages of a fixed, preregistered gate.
**The live effect is not established yet**: it needs at least 200 live games
compared with the 50% baseline.

## 1. Diagnosis: why nothing passed before

Three measurements on the live ledgers framed the work.

1. **The opponent model was miscalibrated.** For each turn of our 120 newest live
   games, the position was rebuilt as the live search rebuilt it and the
   opponent's real action was located in the search's enumeration
   (`search/human/compare_live.py`, 957 decisions with a fully known action):

   | Opponent distribution | Top-1 | Top-3 | Negative log likelihood |
   |---|---:|---:|---:|
   | Deployed: 0.5 equilibrium + 0.5 greedy softmax | 0.176 | 0.400 | 3.70 |
   | Equilibrium alone | 0.164 | 0.385 | 4.76 |
   | Uniform over the kept actions | 0.194 | 0.322 | 3.27 |
   | Learned human policy (v3) | 0.246 | 0.515 | 2.82 |
   | 0.8 human + 0.2 deployed | 0.264 | 0.517 | 2.82 |

   The equilibrium component was worse than guessing uniformly. The probe
   screening also discarded the real reply in about 20% of decisions (kept 553
   of 694), against 11 to 13% for a screening by human probability.

2. **The preview was opponent-blind.** The actor led Archaludon + Politoed in
   1,278 of 1,290 games and brought the same four in 97%. Humans with this
   archetype choose among about five lead pairs depending on the opponent.

3. **The gate opponent was not the ladder.** Every gate played the candidate
   against the deployed search piloting real teams. A candidate tuned to beat
   ladder humans has no reason to win that game, and vice versa.

## 2. Data

**Public replays.** `search/human/fetch_replays.py` pages the format's public
replay index (`replay.pokemonshowdown.com/search.json?format=...&before=T`) and
downloads each log with both players' ratings, resumably, at 2.5 requests per
second. The format has about 2,400 public games per day; the corpus now holds
about 150,000 games from 2026-09-20 on. The models below were trained on the
first 10,600 (move policy) and 12,600 (preview) of them.

**Our own games.** The live ledgers keep each game's full public log, so the
same extractor reads our opponents' decisions in exactly the games the live
search faced. Our account is excluded as an actor. These decisions are the
held-out test set on the deployment distribution.

**From logs to labelled decisions** (`search/human/extract.py`). For every move
turn and each player (the actor):

- the position is rebuilt with the live search's tracker and usage priors, with
  one difference: the actor knows their own team, so everything the actor
  revealed anywhere in the game (moves, item, ability, Mega stone) fixes their
  own side, while the other side is sampled from what the actor had seen;
- the engine enumerates the actor's legal joint actions (`engine.cjs` message
  `human`, with Mega and non-Mega variants);
- the label is the action the log shows. When a Pokemon fainted or flinched
  before acting, its slot's choice is unknown, so the label is the set of all
  consistent joint actions and the loss marginalises over it. Called moves,
  Struggle and locked moves are not choices and are skipped.

Of the labelled decisions, about 81% have a fully known action and 19% a marginal label.

**Previews.** For each replay: both sixes, the lead pair, the Pokemon brought,
the winner and the ratings (`data/ml/human/previews-replays.jsonl`).

## 3. Models (all trained on MPS in seconds)

**Move policy** (`search/human/train_policy.py`). An MLP scores every legal
joint action of a decision; the loss is the negative log of the softmax mass on
the consistent set (a listwise loss). Inputs: the value net's 146 position
features from the actor's side, the screening ranker's 70 action features
(category, power, priority, estimated damage and KO per target, speed order,
switch matchups, focus fire), and one-hot identities of the move used and the
acting species per slot (`humanfeat.cjs`, vocabulary of the 200 most used moves
and 120 species). The one-hot blocks are trained as embeddings and exported as
one dense layer, so the engine evaluates the net with the same loader as the
screening ranker. Splits are by game.

Accuracy on our held-out live opponents (full enumeration, about 60 actions):

| Training replays | Top-1 | Top-3 |
|---:|---:|---:|
| 1,200 | 0.17 | 0.35 |
| 4,600 | 0.20 | 0.41 |
| 10,600 | 0.21 | 0.41 |
| 10,600 + 60% of our own opponents | 0.22 | 0.44 |

The curve flattens quickly: more replays help little.

**Preview policy** (`search/human/preview_model.py`). Same listwise idea over the
90 plans of a six. Features: multi-hot species of own six, opposing six, the
four brought and the lead pair. Lead-pair top-1 on held-out games: 25% (uniform
6.7%).

**Matchup value** (`search/human/preview_value.py`, the promoted model). A small
classifier of P(win | own six, opposing six, plan) on 19,976 human player-games
whose four brought Pokemon are known. On held-out games it is only weakly
predictive (AUC 0.59, log loss 0.675 against 0.684 for the base rate), which is
expected: preview explains a small part of a game's outcome. Applied to our team
against the opponents we actually met, it rated our habitual plan at 42% and the
best plan per opponent about 10 points higher, usually with Garchomp or
Golisopod leading next to Incineroar.

## 4. Evaluation method

**The imitation opponent** (`search/human/imitation.py`). It plays each turn by
the move policy on a position rebuilt from its own request and public stream,
uses the preview policy at team preview, and a one-world search for forced
switches. It pilots the 112 real ladder teams of the gate pool.

**Calibration.** Its strength had to be near the ladder's, or almost no pairs
would differ. Against the deployed search over 24 to 45 games each:

| Imitation variant | Its win rate vs our search |
|---|---:|
| Sampled at temperature 1.0 | under 5% |
| Sampled at temperature 0.5 | 12% |
| Policy top-3 or top-6, then a one-world search | 29 to 33% |
| Argmax of the policy | 36% |
| Real ladder players | 50% |

The gates use the argmax variant.

**Independence.** The imitation's policy was trained on one hashed half of the
replay games and the candidate's on the other half, so a candidate cannot win by
exploiting the gate opponent's own quirks.

**The gate** (`search/value_gate.py --opponent imitation:...`). Paired games: same
seed, opponent team, side and sampled worlds in both arms. A pair counts when the
two arms' results differ. Each stage plays 400 pairs and passes when the one-sided
exact sign test gives p ≤ 0.05 with more gains than losses. A stage stops early
as a rejection once gains ≤ losses after 40 complete pairs. A candidate needs a
development stage and a confirmation stage on fresh seeds, plus a supplementary
200-pair gate against the old search opponent that must not show significant
harm. All rules were written in a protocol file before the first game.

**Amendments** (both recorded before results were known for the affected rows).
The imitation could not translate a locked two-turn move to its request, and one
pool team's "Hisuian Arcanine" spelling failed the set lookup; both made games
error. Errors voided pairs equally in both arms. Both were fixed mid-stage and
verified on the 40 most recent failing positions; no rule, model or seed changed.

## 5. Results

| Campaign and candidate | Development (400 pairs) | Confirmation (fresh seeds) | Decision |
|---|---|---|---|
| c3: human opponent model inside the search | 2 gains / 9 losses at 70 pairs, futility | 11 / 10 at 73, stopped | rejected |
| c4: human preview policy | 86 / 61, p = 0.024 | 5 / 7 at 41, futility | rejected |
| c4: matchup-value preview | 81 / 52, p = 0.007 | 87 / 62, p = 0.024 | **promoted** |

Supplementary gate for the promoted candidate, against the old search opponent:
12 gains / 14 losses at 51 pairs. No gain against the search, and no significant
harm.

**Why c3 failed although its predictions were better.** On 84 live positions the
c3 candidate chose a different action than the deployed search in 52% of them,
with a median gap of 0.15 hand-evaluation units between the top two actions. The
better opponent model moves many close decisions, and the gate says those moves
were not wins. With gaps that small, the noise of the leaf evaluation and of the
sampled worlds bounds the one-turn search; changing the opponent model alone is
not enough.

**Why the preview worked.** Team preview is a one-shot choice with a large,
systematic effect, made by a policy that previously ignored the opponent. The
matchup value encodes what thousands of human games say about which of our
plans works against which opposing six. That is information the turn search
cannot recover once the leads are on the field.

**The rejected human preview** passed development and failed confirmation. With
two candidates in development at p = 0.05, one passing by chance is not rare;
the confirmation stage is what makes the promotion credible.

**Which team to play.** `search/human/team_arena.py` played the deployed search,
with the human preview, on our team and on the six most played ladder archetypes
(exact open-team-sheet sets), 30 games each against the imitation pool with the
same seeds. Ours won 22 of 30; the others 19 to 25. Paired against ours, the best
were 8 gains / 5 losses and 7 / 4 (p about 0.3). In human hands our exact six
won 60% of 53 public games. Nothing supports a team change: the gap is the
pilot, not the team.

## 6. Deployment and how to judge it

Live since 2026-10-14 11:03 CST (code ce7d6b8):

    scripts/browser_play.py --team Rain-Recife-special-stat-fix --continuous --resume-active \
        --chrome --search --preview pvalue:data/ml/human/models-preview/pvalue-661448b6a320fce0.json

The runner logs a `preview_policy` event per run. When the opponent's six is
unknown, the old actor's preview is the fallback. Turn decisions are unchanged.

**Judging it live.** The gate measured the gain against an imitation of the
ladder, not the ladder itself. The live check compares at least 200 games after
deployment with the 1,308 before (50.0%), and also looks at the lead pairs
chosen and at win rate by opponent archetype. With 200 games, a 50% to 57%
change is at the edge of what can be distinguished from noise, so a flat result
after 200 games is not conclusive either way.

**Rollback.** Remove the two `--preview` arguments from the
`dev.pokemon-showdown.browser-player` LaunchAgent plist, then `launchctl bootout`
and `launchctl bootstrap` it (`kickstart -k` alone does not reload an edited
plist). The runner finishes its current game before stopping.

## 7. Limits and next steps

- The imitation is an approximation of ladder play at 36% strength, not 50%.
- The matchup value is a weak predictor (AUC 0.59); a better preview model,
  trained on the full 150,000-game corpus and with our search's own results, is
  the natural next candidate.
- The gate's futility rule is applied when reading summaries; the gate process
  does not stop itself, so stages are stopped by hand.
- The PPO learning service was paused for these campaigns.
- The opponent model inside the search (`opp_model`) stays in the code, off by
  default; a future campaign could combine it with a less noisy evaluation.

## Code and data map

| Path | Role |
|---|---|
| `search/human/fetch_replays.py` | replay download |
| `search/human/extract.py` | human decisions and previews from logs |
| `search/human/humanfeat.cjs`, `vocab.json` | action features with move and species identity |
| `search/human/train_policy.py` | move policy (MPS) |
| `search/human/preview_model.py` | human preview policy |
| `search/human/preview_value.py` | matchup value (promoted preview) |
| `search/human/imitation.py` | imitation opponent |
| `search/human/compare_live.py` | opponent-model accuracy on our live games |
| `search/human/calibrate.py`, `team_arena.py` | imitation strength, team comparison |
| `search/human/store.py`, `campaign.py` | opponent-model pointer and c3 promotion helper |
| `search/value_gate.py` | `--opponent imitation:...`, `--cand-preview`, `make_preview` |
| `search/engine.cjs` | `human` enumeration, `opp_detail`, `opp_model` |
| `data/ml/value/campaign-c3/`, `campaign-c4/` | protocols, gate logs (private) |
| `artifacts/human-policy/` | comparison, calibration and arena results (private) |

Reproduce:

```
.venv/bin/python search/human/fetch_replays.py --out data/replays/regmc
.venv/bin/python search/human/extract.py --replays 'data/replays/regmc*/replays-*.jsonl' --ledgers --out data/ml/human/regmc-v4
.venv/bin/python search/human/train_policy.py --data 'data/ml/human/regmc-v4/replay-*.npz' --live 'data/ml/human/regmc-v4/live-*.npz' --replay-half A
.venv/bin/python search/human/preview_value.py train
.venv/bin/python search/value_gate.py --pairs 400 --cand-preview pvalue:PATH \
    --opponent imitation:POLICY:argmax:PREVIEW --pool data/ml/value/pool-build/pool-teams-new.json --out ...
```
