# Human opponent model: the search plays against ladder players, not against itself

## Why

Every live turn is chosen by one-turn simulator search (`docs/search-agent.md`).
Its opponent model is a blend of the matrix equilibrium and a greedy softmax
prior, and every offline gate so far measured a candidate against that same
search piloting real teams (`docs/value-learning.md`). Two facts from the first
1,300 live games say this is the wrong opponent:

- On our own live games, the deployed opponent model gives the action the
  ladder opponent really chose about 10-15% top-1 probability mass among the
  kept replies (`search/human/compare_live.py`). Ladder players at 1000-1400
  are not equilibrium players; they are predictable in other ways.
- Team preview: our actor led Archaludon + Politoed in 1,278 of 1,290 games.
  Human players with the same archetype pick among about five lead pairs
  depending on the opponent's six (`search/human/preview_model.py` data).

So the leaf-evaluation campaigns (c1, c2) optimised against an opponent that
does not exist on the ladder. This module learns the ladder opponent from
public data and puts that model in three places: inside the search's opponent
model, as the opponent of the offline gate, and (later) in team preview.

## Data: public replays and our own games

`search/human/fetch_replays.py` pages the public replay index of the format
(`replay.pokemonshowdown.com/search.json?format=...&before=T`, 51 per page,
about 2,400 rated games per day back to the format's start) and stores each
replay's log with its players' ratings under `data/replays/<dir>/replays-*.jsonl`
(resumable, about 2.5 requests per second).

`search/human/extract.py` turns every move turn of every log into a decision of
each player (the "actor"):

- the position is rebuilt the way the live search rebuilds it (`search/tracker.py`
  from the public prefix, hidden sets from `search/priors.py`), with one
  difference: the actor knows their own team, so the actor's own moves, items,
  abilities and Mega stone revealed anywhere in the game are used for their
  side, while the other side is sampled from what the actor had seen so far;
- the engine enumerates the actor's legal joint actions (`engine.cjs` message
  `human`, Mega and non-Mega variants) with the features below;
- the label is the joint action the log shows. A Pokemon that fainted or
  flinched before acting leaves its slot's choice unknown; the label is then
  the set of all consistent joint actions (its part is any move, restricted to
  Mega parts when the log shows a Mega Evolution) and the loss marginalises over
  it. Called moves, Struggle and locked moves are not choices and are skipped.

Our own live games (`data/ml/browser-runs` ledgers, full public log in the
terminal event) go through the same extractor with our account excluded, so the
opponents' decisions in exactly the games the live search faced are a held-out
set on the deployment distribution (`live-*.npz`), split from the replay games.

Shards: `data/ml/human/regmc-v2/{replay-*,live-*}.npz` with per-decision
position features `S`, action features `A`, group index `rec`, label mask `y`
and JSON meta (game, side, turn, rating, result, choices). Previews
(`previews-replays.jsonl`): both sixes, the leads and the brought four.

## Features and models

`search/human/humanfeat.cjs`: the screening features (`search/screenfeat.cjs`,
70 per joint action: category, power, priority, estimated damage and KO on each
target, speed order, switch-in matchups, joint focus fire...) plus one-hot
identities per slot of the move used and the acting species over a fixed
vocabulary (`vocab.json`: the 200 most used moves and 120 species of the
format's usage statistics). Position features are the value net's 146
(`valuefeat.cjs`) from the acting side.

`search/human/train_policy.py` (MPS, seconds): a 2x256 MLP scores every action
of a decision; the loss is the listwise negative log of the probability mass on
the consistent set. Splits are by game. The export
(`data/ml/human/models-v2/human-<sha16>.json`, `layers/mean/std`) is the same
format as the screening ranker, so `valuenet.cjs` evaluates it in the engine.

`search/human/preview_model.py`: the same idea for team preview. A plan is one
of the 90 (lead pair, back pair) selections of a six; the features are multi-hot
species of own six, opposing six, the brought four and the lead pair; the label
marginalises over unrevealed back members.

## Inside the search: `opp_model`

`search/config.py` `LIVE['opp_model'] = {path, w, temp}` (None = the old
model). In `engine.cjs searchWorld`:

1. the policy scores all of the opponent's legal joint actions; the kept set is
   its top `keep_opp` by human probability instead of the probe screening (no
   probe simulations for their side: about 30% fewer simulations per decision);
2. after the matrix is solved, the opponent distribution used to value our
   actions is `q = (1-w) * human + w * (alpha * equilibrium + (1-alpha) * greedy)`,
   renormalised over the kept set.

The live runner adopts the team's `opp-current.json` pointer between games
(`search/human/store.py`, SHA-256 verified, falls back to the old model), like
the value and screening pointers. A play-code restart is needed once for the
runner to know the pointer.

## The gate's opponent: a ladder-human imitation

`search/human/imitation.py` `ImitationAgent` samples the policy (temperature)
over its legal actions on a position rebuilt from its own request and public
stream, previews with the human preview policy when given (else the frozen
actor), and uses a one-world search for forced switches, which the policy does
not cover. `search/value_gate.py --opponent imitation:PATH[:temp[:PREVIEW]]`
makes both arms face it piloting the real-team pool. Candidate and imitation
policies are trained on disjoint replay halves so the candidate cannot exploit
the imitation's own idiosyncrasies.

## Operate

```
.venv/bin/python search/human/fetch_replays.py --out data/replays/regmc            # resume download
.venv/bin/python search/human/extract.py --replays 'data/replays/regmc*/replays-*.jsonl' --ledgers --out data/ml/human/regmc-v2
.venv/bin/python search/human/train_policy.py --data 'data/ml/human/regmc-v2/replay-*.npz' --live 'data/ml/human/regmc-v2/live-*.npz'
.venv/bin/python search/human/compare_live.py --model data/ml/human/models-v2/human-<sha>.json --games 120
.venv/bin/python search/value_gate.py --cand-search '{"opp_model": {...}}' --opponent imitation:MODEL:1.0 --pairs 400 --out ...
```

Results of campaign c3 are appended below as they settle.

## Campaign c3 result (2026-10-13)

Candidate: the deployed search with `opp_model` (policy v4 trained on replay half B plus
60% of our own opponents, w = 0.2); opponent: argmax imitation (replay half A, human
preview policy), 400 pairs per stage, dev and confirm concurrently
(`data/ml/value/campaign-c3/PROTOCOL.md`). **Rejected**: dev stopped by futility at 70
complete pairs with 2 gains and 9 losses (candidate 48 wins, incumbent 54); confirm was
stopped by hand at 73 pairs, 11 gains and 10 losses. The candidate used 25% fewer
simulations per decision. Predicting the opponent better (top-1 26% vs 18% on real ladder
decisions) did not make the one-turn search win more games against an opponent drawn
from the same population; about 16% of pairs were discordant.

Imitation strength calibration (45 games each, v2 policy): argmax 36% wins against the
deployed search, top-k-with-search 29-33%, sampled at temperature 0.5 12%, at 1.0 under 5%;
ladder players win 50%.

Decision-level diagnostic (`artifacts/human-policy/disagreement-c3.json`, 84 live
positions, same worlds and seeds): the candidate chose a different joint action in 44
(52%), with a median top-2 margin of 0.15 hand units overall and 0.10 where it differed.
The opponent model changes many close decisions, and the gate says those changes were not
wins. With margins that small, the leaf evaluation's noise, not the opponent model, bounds
the one-turn search.

## Campaign c4: opponent-aware team preview (2026-10-13/14)

Two preview policies for our own team, everything else as live, against the argmax
imitation opponent (`data/ml/value/campaign-c4/PROTOCOL.md`):

| Candidate | Dev stage (400 pairs) | Result |
|---|---|---|
| Human preview policy (argmax; what ladder players with this archetype lead against that six) | 366 complete, 34 void: 86 gains, 61 losses, p = 0.024; score 250 vs 225 | passed |
| Matchup value (argmax; the plan humans won most with against that six) | 378 complete, 22 void: 81 gains, 52 losses, p = 0.007; score 275 vs 245 | passed |

Void pairs came from two infrastructure faults (amendment 2), identical in both arms.
Confirm stages on fresh seeds started 2026-10-13 23:35 CST.

### Confirmation and promotion (2026-10-14)

| Candidate | Confirm stage (fresh seeds) | Supplementary (vs search opponent, 200 pairs) | Decision |
|---|---|---|---|
| Human preview policy | futility rejection: 5 gains / 7 losses at 41 pairs (20 / 20 when stopped at 102) | not run | rejected |
| Matchup value | 393 complete, 7 void: 87 gains, 62 losses, p = 0.024; score 284 vs 258.5 | 12 gains / 14 losses at 51 pairs: no gain, no significant harm | **promoted** |

The matchup-value preview is deployed to the live runner (`scripts/browser_play.py --preview
pvalue:data/ml/human/models-preview/pvalue-661448b6a320fce0.json`): at team preview it plays, among
our 90 plans, the one ladder players won most with against that opponent's six, falling back to
the actor's preview when the opponent's six is unknown. Everything else about live play is
unchanged. The two dev passes with one confirm failure are a reminder that a 400-pair dev stage
at p = 0.05 with two candidates passes a null candidate about 10% of the time; the confirm stage
is what makes the promotion credible.

Team choice (`search/human/team_arena.py`, 30 games per team against the imitation pool, same
seeds, human preview for all): our team 22/30; the six most played ladder archetypes with exact
open-team-sheet sets scored 19 to 25 of 30; paired against ours, the best were 8 gains / 5 losses
and 7 / 4 (p about 0.3). No team change is supported by this evidence.
