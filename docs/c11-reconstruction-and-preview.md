# Reconstruction fixes, M5 scheduling, and a bounded preview experiment

Implemented after the October 23 campaign review. C10 keeps its pinned source,
models, seeds, ledgers, and stopping rule. The preview experiment below is separate
and cannot replace c10 or promote a model from predictive accuracy alone.

## Correctness fixes

An Illusion reveal can be `|replace|IDENT|DETAILS`, without an HP field. The tracker
previously ignored that message. Zoroark's faint then marked its Blaziken disguise
as fainted; when the real Blaziken entered, it retained the faint flag. Search saw
two fainted opposing actives and generated `pass, pass`, which the simulator
rejected when it requested a replacement. Every world failed and the player used
the first legal move. Three recorded positions from one live game reproduced it.

The tracker now transfers the active observations to the revealed identity and
restores any facts previously known about the disguise. A previously unseen
disguise does not count as a fourth revealed team member. Switch-in HP also
updates the faint flag. Regression cases cover reveals with and without HP,
previously observed disguises, and the complete tracker-to-simulator path.

Recharge also needed both public and private state reconstruction. Public
`-mustrecharge` sets the volatile; `cant ... recharge` consumes it; a private
Recharge request is authoritative after reconnect. The engine rebuilds that
volatile. Move translation retains the request index for Recharge because the
simulator's name-based command incorrectly infers a normal target for this
targetless pseudo-move. The regression exercises an actual simulated turn.

The runner logs `reconstruction_version: 2` in its search-agent event. This gives
the fix a separate performance-Elo segment without changing search tuning or
pretending its Elo effect is already known.

## Controlled CPU comparison

`search/compute_bench.py prepare` freezes 48 positions, six worlds per position,
and their input hash. The sequence is a two-minute no-gate baseline, widths
4/8/6/6/8/4, then another baseline. Every width receives the same worlds and seeds,
with adaptive extra worlds disabled. Rebuilt sleep/confusion positions are
excluded because their duration currently uses a separate random source.

`run` temporarily parks c10's supervisor and gate groups, leaving live play and
data ingestion running. Benchmark engines register with live priority. A separate
guardian restores c10 if the benchmark parent exits; normal and interrupted exits
restore the registrations. No experiment spec or candidate is edited by the
benchmark. The parked gates retain memory, and other host workloads are not
controlled, so this measures scheduling under a shared workstation workload,
not an isolated silicon maximum or full-game pairs/hour.

The measured two-pass totals were:

| Engines | Seconds for 96 positions | Positions/second |
|---|---:|---:|
| 4 | 202.10 | 0.475 |
| 6 | 149.48 | 0.642 |
| 8 | 162.64 | 0.590 |

The raw six-engine throughput was 8.8% higher, but live interruptions differed:
four/eight/six engines overlapped about 41.41/15.73/0.03 seconds of live inference.
After subtracting those estimated pauses, throughputs were .597/.653/.642
positions/second. Six is within 1.7% of eight with 25% fewer workers, satisfying
the rule to prefer the smallest width within 5% of the best measured throughput.
This supports six as a provisional resource choice, not a proven speedup.
All passes had exactly the same simulator counts and action-value hashes.
There were only 6/5/0/2 ordinary live moves during widths 0/4/6/8, respectively;
the latency samples cannot establish a live improvement or an Elo gain.

`scripts/campaign_service.py scale --slots N` applies only the protocol's mutable
slot count. It stops the supervisor, restarts unfinished workers on the same
pinned source and ledgers, and restores launchd supervision even on error.
Completed outcomes remain; interrupted seeds are replayed unchanged. A planned
restart uses one of the existing three worker starts. It refuses exhausted
workers or an experiment outside active offline evaluation.

## New preview learning: protocol before training

`search/train_preview_residual.py prepare --out DIR` writes the protocol before
extracting examples or fitting a model. It hashes source files, the incumbent
model, and the training code. Public replay games are deduplicated; our account,
players whose normalized names contain `bot`, and Illusion games are excluded.
This is not a claim that every remaining account is human. Illusion games are
excluded because historical preview extraction may attribute a disguise to the
wrong brought set.

Examples need at least four of the rain team's six species, a known brought four
and lead pair, both ratings, and an opposing Kingambit, Milotic, Annihilape,
Gardevoir, Sylveon, or Floette-Eternal. Whole games are ordered by numeric battle
ID as a chronological proxy and split 70/15/15. Both players from a game stay in
one split. Prepared arrays and manifests are frozen before fitting.

One MPS model learns a bounded correction to the incumbent logit using explicit
lead/bring-by-opponent interactions. A fixed Elo offset accounts for rating
differences during training; inference uses equal ratings. The hidden layer has
32 units; the correction is capped at ±0.35 logits. AdamW uses lr .001, decay .01,
batch 1024, at most 40 epochs, patience five, seed 20261023, two CPU threads and
half the MPS working-set allowance. Checkpoint selection uses validation only.

The single test evaluation requires at least 300 validation and test rows each,
at least .002 lower log loss on both, and no worse test Brier score. Failure does
not trigger retuning. The incumbent may have seen some historical test games;
the new correction has not. Prediction improvement is not evidence that changing
plans would have caused more wins.

Inference is opt-in through `preview_residual`. It acts only on the exact rain
six and the specified opposing matchup groups, and considers only legal plans
within .025 probability of the incumbent's best legal plan. It cannot be stacked
silently with c10's preview guard. It validates the incumbent checksum and model
dimensions. No default live option enables it.

Even a successful predictive test only qualifies the artifact for a separately
preregistered offline strength gate, followed by a fresh ladder A/B after c10.
The trainer contains no promotion path. Existing promotion preflight can stage
the immutable residual JSON without linking the mutable training dataset.

Private evidence is retained under `data/ml/value/implementation-20261023/`.

## First fitted candidate: rejected

The frozen preparation produced 8,868 examples from 8,867 games: 6,207 training,
1,330 validation, and 1,331 test rows. MPS fitting and evaluation took 1.53 seconds.
Validation log loss improved from .659832 to .658777, below the required .002
improvement. Test log loss worsened from .647704 to .648898 and Brier score from
.227871 to .228331. The candidate failed the fixed rule, remains disabled, and
does not enter a strength gate. No follow-up parameters were selected on this test.

C10's allowed slot setting was changed from eight to six. Both workers restarted
from the original pinned commit with their ledgers intact, using their second
start; the supervisor and live player continue running. This is a resource
adjustment, not a change to either candidate or its statistical rule.
