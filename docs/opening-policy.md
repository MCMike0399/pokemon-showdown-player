# Opening and Mega policy

The `opening-v1` feature profile scores team preview and joint turn choices using
our exact sets and the opposing information visible at decision time. Existing
checkpoints keep their original profile. The new profile is an explicit candidate
that needs new rollouts and evaluation before it can replace a playing actor.

## Team preview

Preview compares coverage and opening pressure against the visible opposing six,
with a cost for incoming threat and unfavorable estimated speed. Rain-dependent
attacks depend on bringing a setter; leading the setter provides immediate rain,
while a benched setter needs an entry turn. Intimidate and Fake Out contribute
bounded support estimates.

Each selected four is compared with at most one prospective Mega at a time. Both
Mega candidates may be brought, but the plan never assumes both can evolve.
Bench order does not change the explicit preview prior. All complete legal
commands remain in the mask, and the actor still samples with its checkpoint's
stored preview temperature.

Opponent stats and unrevealed moves are estimates. Observed moves and accepted
team sheets provide stronger inputs when present. A species preview does not
reveal an opposing item, spread, Mega choice or next command. The scorer does
not obtain an omniscient simulator stream or turn a battle win into an expert
preview label.

## Prospective Mega forms

When a selected command includes Mega Evolution, the scorer substitutes the
matching form's typing and ability. With one matching own set and cached format
rules, it calculates exact prospective stats with the pinned simulator's nature
and Champions stat-point rules. Missing own sets preserve observed stats and
mark the prospective stats as uncertain instead of inventing exact values.

Damage pressure considers the selected form, contact bonuses, known immunities,
weather, fresh charge versus locked release, and the partner's Helping Hand.
Incoming threat and estimated ability to act use the new defensive typing and
Speed. There is a small cost for spending the shared Mega option. Protect uses
incoming pressure and repeated protection history; switching also has an entry
cost. These are bounded tactical estimates, not exact predictions of the
opponent's commands or damage rolls.

The prospective form is encoded into the selected action as well as its prior.
The prior is smoothly bounded to avoid assigning identical saturated scores to
distinct strong actions. Public colored HP uses the corrected parser for this
profile. Tensor dimensions stay unchanged, and different profiles are kept
separate during PPO training.

The `mega-v1` ablation preserves legacy state, preview and non-Mega actions. It
adds the difference between prospective-form and base-form valuation only to
commands that include Mega, plus prospective action features. This separates
Mega changes from the combined opening policy during development evaluation.

The narrower `opening-v2` profile preserves those same legacy turn inputs and
non-Mega action scores while adding matchup-aware preview and prospective Mega
valuation. Preview action features canonicalize equivalent bench orders; the
complete legal mask and command probabilities remain explicit. This isolates
the opening adjustment from a general rewrite of turn scoring.

## Learning matchup preferences

The optional `matchup-v1` actor architecture adds a learned product of state and
action embeddings to the existing score. This provides a direct signal for
learning different action preferences against different opposing teams. An
explicit candidate migration copies the existing actor and critic and initializes
the added score to zero, preserving every starting logit. Old checkpoints load
as `concat-v1`; architecture metadata selects the correct network on reload.

A checkpoint can also weight preview decisions more heavily in the PPO objective.
The weight changes optimizer emphasis, not the recorded action distribution or
its collecting likelihood. Existing checkpoints retain weight one. The matchup
experiment uses weight four, with unchanged clipping and KL validation.

## Isolated training and evaluation

```bash
.venv/bin/python scripts/train_opening.py \
  --root data/ml --output artifacts/opening-training \
  --practice 128 --development 96 --final 200
```

Use `--matchups --rounds 4` to train four sequential fresh batches with the
matchup architecture. Each round freezes its own collecting checkpoint and
uses new seeds. `--initial-checkpoint` can warm-start the opening candidate from
a retained local candidate; the comparison incumbent stays separately frozen.
Use `--profile opening-v2` for the focused preview and Mega adjustment.

The experiment freezes the incumbent checkpoint, focused team, opponent pool,
research/scout inputs, source generation and dex signature. It collects fresh
sampled complete local games under `opening-v1`, validates their original
collecting likelihoods, and fits PPO with a KL budget of 0.03. Legacy private
requests are not re-encoded into new-profile PPO trajectories.

Development compares the incumbent, combined prior, Mega-only ablation and
trained combined actor with identical seeds, sides and opposing policies. Only
the trained actor can proceed to the independent final partition, and only
after passing the development gate. Both gates require clean complete games,
at least a ten percentage-point win margin and an exact paired one-sided
p-value at most 0.05. Local scripted and frozen opponents do not establish a
human ladder gain.

The output retains partial progress and supports resuming the same command.
Source or frozen-checkpoint changes invalidate that experiment; retain its
evidence and begin a fresh output directory. Reports do not stage a candidate
automatically. A passed candidate can be staged through the existing promotion
interface and is installed only before matchmaking, with no pending live game
and an unchanged parent revision and source generation.

## Shared resource limits

Experiment simulator waves share advisory slots with production background work,
while their experience and candidate checkpoints remain isolated. CPU, free
memory, swap-out rate, disk and MPS allocation limits continue to apply. Yellow
macOS memory pressure permits work when those limits pass; red pressure stops
new work and causes an unsaved optimizer update to yield without consuming its
source experience. Live recording continues independently.

## Initial validation results

The initial combined-prior candidate trained on 128 complete local games and
1,048 decisions. It won 62 versus 57 games in a 96-case paired development
comparison and failed the promotion gate.

The matchup architecture with four fresh batches trained on 512 games and
3,940 decisions. It passed development at 70 versus 55 wins over 96 paired
cases, with paired p = 0.0068, but tied 126 versus 126 wins over the independent
200-case final partition. That final result rejected the candidate.

The narrower `opening-v2` candidate trained on 512 games and 3,831 decisions.
It tied the incumbent at 70 wins each over 96 paired development cases and
was rejected; its independent final partition was not opened.

All completed comparisons had clean terminal results and matched seeds, sides,
teams and opposing policies. Failed source-change and SQLite-contention attempts
remain separate partial evidence. These experiments support the implementation
and recording checks, but do not establish improved playing strength. Failed
candidates do not enter the ordinary live promotion slot.
