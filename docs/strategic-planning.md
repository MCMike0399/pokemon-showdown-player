# Strategic VGC planning

The strategic feature profiles add a joint-turn evaluator to the existing actor.
They are candidate policies, selected by checkpoint metadata; legacy checkpoints
retain their original features and likelihoods. The network architecture and
public legal mask remain unchanged.

## What a candidate considers

`strategic-v1` changes preview and battle scoring. `strategic-turn-v1` and
`strategic-preview-v1` isolate those changes for experiments. Preview compares
the selected four, lead pressure, coverage, one prospective Mega at a time,
rain availability, possible sun disruption, Intimidate and speed-control fit.
Equivalent bench permutations have the same encoding.

The battle evaluator resolves both own commands against up to six weighted
opponent response hypotheses. It estimates switches, entry weather/terrain,
Intimidate, exact own Mega stats, disclosed opposing Mega possibilities,
priority, effective Speed, Trick Room, Tailwind, paralysis, weather speed
abilities, terrain, screens, contact/item effects, Protect, Wide Guard,
redirection, Helping Hand, Fake Out and charging. A faster knockout can deny an
opposing action; a later attack can retarget after the selected foe faints.
Known survival items and residual damage affect approximate survival.

Opponent scenarios cannot spend two Megas or switch two slots into one reserve.
Only publicly revealed opposing reserves become switch hypotheses. An accepted
team sheet constrains move hypotheses. Unknown spreads remain estimates; the
planner never reads the simulator's private opposing team.

The evaluator adds a small preservation cost for losing a scarce answer to a
remaining threat or a weather enabler. Once four opposing picks are revealed,
unselected preview species leave the future-threat set. Speed-control value
compares offensive initiative across the active pair and remaining own party.
Pessimism increases with an approximate resource advantage and decreases when
behind. These are bounded heuristics, not calibrated winning probabilities.

## How the battle corpus contributes

`scripts/audit_strategy.py` freezes a requested terminal-game cutoff, verifies
unique room attribution, and extracts executed opponent moves from captured
terminal logs or the existing public archive. The initial 500-game audit had
all 500 logs and 4,696 executed opponent moves across 260 species/form names.
Those frequencies inform uncertain replies; they do not reveal an individual's
unseen moves, label optimal choices, or turn old recordings into fresh PPO.

Population evidence is stored inside each candidate checkpoint, with source
episode identities for deduplication. Visible moves and recent observed
behavior influence hypotheses with shrinkage. It is possible for executed
frequencies to misrepresent intended moves because fainting and immobilization
censor actions; predictions remain explicitly limited to this evidence.

`Brain.finish` attaches guarded feedback to its single immutable terminal write:
missing matching executions, own faints, weather changes,
observed failures, dry Electro Shot selections, opposing leads and executed
moves. A malformed auxiliary snapshot is recorded as an auxiliary error and
cannot prevent terminal attribution. Saving a second copy after completion
would discard feedback because the store preserves completed episodes.
Missing execution is not automatically called a mistake, and a dry selection
can be rescued by a partner weather switch.

Routine learning folds new terminal ladder reports into a candidate's behavior
evidence, deduplicates them, and preserves the active checkpoint. PPO uses only
compatible sampled encoded trajectories. The combined weights/evidence update
must pass the existing paired evaluation and between-game promotion gate.

## Experiments and limitations

`scripts/experiment_strategy.py` creates immutable incumbent/candidate copies,
freezes scout/research inputs, declares seeds/sides/opponents/source/dex hashes,
and compares sampled complete games in the pinned official simulator. It tests
the combined, battle-only and preview-only profiles on development cases. Only
the selected development-passing candidate opens an independent final panel.
Promotion still requires clean matching cases, the declared win margin and
paired significance. Partial and rejected evidence remains preserved.

`scripts/inspect_strategy.py` measures CPU decision latency and opposing-team
substitution sensitivity from exact captured historical observations. These
diagnostics do not establish that a recommended alternative wins.

This is approximate one-turn planning. Unknown switch-ins, undisclosed opposing
Mega stones, exact damage rolls, hit/miss branches, field-expiry horizons,
several recovery/setup/status interactions and deeper combinations remain
incomplete. Damage estimates and survival thresholds are not a replacement for
the official simulator. A passing local panel does not establish tournament
strength or a particular live ladder win rate.

The design draws testable hypotheses from two caption-grounded studies:
[new Mega Golisopod tournament](vgc-strategic-research.md) and
[first Champions tournament](vgc-wolfe-second-video.md). It transfers Wolfe's
planning questions rather than copying his team or treating video choices as
demonstrations for a different battle.
