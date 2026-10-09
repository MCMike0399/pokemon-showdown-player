# Strategic VGC policy research

Research date: 2026-10-08, America/Mexico_City. Scope: the existing Champions M-C rain team and improvement work following the 500-battle milestone. This document records primary-source research and proposed experiments; it does not claim that any proposal has won an evaluation or been deployed.

## Wolfe video: verified transcript and useful observations

The requested video is WolfeyVGC's **I Entered a Tournament with a New Mega Pokemon**, published September 12, 2026, duration 2:19:23. It concerns Mega Golisopod in Pokémon Champions. Metadata and the English caption track were retrieved directly from YouTube. All 3,603 timestamped caption segments were available; captions can contain spelling errors. The plain transcript remains temporary research scratch data, outside the repository. No full transcript, signed caption URLs or browser tokens are included here. [Video](https://www.youtube.com/watch?v=Ai3L_iwn5oE).

Retrieval limitation: ordinary signed `timedtext` requests returned HTTP 200 with empty bodies, and the transcript panel/API returned HTTP 400. The background T3 browser successfully loaded the video's normal caption request, including its browser-generated proof-of-origin parameters. Reading that legitimate player request produced the caption track. This is transcript-backed research, despite the panel failure.

The following are short paraphrases, not quotations or expert action labels:

| Timestamp | Observed reasoning |
| --- | --- |
| [01:22](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=82) | Builds around Golisopod setup and healing, then supplies rain protection, redirection and speed control. |
| [05:24](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=324) | Chooses Scarf/Adaptability Basculegion to reduce dependence on rain. |
| [09:03](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=543) | Keeps favorable Golisopod in reserve to avoid opening Intimidate. |
| [10:05](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=605) | Notices Trick Room helping Golisopod can compromise fast Basculegion. |
| [13:50–14:32](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=830) | Reassesses the omitted rain setter after a sand reset undermines his speed plan. |
| [31:05–31:54](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=1865) | Compares redirection with Helping Hand using knockout thresholds and switch coverage. |
| [41:05–41:32](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=2465) | Repositions to preserve the winning pair and future terrain renewal. |
| [1:15:50–1:17:14](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=4550) | Constructs a line preserving healthy Gardevoir even if the other two Pokémon fall to critical hits. |
| [1:24:37](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=5077) | Anticipates an opponent's next-game counterpick and changes offensive mode. |
| [2:11:50–2:13:39](https://www.youtube.com/watch?v=Ai3L_iwn5oE&t=7910) | Audits positioning, terrain renewal and damage checks rather than crediting a knockout alone. |

These episodes inspire the engineering hypotheses below. They do not establish globally optimal actions. His six Pokémon differ from our team; transfer the planning process, not his composition or best-of-three assumptions. Our ladder is a different observation regime, and open-sheet facts must be used only when actually received.

## Primary mechanics that bound a correct planner

The dependency is pinned to Showdown commit `c046106cbe075931b1ff8d8b800ff5be47a85f96`. Champions unmodified-level stats use base + stat points + 20 for non-HP and base + stat points + 75 for HP, followed by nature truncation. Its Trick Room action-speed implementation negates effective Speed. Exact own sets are therefore preferable to generic main-series approximations. [Pinned Champions implementation](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/mods/champions/scripts.ts).

Rain normally lasts five turns, eight with Damp Rock. Its Water damage multiplier is 1.5 and Fire multiplier 0.5. The public weather log emits upkeep events; treating every upkeep as a new weather activation would corrupt duration tracking. [Pinned weather implementation](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/conditions.ts).

Electro Shot boosts Special Attack before attacking and skips charging in effective rain. Helping Hand has priority +5 and multiplies the ally's move power by 1.5. Psychic Terrain protects grounded targets against opposing priority; it does not grant blanket priority immunity to airborne targets or forbid friendly support. Trick Room has priority −7, normally lasts five turns, and using it again removes it. Weather Ball's type and power depend on effective weather. These details make joint action order essential. [Pinned move implementations](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/moves.ts).

Tough Claws boosts contact moves rather than every attack. Armor Tail has its own priority blocking condition; it must not be conflated with terrain grounding. Intimidate and Emergency Exit create entry and survival consequences distinct from raw type coverage. [Pinned ability implementations](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/abilities.ts).

The simulator applies spread damage modifiers and resolves move targets and events. Reuse or compare against those mechanics for regression scenarios, while constructing opponent hypotheses solely from information visible before the decision. Reading the simulator's actual opposing private sets during action selection would contaminate an experiment. [Pinned battle actions](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/sim/battle-actions.ts).

## Current implementation seams inspected

These are local code findings, not external research claims. Recheck line numbers after implementation work.

| Existing seam | Present capability | Highest-value extension |
| --- | --- | --- |
| `battle_state.observe` | Public species, reveals, boosts, field/weather, recent moves, exact own requests | Field activation/remaining-turn intervals, entry timing, move order observations, target history. |
| `ml.preview.score` | Own attacks versus opposing species plus simple rain synergy | Joint bring-four, lead pair and intended Mega/backup plan evaluated against plausible opposing openings. |
| `ml.tactics.damage` | Approximate damage, weather, STAB, boosts and some immunities | Champions-correct stat intervals, prospective forms, terrain/items/abilities and order-dependent action execution. |
| `ml.mechanics.score` | Joint attacks, Helping Hand, charging and selected mechanics | Support value from changed KO/survival outcomes, cancellation of an enemy action, switching/entry consequences. |
| `ml.scout` | Pre-turn public observations predicting executed moves | Targets, switches, Protect/redirection alternatives, uncertainty and within-game adaptation. |
| `ml.brain` / recording | Versioned actor attribution, legal actions and rich snapshots | Observable decision explanations, postgame error tags and separate auxiliary targets. |

`battle_state` currently retains a weather name and field membership without duration. Its recent history retains side/slot/move/turn, but not move targets. `ml.tactics.stat` uses a base-stat-plus-35 fallback that is not the Champions neutral zero-point formula. That fallback should remain explicitly uncertain, rather than be relabeled as an exact inferred opponent stat. `ml.tactics.score` penalizes ordinary switching uniformly and primarily values immediate damage; this creates a plausible mismatch with resource-preserving lines. Proposed extensions must be versioned because old PPO likelihoods depend on their original encoding and policy.

## Actionable hypotheses and experiments

The following are proposed engineering designs. They are not conclusions derived from the video's results.

### 1. Choose a plan at preview, with an escape route

Enumerate the legal bring-four and lead pairs. Score each against a compact distribution of plausible opposing lead pairs and known team-sheet moves, when available. Keep unknown moves/items/abilities probabilistic. Distinguish rain pressure, slow control and fast special offense. Allocate an intended Mega and a backup Mega only as a planning hypothesis; do not force exactly one stone holder into every selection.

For each opening, ask which opposing threats prevent the selected endgame, which Pokémon removes them, who can enter safely, and whether the bench can reset weather or protect priority-sensitive attackers. Preserve a setter in reserve when an immediate lead would surrender weather control. Prefer actual matchup sensitivity over merely lowering preview temperature.

**Falsifiable checks:** opposing-six substitutions change meaningful bring/lead preferences; bench-order permutations do not change utility; left/right slot swaps preserve equivalent choices; selected plans improve turn-two survival and complete-game wins against reserved opposing teams.

### 2. Score the whole joint turn, including action denial

Construct the proposed own board after switches and Mega changes. Update rain, terrain, Intimidate, abilities and exact own stats before estimating outcomes. Order actions by priority and effective speed, with uncertain opponents represented as intervals or a small mixture. Discount damage from an attacker likely to faint or be disabled before its action. Count the value of KOing or flinching a threat before it acts, not only lost HP.

Support should change a concrete outcome. Helping Hand is valuable when it changes a KO or forces a useful switch; Protect is valuable when it denies likely damage, consumes an adverse field turn, or lets the partner progress. A fixed support bonus can encourage stalls. Double-target damage should stop earning value after the target is already removed, and should consider Protect/redirection alternatives.

**Falsifiable checks:** support versus attack rankings flip near a KO threshold; priority moves lose utility only against the relevant blockers; switch-to-rain plus Electro Shot is evaluated as a joint action; sun overwrite is considered before charging/damage; repeated Protect and ally splash damage are handled.

### 3. Make speed and field control global resources

Evaluate speed control against all surviving own and plausible opposing Pokémon, rather than the current slow attacker alone. Track durations or bounded duration intervals for weather, terrain, Trick Room, Tailwind, screens and Protect chains. A switch can give up immediate damage to retain a future reset; value that trade according to the remaining opponent threats.

**Falsifiable checks:** Trick Room can rank below attacking when fast reserves need to finish; it can rank above attacking when slow reserves need to enter; Protect/switch value rises near hostile Tailwind expiration; healthy setter preservation improves games against weather resets without inducing systematic passivity.

### 4. Use opponent behavior without pretending intentions are observed

Executed public moves are trustworthy labels for what happened; they are not always the intended action distribution because fainting, immobilization and called moves censor intentions. Add target/switch/Protect observations where unambiguous. Maintain within-game tendencies with strong shrinkage toward format priors, so one Protect does not imply permanent Protect bias. Compare likely aggressive, defensive and positioning replies instead of assuming one predicted reply is certain.

One possible bounded planner objective is `mean utility − risk_weight × downside`, with risk conditioned on the game state. When ahead, retain robust lines that survive plausible counters; when behind, consider a lower-probability route that actually permits a win. Calibrate that behavior against real outcomes rather than adding an unconditional aggressive or cautious style.

**Falsifiable checks:** prediction log loss/Brier score improves on whole held-out battles; label extraction excludes future reveals and ambiguous replacements; policy improvement persists against opponents outside its training pool and against an adversarial response panel.

### 5. Convert postgame review into trustworthy learning targets

Produce concise outcome-linked error tags: pre-action KO, unproductive support, rain dependency without an entry plan, exposed priority, adverse speed reversal, weather overwritten, ineffective/immunity attack, or lost sole endgame answer. Record evidence and uncertainty; do not infer that an unplayed alternative certainly won.

The 500-battle corpus can support outcome prediction, scout training, plan/outcome associations and mechanical failure mining. Old trajectories remain attributed to their collecting actor. Re-encoding them under new mechanics does not make them on-policy PPO examples. Winner-only imitation can copy lucky or incoherent moves; reserve imitation labels for defensible demonstrations or use explicit off-policy learning with its own validation.

**Falsifiable checks:** replay-prefix reconstruction reproduces recorded facts; failure tags have manually checked precision; complete-battle train/validation splits prevent adjacent-turn leakage; new diagnostics continue after every live game without blocking terminal recording.

## Evaluation and deployment criteria

Freeze the team fingerprint, collecting checkpoint, format/dex pin, data cutoff, seeds and opponent pool. Compare preview-only, turn-only and combined candidates separately before selecting one final candidate. Use paired seeds and alternate sides against rain, sun, sand, Trick Room, Tailwind, terrain/priority and bulky positioning teams. Include several actual opponent policies; a single exploitable heuristic can make a candidate look strong.

VGC-Bench treats team-dependent strategy and generalization as central evaluation problems. Its experiments demonstrate that narrow single-team success does not establish robustness across a broader team distribution. It supplies behavior-cloning, self-play and population-based baselines, while leaving team building open. This supports evaluating policy/team pairs and opponent diversity here; it does not establish a performance guarantee for our small model or Champions mechanics. [VGC-Bench paper](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/angliss2026vgc.pdf), [Authors' implementation](https://github.com/cameronangliss/vgc-bench).

Report paired win/loss changes and uncertainty alongside decision latency, illegal/rejected choices, stall rate, turn-two survival, moves executed before fainting, effective support, setter resets and endgame conversion. Development panels guide iteration; final reserved opponents/seeds determine the promotion claim. A handful of subsequent live games provide behavioral evidence, not a precise estimate of ladder improvement.

Deploy between completed games with matching source/config readers and a preserved rollback checkpoint. Verify the next recorded game identifies the expected source generation, checkpoint hash, feature profile and team. Monitor whole games, inspect decision reasons for remaining mistakes, and feed their public evidence into review/scout data. Separate a proven mechanical correction from an experimentally promoted strength improvement.
