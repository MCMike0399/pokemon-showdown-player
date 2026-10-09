# Politoed rain and the battle policy

Research date: 2026-10-07, America/Mexico_City. Target: `gen9championsvgc2026regmc`, Pokémon Champions Regulation M-C. This is a research and engineering recommendation; experiments and deployments are reported separately.

## Assessment

The campaign's original team is a credible, recently successful Politoed rain team. Its six Pokémon, items, abilities, natures and moves match **Luciano Begot's runner-up team at Recife on October 3, 2026**. The submitted team sheet is available directly from the tournament platform, and its roster connects Begot to the player name KSV S.Proud. The sheet does **not** disclose stat-point allocations. Our spreads are generated hypotheses, so the harness is not playing an exact reproduction of the finalist's complete build. [Submitted public team sheet](https://rk9.gg/teamlist/public/RE002-wN97aUf4Psz0ZJ/r9UAaJtg3GD2FYc6ZRl9), [RK9 roster](https://rk9.gg/roster/RE002-wN97aUf4Psz0ZJ), [RK9 results](https://rk9.gg/pairings/RE002-wN97aUf4Psz0ZJ).

There is no defensible evidence that this is **the best** Politoed team. A tournament placing combines player skill, matchups, build and luck. The main demonstrated construction problem in this repository is an allocation mistake on special-attacking Mega Garchomp Z, rather than evidence that Politoed or the six-Pokémon core needs replacing.

## Format and source checks

Official M-C runs September 9–December 2, 2026. It permits one Mega Evolution per battle and disallows duplicate held items. The new permitted Megas include Garchomp Z and Golisopod. This repository's format uses Champions mechanics, not Scarlet/Violet Regulation F/G/I. Earlier Politoed articles, SV Terastallization builds, and Champions M-A/M-B demonstrations do not establish current M-C strength. [Official M-C regulation](https://champions-news.pokemon-home.com/en/page/816.html).

Recife's competition is branded the **2027** Regional Championships because it belongs to the next competitive season, despite taking place in October **2026**. Do not reject this relevant event solely because its title contains 2027. [Organizer event and final standings](https://rk9.gg/pairings/RE002-wN97aUf4Psz0ZJ).

The simulator is pinned to `c046106cbe075931b1ff8d8b800ff5be47a85f96` in `package.json`; reproduce legality and stat calculations with that version. Champions uses up to 32 points per stat and a different stat formula. Here `evs` is the simulator field representing Champions stat points, not conventional 252-EV spreads. [Pinned Champions mechanics](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/mods/champions/scripts.ts), [Pinned Champions rules](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/mods/champions/rulesets.ts).

## Dated team evidence

| Evidence | Relevant result | Interpretation |
| --- | --- | --- |
| Luciano Begot, Recife, October 3 | Runner-up; current six Pokémon and disclosed sets match | Keep this as the main offensive-rain baseline; undisclosed spreads require separate optimization. |
| Aditya Subramanian, Baltimore, September 19–20 | Runner-up: Charizard Y, Golisopod, Farigiraf, Archaludon, Politoed, Grimmsnarl | Offensive Mystic Water Politoed also succeeds in a dual-weather structure. |
| Brady Smith, Baltimore, September 19–20 | Third: Gengar, Vivillon, Incineroar, Archaludon, Politoed, Rillaboom | Perish/Encore Politoed belongs to a different control strategy. |
| Billy Helm, Baltimore, September 19–20 | Tenth: Gengar, Golisopod, Incineroar, Archaludon, Politoed, Rillaboom | Another trap/control team; changing only Politoed does not recreate the archetype. |
| Anthony Liuzzo, Frankfurt, September 26 | Fifth: Vivillon, Politoed, Gengar, Incineroar, Rillaboom, Kommo-o | Useful whole-team alternative after the policy can execute Perish endgames. |

Recife evidence: [submitted team sheet](https://rk9.gg/teamlist/public/RE002-wN97aUf4Psz0ZJ/r9UAaJtg3GD2FYc6ZRl9), [official platform standings](https://rk9.gg/pairings/RE002-wN97aUf4Psz0ZJ). Baltimore evidence: [dated event](https://limitlessvgc.com/tournaments/441), [operator-maintained team database](https://limitlessvgc.com/tournaments/441/teams). Frankfurt evidence: [dated event](https://limitlessvgc.com/tournaments/443), [operator-maintained team database](https://limitlessvgc.com/tournaments/443/teams). Limitless is a team/results database, not the tournament organizer or a player's tactical report. Its entries support disclosed compositions and sets; they do not reveal the player's hidden stat points or prove optimality.

September Showdown M-C statistics at the 1630 weighting report Politoed at **7.35024%**, Pelipper at **13.89148%**, Archaludon at **16.00513%**, and Mega Golisopod at **15.88017%**, across **1,631,943** battles. These are rating-weighted usage numbers, not win rates or a head-to-head comparison of setters. Pelipper's higher usage does not justify replacing Politoed on a proven Politoed team. Keep Bo1 and Bo3 datasets separate. [Showdown September M-C usage](https://www.smogon.com/stats/2026-09/gen9championsvgc2026regmc-1630.txt), [separate Bo3 statistics](https://www.smogon.com/stats/2026-09/gen9championsvgc2026regmcbo3-1630.txt).

## Why team construction affects model performance

The team defines the actions, speed order, damage thresholds and strategic plans the model can execute. Rain increases Water damage, reduces Fire damage and lets Electro Shot attack immediately while boosting Special Attack. Weather Ball changes type and power with weather; retaining only its static Normal/50-power description misrepresents Politoed's central attack. Muddy Water has 85% accuracy and hits both opponents, so the correct choice can differ from accurate single-target Weather Ball. A policy must also value preserving or repositioning Drizzle before rain expires or opposing weather arrives. [Pinned weather conditions](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/conditions.ts), [Pinned move mechanics](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/moves.ts).

Both Mega stones are legal, but only one Pokémon can Mega Evolve. Choosing the wrong Mega, or failing to Mega Evolve, changes how the other slot should be piloted. Mega Garchomp Z becomes **pure Dragon with Levitate**, rather than Dragon/Ground: Earth Power no longer receives Ground STAB. Mega Golisopod becomes **Bug/Steel with Tough Claws**, base Speed 40. It is not a Swift Swim Water attacker; rain primarily helps its Fire matchup. These distinctions must apply when scoring a proposed Mega action, before the next request arrives. [Pinned species definitions](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/pokedex.ts).

The published VGC-Bench experiments find that strategies depend strongly on both teams. Narrow single-team success does not imply skill across many compositions; increasing team diversity worsened performance for the strongest single-team method while improving generalization to unseen teams. The paper leaves team building open. This supports controlling team variation and evaluating policy/team pairs; it does not prove which rain build our smaller local model will pilot best. [VGC-Bench paper](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/angliss2026vgc.pdf).

## Concrete adjustments to test

Local audit input: original `Campaign-20261007T182143Z-base` in `teams.json`. The following are engineering recommendations based on that frozen baseline, not claims about Begot's undisclosed spread.

1. **Correct Garchomp's attacking stat immediately in new generated candidates.** Its disclosed attacks are all special, but `generated_spreads` chose Attack from the un-Mega species' base stats. Original: `atk:32, spe:32, hp:2`. Corrected: `spa:32, spe:32, hp:2`. The pinned simulator gives Mega Garchomp Z Special Attack **177 → 212**, with Speed **203** and HP **185** unchanged. This is approximately 19.8% more Special Attack; damage and knockout improvements still depend on rounding and the matchup. Preserve the original team for comparison.
2. **Test bulk on Golisopod as a separate hypothesis.** Compare original `atk:32, spe:32, hp:2` with `atk:32, hp:32, spd:2`, keeping its disclosed nature, moves and item. More bulk and less Speed may fit Farigiraf's Trick Room mode; it may worsen some matchups outside Trick Room. Do not promote it simply because the Mega is slow.
3. **Retain offensive Politoed for the first comparison.** Its existing set matches the submitted finalist sheet. Perish Song/Encore with a defensive berry is a different archetype, ideally tested alongside Gengar and positioning support, rather than imported as an isolated supposedly superior Politoed set.
4. **Avoid blind per-species set swaps.** Same-format legality and item uniqueness are insufficient to preserve weather, speed and trapping plans. Preserve entire credited teams; label every generated spread and hybrid as an experiment. The planner should resolve attacking stat from actual moves, nature and possible Mega form, not base Attack versus Special Attack alone.

Validation performed: baseline, the Garchomp-only correction, and the combined Garchomp correction/Golisopod bulk hypothesis all passed the pinned offline `TeamValidator` with **zero errors**. This verifies legality, not competitive superiority. The Garchomp stat values were computed from the pinned `Battle` model after its Mega-Z form change. No live connection or battle was used for this research.

## Model and evaluation recommendations

Use weather-aware action features and preview synergy features in a versioned candidate checkpoint. Add effective Weather Ball type/power, rain-dependent Electro Shot charging, Water/Fire modifiers, prospective Mega typing/ability/stat changes, and the value of selecting the rain setter with its beneficiaries. Keep unknown opponent stats uncertain and do not use hidden simulator state. A raw team fingerprint is useful for attribution; it should not substitute for transferable composition features.

Evaluate four combinations: incumbent/original team, incumbent/corrected team, candidate/original team, candidate/corrected team. Hold opponent teams, seeds and sides fixed across comparisons. Then test the bulky Golisopod candidate separately. Use opponent coverage including sun, sand, fast Tailwind offense, Trick Room and opposing rain. Report complete-game win rates and uncertainty alongside Mega use, weather control, Protect/support use, rejected actions and timeouts. Ladder outcomes identify the performance of a policy/team pair against its encountered opponents; they cannot independently assign a loss to the team or model.

The next-battle recommendation is the mechanically corrected rain baseline with a gated policy candidate, not a claim that a new team or larger model is automatically stronger. Actual campaign counts, policy promotion decisions and any deployed adjustment belong in the accompanying run analysis.
