# Campaign c13: which team should the search play? (2026-10-26)

## Why the team, now

Campaigns c1 to c12 changed the search (Protect logic, endgame depth, preview models,
scouting, thinking budget, a belief-game root solver). None produced a measurable live
gain. The final gates: c10 endgame 108 vs 106 of 785 pairs, c10 preview 8 vs 10, c12
belief game 46 vs 59 (futility stop at 400). Over 6,054 rated games the account wins
49.9% and sits at about 1,240 to 1,300: the search plays exactly at its rating.

Over the same period the team never changed. Since 2026-10-10 every game used
`Rain-Recife-special-stat-fix` (Garchomp-Mega-Z, Politoed, Incineroar, Archaludon,
Farigiraf, Golisopod). The only team comparison was the c4 team arena, which played 30
games per team against the imitation opponent. That is far too few to resolve
anything, and the imitation gate is known to be miscalibrated (docs/c9-live-ab.md).

## What the data says

`search/human/team_effects.py` reads the public replay corpus: 111,564 rated games and
197,708 sides, excluding bots and our account. For every side it computes the score minus
the Elo expectation from both players' ladder ratings. Each player is weighted equally,
so one strong player climbing on a fresh account cannot make a team look strong.

- **Teams matter less than players in human hands.** After the per-player weighting,
  every team with at least 100 games falls within about ±70 Elo of average, with
  confidence intervals of ±40 to ±90 Elo. Our six played by humans: −42 [−129, +39]
  (54 players).
- **Our live games show matchup holes.** These are 4,742 rain games since 2026-10-14,
  scored as Elo residuals. Opponents with Floette-Eternal −114 ±34, Staraptor −104 ±34,
  Milotic −98 ±32, Sylveon −66 ±36 and Gardevoir −53 ±38: fairy attackers against a
  dragon-heavy team, plus Competitive and Defiant punishing Intimidate. Against weather
  mirrors we are fine: Politoed +86, Golisopod +84, Grimmsnarl +75.
- **Long games are lost.** We win 100% of games under 3 turns (opponents quit early),
  67% at 3 to 5 turns, 45% at 6 to 8 turns and 33% at 9 or more turns. Wins by opponent
  forfeit: 1,147 of 2,361. This is a pilot limitation the team cannot fix alone, but a
  team that closes games sooner (spread pressure, speed control) plays to the search's
  strength.
- Accepting open team sheets: −14 ±15 with sheets versus +3 ±13 without. No significant
  difference.

## Candidates (tournament-proven, rain-compatible)

Selection used web research on Pokémon Champions Reg M-C results (Victory Road replica
list, Pikalytics and Limitless top teams), restricted to the rain and weather family the
project already knows. A candidate also needed good ladder evidence in the corpus and
exact moves, items and natures from open team sheets. Stat points are hidden on sheets,
so each set uses the most common ladder spread for its species and nature
(`search/priors.load_usage`). Both teams pass the official validator.

1. **`Rain-Gengar-Helm-Baltimore`.** Billy Helm, Baltimore Regional (Top 16, 11-3).
   The same rain core has won several community events (9-0, 7-0, 6-0). It keeps
   Politoed, Archaludon, Incineroar and Golisopod. Gengar-Mega (Shadow Tag, Perish Song,
   Sludge Bomb) replaces Garchomp, and Rillaboom (Fake Out, Grassy Glide) replaces
   Farigiraf. It removes the fairy weakness: one dragon instead of two, and a Poison
   attacker. Corpus: 336 games, 107 players, +28 [−31, +89] per player.
2. **`SunRain-Mev-Brisbane`.** Kiran Singh, Brisbane Regional champion (10-1) and VR
   October Challenge #2 champion. Pelipper rain plus Charizard-Mega-Y sun, with
   Grimmsnarl screens, Venusaur and Garchomp-Mega-Z. This is the most-played strong team
   on the ladder. Corpus: 870 games, 285 players, +25 [−10, +60] per player, +62 pooled.

Sources: https://www.vrpastes.com/WkQ3iI3V, https://vrpastes.com/4OR2JHdH,
https://victoryroad.pro/champions-replica/, https://www.pikalytics.com/topteams.
The sets are in `data/ml/value/campaign-c13/candidate-teams.json` and `teams.json`.

## Protocol (written before the first game)

- **Design.** King-of-the-hill rounds on the live ladder (`scripts/team_ladder.py`).
  Each round is one A/B (`search/ab_test.py`, spec `data/ml/value/ab/current.json`,
  ids `c13r1`, `c13r2`). The single factor is `team`: base arm = champion,
  factor on = challenger. Games alternate in seeded balanced blocks of two.
  The search configuration (`search/config.py` LIVE, preview `pvalue-661448b6a320fce0`)
  is identical in both arms. Only the six Pokemon change.
- **Order.** Round 1: `Rain-Gengar-Helm-Baltimore` vs `Rain-Recife-special-stat-fix`.
  Round 2: `SunRain-Mev-Brisbane` vs the round-1 champion.
- **Decision.** SPRT of 0 vs +40 Elo, alpha 0.05, beta 0.10, minimum 300 and maximum
  3,000 games per round (the c9 rule). Forfeit wins count as wins, because opponent
  quitting is part of what a team earns on the ladder. H1 makes the challenger the
  champion. H0 or the cap keeps the champion.
- **End.** A new champion keeps playing every game through spec `c13final`, with
  nothing randomised. Making it the LaunchAgent's `--team` is a human step. With no new
  champion the spec goes inactive and the runner plays its own `--team`. Setting the
  spec `"active": false` stops the campaign; the controller then records `stopped`.
- **Safety.** An unknown team name in a spec plays `--team` and logs `ab_error`. A room
  resumed after a restart that the current round does not know plays the round's base
  team. The controller never replaces an active spec belonging to another experiment.

## Operation

    .venv/bin/python scripts/team_ladder.py status --dir data/ml/value/campaign-c13
    .venv/bin/python -m search.ab_test status
    tail data/ml/value/campaign-c13/ladder.log

Controller LaunchAgent: `dev.pokemon-showdown.campaign-c13`, running from the pinned
worktree `<checkout>.c13`.

## Result (2026-10-30)

- Round 1: `Rain-Gengar-Helm-Baltimore` H0 after 302 games, −75 Elo (score 0.437 vs 0.543).
- Round 2: `SunRain-Mev-Brisbane` H1 after 897 games, +62 Elo (score 0.556 vs 0.468),
  LLR 3.05. This is the first live gain since c5. Opponent ratings were balanced between
  the arms. The edge comes from opponents rated 1200 or more, plus more opponent forfeits.
- From 2026-10-30 every game played the Mev team (spec `c13final`): 1344 ± 33 over 452
  games (Elo monitor segment 16). The runner's `--team` became `SunRain-Mev-Brisbane`
  on 2026-10-31, and campaign c14 (docs/c14-preview.md) followed.
