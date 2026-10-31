# Campaign c14: let the Mev team bring what humans bring (2026-10-31)

## Why

Campaign c13 (docs/c13-team-ladder.md) moved the live team to `SunRain-Mev-Brisbane`,
+62 Elo, the first live gain since c5. The search still plays that team as a fixed four.
Over 885 live games the team preview (`pvalue-661448b6a320fce0`, campaign c4) brought:

| Pokemon    | live (pvalue) | humans with this exact six (1,046 games, 298 players) |
|------------|---------------|--------------------------------------------------------|
| Archaludon | 100%          | 85% seen                                               |
| Grimmsnarl | 100%          | 63% seen                                               |
| Charizard  | 95%           | 63% seen                                               |
| Pelipper   | 87%           | 71% seen                                               |
| Garchomp   | 16%           | 53% seen                                               |
| Venusaur   | 3%            | 42% seen (46% at 1400+)                                |

"Seen" means the Pokemon appeared in battle, a lower bound on brought. Humans lead
Charizard + Venusaur most often: sun, Chlorophyll and Sleep Powder. Our bot almost never
plays that mode. The pvalue preview scores plans with a matchup value learned from human
games, and for this six it converges on one plan in 83% of games.

The human-imitation preview `preview-1455f4a55d18112a` (search/human/preview_model.py,
the preview of the imitation opponent since c10) reproduces the human usage when it
samples. Taking its most likely plan for the same 885 opponents, it brings Venusaur 63%,
Garchomp 24% and Grimmsnarl 37%, and leads Charizard + Venusaur in 558 games.

Earlier preview campaigns (c7 ensemble, c10 residual) changed the preview for the rain
team, using offline gates that are known to be miscalibrated. This one is team-specific,
copies what strong humans do with this exact six, and is tested on the live ladder, the
method that found c13's gain.

## Protocol (written before the first game)

- **Design.** One live interleaved A/B (`search/ab_test.py`), driven by
  `scripts/team_ladder.py` with prefix `c14` and factor `preview`. Both arms play
  `SunRain-Mev-Brisbane` with the same search (`search/config.py` LIVE).
  - Off arm: the runner's `--preview pvalue:data/ml/human/models-preview/pvalue-661448b6a320fce0.json`.
  - On arm: `human:data/ml/human/models-preview/preview-1455f4a55d18112a.json:argmax`.
  - Games alternate in seeded balanced blocks of two (seed 20261401).
- **Decision.** SPRT of 0 vs +40 Elo, alpha 0.05, beta 0.10, 300 to 3,000 games (the
  c9 and c13 rule). Forfeit wins count as wins.
- **End.**
  - H1: every game keeps the human preview (spec `c14final`). Making it the runner's
    `--preview` is a human step.
  - H0 or the cap: the spec goes inactive and the runner keeps pvalue.
  - Setting the spec `"active": false` stops the campaign.
- **Safety.** An unusable preview spec plays `--preview` and logs `ab_error`. The
  preview change takes effect between games only (`arm_preview` event).

## Operation

    .venv/bin/python scripts/team_ladder.py status --dir data/ml/value/campaign-c14
    tail data/ml/value/campaign-c14/ladder.log

Controller LaunchAgent: `dev.pokemon-showdown.campaign-c14`, running from the pinned
worktree `<checkout>.c14`.
