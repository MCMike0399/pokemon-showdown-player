# Agent instructions

This harness gives an agent one regular Pokemon Showdown player connection.

For OpenClaw battle/team decisions, read `docs/agent-workflow.md` before playing.
For CPU/MPS training, schedules or feeds, read `docs/continuous-learning.md`.
For corpus changes, read `docs/continuous-learning-research.md`; retain source,
license and exact regulation, and validate observable-state labels.

Use the configured account in `.env`, log in once, and reuse its connection.
Read waiting requests before choosing a complete legal joint action. Keep the
collecting checkpoint fixed during a recorded game. Training uses background
resource budgets and candidate evaluation before promotion.

Keep credentials in `.env` or environment variables. Keep datasets, replays,
team stores, weights, reports and machine-specific settings under gitignored
local directories. Respect Showdown's rules and use the offline simulator for
bulk training. Use the shared host's configured compute headroom.

Run `.venv/bin/python -m pytest -q` after battle, model or feed changes. Run
`scripts/public_snapshot.py` before publishing a clean source snapshot.

## Live player and automatic promotion

- The ladder player (LaunchAgent `dev.pokemon-showdown.browser-player`) runs from a pinned
  worktree `<checkout>.live`, not from this checkout. Edits here go live only when deployed:
  `.venv/bin/python scripts/live_deploy.py --ref origin/main` (graceful restart; `--status`
  shows the pinned commit). Never edit `<checkout>.live` by hand.
- A campaign may be gating a candidate. `scripts/auto_promote.py` can commit `search/config.py`
  to main, push, deploy and restart the player at any time. Leave `search/config.py` unmodified
  here, work on a branch or worktree, and land changes through main. Changes to `search/`, `ml/` or
  `scripts/browser_play.py` reach live at the next deploy or promotion, so run the tests first.
- A running campaign's gates use their own pinned worktree (e.g. `<checkout>.c6`); don't edit it.
