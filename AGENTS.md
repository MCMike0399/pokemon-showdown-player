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
