# Simulator search agent

The live neural policy (PPO actor-critic, strategic-v1) plateaued at **348 W / 499 L
(41.1%)** over 847 ladder games. Further PPO candidates kept failing their paired gates
by a few points. This agent replaces per-turn decisions with **search over the
official Showdown simulator**, while keeping the learned model for team preview.

## How a decision is made

1. **Track the public state** (`search/tracker.py`): from our own player stream only,
   HP, status, boosts, volatiles, revealed moves/items/abilities, Mega Evolution,
   weather/terrain/Trick Room/Tailwind/screens and their remaining turns.
2. **Determinize the opponent** (`search/priors.py`): every hidden set (moves, item,
   ability, nature, stat points, and which unrevealed Pokemon were brought) is drawn
   from Smogon ladder usage statistics for this format, conditioned on everything the
   opponent has revealed. Open team sheets, when offered, replace the sampled
   moves/items/abilities/natures exactly.
3. **Rebuild the battle** (`search/engine.cjs`): for each sampled world the pinned
   simulator is constructed with our exact team and the sampled one, then mutated to
   the observed state.
4. **Simulate one turn for every action pair**: our legal joint actions (pruned of
   ally-damaging moves, Mega whenever available) against the opponent's, after a
   screening pass that keeps the strongest ~24 x 20. Every cell is a real simulated
   turn, so speed order, priority, Protect, Fake Out, spread damage, abilities and
   items come from the engine itself.
5. **Score and solve**: positions are scored by a material/HP/status/boost/field
   evaluation; the simultaneous-move matrix is solved by regret matching, and our
   action maximizes value against a blend of the opponent's equilibrium strategy and
   a greedy prior. Values are averaged over worlds.

Forced switches score each replacement by the value of the following turn. Our
choices are sent to the engine by move id (`move icebeam 2`), because locked requests
list only the locked move.

## Offline evidence (open team sheets off, as most ladder opponents)

| Test | Incumbent | Search |
|---|---|---|
| Rain mirror, head to head (40 games, sides alternated) | 14 | **26** (65%, one-sided p = .040) |
| vs 45 real human teams, same seeds/teams/sides (paired) | 19 (42%) | **30 (67%)** |

The real-team pool is 66 live opponents' open team sheets plus 12 tournament teams
(`search/build_real_teams.py`; stat points sampled from usage). Opponents are piloted
by the search agent. On it the incumbent scores 42% on the paired games (49% over
140), close to its live 41%. Paired outcome changes: 14 gains, 3 losses (McNemar
one-sided p = .0064). Teams built by sampling each Pokemon independently from usage
are not a valid proxy: the incumbent beat them 9-1.

These are simulator opponents, not people. The claim to verify live is the
improvement, not its exact size.

## Use

    .venv/bin/python search/fetch_usage.py                 # once: ladder usage priors
    .venv/bin/python scripts/browser_play.py --team Rain-Recife-special-stat-fix \
        --games 30 --chrome --search [--search-worlds 6 --search-engines 3]

`--search` implies `--no-record` (search choices are not PPO samples); finished games
are still imported into the experience store as `own-live-game` public battles. A
decision takes about 2-3 s.

Offline evaluation:

    .venv/bin/python search/run_match.py --a 'search:worlds=4;screen={"probe":4,"keep_opp":20,"keep_ours":24}' \
        --b incumbent --games 40 --out runs/mirror.jsonl
    .venv/bin/python search/trace_game.py 'search:worlds=2' incumbent 5002 p1 trace.json

`run_match.py` expects a frozen incumbent under `artifacts/search-breakthrough/frozen`
(`models/<format>.pt` plus an inference copy of the experience tables).

## Browser transport changes needed for a slower decider

The live model answers in about 100 ms; search takes seconds. That exposed four races
in the browser transport, all fixed in `ml/browser.py`:

- the persistent profile auto-logs in a few seconds after load (now awaited);
- the preview choice object exists only after controls render (now awaited);
- the client rewrites its local copy of a request while rendering, so freshness and
  the click guard compare the server `rqid`, not the whole object;
- move buttons appear only after turn animations, so clicks wait up to 90 s.

## Not yet done

- `search/selfplay.cjs`, `search/valuefeat.cjs`, `search/train_value.py` and
  `search/valuenet.cjs` produce, train (MPS) and evaluate a learned value function
  that can replace or blend with the hand evaluation (`value_path`, `value_beta`),
  and score team-preview plans (`preview_mode='value'`). Not trained or gated yet.
