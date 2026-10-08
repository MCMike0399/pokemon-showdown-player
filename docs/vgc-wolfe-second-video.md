# Wolfe video: first Champions tournament

Research: 2026-10-08. Primary source: WolfeyVGC, [I Entered the First Pokemon Champions Tournament](https://www.youtube.com/watch?v=inQoYsuK2qE). Retrieved the actual English caption track through YouTube's browser player: 2,654 timestamped segments, last segment 2:17:52. The full transcript is a temporary private research input; this note contains paraphrases and implementation hypotheses, not a transcript.

## Timestamped evidence

| Video passage | Strategic observation |
| --- | --- |
| [18:04–20:05](https://www.youtube.com/watch?v=inQoYsuK2qE&t=1084s) | Weather plus Wide Guard can disable an opposing attack plan; base-form survival can justify delaying Mega. |
| [22:50–25:37](https://www.youtube.com/watch?v=inQoYsuK2qE&t=1370s) | Switches precede Mega weather activation; alternate Megas provide matchup plans. |
| [1:05:26–1:06:34](https://www.youtube.com/watch?v=inQoYsuK2qE&t=3926s) | A losing position warrants variance; four revealed picks eliminate absent support. |
| [1:13:58–1:16:15](https://www.youtube.com/watch?v=inQoYsuK2qE&t=4438s) | Misread boosted speed, unnecessary inaccurate chip, and costly scouting lose a winnable endgame. |
| [1:33:27–1:34:05](https://www.youtube.com/watch?v=inQoYsuK2qE&t=5607s) | Setup targets a specific speed threshold and subsequent attack window. |
| [1:37:01–1:38:59](https://www.youtube.com/watch?v=inQoYsuK2qE&t=5821s) | Preserve the sole answer to an unrevealed threat; numerical advantage alone can mislead. |
| [1:44:23–1:44:47](https://www.youtube.com/watch?v=inQoYsuK2qE&t=6263s) | Accurate single-target finishing removes avoidable comeback risk. |
| [2:12:46–2:13:26](https://www.youtube.com/watch?v=inQoYsuK2qE&t=7966s) | Post-loss diagnosis repairs resource preservation and early Mega commitment before changing picks. |

The video identifies its tournament ruleset as M-A around 4:06. Its metagame and team are historical examples, not current M-C recommendations. Named Pokémon and mechanics must be checked against the pinned simulator.

## Engineering hypotheses for Rain Recife

The following are our proposed experiments, not claims that Wolfe endorses this implementation or that the current model already implements them.

1. **Represent a winning route, not only turn damage.** For each selected four, record a rough plan: rain-enabled Archaludon pressure; fast Mega Garchomp Z coverage; or Farigiraf-controlled slower Mega Golisopod endgame. Identify the opposing threat each remaining Pokémon uniquely answers. Penalize sacrificing that answer unless the trade removes its assigned threat or secures a terminal win. Score the remaining bench as well as active HP.

2. **Treat Mega as an option with information value.** Compare both prospective Mega forms, support-compatible base forms, and staying unevolved. Charge an option cost when committing before the opposing Mega or decisive coverage is known. For this team, prospective Garchomp Z and Golisopod change typing/ability and cannot both Mega; a generic “Mega is stronger” bonus is insufficient. Exact forms are already documented in [the team research](politoed-rain-research.md).

3. **Forecast weather at execution time.** Branch on plausible opposing switch-entry weather and delayed Charizard Mega activation. A Politoed switch does not guarantee that rain persists until Electro Shot resolves. Distinguish setter on the field, setter available in the bench, weather duration, and irreversible loss of the setter. Apply the same principle to priority protection and speed-control expiry.

4. **Value Protect by the turn it creates.** Include partner progress, weather/speed-control turns consumed, residual survival budget, opposing setup/recovery, and the next required Protect success. Protect to scout a Choice lock is useful only if the resulting information can still be acted on. Do not give double Protect a blanket safety reward.

5. **Prefer knockout probability to expected chip when closing.** Calculate damage intervals and move hit probability separately. Compare accurate Weather Ball or other single-target finishes against Muddy Water spread value. Use a conservative preference when an existing winning route survives; allow status/flinch/critical-hit routes when ordinary lines lose. Never label a lucky win as proof of a high-quality decision.

6. **Update opponent beliefs by identity and context.** Keep revealed move/item facts separate from species priors. Track attack targets, setup versus Protect, weather-switch responses, Choice locks and switch preferences. Condition lightweight behavior counts on relevant situations, shrink toward population priors, and clear/retarget slot history on switches. Once four opposing picks are revealed, remove the two unselected species from future-threat hypotheses.

7. **Learn explanations from terminal games.** Add counterfactual tags to post-game review: wrong four/lead, missing threat answer, premature Mega, lost weather setter, speed-order mistake, avoidable accuracy risk, expiry/residual mistake, or opponent prediction error. Tags describe candidate causes, not proven causality. Train against frozen choice-time observations and retain provenance; never leak later revealed information into earlier decisions. Promote changes using fresh paired seeds/opponent teams and stratified outcomes.

## Read-only assessment of the draft planner

Inspected `artifacts/strategic-improvement-20261008/worktree/ml/strategy.py` on 2026-10-08. This is a moving draft, so recheck before applying findings.

The joint action and initiative simulation is a useful first step: it includes action denial, priority, observed speed effects, weather, terrain, screens, survival items, switching and limited opponent move hypotheses. Its explicit uncertainty boundary should remain.

At the time of that initial review, the draft lacked opposing switch/Mega
weather scenarios, residual effects, scarce-answer value and position-dependent
pessimism. Those were subsequently added; see [the implemented planning
profile](strategic-planning.md). Several hypotheses still exceed its scope:

- Unrevealed opposing reserves and undisclosed Mega stones remain outside explicit switch/form scenarios.
- Fixed estimated damage can receive a full knockout bonus without branching hit/miss and damage rolls.
- Field-expiry horizons and several recovery/setup/status interactions remain incomplete.
- Support/setup is valued mainly through local constants; immediate defense can dominate the future winning route.

Do not claim this approximate one-turn evaluator implements full competitive thought or has demonstrated superiority until controlled complete-game comparisons pass. Address the highest-frequency recorded failure first; keep deeper planning, team changes and learned behavioral models as separately attributable experiments.

## Limits

This is caption-grounded strategic research, not frame-by-frame game reconstruction. Captions include transcription errors; timestamps identify passages for review. Tournament adaptation across games does not establish that a ladder opponent will repeat behavior. No runtime code, team, checkpoint, live account or service was changed by this research agent.
