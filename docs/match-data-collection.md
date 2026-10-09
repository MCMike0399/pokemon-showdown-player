# Match data and set provenance

Episodes retain the exact submitted own sets, private Showdown requests,
choice-time observations, public-event prefixes, legal masks, selected choices,
sampling probabilities, encoded policy inputs and terminal outcomes. Player
credentials and chat are excluded. Original feature arrays remain immutable;
retrospective diagnostics do not become fresh PPO or expert labels.

| Information | Our Pokémon | Live opponent |
| --- | --- | --- |
| Nature and `evs` allocation | Submitted team, when available | Only if explicitly disclosed |
| IV allocation | Retain explicit declaration; missing remains unknown | Only if explicitly disclosed |
| Calculated stats and maximum HP | Private request, including observed form changes | Usually unknown; public HP ratios are retained |
| Species, forms, moves, items, abilities | Submitted set plus current requests/events | Preview, actual events and accepted open team sheets |
| Status, boosts, weather, terrain, screens, PP | Requests and chronological events | Public events; hidden PP or counters are not invented |

For the pinned Champions M-C format, `evs` means **stat points**, not traditional
EV training. Its stat calculation applies nature and stat points without using
the declared IV values. Accepted Champions open team sheets include nature but
omit stat-point and IV allocations. See the authoritative
[Champions stat implementation](https://github.com/smogon/pokemon-showdown/blob/master/data/mods/champions/scripts.ts)
and [team-sheet implementation](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle.ts).

Request stats are base calculated stats. Boosts, paralysis, Tailwind, items,
abilities and field effects can change effective battle stats; the associated
observations and events are stored separately. A missing opponent spread is
unknown. Damage and move-order evidence can constrain estimates, but multiple
spreads and natures can fit the same observations.

Produce a read-only, exact-cutoff audit:

```sh
.venv/bin/python scripts/audit_match_collection.py --root data/ml \
  --campaign data/ml/campaigns/CAMPAIGN --start 533 --cutoff 582 \
  --output artifacts/collection-audit
```

`summary.json` reports completeness; `matches.jsonl` provides each game's
submitted-set facts, observed stat transitions, actual opponent disclosures,
format mechanics and unknown fields. Blank public EV/IV fields stay unknown.
Defaults inside an explicitly transmitted allocation are decoded according to
the packed-team protocol. Legacy missing information is not reconstructed from
today's mutable TeamStore.

For completed local simulations, `audit_episode(..., fixture_plan=...)` can
attach the exact declared synthetic opponent fixture after matching its team
fingerprint. This is labelled `postgame-diagnostics-only`, never merged into
choice-time observations or used to infer a live opponent's private set.

The exact 50-game review at 582 verified submitted nature and allocations in
50/50 games, raw private requests and own stats in 454/454 decisions, opponent
public nature in two games and opponent public EVs in zero. The team did not
declare IV overrides; IV values have no stat effect in this Champions format.
This audit adds readable provenance without changing the playing policy,
existing recordings, model inputs or candidate evaluation source generation.
