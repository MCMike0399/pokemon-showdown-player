# Validation performed 2026-10-07

Implementation verification used Python 3.14.6, Torch 2.14.1, Node 26.4.0 and
the pinned official Showdown commit in `package.json` on an Apple-silicon Mac.

| Check | Result |
| --- | --- |
| Offline pytest suite | 23 passed |
| Original team CRUD smoke test | Passed |
| Python compilation and `git diff --check` | Passed |
| Actual MCP stdio initialization/listing | 35 tools registered |
| MCP `ps_ml_status` and `ps_team_recommend` calls | Successful through the real stdio client |
| Champion M-C team validation and packing | Fixture valid; packing matches official simulator |
| Real local Champions games, both player sides | Completed; tested game runs contained no rejected choices |
| PPO updates, consumed-rollout protection, checkpoint reload | Passed |
| Imitation update | Demonstrated labels gained probability; separately consumed from PPO |
| Live submissions/retries/results | Tested with a simulated client, including duplicate and rejected choices |
| Web discovery and exact-format September usage fetch | Verified against live sources |
| Paste parsing/validation | Verified parser and attribution; obsolete SV paste correctly rejected |

The account was not logged in and public-ladder play was not exercised during
this implementation task. The ML tools reuse the existing configured connection.

## Small learning experiment

All games used the **self-built** `examples/champions-rain.json` team for both
sides and a frozen tactical heuristic opponent. These are fixture mirrors, not
held-out team compositions or human opponents. The actor-critic has 97,826
parameters and ran on CPU.

An initial smoke run collected ten sampled PPO games (4–6), then three heuristic
demonstration games supplied one behavior-cloning update. A further **100 PPO
training games**, seeds 200–299 and alternating learner sides, finished **35–65**
with zero rejected actions and no truncations. Sampled training outcomes include
exploration; the evaluation policy selects the highest logit deterministically.

The initial tactical-prior policy and final learned policy were evaluated against
the same scripted opponent on **20 games each**, seeds 2000–2019, alternating
player sides:

| Policy | Wins/losses | Win rate | 95% Wilson interval | Rejected actions |
| --- | --- | --- | --- | --- |
| Initial model (zero updates, tactical prior) | 10–10 | 50% | 29.9–70.1% | 0 |
| Final learned model | 16–4 | 80% | 58.4–91.9% | 0 |

This is a useful functional demonstration of learning on one matchup. The sample
is small, the intervals overlap, and this does not establish broad VGC strength
or improvement against professionals. Earlier weak checkpoints also lost their
evaluation games; updates are not monotonically beneficial.

Local artifacts (gitignored):

- Baseline report: `data/ml-baseline/evaluate-gen9championsvgc2026regmc-2e064ed036d2d5ae11ae.json`
- 100-game training report: `data/ml/local-gen9championsvgc2026regmc-0946a6471ad573ff4b3f.json`
- Final evaluation report: `data/ml/evaluate-gen9championsvgc2026regmc-b220e3f0ccba42ffdfc1.json`
- Learned checkpoint: `data/ml/models/gen9championsvgc2026regmc.pt`
- Experience: `data/ml/experience.sqlite3`

Broader validation should use multiple tournament-valid teams, distinct opponent
policies, frozen checkpoint comparisons and enough games to reduce uncertainty.
