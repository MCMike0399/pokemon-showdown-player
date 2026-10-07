# Continuous-learning and OpenClaw verification

Verified on 2026-10-07 on an M5 Mac with 10 CPU cores and 16 GB RAM.

| Check | Evidence |
| --- | --- |
| Offline suite | 31 tests passed, including pre-turn leak prevention, private-log exclusion, job leases, resource deferral, team materialization and public-snapshot rejection |
| Native OpenClaw MCP | `openclaw mcp probe` discovered 40 harness tools; long battle timeout configured |
| OpenClaw team/model flow | Agent invoked `ps_team_plan`, one `ps_login`, one `ps_ml_ladder`, `ps_ml_wait`, `ps_ml_play`, `ps_ml_finish` and `ps_learning_status` through native MCP |
| Actual Champions M-C game | Completed loss; neural brain submitted 7 decisions; terminal outcome recorded once; 18 public-move scout samples recorded |
| Per-battle background learning | Actual game queued a durable job; candidate and incumbent each won 4/8 paired evaluations; candidate was correctly not promoted |
| Local MPS candidate | PPO and scout trained successfully on Metal; weaker candidate (1/8 versus incumbent 2/8) was correctly rejected |
| CPU/MPS benchmark | One synchronized run measured 2.606 ms CPU versus 1.714 ms MPS per training step; auto mode selected MPS. An earlier small-margin run selected CPU |
| Historical feed | Checksum-pinned M-B archive: 2,334,303 bytes, 324 public games, approximately 6,900 usable move labels; stored separately from M-C |
| Historical scout learning | 6,923 samples, 1,641 held-out samples; held-out cross-entropy decreased from 6.840 to 4.253 on MPS |
| Exact M-C daily feeds | 50-species monthly usage summary, 12 tournament team sheets, 23 public replay games and 407 executed-move samples retrieved |
| Current-format scout | 529 samples, 110 held out; held-out cross-entropy decreased from 6.521 to 5.575 on MPS |
| Always-on scheduler | Project launchd job installed; every 15 minutes with bounded practice and once-per-day feeds; CPU/memory pressure causes deferral |
| Successful scheduled promotion | Eight-team pool, identical evaluation seeds/sides: candidate 13/20 wins versus incumbent 10/20; clean evaluation passed the margin and promoted checkpoint update 112 |
| Other services | Existing OpenClaw and application service process IDs remained unchanged during verification |
| Public Linux CI | GitHub Actions installed the pinned simulator and ran all offline checks successfully |

Team sheets include actual player/event attribution but do not disclose stat
spreads. The OpenClaw test used explicitly generated experimental spreads; these
were not attributed to the tournament player. No second Showdown account was
created, and no chat or external messaging was sent.

These checks prove the data/model/harness pipeline and observed learning of the
auxiliary classifier. One live loss and small scripted evaluations do not measure
general tournament strength. The active actor-critic retained its incumbent when
candidates did not improve the finite evaluation suite.

Raw test outputs, account-specific game identifiers, replay logs, weights,
databases and local configuration are excluded from the public repository.
Publication uses a source allowlist and a new Git root rather than private history.
