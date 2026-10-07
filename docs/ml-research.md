# A learning brain for the Showdown player

Research checked on 2026-10-07. This document separates published evidence from engineering recommendations; it does not claim that a newly initialized model is already a strong VGC player.

## Recommendation

Use a compact **action-conditioned actor-critic** behind the existing `decide(ctx)` interface. The harness owns the connection, team CRUD, matchmaking, legal actions and submitting decisions; OpenClaw orchestrates those operations. The model scores complete legal choices, including both active Pokémon's actions, and learns from recorded outcomes. Start with imitation learning and an interpretable heuristic prior, then train against a local opponent population with masked PPO. This is an engineering recommendation for this repository and the available 16 GB Mac, not an experimentally established best architecture.

The closest direct research match is **VGC-Bench**. Its AAMAS 2026 paper combines human behavior cloning with PPO, self-play, fictitious play and double oracle. It handles doubles joint choices and invalid-action masks, including dependencies between the two actions. Its transformer encodes all twelve Pokémon and optionally a causal history. Reported wins against a professional occur in a restricted single-team mirror setting; generalizing across team compositions remains difficult, and the paper explicitly leaves team building open. These results support the training approach, not a claim of broad or superhuman VGC strength. [VGC-Bench paper](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/angliss2026vgc.pdf)

## Options considered

| Approach | Evidence and fit | Decision |
| --- | --- | --- |
| Compact masked actor-critic, imitation then PPO | Uses the same algorithm family as the direct VGC benchmark; a small network can stay behind the current harness seam. | Recommended first implementation; verify actual learning with held-out evaluation. |
| VGC-Bench transformer and training stack | Direct doubles/VGC BC, population training, pretrained models and replay tools. Requires its pinned Showdown and poke-env versions. | Best external reference and future adapter; validate Champions compatibility before importing checkpoints. |
| Metamon offline RL transformer | Research-grade offline RL, human/self-play data and pretrained policies; documented rulesets focus on singles OU, including Gen 9 OU. | Useful research precedent; its singles policies are not VGC policies. |
| PokéChamp minimax LLM | Official repository includes VGC code, but the published headline evaluation is Gen 9 OU. Adds inference/search complexity and does not by itself implement the requested persistent local RL learner. | Optional future teacher or opponent; retain ML as the runtime decision maker. |
| DQN or MuZero from scratch | DQN needs a stable action representation; learned world models add another major training and validation problem. | Defer until a masked policy and reproducible benchmark exist. |

VGC-Bench's repository provides BC initialization, frame stacking, matchup-specific training, opponent pools and live play. It explicitly warns that its pinned Showdown server must be used. Consequently, replacing this project's working transport wholesale would create avoidable compatibility risk. [VGC-Bench source and setup](https://github.com/cameronangliss/vgc-bench)

Metamon supplies extensive datasets and pretrained agents, but its own supported-rulesets discussion and strength claims concern singles. Transfer to doubles and Champions would require different actions, observations and evaluation. [Metamon repository](https://github.com/sooham/metamon)

PokéChamp's maintained repository includes `llm_vgc_player.py` and VGC runners; that code availability should be distinguished from the paper's OU performance claims. [PokéChamp repository](https://github.com/sethkarten/pokechamp), [PokéChamp paper](https://arxiv.org/abs/2503.04094)

## Target format and mechanical correctness

The existing target `gen9championsvgc2026regmc` is **Pokémon Champions**, not Scarlet/Violet Regulation I. Current upstream defines the format with `mod: champions`, doubles, Flat Rules, VGC Timer and Open Team Sheets. Its Bo3 variant forces team sheets. Champions Flat Rules use level 50, species/item restrictions and automatic picked-team size. Retain the exact format in data, model metadata and evaluation. [Showdown format definitions](https://github.com/smogon/pokemon-showdown/blob/master/config/formats.ts), [Champions rules](https://github.com/smogon/pokemon-showdown/blob/master/data/mods/champions/rulesets.ts)

The official M-C notice runs from September 9 to December 2, 2026. It specifies one Mega Evolution per battle, no duplicate held items, 45-second decisions and a 90-second preview. Do not assume Terastallization because the identifier starts with `gen9`. [Official Regulation M-C announcement](https://champions-news.pokemon-home.com/en/page/816.html)

Upstream Champions explicitly disables Terastallization and uses different stat and PP calculations. Team spreads, damage estimates and legality must therefore be checked against the format's simulator rather than ordinary Gen 9 formulas. [Champions simulator mechanics](https://github.com/smogon/pokemon-showdown/blob/master/data/mods/champions/scripts.ts)

Current poke-env contains a genuine `DoublesEnv`: joint actions include targets, switches, Mega Evolution and other generation-specific mechanics. This supports a later local training adapter; actual compatibility with this repository's current Champions rules should still be exercised in integration tests. [poke-env doubles environment source](https://github.com/hsahovic/poke-env/blob/master/src/poke_env/environment/doubles_env.py)

## Model and learning contract

Recommended first implementation:

1. Build observable features from our request and the public battle log: both active Pokémon, available party, revealed opposing party, moves, HP/status/boosts, field conditions, recent actions and game phase. Mark unknown opponent information explicitly; do not fill it with unrevealed simulator internals.
2. Encode every **complete legal action** alongside that state. Include move identity, target identity, switching destination, gimmick, and interactions such as double targeting, spread attacks, ally support and competing switches. A shared scorer produces one logit per candidate; softmax over legal candidates gives the policy. A separate value head predicts outcome.
3. Select deterministically during evaluation and sample during data collection. Record the policy version, exact candidate list, chosen action, log probability, estimated value, request identity, team identity, format and terminal result. Separate recommendations from choices the server accepted.
4. Pretrain from reviewed state/action demonstrations. Bootstrap heuristics may supply a prior or weak demonstrations, but imitation of those demonstrations alone is not evidence of learning human-level play.
5. Learn from completed trajectories. A small Monte Carlo policy-gradient actor-critic is a reasonable first working learner; label it accurately. PPO needs rollout log probabilities from the collecting policy and a consistent legal-action distribution. Do not call arbitrary historical replay fine-tuning “PPO.”

PPO performs multiple optimization passes on collected policy data with a constrained surrogate objective. It is a practical later trainer, but is not inherently a guarantee of improvement. [Original PPO paper](https://arxiv.org/abs/1707.06347)

Masking invalid actions has a policy-gradient justification and becomes especially valuable when much of the action space is illegal. Generate complete joint choices before scoring so that double switching into the same slot or using a once-per-battle mechanic twice cannot receive policy probability. [Invalid-action masking paper](https://arxiv.org/abs/2006.14171)

Use persistent revealed-state history first, then add a GRU or short causal history encoder when data supports it. Recurrent networks have evidence for handling partial observations, but that evidence is not a Pokémon-specific guarantee. Off-the-shelf SB3 MaskablePPO documents no recurrent-policy support, so combining masking and memory requires a custom policy/trainer or an explicit frame-stack approach. [Recurrent partial-observation research](https://arxiv.org/abs/1507.06527), [SB3 MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)

For local resources, stream trajectories, use bounded minibatches, keep checkpoints small and begin with CPU execution. Measure MPS rather than assuming it improves a tiny action-scoring network. Simulator throughput and data quality will likely matter more than scaling model parameters initially; that is an engineering hypothesis to benchmark.

## Human demonstrations and experience

The current VGC-Bench dataset card is particularly useful: it contains **88,905 Champions M-A/M-B OTS logs**, approximately 177,810 player trajectories and 1.47 million transitions, with last listed logs on June 20, 2026. The roughly 630 MB corpus is a practical initial download. Older Scarlet/Violet A–J logs moved to a separate archive. These are transfer demonstrations, not current M-C expertise. The card links its perspective reconstruction and BC pipeline. [Dataset owner's current card](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs)

Raw replays are public transitions, not exact player requests. They omit some unchosen options and hidden information. Use a verified perspective reconstruction adapter; do not silently synthesize exact legal requests or train with future revealed information. Collecting this harness's own accepted requests/actions provides cleaner experience for online updates. VGC-Bench's replay pipeline and its limitations are a useful reference. [VGC-Bench replay pipeline](https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/logs2trajs.py)

For scalable improvement, run self-play on a local simulator, with no public credentials and a mixture of scripted opponents and past model snapshots. Keep the configured public account for normal ladder play. Begin with several curated teams and broaden the distribution; otherwise the policy may memorize one mirror matchup. Store aborted games separately from losses. Freeze a deployed checkpoint during each rollout batch.

Before claiming a better model, compare candidate and deployed checkpoints on the same held-out teams and opponent pool, with alternating sides and multiple random seeds. Report win rate, sample count and uncertainty. Check illegal/rejected choices and timeout rates. A single win or declining training loss establishes neither strength nor generalization.

## Web research and team improvement

Research should produce **dated, attributed structured evidence**: source URL, fetch date, event date, exact regulation, player, placement, six-Pokémon composition, accessible paste, and whether stats are fully disclosed. Preserve the source and original team; distinguish real sets from inferred spreads. External prose supplies hypotheses and priors. Verified teams feed candidate evaluation; actual state/action demonstrations feed imitation. Reading an article alone is not a neural-weight update.

| Source | Useful input | Qualification |
| --- | --- | --- |
| Official Pokémon / Champions announcements | Regulation dates, permitted mechanics and eligible species | Rules authority; refresh on regulation changes. |
| Limitless VGC | Tournament results, teams, player profiles and format filters | Operator-owned database; event pages expose regulation so older SV teams can be filtered out. |
| Victory Road / VR Circuit | Credited replica teams, team reports and their own tournament results | Track Bo1 closed-sheet versus Bo3 open-sheet context. |
| Smogon monthly Showdown statistics | Usage, moves, items, abilities, spreads and teammates where provided | Ladder distribution differs from tournament results; use format and rating cutoff. |
| LabMaus | Potential tournament-data source | Site could not be inspected with the research browser in this run; no verified API contract. |

Limitless currently identifies M-C on its homepage and lists October 3 Recife and September Frankfurt, Brisbane and Baltimore events. Its event team pages expose moves/items/abilities and regulation metadata. [Limitless VGC](https://limitlessvgc.com/), [Limitless tournaments](https://limitlessvgc.com/tournaments), [Example event team page](https://limitlessvgc.com/tournaments/428/teams)

Victory Road's replica archive credits creators and distinguishes tournament finalists from closed-sheet Bo1 ladder teams. Its current circuit records M-C tournaments and winners. These are useful curated seed candidates, subject to complete set availability and simulator validation. [Credited Champions replica teams](https://victoryroad.pro/champions-replica/), [VR Circuit](https://circuit.victoryroad.pro/)

September 2026 Showdown statistics include the exact M-C format and Bo3 variant at multiple rating cutoffs. The `chaos` JSON is machine-readable but large; use bounded caching and fetch only the desired monthly format. [September statistics index](https://www.smogon.com/stats/2026-09/), [M-C 1630 usage](https://www.smogon.com/stats/2026-09/gen9championsvgc2026regmc-1630.txt), [M-C detailed JSON](https://www.smogon.com/stats/2026-09/chaos/gen9championsvgc2026regmc-1630.json)

Recommended team-building progression: rank existing legal curated teams using actual matchup outcomes; expose weak matchups and uncertainty; propose constrained replacements or spread/move changes; validate every candidate with Showdown; then evaluate against a weighted current-metagame pool. A team-ranking posterior or bandit can learn which existing teams this policy pilots well. That is a useful first module, but it should not be described as a model that has solved novel team generation.

## What remains to prove

The research supports the architecture and training direction. Competitive M-C ability requires a dataset adapter, sufficient real training, simulator-valid teams, robust observable-state reconstruction and controlled evaluation. Existing published singles strength does not supply those proofs, and a working learning loop is not itself evidence that every update improves play.
