# Continuous VGC learning: sources, integration and resource limits

Research verified on 2026-10-07. Recommendations below distinguish source evidence from engineering choices. A working daily training cycle does not establish competitive strength or guarantee that each update improves play.

## Recommended data flow

OpenClaw should orchestrate the persistent MCP harness. The harness supplies legal battle choices, submits the model's choice, and stores accepted state/action/outcome records. A separate background worker prepares candidate checkpoints from new experience and local simulation, evaluates them against a frozen incumbent, and promotes only candidates that meet an explicit evaluation gate. Keep live inference on an immutable checkpoint for an entire battle.

Use three distinct feeds: exact-format experience for policy training, sourced teams for opponent diversity and team selection, and metagame statistics for priors. Source prose, a tournament placing, and a usage percentage are not action labels. Preserve format, team version, checkpoint, collection mode and provenance so transferred historical data cannot masquerade as current M-C ladder evidence.

## Current format and sources

The target remains `gen9championsvgc2026regmc`, with `gen9championsvgc2026regmcbo3` tracked separately. The official M-C notice specifies its September 9–December 2, 2026 period and battle constraints. A daily job can check for rule changes, but a new regulation must trigger revalidation instead of silently mixing its records into the current policy. [Official Regulation M-C notice](https://champions-news.pokemon-home.com/en/page/816.html)

| Feed | Input and proposed cadence | Training use and qualification |
| --- | --- | --- |
| Harness's own completed battles | Accepted requests/actions and outcomes, queued immediately | Most reliable exact-request experience. Separate deterministic inference from sampled rollouts; only sampled policy records with matching probabilities qualify for on-policy PPO. |
| Official Showdown public replays | Exact-format discovery daily, small download cap and cached IDs | Public actions and revealed states for future imitation conversion; metadata and disclosed teams can be useful immediately. |
| Limitless VGC event pages | Discover new events daily, download matching events once | Credited compositions, sets, placement and nature. Missing stat allocations remain unknown. |
| Smogon `chaos` JSON | Check latest completed month daily; cache by month/format/cutoff | Weighted move/item/ability/spread/teammate priors; marginal statistics do not reveal complete coherent sets. |
| VGC-Bench Champions archive | Optional bounded bootstrap, pinned revision | Older M-A/M-B transfer data; raw logs require reconstruction and exact-format filtering. |

### Current tournament teams

Limitless's current homepage identifies M-C and links the October 3 Recife regional and September Frankfurt, Brisbane and Baltimore regionals. Recife's teams page identifies the date and regulation, then publishes credited placements, species, items, abilities, natures and moves. Stat point allocations are absent in the inspected sets. Mark those teams as partial disclosures; any filled spread is an explicit synthetic hypothesis. [Limitless current events](https://limitlessvgc.com/), [Recife M-C teams](https://limitlessvgc.com/tournaments/444/teams)

Do not assume every event ID linked from an old document belongs to the current format: `/tournaments/428/teams` identifies April 2026 Prague, Scarlet/Violet Regulation I. Discover fresh event links and verify their own regulation metadata before accepting them. [Prague archive](https://limitlessvgc.com/tournaments/428/teams)

Engineering recommendation: preserve the original attributed set, validate its moves/species/items against the pinned simulator, and generate clearly labeled candidate spreads only when needed for simulation. Evaluate synthetic candidates independently; do not present a completed approximation as the player's original team. Source caches belong in local data storage, rather than a public repository containing republished archives.

### Machine-readable metagame statistics

September 2026 contains the exact M-C format in the official stats directory. Its detailed `chaos` endpoint is JSON and keyed by Pokémon, with distributions including moves, items, abilities, spreads and teammates. Rating cutoffs change the sampled population; retain the cutoff in provenance and avoid treating ladder popularity as tournament win probability. [September index](https://www.smogon.com/stats/2026-09/), [Exact M-C 1630 JSON](https://www.smogon.com/stats/2026-09/chaos/gen9championsvgc2026regmc-1630.json)

Engineering recommendation: select a completed month, fetch a bounded exact-format file, cache its hash, and refresh only when its source version changes. If a file is unavailable, retain the last successful snapshot and report its age. A daily schedule can check monthly data without repeatedly retraining on an unchanged download.

### Human replay archive and normalization

The dataset owner's current card labels `cameronangliss/vgc-battle-logs` MIT and lists 88,905 Champions M-A/M-B open-team-sheet logs, about 1.47 million reconstructed transitions and a roughly 630 MB total. No M-C file is listed. Each JSON maps a battle ID to `[timestamp, public_battle_log]`; these are not ready-made legal-request decisions. Older Scarlet/Violet logs are in a separate archive. [Dataset card](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs), [SV archive card](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs-sv)

The current file listing offers a useful small bootstrap: `logs_gen9championsvgc2026regmb.json` is 2.33 MB, whereas the M-A Bo3 file is 550 MB. Metadata checked via the publisher's API identifies revision `74f260510b1dbcaf4c42c8bf653f726b51d476ef`. Pin the revision rather than trusting changing `main`; preserve source attribution and the publisher's license metadata. [Publisher file listing](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs/tree/74f260510b1dbcaf4c42c8bf653f726b51d476ef)

The small M-B file was downloaded with a 4 MiB cap and inspected: 2,334,303 bytes, 324 dictionary entries, each mapping an ID to a two-element list of integer timestamp and protocol-log string. The inspected first record has an explicit M-B tier. Its SHA-256 is `7711711fc19ec36eb7f9c46b4f18e3621e33ef8e3ec2757983bf4589fbae87dc`; the pinned dataset README declares `license: mit`. Verify the digest before accepting a bootstrap, and require explicit separate transfer mode for M-B. [Pinned download](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs/resolve/74f260510b1dbcaf4c42c8bf653f726b51d476ef/logs_gen9championsvgc2026regmb.json), [Pinned publisher license metadata](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs/blob/74f260510b1dbcaf4c42c8bf653f726b51d476ef/README.md)

VGC-Bench's converter reconstructs both player perspectives through poke-env, obtains joint action labels from public moves/switches/Mega events, and writes trajectories using its own observation embedding and action encoding. It filters by rating and optionally winner. It records known leads, adds backline preview labels only when both backline Pokémon are revealed, and documents known parsing failures. Its `fake=True` order conversion is a reconstruction mechanism, not proof of an original legal-action request. Porting its outputs therefore requires a tested observation/action adapter for this project's model. [Owner's replay converter](https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/logs2trajs.py)

Engineering recommendation: begin with raw replay caching and disclosed-team extraction, while treating human policy learning as a separate normalization milestone. A conservative future adapter must reject ambiguous targets, actions prevented before execution, forced movement mistaken for a choice, and unrevealed selected Pokémon. It must reconstruct each decision from information available at that point, preserve missing information, and report how many records were rejected. Public event logs never provide the collecting policy's probability distribution, so human demonstrations belong in imitation learning rather than on-policy PPO.

### Public replay API and availability

Showdown documents `.json` replay downloads and `search.json?format=<format>&before=<timestamp>` discovery. Use the last replay's `uploadtime` as the pagination cursor: search returns up to 51 results and a 51st result signals another page. The older `page` parameter is explicitly discouraged. VGC-Bench's scraper additionally checks for open team sheets and distinguishable team members. [Official web API contract](https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md), [Reference scraper](https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/scrape_logs.py)

A direct GET to the exact M-C `search.json` endpoint initially returned HTTP 403 during research; the research browser could not access it either. The subsequent implementation validation successfully retrieved 23 exact-format completed public games and 407 pre-turn executed-move labels. Availability can vary: the worker reports individual errors, retains prior cached data and continues with other feeds. Actual ingestion counts, rather than the existence of an endpoint, establish whether data was fetched. See [round-two validation](round2-validation.md).

### Additional GitHub and Hugging Face resources

| Publisher resource | Verified availability and licensing | Selection |
| --- | --- | --- |
| [VGC-Bench code](https://github.com/cameronangliss/vgc-bench) and [checkpoints](https://huggingface.co/cameronangliss/vgc-bench-models) | MIT code; model publisher API also declares MIT. Its own encoder/actions and dependency pins govern compatibility. | Best direct VGC reference; adapt deliberately instead of loading its weights into an unrelated architecture. |
| [Hanami Reg M-A dataset](https://github.com/HanamiGG/hanami-vgc-dataset) | Owner declares CC BY 4.0. Its release specifies 185,566 parsed replay-side records and 307 tournaments. Public v1.0.0 release asset is 52,118,180 bytes. | Useful optional historical team/meta corpus; exact M-A remains transfer evidence, not M-C. |
| [Metamon code](https://github.com/sooham/metamon) and [parsed replay dataset](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays) | MIT code and CC BY-NC 4.0 parsed dataset; these are distinct licenses. Current owner docs describe 2.7 million raw battles and 5.3 million reconstructed trajectories. | Research reference for reconstruction/versioning. Available policy formats focus on singles Gen1–4 and Gen9 OU; omit bulk download from the default VGC feed. |

Hanami's owner lists `replays_parsed` per-side records with teams, leads, brought Pokémon, results, items and moves; `replays_meta` has format/Elo/anonymized players. Tournament tables and aggregate summaries accompany them, in Parquet and gzipped CSV. The release's internal full schema was not downloaded or executed here. A future importer should inspect that schema and reject unavailable per-turn fields instead of inventing them. [Hanami owner README](https://github.com/HanamiGG/hanami-vgc-dataset), [Published release](https://github.com/HanamiGG/hanami-vgc-dataset/releases/tag/v1.0.0)

The GitHub release API identifies v1.0.0 as latest, published June 19, 2026, with tag commit `caf9c4d49cf81cc376946ad5ebaf0f87b698fccc`. Its asset metadata supplies SHA-256 `37617f43b76ad2428047c786105acb63edd0c5f843b3c11bdfe02079553128e9`. Both the tagged README and release declare CC BY 4.0; the repository contains only a README, without an independent license file or integrity manifest. The asset was not downloaded here, so that digest is publisher metadata rather than an independently computed value. [Release API](https://api.github.com/repos/HanamiGG/hanami-vgc-dataset/releases/tags/v1.0.0), [Tagged license declaration](https://github.com/HanamiGG/hanami-vgc-dataset/blob/caf9c4d49cf81cc376946ad5ebaf0f87b698fccc/README.md), [Pinned release asset](https://github.com/HanamiGG/hanami-vgc-dataset/releases/download/v1.0.0/hanami-vgc-regma-1.0.0.zip)

Metamon's parsed schema includes chosen, legal and missing-action fields plus reconstructed universal states; its publisher explains partially revealed teams and imputation. Those fields are specific to its pipeline, rather than original private Showdown requests. No ready-made exact M-C action-conditioned dataset was verified in this search. [Metamon dataset schema](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)

### A practical replay-trained opponent scout

Engineering recommendation: public logs can train an auxiliary model immediately without claiming exact action reconstruction. Predict an opponent's next **executed public move**, using its species, explicit public team sheet when present, and information revealed before that turn. A small conditional frequency model or classifier can learn Protect frequency, move type/role distributions or switching tendencies. Use its uncertainty-weighted predictions to inform the battle scorer and matchup/team evaluation.

The prediction target must be labeled accurately: an executed public action is not necessarily the choice a player submitted, because action prevention and triggered/called moves exist. Exclude moves marked as originating from another effect, forced movement and ambiguous actor identities. Do not infer a move label for a fainted or immobilized actor. Update features at turn boundaries, so earlier same-turn actions do not leak into a prediction that represents the simultaneous decision phase. Never backfill earlier features from the final revealed team or later HP/item/move information. Keep raw text and chat outside learned features. The official protocol distinguishes move, switch, drag and effect events, providing the basis for conservative extraction. [Showdown battle protocol](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md)

Evaluate the scout on battle-disjoint held-out logs and report prediction accuracy/log loss and coverage. Preserve old-regulation corpora as separate priors; prefer current exact-format observations when available. Scout updates can incorporate new public records daily while the actor-critic continues to learn from its own accepted exact-request decisions.

## OpenClaw integration and process lifetime

Current official documentation distinguishes OpenClaw's native `mcp.servers` registry from the mcporter registry in `config/mcporter.json`. `openclaw mcp serve` exposes OpenClaw itself and is the opposite direction from registering this Showdown server. The native CLI supports adding, configuring and probing downstream servers. Installed OpenClaw 2026.9.8 was checked read-only and exposes these commands; mcporter was absent. [Official MCP overview](https://docs.openclaw.ai/mcp)

A `probe` proves connection and capability discovery, not a successful agent decision. Official lifecycle documentation says session-scoped MCP runtimes survive between turns, while detached one-shot runs retire connections at run end. Reset, Stop, compaction rollover, relevant configuration changes and Gateway shutdown also retire them. Idle eviction is opt-in. Use one retained agent session or a single battle-owning tool call so login and later choices reach the same server process. `openclaw mcp reload` affects the calling process, not an independently running Gateway. [Registry and lifecycle contract](https://docs.openclaw.ai/cli/mcp/registry)

Plugin bundles can also contribute stdio/HTTP MCP servers, and embedded OpenClaw exposes their tools using `serverName__toolName`. The coding and messaging profiles include bundle MCP tools; profile/filter configuration can hide them. A bundle's working directory and runtime dependencies must be valid before the agent can call the harness. [Official bundle mapping](https://docs.openclaw.ai/plugins/bundles)

Engineering recommendation: verify an actual isolated OpenClaw agent turn invokes a model status tool, selects or ranks a team, obtains a legal model decision and completes an offline battle through MCP. Record the agent's tool-use evidence and persisted experience. Report public-ladder behavior separately; a CLI `probe` or a Python unit test is insufficient to claim live OpenClaw gameplay.

## Apple silicon and shared-server resources

Apple documents PyTorch's MPS backend as Metal GPU acceleration, with availability checked through `torch.backends.mps.is_available()`. PyTorch's CPU intraop thread count is configurable and should be set before eager/JIT/autograd work begins. These mechanisms enable benchmarking CPU and MPS; neither establishes which is faster for this small policy. [Apple MPS setup](https://developer.apple.com/metal/pytorch/), [PyTorch CPU threads](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html)

PyTorch also provides an MPS per-process allocator fraction relative to the device's recommended working set. Its allocator documentation warns that disabling the high watermark may cause system failure under system-wide out-of-memory conditions. These GPU allocator limits do not cap Python, simulator processes or total host RAM. [MPS per-process allocation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.set_per_process_memory_fraction.html), [MPS allocator controls](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)

Engineering recommendation: leave CPU and memory headroom for other services, use low-priority worker processes, cap simulator concurrency and training threads independently, and pause new work when free memory or normalized host load crosses configured thresholds. Make limits environment-configurable. Increasing worker count should follow a measured throughput benchmark; oversubscribing simulator workers and BLAS/PyTorch threads can reduce throughput. Benchmark end-to-end games per minute and inference latency, not just a tensor operation. Keep a maximum batch duration and cleanly reap only owned simulator children when interrupted.

For mixed CPU/GPU execution, keep simulator/event parsing and small live inference on CPU, and benchmark batched candidate optimization on MPS. Warm up both paths, use identical batches and bracket GPU timing with `torch.mps.synchronize()`; that call waits for all MPS kernels to complete. Measure CPU work and data-transfer overhead as part of the result. GPU availability alone is not evidence of a faster batch. [PyTorch MPS synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)

Engineering recommendation: expose an explicit MPS training option plus a documented CPU fallback for unsupported operations. Choose a nonzero bounded allocator fraction and independently monitor host memory because Apple silicon CPU and GPU share memory capacity. Neither a positive process nice value nor a GPU allocator fraction provides a complete service isolation guarantee. Start with conservative defaults, raise worker/batch limits only after observing host service latency, and avoid changing unrelated process priorities or stopping services.

## Daily learning acceptance criteria

The following are proposed engineering criteria, not published guarantees:

1. Each completed battle contributes one terminal trajectory; retries do not duplicate it. Aborted games remain separate from losses.
2. The deployed checkpoint does not change mid-battle. Candidate training is isolated from live inference and guarded against concurrent writers.
3. Daily work first checks resource headroom and source versions, then collects bounded experience against varied legal teams and past opponents.
4. Training records the exact experience/checkpoint versions and distinguishes imitation updates from sampled-policy PPO.
5. Candidate evaluation uses fixed held-out matchups, seeds and side alternation. Promotion records game count, incumbent/candidate score, uncertainty and rejection reasons.
6. Losses and rejected candidates remain useful experience; failure to pass a promotion gate does not overwrite the deployed policy.
7. Team selection learns from the same format/versioned outcomes, with uncertainty for untested teams. Tournament priors and synthetic spreads remain attributed separately.
8. Status reports last successful feed, last completed battle, last training cycle, checkpoint version, evaluation result and any paused/error state.

More experience is a mechanism for improvement; evaluation determines whether a particular candidate is better in the measured matchups.
