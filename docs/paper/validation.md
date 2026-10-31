# Paper validation

## Third edition — 31 October 2026

Implementation inspected at public `bee08dc`; simulator pin `c046106`.
Evidence cutoff: **2026-10-31 16:28:15 UTC**. This is a documentation and
read-only evidence audit, not a new playing-strength experiment.

- **Build:** `tectonic --keep-logs main.tex`; 65 pages, 19 included section
  files, 40 bibliography entries, 14 TikZ/pgfplots figures. No undefined
  references/citations or overfull boxes. Two underfull text-box warnings
  remain in the historical recorder diagram; neither clips content.
- **Visual inspection:** rendered all 65 pages into six contact sheets and
  inspected them, with larger renders of the updated architecture, live
  performance figure and C13 results table. No clipping or overlapping chart
  labels were found. Temporary inspection images are not tracked.
- **Formal content:** 32 propositions/theorems/lemmas with proofs and 8 worked
  examples. The new shared-root certificate explicitly applies only to the
  supplied finite matrices. Earlier actor/PPO sections have historical scope
  notes; the defects appendix now distinguishes repairs from remaining limits.
- **Frozen evidence:** `evidence-20261031.json`, collected by
  `collect_evidence.py`, contains aggregate data from 7,734 completed rated
  search games, room-deduplicated internally, without publishing player names,
  room IDs, requests or logs. It includes manifest hashes and saved stopping
  receipts; builds do not refresh it.
- **Independent arithmetic:** `python3 collect_evidence.py --verify` recomputes
  C10/C12 exact sign-test tails, both C13 round scores, Elo transforms,
  normal-approximation LLRs and the post-C13 interval. Mev: 249/448 versus
  210/449 at the 897-game decision, +61.9 Elo, LLR 3.054. The later champion
  cohort joins both sides of a runner restart: 455 games, 226 wins,
  performance 1343.7 ± 33.3. C14 has 9 completed games and no decision.
- **Stopping integrity:** C9 uses each factor's saved first decision rather
  than later terminal arrivals. All four are H0; boost stopped at 1,581 games,
  while the eventual experiment ledger has 1,582. C13 excludes the one late
  result in each round from its 302- and 897-game stopping cohorts. Assignment
  ledgers identify actual team arms even when the run's default team differs.
- **Source checks:** verified referenced repository paths and current C5
  constants, preview models, CRN, root-solver behavior, terminal-only A/B join,
  and the pinned simulator. Primary papers on restricted Nash responses and
  imperfect-information depth limits were checked directly. Sequential-test
  claims qualify the implemented normal approximation and multiplicity.
- **Relevant regression suite:** 58 passed across A/B assignment, Elo monitor,
  gate scheduling and belief solver tests. No production source was changed.
- **Publication checks:** `git diff --check` and the repository publication
  scanner on the article's publication files pass.

No login, battle, training, service restart, model promotion or experiment
mutation was performed for this update. This edition records C5's explicit
post-results amendment and C13's randomized success without relabeling other
campaigns as successful. The public aggregates support arithmetic reproduction;
a full room-level integrity audit still requires the private original logs.

## Second-edition validation receipt (historical)

Second edition, validated on 2026-10-09 against public source `0c00ec2`. This
checks the document's correspondence with the source; it is not a new
battle-strength experiment.

- **Build.** Tectonic compiles `main.tex` with its 16 included section files
  and BibTeX: 49 pages, 37 cited sources, 13 TikZ/pgfplots figures. There are
  no undefined references and no overfull boxes. Every figure page was
  rendered and inspected for overlaps and clipping.
- **Formal content.** 31 propositions, theorems and lemmas, each with a proof,
  and 8 worked examples.
- **Recomputed numbers.** Every worked example and numeric claim was
  recomputed independently:
  - softmax at τ = 1 and τ = 0.25, the strategic squash, GAE, PPO clipping, and
    the Beta posteriors;
  - the tensor-cache bound (482 steps) and the parameter counts (97,826,
    110,210 and 127,946, from instantiated modules);
  - the sign-test threshold table, the Hoeffding radii, the multiplicity bounds
    and the regret-matching gap;
  - the Elo stationary rates, the Wilson intervals and every paired p-value in
    the evidence chapter.
- **Independent review.** A separate review compared the paper with the code
  and the proofs. It found no wrong proposition or example. It did find about a
  dozen small mismatches with the code, all now corrected:
  - profile scope of bench canonicalization and the coloured-HP parser;
  - target weighting and the doubled preview score;
  - world-rebuild contents and matrix-cell invalidation;
  - retry counts and pressure flag values;
  - the screening simulation count;
  - the share scale in the usage lemma;
  - the Elo floor clamp;
  - the status of v5.
- **Live search statistics** were recomputed read-only from the local run
  ledger at 20:07 UTC: battle rooms were de-duplicated and ratings were taken
  from terminal logs. No personal identifiers or paths appear in the paper.
- **Tests and scans.** Public tests: **309 passed, 14 skipped**. The workspace
  publication scanner (`scripts/public_snapshot.py --scan-only`) passes on the
  branch checkout, and `git diff --check` passes.

No live login, battle, training or promotion was performed to write this
paper. Historical results come from the repository's validation documents.
