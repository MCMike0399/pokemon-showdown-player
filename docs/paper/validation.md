# Paper validation

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
