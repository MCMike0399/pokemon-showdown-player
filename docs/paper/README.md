# Technical paper

[Read the compiled paper](main.pdf) · [LaTeX source](main.tex) · [Bibliography](references.bib)

*Learning, Search and Evidence in a Pokémon Showdown VGC Agent*, third edition,
31 October 2026, covers the model and software development through campaigns
C1–C14. It preserves the earlier actor–critic/PPO mathematics as dated history
and explains the current simulator search, public-replay models, experiments,
reliability work, and measured team breakthrough.

C13's Mev team passed its live comparison after 897 games: 249/448 wins versus
210/449 for Recife, estimated +61.9 Elo, sequential LLR 3.054 above 2.890.
The subsequent 455-game champion cohort has performance rating 1344 ± 33
(descriptive approximate 95% interval). The paper distinguishes the randomized
contrast from this observational follow-up, records unsuccessful campaigns
and C5's post-results promotion amendment, and leaves C14 unresolved.

The implementation snapshot is public `bee08dc`; the official simulator pin is
`c046106`. The new read-only audit is frozen at **2026-10-31 16:28:15 UTC**:
7,734 completed search games, deduplicated by room. No live game, training,
deployment, promotion, or experiment change was performed for the update.

## Build

From this directory:

```bash
tectonic --keep-logs main.tex
# Or, with a conventional TeX installation:
latexmk -pdf main.tex
```

The included `sec-*.tex` files hold the article; their filenames retain the
previous edition's numbering. TikZ and pgfplots figures need no external images.
Tectonic may download its TeX bundle on the first build. Auxiliary outputs are
ignored; the compiled PDF is tracked.

## Evidence and verification

[evidence-20261031.json](evidence-20261031.json) contains public aggregates,
stopping receipts and input manifest hashes, without player names, room IDs or
private requests. Verify its numerical claims without private data:

```bash
python3 collect_evidence.py --verify
```

[collect_evidence.py](collect_evidence.py) can also collect a **new** snapshot
with `--collect --output /tmp/evidence.json` when private campaign ledgers are
available. Builds never refresh the committed cutoff. Public aggregates cannot
replace access to the original logs for a room-by-room integrity audit.

[Document validation](validation.md) records build and inspection results.
[Earlier primary-source notes](../paper-references.md) support the retained
mathematics; new primary papers are cited directly in the bibliography.
