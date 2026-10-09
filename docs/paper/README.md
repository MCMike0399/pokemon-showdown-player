# Technical paper

[Read the compiled paper](main.pdf) · [LaTeX source](main.tex) · [Bibliography](references.bib)

*Learning, Search and Evidence in a Pokémon Showdown VGC Agent* (second edition,
9 October 2026) explains the deployed system and the mathematics behind each
claim. It covers:

- the system map and project timeline;
- the battle as a partially observed stochastic game, and the joint-action and
  team-preview constraints;
- the hashed features, the deployed `strategic-v1` scenario prior and its
  squash;
- the exact actor–critic, with its temperature;
- the record kept for every match;
- GAE, PPO, the collecting-likelihood check and the KL stop;
- the CPU/MPS three-lane pipeline and its resource rules;
- the paired sign-test promotion gate and multiplicity;
- the determinized simultaneous-move search;
- the browser transport and its rqid guards;
- scouting and team selection;
- an evidence chapter with all recorded evaluations, the live rating under
  search and an Elo analysis.

The paper has 31 propositions, theorems and lemmas with proofs, 8 worked
examples and 13 diagrams and charts. Appendices list corrections to the first
edition and implementation defects found during the review.

The paper describes public source snapshot `0c00ec2` and simulator pin
`c046106`. It ran no new games, training or promotions. Historical results come
from the repository's validation documents. Live search statistics come from
read-only recomputation of the local run ledger on 9 October 2026.

## Build

From this directory:

```bash
tectonic main.tex
# Or, with a conventional TeX installation:
latexmk -pdf main.tex
```

`main.tex` includes the `sec-*.tex` section files. The figures are TikZ and
pgfplots, with no external images. Tectonic may download its TeX bundle on the
first build. Auxiliary build files are ignored.

[Primary-source notes](../paper-references.md) record how references were
verified and the qualifications on the mathematics.
[Document validation](validation.md) records the build, the numerical checks
and the review.
