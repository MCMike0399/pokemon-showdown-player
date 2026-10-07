# Technical paper

[Read the compiled paper](main.pdf) · [LaTeX source](main.tex) · [Bibliography](references.bib)

The paper explains the OpenClaw/MCP player lifecycle, observation and joint-action
contracts, the exact action-conditioned actor–critic, PPO and behavior cloning,
public-move scouting, Bayesian team selection, and candidate promotion. Proofs
state their assumptions explicitly; the statistical improvement certificate is
an analytical proposal, not an implemented promotion test.

It describes repository source at local commit `151393d` and feature schema 1.
Historical experiments are attributed to the checked-in validation documents;
the paper does not claim to run a new playing-strength experiment.

From this directory, use either:

```bash
tectonic main.tex
# Or, with a conventional TeX installation:
latexmk -pdf main.tex
```

Tectonic may download its TeX bundle on the first build. Both commands process
the adjacent BibTeX bibliography. The PDF is self-contained and has no external
image dependencies. Auxiliary build files are ignored.

[Primary-source research notes](../paper-references.md) document reference
verification and mathematical qualifications. The paper's source audit appendix
maps its claims to implementation files.

[Document validation](validation.md) records compilation, numerical checks,
source review and publication-export checks.
