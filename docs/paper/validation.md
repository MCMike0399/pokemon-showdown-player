# Paper validation

Validated on 2026-10-07. This checks the document and source correspondence;
it is not a new battle-strength experiment.

- Compiled `main.tex` and its BibTeX bibliography using Tectonic 0.17.0.
- Final PDF: 24 pages, 16 proofs, six worked examples and 18 cited sources.
- Checked all citation keys and internal references; the PDF contains no unresolved-reference markers.
- Checked the actual `ActorCritic` parameter count: 97,826.
- Independently recalculated the softmax, GAE, Beta uncertainty and Hoeffding examples.
- Reviewed the mathematical assumptions and implementation qualifications against source and primary references; strengthened the policy-gradient theorem with a finite-trajectory-space assumption.
- Inspected rendered title, contents, architecture diagram, mathematical pages and references. Text-block bounds stay within the page; the final build has no overfull boxes.
- Existing publication tests: **2 passed**.
- A temporary clean-source export includes the PDF, LaTeX, bibliography, build instructions, research notes and this validation note; its secret scan passes and excludes auxiliary TeX build files.
- `git diff --check` passes.

The historical playing results discussed in the paper come from the repository's
existing validation documents. No live login, battle, policy training or promotion
was performed to write this paper.
