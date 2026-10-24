# Game-theoretic search: a shared root strategy

Research date: 2026-10-24. Source inspected: `cc320752b7b50ceb5a8778938a66b66a58e28709`. This is a design recommendation, not an experiment result or deployment. **The smallest meaningful architecture change is to solve for one mixed distribution over our joint actions across hidden worlds, then sample it.** Comparable action coverage across worlds is a prerequisite. There is no demonstrated Elo gain for this proposal.

## What the code actually does

[`search/config.py`](../search/config.py) specifies six worlds, three engines, a 24-by-20 retained matrix, one chance sample per cell, and `alpha=.25`. Close decisions can add six worlds within the adaptive 9-second budget. C5 also uses a sigmoid leaf, selective Protect continuation, and speed inference.

[`search/engine.cjs`](../search/engine.cjs), `searchWorld` and `solveZeroSum`, independently screen each world's actions and run 2,000 regret-matching iterations. They return average strategies `x_w,y_w`, but value our rows against the fixed model

`q_w = .25*y_w + .75*softmax(-column_mean(M_w))`.

[`SearchAgent._move`](../search/agent.py) ignores returned `mix`, averages each action over only the worlds retaining it, subtracts a single `.5` if any successful world omitted it, and returns a pure argmax. This is an approximate **best response to a model**, not execution of an equilibrium strategy. Finite regret iterations also do not certify an exact equilibrium. Regret theory concerns the averaged strategies and their deviation bounds. [Zinkevich et al., 2007, Theorem 2](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).

The distinction is already documented in the repository's [search paper source](paper/sec-11-search.tex), including a matching-pennies counterexample. Root `_move` commits to one action across worlds; **calling that root decision itself strategy fusion would be inaccurate**. Actual fusion occurs in `_force_switch`, which averages per-world best follow-ups, and potentially in `secondTurn`, which chooses continuations with determinized knowledge. Different continuations are valid only when intervening observations distinguish the worlds. The original determinization analysis and information-set search literature explain this information constraint. [Frank and Basin, 1998](https://www.sciencedirect.com/science/article/pii/S0004370297000829), [Cowling et al., 2012, sections I–II](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf).

## Four counterexamples that matter

These are mathematical fixtures, not evidence that a particular live loss had this cause. Arithmetic was checked with Python's exact fractions; no battles were run.

1. **A best response to equilibrium need not be safe.** For `M=[[1,-1],[-1,1]]`, both rows score zero against the equilibrium opponent. A pure tie-break has worst-case value −1; sampling each row with probability 1/2 guarantees zero in expectation. Increasing `alpha` to 1 does not fix pure argmax. Random simulation noise can incidentally vary actions, but gives no guarantee of the required distribution.
2. **Solving worlds independently is not solving the shared decision.** Give two equally likely opponent types matrices `M1=[[2,0],[0,1]]` and `M2=[[0,1],[2,0]]`. Each perfect-information game has value 2/3. If our probability of row A is `p`, the realizable security value is `0.5*min(2p,1-p)+0.5*min(2(1-p),p)`, at most 1/2, attained for `p` between 1/3 and 2/3. Either pure action guarantees zero. Per-world equilibrium row values tie, so averaging them then choosing a pure action loses the guarantee. Averaging the matrices also loses the opponent's ability to condition on its own type: it admits unsafe pure solutions here.
3. **Averaging per-world mixtures is not a general repair.** With a single opponent action, world 1 pays `(A=1,B=0)` and world 2 pays `(A=0,B=3)`. The per-world optimal mixtures average to 50:50, worth 1. The optimal shared policy always chooses B, worth 1.5. Similarly, a forced replacement offering hidden-state-dependent follow-ups `(1,0)` and `(0,1)` is scored as 1 by `mean(max)`, although its best uninformed follow-up is worth .5. A competing replacement guaranteeing .6 should win.
4. **Missing rows are not a probability estimate.** Suppose A survives in one of six worlds with value 1 and would score zero in the other five; B scores .4 in all worlds. Current aggregation gives A `.5` and chooses it, although its full-world expectation is 1/6. Screening is value-dependent, so omissions cannot be assumed random. The `.5` penalty is also insensitive to whether one or five worlds omitted the row. This can confound any mixed-strategy experiment.

## The proposed root game

Let `w` denote an opponent private type, `p_w` its posterior mass, and `M_w` a matrix whose cells already average future chance. Our action identifiers and rows must be shared; opponent action sets may differ by type. For the approximation that the opponent knows its type and our set, solve:

```text
maximize over x:  V(x) = sum_w p_w min_b sum_a x[a] M_w[a,b]
subject to:      x >= 0, sum_a x[a] = 1
```

Equivalently, maximize `sum_w p_w*z_w` with constraints `z_w <= sum_a x[a]*M_w[a,b]` for each `(w,b)`. This formulation is a direct derivation for this finite Bayesian root game. It shares our commitment while permitting informed opponent responses; it does not require enumerating the Cartesian product of opponent policies.

Implementation can adapt the existing regret loop: maintain one row-regret vector and one column-regret vector per type; update row regrets using posterior-weighted payoffs across types, and each opponent vector against the same `x`. Return time-averaged strategies. Log the computable primal-dual gap

`max_a sum_w p_w*(M_w*y_w)[a] - sum_w p_w*min_b (x^T*M_w)[b]`.

This bounds solution error **in the supplied surrogate game**, not full Pokémon exploitability. The averaged-strategy regret argument supplies the convergence basis. [Zinkevich et al., 2007](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).

Sample one complete doubles action from `x`, using a dedicated private RNG independent of world sampling; retain the sample for retries of the same request. Do not sample the two Pokémon's moves independently: correlated target, Protect, switch, and Mega decisions are the actions the matrix evaluates.

**Information boundary:** per-world opponent responses are legitimate for distinctions the opponent knows, such as its own brought Pokémon and stats. They must not condition on future damage rolls, accuracy, or hidden facts about our team. Average chance replicates within a private type before optimizing responses; merge indistinguishable opponent information states. A single random sample per cell is a noisy payoff estimator, not a revealed future. With our hidden set fixed and effectively exposed to the modeled opponent, this remains a one-sided-information approximation, not a full Bayesian equilibrium of VGC.

## Exploit weak opponents without discarding robustness

The existing `.25*Nash + .75*greedy` opponent model fixes both components before our final choice. Its Nash component cannot adapt to that choice, so `.25` is not an exploitability budget.

Restricted Nash responses instead optimize against a mixture of a fixed modeled opponent and an opponent still free to respond. Their poker evidence motivates testing this tradeoff, not transferring an Elo claim to Pokémon. [Johanson, Zinkevich and Bowling, 2007, section 5](https://poker.cs.ualberta.ca/publications/NIPS07-rnash.pdf).

For this root game, a small extension is

`maximize_x lambda*sum_w p_w*x^T M_w*g_w + (1-lambda)*V(x)`.

It can use the same solver with transformed cells `(1-lambda)*M_w[a,b] + lambda*(M_w*g_w)[a]`. Unlike the present fixed blend, the adversarial part responds to the shared `x`. Start with the existing greedy prior to isolate architecture; a nominal `.75` model weight is only an experimental setting, not inherited validation. A more interpretable alternative maximizes model payoff subject to `V(x) >= V* - epsilon`, where `V*` is the shared root maximin value. Report the measured shortfall and solver error.

Neither method gives a real-game safety guarantee with inaccurate leaves, omitted replies, or wrong beliefs. Safe-exploitation results require their specified game and information assumptions; an advantage estimated in a heuristic Pokémon position is not a bankroll of guaranteed future wins. [Ganzfried and Sandholm, *Safe Opponent Exploitation*](https://www.cs.cmu.edu/~sandholm/safeExploitation.ec12.pdf).

## Leaf values and VGC limitations

A scalar evaluation of a fully specified state is not generally enough for depth-limited imperfect-information solving. Brown et al. show that even equilibrium continuation values can erase the strategic constraints needed to recover a robust strategy; their remedy offers information-consistent continuation choices. This is a reason to avoid treating the current sigmoid leaf as a game-theoretic oracle. [Brown, Sandholm and Amos, 2018, section 2](https://noambrown.github.io/papers/18-NIPS-Depth.pdf).

Longer-term value learning should condition on public state and both players' ranges and predict appropriate information-set continuation values. That is substantially more than fitting match winners to full-state material features. Public belief states provide sufficient context under the paper's formal assumptions. [Kovařík et al., *Value Functions for Depth-Limited Solving*, section 4.3.2](https://arxiv.org/html/1906.06412v5). A small information-consistent continuation-policy portfolio is a possible intermediate experiment; its policies must not inspect hidden simulator truth.

Practical constraints from the inspected code:

- **Beliefs are approximate.** [`priors.py`](../search/priors.py) combines usage-based moves, items, abilities and spreads rather than sampling complete correlated teams from a calibrated posterior. [`agent.py`](../search/agent.py) adds revealed facts, team sheets when available, and speed filters. Six independent draws do not repair a misspecified set distribution. Distinguish open-sheet and closed-sheet observations.
- **The horizon matters.** Protect, redirection, speed control and switching change subsequent choices. Selective `secondTurn` retains determinized information and skips forced-switch continuations. A root equilibrium cannot fix those leaf errors. Mechanics should follow the [pinned simulator](https://github.com/smogon/pokemon-showdown/blob/c046106cbe075931b1ff8d8b800ff5be47a85f96/data/moves.ts), not assumptions imported from another regulation.
- **Depth corrections need integration.** Current Protect rescoring changes row `values` after `x,y` are solved, and normalizes the selected replies' correction by their total probability. Returned `mix` is therefore stale relative to those corrected values. Preserve actual corrected cell values and solve afterward; do not present a copied row correction as an information-consistent continuation matrix. First use a matched one-turn control to isolate the root change, then compare the integrated candidate against full C5.
- **Pruning limits the guarantee.** Both Python pruning and per-world screening restrict actions. A small duality gap only concerns retained rows/columns; an omitted adversarial reply can invalidate the apparent security. ISMCTS is not an automatic solution: its authors explicitly distinguish strong practical policies from convergence to Nash policies. [Cowling et al., 2012, section I](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf).

## M5 feasibility and a discriminating evaluation

At six worlds and 24×20 retained actions, the dense root contains only 2,880 payoffs, about 23 KB as float64 before metadata. A shared 2,000-iteration solve visits about 5.76 million matrix cells, the same order as the existing six separate solves. This is a CPU arithmetic task; MPS training is unnecessary. The expensive part remains simulator calls and coverage, not storing the game.

Separate screening from dense simulation: aggregate probe scores into one common bounded set of 24 own actions, then evaluate those rows in every world against that world's retained replies. This preserves the nominal dense-cell budget but changes screening and introduces a coordination barrier, so latency must be measured. Blindly taking the union of six different 24-row screens can require up to 144 rows, a sixfold increase. Never silently fill omitted payoffs with zero or the current fixed penalty. Genuine reconstruction failures should be surfaced separately from screening omissions.

No latency benchmark was run for this report. Existing [M5 measurements](c11-reconstruction-and-preview.md) show interference and live-priority effects, so arithmetic counts are a feasibility estimate only. Use frozen positions to measure median/p95/p99 latency, simulator counts, coverage, errors and timeouts under ordinary load; preserve the 9-second adaptive budget and fallback path. SciPy is absent from the current venv, so an LP implementation would add a dependency; extending the existing JS regret solver avoids that requirement.

Before any promotion, require:

1. Toy-game checks above, explicit probability/legality checks, matrix permutation invariance, and a small certified solver gap. On tiny games compare against independently derived exact solutions.
2. Frozen-position comparison separating coverage repair, mixed commitment, and shared-type solving; then robust exploitation. Report worst-case surrogate value, modeled payoff, entropy, action changes and deeper-search effects. These diagnose behavior, not strength.
3. A preregistered fresh paired strength test against diverse opponents, followed by a fresh randomized ladder test against the actual C5 baseline. Keep preview/team fixed, preserve ongoing campaign protocols, and report rating-adjusted effect with uncertainty. Pure minimax can surrender gains against weak humans; a better surrogate security value can still reduce Elo.

The historical [c6 results](c7-live-review.md) failed to establish a benefit from more worlds or wider screening; they do not prove sampling noise is irrelevant. This report establishes concrete failure modes and a tractable candidate architecture. **Only fresh controlled games can establish an actual Elo improvement.**
