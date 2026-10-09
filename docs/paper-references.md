# Primary sources and proof boundaries for the technical paper

## Second edition (2026-10-09)

The paper was rewritten against public source `0c00ec2`. Where the first-edition
notes below disagree with the current code, the paper's Appendix B is
authoritative. In particular:

- **Promotion.** The gate is `ceil(n*m)` plus an exact one-sided paired sign test
  at p <= .05, not a margin alone.
- **Temperature.** The policy uses a temperature, and the deployed value is 0.25.
- **Prior.** The deployed prior is the squashed `strategic-v1` scenario score.
- **Live turns.** These are decided by search.

The new sources were checked as follows:

- **Search and games.**
  - [Perfect-information Monte Carlo](https://ojs.aaai.org/index.php/AAAI/article/view/7562)
    (Long et al., AAAI 2010) explains when determinized search succeeds.
  - [Strategy fusion](https://www.dai.ed.ac.uk/daidb/papers/documents/rp780.html)
    (Frank and Basin, *Artificial Intelligence* 100, 1998) is the classical
    critique.
  - [Information-set MCTS](https://eprints.whiterose.ac.uk/75048/)
    (Cowling et al., IEEE TCIAIG 2012) is the alternative that avoids fusion.
  - [Regret matching](https://doi.org/10.1111/1468-0262.00153)
    (Hart and Mas-Colell, *Econometrica* 68(5), 2000) and the
    [regret bound used for equilibrium approximation](https://proceedings.neurips.cc/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)
    (Zinkevich et al., NeurIPS 2007) support Proposition 11.2. Only the
    equilibrium folk argument is proved in the paper.
  - [Quantal response](https://resolver.caltech.edu/CaltechAUTHORS:20171128-164515991)
    (McKelvey and Palfrey, *GEB* 10, 1995) motivates the soft-response half of
    the opponent model.
- **Statistics.**
  - [McNemar (1947)](https://link.springer.com/article/10.1007/BF02295996)
    covers the exact conditional sign test used by the gate.
  - [Holm (1979)](https://www.jstor.org/stable/4615733) is the correction used
    by the offline matrix.
  - The k3 KL estimator is from
    [Schulman (2020)](http://joschu.net/blog/kl-approx.html). Its unbiasedness
    and non-negativity are proved in the paper (Proposition 8.8).
- **Ladder.** The [official ladder page](https://pokemonshowdown.com/pages/ladderhelp)
  gives the Elo floor of 1000 and the asymmetric K below 1100 (80/20 at the
  floor, interpolated to 50/50 at 1100; 40 from 1300). These are the inputs to
  Proposition 14.1.
- **Usage data.** [Smogon usage statistics](https://www.smogon.com/stats/):
  September 2026 chaos JSON for `gen9championsvgc2026regmc`, 1,631,943 battles
  (from the file's `info` block).
- **Platform.**
  [`torch.mps.set_per_process_memory_fraction`](https://pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction)
  limits allocations relative to Metal's recommended working-set size. Also
  checked: [server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
  and [Playwright](https://playwright.dev/).
- **Related agents.**
  [PokéChamp](https://proceedings.mlr.press/v267/karten25a.html) (ICML 2025)
  and [Metamon](https://arxiv.org/abs/2504.04395) (RLJ/RLC 2025). They are
  cited as directions, not as comparable measurements.

The first-edition notes follow unchanged, for provenance.

Checked 2026-10-07 against repository source and the primary sources below. This is an editorial research note, not a new runtime implementation or an experimental result.

## Exact implementation facts that change the mathematics

- `ml/model.py` scores each **complete enumerated choice**, using 384 state features, 192 action features, state embedding 128, action embedding 96, joint hidden layer 96 and a separate 64-unit critic hidden layer. The final action feature is added directly to the logit. The actor's final layer starts at zero, so the initial policy is exactly softmax of that heuristic feature for finite inputs. The heuristic is a rough power/accuracy/STAB/type-pressure score, not the simulator damage equation.
- The GAE recurrence uses gamma 0.99 and lambda 0.95, terminal reward in {-1,0,1}, zero terminal bootstrap, and no intermediate reward. PPO uses clip interval [0.8,1.2], value coefficient 0.5, entropy coefficient 0.01, Adam learning rate 0.0003, minibatches of 32 and gradient norm cap 0.5. Behavioral cloning uses negative log likelihood only. Training accepts current-revision, complete stochastic records for PPO and explicit demonstrations for cloning. [Source](../ml/model.py)
- `ml/teams.py` uses a Beta(1,1) prior, adds outcome score (outcome+1)/2 to the first shape parameter, and its complement to the second. **Exploration is independent Beta posterior sampling followed by selecting the maximum: Thompson sampling. It is not Boltzmann team selection.** `rank_teams` displays mean+standard-deviation as `exploration_score`, but the planner does not select with that expression. [Source](../ml/teams.py)
- `ml/worker.py` evaluates candidate and incumbent with the same team pairing, seed and side for each index. Sides alternate across different indices/seeds; it does not execute both sides of every seed. Promotion requires all games complete, no rejected actions, and at least max(1,ceil(n*margin)) more candidate wins. Default n=20 and margin=0.10 require two extra wins. **No confidence bound is used by this promotion gate.** Any Hoeffding certificate in the paper must be explicitly labeled additional mathematical analysis or a proposed extension. [Worker](../ml/worker.py), [defaults](../ml/continuous.py)
- `ml/scout.py` trains a supervised next-executed-public-move classifier from observations captured at the start of the turn. Its labels are executed move events rather than submitted intentions. Called moves and actors that entered during the turn are excluded. Its held-out partition is battle-disjoint. At inference, it restricts outputs to disclosed sheet moves or moves previously observed for that species, then renormalizes and returns the top three. Those three probabilities need not sum to one. [Source](../ml/scout.py)

## Proofs that are appropriate, with explicit hypotheses

1. **Masked normalization and support.** For a nonempty finite enumerated set L(h), define pi(a|h)=exp(z(h,a))/sum over L(h) exp(z(h,b)). This gives zero probability to choices outside L(h), and positive probability inside for finite logits. It only guarantees server legality if enumeration is sound for the request and the server has no undisclosed constraints. The server can reject choices and return new information. Training padding uses `torch.finfo(dtype).min`, a finite numerical sentinel; exact-zero support is an ideal mathematical formulation, with floating-point underflow in ordinary implementations. An all-false padding mask has no valid categorical interpretation and must be excluded by the nonempty-set assumption. [Masking paper](https://arxiv.org/abs/2006.14171), [server choice/error contract](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md)
2. **Permutation equivariance.** A shared scorer over candidates and symmetric softmax preserves candidate probabilities when their list is permuted. With a unique maximum the chosen choice is invariant; deterministic `argmax` resolves exact ties by array order, so tied choices need a caveat. Enumeration is over complete joint choices, allowing coupling between the two active slots. Feature collisions can make different choices indistinguishable.
3. **Heuristic residual as exponential tilting.** With prior q(a)=softmax(h(a)), learned residual f gives pi(a)=q(a)exp(f(a))/Z. Derive the variational identity sum pi*f-KL(pi||q)=log Z-KL(pi||pi_star). Nonnegativity of KL proves the unique optimum pi_star. This explains the existing additive logit without asserting its prior is calibrated battle knowledge.
4. **Trajectory likelihood gradient and baseline cancellation.** Work on full information history h, rather than asserting that the hashed vector is Markov. With finite horizon, differentiable positive policy on parameter-independent support and parameter-independent environment/opponent kernel, differentiating trajectory likelihood yields E[R sum_t grad log pi(a_t|h_t)]. Conditional action-score mean is zero, so any action-independent b(h_t) can be subtracted without changing the expectation. A learned opponent whose parameters are changed jointly would violate the fixed-kernel argument. This does not prove the repository's GAE/minibatch/clipping gradient is an unbiased gradient of raw win probability. [Original policy-gradient paper](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
5. **GAE identity.** Define delta_t=r_t+gamma V_(t+1)-V_t and Ahat_t=sum_l (gamma lambda)^l delta_(t+l), with the finite terminal boundary. The backward recurrence follows immediately by splitting off l=0. For lambda=1 and V_T=0, telescoping gives Ahat_t=G_t-V_t. Lambda below one mixes bootstrap targets and introduces bias when the critic is inaccurate; compressed histories can make that unavoidable. Gamma below one also changes the objective: gamma^(T-1)*terminal_outcome is not raw terminal_outcome if episode lengths vary. The practical PPO loss averages unweighted per-step advantages; do not claim exact equality with the globally discounted start-state policy gradient, which includes temporal discount weighting. [GAE paper](https://arxiv.org/abs/1506.02438)
6. **Pointwise PPO surrogate bound.** min(r*A,clip(r)*A)<=r*A is algebraic. For A>0 the surrogate saturates only above 1+epsilon, and for A<0 only below 1-epsilon. This does not constrain every probability ratio to the interval, bound KL divergence, prove monotonic improvement or establish competitive play. Clipping is a practical optimization heuristic evaluated empirically. [PPO paper](https://arxiv.org/abs/1707.06347)
7. **Team posterior conjugacy.** For binary wins/losses with stationary independent Bernoulli success probability p, multiplying a Beta prior by the Bernoulli likelihood proves Beta(1+w,1+l), mean alpha/(alpha+beta), variance alpha*beta/[(alpha+beta)^2*(alpha+beta+1)]. The actual tie update contributes 0.5 to both counts: this is a fractional-likelihood score model, not literal Bernoulli conjugacy for a three-outcome observation. Outcomes also mix policy revisions/opponents over time; posterior uncertainty is model-based and not a universal confidence interval. Thompson draws are implemented over a bounded, validated candidate subset, not all possible teams. [Thompson original paper](https://academic.oup.com/biomet/article-abstract/25/3-4/285/200862)
8. **Paired Hoeffding certificate, separate from implementation.** Let D_i=W_i(candidate)-W_i(incumbent) in [-1,1]. If independent random evaluation units are drawn in advance and both frozen policies face the prescribed same unit, P(mean D-E mean D>=epsilon)<=exp(-n epsilon^2/2). Thus E mean D>=observed mean D-sqrt(2 log(1/alpha)/n) with probability at least 1-alpha. Independence is needed between units, not between the two results inside a unit. The bound can target an average of nonidentical fixed matchup means, but does not certify unseen matchups or public ladder strength. Reusing adaptively selected validation units or repeatedly testing candidates requires a separate correction. For n=20, alpha=.05, radius is about .547, much larger than the runtime's .10 empirical margin. [Original author's report](https://repository.lib.ncsu.edu/bitstreams/d0e6ed15-3e1c-432f-8419-e55ffb6f3171/download), [journal DOI](https://doi.org/10.1080/01621459.1963.10500830)

## Architecture and lifecycle sources

MCP provides JSON-RPC messages, schema-bearing tool discovery/invocation and negotiated initialization. In stdio mode the client launches a child and communicates through newline-delimited stdin/stdout; normal logging belongs on stderr. It does not define the game policy or magically persist server memory across process termination. [MCP tools specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools), [transport specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports), [lifecycle specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)

OpenClaw's native registry owns downstream MCP server connections. Its session-scoped runtimes can persist between turns; detached one-shot runs retire them when the run ends. Reset, Stop, compaction rollover, relevant config changes and Gateway shutdown retire runtimes. Idle eviction is opt-in. A retained transcript alone does not retain the child. `openclaw mcp reload` applies to the calling process. This explains why a full battle needs one retained runtime or one battle-owning call. [OpenClaw registry and runtime lifecycle](https://docs.openclaw.ai/cli/mcp/registry)

Showdown's official protocol documents request JSON, optional rqid, simultaneous doubles choice components, target signs, invalid/unavailable-choice errors, and terminal win/tie events. rqid can be appended to `/choose` to associate a submission with its intended request. Raw public transitions do not contain every private legal request or an unknown collecting policy's probabilities. [Official simulator protocol](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md)

The current target is Champions Regulation M-C, not generic Scarlet/Violet Gen 9. The official notice permits Mega Evolution once per battle, forbids duplicate held items, and specifies the September 9–December 2, 2026 period. Champions simulator source changes stat/PP mechanics and disables Terastallization. [Official M-C notice](https://champions-news.pokemon-home.com/en/page/816.html), [Champions simulator source](https://github.com/smogon/pokemon-showdown/blob/master/data/mods/champions/scripts.ts)

VGC-Bench is a related system, not this model: its authors use BC/PPO, transformer encoding, masking and population methods. Its professional-player result is a restricted single-team mirror setting; it does not establish broad competitive strength for this repository. Team building is left open. [Original AAMAS paper](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/angliss2026vgc.pdf)

## Verified BibTeX entries

The GAE paper was first posted in 2015, appeared at ICLR 2016, and has a 2018 arXiv revision. Its conference year is also confirmed by the [coauthor's publication list](https://people.eecs.berkeley.edu/~jordan/control.html). The masking paper was first posted in 2020 and published in FLAIRS 2022. Keep these dates consistent with the chosen citation type.

```bibtex
@article{schulman2017ppo,
  author = {John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
  title = {Proximal Policy Optimization Algorithms},
  journal = {arXiv preprint arXiv:1707.06347}, year = {2017},
  doi = {10.48550/arXiv.1707.06347}, url = {https://arxiv.org/abs/1707.06347}
}
@inproceedings{schulman2016gae,
  author = {John Schulman and Philipp Moritz and Sergey Levine and Michael I. Jordan and Pieter Abbeel},
  title = {High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  booktitle = {International Conference on Learning Representations}, year = {2016},
  url = {https://arxiv.org/abs/1506.02438}
}
@inproceedings{sutton1999policygradient,
  author = {Richard S. Sutton and David A. McAllester and Satinder P. Singh and Yishay Mansour},
  title = {Policy Gradient Methods for Reinforcement Learning with Function Approximation},
  booktitle = {Advances in Neural Information Processing Systems}, volume = {12}, year = {1999},
  url = {https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html}
}
@article{huang2022masking,
  author = {Shengyi Huang and Santiago Onta{\~n}{\'o}n},
  title = {A Closer Look at Invalid Action Masking in Policy Gradient Algorithms},
  journal = {The International FLAIRS Conference Proceedings}, volume = {35}, year = {2022},
  doi = {10.32473/flairs.v35i.130584}, url = {https://arxiv.org/abs/2006.14171}
}
@article{thompson1933,
  author = {William R. Thompson},
  title = {On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples},
  journal = {Biometrika}, volume = {25}, number = {3--4}, pages = {285--294}, year = {1933},
  doi = {10.1093/biomet/25.3-4.285},
  url = {https://academic.oup.com/biomet/article-abstract/25/3-4/285/200862}
}
@article{hoeffding1963,
  author = {Wassily Hoeffding}, title = {Probability Inequalities for Sums of Bounded Random Variables},
  journal = {Journal of the American Statistical Association}, volume = {58}, number = {301},
  pages = {13--30}, year = {1963}, doi = {10.1080/01621459.1963.10500830},
  url = {https://doi.org/10.1080/01621459.1963.10500830}
}
@article{smallwood1973,
  author = {Richard D. Smallwood and Edward J. Sondik},
  title = {The Optimal Control of Partially Observable Markov Processes over a Finite Horizon},
  journal = {Operations Research}, volume = {21}, number = {5}, pages = {1071--1088}, year = {1973},
  doi = {10.1287/opre.21.5.1071}, url = {https://pubsonline.informs.org/doi/10.1287/opre.21.5.1071}
}
@inproceedings{angliss2026vgc,
  author = {Cameron L. Angliss and Jiaxun Cui and Jiaheng Hu and Arrasy Rahman and Peter Stone},
  title = {{VGC-Bench}: Towards Mastering Diverse Team Strategies in Competitive {Pok{\'e}mon}},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems},
  publisher = {IFAAMAS}, year = {2026},
  url = {https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/angliss2026vgc.pdf}
}
@misc{mcpspec2025,
  author = {{Model Context Protocol contributors}},
  title = {Model Context Protocol Specification: Lifecycle, Transports and Tools}, year = {2025},
  url = {https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle}, note = {Version 2025-06-18; accessed 2026-10-07}
}
@misc{openclawmcp,
  author = {{OpenClaw contributors}}, title = {Manage Saved MCP Servers},
  url = {https://docs.openclaw.ai/cli/mcp/registry}, note = {Accessed 2026-10-07}
}
@misc{showdownprotocol,
  author = {{Pok{\'e}mon Showdown contributors}}, title = {Pok{\'e}mon Showdown Simulator Protocol},
  url = {https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md}, note = {Accessed 2026-10-07}
}
@misc{championsmc2026,
  author = {{The Pok{\'e}mon Company}}, title = {Regulation Set M-C (Updated on September 9)}, year = {2026},
  url = {https://champions-news.pokemon-home.com/en/page/816.html}, note = {Accessed 2026-10-07}
}
```

Access limitation: the Hoeffding journal publisher page did not open in the research browser; its DOI and metadata were cross-checked, and the author-written 1962 report was indexed in the original institution's repository. That report's direct fetch encountered bot detection. Do not claim the journal PDF was read end to end. Thompson's original publisher metadata was verified, while its complete PDF requires access. The posterior calculation and concentration proof can be supplied independently in the paper.
