# RP1 specification audit

Authority: the supplied RP1.pdf, Xinyi Tian et al., *SeBot-MLS: Multi-level structural feature learning for graph-based social bot detection*, Journal of Safety Science and Resilience 7 (2026), 100310, DOI https://doi.org/10.1016/j.jnlssr.2026.100310. All 14 supplied pages were rendered; pages 3-7 and Appendix Table A on page 13 were inspected visually. The PDF is image-only, so text extraction cannot verify the equations. No architecture extension is included.

**Status:** an equation-driven reimplementation, not an authenticated copy of the authors' SeBot-MLS code. Page 13 states that code is available on request. Exact numerical reproduction remains unverified until official MGTAB training and reconciliation of unspecified conventions. The paper reports MGTAB accuracy 91.76%, bot F1 84.62%, recall 85.56%, precision 83.70%; these are reference results, never generated results.

## Verified dimensions and equations

| Source | Implementation | Dimensions / behavior |
|---|---|---|
| p.3 Eq.2 | `SeBotBranch` | Global SEP_U, subgraph SEP_G, relational FACN; 32 each -> 96 |
| p.3 Eq.3-5 | `SeBotMLS` | MLS 32; concatenate to 128; Linear -> LeakyReLU -> Linear -> two logits |
| p.3 Eq.6, p.4-5 Eq.7-11 | `dyadic` | MGTAB dyadic 12 = intimacy 6 + reciprocity 6 |
| p.13 Table A | `INTIMACY_WEIGHTS` | follower .60, friend .80, mention 1.00, reply .80, quote .60, URL .35, hashtag .30 |
| p.4 Eq.7-8 | `dyadic` | Sum interaction weight × frequency per ordered user pair; five global-quantile bucket counts + total relationship count |
| p.4 Eq.9, p.5 Eq.10 | `dyadic` | `1 - 2*abs(Iij/(Iij+Iji)-.5)` over five directed relation types; unilateral, three interior bins, complete, self-loop |
| p.5 Eq.13 / Fig.2 | `motif_counts` | 8 counts: path3, triangle3, star4, path4, paw4, cycle4, diamond4, clique4 |
| p.5 Eq.14 | `local_features` | Two-hop ego graph: nodes, edges, density, center's Burt constraint, average clustering, transitivity |
| p.5 Eq.15 | `global_features` | PageRank, eigenvector, betweenness, closeness, in that order |
| p.5-6 Eq.16-23 | `LevelEncoder` | Separate 6+6, 2+6, 3+3 MLPs -> 8+8; global 4 -> 16 |
| p.6 Eq.24-27 | `HGNN` | GraphSAGE -> ReLU -> self-attention across users -> projection -> LayerNorm; four 16-dimensional levels |
| p.6 Eq.28-29 | `AFusion` | Stack four levels into 64; attention across the four level tokens of each user |
| p.6 Eq.30-34 | `AFusion` | Dynamic `softmax(W e+b)`; static sigmoid of four trainable parameters; normalize product; weighted sum -> Linear 16 to 32 |
| p.6 Eq.35-36 | training loop | Classification + .1 NCL + .1 SCL; only train labels supervise classification |
| p.6 §4.1(1) | `make_split` | MGTAB random 8:1:1; TwiBot-20 7:2:1, not implemented here |
| p.7 §4.1(4) | `mgtab.yaml` | Adam, lr .01, maximum 200 epochs, early stopping; MGTAB batch = all 10,199 users |

MGTAB counts are 8,159 / 1,020 / 1,020 using floor boundaries on a seeded shuffled array. This intentionally supersedes the original MGTAB benchmark and original SeBot scripts' 70/20/10 setting. Labels are never used in motif, centrality, tree, quantile or graph construction. Full graph topology and unlabeled user attributes participate in the transductive encoders; holdout labels do not. Standardization uses training users only.

## Choices the paper does not resolve

These are disclosed implementation conventions, not claims that the authors used them. Author clarification/source code is needed for strict result matching.

1. **Seed and split indices:** use seed 42 inherited from the original SeBot script; non-stratified NumPy RandomState permutation. Save actual indices and input hashes.
2. **Graph projection:** use a simple undirected union of all seven relation types for polyadic/local/global topology, remove self-loops there. Retain original directed weighted edges for dyadic features. The paper does not identify the exact projection/multigraph/directed conventions.
3. **Frequency gap in standardized data:** inspection of the official release found unit weights for the five explicit types, and noninteger weights for URL/hashtag. The MGTAB paper §4 (https://arxiv.org/pdf/2301.01123) describes implicit weights as average pointwise mutual information. They are **not interaction counts**. RP1 does not explain how it recovered frequency. Default `frequency_source: edge_occurrences` counts each observed edge record once; repeated records accumulate. This is an explicit fallback and a material reproduction limitation, not recovered raw tweet frequency. To use real counts, provide an aligned `interaction_frequency.pt` tensor and set `frequency_source: file`; its bytes enter the dataset fingerprint. PMI weights are never substituted for counts. Quantile populations use positive ordered-pair intimacy values, global across the full observed graph; ties go in the lower bucket. `intimacy_total` counts relationships, not sum of intimacy scores, per the prose of Eq.8.
4. **Reciprocity:** merge five directed types before Eq.9; include both participants of a unilateral pair. Exclude exact 0/1 and self-loops when calculating the P10/P90 interior population so the six categories are disjoint. The paper does not state whether extrema participate in those quantiles or how ties are handled.
5. **Motifs:** count connected **induced** 3/4-node graphlets containing each user. ORCA orbit sums match independent brute enumeration. No sampling or degree cap is used for MLS motifs. The paper does not state induced vs non-induced orbits; Fig.2 labels triangle+1E and triangle+2E are interpreted as paw and diamond.
6. **Local statistics:** compute constraint for the ego center, ACC across every vertex in the radius-2 induced ego, and transitivity across that ego. Set undefined constraint of isolates to zero. Empty clustering/transitivity are zero. Disconnected closeness uses reachable-node count (NetworkX `wf_improved=False`). Betweenness is unnormalized as in Fig.2; PageRank damping .85 and L2-normalized eigenvector centrality are standard conventions. These normalization/disconnected details are not specified.
7. **Attention and MLPs:** four heads, single Linear+LeakyReLU sub-encoders (two equal eight-dimensional parts), 32 classifier hidden units. A-Fusion reshapes 64 into four 16-dimensional tokens. Dynamic weights use pre-attention `e`, following Eq.30. Cross-node H-GNN attention sees the full user population. Head counts, internal MLP depth, dropout placement, and tokenization are not published.
8. **Training:** dropout .3, patience 50, temperature .1, tree depth 4, two global pooling blocks, three subgraph convolutions and ten outgoing neighbor entries are inherited from original SeBot source. Adam has no weight decay/scheduler here, because RP1 explicitly names Adam but does not specify decay or scheduling. Validation loss selects the checkpoint; tests are evaluated only by the evaluate command. Classification uses mean cross-entropy, matching SeBot practice; Eq.36 writes a sum.
9. **Trees:** preserve upstream structural entropy tree generation and its neural modules, not Louvain substitutes. A disconnected-component wrapper is used to keep isolates valid. The subgraph cap and random star for isolated users are upstream SeBot behaviors; they do not affect MLS extraction. The original tree caches' sampling seed and exact construction provenance are not in RP1. This project rebuilds trees on the downloaded graph.
10. **Memory and numerical platform:** modern PyTorch 2.5.1 / PyG 2.6.1 replaces the paper's PyTorch 1.12 / PyG 2.1 environment for Windows Python 3.12 compatibility. Full cross-node attention uses PyTorch scaled-dot-product kernels (`need_weights=False`). Exact symmetric NT-Xent query chunks include all users as negatives; subgraph activations are checkpointed. ORCA and igraph C routines accelerate exact graphlet/topology calculations, checked against independent Python implementations. These change execution/memory, not the objective or graph coverage. Floating-point/GPU kernel differences can affect exact results. Disconnected graphs with tied largest eigenvalues may have nonunique eigenvectors; backend choice is recorded.

The original FACN draws random logistic noise even during evaluation. This behavior is retained, with a fixed evaluation RNG seed for repeatable validation/test outputs. CUDA scatter reductions may remain nondeterministic at floating-point rounding level.

## Original SeBot reuse

Source https://github.com/846468230/SEBot, commit `59b5407d2d5600aee06f507c2b0d343fa599b482`, Apache-2.0. `scripts/prepare_legacy.py` makes compatibility copies in `src/sebot_mls/legacy/`: relative imports, removal of unused sparse-only dispatch/imports, rank-two empty COO edge matrices and removal of upstream training entry points. COO message passing, pooling, GCN/FACN, feature encoders, and structural entropy algorithms are retained. Upstream sources and license remain in `vendor/SEBot` for review. The upstream trainer is not used because it has a different split, optimizer and evaluation flow.

ORCA source https://github.com/thocevar/orca, commit `e146a8b1a99a90f5e3096b7bcc2ab0ea246c3ca7`, GPL-3.0, is invoked as a separate CLI for exact graphlet counts. Sources and license accompany the executable. See NOTICE for third-party provenance.
