# Validation record — 6 October 2026

## Verified locally

- Supplied RP1.pdf: rendered 14 scanned pages; methodology/equations, implementation settings and Table A inspected visually. The authoritative mathematical audit is in SPECIFICATION.md.
- Installed environment: Python 3.12.14, PyTorch 2.5.1+cu121, PyG 2.6.1, NumPy 1.26.4, NetworkX 3.4.2, igraph 0.11.8. `pip check` reports no broken requirements.
- Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU; CUDA backward diagnostic passed. Host i7-13620H / 16GB RAM.
- Official MGTAB archive successfully downloaded and loaded: 10,199 users, 788 features, 1,700,108 original relation edges; labels 7,451 humans / 2,748 bots; all seven relation IDs checked. Undirected union has 902,856 simple edges; relation-wise symmetrization has 2,553,457 edges.
- Released explicit edge weights are all 1; URL/hashtag weights are noninteger and their PMI semantics were checked against the MGTAB paper. They are not treated as event-frequency counts. The frequency fallback is disclosed in SPECIFICATION and PHASE2_LEDGER.
- 16 tests pass: independent exact graphlet enumeration and ORCA agreement on five random graphs; 80/10/10 split coverage; training-only scaling; dyadic categories; exact local/global metrics; igraph/NetworkX agreement including an isolate; dense vs chunked NT-Xent values and gradients; H-GNN/A-Fusion shapes/gradients; full vs query-chunked attention values/gradients; explicit-frequency/PMI separation; resumed local extraction; disconnected encoding-tree validity; interrupted-vs-uninterrupted CPU training and held-out evaluation.
- Full pipeline smoke: 40 explicitly synthetic users; exact feature/tree preparation; original SEP_U, SEP_G, FACN and both contrastive losses; CPU and CUDA training; validation-selected checkpoint; CUDA and CPU evaluation of the GPU checkpoint; CSV/JSON exports. Synthetic GPU run completed three epochs with ~21.6MB peak allocated CUDA memory. These scores have no benchmark meaning.
- Full-size MLS component stress check: actual 10,199-user MGTAB topology with 2,553,457 relation-symmetrized edges, random structural/contrastive inputs, H-GNN + A-Fusion + exact NT-Xent forward/backward. Finite loss/gradients, ~750.1MB peak allocated CUDA memory, ~2.68 seconds measured compute. This is a **component capacity check**, not full SeBot-MLS training: real entropy-tree/FACN activation memory is excluded.
- Representative exact two-hop statistics on real user 0: 7,118 ego vertices / 685,419 edges; compiled ACC/transitivity calculation measured ~1.43 seconds. This illustrates preprocessing scale; it is not a total runtime estimate.
- Colab/Kaggle notebook Python cells compile. Both call the same installed package and preserve output/cache paths. ORCA source builds successfully with local MinGW g++.

## Still unverified

**No full MGTAB training/evaluation result is claimed.** Exact full-graph ego extraction and global/subgraph entropy-tree preparation have not been completed for all real users in this session. The prepared end-to-end checkpoint shipped locally is synthetic only. Full model VRAM fit depends on the real tree inputs in addition to the tested MLS component.

Colab execution remains unverified. Kaggle synthetic execution was verified in the user's account on 6 October 2026: private notebook `viswasrbhat/sebot-mls-mgtab-baseline`, version 1 (`scriptVersionId=355677945`), Save & Run All, GPU T4 x2 allocated with only GPU 0 used. The isolated Python 3.12.3 / PyTorch 2.5.1+cu121 environment installed successfully; Tesla T4 had 14.56 GiB accessible GPU memory and passed CUDA backward diagnostics. All 16 tests passed in 7.23 seconds; synthetic preparation, three-epoch training, checkpoint saving and evaluation completed successfully. The Linux ORCA executable compiled; the existing optional ORCA test path targets the Windows executable, so this cloud test run does not establish Linux ORCA agreement. Total batch duration included approximately four minutes of fresh dependency setup. Synthetic metrics are not MGTAB results. Full MGTAB preprocessing/training remains unverified.

The authors' SeBot-MLS source, exact interaction-frequency reconstruction, graph/motif conventions, head counts, internal MLP choices and split seed are not available in RP1. This implementation discloses those choices; it cannot certify identical official-paper results. Resolve the D/M entries in PHASE2_LEDGER before claiming exact numerical reproduction.

## Recheck commands

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m sebot_mls.cli doctor
.\.venv\Scripts\python.exe scripts/check_gpu_capacity.py
```

The full-size capacity script uses random inputs on actual topology and reports only execution checks. It does not train a classifier or produce benchmark scores.
