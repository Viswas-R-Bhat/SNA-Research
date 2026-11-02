# SeBot-MLS baseline for MGTAB

A modular Python project for the SNA college baseline: original SeBot global/subgraph/relational user encoders, 30 structural inputs across four levels, H-GNN, A-Fusion, two-class classifier, contrastive + supervised training, validation-selected checkpoints and held-out evaluation.

**Read [the specification audit](docs/SPECIFICATION.md) before presenting this as a reproduction.** Equations/dimensions and the MGTAB 80/10/10 split were checked against the supplied scanned RP1.pdf. The paper does not publish enough implementation detail to guarantee an identical run; unresolved choices are listed explicitly. Reference paper scores are not this project's results. No proposed extensions are included.

Everything beyond RP1, including assumptions, frequency fallback, execution optimizations, testing, checkpointing and cloud support, is recorded in the [Phase 2 additions ledger](docs/PHASE2_LEDGER.md). These additions are not claimed as accuracy improvements.

## Recommended platform

Write/debug in local VS Code, verify on the RTX 3050 6GB, and use Kaggle optionally for larger-memory training. Colab is also supported. See [platform comparison](docs/PLATFORMS.md) and the two notebooks under `notebooks/`. They use the same package and config as Windows.

## Windows setup

Install Python **3.12 (64-bit)** and a working NVIDIA driver. The computer's default Python 3.15 is outside this project's supported range; do not create the environment with that interpreter. Open this folder in VS Code, then use a PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\python.exe -m sebot_mls.cli doctor
```

The setup uses PyTorch 2.5.1 CUDA 12.1 wheels and PyG 2.6.1; no local CUDA toolkit or `torch_sparse` compilation is required for the COO message-passing path. Select `.venv\Scripts\python.exe` as the VS Code interpreter. If Python 3.12 has a custom location, pass `-Python 'C:\path\python.exe'`. For CPU-only installation add `-Cpu`, and use `--device cpu` when running training/evaluation. CUDA training requires `doctor` to report a successful CUDA backward pass. Official wheel installation reference: https://pytorch.org/get-started/previous-versions/#v251.

## First verify the runnable pipeline

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m sebot_mls.cli synthetic --config configs/smoke.yaml
.\.venv\Scripts\python.exe -m sebot_mls.cli prepare --config configs/smoke.yaml
.\.venv\Scripts\python.exe -m sebot_mls.cli train --config configs/smoke.yaml --device cuda
.\.venv\Scripts\python.exe -m sebot_mls.cli evaluate --config configs/smoke.yaml --device cuda
```

This creates a small **synthetic** fixture and tests all branches/losses. It is not a bot-detection benchmark or evidence of the paper's accuracy. Synthetic targets must be empty on creation; on later runs reuse the data and prepared cache.

## Obtain official MGTAB

```powershell
.\.venv\Scripts\python.exe -m sebot_mls.cli download
```

This downloads the official MGTAB ZIP and extracts only these tensors into `data/MGTAB`:

```text
features.pt       [N,788] float user features
labels_bot.pt     [N]     0 human / 1 bot
edge_index.pt     [2,E]   directed original endpoints
edge_type.pt      [E]     0 follower, 1 friend, 2 mention, 3 reply,
                         4 quote, 5 URL, 6 hashtag
edge_weight.pt    [E]     released edge strengths; URL/hashtag use PMI
```

If Google Drive blocks automated retrieval, download using the official [MGTAB release link](https://drive.google.com/uc?export=download&id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe) from the [benchmark repository](https://github.com/GraphDetec/MGTAB), and place those five files directly in `data/MGTAB`. Standard MGTAB only: 10,199 labeled users. MGTAB-large is not supported in this baseline. Keep the official downloaded tensors unchanged. Existing tensor files are preserved by download; validation rejects shape/type mismatches.

**Frequency limitation:** the standard tensors do not supply raw interaction counts required by RP1 Eq.7. The default counts observed edge records; it does not treat PMI weights as event counts. If authoritative per-edge counts are available, place `interaction_frequency.pt` alongside the tensors and set `frequency_source: file`. This gap prevents a claim of exact dyadic-feature reproduction; see the audit.

## Exact graphlets, preprocessing and training

The supplied Windows ORCA executable lives in `tools/orca.exe`; corresponding GPL-3.0 source/license is in `vendor/orca`. To rebuild with MinGW g++:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_orca.ps1
```

Alternatively compile `vendor/orca/orca.cpp` with an appropriate C++11 compiler. On Linux/cloud: `g++ -O3 -std=c++11 vendor/orca/orca.cpp -o tools/orca`, then set `orca_executable: tools/orca` in that cloud config. `motif_backend: enumerate` is an exact independent Python fallback for small fixtures, not a practical choice for dense MGTAB.

```powershell
.\.venv\Scripts\python.exe -m sebot_mls.cli prepare --config configs/mgtab.yaml
.\.venv\Scripts\python.exe -m sebot_mls.cli train --config configs/mgtab.yaml
.\.venv\Scripts\python.exe -m sebot_mls.cli evaluate --config configs/mgtab.yaml
```

Preparation uses the full unlabeled topology, exact motifs, radius-2 ego statistics, exact global centralities, and original structural-entropy coding trees. MGTAB defaults to the compiled igraph backend for exact local/global statistics; tests compare it with NetworkX. It can be CPU- and RAM-intensive. The original global tree algorithm constructs dense CPU adjacency (~0.8GB before additional copies/algorithm state for 10,199 users). Two-hop ego statistics on a dense graph may dominate runtime. This is not a near-instant preprocessing promise.

The processed cache records input SHA-256, split IDs, quantile boundaries, train-only normalization statistics, feature order and trees. Changing data/split/extraction settings requires a separate cache or `prepare --force`. Holdout labels do not influence feature extraction or normalization. Graph-derived global quantiles use full transductive topology, as disclosed in the audit.

Preparation resumes completed stages and periodically saves ego rows and subgraph trees when rerun with the same settings. A global tree interrupted before completion must restart its optimization. `--force` deliberately rebuilds the derived stages. Keep the cache directory when moving between cloud sessions.

On the 6GB GPU, the model keeps 32-dimensional SeBot views and 16-dimensional structural levels. Cross-node attention chunks query rows while keeping **all users** as keys/values; NT-Xent is computed in exact query chunks and subgraph activations are recomputed during backward. If memory is tight, reduce `attention_query_chunk`, `contrastive_chunk` or `subgraph_batch` in the config. Those change execution chunk size, not the negatives, sampled graph or model dimensions. Do not reduce node coverage/attention scope to claim the same baseline. If it still does not fit, use Kaggle or CPU.

Resume interrupted training with the latest optimizer/RNG state:

```powershell
.\.venv\Scripts\python.exe -m sebot_mls.cli train --config configs/mgtab.yaml --resume
```

## Outputs and demonstration

`outputs/mgtab/` contains:

- `best.pt`: model, optimizer, selected epoch, config, split, scaler and dataset identity.
- `latest.pt`: last completed epoch, optimizer/RNG and early-stopping state for resumption.
- `history.csv`: training components and validation metrics; no test-based selection.
- `test_metrics.json`: accuracy, bot precision/recall/F1, macro F1, ROC-AUC and confusion matrix.
- `predictions.csv`: test node IDs, labels, probabilities and four fusion weights.
- `config.json`, `run.json`: actual settings, software/device, duration and peak GPU memory.

The evaluation command loads the selected checkpoint's architecture/config and checks it against the prepared data. Classification uses both logits with softmax; no test-threshold tuning. You can demonstrate a trained checkpoint locally without retraining:

```powershell
.\.venv\Scripts\python.exe -m sebot_mls.cli evaluate --config configs/mgtab.yaml --checkpoint outputs/mgtab/best.pt --device cpu
```

## Project layout

```text
configs/             MGTAB and synthetic smoke settings
src/sebot_mls/
  data.py            checked tensor loading, splits, scaling, relation projection
  features.py        four MLS feature groups and exact graphlets
  trees.py           original SeBot structural entropy tree preparation
  models.py          SeBot branch, H-GNN, A-Fusion, classifier, exact NT-Xent
  engine.py          preprocessing, train/validation, checkpoint/evaluate
  cli.py             commands and acquisition
  legacy/            reviewed compatibility copies of upstream SeBot modules
tests/               independent graphlet, leakage, loss-gradient and dimension checks
scripts/             setup and tool builds
notebooks/           Colab / Kaggle portable entry points
vendor/              preserved upstream sources and licenses
docs/                specification, platform decision, validation record
```

## Move to another machine

Use the supplied project ZIP, or copy this folder **excluding `.venv`, transient caches and vendor `.git` folders**. Recreate the environment on the target machine. To move a trained demo, additionally copy `data/MGTAB`, the prepared cache and `outputs/mgtab`. Preserve the dataset license and keep it private; the source ZIP does not redistribute MGTAB. The synced SNA reference files were not edited.

Third-party provenance and licenses: [NOTICE](NOTICE). Rebuild compatibility modules only after reviewing upstream changes: `python scripts/prepare_legacy.py`.
# Self-contained Kaggle notebook

The full MGTAB run completed on 7 October 2026: 91.47% test accuracy and 84.04% bot F1. See [the measured results and limitations](docs/MGTAB_RESULTS.md). Training stopped at epoch 187; evaluation used the best validation checkpoint from epoch 137.

The separate trained-demo ZIP includes this source project plus `SNA-SeBot-MLS/` with the downloaded tensors, prepared cache and checkpoints. Open its `sebot-mls-baseline` folder in VS Code, follow the Windows environment setup above, and run:

```powershell
.\.venv\Scripts\python.exe -m sebot_mls.cli evaluate --config configs/trained-demo.yaml --checkpoint SNA-SeBot-MLS/outputs/best.pt --device cuda
```

Use `--device cpu` when CUDA is unavailable. This command evaluates the already trained checkpoint and saves a separate `outputs/demo-evaluation` report. It does not repeat preprocessing or training. Keep the dataset-containing demo package private.

Import `notebooks/kaggle_standalone.ipynb` using **File → Import Notebook** in Kaggle.
It includes the modular source snapshot and installs the verified dependencies in an isolated Python 3.12 environment.
Start with `STAGE = 'smoke'`, then run `prepare` and save its outputs. Attach that saved notebook output in the next run and set `RESTORE_FROM` to its `SNA-SeBot-MLS` folder. Use `benchmark` for a separate ten-epoch timing run or `train` for the baseline's 200-epoch maximum. Select a GPU for training; enable Internet for setup/downloads. Save & Run All runs as a server-side batch job. The notebook uses one GPU.

The synthetic smoke stage verifies software only. Full MGTAB results must come from the `train` and `evaluate` stages. The standalone workflow is an engineering addition recorded as E17 in `docs/PHASE2_LEDGER.md`.

