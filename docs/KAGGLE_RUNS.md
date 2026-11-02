# Kaggle execution record — 6–7 October 2026

Notebook: https://www.kaggle.com/code/viswasrbhat/sebot-mls-mgtab-baseline

Privacy was verified as Private. Source snapshot embedded in the submitted versions:
`abc7630cb1315b8e2e62b2d9dd5688e620244b2c8892b328f9a22baf0888e62e`.
The source is the modular RP1-traced reimplementation with the assumptions and engineering additions documented in the embedded specification and Phase 2 ledger.

## Version 1: Synthetic GPU verification

https://www.kaggle.com/code/viswasrbhat/sebot-mls-mgtab-baseline?scriptVersionId=355677945

- Save & Run All batch: successful, 292-second total job duration including setup.
- T4 x2 allocated; GPU 0 used (Tesla T4, 14.56 GiB accessible memory).
- Isolated Python 3.12.3, PyTorch 2.5.1+cu121; CUDA backward diagnostic passed.
- 16 tests passed in 7.23 seconds; Linux ORCA built. The optional test's Windows ORCA path was absent, so cloud ORCA output agreement is not established by this run.
- Forty synthetic users: complete feature/tree preparation, three training epochs, best/latest checkpoints, held-out evaluation and output ZIP saved.
- These are software checks, not MGTAB accuracy or runtime estimates. Full-size training duration remains unmeasured.
- The executed notebook was downloaded to `../output/kaggle-verified-smoke.ipynb` relative to the project root's parent.

## Version 2: Full MGTAB preprocessing

https://www.kaggle.com/code/viswasrbhat/sebot-mls-mgtab-baseline?scriptVersionId=355679359

Save & Run All batch submitted with accelerator None to use CPU resources without GPU quota consumption. Completion verified successful on 7 October: total job duration 31,560 seconds (8 hours 46 minutes), saved output 220.36 MB. This version downloaded the official MGTAB tensors and prepared structural features and entropy trees. Local two-hop ego calculations finished at 20,210.9 seconds; global features finished at 20,490.8 seconds; tree preparation finished around 31,540 seconds. The final cache is `/kaggle/working/SNA-SeBot-MLS/processed/prepared.pt`. It did not train or evaluate the MGTAB classifier.

After successful preparation, attach this version's output, set `RESTORE_FROM` to its `SNA-SeBot-MLS` directory, select T4 GPUs, and run `benchmark` in a separate output folder before `train`. Keep the prepared cache and split unchanged. If preparation reaches the session deadline, retain completed stage outputs and restore them to resume; an unfinished global-tree optimization restarts.

## Version 3: Ten-epoch MGTAB GPU benchmark

https://www.kaggle.com/code/viswasrbhat/sebot-mls-mgtab-baseline?scriptVersionId=356063931

Submitted with Save & Run All on 7 October 2026 at 17:54 IST, version name `MGTAB GPU benchmark - 10 epochs`. Attached version 2 notebook output (source ID 355679359). Controls: `STAGE = 'benchmark'`, `RESTORE_FROM = '/kaggle/input/notebooks/viswasrbhat/sebot-mls-mgtab-baseline/SNA-SeBot-MLS'`. T4 x2 allocated; existing code uses GPU 0. Runtime diagnostic confirmed CUDA available, Tesla T4 with 14.56 GiB accessible memory, and a successful CUDA backward pass. Configuration confirms 10 epochs and the saved 80/10/10 split. Epochs 1 and 2 completed at 130.0 and 138.4 seconds of job time. Status last checked running; completion and final benchmark report remain pending. Benchmark outputs are separate from the full baseline training folder; the baseline maximum remains 200 epochs.

Completion verified: Successfully ran in 215.9 seconds total (3 minutes 36 seconds), including environment setup and output saving. All 10 epochs completed. Training report: 85.9001 seconds, CUDA Tesla T4, peak allocated CUDA memory 2,403.21 MB. Steady epochs took roughly 8.4–8.7 seconds; linear projection to 200 epochs is 28.63 minutes of training, with separate setup/evaluation overhead and possible early stopping. Epoch 10 validation bot F1 was 0.5682; this is not a held-out test result or a reproduction of the paper's final metrics. Saved best/latest benchmark checkpoints and benchmark-results.zip. Screenshot evidence: output/kaggle-benchmark-status.jpg in the workspace parent. Full baseline training and held-out evaluation remain pending.

## Version 5: Full MGTAB training and evaluation

https://www.kaggle.com/code/viswasrbhat/sebot-mls-mgtab-baseline?scriptVersionId=356065698

Submitted 7 October 2026 at 18:01 IST with stage train, version 2 preprocessing input, GPU T4 x2 (GPU 0 used), maximum 200 epochs, original validation-loss early stopping. Runtime logs confirmed CUDA available and epochs: 200. Full run starts from scratch in outputs, separate from benchmark. Held-out evaluation runs after training, then train-results.zip is saved. Version 4 was canceled because the editor inserted new controls ahead of old benchmark controls; the duplicate was cleared and verified before version 5 submission. Completion and final metrics pending.

Version 5 completion verified successful. Early stopping ended at epoch 187; best validation checkpoint epoch 137. Training 1502.94 seconds; peak allocated CUDA memory 2406.85 MB. Test accuracy 91.4706%, bot F1 84.0367%, recall 84.8148%, precision 83.2727%; 1,020 test users. All output downloaded and extracted to workspace output/kaggle-full-run. Executed notebook preserved as output/kaggle-full-run.ipynb. See MGTAB_RESULTS.md for interpretation and paper comparison.
