# MGTAB baseline run — 7 October 2026

Full training and held-out evaluation completed successfully on Kaggle, version 5:
https://www.kaggle.com/code/viswasrbhat/sebot-mls-mgtab-baseline?scriptVersionId=356065698

Maximum 200 epochs; early stopping ended training at epoch 187. The lowest validation loss selected epoch 137 (0.2419154644). The test set contains 1,020 users; the saved split is 8,159 train / 1,020 validation / 1,020 test. Test labels were not used for checkpoint selection.

| Metric | This run | RP1 reference |
|---|---:|---:|
| Accuracy | 91.47% | 91.76% |
| Bot F1 | 84.04% | 84.62% |
| Bot recall | 84.81% | 85.56% |
| Bot precision | 83.27% | 83.70% |
| Macro F1 | 89.11% | — |
| ROC AUC | 0.96520 | — |

Confusion matrix (rows true human/bot, columns predicted human/bot): `[[704, 46], [41, 229]]`.

Training took 1,502.94 seconds (25 minutes 3 seconds) on one Tesla T4, with peak allocated CUDA memory 2,406.85 MB. Total notebook runtime was approximately 27 minutes 31 seconds, including environment setup, cache loading, evaluation and output saving. Preprocessing was a separate completed CPU job taking 8 hours 46 minutes. The timing benchmark took 85.90 seconds for ten epochs.

Dataset fingerprint: `daddd8c9835d60349c85df2e5c80601fd6c8c628952b88b88b21d288d7d9a288`.
Embedded source snapshot: `abc7630cb1315b8e2e62b2d9dd5688e620244b2c8892b328f9a22baf0888e62e`.

This is one seed of the documented reimplementation, not proof of exact numerical reproduction. Paper-unspecified conventions and the interaction-frequency fallback remain as disclosed in SPECIFICATION.md and PHASE2_LEDGER.md. The frequency fallback is a material limitation. No settings were tuned against this test result. TwiBot-20 has not been trained; its paper results must not be presented as this project's results.

Saved artifacts include best/latest checkpoints, history.csv, predictions.csv, config.json, run.json, test_metrics.json, the complete prepared cache, executed notebook, and train-results.zip. A private trained-demo package can retain the downloaded MGTAB files for the user's local demonstration; do not publish or redistribute that package without checking dataset permissions. Packaging is covered by ledger E16/E17 and introduces no model change.

Windows local verification: the downloaded epoch-137 checkpoint evaluated successfully on NVIDIA GeForce RTX 3050 6GB Laptop GPU. All reported test metrics and the confusion matrix exactly matched the saved Kaggle report. The trained-demo config writes local evaluation separately and does not modify the original cloud outputs.
