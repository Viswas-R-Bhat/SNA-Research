# Platform decision (6 October 2026)

**Author and debug locally; use Kaggle as the preferred optional cloud training platform. Keep Colab as a convenient alternative.** One Python package is used everywhere; notebooks only install it, set paths, and call the same commands. There are no cloud-specific model variants.

| Platform | Advantages for this baseline | Constraints | Role |
|---|---|---|---|
| Windows / VS Code | Persistent storage, full project editing, no hosted-session deadline; actual college demonstration hardware | Detected RTX 3050 6GB, 16GB system RAM, i7-13620H. Full attention and original trees need memory care. | Main source code, tests, debugging, final demo |
| Kaggle | Documented ~29GB host RAM for GPU sessions; free GPU options; saved versions/outputs; batch Save & Run All | Documentation lists 12-hour CPU/GPU sessions, 20GB auto-saved working output, GPU quota/availability. Input datasets are read-only; use `/kaggle/working`. Two T4 cards do not automatically pool memory for this single-GPU implementation. | Preferred optional full training and CPU preprocessing if local RAM is insufficient |
| Colab | Quick interactive notebooks, Google Drive mounting, easy sharing; paid plans can offer more resources | GPU models, quotas, RAM and runtime availability vary; free sessions at most 12 hours and can stop earlier. Paid compute is not assumed. | Interactive fallback and troubleshooting |

This recommendation is an engineering inference from the documented resources, not a measured cloud speed comparison. Faster GPU hardware alone does not accelerate NetworkX graph extraction. Use the compiled ORCA counter for exact graphlets. Cache raw features and trees once, then train repeatedly from those caches. Never recompute features on a sampled graph just to fit cloud time limits and describe that as the paper baseline.

Preprocessing and training should be separate runs. For Kaggle, attach the project ZIP and official tensors (or a prepared bundle) as private inputs, write derived artifacts in working, and use Save & Run All for unattended runs. For Colab, persist data/cache/output directories in Drive and install the source on the runtime disk. Checkpoints are device-independent, and evaluation can use `--device cpu` after downloading the output.

Cloud notebooks have been syntax-checked locally; their hosted environments and quotas are not yet exercised in the user's accounts. No cloud session or paid subscription has been created.

Sources checked:
- Kaggle notebook technical specifications, versions, input/output and environment: https://www.kaggle.com/docs/notebooks
- Kaggle GPU quota guidance: https://www.kaggle.com/docs/efficient-gpu-usage
- Colab resource policy: https://research.google.com/colaboratory/faq.html
- ORCA exact graphlet counter: https://github.com/thocevar/orca
