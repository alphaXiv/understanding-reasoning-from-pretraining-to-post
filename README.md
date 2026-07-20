# Reproducing pretraining-to-RL scaling in chess

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/understanding-reasoning-from-pretraining-to-post/blob/main/notebooks/pretraining_to_rl.py)

This repository reproduces two claims from [*Understanding Reasoning from Pretraining to Post-Training* (arXiv:2607.16097)](https://arxiv.org/abs/2607.16097): at fixed model size and matched post-training, a longer-pretrained checkpoint improves faster under RL, and pretraining validation loss ranks post-RL pass@1.

**Assessment: reproduced at reduced scale.** Eight paired seeds of the same 20.54M-parameter chess model saw either 9.18M or 36.70M nested public Lichess tokens, then the same 120-step verified-trace SFT and 60-update GRPO schedule. Mean RL slope increased from **0.03395 to 0.05959 pass@1 per log10 update** (paired difference +0.02564, 95% CI +0.00140 to +0.04989). Mean pass@1 at update 60 increased from **8.20% to 11.95%**. The long arm had lower pretraining loss in 8/8 seed pairs and higher aggregate pass@1 at every matched update.

The paper reports a slope–token Pearson correlation of **+0.84** across its multi-size, multi-checkpoint sweep and pretraining-loss/post-RL Spearman magnitudes of **0.93–0.99**. This reproduction tests the same directions with two exposure levels rather than re-estimating those multi-point correlations.

The substitutions are explicit: 9.18M versus 36.70M tokens instead of the paper's roughly 200M–52B-token sweep; 160 held-out puzzles instead of the full benchmark; 60 GRPO updates instead of 1,000–5,000; context 256; and eight independent single-GPU replicas per arm because this cluster's NCCL collective path faulted on the Blackwell nodes. A three-epoch-SFT robustness round at 9.18M/18.35M/36.70M tokens preserved the mean loss and slope orderings, though the midpoint endpoint was essentially tied with the short arm. Training used Kubernetes, NVIDIA RTX PRO 6000 Blackwell GPUs, a peak of 16 GPUs concurrently, and 0.698 hours of elapsed campaign wall time through the final evidence run.

- [Detailed visual report](reports/pretraining-to-rl/report.md)
- [Self-contained tutorial notebook](notebooks/pretraining_to_rl.py)
- [Measured seed-level results](reports/pretraining-to-rl/seed_summary.csv)
- [Mean trajectories used in the headline figure](reports/pretraining-to-rl/trajectory_summary.csv)

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, configs, and implementation | Not run as an experiment (publication surface) | Presentation-only | No experiment allocation |
| [`orx/validated-end-to-end-scout`](https://github.com/alphaXiv/understanding-reasoning-from-pretraining-to-post/tree/orx/validated-end-to-end-scout) | Combined launcher, puzzle-state, and CUDA-device fixes | `bash run.sh` | Failed before metrics: reproducible NCCL illegal-memory fault motivated independent replicas | Kubernetes; 8× NVIDIA RTX PRO 6000 Blackwell; 6m07s |
| [`orx/independent-gpu-seed-ensemble`](https://github.com/alphaXiv/understanding-reasoning-from-pretraining-to-post/tree/orx/independent-gpu-seed-ensemble) | End-to-end one-shard validation with eight independent seeds | `bash run.sh` | Successful; eight terminal result records | Kubernetes; 8× NVIDIA RTX PRO 6000 Blackwell; 42s |
| [`orx/formal-short-8-shards`](https://github.com/alphaXiv/understanding-reasoning-from-pretraining-to-post/tree/orx/formal-short-8-shards) | Primary 9.18M-token arm | `bash run.sh` | Mean loss 1.03150; pass@1 3.98%→8.20%; slope 0.03395 | Kubernetes; 8× NVIDIA RTX PRO 6000 Blackwell; 1m35s |
| [`orx/formal-long-32-shards`](https://github.com/alphaXiv/understanding-reasoning-from-pretraining-to-post/tree/orx/formal-long-32-shards) | Primary 36.70M-token arm | `bash run.sh` | Mean loss 0.87191; pass@1 5.70%→11.95%; slope 0.05959 | Kubernetes; 8× NVIDIA RTX PRO 6000 Blackwell; 3m11s |
| [`orx/matched-short-exposure-8-replicas`](https://github.com/alphaXiv/understanding-reasoning-from-pretraining-to-post/tree/orx/matched-short-exposure-8-replicas) | Three-epoch-SFT 9.18M-token robustness arm | `bash run.sh` | Mean loss 1.02863; final pass@1 21.33%; slope 0.10178 | Kubernetes; 8× NVIDIA RTX PRO 6000 Blackwell; 3m00s |
| [`orx/matched-midpoint-exposure-8-replicas`](https://github.com/alphaXiv/understanding-reasoning-from-pretraining-to-post/tree/orx/matched-midpoint-exposure-8-replicas) | Three-epoch-SFT 18.35M-token midpoint | `bash run.sh` | Mean loss 0.94129; final pass@1 21.41%; slope 0.10681 | Kubernetes; 8× NVIDIA RTX PRO 6000 Blackwell; 2m13s |
| [`orx/matched-long-exposure-8-replicas`](https://github.com/alphaXiv/understanding-reasoning-from-pretraining-to-post/tree/orx/matched-long-exposure-8-replicas) | Three-epoch-SFT 36.70M-token robustness arm | `bash run.sh` | Mean loss 0.87279; final pass@1 27.19%; slope 0.13810 | Kubernetes; 8× NVIDIA RTX PRO 6000 Blackwell; 6m29s |

## Running the code

The frozen configs are in `reproduction/configs/`. The recorded experiment branches all use the exact command `bash run.sh`; to rerun one, check out that branch so its committed `reproduction/config.json` is selected:

```bash
git switch orx/formal-short-8-shards
bash run.sh

git switch orx/formal-long-32-shards
bash run.sh
```

The script downloads public Hugging Face inputs at runtime and does not redistribute Lichess games or puzzle data. Each 8-GPU job emits one `RESULT_JSON` terminal record per independent seed.
