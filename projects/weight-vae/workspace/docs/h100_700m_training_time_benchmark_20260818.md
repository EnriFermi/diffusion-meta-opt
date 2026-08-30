# H100 700M training-time benchmark (2026-08-18)

## Hardware

- GPU: NVIDIA H100 NVL, 95,830 MiB
- CPU: 32 logical cores
- GPU was idle before the benchmark

## Synthetic compute-only benchmark

The benchmark used a 737,356,800-parameter Transformer core intended only to
approximate the parameter scale of the proposed WeightCLIP-sized model:

- width: 1600
- 24 Transformer blocks total
- sequence length: 512
- BF16
- flash scaled-dot-product attention
- fused AdamW
- complete training step: forward, backward, optimizer update

Results:

| Batch | Seconds / optimizer step | Samples / second | Peak GPU memory | 500k steps |
|---:|---:|---:|---:|---:|
| 16 | 0.1132 | 141.4 | 13.97 GiB | 15.7 h / 0.65 d |
| 32 | 0.2163 | 147.9 | 23.55 GiB | 30.0 h / 1.25 d |

These numbers are an optimistic compute floor, not a production ETA. They omit
the actual BigVAE encoder/decoder topology, activation-context construction,
behavioral and structural loss work, data loading, validation, checkpointing,
and expensive gradient diagnostics.

## Evidence from existing BigVAE runs

Existing approximately 20M-parameter runs commonly show roughly 1--3.4
optimizer steps/s during normal regions, with much slower logged intervals when
data waits or frequent per-layer gradient monitoring occur:

- `projects/shared/storage/artifacts/training/runs/train_big_vae_20260401_190025_pid516991_f3762e/logs/train_rank0.log`
- `projects/shared/storage/artifacts/training/runs/train_big_vae_20260503_230432_pid3605482_146b9a/logs/train_rank0.log`

Relevant existing checkpoints:

- `projects/shared/storage/artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt`: 15,908,896 parameters, step 480k
- `projects/shared/storage/artifacts/training/checkpoints/weight_quantile_vae/stage_1/latest.pt`: 19,609,376 parameters, step 440k

## Planning range

For 500,000 optimizer steps:

| Sustained seconds / step | Wall time |
|---:|---:|
| 0.2 | 27.8 h / 1.16 d |
| 0.5 | 69.4 h / 2.89 d |
| 1.0 | 138.9 h / 5.79 d |
| 2.0 | 277.8 h / 11.57 d |

Current planning estimate for a properly optimized offline pipeline is 1.5--3
days. A conservative production interval including real losses, evaluation,
and checkpoints is 3--7 days. Leaving frequent full per-layer gradient
monitoring or a stalling input pipeline enabled can extend this to 7--14+ days.

Before scheduling the production run, instantiate the exact proposed model and
run 100--500 production-shaped optimizer steps. Report both optimizer steps and
the total number of training examples/tokens: at batch 16, 500k steps are 8M
examples; at batch 32 they are 16M examples, so a step-only budget is not a fair
compute or data budget.
