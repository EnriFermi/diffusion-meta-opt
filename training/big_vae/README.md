# BigVAE Training Package

`experiments/train_big_vae.py` is now a compatibility entrypoint. The actual
implementation is split here by responsibility:

- `launcher.py`: Hydra entrypoint, artifact setup, distributed process launch.
- `worker.py`: the stateful training worker loop.
- `runtime.py`: config normalization and model config construction.
- `data.py`: public data helpers re-exported from focused modules.
- `source_sampling.py`, `source_pool.py`, `source_batching.py`: source sample
  collection, filtering, and batch assembly.
- `batch_padding.py`, `batch_debug.py`, `presliced.py`, `data_types.py`:
  batch shaping and offline/presliced dataset helpers.
- `checkpointing.py`: checkpoint and resume-state helpers.
- `tracking.py`: logging, telemetry, and scalar helpers.
- `grad_monitoring.py`: compatibility re-export for gradient monitoring,
  backed by `grad_stats.py`, `grad_plots.py`, and `collector_monitoring.py`.

Keep old imports through `experiments.train_big_vae` working unless all callers
are migrated.
