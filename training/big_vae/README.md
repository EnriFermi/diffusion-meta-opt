# BigVAE Training Package

The active training implementation is split here by responsibility:

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
- `grad_stats.py`, `grad_plots.py`, and `collector_monitoring.py`: gradient
  telemetry helpers.
