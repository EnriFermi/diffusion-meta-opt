# Repository Agent Rules

## Verbose Experiment Runs

When adding or changing a long-running training, evaluation, notebook, or research pipeline, include clear runtime visibility by default.

- Print or log the resolved config, device, dtype, seed, cache mode, and main artifact output directory at startup.
- Show the current pipeline stage, for example data loading, cache hit/miss, model build, training, geometry evaluation, downstream evaluation, and output writing.
- Add progress bars or periodic logs for loops that can take more than a few seconds.
- Progress output must include enough context to identify what is running: experiment label, seed, method, LR, batch/step counts, current loss or metric, and elapsed/rate when practical.
- On cache hits, print or log which files were reused instead of silently skipping work.
- At the end, print or log the written artifact paths and the most important summary metrics.
- Keep verbosity configurable, but default it to enabled for research notebooks and launcher scripts.

