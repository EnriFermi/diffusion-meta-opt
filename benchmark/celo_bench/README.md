# Celo Benchmark

This package runs the 17-task Celo/VeLOdrome benchmark from the Celo paper with
the original JAX `learned_optimization` task constructors.

It is intentionally separate from BigVAE code. The runner consumes optimizer
adapters that produce learned-optimization-style optimizers; future
metaoptimizers can be added by import path without changing the task registry.

Install the optional environment separately:

```sh
pip install -r benchmark/celo_bench/requirements-celo.txt
```

The upstream `google-research/vision_transformer` repository installs the
package as `vit_jax`; this benchmark provides a compatibility alias for the
older `vision_transformer.vit_jax` import used by `amoudgl/learned_optimization`.

Run a dry run:

```sh
scripts/launchers/benchmark/celo_bench.sh benchmark.dry_run=true
```

Run with a learned optimizer checkpoint:

```sh
scripts/launchers/benchmark/celo_bench.sh \
  methods.items='[{name: celo, kind: celo_factory, optimizer_name: celo, checkpoint_path: /path/to/theta.state}]'
```

Run the AdamW smoke config:

```sh
scripts/launchers/benchmark/celo_bench.sh --config-name adamw_smoke
```

Run the full 17-task AdamW benchmark:

```sh
scripts/launchers/benchmark/celo_bench.sh --config-name adamw_full
```

`adamw_full` keeps the exact 17-task list, runs the fragile Wikipedia task
first, and has `benchmark.continue_on_task_error=true` so a dead external TFDS
URL is recorded in `skipped.csv` instead of aborting the whole run.

Run all currently downloadable tasks except the old Wikipedia snapshot:

```sh
scripts/launchers/benchmark/celo_bench.sh --config-name adamw_full_no_wikipedia
```

Artifacts are written to:

```text
artifacts/celo_bench/runs/<timestamp>__<label>/
```

The default task preset is `celo_paper_17`, with `steps=2000`,
`seeds=[0,1,2]`, `eval_every=10`, `eval_batches=5`, `last_eval_batches=10`,
and `metrics_every=10`.

TensorFlow GPU visibility is disabled by default via
`runtime.tensorflow_hide_gpus=true`, matching upstream Celo evaluation. TFDS
data preprocessing then stays on CPU while JAX can use the GPU.

The exact 17-task set includes `RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128`,
which uses TFDS `wikipedia/20201201.en` from the upstream
`learned_optimization` code. TFDS no longer serves the raw 2020 Wikimedia dump
from its old mirror, so `runtime.tfds_try_gcs_for_wikipedia=true` patches only
that dataset load to use TFDS' public prepared GCS cache for the same
`wikipedia/20201201.en/1.0.0` snapshot. This avoids substituting a newer
Wikipedia dump. If the GCS cache/network is unavailable, provide a prebuilt TFDS
cache via `TFDS_DATA_DIR` or run `--config-name adamw_full_no_wikipedia`. The
`adamw_full` config keeps all 17 tasks but schedules the Wikipedia task first,
so a remaining failure is recorded in `skipped.csv` and the rest of the run
continues.

Curve-level caching is enabled when `benchmark.cache_first=true`. If a long run
fails, rerun with the same `benchmark.run_dir=/path/to/failed/run` to reuse
already written raw `.npz` curves.
