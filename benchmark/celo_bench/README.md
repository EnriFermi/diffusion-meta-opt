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
