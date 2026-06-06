# BigVAE Eval Suite

One-command post-train evaluation for a BigVAE checkpoint.

The suite uses a single Hydra config:

```text
conf/big_vae_eval_suite/config.yaml
```

It writes one top-level run directory:

```text
artifacts/big_vae/eval/suite/runs/<timestamp>__<label>/
  config_resolved.json
  suite_context.json
  suite.log
  summary.json
  heldout_eval/
  scaling_check/
  landscape_ablation/
  logs/
```

Run:

```sh
BIG_VAE_CHECKPOINT=/path/to/big_vae/latest.pt \
LATENT_DIFFUSION_PRIOR_CHECKPOINT=/path/to/prior/latest.pt \
post_train_research/big_vae_eval_suite/run_big_vae_eval_suite.sh
```

Use `DRY_RUN=true` to write the planned commands without running the stages.
Dry-run still requires required config fields to be non-empty, but it does not
require future checkpoint paths to exist.

Stage toggles:

```sh
post_train_research/big_vae_eval_suite/run_big_vae_eval_suite.sh \
  stages.heldout_eval.enabled=true \
  stages.scaling_check.enabled=true \
  stages.landscape_ablation.enabled=false
```

Scaling profiles are configured under `stages.scaling_check.jobs`. Each job can
override `profile`, `max_steps`, `lr`, `batch_size`, `eval_batch_size`,
`train_subset`, `test_subset`, and whether to run latent/raw variants.

If the stages need different environments, keep the top-level launcher in any
environment that can run Hydra and set per-stage conda envs:

```sh
HELDOUT_CONDA_ENV=onerec \
SCALING_CONDA_ENV=diff-meta-opt312 \
LANDSCAPE_CONDA_ENV=diff-meta-opt312 \
post_train_research/big_vae_eval_suite/run_big_vae_eval_suite.sh
```

The same knobs are available in `conf/big_vae_eval_suite/config.yaml` as
`suite.conda_env` and `stages.<stage>.conda_env`.
