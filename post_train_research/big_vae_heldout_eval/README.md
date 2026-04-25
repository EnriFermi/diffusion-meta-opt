# BigVAE Held-Out Evaluation

This directory contains a two-stage post-train pipeline:

1. `build_heldout_offline_dataset.py` collects a reusable offline dataset from dataset/model pairs that were not in the BigVAE train profile.
2. `evaluate_big_vae_heldout.py` loads a trained BigVAE checkpoint and evaluates all BigVAE loss components on the prebuilt dataset.

Held-out pairs:

```text
chexpert: vit_large_p16_224, clip_vit_l14
flickr30k: clip_vit_l14, vit_base_p16_224
food101: siglip_so400m_p14_384, vit_large_p16_224
openimages_v7: detr_resnet50, clip_vit_l14
pascal_voc_2012: segformer_b5_cityscapes, detr_resnet50
rvl_cdip: donut_rvlcdip, trocr_large_printed
sun397: vit_large_p16_224, clip_vit_l14
```

Build once:

```bash
HELDOUT_ROOT=post_train_research/big_vae_heldout_eval/artifacts/offline_dataset \
HELDOUT_LOG_DIR=post_train_research/big_vae_heldout_eval/artifacts/logs \
HELDOUT_RECORDS_PER_PAIR=1024 \
HELDOUT_TARGET_SIZE_GB=20 \
HELDOUT_OVERWRITE=true \
HELDOUT_SAMPLE_WAIT_TIMEOUT_S=1800 \
post_train_research/big_vae_heldout_eval/run_build_heldout_offline_dataset.sh
```

Evaluate any checkpoint:

```bash
BIG_VAE_CHECKPOINT=artifacts/training/checkpoints/weight_quantile_vae/stage_1/latest.pt \
HELDOUT_ROOT=post_train_research/big_vae_heldout_eval/artifacts/offline_dataset \
HELDOUT_LOG_DIR=post_train_research/big_vae_heldout_eval/artifacts/logs \
post_train_research/big_vae_heldout_eval/run_evaluate_big_vae_heldout.sh
```

Run logs are written to `HELDOUT_LOG_DIR` by default:

```text
post_train_research/big_vae_heldout_eval/artifacts/logs/build_big_vae_heldout_offline_dataset_rank0.log
post_train_research/big_vae_heldout_eval/artifacts/logs/evaluate_big_vae_heldout_rank0.log
```

During dataset build, waiting for a new collector sample is logged every
`HELDOUT_SAMPLE_WAIT_STATUS_EVERY_S` seconds and fails after
`HELDOUT_SAMPLE_WAIT_TIMEOUT_S` seconds by default. Set
`HELDOUT_FAIL_ON_SAMPLE_WAIT_TIMEOUT=false` only if you want a partial dataset
instead of a failing build.

Main outputs:

```text
metrics_summary.json
coverage.json
metrics_global.json
metrics_macro.json
metrics_by_model.csv
metrics_by_dataset.csv
metrics_by_dataset_model_pair.csv
metrics_by_layer.csv
record_metrics.csv
latent_dump.pt
latent_dump.metadata.csv
latent_plots/latent_embedding_pca.csv
latent_plots/latent_counts_by_<field>.csv
latent_plots/latent_pca_by_dataset.png
latent_plots/latent_pca_by_model.png
latent_plots/latent_pca_by_layer_type.png
latent_plots/latent_pca_by_depth_label.png
latent_plots/latent_embedding_tsne.csv
latent_plots/latent_tsne_by_dataset.png
latent_plots/latent_tsne_by_model.png
latent_plots/latent_tsne_by_layer_type.png
latent_plots/latent_tsne_by_depth_label.png
```

`metrics_summary.json` contains micro-global metrics plus macro averages over models, datasets, and dataset/model pairs. `record_metrics.csv` is intentionally verbose for outlier debugging.

Latent dump/plots:

```bash
EVAL_LATENT_DUMP_ENABLED=true \
EVAL_LATENT_DUMP_MAX_ENTRIES=256 \
EVAL_LATENT_DUMP_MAX_SLICES_PER_SOURCE=1 \
EVAL_LATENT_DUMP_BALANCE_ENABLED=true \
EVAL_LATENT_DUMP_BALANCE_KEYS=dataset,model,layer_type,depth_label \
EVAL_LATENT_DUMP_MAX_PER_GROUP=1 \
EVAL_LATENT_PLOT_ENABLED=true \
EVAL_LATENT_PLOT_TSNE=true \
post_train_research/big_vae_heldout_eval/run_evaluate_big_vae_heldout.sh
```

The dump stores one latent vector per evaluated slice, capped by
`EVAL_LATENT_DUMP_MAX_ENTRIES`. For VAE checkpoints this is posterior `mu`; for
non-sampling checkpoints this is the deterministic base latent. Metadata
columns include `dataset`, `model`, `layer`, `layer_type`, and `depth_label`.
By default the dump keeps only one slice per source record, so a large batch
from one layer cannot fill the whole latent sample. It also balances candidates
over the joint key `(dataset, model, layer_type, depth_label)` and then applies a
final greedy marginal balancing pass if there are more candidate groups than
`EVAL_LATENT_DUMP_MAX_ENTRIES`. Rebuild the PCA/t-SNE plots without rerunning
held-out eval:

```bash
python post_train_research/big_vae_heldout_eval/plot_latent_dump.py \
  "$HELDOUT_ROOT/eval/<checkpoint_dir>_<checkpoint_name>/latent_dump.pt"
```

Coverage checks:

```bash
cat "$HELDOUT_ROOT/eval/<checkpoint_dir>_<checkpoint_name>/coverage.json"
```

`coverage.ok=true` means all expected held-out datasets, models, and dataset/model pairs produced finite metrics and no eval samples were skipped. By default `run_evaluate_big_vae_heldout.sh` sets `EVAL_REQUIRE_FULL_COVERAGE=true`, so the eval process exits non-zero if coverage fails. For smoke runs with `EVAL_MAX_RECORDS>0`, set `EVAL_REQUIRE_FULL_COVERAGE=false`.

Post-facto coverage check for runs started before `coverage.json` existed:

```bash
python post_train_research/big_vae_heldout_eval/check_heldout_coverage.py \
  --heldout-root "$HELDOUT_ROOT" \
  --latest-eval
```

Or point at a concrete eval output directory:

```bash
python post_train_research/big_vae_heldout_eval/check_heldout_coverage.py \
  --eval-dir "$HELDOUT_ROOT/eval/<checkpoint_dir>_<checkpoint_name>"
```

Build dataset coverage can be checked from the already written build summary:

```bash
python post_train_research/big_vae_heldout_eval/check_heldout_coverage.py \
  --heldout-root "$HELDOUT_ROOT" \
  --build
```

If build stalls or the async collector exits, inspect the build failure context:

```bash
post_train_research/big_vae_heldout_eval/run_diagnose_heldout_build_failure.sh
```

This reads the latest heldout build log, collector crash report, dataset worker
status files, and recent fatal reports. It also writes a JSON copy to
`post_train_research/big_vae_heldout_eval/artifacts/heldout_diag.json` by
default. Override that path with `HELDOUT_DIAG_JSON_OUT=/tmp/heldout_diag.json`
if needed.
