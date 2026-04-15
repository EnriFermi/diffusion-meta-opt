# BigVAE Held-Out Evaluation

This directory contains a two-stage post-train pipeline:

1. `build_heldout_offline_dataset.py` collects a reusable offline dataset from dataset/model pairs that were not in the BigVAE train profile.
2. `evaluate_big_vae_heldout.py` loads a trained BigVAE checkpoint and evaluates all BigVAE loss components on the prebuilt dataset.

Held-out pairs:

```text
bigearthnet: vit_large_p16_224, siglip_so400m_p14_384
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
HELDOUT_RECORDS_PER_PAIR=1024 \
HELDOUT_TARGET_SIZE_GB=20 \
HELDOUT_OVERWRITE=true \
post_train_research/big_vae_heldout_eval/run_build_heldout_offline_dataset.sh
```

Evaluate any checkpoint:

```bash
BIG_VAE_CHECKPOINT=artifacts/training/checkpoints/weight_quantile_vae/stage_1/latest.pt \
HELDOUT_ROOT=post_train_research/big_vae_heldout_eval/artifacts/offline_dataset \
post_train_research/big_vae_heldout_eval/run_evaluate_big_vae_heldout.sh
```

Main outputs:

```text
metrics_summary.json
metrics_global.json
metrics_macro.json
metrics_by_model.csv
metrics_by_dataset.csv
metrics_by_dataset_model_pair.csv
metrics_by_layer.csv
record_metrics.csv
```

`metrics_summary.json` contains micro-global metrics plus macro averages over models, datasets, and dataset/model pairs. `record_metrics.csv` is intentionally verbose for outlier debugging.
