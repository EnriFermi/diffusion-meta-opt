# Train/Test Dataset -> Models Mapping

Generated: 2026-02-17 20:45:00 CET

## Scope
- Mini profile: `legacy/mini_vae/conf/run_profiles/train_mini_vae.yaml` + `legacy/mini_vae/conf/data_collection_runtime/data_profiles/data_profile_for_mini_vae_training.yaml`
- Big profile: `conf/run_profiles/train_big_vae.yaml` + `conf/data_collection_runtime/data_profiles/data_profile_for_big_vae_training.yaml`
- Test profile: `conf/data_collection_runtime/data_profiles/data_profile_for_hf_assets_test.yaml`
- Dataset model lists are taken from `conf/data/datasets/<dataset>.yaml` (field `models`).

## Mini-VAE Train Profile

| Dataset | Models |
|---|---|
| `coco2017` | `clip_vit_b32`, `siglip_base_p16_384`, `dinov2_base`, `deit_base`, `vit_mae_base` |
| `cc12m` | `clip_vit_b32`, `siglip_base_p16_384`, `dinov2_vits14`, `beit_base`, `blip_image_captioning_base` |
| `visual_genome` | `clip_vit_b32`, `dinov2_base`, `dinov2_vits14`, `grounding_dino_tiny`, `blip_image_captioning_base` |
| `scene_parse_150` | `swinv2_base`, `mask2former_swin_base`, `grounding_dino_tiny`, `deit_base`, `beit_base` |

## Big-VAE Train Profile

| Dataset | Models |
|---|---|
| `bdd100k` | `swinv2_base`, `dinov2_base`, `vit_mae_base`, `deit_base`, `clip_vit_b32` |
| `cc12m` | `clip_vit_b32`, `siglip_base_p16_384`, `dinov2_vits14`, `beit_base`, `blip_image_captioning_base` |
| `coco2017` | `clip_vit_b32`, `siglip_base_p16_384`, `dinov2_base`, `deit_base`, `vit_mae_base` |
| `mapillary_vistas_v2` | `swinv2_base`, `mask2former_swin_base`, `grounding_dino_tiny`, `dinov2_base`, `deit_base` |
| `relaion400m` | `clip_vit_b32`, `siglip_base_p16_384`, `blip_image_captioning_base`, `dinov2_vits14`, `vit_mae_base` |
| `scene_parse_150` | `swinv2_base`, `mask2former_swin_base`, `grounding_dino_tiny`, `deit_base`, `beit_base` |
| `visual_genome` | `clip_vit_b32`, `dinov2_base`, `dinov2_vits14`, `grounding_dino_tiny`, `blip_image_captioning_base` |
| `stanford_cars` | `clip_vit_b32`, `siglip_base_p16_384`, `dinov2_base`, `deit_base`, `beit_base` |
| `dtd_textures` | `clip_vit_b32`, `dinov2_vits14`, `vit_mae_base`, `beit_base`, `deit_base` |
| `eurosat_rgb` | `clip_vit_b32`, `dinov2_base`, `vit_mae_base`, `swinv2_base`, `deit_base` |
| `patchcamelyon` | `clip_vit_b32`, `dinov2_base`, `dinov2_vits14`, `vit_mae_base`, `beit_base` |
| `oxford_pets` | `clip_vit_b32`, `siglip_base_p16_384`, `dinov2_base`, `deit_base`, `mask2former_swin_base` |
| `wider_face` | `clip_vit_b32`, `dinov2_vits14`, `swinv2_base`, `grounding_dino_tiny`, `mask2former_swin_base` |
| `doclaynet_v11` | `donut_base`, `trocr_base_printed`, `blip_image_captioning_base`, `clip_vit_b32`, `swinv2_base` |
| `cord_v2` | `donut_base`, `trocr_base_printed`, `blip_image_captioning_base`, `clip_vit_b32`, `dinov2_base` |
| `funsd` | `donut_base`, `trocr_base_printed`, `clip_vit_b32`, `dinov2_vits14`, `blip_image_captioning_base` |
| `docvqa_1200` | `donut_base`, `trocr_base_printed`, `clip_vit_b32`, `blip_image_captioning_base`, `deit_base` |

## Test Profile (HF Assets)

| Dataset | Models |
|---|---|
| `sun397` | `vit_large_p16_224`, `clip_vit_l14` |
| `food101` | `siglip_so400m_p14_384`, `vit_large_p16_224` |
| `rvl_cdip` | `donut_rvlcdip`, `trocr_large_printed` |
| `chexpert` | `vit_large_p16_224`, `clip_vit_l14` (grayscale source -> RGB/3-channel at decode time) |
| `openimages_v7` | `detr_resnet50`, `clip_vit_l14` |
| `pascal_voc_2012` | `segformer_b5_cityscapes`, `detr_resnet50` |

## Notes
- In current config layout, Big profile dataset set is a superset of Mini profile dataset set.
- Train split remains the existing current training datasets from Big/Mini profiles.
- Collector/device settings do not change the `dataset -> models` declarations; they only affect runtime eligibility/scheduling.
