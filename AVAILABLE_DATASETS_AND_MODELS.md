# Доступные датасеты и модели (текущее состояние кода)

Собрано из:
- `conf/data/datasets/*.yaml` (кроме шаблона `_data_raw_.yaml`)
- `conf/data/models/*.yaml` (кроме шаблона `_model_.yaml`)
- `dataset/data_raw/providers/hf/__init__.py`
- `dataset/models/registry.py`
- `train.py` и `conf/model/weight_quantile_vae.yaml`

## 1) Датасеты (`dataset/data_raw`)

| Dataset | enabled | gated | HF repo | streaming | Модели из `models:` |
|---|---:|---:|---|---:|---|
| `bdd100k` | true | false | `dgural/bdd100k` | true | `dinov2_base` |
| `cc12m` | true | false | `flax-community/conceptual-captions-12` | true | `clip_vit_b32` |
| `coco2017` | true | false | `phiyodr/coco2017` | true | `clip_vit_b32`, `siglip_base_p16_384` |
| `flickr30k` | false | false | `nlphuji/flickr30k` | false | `clip_vit_b32`, `dinov2_base` |
| `mapillary_vistas_v2` | true | true | `candylion/mapillary-vistas-v2` | true | `swinv2_base` |
| `relaion400m` | true | true | `laion/relaion400m` | true | `clip_vit_b32` |
| `scene_parse_150` | true | false | `zhoubolei/scene_parse_150` | true | `swinv2_base` |
| `visual_genome` | true | false | `ranjaykrishna/visual_genome` | true | `clip_vit_b32` |

Runtime-регистрация адаптеров (`register_dataset`) сейчас покрывает эти же 8 имен:
`bdd100k`, `cc12m`, `coco2017`, `flickr30k`, `mapillary_vistas_v2`, `relaion400m`, `scene_parse_150`, `visual_genome`.

## 2) Модели для сбора признаков (`dataset/models`)

| Model | runner | family | run_mode | HF repo |
|---|---|---|---|---|
| `beit_base` | `hf_vit_runner` | `vit` | `encoder_only` | `microsoft/beit-base-patch16-224` |
| `blip_image_captioning_base` | `hf_encdec_runner` | `blip` | `vision_encoder_only` | `Salesforce/blip-image-captioning-base` |
| `clip_vit_b32` | `hf_clip_runner` | `clip` | `vision_only` | `openai/clip-vit-base-patch32` |
| `deit_base` | `hf_vit_runner` | `vit` | `encoder_only` | `facebook/deit-base-patch16-224` |
| `dinov2_base` | `hf_vit_runner` | `vit` | `encoder_only` | `facebook/dinov2-base` |
| `dinov2_vits14` | `hf_vit_runner` | `vit` | `encoder_only` | `facebook/dinov2-small` |
| `donut_base` | `hf_encdec_runner` | `donut` | `encoder_only` | `naver-clova-ix/donut-base` |
| `grounding_dino_tiny` | `hf_dense_runner` | `dense` | `vision_only` | `IDEA-Research/grounding-dino-tiny` |
| `mask2former_swin_base` | `hf_dense_runner` | `dense` | `full` | `facebook/mask2former-swin-base-coco-panoptic` |
| `siglip_base_p16_384` | `hf_siglip_runner` | `siglip` | `vision_only` | `google/siglip-base-patch16-384` |
| `swinv2_base` | `hf_vit_runner` | `vit` | `encoder_only` | `microsoft/swinv2-base-patch4-window8-256` |
| `trocr_base_printed` | `hf_encdec_runner` | `trocr` | `encoder_only` | `microsoft/trocr-base-printed` |
| `vit_mae_base` | `hf_vit_runner` | `vit` | `encoder_only` | `facebook/vit-mae-base` |

Зарегистрированные раннеры в `dataset/models/registry.py`:
`hf_vit_runner`, `hf_clip_runner`, `hf_siglip_runner`, `hf_dense_runner`, `hf_encdec_runner`.

## 3) Модель обучения в `train.py`

`train.py` обучает `WeightQuantileVAE` из `models/weight_quantile_vae.py` с гиперпараметрами из `conf/model/weight_quantile_vae.yaml`.

## 4) Что активно по умолчанию в текущем профиле данных

В `conf/data/test_dataset.yaml` в `enabled_datasets` включены:
`coco2017`, `cc12m`, `visual_genome`, `scene_parse_150`, `mapillary_vistas_v2`, `relaion400m`.

