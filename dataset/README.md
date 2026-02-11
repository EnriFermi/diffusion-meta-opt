# Shared Data Collection Pipeline

Проект использует единственный production-путь:

- `dataset/data_raw/`: виртуализация HF raw-датасетов с chunk-cache.
- `dataset/models/`: виртуальные модели с хуками `nn.Linear` и `LayerIORecord`.
- `dataset/shared/`: model-first оркестрация (`CollectorService`, `SharedModelDataset`, кэш, scheduler).

## Конфиги

- `conf/config.yaml`: top-level + `defaults`.
- `conf/data/test_dataset.yaml`: профиль набора датасетов/override-ов.
- `conf/data/datasets/*.yaml`: отдельные raw-датасеты.
- `conf/data/models/*.yaml`: отдельные модели.

Выбор профиля делается через `defaults` в `conf/config.yaml`:

```yaml
defaults:
  - data: test_dataset
  - collector: interleaved
  - _self_
```

Локальные override-ы датасетов задаются в профиле `data.dataset_overrides`, например:

```yaml
dataset_overrides:
  flickr30k:
    models: [clip_vit_b32, dinov2_base]
```

## Model-First Flow

Один collector job:

1. Выбор модели (`collector.model_policy`, `collector.model_burst_jobs`).
2. Mixed batch из совместимых датасетов (`dataset_mix_policy`, `mix_cap_per_dataset`).
3. Инференс модели (на collector device, LRU cap = 1).
4. Атомизация LayerIO и запись в bounded cache.

## Режимы

- `async`: `collector.device != train.device`.
- `interleaved`: `collector.device == null` или `collector.device == train.device`.

## Запуск демо

Async:

```bash
pipenv run python -m dataset.shared.demo_async \
  train.device=cuda:0 collector.device=cuda:1 collector.mode=auto
```

Interleaved:

```bash
pipenv run python -m dataset.shared.demo_interleaved \
  train.device=cuda:0 collector.device=null collector.mode=auto
```

End-to-end:

```bash
pipenv run python -m dataset.shared.demo_end_to_end \
  train.device=cuda:0 collector.device=cuda:1 \
  data.enabled_datasets=[flickr30k,coco2017] \
  data.dataset_overrides.flickr30k.models=[clip_vit_b32,dinov2_base] \
  data.dataset_overrides.coco2017.models=[clip_vit_b32] \
  collector.cache.max_items=2000 collector.cache.low_watermark=1200 collector.cache.fill_target=2000 \
  demo.num_samples=200
```

## Инспекция raw-датасета

```bash
pipenv run python -m dataset.data_raw.tools.inspect_dataset flickr30k --n 3
```
