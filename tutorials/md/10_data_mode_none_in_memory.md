# Туториал 1: In-Memory режим (`streaming.mode=none`)

Это самый простой режим: collector складывает `SharedSample` в память, training читает из памяти.

## Когда использовать

1. Локальная отладка.
2. Быстрый smoke test.
3. Небольшие эксперименты без долгого буферинга на диск/S3.

## Минимальный запуск

```bash
pipenv run python -m dataset.shared.demo_end_to_end \
  data/streaming=none \
  train.device=cuda:0 \
  collector.device=null \
  collector.mode=auto \
  data.enabled_datasets=[coco2017,cc12m] \
  data.dataset_overrides.coco2017.models=[clip_vit_b32] \
  data.dataset_overrides.cc12m.models=[clip_vit_b32]
```

`target_samples` и остальные demo-only параметры задаются в шапке
`dataset/shared/demo_end_to_end.py` и печатаются таблицей при запуске.

## Что вы увидите

1. Collector job логи (`model=... images=... layers=...`).
2. Рост/падение `cache size` (in-memory queue).
3. В конце — summary по model/layer/dataset mix.

## Как понять, что данные реально идут в обучение

В вашем train-loop вы используете `SharedModelDataset`, и каждый item — это `SharedSample`:

- `model_name`
- `layer_name`
- `x` / `y` (тензоры, CPU float32)
- `meta` (dataset/source mapping)

## Практические параметры

1. `collector.cache.max_items/fill_target/low_watermark`
2. `collector.model_burst_jobs`
3. `collector.atomization.chunk_rows`

## Ограничения режима

1. Буфер ограничен RAM.
2. При рестарте процесса in-memory очередь теряется.
