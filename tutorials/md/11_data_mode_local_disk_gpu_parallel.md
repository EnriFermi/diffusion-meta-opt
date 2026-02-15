# Туториал 2: Local-Disk streaming (`streaming.mode=local_disk`)

Collector пишет финальные chunk-файлы на локальный диск, training читает их через `ChunkReader`.

Профиль по умолчанию для этого режима:
- `data/streaming=gpu_parallel_streaming`

## Когда использовать

1. Single-node multi-GPU.
2. Нужна устойчивость к временным лагам между collector и training.
3. Нужен DDP chunk-sharding на одной машине.

## Минимальный запуск (dedicated collector GPU)

```bash
pipenv run python -m dataset.shared.demo_streaming_local \
  data/streaming=gpu_parallel_streaming \
  train.device=cuda:0 \
  collector.device=cuda:1 \
  collector.mode=auto \
  data.enabled_datasets=[coco2017,scene_parse_150] \
  data.dataset_overrides.coco2017.models=[clip_vit_b32] \
  data.dataset_overrides.scene_parse_150.models=[dinov2_base]
```

`target_samples` и остальные demo-only параметры задаются в шапке
`dataset/shared/demo_streaming_local.py` и печатаются таблицей при запуске.

## Что важно в этом режиме

1. Hysteresis по chunk-store:
- fill до `streaming.local_disk.max_ready_chunks`
- stop на max
- resume когда `ready <= streaming.local_disk.low_watermark_chunks`

2. Producer-side spool:
- `streaming.producer.local_max_chunks`

3. Consumer-side prefetch:
- `streaming.consumer.local_max_chunks`

## DDP шардирование

В профиле `gpu_parallel_streaming` уже выставлено:
- `streaming.distributed.enabled=true`
- `streaming.distributed.shard_by=chunk`

Assignment:
- `assigned_rank = sha1(chunk_id) % world_size`

Каждый rank потребляет только свои chunks.

## Dedicated GPU поведение

В async-профиле collector:
- `pin_gpu: true`
- `release_device_on_unload: false`
- `empty_cuda_cache_on_unload: false`

Это означает: collector не пытается агрессивно освобождать память своей выделенной GPU “под VAE”.
