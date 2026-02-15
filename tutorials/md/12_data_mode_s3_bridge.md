# Туториал 3: S3 Bridge streaming (`streaming.mode=s3_bridge`)

Collector публикует финальные chunk-файлы в S3, training скачивает и потребляет.

Профиль:
- `data/streaming=s3_bridge_streaming`

## Когда использовать

1. Producer и training на разных машинах.
2. Нужна decoupled pipeline архитектура.
3. Нужен общий удалённый буфер между окружениями.

## Минимальный запуск

```bash
pipenv run python -m dataset.shared.demo_streaming_s3_bridge \
  data/streaming=s3_bridge_streaming \
  streaming.s3.bucket=YOUR_BUCKET \
  streaming.s3.prefix=diffusion-meta-opt/streaming \
  train.device=cuda:0 \
  collector.device=cuda:1 \
  collector.mode=auto \
  data.enabled_datasets=[coco2017,cc12m] \
  data.dataset_overrides.coco2017.models=[clip_vit_b32] \
  data.dataset_overrides.cc12m.models=[clip_vit_b32] \
  demo.num_samples=120
```

## Ключевые параметры

1. Remote capacity:
- `streaming.s3.max_remote_chunks`

2. Upload pattern:
- staging key -> copy to ready key -> delete staging key

3. Delete policy:
- `streaming.consumer.delete_remote_after=consume` (default, рекомендовано)
- альтернатива: `download`

4. Локальные кэши:
- `streaming.producer.local_spool_dir`
- `streaming.consumer.local_cache_dir`

## Что проверять в первую очередь

1. Права на `PutObject/GetObject/DeleteObject`.
2. Корректный `bucket/prefix`.
3. Если используете S3-compatible storage — `streaming.s3.endpoint_url`.

## Как убедиться, что всё работает

1. Producer логи: рост ready chunks в S3.
2. Consumer логи: скачивание, consume, и удаление remote согласно policy.
3. На стороне training получаете `SharedSample` без блокировок по raw/model этапу.
