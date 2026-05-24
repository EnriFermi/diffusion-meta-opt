# Как выбрать режим получения данных (3 варианта)

Этот гайд нужен, чтобы быстро выбрать рабочий режим и сразу начать получать `SharedSample` для обучения.

## TL;DR

1. `streaming.mode=none`:
- Самый простой старт.
- Всё в памяти (очередь).
- Подходит для локальной отладки и коротких запусков.

2. `streaming.mode=local_disk` (профиль `data/streaming=gpu_parallel_streaming`):
- Collector пишет финальные chunk-файлы на локальный диск.
- Training читает эти chunk-файлы.
- Подходит для single-node multi-GPU и DDP.

3. `streaming.mode=s3_bridge` (профиль `data/streaming=s3_bridge_streaming`):
- Producer (collector) публикует chunks в S3.
- Consumer (training) забирает и потребляет.
- Подходит для разделённых машин.

## Быстрый выбор по сценарию

1. Хочу просто проверить пайплайн локально:
- Берите `none`.

2. У меня одна нода, есть выделенная collector GPU, хочу стабильный поток в train:
- Берите `local_disk`.

3. Collector и training на разных машинах/окружениях:
- Берите `s3_bridge`.

## Общий паттерн запуска

Во всех режимах вам нужно:

1. Задать токен и data root в `conf/config.yaml`:
- `hf.token`
- `data.path`

2. Выбрать датасеты:
- `data.enabled_datasets=[...]`

3. Убедиться, что у датасетов прописан `models: [...]`.

4. Запустить demo/тренировку с нужным streaming mode.

См. подробные инструкции:

1. `legacy/tutorials/10_data_mode_none_in_memory.md`
2. `legacy/tutorials/11_data_mode_local_disk_gpu_parallel.md`
3. `legacy/tutorials/12_data_mode_s3_bridge.md`
