# Dataset Pipeline Entry README

Полная техническая документация пайплайна находится в:
- `docs/data_pipeline_README.md`

Этот файл оставлен как короткий entry-point.

## Что сейчас production-path

Используется только связка:
- `dataset/data_raw/` — виртуализация HF raw datasets + chunk-cache
- `dataset/models/` — model virtualization + `nn.Linear` hooks
- `dataset/shared/` — model-first orchestration + shared cache/streaming + iterable dataset

Streaming modes:
- `streaming.mode=none` (in-memory cache)
- `streaming.mode=local_disk` (final chunks on local disk)
- `streaming.mode=s3_bridge` (producer/consumer через S3)

Подробности и тюнинг streaming вынесены в:
- `docs/data_pipeline_README.md` (разделы 8-15)
- `legacy/tutorials/00_how_to_choose_data_mode.md` (выбор режима)
- `legacy/tutorials/10_data_mode_none_in_memory.ipynb`
- `legacy/tutorials/11_data_mode_local_disk_gpu_parallel.ipynb`
- `legacy/tutorials/12_data_mode_s3_bridge.ipynb`

## Быстрые команды

Inspect raw dataset:

```bash
pipenv run python -m dataset.data_raw.tools.inspect_dataset coco2017 --n 3
```

Interleaved demo:

```bash
pipenv run python -m dataset.shared.demo_interleaved \
  train.device=cuda:0 \
  collector.device=null \
  collector.mode=auto
```

Async demo:

```bash
pipenv run python -m dataset.shared.demo_async \
  train.device=cuda:0 \
  collector.device=cuda:1 \
  collector.mode=auto
```

End-to-end demo:

```bash
pipenv run python -m dataset.shared.demo_end_to_end \
  train.device=cuda:0 \
  collector.device=null \
  collector.mode=auto \
  data.enabled_datasets=[coco2017,cc12m,scene_parse_150]
```

Параметры demo-run (`target_samples`, `steps`, `runtime_seconds`) задаются локальными
константами в начале каждого `dataset/shared/demo_*.py` и логируются таблицей при старте.

Local streaming demo:

```bash
pipenv run python -m dataset.shared.demo_streaming_local \
  data/streaming=gpu_parallel_streaming \
  train.device=cuda:0 \
  collector.device=cuda:1 \
  collector.mode=auto
```

S3 bridge streaming demo:

```bash
pipenv run python -m dataset.shared.demo_streaming_s3_bridge \
  data/streaming=s3_bridge_streaming \
  streaming.s3.bucket=YOUR_BUCKET \
  train.device=cuda:0 \
  collector.device=cuda:1 \
  collector.mode=auto
```

Логи любого запуска дублируются:
- в консоль;
- в файл `${logging.dir}/${logging.file_name}` (по умолчанию `logs/*.log`).

## Extra HF raw datasets (10-pack)

Добавлен отдельный профиль данных:
- `conf/data/hf_extra_10.yaml`

Он включает новые датасеты:
- `stanford_cars`
- `dtd_textures`
- `eurosat_rgb`
- `patchcamelyon`
- `oxford_pets`
- `wider_face`
- `doclaynet_v11`
- `cord_v2`
- `funsd`
- `docvqa_1200`

Базовые YAML лежат в:
- `conf/data/datasets/*.yaml` (по имени датасета)

Как запустить профиль:

```bash
pipenv run python train.py data=hf_extra_10
```

Smoke-проверка каждого датасета:

```bash
pipenv run python -m dataset.data_raw.tools.inspect_dataset stanford_cars --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset dtd_textures --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset eurosat_rgb --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset patchcamelyon --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset oxford_pets --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset wider_face --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset doclaynet_v11 --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset cord_v2 --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset funsd --n 8 --output-format pil
pipenv run python -m dataset.data_raw.tools.inspect_dataset docvqa_1200 --n 8 --output-format pil
```
