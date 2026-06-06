# Data Pipeline README

Этот документ описывает текущую реализацию data pipeline с фокусом на `dataset/data_raw`:

1. как задается каждый raw dataset,
2. как датасет фетчится и кэшируется,
3. что означает каждый ключ в конфиге,
4. как корректно добавить новый датасет, чтобы агент мог сделать это без догадок.

Все ниже соответствует текущему коду в:

- `dataset/data_raw/*`
- `dataset/shared/compatibility_index.py`
- `dataset/shared/raw_dataset_pool.py`
- `conf/data/datasets/*.yaml`

## 1. Где и как выбираются датасеты

Точка входа:

- `conf/big_vae/train/default.yaml` -> `defaults: - data: test_dataset`
- `conf/data/test_dataset.yaml`:
  - `dataset_config_dirs`: где искать yaml-файлы датасетов,
  - `enabled_datasets`: какие имена датасетов включены в ран,
  - `dataset_overrides`: runtime override конкретных полей датасета.

Лоадер совместимости:

- `dataset/shared/compatibility_index.py`
  - читает `data.enabled_datasets`,
  - грузит `conf/data/datasets/<name>.yaml`,
  - накладывает `data.dataset_overrides.<name>`,
  - отбрасывает отключенные (`enabled: true`) и несовместимые по `collector_device` (в async-режиме),
  - строит связи `model -> datasets`.

Важно:

- Имя датасета в `enabled_datasets` должно совпадать с именем файла без `.yaml`.
- Датасет должен быть зарегистрирован в реестре `dataset/data_raw/registry.py` через adapter-модуль.

## 2. Какие датасеты сейчас есть и как они описаны

Реальные конфиги лежат в `conf/data/datasets/`:

- `coco2017.yaml` (`phiyodr/coco2017`, URL mode, streaming)
- `cc12m.yaml` (`flax-community/conceptual-captions-12`, URL mode, streaming)
- `visual_genome.yaml` (`ranjaykrishna/visual_genome`, URL mode, streaming, `trust_remote_code=true`)
- `scene_parse_150.yaml` (`zhoubolei/scene_parse_150`, image mode, streaming)
- `bdd100k.yaml` (`dgural/bdd100k`, image mode, streaming)
- `mapillary_vistas_v2.yaml` (gated, image mode, streaming)
- `relaion400m.yaml` (gated, URL mode, streaming)
- `flickr30k.yaml` (image mode, currently `enabled: true`)

Шаблон с комментариями:

- `conf/data/datasets/_data_raw_.yaml`

Этот файл нужно использовать как стартовую точку для новых датасетов.

## 3. Полная схема dataset-конфига (runtime semantics)

Ниже поля, которые реально используются runtime-кодом.

### Корневые поля

- `name: str`
  - логическое имя датасета (обычно равно имени yaml-файла).
- `enabled: bool`
  - локальный флаг включения.
- `gated: bool`
  - если `true`, обязателен `hf.token` в `conf/big_vae/train/default.yaml`.
- `sampling_weight: float`
  - вес датасета в mixed batching.
- `models: list[str]`
  - список model config names из `conf/data/models/*.yaml`.
- `collector_device: null | "cuda:X"`
  - фильтрация датасета для async collector по устройству.
- `licensing_note: str`
  - текстовая заметка по лицензии/условиям.

### `hf`

- `hf.repo: str`
  - HF dataset id.
- `hf.subset: str | null`
  - subset/config name для `load_dataset`.
- `hf.split: str`
  - split (`train`, `validation`, `test`, ...).
- `hf.streaming: bool`
  - `true`: итератор + `.shuffle(buffer_size=...)`,
  - `false`: индексный доступ по random index.
- `hf.shuffle_buffer: int`
  - буфер shuffle в streaming-режиме.
- `hf.trust_remote_code: bool`
  - включать только если dataset script этого требует.

### `schema`

- `schema.image_mode: "image_field" | "url_field"`
  - режим извлечения изображения.
- `schema.image_field: str | null`
  - основная колонка с изображением (HF Image feature / bytes / PIL-like).
- `schema.image_field_candidates: list[str]`
  - fallback-кандидаты колонки с изображением.
- `schema.url_field: str | null`
  - основная колонка URL.
- `schema.url_field_candidates: list[str]`
  - fallback-кандидаты URL колонки.
- `schema.id_field: str | null`
  - колонка с sample id.
- `schema.extra_fields: list[str]`
  - поля, сохраняемые в `ImageSample.meta`.

### `cache`

- `cache.chunk_size_images: int`
  - сколько изображений собирать в один raw chunk.
- `cache.num_chunks_kept: int`
  - сколько raw chunk-ов хранить на диске одновременно.

### `worker`

- `worker.idle_sleep_s: float`
  - sleep, когда кэш заполнен или нет работы.
- `worker.get_timeout_s: float`
  - timeout чтения для consumer-side `get_batch`.
- `worker.startup_get_timeout_s: float` (опционально)
  - увеличенный timeout на прогрев первого chunk.
- `worker.request_timeout_s: int`
  - timeout HTTP-запросов для URL mode.
- `worker.max_retries: int`
  - число ретраев URL download.
- `worker.max_worker_restarts: int`
  - лимит перезапусков prefetch worker.
- `worker.max_chunk_build_seconds: float` (опционально)
  - лимит времени сборки одного chunk перед partial finalize.

Поля в шаблоне, которые сейчас в runtime не используются напрямую:

- `cache.root_subdir`
- `cache.decode_threads`
- `transforms.*`

Их можно оставлять как документационные, но опираться на них в логике пока нельзя.

## 4. Как именно датасет фетчится (шаг за шагом)

### 4.1 Инициализация

1. `RawDatasetPool` (`dataset/shared/raw_dataset_pool.py`) вызывает:
   - `validate_gated_datasets_token(...)`,
   - `init_hf_auth(...)`.
2. По каждому датасету вызывается `create_dataset(...)`.
3. Для HF-датасетов создается `HFVirtualDataset`.
4. `HFVirtualDataset`:
   - создает root: `${data.path}/${dataset_name}`,
   - создает `meta.json`, если отсутствует,
   - поднимает `ChunkCache`.

### 4.2 Старт worker-процесса

`HFVirtualDataset.start()` запускает `hf_dataset_prefetch_worker(...)` (multiprocessing, spawn).

Worker делает:

1. auth (`init_hf_auth`),
2. `load_hf_dataset(...)`,
3. бесконечный цикл prefetch.

### 4.3 Как формируется chunk

Внутри worker loop:

1. `cache.evict_old_chunks()` удаляет лишние старые chunk-и.
2. Если `len(chunks) >= num_chunks_kept`, worker спит (`idle_sleep_s`).
3. Иначе открывается новый `chunk_<timestamp>_<rand>`.
4. До `chunk_size_images`:
   - берется следующий record:
     - streaming: `next(iterator)`, с перезапуском при `StopIteration`,
     - non-streaming: `dataset[random_index]`.
   - извлекается `sample_id`,
   - materialize image:
     - `image_field`: `decode_to_pil(record[field])`,
     - `url_field`: `fetch_image_to_cache(url, timeout, retries)`.
5. Если собрано > 0 записей -> `finalize_chunk(...)` + event `chunk_ready`.
6. Если 0 -> chunk удаляется.

### 4.4 Как chunk читается потребителем

`HFVirtualDataset.get_batch(n)`:

1. забирает worker events (`chunk_ready`, `chunk_evicted`, `error`),
2. загружает `manifest.json` chunk-а в локальную очередь,
3. выдает `ImageSample` по одному, читая файлы с диска,
4. когда chunk исчерпан -> удаляет этот chunk с диска (`remove_chunk`),
5. если данных мало, возвращает partial batch и логирует underfill.

Итог:

- raw cache всегда ограничен `num_chunks_kept`,
- полностью датасет в raw chunk cache не складывается.

## 5. Структура файлов raw cache

Для датасета `<name>` под `${data.path}`:

```text
${data.path}/${name}/
  meta.json
  chunks/
    index.json
    chunk_<id_1>/
      manifest.json
      <img files>.jpg
    chunk_<id_2>/
      manifest.json
      <img files>.jpg
```

`meta.json` содержит:

- hf repo/subset/split,
- schema,
- streaming/trust_remote_code flags,
- `dataset_size` (если удалось оценить),
- gated/licensing info,
- timestamp.

## 6. Как добавить новый датасет (инструкция для агента)

Ниже канонический workflow.

### Шаг 1. Создать YAML в `conf/data/datasets/`

1. Скопировать `conf/data/datasets/_data_raw_.yaml`.
2. Заполнить:
   - `name`,
   - `hf.repo/subset/split`,
   - `schema.image_mode` + соответствующие поля,
   - `cache.chunk_size_images`, `cache.num_chunks_kept`,
   - `models`,
   - `gated`, `licensing_note`.
3. Если датасет gated:
   - `gated: true`,
   - убедиться, что `hf.token` задан в `conf/big_vae/train/default.yaml`,
   - лицензия принята на HF странице датасета.

### Шаг 2. Добавить adapter-модуль

Создать файл `dataset/data_raw/providers/hf/<dataset_name>.py`:

```python
from __future__ import annotations

from typing import Any

from dataset.data_raw.providers.hf.virtual_dataset import HFVirtualDataset
from dataset.data_raw.registry import register_dataset

DATASET_NAME = "my_dataset"


def build_dataset(cfg: Any, global_data_root: str, seed: int, hf_cfg: Any) -> HFVirtualDataset:
    return HFVirtualDataset(cfg=cfg, global_data_root=global_data_root, seed=seed, hf_cfg=hf_cfg)


register_dataset(DATASET_NAME, build_dataset)
```

### Шаг 3. Зарегистрировать модуль в пакете adapter-ов

В `dataset/data_raw/providers/hf/__init__.py` добавить модуль в `_ADAPTER_MODULES`.

Без этого `register_all_adapters()` его не импортирует, и датасет не попадет в registry.

### Шаг 4. Включить датасет в data-профиль

В `conf/data/<profile>.yaml`:

- добавить имя в `enabled_datasets`,
- при необходимости задать `dataset_overrides.<name>`.

### Шаг 5. Проверить совместимость с моделями

В dataset yaml `models: [...]` должны ссылаться только на существующие model yaml names.

Иначе `CompatibilityIndex` завершится ошибкой при старте.

### Шаг 6. Smoke-проверка

Команда:

```bash
python -m dataset.data_raw.tools.inspect_dataset <dataset_name> --n 8 --output-format pil
```

Проверить:

1. worker стартует,
2. chunk-и появляются в `${data.path}/${dataset_name}/chunks`,
3. возвращаются sample id и изображения,
4. нет бесконечных `underfilled` предупреждений.

## 7. Специфика image vs url mode

### image_field mode

Когда использовать:

- HF датасет уже хранит изображения как `datasets.Image` feature или bytes/PIL-подобный объект.

Что важно:

- корректно указать `image_field`,
- задать `image_field_candidates` для устойчивости к вариациям схемы.

### url_field mode

Когда использовать:

- датасет дает URL, не бинарные картинки.

Что важно:

- корректно указать `url_field`,
- добавить fallback `url_field_candidates`,
- понимать, что часть URL может быть битой/недоступной,
- под это настроить `worker.request_timeout_s` и `worker.max_retries`.

## 8. Ошибки и диагностика

### `HF token missing in top-level config: set hf.token`

Причина:

- включен gated датасет, но токен не задан/плейсхолдер.

Исправление:

1. заполнить `hf.token` в `conf/big_vae/train/default.yaml`,
2. принять лицензию на HF dataset page,
3. убедиться, что токен имеет доступ.

### Датасет не находится

Причины:

- нет yaml в `dataset_config_dirs`,
- имя в `enabled_datasets` не совпадает с файлом,
- adapter не зарегистрирован в `_ADAPTER_MODULES`.

### Пустые батчи / underfilled

Причины:

- медленный network (URL mode),
- много битых URL,
- слишком маленькие timeout/retry,
- стриминг датасета с временными проблемами.

Что сделать:

1. увеличить `request_timeout_s`, `max_retries`,
2. уменьшить `chunk_size_images`,
3. временно включить non-streaming, если dataset это поддерживает.

## 9. Минимальный чеклист для PR с новым датасетом

1. Добавлен `conf/data/datasets/<name>.yaml`.
2. Добавлен adapter `dataset/data_raw/providers/hf/<name>.py`.
3. Модуль добавлен в `_ADAPTER_MODULES`.
4. `models` в dataset yaml валидны.
5. Для gated датасета стоит `gated: true` + корректный `licensing_note`.
6. `inspect_dataset` smoke проходит.
7. В `meta.json` создаются корректные поля после первого запуска.

---

Если этот файл используется как вход для внешнего “agent that writes implementation instructions”, передавайте агенту минимум:

1. разделы 3-6 (schema + fetch lifecycle + onboarding steps),
2. шаблон `conf/data/datasets/_data_raw_.yaml`,
3. пример adapter-кода из раздела 6.
