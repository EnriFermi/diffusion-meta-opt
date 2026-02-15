# Data Pipeline README (финальная архитектура)

Этот документ описывает **текущую боевую реализацию** data pipeline в репозитории после добавления
**second-level streaming virtualization** (виртуализация финальных post-inference sample chunks).

Документ отражает код в:
- `dataset/data_raw/*`
- `dataset/models/*`
- `dataset/shared/*`
- `dataset/shared/streaming/*`
- `conf/*`
- `tests/*`

---

## 1. Зачем нужен этот пайплайн

Цель пайплайна: готовить обучающие объекты не напрямую из raw-изображений, а из
**layer-level представлений моделей** (inputs/outputs linear-слоёв), при этом:

1. Не держать все raw datasets целиком на диске.
2. Не держать все модели одновременно в GPU.
3. Поддерживать model-first scheduling и mixed batching из нескольких датасетов.
4. Поддерживать два уровня буферизации:
   - уровень raw images (chunk cache per dataset),
   - уровень final samples (in-memory cache или streaming chunks: local disk / S3 bridge).

---

## 2. Главные инварианты системы

Текущая реализация гарантирует:

1. **Model-first scheduling**: сначала выбирается модель, потом датасеты, совместимые с этой моделью.
2. **Mixed batch**: батч модели собирается из нескольких raw datasets, а не из одного.
3. **Single collector model on collector device**: в `CollectorService` жёстко `collector.max_loaded_models=1`.
4. **Hooks на все `nn.Linear`** (с фильтрами regex) + flatten `(..., D) -> (N, D)`.
5. **CPU float32 capture** для hook-тензоров.
6. **Гейтед HF-доступ**: для gated datasets/models обязателен `hf.token` в top-level config.
7. **Second-level streaming**: финальные `SharedSample` могут идти:
   - в in-memory queue (`streaming.mode=none`),
   - в chunk store на локальном диске (`local_disk`),
   - в chunk store через S3 bridge (`s3_bridge`).

---

## 3. Структура кода

```text
dataset/
  data_raw/
    core/
    providers/hf/
    registry.py
    tools/inspect_dataset.py

  models/
    providers/transformers/
    hooks.py
    model_pool.py
    model_runner.py
    registry.py

  shared/
    compatibility_index.py
    model_scheduler.py
    raw_dataset_pool.py
    atomizer.py
    cache.py
    collector_service.py
    shared_dataset.py

    streaming/
      chunk_format.py
      chunk_writer.py
      chunk_reader.py
      config.py
      factory.py
      backends/
        base.py
        local_disk.py
        s3.py

    demo_async.py
    demo_interleaved.py
    demo_end_to_end.py
    demo_streaming_local.py
    demo_streaming_s3_bridge.py
```

---

## 4. Конфигурация (Hydra)

## 4.1 Top-level `conf/config.yaml`

`defaults`:
- `data: test_dataset`
- `data/collector: interleaved`
- `data/streaming: none`
- `_self_`

Ключевые секции:
- `hf.token`, `hf.hf_home`, `hf.datasets_cache`, `hf.hub_cache`
- `train.device`
- `data.*` (через профиль `conf/data/*.yaml`)
- `collector.*` (через профиль `conf/data/collector/*.yaml`)
- `streaming.*` (через профиль `conf/data/streaming/*.yaml`)
- `logging.*` (единая конфигурация console+file логирования)

Логирование ранa:
- логи пишутся одновременно в консоль и в файл;
- директория по умолчанию: `logs/`;
- формула имени файла:
  - `${logging.project_name}_${hydra:job.name}_${now:%Y-%m-%d_%H-%M-%S}.log`
- итоговый путь:
  - `${logging.dir}/${logging.file_name}`.

## 4.2 Профили streaming (`conf/data/streaming/*`)

### `none.yaml`
- `streaming.mode=none`
- Используется только in-memory `SharedSampleCache`.

### `gpu_parallel_streaming.yaml`
- `streaming.mode=local_disk`
- `streaming.distributed.enabled=true`
- Предназначен для single-node multi-process DDP consumption через chunk sharding.

### `s3_bridge_streaming.yaml`
- `streaming.mode=s3_bridge`
- Producer публикует chunk-файлы в S3 ready-prefix, consumer скачивает и потребляет.

### Канонические mode values
- `none`
- `local_disk`
- `s3_bridge`

Для обратной совместимости принимается alias:
- `in_memory` -> `none`

## 4.3 Data profile (`conf/data/*.yaml`)

Пример: `conf/data/test_dataset.yaml`.

Содержит:
- `path` — общий root для raw cache, model cache, streaming dirs.
- `enabled_datasets` — список dataset YAML имён.
- `dataset_config_dirs` — где лежат dataset YAML.
- `dataset_overrides` — точечные override для текущего эксперимента.

## 4.4 Dataset configs (`conf/data/datasets/*.yaml`)

Каждый датасет описывает:
- HF источник (`hf.repo`, `subset`, `split`, `streaming`, `trust_remote_code`)
- схему (`schema.image_mode`, `image_field`/`url_field`)
- raw cache параметры (`cache.chunk_size_images`, `cache.num_chunks_kept`)
- mapping `models: [...]`
- `sampling_weight`
- `collector_device` (опционально)
- `gated` + `licensing_note`

## 4.5 Model configs (`conf/data/models/*.yaml`)

Каждая модель задаёт:
- runner / provider параметры
- `hf_repo`, `revision`, `gated`
- `cache_subdir` (локальный model cache)
- `batch_size`, `sampling_weight`
- `device`, `dtype`, `run_mode`
- hook filters (`include_regex`, `exclude_regex`)
- limits (`max_records_per_layer`, `max_layers`)

---

## 5. Уровень raw datasets (`dataset/data_raw`)

## 5.1 Что происходит

1. Для каждого enabled датасета стартует отдельный worker process.
2. Worker держит на диске ограниченное число chunk-ов (`num_chunks_kept`).
3. По потреблению chunk-ов старые удаляются (FIFO), новые догружаются.
4. Поддерживается `image`-field и `url`-field режимы.

## 5.2 Что важно понимать

- Raw virtualization **не скачивает весь dataset в ваш chunk cache**.
- Но HF экосистема может хранить свои служебные artifacts в `HF_HOME/HF_DATASETS_CACHE/HF_HUB_CACHE`.

---

## 6. Уровень моделей (`dataset/models`)

1. `ModelPool` реализует LRU (с жёстким лимитом `max_loaded_models=1` на collector path).
2. Модель загружается из локального cache dir `${data.path}/${model.cache_subdir}`.
3. `run(batch_pil)` возвращает `LayerIORecord` по linear-слоям.
4. `merge_layer_records` объединяет micro-batches в единый per-layer output.

---

## 7. Shared orchestration (`dataset/shared`)

## 7.1 `CompatibilityIndex`

Строит:
- `model -> datasets`
- `model -> dataset_weights`
- `model_weights`

Учитывает:
- `data.enabled_datasets`
- `data.dataset_overrides`
- `collector.mode/device`
- dataset-level `collector_device` фильтрацию для async режима.

## 7.2 `ModelScheduler`

Поддерживает:
- `shuffled_cycle`
- `weighted`

`model_burst_jobs` контролирует частоту переключения модели.

## 7.3 `RawDatasetPool`

Для выбранной модели:
1. Берёт совместимые датасеты.
2. Делит `batch_size` по policy (`multinomial`/`uniform`) + cap (`mix_cap_per_dataset`).
3. Запрашивает PIL batch у каждого датасета.
4. Собирает unified mixed batch + `MixedImageMeta`.

## 7.4 `Atomizer`

Преобразует `LayerIORecord` в `SharedSample`:
- `atom_mode=row|chunk`
- `chunk_rows`
- meta: `model_run_id`, timestamps, `image_meta`, `row2img`, ranges.

---

## 8. Second-level streaming virtualization

## 8.1 Зачем

Раньше финальные `SharedSample` шли только в in-memory queue.
Теперь есть второй слой: **final chunks** (post-inference) в backend store.

Это даёт:
- decoupling producer/consumer,
- контроль диска/remote буфера,
- bridge между машинами (S3 producer -> trainer consumer),
- DDP-friendly chunk sharding в local mode.

## 8.2 Компоненты (`dataset/shared/streaming`)

### `chunk_format.py`
- `save_chunk(path, samples, meta, compression)`
- `load_chunk(path) -> (samples, meta)`
- формат: `chunk_id`, `created_at`, `num_samples`, `samples`, `meta`
- поддержка `compression=none|gzip`

### `backends/base.py`
- `ChunkStore` интерфейс:
  - `put_ready`
  - `list_ready`
  - `fetch_to_local`
  - `delete_ready`
  - `count_ready`
  - `capacity_state`

### `backends/local_disk.py`
- структура: `root/staging`, `root/ready`, `root/consumed`
- publish pattern: staging -> atomic rename в ready
- лимиты:
  - `max_ready_chunks`
  - `low_watermark_chunks`

### `backends/s3.py`
- publish pattern:
  - upload в staging key
  - copy в ready key
  - delete staging key
- готовые chunk-и листаются в `ready_prefix`
- лимит remote: `max_remote_chunks`

### `chunk_writer.py`
- буферизует `SharedSample`
- flush по `chunk_size_samples`
- умеет `flush(force_partial=True)` на shutdown

### `chunk_reader.py`
- prefetch готовых chunk-ов в локальный cache
- выдаёт `SharedSample` по одному
- delete policy:
  - `download` — удалять remote сразу после download
  - `consume` — удалять remote после полного потребления chunk-а (**default**)
- поддерживает distributed chunk sharding.

### `config.py`
- normalize mode (`in_memory -> none`)
- resolve distributed rank/world_size из env.

### `factory.py`
- единая сборка:
  - streaming cfg
  - chunk store
  - chunk writer
  - chunk reader

---

## 9. Как это интегрировано в `CollectorService`

`CollectorService` теперь использует sink abstraction:

1. `_InMemorySampleSink` (`streaming.mode=none`)
   - пишет в `SharedSampleCache`.

2. `_ChunkedSampleSink` (`local_disk|s3_bridge`)
   - пишет в `ChunkWriter` + backend store.

Поведение fill control:
- `none`: по `cache.fill_target/low_watermark`.
- `local_disk`: hysteresis fill-to-max/resume-at-low:
  - fill пока `ready < max_ready_chunks`,
  - остановка на max,
  - возобновление когда `ready <= low_watermark_chunks`.
- `s3_bridge`: fill пока backend может принять (`ready < max_remote_chunks`) и writer может spool.

Shutdown semantics:
- на остановке sink закрывается,
- для chunked sink вызывается `writer.flush(force_partial=True)` через `close()`.

---

## 10. Как это интегрировано в `SharedModelDataset`

`SharedModelDataset` сохранил публичный контракт, но источник теперь mode-aware:

1. `streaming.mode=none`:
   - читает `collector.cache.get(block=True)`.

2. `streaming.mode=local_disk|s3_bridge`:
   - создаёт `ChunkReader` через `collector.create_chunk_reader()`
   - читает `reader.next_sample()`.

Методы:
- `__iter__()` — бесконечный stream sample-ов.
- `maybe_collect(step_idx)` — interleaved hook.
- `cache_size()`:
  - для `none`: count sample-ов,
  - для chunked: backend ready chunks (метрика уровня chunk store).
- `try_next_sample()` — non-blocking удобный путь для демо.

---

## 11. Async vs Interleaved режимы

## 11.1 Async mode

Условие:
- `collector.device != train.device` и mode resolves to `async`.

Поведение:
- отдельный collector process,
- непрерывно выполняет collector jobs пока sink требует fill.

## 11.2 Interleaved mode

Условие:
- `collector.device == train.device` или `collector.device=null`.

Поведение:
- тот же процесс что и training loop,
- `maybe_collect(step_idx)` запускает burst по:
  - `every_n_steps`
  - `burst_jobs`
  - только если sink low.

Важно:
- Нельзя пытаться параллелить training и collector в разных process на одной GPU.

---

## 12. DDP chunk sharding (`gpu_parallel_streaming`)

В `streaming.distributed.enabled=true`:

1. rank/world_size читаются из env (`RANK`, `WORLD_SIZE` по умолчанию).
2. Chunk assignment:
   - `assigned_rank = sha1(chunk_id) % world_size`
3. Каждый rank потребляет только свои chunk-и.
4. После consume локальные файлы удаляются reader-ом.

Это позволяет multi-process consumption без централизованного coordinator.

---

## 13. Рекомендованные hyperparameters

Ниже стартовые ориентиры, потом подбирать под throughput/latency:

## 13.1 Producer/consumer буфер

- `streaming.chunk_size_samples`: `128..512`
- `streaming.producer.local_max_chunks`: `32..128`
- `streaming.consumer.local_max_chunks`: `8..32`

## 13.2 Local disk backend

- `streaming.local_disk.max_ready_chunks`: `200..1000`
- `streaming.local_disk.low_watermark_chunks`: `~50% от max_ready_chunks`

## 13.3 S3 bridge backend

- `streaming.s3.max_remote_chunks`: `500..5000`
- `streaming.consumer.delete_remote_after=consume` (default)
  - safer, потому что remote удаляется только после полного consume.

## 13.4 Collector knobs

- `collector.model_burst_jobs`: `1..2`
- `collector.num_inflight_jobs`: `1..2`
- `collector.mix_cap_per_dataset`: `0.4..0.7`

## 13.5 Atomization

- `collector.atomization.atom_mode=chunk`
- `collector.atomization.chunk_rows=128..256`

---

## 14. Failure modes и recovery

## 14.1 `HF token missing in top-level config: set hf.token`

Причина:
- gated dataset/model включён без валидного token.

Recovery:
1. Указать `hf.token` в `conf/config.yaml`.
2. Принять лицензию/условия на странице HF.
3. Проверить, что token имеет доступ.

## 14.2 Chunk store переполнен

Симптом:
- producer перестаёт пополнять.

Это ожидаемо при backpressure.

Recovery:
- увеличить consumption rate,
- уменьшить `chunk_size_samples`,
- увеличить `max_ready_chunks`/`max_remote_chunks`.

## 14.3 S3 backlog не уменьшается

Проверить:
- `delete_remote_after` (consume/download),
- доступы на `DeleteObject`,
- корректный `bucket/prefix`.

## 14.4 Cache не наполняется

Проверить:
- совместимость dataset->models в `CompatibilityIndex`,
- `collector.mode/device`,
- gated token/license,
- worker-логи raw datasets,
- `collector.stats()` (`sink`, `cache_size`, `jobs_by_model`).

---

## 15. Команды запуска

## 15.0 Быстрые туториалы по 3 режимам

1. Выбор режима:
`/Users/artemon/Library/Mobile Documents/com~apple~CloudDocs/Programming/python_projects/diffusion-meta-opt/tutorials/00_how_to_choose_data_mode.md`
2. In-memory (`streaming.mode=none`):
`/Users/artemon/Library/Mobile Documents/com~apple~CloudDocs/Programming/python_projects/diffusion-meta-opt/tutorials/10_data_mode_none_in_memory.ipynb`
3. Local disk (`streaming.mode=local_disk`):
`/Users/artemon/Library/Mobile Documents/com~apple~CloudDocs/Programming/python_projects/diffusion-meta-opt/tutorials/11_data_mode_local_disk_gpu_parallel.ipynb`
4. S3 bridge (`streaming.mode=s3_bridge`):
`/Users/artemon/Library/Mobile Documents/com~apple~CloudDocs/Programming/python_projects/diffusion-meta-opt/tutorials/12_data_mode_s3_bridge.ipynb`

## 15.1 Unit tests

```bash
pipenv run pytest -q tests
```

## 15.2 Raw dataset inspect

```bash
pipenv run python -m dataset.data_raw.tools.inspect_dataset coco2017 --n 3
```

## 15.3 End-to-end (in-memory)

```bash
pipenv run python -m dataset.shared.demo_end_to_end \
  streaming.mode=none \
  train.device=cuda:0 \
  collector.device=null \
  collector.mode=auto
```

## 15.4 Streaming local disk

```bash
pipenv run python -m dataset.shared.demo_streaming_local \
  data/streaming=gpu_parallel_streaming \
  train.device=cuda:0 \
  collector.device=cuda:1 \
  collector.mode=auto
```

## 15.5 Streaming S3 bridge

```bash
pipenv run python -m dataset.shared.demo_streaming_s3_bridge \
  data/streaming=s3_bridge_streaming \
  streaming.s3.bucket=YOUR_BUCKET \
  train.device=cuda:0 \
  collector.device=cuda:1 \
  collector.mode=auto
```

---

## 16. Тестовое покрытие streaming слоя

Добавлены тесты:
- `tests/test_streaming_chunk_format.py`
- `tests/test_streaming_local_disk_backend.py`
- `tests/test_streaming_sharding.py`
- `tests/test_streaming_chunk_writer_reader_local.py`
- `tests/test_streaming_mode_resolution.py`
- `tests/test_streaming_s3_backend.py`

Также обновлены token-тесты:
- `tests/test_hf_token_requirements.py`

---

## 17. Что расширять дальше

1. Добавить retry/persistence для writer buffer при длительном backend full.
2. Добавить batch IPC протокол для chunk reader/writer telemetry.
3. Добавить richer metrics endpoint (Prometheus-friendly counters).
4. Добавить multi-collector (per-device) поверх текущего single-collector контракта.
