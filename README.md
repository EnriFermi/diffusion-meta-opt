# MetaOpt

Прототип “meta-optimizer”: мы учим оптимизатор как политику RL (SAC), которая на каждом шаге inner-loop предлагает обновления параметров downstream‑модели. В зависимости от `downstream.name` downstream‑задача может быть:

- `regression`: игрушечные регрессионные задачи (sine/cosine/poly) + MLP.
- `cifar10`: классификация CIFAR‑10 + маленький сверточный ResNet.

Код ориентирован на быстрые итерации и эксперименты (torch.compile, AMP, параллельная загрузка данных для CIFAR).

---

## Как запускать

Всё конфигурируется через OmegaConf:

- базовый конфиг в `MetaOpt/config.py` (`default_cfg()`)
- при запуске можно передать:
  - YAML: `--config path/to/config.yaml`
  - dotlist overrides: `key=value` (например `downstream.name=cifar10`)

Примеры:

```bash
# 1) Игрушечная регрессия (по умолчанию)
python -m MetaOpt.meta_learning

# 2) CIFAR-10 + tiny ResNet
python -m MetaOpt.meta_learning downstream.name=cifar10 cifar10.data_dir=./data cifar10.download=true

# 3) Быстро прогнать мало эпизодов
python -m MetaOpt.meta_learning rl.train_episodes=200 baseline.episodes=10

# 4) Выключить компиляцию/AMP (если нужно)
python -m MetaOpt.meta_learning perf.compile=false perf.amp=off
```

## Эксперименты

Готовые конфиги/скрипты лежат в `MetaOpt/experiments/`.

Пример CIFAR‑10 запуска:

```bash
bash MetaOpt/experiments/run_cifar10_fast.sh
```

Можно докинуть dotlist overrides:

```bash
bash MetaOpt/experiments/run_cifar10_fast.sh rl.train_episodes=1000 cifar10.cache_on_gpu=true

# сменить meta-алгоритм на VeLO-style ES
bash MetaOpt/experiments/run_cifar10_fast.sh meta.algorithm=velo
```

Значимые параметры (см. `MetaOpt/config.py`):

- `downstream.name`: `regression` | `cifar10`
- `meta.algorithm`: `sac` | `velo`
- `meta.inner_steps`: длина inner‑loop оптимизации (сколько шагов оптимизатор делает внутри эпизода)
- `rl.*`: гиперпараметры SAC (используются только при `meta.algorithm=sac`)
- `velo.*`: гиперпараметры VeLO-style ES (используются только при `meta.algorithm=velo`)
- `perf.*`: ускорение (compile/AMP/TF32/channels_last/num_workers)

---

## Что именно “учится”

Мы рассматриваем оптимизацию downstream‑модели как среду RL:

- **Состояние** `s_t`: статистики текущих весов и градиентов downstream‑модели
- **Действие** `a_t`: шаги обновления для параметров downstream‑модели
- **Награда** `r_t`: насколько уменьшилась train‑ошибка (плюс shaping по val‑ошибке на терминальном шаге)

### Представление состояния/действия (blockwise, по тензорам параметров)

Пусть downstream‑модель имеет `B` тензоров параметров (все `requires_grad=True`), тогда:

- `action_dim = B`
- `state_dim = 4 * B`

Для каждого тензора параметров `p_i` считаем:

- `w_rms[i]` — RMS весов `p_i`
- `g_rms[i]` — RMS градиента `∂L/∂p_i`
- `m[i]` — EMA по `g_rms` (моментум на уровне блока)
- `v[i]` — EMA по `g_rms^2` (аналог второй моменты на уровне блока)

Состояние:

```
s = concat([w_rms, g_rms, m, v])  # shape: (4*B,)
```

Действие `a` — вектор длины `B` (политика ограничивает его через `tanh` и масштабирует на `rl.max_action`).

Применение действия (в среде) — “SGD‑шаг с learned LR на блок”:

```
p_i <- p_i - a[i] * grad(p_i)
```

Реализация: `MetaOpt/env.py`.

---

## Награда и эпизод

Эпизод = оптимизация одной downstream‑модели на маленьком train‑подмножестве:

- на каждом inner‑шаге: `reward = -train_loss`
- на последнем шаге дополнительно: `reward += -val_loss` (терминальное shaping)

Поэтому “return” эпизода примерно:

```
sum_t -train_loss_t  -  final_val_loss
```

Для regression `loss = MSE`, для CIFAR‑10 `loss = CrossEntropy`.

---

## Meta алгоритмы

Выбор через `meta.algorithm`: `sac` | `velo`.

### SAC (reinforcement learning)

Soft Actor-Critic (`MetaOpt/rl/sac.py`).

- actor: `TransformerActor` (`MetaOpt/models.py`) — работает по токенам длины `B`, где каждый токен = один тензор параметров (фичи `[w, g, m, v]`)
- critics: простые MLP (`MetaOpt/models.py: Critic`)
- replay buffer: `MetaOpt/replay.py`
- гиперпараметры: `rl.*`

### VeLO-style ES (эволюционная оценка градиента)

Обучаем **только actor** чёрным‑ящиком через antithetic ES‑оценку градиента по return эпизода:

```
g ~ (1/(2*sigma*N)) * sum_i (R(θ+σϵ_i) - R(θ-σϵ_i)) * ϵ_i
```

- без critics / replay buffer
- гиперпараметры: `velo.*`

Тренировка в обоих случаях запускается из `MetaOpt/train.py`.

---

## Downstream задачи

### regression

- распределение задач задаётся через `cfg.tasks.*` (mix sine/cosine/poly)
- данные для каждой задачи сэмплируются “на лету” (`MetaOpt/tasks.py`)
- модель: `TwoLayerMLP` (`MetaOpt/models.py`)

### cifar10

- сэмплируем небольшие train/val подмножества для каждого эпизода (`MetaOpt/cifar10.py`)
- модель: `TinyResNet` (`MetaOpt/models.py`)
- зависимости: нужен `torchvision` (иначе будет понятная ошибка при выборе `downstream.name=cifar10`)

---

## Ускорение (compile / AMP / параллелизм)

Настройки в `cfg.perf.*` (см. `MetaOpt/config.py`).

Что включено:

- `torch.compile` для:
  - meta‑сетей (actor всегда; critics/targets только при `meta.algorithm=sac`)
  - downstream‑модели, которая переиспользуется между эпизодами (параметры сбрасываются через `reset_parameters_`)
- AMP:
  - `perf.amp=auto` на CUDA выберет bf16, если доступно (иначе fp16 + GradScaler)
- `channels_last` для CIFAR‑10 (модель и входы приводятся к channels_last)
- `TF32` и `cudnn.benchmark` на CUDA (ускоряет матмулы/свертки)
- “параллелизм” по данным CIFAR‑10:
  - `Cifar10EpisodeSampler` может использовать `DataLoader` с `num_workers>0` для подготовки батчей
    - на macOS/Windows по умолчанию `num_workers=0` (spawn), чтобы не плодить копии датасета в воркерах
  - `pin_memory` + `non_blocking=True` для копирования на GPU
  - опционально `cifar10.cache_on_gpu=true` — держать CIFAR‑тензоры на GPU между эпизодами (быстрее, но расходует VRAM)

Реализация: `MetaOpt/perf.py`, `MetaOpt/cifar10.py`, `MetaOpt/train.py`, `MetaOpt/env.py`.

---

## Чекпоинты

Сохранение/загрузка: `MetaOpt/checkpoint.py`.

В чекпоинт кладём:

- `cfg` (как контейнер OmegaConf)
- `meta_algo` + `agent` (state_dict выбранного meta‑алгоритма)
- `global_step`, `episode`

---

## Карта кода

- `MetaOpt/train.py` — основной тренировочный цикл (SAC или VeLO-style ES) + чекпоинты + eval
- `MetaOpt/config.py` — дефолтный конфиг + сборка конфига из CLI/YAML (`get_cfg`)
- `MetaOpt/meta_algos.py` — выбор meta‑алгоритма (`meta.algorithm`)
- `MetaOpt/velo.py` — VeLO-style ES агент
- `MetaOpt/downstream.py` — выбор downstream‑модели/лосса/сэмплера по `downstream.name`
- `MetaOpt/env.py` — “среда”: inner‑loop оптимизация + формирование state/action/reward
- `MetaOpt/rl/sac.py` — SAC агент
- `MetaOpt/models.py` — `TwoLayerMLP`, `TinyResNet`, actor/critic сети
- `MetaOpt/tasks.py` — toy regression задачи и их микс
- `MetaOpt/cifar10.py` — эпизодный сэмплер CIFAR‑10
- `MetaOpt/perf.py` — настройки производительности (compile/AMP/TF32/channels_last/reset)

---

## Добавление новой downstream задачи

1) Добавить модель/лосс/сэмплер.
2) Подключить в `MetaOpt/downstream.py`:
   - `build_downstream_model`
   - `build_episode_sampler`
   - `build_loss_fn`
3) При необходимости добавить параметры в `MetaOpt/config.py`.
4) Обновить этот README (важно: держать его синхронным с кодом).
