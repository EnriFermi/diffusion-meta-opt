# ViT Latent Scaling

Новый пайплайн здесь намеренно простой:

- один Hydra-конфиг: [conf/vit_latent_scaling/config.yaml](/Users/enrifermi/Projects/diff-meta-opt/conf/vit_latent_scaling/config.yaml)
- один launcher: [run_vit_latent_scaling.sh](/Users/enrifermi/Projects/diff-meta-opt/post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh)
- единое хранилище запусков: `artifacts/big_vae/eval/vit_latent_scaling/runs/<run_label>/<run_id>/`
- shared checkpoint store: `artifacts/big_vae/eval/vit_latent_scaling/checkpoints/<shared_checkpoint_label>/`

## Что умеет

- `raw` и `latent` setup
- latent init:
  - `fresh` + `base|random`
  - `source` из любого предыдущего run-а
  - `diffusion_prior`
- smart source transplant:
  - head mismatch не валит запуск
  - patch embedding и `pos_embed` адаптируются под новый shape
  - если latent-layout совпал, можно напрямую поднять exact latent state
- checkpoints:
  - `init.pt`
  - `latest.pt`
  - `best.pt`
  - `final.pt`
  - `step_XXXXXX.pt`
- локальный лог `run.log`
- `metrics.csv`
- `summary.json`
- Comet logging

## Структура run-dir

Каждый запуск пишет сюда:

```text
artifacts/runs/<run_label>/<run_id>/
  config_resolved.yaml
  config_resolved.json
  run.log
  metrics.csv
  summary.json
  checkpoints/

artifacts/checkpoints/<shared_checkpoint_label>/
  init.pt
  latest.pt
  best.pt
  final.pt
  step_XXXXXX.pt
```

Пути не перезатираются: каждый run получает новый `run_id` внутри директории своего `run_label`.
При этом checkpoints дополнительно зеркалятся в shared директорию с отдельным именем, чтобы не искать модель по per-run путям.

## Как запускать

Редактируй переменные вверху:

[run_vit_latent_scaling.sh](/Users/enrifermi/Projects/diff-meta-opt/post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh)

Для shared checkpoint store редактируй:

```text
SHARED_CHECKPOINT_LABEL="my_model"
SHARED_CHECKPOINT_ROOT=""
```

Если `SHARED_CHECKPOINT_ROOT=""`, то используется дефолт:

```text
artifacts/big_vae/eval/vit_latent_scaling/checkpoints/<shared_checkpoint_label>/
```

И потом:

```bash
./post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh
```

## Decoder Adapter

Latent setup can decode through a post-train BigVAE decoder adapter:

```bash
EVAL_DECODER_ADAPTER=latent_flattening_flow \
EVAL_DECODER_ADAPTER_CHECKPOINT=/path/to/latent_flattening/checkpoints/latest.pt \
./post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh
```

For latent flattening this means trainable latents are interpreted as
`z_prime`, and weights are decoded as `decoder(flow.inverse(z_prime))`. This is
available for `init.kind=diffusion_prior`, where latent slots live in
`decoder_z` space.

## Source init

Чтобы дообучить новый запуск из старого:

```text
INIT_KIND="source"
SOURCE_RUN_DIR="<run_id или path/to/run_dir>"
SOURCE_CHECKPOINT="best"
```

Для latent setup логика такая:

- если exact latent layout совпал и `SOURCE_PREFER_DIRECT_LATENT=true`, грузится прямой latent state
- иначе берутся `named_tensors` из source checkpoint и smart-перекладываются в новую архитектуру
- после этого latent-модель стартует через encode от этих source weights

То есть chain из многих запусков делается без отдельного zoo из shell-скриптов.

## Профили

Сейчас profiles живут в коде, а не в россыпи yaml:

[config.py](/Users/enrifermi/Projects/diff-meta-opt/post_train_research/vit_latent_scaling/config.py)

Например:

- `cifar10_50k`
- `cifar10_small`
- `mnist_tiny`
- `mnist_mili`
- `cifar_mili`
- `imagenet_small`

Выбирается через:

```text
PROFILE="cifar10_50k"
```
