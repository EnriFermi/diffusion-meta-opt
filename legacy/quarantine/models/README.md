# WeightQuantileVAE: Architecture And Full Parameter Reference

Документ описывает все параметры моделей и связанных конфигов в текущей кодовой базе:

- `models/weight_quantile_vae.py` (backward-compatible re-export shim)
- `models/distribution_encoder.py`
- `models/mini_patch_vae.py`
- `models/patch_tokenizers.py`
- `models/big_weight_vae.py`
- `models/big_weight_vae_parts/`
- `models/vae_shared.py`
- `legacy/mini_vae/pretrain_mini_patch_vae.py`
- `conf/big_vae/train/`

Основная реализация больше не лежит в одном файле: `models/weight_quantile_vae.py`
сохранен для обратной совместимости и просто реэкспортирует публичные классы.
MiniVAE training is legacy; new training runs should use `conf/big_vae/train/`
and `scripts/launchers/big_vae/`.

## Modules

- `InputDistributionEncodingModule`: кодирует распределение входов `X` для одного patch по `d_in`.
- `MiniPatchVAE`: patch-level VAE для `w_patch`.
- `BigWeightVAE` (alias `WeightQuantileVAE`): full-matrix VAE для `W`.
- `DCNv2`: блок кросс-признаков для `dist_patch_embed`.

## Tensor conventions

- `X`: `[B, n, d_in]` или `[n, d_in]` (unbatched для `BigWeightVAE.forward`).
- `W`: `[B, d_in, d_out]` или `[d_in, d_out]`.
- `p = patch_size`.
- `T = ceil(d_in / p)`.

## Encoder Details (Current Code)

### InputDistributionEncodingModule (`models/weight_quantile_vae.py`)

Входы:
- `X`: `[B, n, d_in]`
- `patch_idx`: `[B, p]`

Выходы:
- `dist_var_tokens`: `[B, p, d_var]`
- `dist_patch_embed`: `[B, d_dist]`

Пайплайн:
1. По `patch_idx` берется `X_I = X[..., patch_idx]` формы `[B, n, p]`.
2. По оси `n` считаются эмпирические квантили `k_s`: `q` формы `[B, k_s, p]`.
3. `q` нормируется в `[-1, 1]` по переменной (с фиксированными концами `-1` и `1`).
4. Считаются статистики `mu` и `sigma` по оси `n`, затем `mu_log = sign(mu) * log1p(|mu|)` и `sigma_log = log(sigma + eps)`.
5. Квантильный сигнал гонится через одноканальный `Conv1d(kernel=3, padding=0)` вдоль квантильной оси, получается `q_conv: [B, p, k_s-2]` (без усреднения по квантилям).
6. Формируются per-variable признаки `f = [q_conv, mu_log, sigma_log]` размера `[B, p, k_s]`.
7. `f -> var_mlp -> var_encoder(TransformerEncoder)` дает `dist_var_tokens: [B, p, d_var]`.
8. `dist_var_tokens` mean-pool по `p`, затем `DCNv2` дает `dist_patch_embed: [B, d_dist]`.

### MiniPatchEncoder (`models/weight_quantile_vae.py`)

Входы:
- `w_patch`: `[B, p]`
- `dist_var_tokens`: `[B, p, d_var]`

Пайплайн:
1. Строится детерминированный sinusoidal positional embedding по индексу элемента патча: `[p, pos_dim]`.
2. На каждом элементе патча конкатенируются `[w_i, dist_var_token_i, pos_i]`, итого `[B, p, 1 + d_var + pos_dim]`.
3. `elem_embed` (MLP с GELU/Dropout) переводит в `[B, p, d_e]`.
4. Learnable `CLS` token prepended к последовательности, затем `TransformerEncoder` (`num_attn_layers_encoder`) добавляет контекст по элементам патча.
5. В качестве агрегированного представления берется выход `CLS`: `h = e_ctx[:, 0, :]` (`[B, d_e]`).
6. Перед head-ами применяется `LayerNorm` к `h`.
7. Линейные головы дают `mu` и `logvar`: `[B, z_dim]`.

Важно:
- В текущей реализации `MiniPatchDecoder` conditioned только на `z` и positional (`dist_var_tokens` в декодер не подаются).
- Поэтому distribution conditioning входит в mini-VAE через энкодерную ветку.

## Model dataclasses (models/weight_quantile_vae.py)

### DistributionConfig

| Param | Type | Meaning |
|---|---|---|
| `k_s` | `int` | Число квантилей на входную переменную. Должно быть `>= 3` (kernel=3 без padding). |
| `Kq` | `int` | Legacy-поле (в текущем quantile-пути не используется). |
| `d_var` | `int` | Размер per-variable токена после `var_mlp` и variable self-attention. |
| `d_dist` | `int` | Размер итогового patch-level distribution embedding (`dist_patch_embed`). |
| `num_var_attn_layers` | `int` | Количество слоев `TransformerEncoder` над variable tokens. |
| `var_attn_heads` | `int` | Число attention-heads в variable encoder. `d_var % var_attn_heads == 0`. |
| `dcn_num_cross_layers` | `int` | Число cross-слоев в `DCNv2`. |
| `dcn_deep_hidden` | `int` | Hidden размер deep tower в `DCNv2` (если `>0`). |
| `dcn_deep_layers` | `int` | Число deep-слоев в `DCNv2` (если `>0`). |
| `dropout` | `float` | Dropout в `var_mlp` и variable-attention/FFN. |

### MiniVAEConfig

| Param | Type | Meaning |
|---|---|---|
| `z_dim` | `int` | Размер латента mini-VAE. |
| `d_e` | `int` | Hidden размер encoder/decoder mini-VAE. |
| `pos_dim` | `int` | Размер детерминированного positional embedding по индексу элемента патча в mini encoder. |
| `num_attn_layers_encoder` | `int` | Количество self-attention слоев в mini encoder. |
| `num_layers_decoder` | `int` | Количество self-attention слоев в mini decoder context (0 = identity). |
| `n_heads` | `int` | Число heads для attention в mini encoder/decoder. `d_e % n_heads == 0`. |
| `d_patch` | `int` | Размер patch token после `encode_patch` (проекция из `mu`). |
| `dropout` | `float` | Dropout mini encoder/decoder. |

### EncoderConfig

| Param | Type | Meaning |
|---|---|---|
| `self_attn_mode` | `str` | Локальное внимание по output-group: `"full"` или `"cls_only"`. |
| `cross_attend_only_cls` | `bool` | Если `True`, латенты cross-attend только к CLS токенам; иначе к CLS+patch tokens. |

### BigVAEConfig

| Param | Type | Meaning |
|---|---|---|
| `d_model` | `int` | Размер токенов encoder/decoder (`patch`, `cls`, `queries`). |
| `d_lat` | `int` | Размер латентных токенов в глобальном bottleneck. |
| `num_latents` | `int` | Число глобальных латентных токенов. |
| `num_encoder_layers` | `int` | Число encoder blocks (local + latent cross/self + FFN). |
| `num_decoder_layers` | `int` | Число decoder blocks (query-to-latent cross-attn + FFN). |
| `n_heads` | `int` | Число attention heads в big encoder/decoder. Требует `d_model % n_heads == 0` и `d_lat % n_heads == 0`. |
| `ffn_mult` | `float` | Множитель hidden-size для FFN блоков. |
| `dropout` | `float` | Dropout big encoder/decoder. |
| `pos_fourier_dim` | `int` | Размер 1D sinusoidal embedding для `o` и `t`; итог в декодере использует `2 * pos_fourier_dim`. |
| `encoder` | `EncoderConfig` | Подконфиг режима локального внимания и cross-attention источника. |

### ModelConfig

| Param | Type | Meaning |
|---|---|---|
| `patch_size` | `int` | Размер patch по оси `d_in`. |
| `distribution` | `DistributionConfig` | Конфиг distribution encoder. |
| `mini_vae` | `MiniVAEConfig` | Конфиг mini patch VAE encoder/decoder. |
| `big_vae` | `BigVAEConfig` | Конфиг full matrix VAE. |
| `beta` | `float` | KL коэффициент (используется в training scripts). |
| `mini_encoder_ckpt_path` | `str` | Путь к pretrained mini encoder checkpoint для загрузки в `BigWeightVAE`. |

### ResamplerConfig

`ResamplerConfig.n_layers` оставлен как backward-compat placeholder и не используется в текущей реализации `BigWeightVAE`.

## Legacy Synthetic Pretrain Config

Все параметры synthetic pretrain вынесены в `PretrainConfig`.

### Training loop

| Param | Meaning |
|---|---|
| `steps` | Число optimizer steps. |
| `log_every` | Частота логов по шагам. |
| `val_every_epochs` | Частота валидации по завершенным эпохам train-split. |
| `val_split` | Доля holdout-валидации. |
| `val_enabled` | Включение holdout-валидации. |
| `val_random_z_probe` | Печатать отдельную валидацию с случайным `z` (probe зависимости от латента). |
| `ablation_steps` | Число начальных шагов абляции. |
| `ablation_zero_dist_conditioning` | В абляции занулять `dist_var_tokens`. |
| `ablation_deterministic_z` | В абляции использовать `z=mu` без reparameterization noise. |
| `grad_monitor_enabled` | Сбор градиентных статистик по слоям. |
| `grad_monitor_every` | Частота сбора grad-RMS. |
| `grad_monitor_weights_only` | Учитывать только weight-тензоры (`ndim>=2`). |
| `grad_monitor_topk_layers` | Сколько слоев показывать на итоговом графике. |
| `grad_monitor_log_scale` | Лог-шкала по оси `y` на графике grad-RMS. |
| `grad_live_csv` | Дописывать CSV в процессе обучения. |
| `grad_live_plot_every` | Частота live-перерисовки PNG графика grad-RMS. |
| `grad_plot_path` | Путь к PNG графику grad-RMS. |
| `grad_csv_path` | Путь к CSV grad-RMS. |
| `xavier_init_enabled` | Применять Xavier init к `dist_encoder` и `mini_vae`. |
| `seed` | Сид для `torch.manual_seed`. |
| `batch_size` | Synthetic batch size для `W` и `X`. |
| `lr` | Learning rate AdamW. |
| `cosine_scheduler_enabled` | Включение warmup+cosine scheduler (`LambdaLR`). |
| `cosine_warmup_steps` | Число warmup шагов. |
| `cosine_min_lr_ratio` | Минимальный множитель LR в конце cosine decay. |
| `weight_decay` | Weight decay AdamW. |
| `adam_beta1` | `beta1` AdamW. |
| `adam_beta2` | `beta2` AdamW. |
| `adam_eps` | `eps` AdamW. |
| `grad_clip_norm` | Целевая глобальная норма для ренормализации градиентов (не clipping). |
| `recon_y_weight` | Вес функционального reconstruction loss по `y`. |
| `recon_w_weight` | Вес reconstruction loss по `w_patch`. |
| `beta` | KL коэффициент в `loss = recon + beta * KL`. |
| `device` | Устройство (`cpu`, `cuda:0`, ...). |
| `save_path` | Путь итогового checkpoint. |
| `print_model_summary` | Печатать ли `MiniPatchVAE` перед train loop. |

### Dataset/cache

| Param | Meaning |
|---|---|
| `dataset_path` | Путь к cached synthetic dataset (`X`, `W`, `meta`). |
| `dataset_num_samples` | Размер synthetic dataset. |
| `regenerate_dataset` | Пересоздать датасет даже если файл уже есть. |
| `shuffle_dataset_each_epoch` | Перемешивать train split каждый epoch. |
| `dataset_seed` | Seed генерации synthetic dataset. |
| `dataset_only` | Только подготовить/проверить датасет и выйти без train loop. |

### Synthetic tensor shapes

| Param | Meaning |
|---|---|
| `n` | Число строк активаций в synthetic `X`. |
| `d_in` | Входная размерность. |
| `d_out` | Выходная размерность synthetic слоя. |
| `patch_size` | Размер patch для `w_patch` и `X_I`. |

### Synthetic input distribution (mixture of Gaussians)

| Param | Meaning |
|---|---|
| `mixture_components` | Количество компонент смеси на feature. |
| `mixture_mean_scale` | Масштаб центров компонент. |
| `mixture_std_min` | Минимальный std компоненты. |
| `mixture_std_max` | Максимальный std компоненты. |
| `mixture_weight_eps` | `eps` для нормализации mixture weights. |

### Synthetic weight distribution

| Param | Meaning |
|---|---|
| `weight_mean` | Mean synthetic весов `W`. |
| `weight_std` | Std synthetic весов `W`. |
| `weight_latent_dim` | Размер низкоразмерного латента для генерации `W` через projection. |
| `weight_projection_seed` | Seed для projection-параметров генератора `W`. |
| `weight_projector_from_input_distribution` | Если `True`, projector `P_b` строится из статистик `X_b`. |
| `weight_projector_input_noise_std` | Доп. шум в projector `P_b(X)` при генерации. |

Режимы `weight_generation` (в `dataset.meta`):
- `input_conditioned_local_feature_projection_v2` (текущий режим при `weight_projector_from_input_distribution=True`): `P_b[:, d]` зависит только от локальных статистик координаты `d`.
- `per_matrix_random_projection` (fallback при `False`): случайный `P_b`, не зависящий от `X`.

### InputDistributionEncodingModule config in synthetic pretrain

| Param | Meaning |
|---|---|
| `dist_k_s` | Число квантилей. |
| `dist_Kq` | Число conv-каналов по quantile axis. |
| `dist_d_var` | Размер per-variable token. |
| `dist_d_dist` | Размер patch distribution embedding. |
| `dist_num_var_attn_layers` | Число variable self-attention слоев. |
| `dist_var_attn_heads` | Число heads variable self-attention. |
| `dist_dcn_num_cross_layers` | Число cross layers в DCNv2. |
| `dist_dcn_deep_hidden` | Hidden-size deep tower DCNv2. |
| `dist_dcn_deep_layers` | Число deep layers DCNv2. |
| `dist_dropout` | Dropout для distribution encoder. |

### MiniPatchVAE config in synthetic pretrain

| Param | Meaning |
|---|---|
| `mini_z_dim` | Размер латента mini-VAE. |
| `mini_d_e` | Hidden размер mini encoder/decoder. |
| `mini_pos_dim` | Размер positional embedding по индексу элемента patch в mini encoder. |
| `mini_num_attn_layers_encoder` | Число encoder-attention слоев. |
| `mini_num_layers_decoder` | Число decoder-attention слоев. |
| `mini_n_heads` | Число attention heads mini-VAE. |
| `mini_d_patch` | Размер patch token из `encode_patch`. |
| `mini_dropout` | Dropout mini-VAE. |

## Hydra configs

### Full model config

`conf/model/weight_quantile_vae.yaml` соответствует `ModelConfig` и его вложенным dataclass.

### Legacy Mini pipeline configs

MiniVAE training configs were moved under `legacy/mini_vae/conf/`.

## Quick run

### Big model smoke test

```bash
python models/weight_quantile_vae.py
```

### Synthetic mini pretrain

```bash
python -m legacy.mini_vae.pretrain_mini_patch_vae
```

### Real mini pipeline (dataset/collector based)

```bash
legacy/mini_vae/scripts/train_mini_vae.sh
```
