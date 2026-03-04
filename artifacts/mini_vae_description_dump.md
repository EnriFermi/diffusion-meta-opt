# Mini VAE: подробный технический dump

## 0) Что именно описано
Этот документ описывает **реальный** (`implementation=real`) Mini patch-level VAE и его обучение в текущем коде.

Основные точки входа:
- Модель: `models/weight_quantile_vae.py`
  - `InputDistributionEncodingModule`
  - `MiniPatchEncoder`
  - `CrossAttnPatchDecoder`
  - `MiniPatchVAE`
- Тренировочная обертка: `training/mini_patch_training_model.py`
  - `MiniPatchTrainingModel.forward`
- Train loop: `experiments/train_mini_vae.py`
  - `sample_patch_batch`
  - основной шаг оптимизации

## 1) End-to-end call graph (кто куда идет)

```text
run_worker (experiments/train_mini_vae.py)
  -> fetch_batch(...) -> (x, W)
  -> sample_patch_batch(x, W, patch_size, patches_per_sample)
       returns X_full, X_patch, w_patch, patch_idx, out_idx
  -> model(...)
       where model = MiniPatchTrainingModel
       -> MiniPatchTrainingModel.forward(...)
            1) normalize_patch_weights(w_patch) -> w_patch_norm
            2) project_patch_weights_to_unit_sphere(w_patch_norm) -> w_patch_unit
            3) distribution_encoder(X_full, patch_idx)
               -> dist_var_tokens, dist_patch_embed
            4) mini_vae(w_patch_unit, dist_var_tokens, dist_patch_embed)
               -> w_hat_raw, mu, logvar, z
            5) normalize_prediction_with_stopgrad_norm(w_hat_raw) -> w_hat_unit
            6) structural_loss = MSE(w_hat_unit, w_patch_unit)
            7) behavioral_loss = MSE(X_patch_norm @ w_hat_unit, X_patch_norm @ w_patch_unit)
            8) contrastive_loss (optional)
            9) kl_loss = mini_vae.kl_loss(mu, logvar)
           10) total_loss = recon_mix + contrastive_coef * contrastive_loss + kl_coef * kl_loss
  -> backward / clip / optimizer.step / scheduler.step
```

## 2) Откуда берутся тензоры для MiniVAE

### 2.1 Исходный layer sample
В train loop из датасета приходит один sample:
- `x`: `[n, d_in]`
- `W`: `[d_in, d_out]`

### 2.2 Patch sampling (`sample_patch_batch`)
Функция: `experiments/train_mini_vae.py::sample_patch_batch`.

Внутри:
1. Выбирается `out_idx`: случайные индексы выходных каналов
   - `out_idx ~ randint(0, d_out, size=[B_p])`
2. Выбирается `patch_t`: случайный номер patch-слота
   - `num_patch_slots = ceil(d_in / patch_size)`
   - `patch_t ~ randint(0, num_patch_slots, size=[B_p])`
3. Формируется `patch_idx`:
   - `patch_idx = patch_t[:,None] * patch_size + arange(patch_size)[None,:]`
   - `clamp(max=d_in-1)`
4. Строятся тензоры:
   - `X_full`: `[B_p, n, d_in]` (реплика `x` по batch-патчам)
   - `X_patch`: `[B_p, n, p]` (gather по `patch_idx`)
   - `W_col`: `[B_p, d_in]` (выбранные колонки `W` по `out_idx`)
   - `w_patch`: `[B_p, p]` (gather `W_col` по `patch_idx`)

Выход:
- `X_full, X_patch, w_patch, patch_idx, out_idx`

## 3) Distribution conditioning path

Класс: `InputDistributionEncodingModule` (`models/weight_quantile_vae.py`).

Вход:
- `X`: `[B, n, d_in]`
- `patch_idx`: `[B, p]`

Выход:
- `dist_var_tokens`: `[B, p, d_var]`
- `dist_patch_embed`: `[B, d_dist]`

Пайплайн внутри `forward`:
1. `X_I = gather(X, patch_idx)` -> `[B, n, p]`
2. Квантили по оси sample `n`:
   - `_quantile_over_samples(X_I, quantile_probs)` -> `[k_s, B, p]`
   - перестановка -> `q: [B, k_s, p]`
3. Нормализация квантилей в `[-1, 1]` по каждой переменной (`p`),
   плюс фиксация крайних квантилей в -1 и 1.
4. Моменты исходного `X_I`:
   - `mu = mean(X_I, dim=1)` `[B, p]`
   - `sigma = std(X_I, dim=1)` `[B, p]`
   - `mu_log = sign(mu) * log1p(abs(mu))`
   - `sigma_log = log(sigma + eps)`
5. 1D conv по quantile-оси:
   - `q_var_major: [B, p, k_s]`
   - reshape -> `[B*p, 1, k_s]`
   - `Conv1d(kernel=3, no padding)` -> `[B*p, 1, k_s-2]`
   - reshape -> `q_conv: [B, p, k_s-2]`
6. Concatenate features:
   - `f = concat(q_conv, mu_log, sigma_log)` -> `[B, p, k_s]`
7. Shared var MLP:
   - `v = var_mlp(f)` -> `[B, p, d_var]`
8. Variable Transformer encoder:
   - `dist_var_tokens = var_encoder(v)` -> `[B, p, d_var]`
9. Mean pool по `p`:
   - `v_pool = mean(dist_var_tokens, dim=1)` -> `[B, d_var]`
10. DCNv2:
   - `dist_patch_embed = dcn(v_pool)` -> `[B, d_dist]`

## 4) MiniPatchVAE (реализация `real`)

### 4.1 Encoder: `MiniPatchEncoder.encode`
Вход:
- `w_patch`: `[B, p]`
- `dist_var_tokens`: `[B, p, d_var]`

Шаги:
1. Позиционные эмбеддинги токенов патча:
   - `token_tau = (arange(p) + 0.5) / p` -> `[p]`
   - `pos = sinusoidal_embedding(token_tau, pos_dim)` -> `[p, pos_dim]`
   - expand -> `[B, p, pos_dim]`
2. Сбор per-element токена:
   - `t = concat(w_patch[...,None], dist_var_tokens, pos)`
   - `t: [B, p, 1 + d_var + pos_dim]`
3. Проекция в model-space:
   - `e = elem_embed(t)` -> `[B, p, d_e]`
4. Выбор размерности латентов энкодера:
   - `latent_dim = encoder_latent_dim`, если `encoder_latent_dim > 0`, иначе `latent_dim = d_e`
5. Slot-anchoring латентов:
   - фиксированные позиции слотов: `slot_tau = (arange(L_lat) + 0.5) / L_lat` -> `[L_lat]`
   - `slot_pos = sinusoidal_embedding(slot_tau, pos_lat_dim)` -> `[L_lat, pos_lat_dim]`
   - `latents0 = slot_pos_to_latent(slot_pos)` -> `[L_lat, latent_dim]`
   - expand -> `latents: [B, L_lat, latent_dim]`
6. Токены для cross-attn в Perceiver:
   - `tokens_with_w = concat(e, w_patch[...,None])` -> `[B, p, d_e + 1]`
7. Perceiver resampling (с RoPE в attention):
   - для каждого `PerceiverResamplerBlock(d_latent=latent_dim, d_token=d_e+1)`:
     - cross-attn: `latents <- tokens_with_w`, RoPE с позициями
       - `latent_pos = slot_tau`
       - `token_pos = token_tau`
     - self-attn по латентам: RoPE с `latent_pos`
     - FFN по латентам
8. Нормализация и heads:
   - `latents = LayerNorm(latents)` (по последней оси `latent_dim`)
   - `h = reshape(latents, [B, L_lat * latent_dim])`
   - `mu = to_mu(h)` -> `[B, z_dim]`
   - `logvar = to_logvar(h)` -> `[B, z_dim]`
9. Важный момент про размерности:
   - в `real` сейчас `z_dim` может быть произвольным, heads делают проекцию `Linear(L_lat*latent_dim -> z_dim)`.

### 4.2 Reparameterization
Если `use_latent_sampling=True`:
- `eps ~ N(0, I)`
- `z = mu + eps * exp(0.5 * logvar)`

Иначе:
- `z = mu`

### 4.3 Decoder: `CrossAttnPatchDecoder.forward`
Вход:
- `z`: `[B, z_dim]`
- `patch_size = p`
- `dist_patch_embed`: `[B, d_dist]` (optional)

Примечание по размерностям:
- `decoder` работает в `d_model = d_e` (он не использует `encoder_latent_dim` напрямую).

Шаги:
1. Латентные токены из `z`:
   - `lat = z_to_latents(z).view(B, L_lat, d_model)`
2. Опциональное conditioning от distribution:
   - `dist_lat = dist_to_latents(dist_patch_embed).view(B, L_lat, d_model)`
   - если `decoder_dist_mode="add"`: `lat = lat + dist_lat`
   - если `decoder_dist_mode="concat"`: `lat = concat(lat, dist_lat, dim=1)`
3. Query-токены:
   - learnable `query_seed: [d_model]`
   - `q = query_seed[None,None,:].expand(B, p, d_model)`
4. Позиции для RoPE:
   - `q_pos = arange(p)` для query-токенов
   - `kv_pos = arange(lat_len)` для latent key/value токенов (`lat_len = L_lat` или `2*L_lat` при `concat`)
5. Decoder blocks (`CrossAttnBlock`) на каждом слое:
   - cross-attn `q <- lat` с RoPE (`q_pos`, `kv_pos`)
   - self-attn по `q` с RoPE (`q_pos`)
   - FFN
6. Heads:
   - `u_hat = direction_head(q).squeeze(-1)` -> `[B, p]`
   - `s_hat = scale_head(mean(q,dim=1)).squeeze(-1)` -> `[B]`
7. Direction + scale decode (`_decode_direction_and_logscale`):
   - `u = u_hat / (||u_hat|| + eps)`
   - log-scale barrier:
     - нижняя: `s = s_min + softplus(s - s_min)`
     - верхняя (если включена): `s = s_max - softplus(s_max - s)`
   - итог: `w_hat = u * exp(s)`

Выход `MiniPatchVAE.forward`:
- `w_hat, mu, logvar, z`

## 5) Wrapper-level нормализация и objective

Класс: `MiniPatchTrainingModel.forward`.

### 5.1 Нормализация target patch
1. `w_patch_norm = w_patch / rms(w_patch)`
2. `w_patch_unit = normalize(w_patch_norm, p=2)`

### 5.2 Прямой проход VAE
- `dist_var_tokens, dist_patch_embed = distribution_encoder(X_full, patch_idx)`
- `w_hat_raw, mu, logvar, z = mini_vae(w_patch_unit, dist_var_tokens, dist_patch_embed)`

### 5.3 Нормализация prediction (stop-grad denom)
- `w_hat_unit = w_hat_raw / detach(||w_hat_raw|| + eps)`

Важно: denom detached, т.е. градиент через норму предсказания не идет.

### 5.4 Лоссы
1. Structural:
- `L_struct = MSE(w_hat_unit, w_patch_unit)`

2. Behavioral:
- Сначала нормируется `X_patch` по RMS на sample-level
- `y = einsum("bnp,bp->bn", X_patch_norm, w_patch_unit)`
- `y_hat = einsum("bnp,bp->bn", X_patch_norm, w_hat_unit)`
- `L_beh = MSE(y_hat, y)`

3. Contrastive (если `contrastive_coef > 0`):
- из `X_full[0], W_full` строятся две function-preserving view
  - permute input dims
  - random sign flips
- те же `patch_idx` переводятся в индексы каждой view
- строятся `w_patch_view1/2`, `dist_var_view1/2`
- кодируются `mu_view1/2`
- `L_con = NT-Xent(mu_view1, mu_view2, temperature)`

4. KL:
- `L_kl = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)` усредненный по batch

Итог:
- `L_recon_mix = structural_coef * L_struct + behavioral_coef * L_beh`
- `L_total = L_recon_mix + contrastive_coef * L_con + kl_coef * L_kl`

## 6) Как именно идет backward и optimizer-step

В основном цикле (`experiments/train_mini_vae.py`):

1. На каждом global step:
- `optimizer.zero_grad(set_to_none=True)`
- есть `grad_accum_steps` micro-итераций

2. В micro-итерации:
- fetch sample (`x, W`)
- `sample_patch_batch(...)`
- forward под `autocast`
- `loss_for_backward = total_loss / grad_accum_steps`
- проверка finite (в DDP через `all_reduce(MIN)`)
- backward:
  - `scaler.scale(loss).backward()` при FP16 AMP
  - или обычный `loss.backward()`

3. После накопления micro-step:
- `scaler.unscale_(optimizer)` (если AMP scaler)
- если активен `freeze_decoder_steps` и `global_step <= freeze_decoder_steps`:
  - у optimizer group `mini_decoder` градиенты принудительно `None`
- grad clipping (`clip_grad_norm_`) при `grad_clip_norm > 0`
- `optimizer.step()` / `scaler.step(optimizer)`
- `scheduler.step()`

## 7) Мониторинг градиентов в train loop

Сейчас есть:
- global/group grad stats (`distribution_encoder`, `mini_encoder`, `mini_decoder`, ...)
- per-layer monitor (при включении telemetry):
  - `grad_rms` pre/post clip
  - `param_rms`
  - `grad_to_param_ratio` pre/post
  - CSV + optional plots

И дополнительные сигналы:
- `dL/dz`, `dL/dmu` (retain_grad на латентах)
- `patch_unique_ratio`, `out_idx_unique_ratio`

## 8) Конфигурационные переключатели (ключевые)

`MiniVAEConfig`:
- `z_dim, d_e, encoder_latent_dim, pos_dim, n_heads`
- `pos_lat_dim`
- `num_attn_layers_encoder, num_layers_decoder, decoder_L_latents`
- `decoder_use_dist_conditioning, decoder_dist_mode`
- `use_latent_sampling`
- `implementation` (`real` / `mlp_stub`)
- `stub_resampler_d_model` (для `mlp_stub`)

Размерные инварианты:
- `real`: `z_dim` свободный (через `Linear(L_lat*latent_dim -> z_dim)`), где `latent_dim = encoder_latent_dim` (если `>0`), иначе `d_e`
- `mlp_stub`: `z_dim = decoder_L_latents * resampler_d_model`, где `resampler_d_model = stub_resampler_d_model` (если `>0`), иначе `d_e`

Trainer/loss:
- `structural_coef, behavioral_coef, contrastive_coef, kl_coef`
- `contrastive_temperature`
- `freeze_decoder_steps`
- AMP/grad_accum/grad_clip/scheduler

## 9) Что меняется в stub-варианте (`implementation=mlp_stub`)

Если выбран stub:
- encoder: `TransformerNoCompressionPatchEncoder`
  - игнорирует `dist_var_tokens`
  - вход в `elem_embed`: `concat(w_i, sinusoidal(arange(p)))`
  - токены после `elem_embed`: `[B, p, d_e]`
  - латенты: размер `resampler_d_model`, где
    - `resampler_d_model = stub_resampler_d_model`, если `> 0`
    - иначе `resampler_d_model = d_e`
  - токены для cross-attn: `tokens_with_w = concat(e, w_i)` -> `[B, p, d_e+1]`
  - в `PerceiverResamplerBlock`:
    - cross-attn `latents <- tokens_with_w` (RoPE, `latent_pos=latent_tau`, `token_pos=token_tau`, где `token_tau=(i+0.5)/p`)
    - self-attn по латентам (RoPE, `latent_pos`)
  - дальше `latents` flatten: `[B, L_lat * resampler_d_model]`, затем `LayerNorm(flat)` -> `mu, logvar`
  - жесткая проверка: `z_dim = L_lat * resampler_d_model`
- decoder: `MLPNoCompressionPatchDecoder`
- `MiniPatchVAEStub.kl_loss` возвращает ноль

То есть реальная VAE-геометрия и KL-динамика здесь intentionally упрощены.

---

## 10) Короткий shape-checkpoint (реальный путь)

- `x`: `[n, d_in]`
- `W`: `[d_in, d_out]`
- `X_full`: `[B_p, n, d_in]`
- `X_patch`: `[B_p, n, p]`
- `w_patch`: `[B_p, p]`
- `patch_idx`: `[B_p, p]`
- `out_idx`: `[B_p]`
- `dist_var_tokens`: `[B_p, p, d_var]`
- `dist_patch_embed`: `[B_p, d_dist]`
- `mu, logvar, z`: `[B_p, z_dim]`
- `w_hat_raw, w_hat_unit`: `[B_p, p]`
