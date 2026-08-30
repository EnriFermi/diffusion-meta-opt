Да. Я бы разделил работу на **две независимые задачи**:

1. сделать block inference реально дешёвым на уровне CUDA/KV-cache;
2. только после этого решать алгоритмическую проблему качества при малом `steps/block`.

Сейчас у вас обе проблемы наложены друг на друга. В текущем виде BD3 платит слишком большой системный overhead за каждый блок.

## 1. Первое исправление — полностью убрать `torch.cat` из KV-cache

Сейчас концептуально происходит что-то вроде:

```python
k = torch.cat([cached_k, current_k], dim=-2)
v = torch.cat([cached_v, current_v], dim=-2)
```

на **каждом слое, каждом denoising step, каждом block**.

Так делать не надо.

Нужен один заранее выделенный буфер на каждый layer:

```text
K_cache[layer]:
    [batch, heads, max_seq_len, head_dim]

V_cache[layer]:
    [batch, heads, max_seq_len, head_dim]
```

и целочисленный

```text
prefix_len
```

Например, генерируем позиции 64:80:

```text
0                         64        80
|------ clean cache -------| current |
```

На очередном denoising step:

```python
K_cache[:, :, 64:80] = current_k
V_cache[:, :, 64:80] = current_v
```

а attention получает view:

```python
K = K_cache[:, :, :80]
V = V_cache[:, :, :80]
```

Никаких reallocations и копирования prefix.

То есть текущий noisy block просто **перезаписывает одни и те же 16 рабочих slots** на каждом diffusion step.

Это должно быть исправлением №1.

---

## 2. Prefix KV никогда не пересчитываем

После того как block стал clean:

```text
block 0 = clean
block 1 = clean
block 2 = currently denoising
```

K/V блоков 0–1 должны физически лежать в кеше и больше никогда не трогаться:

```text
persistent:
[ block0 ][ block1 ]

scratch:
                   [ block2 ]
```

На каждом шаге block2 считаются только:

[
Q_{b},K_b,V_b,
]

а attention:

[
Q_b
\quad\text{against}\quad
[K_{\rm prefix},K_b].
]

Не должно быть никакого повторного projection prefix через Transformer.

В идеальном случае стоимость layer становится примерно:

[
O(Bd^2) + O(BLd)
]

вместо MDLM:

[
O(Ld^2)+O(L^2d).
]

Вот именно это и является реальным computational advantage block diffusion.

---

# 3. Использовать настоящий KV-cache kernel

Простой preallocated PyTorch cache уже лучше, но на H100 я бы в итоге не оставлял:

```python
torch.cat(...)
F.scaled_dot_product_attention(...)
```

а использовал kernel типа **FlashAttention KV-cache**.

Нужная семантика выглядит примерно так:

```text
q: current 16 tokens

K_cache:
[clean prefix | current noisy block]

cache_seqlen = prefix_len
```

На каждом denoising step новые K/V текущего блока записываются в:

```text
[prefix_len : prefix_len+B]
```

при этом `prefix_len` не увеличивается.

Следующий diffusion step просто перезаписывает эти же slots.

Таким образом:

```text
step 1:
[prefix clean][noisy block v1]

step 2:
[prefix clean][noisy block v2]

step 3:
[prefix clean][noisy block v3]
```

Prefix вообще не двигается в памяти.

На H100 это как раз тот режим, где FlashAttention/FA3 может дать заметное преимущество.

---

# 4. Самая важная оптимизация: убрать отдельный `_append_cache()`

Вот это я считаю наиболее интересным изменением.

Сейчас после получения:

```text
block_i = CLEAN
```

вы делаете ещё один Transformer forward:

[
\text{clean block}_i
\longrightarrow
KV_i,
]

чтобы следующий block мог использовать его как prefix.

При одном denoising step на block получается ужасная ситуация:

```text
1 useful diffusion forward
+
1 cache-building forward
```

То есть **50% backbone calls вообще уходят на cache maintenance**.

### Можно сделать лучше

Не кешировать clean block сразу.

Допустим закончили:

```text
block_i
```

и хотим начать:

```text
block_{i+1}.
```

Первый forward следующего блока делаем сразу на:

```text
[ CLEAN block_i | NOISY block_{i+1} ]
```

а старый prefix уже лежит в cache:

```text
CACHE:
[0 ... i-1]

ACTIVE:
[clean i | noisy i+1]
```

Attention mask:

```text
clean block_i:
    sees cached prefix + itself

noisy block_i+1:
    sees cached prefix + clean block_i + itself
```

То есть в **одном Transformer forward** мы одновременно:

1. вычисляем правильные clean K/V для предыдущего блока;
2. записываем их в persistent cache;
3. делаем первый denoising step следующего блока.

Получается pipeline:

```text
block 0:
    denoise

block 1, step 1:
    [clean block0 | noisy block1]
       │               │
       └→ cache         └→ logits

block 1, step 2...:
    only [noisy block1]

block 2, step 1:
    [clean block1 | noisy block2]
       │               │
       └→ cache         └→ logits
```

Это особенно полезно именно для вашего low-NFE режима.

Вместо:

[
N_{\text{blocks}}
(N_{\text{step}}+1)
]

backbone calls получаем почти:

[
N_{\text{blocks}}N_{\text{step}}.
]

При `1 step/block`:

сейчас примерно

[
16+16=32
]

calls.

После fusion:

[
\boxed{16}
]

calls.

Это потенциально очень большое изменение latency.

---

# 5. Причём последний block вообще не надо кешировать

Сейчас проверьте, не вызывается ли cache append для последнего блока.

Если sequence уже закончена:

```text
block15 → final tokens → return
```

его KV никому больше не нужны.

То есть:

```python
if not last_block:
    append_cache(...)
```

Минимальная оптимизация, но бесплатная.

---

# 6. Не делать CPU↔GPU synchronization внутри sampler

Вот такие вещи:

```python
tensor.sum().item()
entropy.mean().item()
mask.any().item()
```

внутри diffusion loop очень плохи для latency measurement.

`.item()` заставляет CPU ждать завершения CUDA.

На 256-token MDLM forward это может быть терпимо.

На 16-token BD3 forward:

[
T_{\rm kernel}
]

становится маленьким, и synchronization начинает занимать существенную долю общего времени.

В fast path должно быть примерно:

```text
GPU:
forward
sampling
mask update
forward
sampling
mask update
...
```

без возврата на CPU.

Метрики/entropy/logging:

```python
if debug:
    ...
```

полностью отключаются во время benchmark.

---

# 7. Sampling тоже полностью оставить на GPU

Никаких:

```python
cpu()
numpy()
Python loops over tokens
```

между NFE.

Все операции:

```text
softmax
categorical sampling
mask selection
first-hitting
token update
```

должны быть batched CUDA ops.

Например:

```python
tokens = torch.where(reveal_mask, sampled, tokens)
```

а не циклы по позициям.

---

# 8. Static shapes

Для H100 это очень важно.

Сейчас длина prefix:

```text
0
16
32
48
...
240
```

меняется.

Лучше cache всегда иметь:

```text
[batch, heads, 256, head_dim]
```

и передавать отдельно:

```text
cache_seqlen
```

а не создавать tensors новой формы.

Это сильно облегчает:

* `torch.compile`;
* CUDA graphs;
* FlashAttention;
* memory allocation.

---

# 9. `torch.compile` — только после исправления cache

Я бы не начинал с compile.

Сначала:

```text
static preallocated KV
no torch.cat
no .item()
no allocations
```

и только потом:

```python
model = torch.compile(model, mode="reduce-overhead")
```

Особенно полезно компилировать именно:

```text
one_block_denoise_step()
```

с фиксированным:

```text
B=16
batch=1
```

---

# 10. CUDA Graph — очень подходит вашему случаю

У вас:

```text
batch = 1
block size = 16
Transformer shape fixed
```

Это почти идеальный сценарий для CUDA Graph.

Основная проблема — растущий `cache_len`, но сам cache tensor можно сделать фиксированного размера, а актуальную длину передавать как tensor/scalar.

Тогда можно capture'нуть почти весь denoising step:

```text
QKV
attention
MLP
logits
sampling
token update
```

и replay его без Python/kernel-launch overhead.

Для маленького block=16 это может быть особенно важно.

---

# 11. Профилировать надо не весь sampler, а один layer/forward

Я бы сделал отдельный benchmark:

### MDLM

```text
forward(L=256)
```

### BD3

```text
forward(
    q_len=16,
    cache_len=0
)

forward(
    q_len=16,
    cache_len=64
)

forward(
    q_len=16,
    cache_len=128
)

forward(
    q_len=16,
    cache_len=240
)
```

И отдельно:

```text
cache update
sampling
Python overhead
```

После оптимизации хочется увидеть что-нибудь вроде:

[
T(B=16,prefix=128)
\ll
T(L=256).
]

Если сейчас получается:

[
T_{16}\approx0.6T_{256},
]

BD3 практически обречён.

Если доведём до:

[
T_{16}\approx0.1-0.2T_{256},
]

block generation уже начинает иметь смысл.

---

# 12. Затем подобрать block size заново

`B=16` — не обязательно оптимум по latency.

При (L=256):

### B=16

[
16\text{ sequential blocks}
]

### B=32

[
8\text{ blocks}
]

### B=64

[
4\text{ blocks}.
]

Чем больше block:

* меньше sequential launches;
* меньше cache transitions;
* лучше GPU utilization;

но хуже качество при агрессивном parallel decoding.

Поэтому после cache optimization я бы построил sweep:

| block size | steps/block |
| ---------: | ----------: |
|          8 |     1,2,4,8 |
|         16 |     1,2,4,8 |
|         32 |     1,2,4,8 |
|         64 |     1,2,4,8 |

и сравнивал строго:

[
\boxed{\text{quality vs measured wall-clock}}
]

Вполне возможно, что на H100 + length 256 оптимум окажется **B=32**, а не 16.

---

# 13. Но есть ещё алгоритмическая проблема

Допустим мы идеально ускорили KV-cache.

BD3 `1 step/block` всё равно может иметь плохой MAUVE/PPL.

Это уже не implementation bug.

Стандартный BD3 objective учит хорошую модель:

[
p_\theta(x_0\mid x_t,prefix),
]

но не оптимизирует специально:

> «сгенерируй весь 16-token block хорошо ровно за один forward».

Поэтому после инженерных исправлений следующий этап — **дистилляция под малое NFE**.

И как раз здесь ваша работа с CFM/Flow Maps может стать гораздо интереснее обычного BD3.

Например:

```text
MDLM
   ↓
BD3
   ↓
block model trained for 1–4 jumps
```

То есть вместо:

[
16\text{ diffusion steps/block}
]

добиться:

[
1-4\text{ learned jumps/block}.
]

Тогда KV-cache начинает давать уже настоящий выигрыш.

---

# Я бы сделал это в таком порядке

1. **Static preallocated KV-cache.** Убрать абсолютно все `torch.cat`.
2. **Scratch slots для текущего block.** Каждый diffusion step перезаписывает те же `B` K/V positions.
3. **Убрать `.item()`/logging/synchronizations** из fast sampler.
4. **Не cache'ить последний block.**
5. **Fuse cache commit предыдущего block с первым forward следующего** через `[clean_prev | noisy_current]`.
6. Перевести attention на **FlashAttention KV-cache / equivalent fused kernel**.
7. `torch.compile`.
8. CUDA Graph.
9. Перемерить `B=8/16/32/64`.
10. Построить crossover по длинам:
    [
    L=256,512,1024,2048,4096.
    ]
11. И только потом, если BD3 всё равно требует 8–16 steps/block, заниматься **low-NFE distillation / Flow Map block model**.

### Какой результат я бы считал успехом

Не обязательно заставлять block model выигрывать MDLM уже при (L=256). На H100 это может быть физически невыгодный режим из-за очень дешёвого full-sequence forward.

Я бы хотел получить такую картину:

```text
L=256     MDLM ≲ block
L=512     примерно одинаково
L=1024    block быстрее
L=2048    block значительно быстрее
L=4096    block сильно быстрее
```

А затем low-NFE дистилляцией сдвигать crossover влево.

**Самое перспективное конкретное изменение для вашего текущего кода — это комбинация `preallocated KV + in-place scratch slots + fused clean-prev/current-first-step`.** Она одновременно убирает `torch.cat` и практически ликвидирует тот самый дополнительный (s=t=1) forward между блоками. Это я бы реализовывал первым.
