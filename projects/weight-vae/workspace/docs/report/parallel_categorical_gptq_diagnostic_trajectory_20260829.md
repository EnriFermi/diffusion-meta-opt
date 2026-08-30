# Categorical GPTQ: расхождение optimized и detached diagnostics

Дата: 2026-08-29 UTC. Срез production-run до шага 3660.

## Наблюдаемый эффект

После быстрого выхода на unconditional categorical prior основной objective
почти плоский, но direction и same-layout contrastive diagnostics улучшаются.
Вопрос этого аудита: является ли это шумом/сменой состава minibatch или модель
действительно начала извлекать слабый sample-specific directional signal.

Raw source:
`/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/train_metrics.jsonl`.
Точный `size + mtime_ns` среза записан в `snapshot_metadata.json`.

## Validity и composition controls

- В live-файле нет дубликатов, пропущенных ожидаемых шагов и non-finite
  значений; `committed_logical_index == step * 32` для каждой строки. Trainer
  был жив при проверке.
- Все `diagnostic_old_*` считаются под `no_grad` после hard argmax decode и не
  входят в backward. Backward objective равен `code_ordinal + scale_ordinal`.
- Случайный top-1 для каждого contrastive batch вычислен точно как
  `group_count / eligible_samples`. Он не изменился материально: 0.3929 в окне
  510--1000 против 0.3894 в окне 3010--3660.
- Fixed-effect regression с одинаковой сигнатурой состава негативов
  `(eligible, groups, min/median/max group size)` сохраняет contrastive-тренд:
  CE slope -0.0419 на 1000 шагов, top-1 advantage slope +0.0646 на 1000 шагов,
  diagonal-minus-hardest-negative slope +0.00827 на 1000 шагов.
- Independent target/composition proxies стабильны: teacher continuous NRMSE
  0.1621 -> 0.1635, teacher operator metric 0.0012460 -> 0.0012464, code target
  boundary fractions практически не изменились.

## Matched-window results

Средние; `510--1000` содержит 50 logged batches, `3010--3660` содержит 66.

| Metric | 510--1000 | 3010--3660 | Изменение |
|---|---:|---:|---:|
| optimized total ordinal | 0.085438 | 0.084946 | -0.58% |
| optimized code ordinal | 0.082641 | 0.082380 | -0.32% |
| optimized scale ordinal | 0.002797 | 0.002566 | -8.27% |
| code accuracy | 0.16994 | 0.17040 | +0.046 pp |
| hard raw NRMSE | 1.001981 | 1.001324 | -0.000657 |
| behavioral direction loss | 0.97523 | 0.94925 | -0.02598 |
| structural direction loss | 0.96427 | 0.94717 | -0.01711 |
| same-layout contrastive CE | 0.88137 | 0.78161 | -11.32% |
| same-layout top-1 | 0.44576 | 0.61886 | +17.31 pp |
| exact random top-1 chance | 0.39289 | 0.38944 | -0.35 pp |
| top-1 advantage over chance | 0.05286 | 0.22942 | 4.34x |
| diagonal similarity | 0.00765 | 0.03391 | 4.43x |
| off-diagonal similarity | 0.00220 | 0.00488 | 2.22x |
| diagonal - hardest negative | 0.00151 | 0.02290 | 15.1x |
| latent interaction RMS ratio | 0.04501 | 0.25090 | 5.57x |
| actual operator diagnostic | 0.028115 | 0.028086 | flat |
| detached old total | 28.425 | 24.309 | -14.48% |

`code_ordinal` по batch means остается статистически почти плоским: raw slope
после шага 510 равен -0.000101 на 1000 шагов с SE 0.000063. Total loss падает
слабо главным образом благодаря scale branch. На этом фоне direction и
contrastive trends велики и устойчивы.

## Почему скаляры могут расходиться

Они оценивают разные свойства и имеют разные редукции.

1. Code ordinal loss использует **soft** cumulative class probabilities и
   усредняется по всем valid scalar codes. После того как модель выучила
   unconditional distribution, небольшая conditional поправка растворяется в
   огромной prior-составляющей.
2. Detached structural direction сначала берет **hard argmax**, затем считает
   `1 - cosine` для 16-мерных patches и сильнее весит patches с большой target
   norm. Небольшой сдвиг logits через argmax boundary может почти не изменить
   soft NLL, но резко изменить направление hard patch.
3. Same-layout contrastive требует лишь, чтобы правильный target был немного
   ближе 1--3 negatives. Поздний absolute diagonal similarity все еще мал
   (0.0339), но off-diagonal еще меньше (0.0049). Поэтому top-1 может стать 62%,
   хотя абсолютная реконструкция остается плохой.
4. Positive per-output scale сокращается при нормализации structural patch
   directions и не может объяснить structural/contrastive improvement. Значит,
   этот сигнал находится именно в hard code pattern, а не только в scale head.

## Mechanism verdict

Старое описание «code branch полностью сидит в static prior и latent collapsed»
уже неполно. Начиная примерно с 1000--1500 шагов идет медленный выход:
sample-by-slot latent interaction растет монотонно, а hard code patterns несут
все больше same-layout sample identity и слабого target-direction alignment.

Это еще не полезная weight reconstruction. Code accuracy практически не
сдвинулась, hard NRMSE остается на zero-output baseline, actual operator metric
плоский. Поддержан узкий вывод: **возник слабый sample-specific directional
carrier, который direction/contrastive diagnostics замечают намного раньше,
чем глобально усредненный categorical objective**.

Не установлено, превратится ли carrier в снижение абсолютной ошибки, или модель
остановится на contrastive fingerprint. Для различения нужен следующий
checkpoint probe: exact-layout `z` swap плюс distribution argmax-margin и
energy-stratified code metrics. Текущий рост latent interaction сам по себе не
дает основания добавлять anti-collapse loss: он уже перестал быть активной
проблемой, а auxiliary может поощрить бесполезную вариативность.

## Artifacts

- `docs/report/parallel_categorical_gptq_diagnostic_trajectory_20260829/window_summary.csv`
- `docs/report/parallel_categorical_gptq_diagnostic_trajectory_20260829/plateau_slopes.csv`
- `docs/report/parallel_categorical_gptq_diagnostic_trajectory_20260829/trajectory.png`
- `docs/report/parallel_categorical_gptq_diagnostic_trajectory_20260829/snapshot_metadata.json`
- `scripts/analyze_categorical_diagnostic_trajectory.py`
