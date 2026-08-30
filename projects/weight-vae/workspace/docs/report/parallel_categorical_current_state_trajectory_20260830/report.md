# Parallel categorical GPTQ: mechanistic trajectory audit at 98k

Дата среза: 2026-08-30 UTC. Анализ полностью read-only; trainer не получал
сигналов и его состояние не изменялось. Срез содержит train metrics до шага
98,590, gradient telemetry до 98,500 и 98 полных 1000-step bins. Для выводов о
тренде частичный live-tail после шага 98,000 не использовался.

## Failure definition

Модель уже вышла из исходного unconditional-prior basin, но после примерно
32k перешла в длинное плато, существенно выше доступного GPTQ teacher floor.

Последнее полное 5k-окно, шаги 93,010--98,000, имеет медианы:

| Metric | Current | Reference |
|---|---:|---:|
| total optimized loss | 0.077923 | 0.085346 на prior plateau 510--1000 |
| code ordinal | 0.076207 | 0.082706 |
| scale ordinal | 0.001629 | 0.002795 |
| scale fraction of optimized loss | 2.08% | 3.31% |
| hard raw NRMSE | 0.93259 | zero-output = 1; GPTQ teacher = около 0.163 |
| actual operator metric | 0.019060 | teacher = около 0.001206; ratio 16.18x |
| behavioral direction loss | 0.67619 | 0.98023 |
| structural direction loss | 0.76372 | 0.96530 |
| same-layout contrastive top-1 | 0.76095 | batch-exact chance 0.38889 |
| code accuracy / MAE | 17.88% / 1.862 bins | 17.02% / 1.939 bins |
| scale accuracy / MAE | 8.51% / 4.577 bins | 6.25% / 5.926 bins |

То есть reconstruction теперь лучше нулевого выхода и sample identity реально
присутствует, но большая часть teacher-achievable reconstruction еще не
извлечена. Scale стал лучше исходного состояния, однако после раннего падения
его MAE практически застыл около 4.5--4.7 bins.

Источник чисел: `window_summary.csv`; полная траектория показана в
`trajectory_overview.png`.

## Validity checks

- 9,860 metric rows и 986 gradient rows; дубликатов и пропущенных ожидаемых
  logged steps нет.
- Все исходные числовые поля конечны; `loss = code + scale` с максимальной
  ошибкой 1.49e-8.
- `committed_logical_index = step * 32` для всех строк; LR всюду 5e-5.
- Schema одинакова во всех строках; elapsed time монотонен.
- Ни одного clipping event: максимум за весь run 2.472 при threshold 5.0;
  late-5k p99 только 0.0665.
- Все parameter groups имеют ненулевой gradient; `none_parameter_tensors = 0`
  в текущем окне.
- Поздняя median local throughput 1.81 step/s, GPTQ tokenization 0.0214 s,
  CUDA peak 13.846 GiB. Признаков runtime/OOM/NaN failure нет.

Точный `path + size + mtime_ns` live-среза и все проверки сохранены в
`snapshot_metadata.json`.

## Data-driven phases

Phase changes искались не по выбранной вручную одной кривой. Десять метрик
(code/scale loss, NRMSE, два direction loss, contrastive CE, latent ratio/RMS,
operator metric и global gradient norm) сначала агрегировались медианой в
полных 1k bins, затем robust-standardized через median/MAD и совместно
сегментировались piecewise-linear dynamic programming. Breakpoints 8k, 17k и
32k воспроизводятся при всех четырех penalties 25/30/35/40. Поздние кандидаты
55k/80k/89k появляются только при одном penalty каждый и не считаются
установленными фазовыми переходами.

| Phase | Mechanically observed state | Representative robust slopes per 1k |
|---|---|---|
| 1--8k | prior capture и появление directional carrier без reconstruction | code -2.8e-6; NRMSE -6.1e-5; latent interaction ratio +0.0550 |
| 8--17k | conditional breakout | code -1.74e-4; NRMSE -0.00262; behavioral direction -0.0134 |
| 17--32k | максимальный reconstruction growth и revival content gradients | code -2.83e-4; NRMSE -0.00340; operator -3.10e-4 |
| 32--98k | noisy plateau / очень медленный drift | code -1.43e-5; scale -9.15e-7; NRMSE +1.61e-4; operator -6.92e-6 |

Полные segment fits находятся в `phase_summary.csv`, а stability audit — в
`phase_changes.json`.

Последнее 5k-окно хуже 88--93k по raw NRMSE, но это пока не доказанная
деградация модели: online batches не являются fixed evaluation panel, teacher
NRMSE в последнем окне также выше (около 0.1657 против 0.1653 в 88--93k и
0.1623 в 83--88k), а robust joint segmentation не выделяет устойчивого позднего
breakpoint. Для temporal degradation claim нужен один fixed panel на двух
checkpoints.

## Latent geometry

Текущий state нельзя корректно назвать повторным latent collapse.

- interaction/total RMS достигает примерно 0.55 около 20--32k, затем падает до
  0.444 в последнем полном bin;
- одновременно total latent RMS непрерывно растет: 2.668 в первом late-phase
  bin и 3.728 в последнем;
- поэтому абсолютная оценка interaction RMS, вычисленная как
  `interaction_ratio * latent_RMS`, не падает, а растет 1.443 -> 1.658.

Иными словами, относительная доля interaction уменьшается из-за более быстрого
раздувания denominator/main effects, а не из-за исчезновения sample-by-slot
carrier. Рост latent norm при почти плоском task quality является реальным
geometry drift. По одной траектории не установлено, является ли он причиной
плато или свободной gauge-like степенью, которую downstream RMS-normalized
decoder в основном компенсирует.

## Gradient mechanism

На раннем prior plateau gradient от code objective в latent действительно был
почти выключен: phase-1 median `d code / dz = 2.06e-9`. Затем он ожил вместе с
reconstruction breakout:

| Gradient metric | Phase 1: 1--8k | Phase 3: 17--32k | Latest 5k |
|---|---:|---:|---:|
| d code / dz RMS | 2.06e-9 | 6.86e-8 | 6.76e-8 |
| d scale / dz RMS | 3.80e-9 | 1.74e-8 | 2.75e-8 |
| scale/code RMS | 1.75 | 0.274 | 0.414 |
| code-scale cosine | +0.014 | +0.061 | +0.070 |

В latest-5k p10 cosine также положителен (+0.034). Поэтому текущий plateau не
объясняется прежней гипотезой о противонаправленности code/scale gradients.
Scale signal не мертв, но примерно в 2.4 раза слабее code signal в latent.

Текущие per-parameter RMS также конечны: decoder block 1 = 3.56e-6, block 5 =
5.48e-7, categorical tail = 3.43e-7, distribution encoder = 2.85e-7, encoder
block 1 = 8.24e-7, encoder block 13 = 6.53e-8. Есть depth attenuation, но нет
dead graph. Эти величины нельзя интерпретировать как aggregate energy shares,
поскольку размеры групп различаются.

## Competing mechanisms and verdict

### H1: latent slots снова collapsed

Prediction: interaction должен падать и в относительных, и в абсолютных
единицах; contrastive top-1 вернется к chance; `d code / dz` исчезнет.

Результат: исключено для текущего state. Absolute interaction растет,
contrastive top-1 0.761 против chance 0.389, code latent gradient жив.

### H2: code и scale уничтожают друг друга в bottleneck

Prediction: устойчиво отрицательный cosine и/или сильная cancellation в
latent.

Результат: исключено как основной текущий механизм. Latest median cosine +0.070
и p10 +0.034.

### H3: clipping, exploding gradients или dead deep path остановили обучение

Prediction: clipping events, non-finite/spiky gradient, исчезнувшие группы либо
runtime failures.

Результат: исключено. Clipping не активировался ни разу, late gradients малы и
конечны, все группы живы.

### H4: исходный unconditional prior остается единственным решением

Prediction: hard NRMSE около 1, contrastive около chance и shuffled/sample
identity неразличима.

Результат: это описывает ранние шаги, но не current state. Траектория уже несет
сильный conditional carrier и дает NRMSE ниже 1. Остаточная marginal shortcut
составляющая возможна, но больше не является полным объяснением.

### H5: objective-pressure mismatch плюс latent norm drift удерживают позднее плато

Установленные proximal facts:

1. После 32k все reconstruction slopes уменьшаются на порядок или становятся
   плоскими, хотя gradients остаются живыми.
2. Scale составляет только 2.08% optimized objective при MAE 4.58 bins и
   получает меньше latent gradient, чем code.
3. Реализованный ordinal loss делит threshold cost на `(classes - 1)^2`.
   Поэтому несколько scale bins в vocabulary 256 почти бесплатны относительно
   нескольких code bins в vocabulary 15.
4. Total latent RMS продолжает расти при почти неизменной reconstruction, тогда
   как абсолютный interaction carrier сохраняется.

Это наиболее согласованное описание текущего режима: optimizer больше не
застрял из-за отсутствия conditional information; он делает небольшие soft
ordinal улучшения и свободный geometry drift, но objective слабо ценит
оставшуюся row-wise scale/hard reconstruction ошибку. Однако trajectory alone
не различает две более глубокие версии механизма: (a) mismatch soft ordinal с
hard argmax/dequantization или (b) decoder/Jacobian уже плохо превращает живой
latent carrier в точные patch values. Для этого нужен fixed-checkpoint probe
soft expected decode vs argmax плюс decoder Jacobian/z-swap на том же batch.

## Narrow conclusion

Текущая модель не collapsed и не технически разваливается. Она прошла три
реальных фазы обучения, сформировала conditional latent carrier и достигла
полезной, но слабой reconstruction. Начиная примерно с 32k она находится в
длинном noisy plateau: code продолжает улучшаться очень медленно, scale почти
перестал улучшаться, hard reconstruction остается примерно в 5.7 раза хуже
GPTQ teacher по NRMSE и в 16 раз хуже по operator metric. Главный текущий
кандидат — objective/readout mismatch при продолжающемся norm drift, а не
latent collapse, code-scale conflict или optimization instability.

## Artifacts

- `trajectory_overview.png` — inspected, readable overview.
- `gradient_trajectory.png` — inspected, readable gradient paths.
- `window_summary.csv` — raw-window robust summaries.
- `phase_summary.csv` — per-phase medians and Theil--Sen slopes.
- `phase_changes.json` — breakpoint stability across penalties.
- `gradient_phase_summary.csv` and `latest_5k_gradient_summary.csv`.
- `robust_1k_metric_bins.csv` and `robust_1k_gradient_bins.csv`.
- `snapshot_metadata.json` — exact source stats and validity checks.
- `scripts/analyze_parallel_categorical_current_state.py` — reproducible
  read-only analysis.
