# Weight-AE runs: facts-only retrospective

Date: 2026-08-29

## Scope and interpretation contract

This report reconstructs the relevant Weight-AE runs directly from persisted
configs, JSONL metrics, gradient ledgers, intervention reports, and the dated
decision record in `discussion.md`. It does not reuse earlier causal verdicts.

The labels below are operational:

- **learns**: the training objective moves materially below its non-informative
  reference and retains sample/operator variation in the latent or output where
  that measurement exists;
- **direction/latent collapse**: direction-like losses remain near their
  random/zero-output reference while rank or sample sensitivity approaches a
  common-template solution;
- **not classified**: the required rank/sensitivity or matched reference was
  not persisted.

Raw total losses are not compared across different objectives. The fixed-64
experiments have 48 train operators (432 train tiles) and 16 diagnostic
held-out operators unless explicitly stated otherwise. The production stream
has 14,000 operator groups and 135,100 canonical tiles. Held-out performance is
not used as a success criterion here.

The latent-rooted training file was still receiving rows during this audit.
Its table entry is a frozen read-only snapshot through step 6,200; its step-1k
checkpoint probes are immutable.

## Run ledger: bounded and exact64 experiments

| Run and exact root | Architecture, backward loss, data, horizon | Observed loss trend | Observed latent/rank/sensitivity | Observed gradient path | What changed from the nearest run |
|---|---|---|---|---|---|
| **4-arm action-loss tokenizer panel** `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_v1_20260827T151252Z` | About 34.85M params per arm; 4 encoder + 4 decoder blocks, width 512, p16 input, bottleneck 32x384. Arms: normalized float, GPTQ continuous, GPTQ token, activation-sqrt. Sole backward target is full-operator relative activation-action error. Fixed 48 train / 16 diagnostic operators, B6, 512 steps, LR 1e-4, warmup 32. | Train action NRMSE: normalized `1.90757 -> 0.91624`; GPTQ-continuous `1.90792 -> 0.91056`; GPTQ-token `1.87361 -> 0.96119`; activation-sqrt `1.88923 -> 0.90765`. The zero-output reference is 1. Final diagnostic values remain around 1 (`0.9904..1.0498`). Operational label: **near-zero-output collapse**. | Train participation rank: normalized `12.43 -> 1.95`; GPTQ-continuous `12.39 -> 1.74`; token `8.31 -> 1.13`; activation-sqrt `7.24 -> 2.60`. On the held-out probe, matched vs shuffled/zero action NRMSE differs only by hundredths. | Every encoder/decoder/bottleneck group has a finite nonzero gradient. For the normalized arm, output-head share of raw squared gradient energy is `95.20%` at step 1 and `98.93%` at step 512. | First relevant panel. All common arms share start SHA `dd56a752...`. |
| **5-arm structural tokenizer panel** `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_10k_v2_20260827T154000Z` | Same 4+4, width-512, 32x384 architecture and same common-arm start SHA as the 4-arm panel. Adds raw-float arm. Backward target is structural `direction + 0.1 * log-scale`; fixed-64 split, B6, planned 10k, stopped after the step-4k evaluation. | Train structural loss at step 4k: normalized `1.01218 -> 0.01308`; GPTQ-continuous `1.01218 -> 0.01240`; GPTQ-token `1.01194 -> 0.11245`; activation-sqrt `1.01195 -> 0.02770`; raw `1.01212 -> 0.55739`. For the normalized arm it is already `0.26615` at step 500. Continuous normalized and GPTQ arms operationally **learn**. | Rank at 4k: normalized `20.78`, GPTQ-continuous `21.17`, token `5.86`, activation-sqrt `7.69`, raw `2.40`; normalized starts at `12.43` and reaches `17.72` by step 500. | All groups remain live. In the normalized arm, output-head raw-energy share changes `94.76% -> 15.49%`; at 4k the largest shares are input/query `62.75%`, head `15.49%`, decoder block 1 `12.49%`. | For the four common arms: backward objective and training horizon changed; raw arm was added. Architecture, data split, common start hash, LR and warmup stayed matched. The matched-step normalized contrast is action loss/rank `0.916/1.95` at step 512 versus structural loss/rank `0.266/17.72` at step 500 (different scalar objectives, identical start). |
| **Raw vs normalized with production Distribution Encoder** `/mnt/shared/weightclip_benchmark/raw_vs_normalized_prod_dist_3k_v1_20260827T1648Z` | About 41.97M params; same 4+4 downstream core and 32x384 bottleneck, plus production Distribution Encoder context added to encoder keys and decoder queries. Raw and normalized arms. Structural `direction + 0.1 * log-scale`, fixed-64 split, B6, 3k. | Raw `1.00970 -> 0.32712`; normalized `1.00970 -> 0.02227`. Normalized operationally **learns**. | Rank raw `7.51 -> 5.54`; normalized `12.63 -> 50.17`. Shuffling activation context changes normalized structural loss `0.02227 -> 0.79848`; normalized latent relative RMS changes `0.459`. | All groups live. Normalized output-head raw-energy share `98.45% -> 20.83%`; Distribution Encoder has `64.22%` at step 3k. | Relative to the 5-arm panel: adds production activation conditioning and its parameters, reduces to raw/normalized arms, and runs 3k from a new shared initialized state. This is not a one-variable conditioning ablation, but conditioning clearly did not prevent normalized structural fitting. |
| **Tokenizer-B structural run** `/mnt/shared/weightclip_benchmark/conditioned_tokenizer_b_small_3k_v1_20260827T1950Z` | About 41.73M params. Same Distribution-conditioned 4+4 core and bottleneck. Token record is `[Wnorm, Wnorm*tanh(g(X)), standardized scale, zero flag, padding]`, then one bias-free `64 -> 512` projection. Same structural loss, fixed-64 split, B6, 3k. | Structural `1.00969 -> 0.02347`, direction `1.00491 -> 0.02069`. Operationally **learns**, matching the prior normalized baseline (`0.02227`) within about `0.0012`. | Rank `12.34 -> 43.24`. Final matched loss `0.02344`; shuffled latent `0.97760`; zero latent `0.98266`. Latent shuffle changes output by relative RMS `1.509`. Tokenizer-only activation shuffle barely changes loss `0.02344 -> 0.02351`. | Every common and tokenizer-specific group is live. Output-head raw-energy share `98.45% -> 29.64%`; Distribution Encoder `39.67%`, tokenizer projection `6.80%` at step 3k. | Relative to the preceding normalized Distribution run, common tensors are matched to the archived baseline; the tokenizer record/projection is the intended change. This tokenizer did not cause a learn-to-collapse transition. |
| **Original heavy V4, normalized exact64** `/mnt/shared/weightclip_benchmark/ae_v4_normalized_exact64_v3_5000step` | Total 725.33M / trainable 706.31M; conditioned-MLP patch tokenizer, 10 encoder blocks with latent feedback, 8 decoder blocks, 32x384 bottleneck, Distribution Encoder. Structural `direction + 0.1 * scale`; fixed 64-operator overfit schedule, B6 with grad accumulation, LR 5e-5, no warmup. Persisted to train step 4,330; eval through 4,096. | Exact64 matched mean direction: `0.99609` at step 0, `0.96244` at 256, `0.96146` at 1,024, then `0.75055` at 3,072 and `0.29747` at 4,096. Last persisted train structural loss in `discussion.md` is `0.218870` at 4,330. This run shows delayed but material fitting; it is **not a clean collapse label**. | No participation-rank series is stored. Z-shuffle output relative effect is `0.05460` at initialization, `0.00217` at 256, and `0.01490` at 4,096; median z-shuffle direction delta stays approximately zero. | No group-gradient JSONL/CSV is present in this root, so a complete gradient-path comparison is missing. | Relative to Tokenizer-B, nearly every architecture component and optimizer setting changes. It establishes only that a heavy tokenizer/feedback V4 can eventually lower fixed64 structural loss; it is not an isolated tokenizer comparison. |
| **Scaled Direct Normalized 700M exact64** `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_3k_v1_20260827T2100Z` | 706.284M; direct normalized p32 input (512 tokens), 13 encoder + 6 decoder blocks, width 1536, 24 heads, same 32x384 bottleneck, production Distribution Encoder. Structural `direction + 0.1 * log-scale`; 48 train / 16 diagnostic operators, 54 tiles/update, LR 1e-4, warmup 32. Planned 3k, stopped after step 1k. | Train structural trajectory: `1.00642` (0), `0.94645` (250), `0.93017` (500), `0.53331` (750), `0.07036` (1k); direction is `0.06798` at 1k. Operationally **learns/memorizes**. | Rank `85.41 -> 13.90 -> 16.99 -> 18.13 -> 18.43`; latent RMS `0.783 -> 1.781`. No checkpoint-based shuffle intervention was persisted. | All 13 encoder and 6 decoder blocks are live. Output-head raw-energy share is `97.48%` at step 1 and `97.22%` at step 1k. At step 1k encoder block RMS spans `3.29e-7..1.90e-5`. | Relative to the small Direct/Distribution family: width/depth and parameter count scale up, input granularity changes p16 to p32, and decoder depth changes 4 to 6. Bottleneck, normalized representation, structural objective and fixed64 data contract remain. Model scale and the `32 -> 1536` projection did not by themselves produce collapse. |
| **700M exact64 activation-relative-MSE** `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_activation_metric_3k_v2_20260828T130359Z` | Same 706.284M 13+6 Direct core, normalized p32 and fixed64 split. Sole backward target is per-tile relative activation MSE. Adds cosine Q/K with fixed scale 2 and uses warmup 320; B6; complete 3k. Legacy structural/behavioral quantities are detached diagnostics. | Full-train activation loss `1.44692 -> 1.10415` (250) `-> 1.05802` (500), fluctuates, and ends `0.96820` at 3k. Zero-output reference is 1. Predicted/target RMS ratio median is `0.127` at 3k. Structural direction diagnostic is `0.96337`. Operationally **near-zero-output collapse**. | Rank `74.60 -> 14.18` (250) `-> 2.56` (500) `-> 1.33` (1,250) `-> 1.77` (3k); latent RMS `0.783 -> 2.183`. | All groups remain finite/nonzero in telemetry, but all 301 logged pre-clip norms exceed 1. Output-head raw-energy share `97.59% -> 99.9993%`; late encoder RMS spans `5.14e-8..3.26e-6`. | Nearest clean comparator is the 700M exact64 structural run: same core, data split, batch and nominal peak LR, but backward objective, Q/K parameterization and warmup schedule all change. Therefore this is not a loss-only ablation. It does show that fixed64 data alone is not sufficient for learning. |

Primary evidence for this table:

- 4-arm: `config.json`, `model_contract.json`, `metrics.jsonl`,
  `gradient_telemetry.jsonl`, `bottleneck_usage.json` in its run root.
- 5-arm: `config.json`, `model_contract.json`, `metrics.jsonl`,
  `gradient_telemetry.jsonl` in its run root.
- Distribution and Tokenizer-B: their `config.json`, `metrics.jsonl`,
  `gradient_telemetry.jsonl`, plus `analysis/conditioning_usage.json`,
  `analysis/tokenizer_b_usage.json`, and `final_interventions.json`.
- V4: `resolved_run_config.json`, `operator_set_metrics.jsonl`, and
  `analysis/current_train_loss_curve.png`.
- 700M exact64 runs: their `config.json`, `metrics.jsonl`, and
  `gradient_telemetry.jsonl`; the activation run also has `summary.json` and
  `train_metrics.jsonl`.

## Run ledger: production-stream experiments

| Run and exact root | Architecture, backward loss, data, horizon | Observed loss trend | Observed latent/rank/sensitivity | Observed gradient path | What changed from the nearest run |
|---|---|---|---|---|---|
| **Production behavioral + operator** `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_production_500k_v1` | 706.301M Direct normalized p32, 13+6, 32x384. Full production bank, B32, LR 5e-5, no warmup, clip 5. Loss is `50*behavioral_operator + behavioral_direction + 10*behavioral_scale + structural_direction + 10*structural_scale`. Metrics through step 5,460. | Total first/last `5.3711 -> 4.6781`; mean steps 1-1k `4.7939`, steps 5,001-5,460 `4.0045`. The decreases are mainly scale/operator terms. Behavioral direction means `0.9432 -> 0.9182`; structural direction `0.9597 -> 0.9574`, still near the non-informative regime. Operational label: **direction non-learning**. | No checkpoint rank or latent intervention was persisted for this short run. | All logged pre-clip norms exceed 5. Output-head raw squared-gradient share `97.30%` at step 1 and `99.9956%` at step 5,400; all groups remain numerically nonzero. | Relative to 700M exact64 structural: full heterogeneous bank/layouts, B6 to B32, LR halved, warmup removed, scale weight `0.1 -> 10`, behavioral direction/scale and operator terms added, and production mask/position metadata added. This transition is heavily confounded. |
| **Production no-operator** `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1` | Same 706.301M architecture/data/optimizer as above. Only behavioral-operator coefficient is zero; other direction/scale coefficients remain `1/10`. Stopped at optimizer step 66,889; metric rows through 66,880. | Total mean `2.7681` (steps 1-1k), `2.1747` (5,001-10k), `2.1446` (40,001-66,880). Behavioral direction means `0.9294`, `0.9043`, `0.9033`; structural direction `0.9560`, `0.9539`, `0.9520`. Scale terms fall strongly. Operational label: **direction/latent collapse**. | Step-60k centered rank `2.82`, raw pairwise latent cosine `0.95484`. Weight shuffle gives final latent cosine `0.99450`. On the fixed probe, matched vs shuffled-latent structural direction is `0.99113` vs `0.99136`; matched action NRMSE `1.696` vs shuffled `1.707`. | Every train row has pre-clip norm above 5. Output-head raw-energy share `97.29% -> 99.61%`. At step 60k, encoder-layer-1 latent-query entropy fraction is `0.0578` and latent-state RMS `172.27`; these are observations, not a causal conclusion. | The only intended loss change from the preceding production run is `behavioral_operator: 50 -> 0`; the step-1 structural and direction/scale components are exactly the same. Removing operator loss does not restore direction learning. |
| **Bounded + gradient-balanced production** `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2` | Same 706.301M/full bank/B32/LR/clip. Adds cosine Q/K, bounded residual/SwiGLU writes, changes raw scale coefficients `10 -> 1`, and uses per-batch bottleneck gradient balancing. Stopped at step 1,898; metric rows through 1,890. | Because objective scaling changed, total is not comparable to no-operator. Mean total `1.9693` (1-1k) and `1.9424` (1,001-1,890). Behavioral direction means `0.9212 -> 0.9135`; structural direction `0.9436 -> 0.9383`. Operational label: **direction/latent collapse**. | Exact probe rank `33.55 -> 3.02` at step 1k; top-1 centered energy `0.110 -> 0.541`; weight-shuffle latent relative delta `0.988 -> 0.678`. Matched/shuffled/zero-latent action NRMSE is `4.43 / 4.68 / 3.73`. | Every logged norm exceeds clip 5. Output-head raw-energy share `97.65% -> 99.94%`. Logged production-batch bottleneck direction/scale cosine is `-0.467` at step 1 and near zero late; the fixed exact64 component probe has different geometry, so no global conflict claim follows. Attention entropies remain high at step 1k. | Multiple simultaneous changes from no-operator: attention, residual/MLP parameterization, scale coefficients, and backward balancing. This run does not isolate which, if any, affects collapse. |
| **Polar p16 tails, production** `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1` | 742.121M. Same p32 input, 13-layer encoder and 32x384 bottleneck; five shared decoder blocks followed by separate one-block p16 direction/scale tails and minimal heads. Coordinate-owned stop-gradient keeps private tails isolated. Same no-operator production loss with `1/10` direction/scale weights, full bank, B32, LR 5e-5, clip 5. Stopped at 18,511. | Mean total `3.0965` (1-1k), `3.0816` (1,001-5k), `3.0917` (5,001-10k), `3.0933` (10,001-18,510). Behavioral direction stays about `0.918..0.922`; structural direction about `0.934..0.938`. Scale improves early. Operational label: **direction/latent collapse**. | Step-7k latent effective rank `1.325`, cross-sample cosine `0.999679`; predicted-direction cross-sample cosine `0.999987` while target cosine is near zero. Latent roll changes direction by `1.76e-5`; zero latent changes direction by `2.86e-4` but changes scale by `0.743`. | Private opposite-coordinate gradients are exactly zero by contract. All groups are finite/nonzero, but late raw energy is `98.68%` in the scale head. Only `1.13%` of training rows have pre-clip norm above 5, so continuous clipping is absent. | Relative to no-operator: production data/loss/optimizer remain; the sixth shared decoder block is replaced by two private tails, polar heads/recombination are explicit, and cross-coordinate private backward is stopped. It is an architecture plus gradient-routing intervention, not topology alone. |
| **Latent-rooted polar tails, production** `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_latent_rooted_polar_tails_500k_v1` | Same 742.121M parameters, data, loss, batch, LR and seeded state family as polar. Forward-graph-only change: the first shared and private-tail reads use query-only addresses with latent/shared K/V and no additive query residual. Snapshot through step 6,200. | Mean total `3.3006` (1-1k), `3.3175` (1,001-5k), `3.3298` (5,001-6,200). Behavioral direction means `0.9229`, `0.9222`, `0.9263`; structural direction `0.9349`, `0.9350`, `0.9370`. Operational label: **direction/latent collapse**. | Immutable step-1k probe: rank `96.14 -> 1.17`, cross-sample latent cosine `0.205 -> 0.99938`; direction prediction cosine `-0.038 -> 0.999999`; latent-roll output relative effect `1.393 -> 0.00134`. Native B4 probe gives latent-gradient RMS `1.263e-5 -> 9.41e-12`; zero latent still changes direction by exactly 1. | Step-1 gradients are live across all groups. By step 6,200, scale head owns `99.85%` of raw squared energy; encoder-block RMS is approximately `1e-9..1e-8`, although values remain nonzero. Only an initial minority of steps are clipped (about `1.7%` through the earlier 5.9k snapshot). | Relative to polar, parameter count, loss/data/optimizer and initialization contract are matched; only decoder value construction changes. It produces strong step-0 latent dependence but does not preserve it through step 1k. |

Primary evidence for this table:

- Every production root: `resolved_config.json`, `train_metrics.jsonl`, and
  `gradient_telemetry.jsonl`.
- No-operator probes:
  `analysis_checkpoint_060000_latent_use_v2.json` and
  `analysis_attention_path_060000_v5.json`.
- Bounded/balanced probes: `analysis_latent_use_initial.json`,
  `analysis_latent_use_step_001000.json`,
  `analysis_attention_path_initial.json`, and
  `analysis_attention_path_step_001000.json`.
- Polar probe:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1/mechanistic_step_0007000_v1/report.json`.
- Latent-rooted probes:
  `/home/coder/project/artifacts/weightclip_benchmark/latent_rooted_hostile_review_step_0001000_v1/report.json`
  and
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_latent_rooted_polar_tails_500k_v1/latent_rooted_state_step_0001000_v1/report.json`.

## Observed facts that discriminate proposed explanations

1. **The cleanest learn/collapse edge is the backward objective on the small
   normalized model.** The 4-arm action run and 5-arm structural run use the
   same common-arm start SHA, architecture, fixed train set, batch, LR, and
   warmup. Near the same horizon, action training has rank `1.95` and action
   NRMSE `0.916` at step 512 (near its zero-output reference 1), while
   structural training has rank `17.72` and structural loss `0.266` at step
   500, then `0.0131` at 4k. This is the smallest archived intervention that
   coincides with a clear change in learning dynamics.

2. **Tokenizer complexity is not sufficient for collapse in these data.**
   Production Distribution conditioning reaches normalized structural loss
   `0.0223`; Tokenizer-B reaches `0.0235`; even the heavy V4 eventually lowers
   exact64 direction loss to `0.297` by step 4,096.

3. **700M scale, p32 tokenization, the `32 -> 1536` projection, and raw
   output-head gradient dominance are not individually sufficient for
   collapse.** The 700M exact64 structural run reaches `0.0704` at step 1k even
   though its output head already owns `97.22%` of raw squared gradient energy.
   Therefore a high raw head-energy share, without a matched threshold or
   optimizer-update measurement, does not distinguish the successful exact64
   run from production failures.

4. **Operator behavioral loss is not necessary for the production failure.**
   Setting only its coefficient from 50 to zero preserves the near-random
   direction plateau through 66.9k and yields low-rank, weakly input-sensitive
   latents.

5. **Full-dataset diversity is not necessary for all collapse modes.** The
   exact64 activation-relative-MSE run also ends near its zero-output basin with
   rank `1.77`. This does not prove that data diversity is irrelevant to the
   production direction/scale loss, because that exact64 run changes the
   objective, attention, and schedule.

6. **Continuous clipping and one-hot attention are not necessary correlates.**
   Polar and latent-rooted runs collapse while fewer than about 2% of their
   logged steps exceed clip 5. Bounded/cosine-attention and the high-entropy
   latent-rooted checkpoint also collapse. Conversely, old no-operator
   production has continuous clipping and low layer-1 attention entropy. These
   mechanisms separate across runs rather than forming one invariant failure
   signature.

7. **Low operator/sample rank and lost permutation sensitivity are the most
   consistent measured state correlates.** Explicitly probed failed runs end at
   rank roughly `1.2..3.0`: action panel `1.1..2.6`, no-operator `2.82`,
   bounded/balanced `3.02`, activation `1.77`, polar `1.32`, latent-rooted
   `1.17`. Learned normalized structural runs retain ranks `18..50`. This is a
   correlation/localization, not a demonstrated cause. V4 has no stored rank
   series and is an important gap.

## Narrow inference: what is the minimum change correlated with learns -> collapses?

### Supported

Across the archive, the minimum experimentally isolated change is **the
backward objective class**:

- same small normalized architecture;
- same fixed data and optimizer settings;
- exact same initialized common weights;
- activation-action relative loss gives a near-zero-output, rank-about-2
  solution;
- structural direction plus `0.1 * log-scale` gives high-rank memorization.

This does not establish the microscopic mechanism inside either objective. It
does establish that neither raw weight tokenization nor encoder depth is needed
to create the observed collapse transition.

### Not established for the production boundary

There is no matched experiment that changes only exact64 to full production
data while keeping the same architecture, objective, scale coefficients,
batch, LR, warmup, masks/layouts, and start. The first 700M production launch
changed all of those together. Later runs remove or alter one part of the
failed production package, but none reconstructs a single-factor
learns-to-collapses comparison:

- removing operator loss alone does not rescue;
- balancing and bounded attention change several factors and do not rescue;
- polar tails change topology and backward routing and do not rescue;
- latent-rooting changes only decoder value construction relative to polar and
  does not rescue;
- activation loss collapses on exact64, but it is not the production mixed
  direction/scale objective.

Therefore the smallest production-specific set is **not identified**. The
narrowest factual description is: the known learned 700M point uses fixed64
data plus structural `direction + 0.1*scale`; every measured full-production
point uses a different backward objective/schedule package and exhibits
direction/latent collapse. Dataset diversity, mixed behavioral/structural
losses, the `0.1 -> 10` scale weighting change, and optimizer/batch changes
remain confounded.

## Missing evidence

- A same-start, same-architecture, same-optimizer comparison of fixed64 versus
  full production data under one identical objective.
- A same-start 700M fixed64 comparison of structural loss versus the exact
  production direction/scale mixture, with only the loss changed.
- AdamW update-energy and parameter-displacement ledgers; current percentages
  are raw gradient energy before clipping and optimizer preconditioning.
- V4 latent rank and direct latent-permutation checkpoints; only output
  sensitivity was persisted.
- A final checkpoint and intervention probe for the successful 700M exact64
  structural run; the run stopped at step 1k and saved final-only checkpoints.
- Replicated seeds. Nearly all decisive comparisons are seed 42, so seed
  robustness is not established.

## Excluded from the scientific ledger

`/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_structural_only_500k_v1`
was a short misinterpreted launch that was stopped after the user clarified the
loss contract. It is not used as evidence for or against production learning.

