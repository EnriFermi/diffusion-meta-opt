# WeightCLIP ResNet protocol audit and deviation ledger

Date: 2026-08-18.

Purpose: prevent silent substitutions between the WeightCLIP paper, its
released code, and our controlled comparison. This is a planning artifact, not
an experimental result.

Primary sources:

- paper: `https://arxiv.org/html/2607.03551`, arXiv v1;
- official repository: `https://github.com/HSG-AIML/weightCLIP`, inspected at
  commit `be080677a6eceacdbe3b2823caffe3c0cc73fa7e`;
- relevant released files: `scripts/train_resnet18slim_zoos.sh`,
  `scripts/train_resnet18slim_datasetpt_zoo.py`,
  `config/contrastive_multi_zoo_resnet_alignment.yaml`,
  `config/data/train_resnet.yaml`,
  `config/data/checkpoints_dataset/*_resnet.yaml`,
  `scripts/dataset_to_model.py`, and `scripts/ood_utils.py`.

## Source hierarchy

1. Reproduce an explicit paper statement when the released code does not
   contradict it.
2. Use the released runnable implementation for details omitted by the paper.
3. When paper and code conflict, do not silently choose. Record both, make one
   explicit decision, and preserve a native released-code result separately if
   the conflict affects evaluation.
4. Never label a run on our newly reconstructed checkpoint population an exact
   paper reproduction. The original ResNet zoo is unavailable.

## Audited contract

| Item | Paper | Released code | Frozen decision |
|---|---|---|---|
| Source datasets | Artworks, Blood Cells, Breast Cancer Tissues, Aerial Cactus, Cassava Leaf, CT Images, Land Use, Lego Bricks, Real/Fake Legos, Casting | `train_resnet.yaml` substitutes `asl_resnet` for the paper's separate Lego Bricks dataset and retains `lego_vs_generic_resnet` | Use the ten **paper** datasets. Do not silently replace Lego Bricks with ASL. Build the missing paper dataset entry in our manifest. |
| Dataset materialization | Refers to TANS datasets; says classes are capped at 20, but Table 10 itself reports ResNet datasets with 51 classes | Builder reads the published TANS `tr/va/te` folders, resizes to 32x32 RGB, converts to float and normalizes to `[-1,1]`; it performs no class cap and exposes separate train/validation/test sets | Use the released TANS-folder builder and preserve all classes, matching the table/released behavior rather than the contradictory prose. Hash raw file lists and all three materialized splits. |
| Population | Main Sec. 4.2: 20 independent runs/dataset and five terminal snapshots at epochs 41--45 = 100 checkpoints/dataset, 1,000 total | Launch shell defaults to 50 seeds; ResNet checkpoint configs load only zero-based epochs 43 and 44, also producing 100 checkpoints/dataset. Both variants train every lineage for the full 45 epochs. | Treat the released runnable configuration as authoritative here: 50 lineages/dataset and zero-based epochs 43/44 (one-based 44/45), still 1,000 primary checkpoints but 500 independent trainings. |
| Zoo optimizer | SGD with momentum, LR sweep, weight decay `5e-4`, dropout `0.15`, batch 256, 45 epochs, OneCycleLR, normal init, no gradient clipping | Shell resolves momentum `0.9`, 45 training epochs, scheduler horizon 50, no clipping, normal init, dropout `0.15`. Standalone parser and per-dataset metadata instead default to uniform/`kaiming_uniform` and dropout `0.2`. | Treat the launch shell as authoritative and reproduce normal init/`0.15`; resolved run manifests must prove the override occurred. |
| LR sweep | Says only `Sweep` | Shell uses 2 seeds for each of `{0.05,0.07,0.1,0.2,0.3,0.5,0.7}`. The Python parser's standalone defaults differ (4 seeds and a different five-LR list). The released selector ranks by held-out/test metrics. | The shell is the official launch surface: mandatory 2x7 sweep per source dataset, cached once. Log that LR selection follows released code. No no-sweep production shortcut. |
| Zoo image preprocessing | Not fully specified beyond the released dataset construction | Trainer consumes materialized `dataset.pt`; no extra online image augmentation is applied by the cached tensor path | Use the exact rebuilt `dataset.pt` tensors and hashes; add no unreported augmentation. |
| Checkpoint archive | Primary epochs 41--45 | Released trainer normally begins saving at epoch index 20 and duplicates optimizer/scheduler with each saved epoch | Save initialization and every epoch model state in FP32; keep only one rolling optimizer/scheduler resume state. Only one-based epochs 44--45 enter the primary population. This storage extension does not alter training. |
| WeightCLIP training population | Paper population above | ResNet configs use 50 seeds x 2 terminal checkpoint indices, `max_seeds=100`, `max_checkpoints=500` | Released checkpoint remains frozen. If matched retraining is triggered, train it on exactly the same 50-lineage/two-terminal-checkpoint manifest used by ours. |
| WeightCLIP split | Paper says held-out model instances but gives no ratio | Random checkpoint-level `70/15/15`; its splitter warns that epochs from one lineage may leak | Our new data split is lineage-safe `35/7/8` inside every source dataset. This is an intentional validity deviation, reported explicitly. All methods newly trained on our data receive the same split. |
| Checkpoint symmetry augmentation | Not numerically specified in the paper | `align=True`, five permutations/checkpoint in ResNet configs | Apply five graph-consistent views only after splitting. Do not count them as independent checkpoints or uncertainty units. |
| Weight backbone | Table 12: token 288, latent 192, `d_model=1600`, 12 layers, 20 heads, window/block 512, dropout `0.05` | Config matches those values; joint trainer uses seed 2020, 100 epochs, BF16, clip 1.0, batch 32, two GPUs, AdamW LR `1e-3`, WD `3e-9`, OneCycleLR | Frozen released checkpoint is primary. These exact resolved values are used only if matched retraining is triggered. |
| Dataset encoder | DeepSets, 32x32, set size 10, sum, embedding 192 for ResNet; paper says classification weight anneals for 600 epochs | ResNet config sets `classification_weight_anneal_epochs: 0`; joint run itself is 100 epochs | Use the released frozen encoder as-is. Record the paper/code annealing conflict; do not retrain or silently alter it in the first campaign. |
| Alignment | SigLIP-style, temperature 1, weight 0.25, bias -4 | Config matches; reconstruction contrastive gamma is 0, token mask/noise are 0 | Frozen released checkpoint as-is. |
| Candidate budget | 100 dataset prompts/subsamples, report top 5 | `dataset_to_model.py` defaults to 100 and top 5 | Run all three views: single-sample controlled primary; matched validation-selected 100-to-top-5; and mandatory paper-style test-selected 100-to-top-5 for every generative method. |
| Candidate selector | Paper does not name the split used to select top 5 | Released code evaluates every candidate on `test_loader`, sorts by that test accuracy, then fine-tunes/evaluates the same top 5 on the same test loader | Reproduce this for WeightCLIP **and ours** in a separately labeled `test-selected oracle` table. The controlled scientific table still selects on a target-train-derived validation split and touches test only for final curves. Never merge the tables. |
| Classifier head | Paper notes head mismatch but does not specify exact adaptation | Released decoder reconstructs the full tokenized head and `_adapt_output_layer` copies/slices decoded rows, adding random rows only when expansion is necessary | Native table keeps released behavior. Controlled table freshly default-initializes the entire classifier head for every arm, as required by our comparison contract. |
| BatchNorm | Paper says BN conditioning | Released decode retains generated BN affine values and recalibrates running state for up to 200 train-loader batches | Native WeightCLIP table uses exactly this. In the controlled table our method excludes every BN key and uses default affine/state followed by the same 200-batch calibration schedule; WeightCLIP remains native. No BN-reset ablation in the first campaign. The system difference is reported. |
| Fine-tuning | Report epoch 0, 1 and 10 | ResNet path actually uses batch 10, SGD momentum 0.9, LR `1.5e-4`, WD 0, no scheduler, full target train loader; evaluates test after every epoch. Its CLI help incorrectly says the default is `1e-4`. | Native and controlled tables use the executed `1.5e-4` value unless a paper artifact proves otherwise. Log the resolved value and every step/epoch so early AULC can also be computed. |
| Reported uncertainty | Table 3 says mean/std over seeds but exact seeds/unit are omitted | Direct released evaluator defaults to seed 0; another released ResNet evaluator defaults to `{42,0,777}` | Paper numbers remain `reported by authors`; no significance test treats their std as ours. Our controlled table uses fixed paired seeds and exposes raw candidate/seed/dataset values. |

## Two tables, two questions

### Native fidelity table

This table asks whether the released checkpoint transfers to our reconstructed
zoo and reproduces the paper-scale numbers. It keeps released head adaptation,
generated BN affine values, 200-batch BN conditioning, 100 candidates, top-5
selection by target test accuracy, and released fine-tuning defaults. It is
explicitly labeled as test-selected/oracle and is never the sole evidence for
our method's superiority.

### Controlled architecture table

This table asks whether our system gives better initializations under matched
access. It uses the same target images/dataset encoder, random full classifier
head, paired downstream seeds, same optimizer/batches, same candidate budget,
and validation-only candidate selection. Ours and WeightCLIP each get their own
matched flow in their own latent space. The primary estimator is one random
sample; matched best-of-100/top-5 is secondary.

## Source-validation decision

Do not hold out one fixed pair of source datasets for all flow decisions. The
large between-dataset variation makes such a two-dataset score high variance.
Use `35/7/8` lineages inside **each** of all ten source datasets. Select AE/flow
settings on the macro average over all 70 validation lineages, while also
reporting the worst source dataset; open the 80 internal-test lineages only
after freezing choices. The six paper OOD datasets remain the only
dataset-generalization test.

## Known irreproducibilities that must remain visible

- Original ResNet checkpoint files and random seeds are absent.
- Paper and repo disagree on the ResNet source-dataset list.
- Paper's 20-class cap conflicts with its own class-count table and released
  builder, which do not cap classes.
- Paper main text's 20 lineages x 5 terminal snapshots and released config's 50
  lineages x 2 terminal snapshots both yield 100 checkpoints/dataset but
  different statistical populations. This is not 5-epoch versus 2-epoch
  training: both variants train a lineage for 45 epochs.
- Paper omits the LR grid/selection rule, split implementation, candidate
  selection split, exact evaluation seeds, and several optimizer details.
- Paper says dataset-classification annealing lasts 600 epochs; released ResNet
  config sets it to zero.
- The released candidate selector uses target test labels.
- Several Python parser/config defaults differ from the official launch shell;
  a launcher must log resolved normal init, dropout `0.15`, two sweep seeds,
  seven LR candidates, 45 epochs, scheduler horizon 50, and no gradient clip.

## Frozen population decision and pre-production probe (2026-08-18)

The campaign follows the released configuration: 50 independent lineages per
dataset, each trained for 45 epochs, with only zero-based checkpoints 43/44
(one-based epochs 44/45) entering the primary WeightCLIP/AE population. The
paper's alternative 20-lineage/five-terminal-snapshot population is not mixed
into that population.

Before the full zoo launch, run the released two-seed/seven-LR sweep and two
full 45-epoch final lineages on one real official TANS source dataset. Save the
five late checkpoints at zero-based indices 40--44 for this diagnostic only.
Review train/validation/test loss and accuracy, the OneCycle LR trace, and
relative/cosine weight displacement across the late checkpoints. This probe
tests whether the two selected terminal snapshots sit in the same late regime;
it does not decide whether training lasted two epochs and cannot establish
population-level diversity from two lineages.

Evaluation has two deliberately separate estimands. The controlled scientific
table uses equal candidate budgets and no test-label selection. A mandatory
native-fidelity table reproduces the released WeightCLIP selector exactly:
generate 100 candidates and report the five selected by target test accuracy.
The latter is explicitly labeled `test-selected oracle`, is run for both
WeightCLIP and our sampler, and is never presented as an unbiased estimate.

These are limitations of exact paper reproduction, not permission to improvise
silently. Every run writes a resolved protocol diff against this ledger.
