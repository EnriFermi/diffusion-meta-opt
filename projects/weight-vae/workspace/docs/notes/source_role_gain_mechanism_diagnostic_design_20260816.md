# Source role-gain mechanism diagnostic (post-hoc, source-only)

Date: 2026-08-16 UTC

## Status and scientific scope

This is an explicitly **post-hoc mechanism diagnostic** for the completed
source latent-code factorial.  It cannot change the locked raw-factorial
verdict (`6/8`, inconclusive for the frozen 5% screen), cannot retroactively
pass G1, and cannot by itself authorize access to the sealed data2vec
text/audio target panel.

The diagnostic separates three explanations for the raw fixed-context result:

1. **Radial/scale miscalibration:** the correct latent code has useful operator
   direction, but the decoder prediction has a transferable role-dependent
   norm bias.
2. **Decoder-prior dominance:** the decoded `z=0` prior is as informative as the
   correct code, so rescaling cannot create a code-specific advantage.
3. **Role-quality failure:** some roles lack useful angular information; a
   global aggregate is rescued only by energy-heavy roles.

## Frozen inputs

No model forward pass and no target-domain load is allowed.

- Factorial sufficient statistics:
  `/home/coder/project/artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816/factorial_matrix_sufficient_stats.csv`
  SHA-256 `3ea4e95e5570547d17e6ab718e1a1f9b1e97a35e99266aa36571440604e5143a`.
- Source-fit gains:
  `/home/coder/project/artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/source_fit_role_gains.csv`
  SHA-256 `e05f2339ba33eb65feee00df2f095655c9319ecf118cd81d7159f6e14f70c4f3`.
- Gain sampling manifest SHA-256
  `2512a92f848d27ab589fb69ebc5989ed0cd15b46545d6f763dc3aac068648791`.
- Heldout-panel manifest SHA-256
  `10020aeabfcce6fe2df3003a7a626f21735cd0763c664e178f27ebcbf42ff985`.
- Parent confirmatory matrix metrics are used only for an exact parity audit:
  `/home/coder/project/artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/matrix_metrics.csv`,
  SHA-256 `d9d42484c55bd74a51da773edd3596306ad7f4b7d7f6055da90e1f284e7d19a1`.

The method is fixed to `ae_cell_mean_c0`, with these six positive,
zero-intercept source-bank gains:

| Role | Gain |
|---|---:|
| `attn_query` | 0.5270182885626962 |
| `attn_key` | 0.5562907139098529 |
| `attn_value` | 0.3990051254514762 |
| `attn_output` | 0.3564476277035192 |
| `ffn_up` | 1.0280206811266708 |
| `ffn_down` | 0.3048084709039325 |

The gain panel contains 72 source-bank matrices (12 per role).  Its models,
datasets, and source keys must be checked as disjoint from the 72-matrix
ViT-B/Flickr heldout panel before analysis.  Every code arm receives the same
gain associated with its decoded matrix role; gains are never refit by code
condition, tiling, context, or heldout result.

## Computation

For stored sufficient statistics `(T, P, D) = (||Y||^2, ||Y_hat||^2,
<Y,Y_hat>)`, evaluate

`E(g) = (T - 2 g D + g^2 P) / T`.

Also report `r=sqrt(P/T)`, `c=D/sqrt(TP)`, and the exact decomposition

`E(g) = (g r - c)^2 + (1 - c^2)`.

The first term is the radial penalty and the second is the angular floor.  The
nonnegative heldout oracle `g*=max(0,D/P)` is descriptive only and must be
labelled as heldout-fitted.

Calibrations/controls:

- raw `g=1`;
- source role-matched gains above;
- a source-derived generic shrink equal to the median of the six frozen gains
  (`0.46301170700708616`) for every role; the arithmetic mean
  (`0.5285984846096913`) is reported as an additional sensitivity;
- all 720 assignments of the six frozen gains to the six roles (an exact
  role-label permutation control);
- an intentionally optimistic heldout-fitted common scalar, separately for
  macro and micro objectives;
- an intentionally optimistic heldout-fitted per-role scalar for each control
  arm, used only to test whether a calibration advantage for `correct` can
  explain the result;
- the literal-zero output baseline `E_X=1`, kept distinct from decoder
  `zero_code = decode(z=0,C)`.

Hard validity checks:

- all input hashes match;
- the target-access seal remains false;
- all 4,608 factorial rows are unique, finite, and complete;
- recomputing with `g=1` matches stored raw `E_X/E_W` to relative tolerance
  `1e-12`;
- fixed `cell/cell/correct` calibrated rows match the parent
  `ae_cell_mean_c0` rows to relative tolerance `1e-12`;
- gain/heldout panels are disjoint as specified;
- both tilings are analyzed separately and never pooled.

## Frozen analysis and interpretation rule

Primary scope: `encoder_condition=cell`, `decoder_condition=cell`.
`native/native` is a separately labelled robustness analysis.  Other context
crosses remain descriptive.

Use 10,000 paired bootstrap draws over the 12 transformer blocks, carrying all
six roles together in each draw, with the same paired-code bootstrap seed as
the completed factorial.  For each tiling separately, compare calibrated
`correct` against `permuted_within_row` and decoder `zero_code` for macro and
micro operator error.

The post-hoc scale-mechanism screen passes only if all eight primary aggregate
comparisons satisfy both:

- comparator/correct point ratio `>= 1.05`; and
- paired-bootstrap ratio lower 5% bound `> 1`.

Additional evidence is required, not silently folded into that 8/8 count:

- correct-minus-permuted and correct-minus-`zero_code` operator-cosine lower
  5% bounds remain `>0` for macro and micro on both tilings;
- per-role calibrated correct-vs-zero points are all reported, including any
  role below the 5% effect size;
- the paired difference-in-differences
  `[E_g(correct)-E_g(control)]-[E_1(correct)-E_1(control)]` is reported; for
  the previously failing `zero_code` contrast it must have upper 95% bound
  `<0` for macro and micro on both tilings to count as differential rescue;
- source gain must reduce the correct-code radial penalty in each previously
  raw-failing role (`attn_value`, `attn_output`, `ffn_down`) on both tilings;
- source role-matched correct must beat the frozen median common shrink for
  macro and micro on both tilings, and identity must rank in the top 5% among
  all 720 role-gain assignments for both aggregates and tilings;
- source-common, heldout-oracle-common, and heldout-oracle-control results are
  reported even when unfavorable;
- sensitivity curves `E(g)` around every fixed role gain are stored and
  plotted; uncertainty intervals condition on this one frozen gain panel and
  do not include gain-estimation uncertainty;
- raw and weight-space results remain visible.

Interpretation:

- A scale-rescue mechanism conclusion requires the 8/8 screen **and** every
  additional directional, difference-in-differences, radial, generic-shrink,
  and role-mapping condition above.  Passing only the calibrated 8/8 screen is
  insufficient.  A full pass supports the narrow claim that the latent
  contains real operator signal and that radial miscalibration materially
  caused the raw zero-code failure.
- It does not establish a standalone weight codec, universal role quality, or
  cross-domain transfer.
- Regardless of outcome, data2vec text/audio remains sealed after this run.
  A target test requires a separately frozen prospective protocol and new
  independent source replication evidence (or an explicit decision to treat
  target results as exploratory rather than confirmatory).
