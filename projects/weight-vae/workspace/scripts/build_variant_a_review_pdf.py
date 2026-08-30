from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
REPORT_DIR = BASE / "a_case_causal_report"
OUT_DIR = BASE / "a_case_review_packet"
OUT_PDF = OUT_DIR / "variant_A_complete_review_packet_2026-07-06.pdf"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _wrap(text: str, width: int = 100) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width, replace_whitespace=False))
    return lines


def add_text_page(pdf: PdfPages, title: str, body: list[str], *, subtitle: str | None = None) -> None:
    fig = plt.figure(figsize=(8.5, 11.0), dpi=150)
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.90])
    ax.axis("off")
    y = 0.985
    ax.text(0.0, y, title, ha="left", va="top", fontsize=18, fontweight="bold")
    y -= 0.045
    if subtitle:
        ax.text(0.0, y, subtitle, ha="left", va="top", fontsize=10.5, color="#444444")
        y -= 0.040
    for block in body:
        if y < 0.055:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11.0), dpi=150)
            ax = fig.add_axes([0.07, 0.05, 0.86, 0.90])
            ax.axis("off")
            y = 0.985
            ax.text(0.0, y, title + " (continued)", ha="left", va="top", fontsize=18, fontweight="bold")
            y -= 0.060
        if block.startswith("## "):
            y -= 0.010
            ax.text(0.0, y, block[3:], ha="left", va="top", fontsize=13, fontweight="bold")
            y -= 0.032
            continue
        if block.startswith("- "):
            wrapped = _wrap(block[2:], 98)
            ax.text(0.012, y, u"\u2022", ha="left", va="top", fontsize=10)
            ax.text(0.038, y, wrapped[0] if wrapped else "", ha="left", va="top", fontsize=9.4)
            y -= 0.023
            for line in wrapped[1:]:
                ax.text(0.038, y, line, ha="left", va="top", fontsize=9.4)
                y -= 0.023
            continue
        wrapped = _wrap(block, 104)
        if not wrapped:
            y -= 0.018
            continue
        for line in wrapped:
            if line == "":
                y -= 0.018
            else:
                ax.text(0.0, y, line, ha="left", va="top", fontsize=9.6)
                y -= 0.023
        y -= 0.006
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _fmt_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        if not math.isfinite(value_f):
            return "NA"
        if abs(value_f) >= 1000 or (abs(value_f) > 0 and abs(value_f) < 1e-3):
            return f"{value_f:.3g}"
        return f"{value_f:.6f}".rstrip("0").rstrip(".")
    return str(value)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, *, note: str | None = None, max_rows: int = 24) -> None:
    if df.empty:
        add_text_page(pdf, title, ["No rows available.", note or ""])
        return
    chunks = [df.iloc[i : i + max_rows].copy() for i in range(0, len(df), max_rows)]
    for chunk_idx, chunk in enumerate(chunks, start=1):
        fig = plt.figure(figsize=(11.0, 8.5), dpi=150)
        ax = fig.add_axes([0.025, 0.05, 0.95, 0.84])
        ax.axis("off")
        page_title = title if len(chunks) == 1 else f"{title} ({chunk_idx}/{len(chunks)})"
        fig.text(0.025, 0.965, page_title, ha="left", va="top", fontsize=16, fontweight="bold")
        if note and chunk_idx == 1:
            fig.text(0.025, 0.925, "\n".join(_wrap(note, 145)), ha="left", va="top", fontsize=8.5, color="#444444")
        table_data = [[_fmt_value(v) for v in row] for row in chunk.to_numpy(dtype=object)]
        table = ax.table(
            cellText=table_data,
            colLabels=[str(c) for c in chunk.columns],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.0 if len(chunk.columns) > 9 else 7.0)
        table.scale(1.0, 1.35)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e9eef5")
                cell.set_text_props(weight="bold")
            cell.set_edgecolor("#b8b8b8")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, *, caption: str | None = None, section: str | None = None) -> bool:
    if not image_path.is_file():
        return False
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False
    fig = plt.figure(figsize=(11.0, 8.5), dpi=150)
    ax = fig.add_axes([0.035, 0.105, 0.93, 0.78])
    ax.imshow(image)
    ax.axis("off")
    header = _rel(image_path)
    if section:
        fig.text(0.035, 0.972, section, ha="left", va="top", fontsize=8.5, color="#666666")
        fig.text(0.035, 0.943, header, ha="left", va="top", fontsize=10.0, fontweight="bold")
    else:
        fig.text(0.035, 0.955, header, ha="left", va="top", fontsize=10.0, fontweight="bold")
    if caption:
        fig.text(0.035, 0.057, "\n".join(_wrap(caption, 145)), ha="left", va="top", fontsize=8.2, color="#222222")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return True


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def downstream_summary_table() -> pd.DataFrame:
    df = read_csv(REPORT_DIR / "a_case_proxy_downstream_intervention_summary.csv")
    if df.empty:
        return df
    keep = [
        "variant",
        "li_A_p95",
        "test_mean_loss_delta_mean",
        "test_mean_loss_delta_median",
        "test_mean_loss_delta_ci_low",
        "test_mean_loss_delta_ci_high",
        "test_mean_loss_delta_worse_count",
        "test_mean_loss_delta_better_count",
        "step0_test_loss_delta_mean",
        "post0_test_loss_mean_delta_mean",
    ]
    return df[keep].rename(
        columns={
            "variant": "variant",
            "li_A_p95": "li_A p95",
            "test_mean_loss_delta_mean": "test_mean_delta",
            "test_mean_loss_delta_median": "test_median_delta",
            "test_mean_loss_delta_ci_low": "ci_low",
            "test_mean_loss_delta_ci_high": "ci_high",
            "test_mean_loss_delta_worse_count": "worse",
            "test_mean_loss_delta_better_count": "better",
            "step0_test_loss_delta_mean": "step0_delta",
            "post0_test_loss_mean_delta_mean": "post0_delta",
        }
    )


def alpha_debug_table() -> pd.DataFrame:
    df = read_csv(BASE / "a_case_run_summary_with_ablation.csv")
    if df.empty:
        return df
    keep = [
        "run_pretty",
        "decoder_aulc_median",
        "decoder_delta_vs_control_mean",
        "decoder_delta_vs_control_max",
        "decoder_worse_than_control_count",
        "diag_li_A_full_p95",
        "quality_decoded_acc_min",
        "train_val_recon_last",
    ]
    return df[keep].rename(
        columns={
            "run_pretty": "run",
            "decoder_aulc_median": "decoder_AULC_median",
            "decoder_delta_vs_control_mean": "mean_delta",
            "decoder_delta_vs_control_max": "max_delta",
            "decoder_worse_than_control_count": "worse/16",
            "diag_li_A_full_p95": "exact_A_p95",
            "quality_decoded_acc_min": "min_decoded_acc",
            "train_val_recon_last": "final_val_recon",
        }
    )


def alpha_recon_table() -> pd.DataFrame:
    df = read_csv(BASE / "a_alpha_recon_li_a_summary.csv")
    if df.empty:
        return df
    keep = [
        "run_pretty",
        "alpha",
        "family",
        "train_val_recon_last",
        "train_precond_a_loss_median",
        "train_precond_a_loss_p99",
        "train_precond_effective_loss_median",
        "train_precond_effective_loss_p99",
        "diag_li_A_full_p95",
        "decoder_delta_vs_control_mean",
    ]
    return df[keep].rename(
        columns={
            "run_pretty": "run",
            "train_val_recon_last": "val_recon",
            "train_precond_a_loss_median": "raw_A_med",
            "train_precond_a_loss_p99": "raw_A_p99",
            "train_precond_effective_loss_median": "eff_A_med",
            "train_precond_effective_loss_p99": "eff_A_p99",
            "diag_li_A_full_p95": "exact_A_p95",
            "decoder_delta_vs_control_mean": "decoder_delta",
        }
    )


def extended_summary_table() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    mapping = {
        "A_unclipped": BASE / "extended_downstream/c0p001_unclipped_fixedlr_64eval/paired_downstream_summary.csv",
        "A_clip5": BASE / "extended_downstream/c0p001_clip5_fixedlr_64eval/paired_downstream_summary.csv",
        "A_clip20": BASE / "extended_downstream/c0p001_clip20_fixedlr_64eval/paired_downstream_summary.csv",
    }
    for label, path in mapping.items():
        df = read_csv(path)
        if df.empty:
            continue
        df = df[(df["method"].astype(str) == "decoder_latent") & df["metric"].astype(str).isin(["test_mean_loss_delta", "step0_test_loss_delta", "post0_test_loss_mean_delta"])].copy()
        df.insert(0, "variant", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["variant", "metric", "n", "mean", "median", "bootstrap95_low", "bootstrap95_high", "worse_count_A_gt_control", "better_count_A_lt_control", "min", "max"]
    return out[keep]


def selected_config_table() -> pd.DataFrame:
    root = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    names = {
        "baseline_pretrained": "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0",
        "current_control": "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "current_A": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "current_A_clip5": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_clip5_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
        "current_A_clip20": "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_clip20_fc2recon_lam0p03_marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_clean_harness_v1_seed0",
    }
    keys = [
        "latent_dim",
        "vae_loss_kind",
        "beta_kl",
        "vae_steps",
        "vae_lr",
        "vae_batch_size",
        "vae_block_recon_coeff",
        "vae_function_anchor_coeff",
        "vae_function_anchor_loss_kind",
        "vae_precond_loss_kind",
        "vae_precond_coeff",
        "vae_precond_max_grad_ratio",
        "vae_precond_loss_clip",
        "vae_precond_every",
        "vae_precond_samples",
        "vae_precond_pairs",
        "vae_precond_estimator_scope",
        "vae_precond_hvp_mode",
        "tune_starts",
        "eval_starts",
        "downstream_steps",
    ]
    rows: list[dict[str, object]] = []
    for label, name in names.items():
        path = root / name / "config.json"
        if not path.is_file():
            continue
        import json

        cfg = json.loads(path.read_text(encoding="utf-8")).get("config", {})
        row: dict[str, object] = {"label": label}
        for key in keys:
            row[key] = cfg.get(key, "NA")
        rows.append(row)
    return pd.DataFrame(rows)


def key_figures() -> list[tuple[Path, str]]:
    pairs = [
        (REPORT_DIR / "a_case_proxy_downstream_intervention_summary.png", "Final 64-start intervention summary: clipping changes the A proxy, but downstream does not robustly improve."),
        (REPORT_DIR / "a_case_path_latent_metric_summary.png", "Path decomposition and latent metric summary: downstream-relevant decoded-start and tangent/optimizer components barely move."),
        (BASE / "extended_downstream/c0p001_unclipped_fixedlr_64eval/paired_downstream_deltas.png", "64 paired starts, A unclipped vs control. Raw deltas are zero by construction; decoder-latent deltas are near zero."),
        (BASE / "extended_downstream/c0p001_clip5_fixedlr_64eval/paired_downstream_deltas.png", "64 paired starts, A clip5 vs control. Proxy tail improves, but step0 decoded-start outliers appear."),
        (BASE / "extended_downstream/c0p001_clip20_fixedlr_64eval/paired_downstream_deltas.png", "64 paired starts, A clip20 vs control. Proxy tail improves strongly, downstream remains near control."),
        (BASE / "a_alpha_recon_li_a_by_alpha.png", "Earlier alpha/reconstruction/li_A diagnostic; this plot was suspicious because the old raw loss looked better under lower alpha."),
        (BASE / "a_alpha_recon_raw_li_a_clean.png", "Cleaned alpha plot separating raw li_A from reconstruction."),
        (BASE / "a_alpha_effective_li_a_fixed_cap_clean.png", "Effective capped A contribution: higher alpha is mostly absorbed by the relative gradient cap."),
        (BASE / "a_fixed_alpha_cap_mechanics.png", "Fixed-alpha cap mechanics: alpha-weighted contribution and gradient scale move inversely under the cap."),
        (BASE / "a_exact_li_full_outliers_by_alpha.png", "Exact li_A outlier diagnostic by alpha."),
        (BASE / "a_gradient_domination_by_alpha.png", "A-gradient pressure diagnostic by alpha."),
        (BASE / "a_downstream_delta_with_ablation.png", "Old vs fixed harness downstream deltas with ablations."),
        (BASE / "a_training_pressure_with_ablation.png", "Training pressure under the A loss and the cap."),
        (BASE / "a_quality_decoded_acc_with_ablation.png", "Decoded-start quality under old/fixed/no-cap ablations."),
        (BASE / "arch_invariant_causal_analysis/path_decomposition_vs_downstream.png", "Architecture-invariant path decomposition vs downstream."),
        (BASE / "mechanism_debug_outliers/latent_metric_projection_summary.png", "Latent metric projection summary: raw optimizer direction mostly lies outside the decoder tangent."),
        (BASE / "mechanism_debug_outliers/reconstruction_error_task_alignment.png", "Reconstruction/task alignment diagnostic; this figure should be treated cautiously because it was earlier flagged as visually weak."),
        (BASE / "function_anchor_analysis/anchor_fix_summary_bars.png", "Function-anchor attempt: included as a rejected/fix attempt, not a final solution."),
        (BASE / "block_recon_analysis/lam0p03/optimizer_residual/mean_curve_deltas.png", "Optimizer residual decomposition for the block-reconstruction setup."),
        (BASE / "block_recon_analysis/lam0p03/optimizer_residual/block_aulc_gap_decomposition.png", "Block AULC gap decomposition."),
        (BASE / "block_recon_analysis/margin_mechanism/marginhuber_c0p001_clip20_training_and_downstream.png", "Current c=0.001 margin-huber clip20 training/downstream diagnostic."),
        (BASE / "block_recon_analysis/margin_mechanism/marginhuber_c0p001_clip20_margin_tail_vs_step0_gap.png", "Margin-tail vs step0 gap under clip20."),
        (BASE / "clean_harness_analysis/robustness/robustness_test_curve_decomposition.png", "Robustness decomposition for clean harness."),
        (BASE / "estimator_variance/current_A_pairs4/a_estimator_distribution.png", "Estimator variance: current A, 4 pairs."),
        (BASE / "estimator_variance/current_A_pairs16/a_estimator_distribution.png", "Estimator variance: current A, 16 pairs."),
    ]
    return [(p, c) for p, c in pairs if p.is_file()]


def all_pngs_ordered() -> list[Path]:
    priority = [p for p, _ in key_figures()]
    seen = {p.resolve() for p in priority}
    rest = sorted(p for p in BASE.rglob("*.png") if p.resolve() not in seen)
    return priority + rest


def build_pdf() -> tuple[Path, int, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page_count = 0
    image_count = 0
    with PdfPages(OUT_PDF) as pdf:
        add_text_page(
            pdf,
            "Variant A complete review packet",
            [
                "Scope: CELO TinyBigVAE / Variant A (`li_a_hvp`) causal debugging and downstream evaluation artifacts produced up to 2026-07-06.",
                "This packet is meant for an external reviewer/model. It contains: the corrected high-level conclusion, implementation setup, experiment setup, key numerical tables, claim-to-evidence map, and a full PNG appendix with every graph currently stored under the Variant A causal-debug artifact directory.",
                "Sign convention: A-control downstream deltas are positive when A is worse.",
                "Main output path: " + _rel(OUT_PDF),
            ],
            subtitle="Generated from stored CSV/PNG/MD artifacts; no new training is run.",
        )
        page_count += 1

        add_text_page(
            pdf,
            "Executive conclusion",
            [
                "## Narrow conclusion",
                "Current A did not robustly outperform the matched control on downstream optimization. A/HVP is active and can move the local curvature proxy, but the downstream-relevant decoded-start and latent optimizer/path components barely improve.",
                "## Important correction",
                "This is not evidence that Li's `c_3` objective or Kron/PSGD is bad. Kron is a positive-control example for the idea that a faithful `P=Q^TQ` preconditioner fitted by Li-style criteria can beat Adam-like baselines. The result here is evidence against the current VAE Variant A approximation/harness as a faithful downstream-improving realization.",
                "## Current best interpretation",
                "The most plausible unresolved cause is a surrogate-faithfulness / implementation gap: the current rectangular VAE decoder chart and latent quadratic majorant may not approximate the full-space Li preconditioner well enough. Latent Adam mismatch remains a harness confounder, but it is not yet an A-specific final cause.",
                "## What would distinguish the remaining mechanisms",
                "Run exact-CG/full-space `c_3` audits separating first and inverse/barrier terms, and run metric-aware/projected or direct PSGD/Kron downstream controls. If exact `c_3` is not better despite A proxy improvement, the cause is surrogate faithfulness. If exact `c_3` is good but latent Adam fails, the cause moves to optimizer/harness mismatch.",
            ],
        )
        page_count += 1

        add_text_page(
            pdf,
            "Implementation setup: losses",
            [
                "## Base VAE loss",
                "The VAE reconstructs the whole flattened CELO MLP weight vector. The accepted pretrained baseline has weight pool shape `(15744, 17098)` and latent dimension `512`. The base loss in `core.py:vae_loss` is `MSE(recon_norm, x_norm) + beta_kl * KL`, with `vae_loss_kind=mse` and `beta_kl=1e-6` in the current A/control runs.",
                "## Variant A surrogate",
                "Implemented in `post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py`. For `vae_precond_loss_kind=li_a_hvp`, the code samples normalized Gaussian latent probes `u` with norm `sqrt(latent_dim) * probe_scale`, computes latent HVP vectors `h`, then applies the two-sample majorant:",
                "`L_A = mean((h1 dot h2)^2 - ||h1||^2 - ||h2||^2 + dim) / dim`.",
                "This is the quadratic majorant `||M-I||_F^2/dim` for the latent HVP covariance target. It is not the exact full-space Li criterion because it does not include the explicit `P^{-1}` barrier term.",
                "## HVP mode",
                "The current runs use `vae_precond_hvp_mode=stopped_composite`. The implementation computes the composite latent Hessian with an x-space Hessian term through `J^T H_x J u` plus the decoder-curvature term from the stopped x-gradient path.",
                "## A-loss scheduling and caps",
                "The preconditioner regularizer runs every `vae_precond_every=10` VAE steps, uses `vae_precond_samples=2`, `vae_precond_pairs=4`, batch size `128`, local estimator scope, and a 1000-step ramp. The total regularizer is `vae_precond_coeff * clipped_A_loss + burg_coeff * burg_loss`; current A uses `coeff=0.01` and `burg_coeff=0`.",
                "The relative gradient cap computes base VAE gradient norm and A-gradient norm separately, then scales the A gradient so the scaled A norm is at most `vae_precond_max_grad_ratio * base_grad_norm`. In current A this ratio is `0.25`. The clip5/clip20 variants additionally clamp the raw A loss to `[-5,5]` or `[-20,20]` before multiplying by alpha.",
                "## Current CELO quality guards",
                "The current c=0.001 setup adds `fc2.weight` block reconstruction with `vae_block_recon_coeff=0.03`, normalized block space, and 1000-step ramp. It also adds a function/head margin guard with `vae_function_anchor_coeff=0.001`, `head_margin_drop_huber`, `huber_delta=0.05`, `head_top_fraction=1.0`, `head_example_top_fraction=0.10`. These are matched between current A and current control.",
            ],
        )
        page_count += 1

        add_text_page(
            pdf,
            "Implementation setup: experiments",
            [
                "## Phase 1: old A-case alpha/debug harness",
                "Compared control, old global alpha .01/.05, fixed local+cap alpha .01/.05, and local no-cap alpha .01. This phase diagnosed the original non-monotonic/bad A behavior. Key finding: the old global estimator and uncapped relative A-gradient pressure were a technical training/harness failure; fixed local+cap stabilized the run but did not prove downstream improvement.",
                "## Phase 2: current c=0.001 block-reconstruction/margin-huber setup",
                "Compared current control, A unclipped, A clip5, and A clip20. All use the same CELO TinyBigVAE architecture and the same fc2/block/function guard setup. The extended downstream evaluation uses 64 paired heldout-final starts, fixed selected LR (`raw=0.0003`, `decoder_latent=0.01`), 300 steps, and deterministic train minibatches.",
                "## Downstream harness",
                "`downstream.py` implements `raw` as Adam directly on `w0`, and `decoder_latent` as Adam on `z0=E(w0)` with decoded weights `D(z)` used for task loss. Test/train curves are recorded every `downstream_eval_every=25` steps. AULC is the mean of recorded train losses; the extended report also uses test-curve deltas.",
                "## Extended downstream cache guard",
                "`scripts/evaluate_variant_a_extended_downstream.py` validates per-label cache metadata: run directory, config/checkpoint/weight-pool/selected-LR signatures, methods, steps, raw/decoder LR, and exact start bank. The paired merge includes `lr`, which prevents silent LR mismatch.",
                "## Metric/path diagnostics",
                "Path decomposition compares raw Adam from `w0`, raw Adam from decoded `D(E(w0))`, and latent Adam from the same decoded start. Latent metric probes measure raw Adam's normal residual relative to decoder tangent space, tangent projection norm, latent Adam cosine with `-grad`, and metric condition.",
                "## Literature correction",
                "After reading Li/PSGD/Kron sources, the correct framing is: current A approximates a Li-style preconditioner poorly or incompletely; it is not a direct test of the exact Li criterion or of Kron/PSGD optimizers.",
            ],
        )
        page_count += 1

        cfg_table = selected_config_table()
        if not cfg_table.empty:
            add_table_page(pdf, "Selected run config fields", cfg_table, max_rows=8)
            page_count += math.ceil(len(cfg_table) / 8)

        for title, table, note in [
            (
                "Current c=0.001 64-start downstream intervention summary",
                downstream_summary_table(),
                "A-control deltas; positive is worse. Proxy improves under clipping, downstream does not robustly improve.",
            ),
            (
                "Extended downstream summaries by variant",
                extended_summary_table(),
                "Only decoder_latent rows are shown here; raw paired deltas are exactly zero in these artifacts.",
            ),
            (
                "Old/fixed/no-cap alpha debug summary",
                alpha_debug_table(),
                "This phase diagnosed the old A harness; it is not the final current c=0.001 setup.",
            ),
            (
                "Alpha vs reconstruction and raw/effective A loss",
                alpha_recon_table(),
                "Raw A loss is logged before alpha; effective A contribution includes coefficient/cap mechanics.",
            ),
            (
                "Path component absolute summary",
                read_csv(REPORT_DIR / "path_component_absolute_summary.csv"),
                "Decoded-start and latent-only penalties are large and nearly shared by A/control.",
            ),
            (
                "Path component paired deltas",
                read_csv(REPORT_DIR / "path_component_paired_delta_summary.csv"),
                "A-control changes in downstream-relevant path components are tiny.",
            ),
            (
                "Latent metric key summary",
                read_csv(REPORT_DIR / "latent_metric_key_summary.csv"),
                "Raw optimizer direction is mostly outside the decoder tangent for both A and control.",
            ),
        ]:
            if not table.empty:
                add_table_page(pdf, title, table, note=note)
                page_count += math.ceil(len(table) / 24)

        add_text_page(
            pdf,
            "Claim-to-evidence map",
            [
                "- Claim: Old global A alpha runs were invalid evidence. Evidence: old global alpha .01/.05 worsened downstream and exact-A outliers; fixed local+cap stabilized behavior.",
                "- Claim: Reducing A proxy tail alone is insufficient. Evidence: clip5/clip20 reduce `li_A_p95`, but 64-start downstream deltas do not become robustly negative; clip5 adds step0 decoded-start outliers.",
                "- Claim: Downstream components barely moved. Evidence: path decomposition median A-control deltas are around 1e-4 or smaller for decoded-start and latent-only penalties.",
                "- Claim: Shared latent chart/optimizer bottleneck remains. Evidence: normal residual fraction is about 0.90 for control, A, and A_clip20; latent Adam cosine with `-grad` is about 0.106.",
                "- Claim: This is not a negative result about Li/Kron. Evidence: current A uses rectangular low-rank `J J^T`, a latent quadratic surrogate, offline VAE training, and latent Adam; Li/Kron fits invertible `P=Q^TQ` and applies the preconditioner directly.",
                "- Leading unresolved mechanism: surrogate-faithfulness gap. Exact-CG/full-space c3 audit is needed to decide whether current A proxy improvement corresponds to the actual Li criterion.",
            ],
        )
        page_count += 1

        add_text_page(
            pdf,
            "Hypotheses still alive after this packet",
            [
                "H1 surrogate-faithfulness gap: A proxy improves while exact full-space `c3`, especially inverse/barrier term, does not. This is currently the strongest Li/Kron-consistent suspicion.",
                "H2 rank/tangent bottleneck: latent dimension 512 is much smaller than weight dimension 17098, so `J J^T` is singular in full weight space. Current latent-space A/B/E surrogates can be blind to normal-space loss.",
                "H3 optimizer/harness mismatch: latent Adam is not the Li/Kron update. It may invalidate using latent Adam downstream as the only measure of preconditioner quality, but current one-step probes show this bottleneck is shared by A and control.",
                "H4 offline amortization mismatch: Li/Kron updates preconditioner online along the trajectory, while the VAE decoder is trained offline and frozen for downstream.",
                "H5 estimator/training instability: quartic HVP tails and gradient pressure are real, but clipping did not fix downstream; therefore this is a mediator/side pathology, not a complete cause by itself.",
                "H6 noise-floor/damping mismatch: Li's criterion has a noise-aware inverse/regularization effect; clipping and finite HVP batching are not equivalent to the exact noise damping.",
            ],
        )
        page_count += 1

        add_text_page(
            pdf,
            "Key figures",
            [
                "The next pages show the most important figures with captions. After that, the PDF includes a full PNG appendix containing every graph under `docs/reparam_preconditioning_experiments/variant_A_causal_debug` exactly once. Some earlier plots were already flagged as weak or suspicious; they are included for auditability, not because every one is accepted evidence.",
            ],
        )
        page_count += 1

        key_paths = {p.resolve() for p, _ in key_figures()}
        for path, caption in key_figures():
            if add_image_page(pdf, path, caption=caption, section="Key figure"):
                page_count += 1
                image_count += 1

        add_text_page(
            pdf,
            "Full PNG appendix",
            [
                "Every PNG under the Variant A causal-debug artifact directory follows, excluding key-figure duplicates already shown above. Each page title is the repository-relative path. This is intentionally verbose so the external reviewer can inspect plots I may not have emphasized.",
            ],
        )
        page_count += 1

        for path in all_pngs_ordered():
            if path.resolve() in key_paths:
                continue
            if add_image_page(pdf, path, section="Full graph appendix"):
                page_count += 1
                image_count += 1

        add_text_page(
            pdf,
            "Source artifacts and scripts",
            [
                "Core theory/report sources:",
                "- docs/reparam_preconditioning.tex",
                "- docs/reparam_preconditioning_experiments/variant_A_causal_debug_report.md",
                "- docs/reparam_preconditioning_experiments/variant_A_causal_debug/a_case_causal_report/variant_A_causal_report.md",
                "- docs/reparam_preconditioning_experiments/variant_A_causal_debug/agent_journals/li_psgd_kron_literature_notes.md",
                "Core implementation sources:",
                "- post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py",
                "- post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py",
                "- post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/downstream.py",
                "- scripts/run_variant_a_frontier.py",
                "- scripts/run_variant_a_block_recon_guard.py",
                "- scripts/evaluate_variant_a_extended_downstream.py",
                "- scripts/evaluate_variant_a_metric_aware_downstream.py",
                "Key CSVs:",
                "- " + _rel(REPORT_DIR / "a_case_proxy_downstream_intervention_summary.csv"),
                "- " + _rel(REPORT_DIR / "path_component_absolute_summary.csv"),
                "- " + _rel(REPORT_DIR / "path_component_paired_delta_summary.csv"),
                "- " + _rel(REPORT_DIR / "latent_metric_key_summary.csv"),
                "- " + _rel(BASE / "a_case_run_summary_with_ablation.csv"),
                "- " + _rel(BASE / "a_alpha_recon_li_a_summary.csv"),
                "- " + _rel(BASE / "extended_downstream/c0p001_unclipped_fixedlr_64eval/paired_downstream_summary.csv"),
                "- " + _rel(BASE / "extended_downstream/c0p001_clip5_fixedlr_64eval/paired_downstream_summary.csv"),
                "- " + _rel(BASE / "extended_downstream/c0p001_clip20_fixedlr_64eval/paired_downstream_summary.csv"),
            ],
        )
        page_count += 1

    return OUT_PDF, page_count, image_count


def main() -> None:
    pdf_path, pages, images = build_pdf()
    size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"[variant_a_review_pdf] wrote {pdf_path}")
    print(f"[variant_a_review_pdf] pages={pages} image_pages={images} size_mb={size_mb:.2f}")


if __name__ == "__main__":
    main()
