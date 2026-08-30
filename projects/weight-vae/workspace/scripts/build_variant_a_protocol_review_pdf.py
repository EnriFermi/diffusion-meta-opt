from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

# Running files from scripts/ can otherwise shadow stdlib modules via scripts/inspect.
SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
OUT_DIR = BASE / "a_case_review_packet"
OUT_PDF = OUT_DIR / "variant_A_protocol_review_packet_2026-07-06.pdf"
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def bpath(relative: str) -> Path:
    return BASE / relative


def exists(relative: str) -> bool:
    return bpath(relative).is_file()


def read_csv(relative: str) -> pd.DataFrame:
    path = bpath(relative)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(relative: str) -> dict[str, Any]:
    path = bpath(relative)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"value": value}


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if not math.isfinite(val):
            return "NA"
        if val == 0.0:
            return "0"
        if abs(val) >= 1000 or abs(val) < 1e-3:
            return f"{val:.{digits}g}"
        return f"{val:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def select_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]].copy()


def shorten(text: object, limit: int = 90) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    keep = max(8, (limit - 3) // 2)
    return value[:keep] + "..." + value[-keep:]


def wrap_lines(text: str, width: int = 106) -> list[str]:
    lines: list[str] = []
    for para in str(text).splitlines():
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width, replace_whitespace=False, break_long_words=False))
    return lines


def new_text_fig(title: str, subtitle: str | None = None) -> tuple[plt.Figure, plt.Axes, float]:
    fig = plt.figure(figsize=(8.5, 11.0), dpi=150)
    ax = fig.add_axes([0.065, 0.055, 0.87, 0.90])
    ax.axis("off")
    y = 0.99
    ax.text(0.0, y, title, ha="left", va="top", fontsize=17, fontweight="bold")
    y -= 0.042
    if subtitle:
        for line in wrap_lines(subtitle, 112):
            ax.text(0.0, y, line, ha="left", va="top", fontsize=8.8, color="#4a4a4a")
            y -= 0.020
        y -= 0.010
    return fig, ax, y


def save_text_fig(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def add_text_pages(pdf: PdfPages, title: str, body: list[str], *, subtitle: str | None = None) -> None:
    fig, ax, y = new_text_fig(title, subtitle)
    page = 1

    def ensure_space(needed: float = 0.035) -> None:
        nonlocal fig, ax, y, page
        if y >= needed:
            return
        save_text_fig(pdf, fig)
        page += 1
        fig, ax, y = new_text_fig(f"{title} (continued {page})", subtitle=None)

    for block in body:
        if block is None:
            continue
        text = str(block)
        if text == "":
            ensure_space(0.04)
            y -= 0.018
            continue
        if text.startswith("## "):
            ensure_space(0.065)
            y -= 0.008
            ax.text(0.0, y, text[3:], ha="left", va="top", fontsize=12.6, fontweight="bold")
            y -= 0.030
            continue
        if text.startswith("### "):
            ensure_space(0.055)
            ax.text(0.0, y, text[4:], ha="left", va="top", fontsize=10.8, fontweight="bold")
            y -= 0.026
            continue
        if text.startswith("- "):
            lines = wrap_lines(text[2:], 104)
            ensure_space(0.028 * max(1, len(lines)))
            ax.text(0.012, y, "-", ha="left", va="top", fontsize=8.8)
            ax.text(0.038, y, lines[0] if lines else "", ha="left", va="top", fontsize=8.4)
            y -= 0.020
            for line in lines[1:]:
                ensure_space(0.030)
                ax.text(0.038, y, line, ha="left", va="top", fontsize=8.4)
                y -= 0.020
            continue
        if text.startswith("    "):
            lines = wrap_lines(text.strip(), 96)
            for line in lines:
                ensure_space(0.026)
                ax.text(0.025, y, line, ha="left", va="top", fontsize=7.9, family="monospace")
                y -= 0.019
            y -= 0.004
            continue
        for line in wrap_lines(text, 108):
            ensure_space(0.030)
            if line == "":
                y -= 0.016
            else:
                ax.text(0.0, y, line, ha="left", va="top", fontsize=8.8)
                y -= 0.021
        y -= 0.004
    save_text_fig(pdf, fig)


def table_cells(df: pd.DataFrame, max_chars: int = 62) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in df.to_numpy(dtype=object):
        rows.append([shorten(fmt(value), max_chars) for value in row])
    return rows


def add_table_pages(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    *,
    note: str | None = None,
    max_rows: int = 24,
    max_chars: int = 64,
) -> None:
    if df.empty:
        add_text_pages(pdf, title, ["No rows available.", note or ""])
        return
    chunks = [df.iloc[i : i + max_rows].copy() for i in range(0, len(df), max_rows)]
    for idx, chunk in enumerate(chunks, start=1):
        fig = plt.figure(figsize=(11.0, 8.5), dpi=150)
        ax = fig.add_axes([0.020, 0.045, 0.960, 0.820])
        ax.axis("off")
        page_title = title if len(chunks) == 1 else f"{title} ({idx}/{len(chunks)})"
        fig.text(0.020, 0.970, page_title, ha="left", va="top", fontsize=14.5, fontweight="bold")
        if note and idx == 1:
            fig.text(0.020, 0.930, "\n".join(wrap_lines(note, 150)), ha="left", va="top", fontsize=7.7, color="#333333")
        table = ax.table(
            cellText=table_cells(chunk, max_chars=max_chars),
            colLabels=[shorten(col, 36) for col in chunk.columns],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        font_size = 5.3 if len(chunk.columns) >= 8 else 6.2
        table.set_fontsize(font_size)
        table.scale(1.0, 1.33)
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor("#b7b7b7")
            if row == 0:
                cell.set_facecolor("#e7edf6")
                cell.set_text_props(weight="bold")
        pdf.savefig(fig)
        plt.close(fig)


def add_image_page(pdf: PdfPages, relative: str, *, caption: str, section: str = "Figure") -> bool:
    path = bpath(relative)
    if not path.is_file():
        return False
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return False
    fig = plt.figure(figsize=(11.0, 8.5), dpi=150)
    ax = fig.add_axes([0.035, 0.115, 0.930, 0.760])
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.035, 0.972, section, ha="left", va="top", fontsize=8.0, color="#666666")
    fig.text(0.035, 0.946, rel(path), ha="left", va="top", fontsize=8.8, fontweight="bold")
    fig.text(0.035, 0.060, "\n".join(wrap_lines(caption, 150)), ha="left", va="top", fontsize=7.8, color="#222222")
    pdf.savefig(fig)
    plt.close(fig)
    return True


def add_image_grid(pdf: PdfPages, title: str, relatives: list[str], *, captions: dict[str, str] | None = None) -> None:
    existing = [r for r in relatives if bpath(r).is_file()]
    if not existing:
        return
    for page_idx in range(0, len(existing), 4):
        batch = existing[page_idx : page_idx + 4]
        fig = plt.figure(figsize=(11.0, 8.5), dpi=150)
        fig.text(0.025, 0.970, title, ha="left", va="top", fontsize=14.5, fontweight="bold")
        for i, relative in enumerate(batch):
            row, col = divmod(i, 2)
            left = 0.035 + col * 0.485
            bottom = 0.525 - row * 0.425
            ax = fig.add_axes([left, bottom, 0.445, 0.315])
            try:
                image = Image.open(bpath(relative)).convert("RGB")
                ax.imshow(image)
            except Exception:
                ax.text(0.5, 0.5, "Unreadable image", ha="center", va="center")
            ax.axis("off")
            fig.text(left, bottom - 0.016, shorten(relative, 92), ha="left", va="top", fontsize=6.2, fontweight="bold")
            if captions and relative in captions:
                fig.text(left, bottom - 0.040, "\n".join(wrap_lines(captions[relative], 72)[:3]), ha="left", va="top", fontsize=5.8)
        pdf.savefig(fig)
        plt.close(fig)


def csv_shape_and_cols(path: Path) -> tuple[str, list[str]]:
    try:
        header = pd.read_csv(path, nrows=0)
        cols = [str(c) for c in header.columns]
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            rows = max(0, sum(1 for _ in handle) - 1)
        return f"{rows}x{len(cols)}", cols
    except Exception as exc:
        return f"csv unreadable: {exc}", []


def infer_key_columns(cols: list[str]) -> str:
    preferred = [
        "variant",
        "variant_label",
        "label",
        "suffix",
        "run_label",
        "method",
        "metric",
        "task_name",
        "source_weight_index",
        "start_index",
        "step",
        "alpha",
        "cap",
        "lam",
        "block_group",
        "action",
        "direction_method",
        "scale_anchor",
    ]
    hits = [col for col in preferred if col in cols]
    if hits:
        return ", ".join(hits[:8])
    return ", ".join(cols[:8])


GROUP_RULES: list[tuple[str, str, str]] = [
    ("a_case_causal_report/harness_validation", "harness validation", "checks cache/start/LR validity for extended downstream harness"),
    ("a_case_causal_report", "current A causal report", "summarizes c=0.001 intervention, path decomposition, latent metric evidence"),
    ("extended_downstream", "extended downstream", "64-start fixed-LR A-control downstream comparisons and clip interventions"),
    ("frontier_analysis_current_control_recheck", "current-control recheck", "matched current-control step0, curve, and frontier mechanism deltas"),
    ("frontier_analysis_current_control", "current-control analysis", "matched current-control step0, curve, and frontier mechanism deltas"),
    ("frontier_analysis", "frontier analysis", "alpha/cap frontier proxy-downstream tradeoff and paired start deltas"),
    ("frontier_analysis_partial", "frontier partial analysis", "partial alpha/cap frontier checks"),
    ("frontier_single_summaries", "frontier run summaries", "single-run summaries feeding frontier analysis"),
    ("block_recon_runs", "block recon run summaries", "fc2 reconstruction/function/head guard run-level comparisons"),
    ("block_recon_analysis/lam0p03/optimizer_residual", "optimizer residual", "step0/path residual and latent optimizer geometry for block lam0.03"),
    ("block_recon_analysis/lam0p03/residual_decomposition", "block residual decomposition", "A-control step0/path decomposition for lam0.03"),
    ("block_recon_analysis/lam0p03_cap0p25/head_splice", "head splice cap0.25", "cross-variant classifier-head splice for capped block-recon run"),
    ("block_recon_analysis/lam0p03/head_splice", "head splice lam0.03", "cross-variant classifier-head splice for lam0.03"),
    ("block_recon_analysis/lam0p06/head_splice", "head splice lam0.06", "cross-variant classifier-head splice for lam0.06"),
    ("block_recon_analysis/lam0p06/raw_decoded_block_splice", "raw-decoded splice lam0.06", "raw-block rescue/intervention on decoded starts"),
    ("block_recon_analysis/lam_sweep", "lambda sweep", "block-recon lambda tradeoff against downstream and step0"),
    ("block_recon_analysis/lam_cap_sweep", "lambda/cap sweep", "joint block-recon lambda and A cap tradeoff"),
    ("block_recon_analysis/head_error_sensitivity", "head error sensitivity", "head MSE/logit error sensitivity against step0 decoded loss"),
    ("block_recon_analysis/margin_mechanism", "margin mechanism", "head CE/margin tail, row direction, and fc2 scale/direction interventions"),
    ("block_recon_analysis", "block recon analysis", "block reconstruction and decoded-start diagnostics"),
    ("block_splice_fixed_nocap_full16", "block splice fixed no-cap", "raw-block rescue interventions for fixed/no-cap variants"),
    ("causal_pair_analysis/downstream_weight_paths", "causal pair weight paths", "raw vs decoded-start vs latent downstream path decomposition"),
    ("causal_pair_analysis/latent_metric", "causal pair latent metric", "J^T J, tangent projection, natural/projected one-step probes"),
    ("causal_pair_analysis", "causal pair analysis", "paired validity, downstream deltas, spikes, and pareto summaries"),
    ("clean_harness_analysis/anchor_sweep", "anchor sweep", "logit/function-anchor coefficient sweep under matched clean harness"),
    ("clean_harness_analysis/block_splice_anchor_sweep", "anchor block splice", "block-splice localization for anchor sweep tail starts"),
    ("clean_harness_analysis/causal_mechanism", "clean causal mechanism", "path decomposition and block-splice mechanism diagnostics"),
    ("clean_harness_analysis/cross_variant_head_splice", "cross-variant head splice", "A/control decoded head swap diagnostics"),
    ("clean_harness_analysis/latent_metric", "clean latent metric", "metric-aware optimizer probes on clean/anchor starts"),
    ("clean_harness_analysis/robustness", "clean robustness", "bootstrap/sign-flip/window/leave-one-out robustness checks"),
    ("clean_harness_analysis/weight_paths_anchor_sweep", "anchor weight paths", "path decomposition for anchor sweep"),
    ("clean_harness_analysis", "clean harness", "matched clean A/control analysis"),
    ("estimator_variance", "estimator variance", "A estimator sample distribution and HVP tail checks"),
    ("arch_invariant_causal_analysis", "arch-invariant synthesis", "joined path, estimator, latent-metric, and downstream summaries"),
    ("mechanism_debug_outliers", "outlier mechanism debug", "reconstruction, logit saturation, and downstream predictor correlations"),
    ("function_anchor_runs", "function anchor runs", "standalone function/logit anchor control vs A run summaries"),
    ("head_guard_runs", "head guard runs", "standalone classifier-head guard control vs A run summaries"),
    ("agent_journals", "agent journals", "mechanism notes, literature correction, and design caveats"),
]


def group_and_support(relative: str) -> tuple[str, str]:
    for prefix, group, support in GROUP_RULES:
        if relative.startswith(prefix):
            return group, support
    if relative.startswith("a_"):
        return "top-level alpha/ablation diagnostics", "alpha/global-local/cap estimator, quality, and downstream ablation evidence"
    return "variant A root", "root-level figures and summaries for Variant A debug packet"


def build_inventory() -> pd.DataFrame:
    suffixes = {".csv", ".json", ".md", ".png"}
    rows: list[dict[str, str]] = []
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = rel(path)
        local_rel = str(path.relative_to(BASE))
        group, support = group_and_support(local_rel)
        shape = ""
        key_columns = ""
        if path.suffix.lower() == ".csv":
            shape, cols = csv_shape_and_cols(path)
            key_columns = infer_key_columns(cols)
        elif path.suffix.lower() == ".png":
            try:
                with Image.open(path) as image:
                    shape = f"{image.width}x{image.height}px"
            except Exception as exc:
                shape = f"png unreadable: {exc}"
            key_columns = "PNG figure"
        elif path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    shape = f"json object; keys={len(payload)}"
                    key_columns = ", ".join(list(payload.keys())[:8])
                elif isinstance(payload, list):
                    shape = f"json list; items={len(payload)}"
                    key_columns = "list items"
                else:
                    shape = type(payload).__name__
                    key_columns = "json scalar"
            except Exception as exc:
                shape = f"json unreadable: {exc}"
                key_columns = ""
        elif path.suffix.lower() == ".md":
            try:
                text = path.read_text(encoding="utf-8")
                shape = f"{len(text.splitlines())} lines"
            except Exception as exc:
                shape = f"md unreadable: {exc}"
            key_columns = "Markdown notes"
        rows.append(
            {
                "group": group,
                "path": relative,
                "shape": shape,
                "key columns": key_columns,
                "supports": support,
            }
        )
    return pd.DataFrame(rows)


def compact_table(relative: str, columns: list[str], *, limit: int | None = None) -> pd.DataFrame:
    df = read_csv(relative)
    if df.empty:
        return df
    out = select_existing(df, columns)
    if limit is not None:
        out = out.head(limit)
    return out


def downstream_combined_summary() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, relative in [
        ("unclipped", "extended_downstream/c0p001_unclipped_fixedlr_64eval/paired_downstream_summary.csv"),
        ("clip5", "extended_downstream/c0p001_clip5_fixedlr_64eval/paired_downstream_summary.csv"),
        ("clip20", "extended_downstream/c0p001_clip20_fixedlr_64eval/paired_downstream_summary.csv"),
    ]:
        df = read_csv(relative)
        if df.empty:
            continue
        df.insert(0, "variant", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return select_existing(
        combined,
        [
            "variant",
            "method",
            "metric",
            "n",
            "mean",
            "median",
            "worse_count_A_gt_control",
            "better_count_A_lt_control",
            "bootstrap95_low",
            "bootstrap95_high",
            "min",
            "max",
        ],
    )


def causal_pair_combined_summary() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, relative in [
        ("marginhuber_c0p001", "causal_pair_analysis/marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_downstream_summary.csv"),
        ("marginhuber_c0p001_clip5", "causal_pair_analysis/marginhuber_c0p001_clip5_downstream_summary.csv"),
        ("marginhuber_c0p001_clip20", "causal_pair_analysis/marginhuber_c0p001_clip20_downstream_summary.csv"),
    ]:
        df = read_csv(relative)
        if df.empty:
            continue
        df.insert(0, "pair", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return select_existing(
        combined,
        [
            "pair",
            "method",
            "metric",
            "n",
            "mean",
            "median",
            "worse_count_A_gt_control",
            "better_count_A_lt_control",
            "exact_sign_p",
            "bootstrap95_low",
            "bootstrap95_high",
        ],
    )


def block_recon_combined_summary() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for relative in [
        "block_recon_runs/block_recon_summary_lam0p03.csv",
        "block_recon_runs/block_recon_summary_lam0p03_cap0p25.csv",
        "block_recon_runs/block_recon_summary_lam0p06.csv",
        "block_recon_runs/block_recon_marginhuber_summary_c0p001_m0_d0p05_toprec1_topex0p1.csv",
        "block_recon_runs/block_recon_marginhuber_clip_intervention_summary.csv",
    ]:
        df = read_csv(relative)
        if not df.empty:
            df.insert(0, "source", relative.rsplit("/", 1)[-1])
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return select_existing(
        combined,
        [
            "source",
            "suffix",
            "alpha",
            "cap",
            "block_coeff",
            "head_coeff",
            "head_loss_kind",
            "precond_loss_clip",
            "decoder_eval_starts",
            "decoder_aulc_mean",
            "li_A_full_per_dim_p95",
            "reconstruction_rel_l2_median",
            "selected_decoder_lr",
            "train_block_recon_effective_loss_median",
            "train_function_anchor_effective_loss_median",
            "train_precond_effective_loss_median",
        ],
    )


def head_splice_combined_summary() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, relative in [
        ("lam0p03", "block_recon_analysis/lam0p03/head_splice/block_lam0p03_head_splice_summary.csv"),
        ("lam0p03_cap0p25", "block_recon_analysis/lam0p03_cap0p25/head_splice/block_lam0p03_cap0p25_head_splice_summary.csv"),
        ("lam0p06", "block_recon_analysis/lam0p06/head_splice/block_lam0p06_head_splice_summary.csv"),
    ]:
        df = read_csv(relative)
        if df.empty:
            continue
        df.insert(0, "splice_set", label)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    return select_existing(
        combined,
        [
            "splice_set",
            "pair",
            "start_group",
            "block_group",
            "action",
            "starts",
            "A_minus_control_test_loss_mean",
            "A_minus_control_test_loss_median",
            "control_block_rescue_fraction_test_mean",
            "control_block_rescue_fraction_test_median",
            "gap_after_candidate_vs_control_test_loss_mean",
        ],
    )


def code_reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file": "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/preconditioning.py",
                "used for": "preconditioner enablement, latent HVP modes, two-sample A majorant, Burg sketch, exact latent Hessian diagnostics",
            },
            {
                "file": "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/core.py",
                "used for": "CeloMetaMLP/FlatSpec, weight-pool generation, VAE architectures, block recon, row-direction, function-anchor losses, grad caps",
            },
            {
                "file": "post_train_research/loss_landscape_analysis/sage_cnn_vae_smoothing/downstream.py",
                "used for": "raw vs decoder_latent downstream optimization, LR tuning, AULC, reconstruction mismatch, train/test curve logging",
            },
            {
                "file": "scripts/run_variant_a_frontier.py",
                "used for": "frontier alpha/cap run labels, finetune setup, precond configs, downstream eval setup",
            },
            {
                "file": "scripts/run_variant_a_block_recon_guard.py",
                "used for": "fc2 block recon, head CE/Huber/margin, row-direction guard configs and run labels",
            },
            {
                "file": "scripts/evaluate_variant_a_extended_downstream.py",
                "used for": "64-start fixed-LR downstream cache signatures, start-bank selection, paired test_mean/step0/post0 metrics",
            },
            {
                "file": "scripts/analyze_variant_a_causal_pair.py",
                "used for": "completion audits, start-bank audit, curve completeness, paired deltas, bootstrap/sign tests",
            },
            {
                "file": "scripts/analyze_latent_metric_optimizer_confounds.py",
                "used for": "decoder Jacobian metric, tangent projection, normal residual, natural/projected update probes",
            },
            {
                "file": "scripts/analyze_decoder_block_splice_interventions.py",
                "used for": "raw-to-decoded block splice rescue/swap definitions and block groups",
            },
            {
                "file": "scripts/analyze_block_recon_head_splice.py",
                "used for": "A/control decoded classifier-head splice, recompute-vs-logged validation",
            },
            {
                "file": "docs/reparam_preconditioning_experiments/variant_A_causal_debug/agent_journals/li_psgd_kron_literature_notes.md",
                "used for": "scope correction: current A surrogate is not Li/Kron/PSGD negative evidence",
            },
        ]
    )


def catalog_entries() -> list[dict[str, Any]]:
    base_setup = (
        "Shared setup unless overridden: CeloMetaMLP weight vectors over MNIST and Fashion-MNIST, image_size=16, "
        "hidden_dim=64, num_classes=10, flat weight_dim=17098; baseline CELO config uses 384 source runs, 2000 "
        "source Adam steps, snapshots every 50, tau log-uniform in [0.5, 2.0], source LRs {3e-4, 1e-3, 3e-3}. "
        "VAE is tiny_big_vae/direct with latent_dim=512, hidden_dim=2048, beta_kl=1e-6. Variant finetunes use "
        "baseline weight_pool and VAE checkpoint, usually vae_steps=5000, vae_lr=3e-5, batch=128."
    )
    precond_setup = (
        "A runs use vae_precond_loss_kind=li_a_hvp, coeff alpha, every=10, samples=2, pairs=4 unless stated, "
        "batch_size=128, probe_scale=1.0, estimator_scope=local for current runs, hvp_mode=stopped_composite, "
        "ramp_steps=1000, diagnostic_grad_batches=64. Grad-ratio cap scales the precond gradient by "
        "min(1, cap * ||base_grad|| / ||precond_grad||)."
    )
    downstream_setup = (
        "Downstream compares raw Adam in weight space with decoder_latent Adam in z. Standard run-level eval uses "
        "tune_starts=8, eval_starts=16, downstream_steps=300, eval_every=25, batch=128. Extended downstream fixes "
        "selected LRs and evaluates 64 heldout-final starts after skip_starts=8."
    )
    metrics = (
        "Sign convention: A-control deltas; positive loss delta is worse. Downstream train AULC is mean train_loss "
        "over recorded curve rows. Extended test_mean_loss is mean test_loss per start/method; test_trapz_loss is "
        "trapezoid(test_loss, step)/(last-first). step0 is first curve row; post0 is mean over rows with step>0. "
        "li_A_full_per_dim is mean((eig(H_z)^2 - 1)^2). HVP proxy A uses (dot(h1,h2)^2 - ||h1||^2 - ||h2||^2 + dim)/dim."
    )
    return [
        {
            "title": "Top-Level Alpha, Scope, Cap, and Ablation Diagnostics",
            "question": "Does Variant A reduce latent curvature proxy without damaging decoded quality or downstream performance?",
            "runs": "control; old global alpha=.01/.05; local no-cap alpha=.01; fixed local cap alpha=.01/.05. Source summaries and plots at the root of variant_A_causal_debug.",
            "control": "No-A finetune and raw downstream are controls; A interventions vary estimator scope, alpha, and grad-ratio cap.",
            "setup": f"{base_setup} {precond_setup} {downstream_setup}",
            "measured": "decoder train-AULC, final test loss/acc, decoded quality, recon rel-L2, exact li_A_full, HVP proxy, precond loss tails, gradient scale/norm.",
            "metrics": metrics,
            "sources": [
                "a_case_run_summary_with_ablation.csv",
                "a_alpha_recon_li_a_summary.csv",
                "a_case_downstream_per_start_with_ablation.csv",
                "a_case_probe_estimator_summary_with_ablation.csv",
                "a_case_probe_gradient_summary_with_ablation.csv",
                "a_case_probe_outlier_summary_256_with_ablation.csv",
                "a_downstream_delta_with_ablation.png",
                "a_alpha_recon_li_a_by_alpha.png",
                "a_gradient_domination_with_ablation.png",
                "a_exact_li_full_outliers_with_ablation.png",
            ],
            "result": "No-cap local A lowered exact li_A p95 relative to control but worsened decoder AULC; old global variants had very large exact A tails and downstream damage. Fixed capped local variants kept downstream near control but did not improve it.",
            "conclusion": "This localizes a real training/proxy intervention but does not establish a downstream-improving mechanism.",
            "table": ("a_case_run_summary_with_ablation.csv", ["run_pretty", "decoder_aulc_mean", "decoder_delta_vs_control_mean", "decoder_delta_vs_control_max", "diag_li_A_full_p95", "train_precond_a_loss_p99", "train_precond_grad_scale_median", "quality_decoded_acc_min"]),
        },
        {
            "title": "Current-Code Alpha/Cap Frontier and Step0 Recheck",
            "question": "Can grad-ratio caps and alpha choices produce a Pareto frontier where A proxy improves without step0/downstream harm?",
            "runs": "scripts/run_variant_a_frontier.py plus scripts/run_variant_a_current_control.py; run labels sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_frontier_* and current control.",
            "control": "Current-code control finetuned from same baseline checkpoint; interventions are alpha/cap frontier specs.",
            "setup": f"{base_setup} {precond_setup} Frontier specs include alpha=0.01 caps 0.05/0.1/0.25/0.5/1.0 and no-cap alpha variants where available.",
            "measured": "Run-level exact A p95, train precond grad scale, recon, decoder AULC; per-start AULC and step0 deltas against current control.",
            "metrics": metrics + " Step0 closure curves compute loss_delta(step) relative to step0 delta to show whether post-step optimization closes the initial decoded-start gap.",
            "sources": [
                "frontier_runs_summary.csv",
                "frontier_analysis/variant_a_frontier_summary.csv",
                "frontier_analysis_current_control/variant_a_step0_mechanism_summary.csv",
                "frontier_analysis_current_control/variant_a_step0_result_deltas.csv",
                "frontier_analysis_current_control/variant_a_step0_vs_aulc_delta.png",
                "frontier_analysis_current_control/variant_a_gap_closure_curves.png",
                "frontier_analysis_current_control/variant_a_quality_frontier.png",
            ],
            "result": "Capping prevents catastrophic no-cap behavior but step0 loss remains correlated with AULC deltas; no alpha/cap point robustly beats control.",
            "conclusion": "Operational cap matters, but the mechanistic failure is not solved by cap/alpha tuning alone.",
            "table": ("frontier_analysis_current_control/variant_a_step0_mechanism_summary.csv", ["label", "n_eval_starts", "mean_delta_aulc", "median_delta_aulc", "mean_delta_step0_test_loss", "corr_step0_delta_vs_aulc_delta", "mean_delta_test_loss_step300"]),
        },
        {
            "title": "Clean Harness Matched A-Control Robustness",
            "question": "Are clean A-control deltas stable under matched starts, task centering, windows, leave-one-out, and curve completeness checks?",
            "runs": "clean_harness_analysis/* generated from clean control/A runs; notes in clean_harness_analysis_notes.md and robustness_notes.md.",
            "control": "control_clean vs a_cap1_clean with same start bank, selected decoder LR=0.01, raw path exactly matched.",
            "setup": f"{base_setup} Clean cap1 uses alpha=0.01, cap=1.0. Standard 16 eval-start downstream and full curve CSVs.",
            "measured": "AULC deltas, test curve AULC, step0 contribution, post0 contribution, sign-flip p values, bootstrap CI, task summary, leave-one-out influence.",
            "metrics": "Exact sign-flip p enumerates all sign flips for n=16; bootstrap CIs resample starts. Test curve AULC is recomputed from downstream_curves, not copied from train AULC.",
            "sources": [
                "clean_harness_analysis/clean_decoder_latent_per_start_deltas.csv",
                "clean_harness_analysis/clean_decoder_latent_curve_deltas.csv",
                "clean_harness_analysis/clean_harness_analysis_notes.md",
                "clean_harness_analysis/robustness/robustness_summary.csv",
                "clean_harness_analysis/robustness/task_summary.csv",
                "clean_harness_analysis/robustness/curve_window_summary.csv",
                "clean_harness_analysis/robustness/robustness_task_centered_step0_vs_aulc.png",
                "clean_harness_analysis/robustness/robustness_test_curve_decomposition.png",
            ],
            "result": "Mean train-AULC delta is small positive while median is near zero; step0 test-loss delta explains most mean test-curve AULC and is task-sensitive.",
            "conclusion": "Clean harness supports a weak symptom/localization to decoded step0 quality, not a decisive A-specific causal mechanism.",
            "table": ("clean_harness_analysis/robustness/robustness_summary.csv", ["metric", "n", "mean", "median", "sign_flip_p_two_sided", "bootstrap95_lo", "bootstrap95_hi", "corr_with_step0_test_loss_delta", "step0_mean_contribution_to_test_curve_auc", "post0_mean_contribution_to_test_curve_auc"]),
        },
        {
            "title": "Function-Anchor and Anchor-Sweep Experiments",
            "question": "Does directly anchoring logits/function values prevent decoded-start damage or interact beneficially with A?",
            "runs": "function_anchor_runs/function_anchor_summary_c0p0010.csv; clean_harness_analysis/anchor_sweep with c=0, 1e-4, 3e-4 paired runs.",
            "control": "For each anchor coefficient, compare matched control anchor vs A anchor; c=0 is no-anchor clean baseline.",
            "setup": f"{base_setup} Function anchor samples deterministic task batches. logit_mse/CE-style anchor penalties are applied to decoded logits or selected head blocks.",
            "measured": "Function-anchor loss/effective loss, CE delta, acc delta, reconstruction, exact A p95, train/test curve deltas, step0 deltas.",
            "metrics": "Function-anchor logit_mse is MSE(decoded_logits, raw_logits). CE gap variants penalize relu(decoded_CE - raw_CE - margin), optionally Huber/top-k. Anchor sweep deltas are A-control within coefficient.",
            "sources": [
                "function_anchor_runs/function_anchor_summary_c0p0010.csv",
                "clean_harness_analysis/anchor_sweep/anchor_sweep_pair_summary.csv",
                "clean_harness_analysis/anchor_sweep/anchor_sweep_notes.md",
                "clean_harness_analysis/anchor_sweep/anchor_sweep_pair_deltas.png",
                "clean_harness_analysis/anchor_sweep/anchor_sweep_step0_vs_aulc.png",
            ],
            "result": "Weak anchors did not produce a robust downstream win; higher anchor settings showed larger step0 damage in the anchor sweep.",
            "conclusion": "Function preservation is relevant but this anchor implementation did not supply a sufficient repair.",
            "table": ("clean_harness_analysis/anchor_sweep/anchor_sweep_pair_summary.csv", ["pair", "anchor_coeff", "n_eval_starts", "mean_delta_train_aulc", "median_delta_train_aulc", "mean_delta_test_curve_aulc", "mean_delta_step0_test_loss", "corr_step0_vs_train_aulc_delta", "delta_li_A_full_per_dim_p95_run_level"]),
        },
        {
            "title": "Extended 64-Start Downstream and A-Loss Clipping",
            "question": "If A proxy/tail is reduced by clipping, does downstream improve on a larger fixed-LR heldout-final bank?",
            "runs": "scripts/evaluate_variant_a_extended_downstream.py over control, A_unclipped, A_clip5, A_clip20; output under extended_downstream/*_64eval.",
            "control": "Same 64 heldout-final starts and same selected LRs; raw method is a zero-delta sanity control. Interventions are unclipped A, clip5, clip20.",
            "setup": f"{base_setup} {precond_setup} Extended downstream uses eval_starts=64, skip_starts=8, fixed raw_lr=3e-4 and decoder_lr=1e-2 from selected_lrs unless overridden.",
            "measured": "Per-start and aggregate test_mean_loss, test_trapz_loss, step0_test_loss, post0_test_loss_mean, train_mean_loss; cache metadata signatures and start bank.",
            "metrics": "test_mean_loss_delta = mean_step(test_loss_A) - mean_step(test_loss_control). Raw deltas must be zero because both labels evaluate identical raw starts with identical LR.",
            "sources": [
                "a_case_causal_report/a_case_proxy_downstream_intervention_summary.csv",
                "extended_downstream/c0p001_unclipped_fixedlr_64eval/paired_downstream_summary.csv",
                "extended_downstream/c0p001_clip5_fixedlr_64eval/paired_downstream_summary.csv",
                "extended_downstream/c0p001_clip20_fixedlr_64eval/paired_downstream_summary.csv",
                "extended_downstream/c0p001_unclipped_fixedlr_64eval/paired_downstream_deltas.png",
                "extended_downstream/c0p001_clip5_fixedlr_64eval/paired_downstream_deltas.png",
                "extended_downstream/c0p001_clip20_fixedlr_64eval/paired_downstream_deltas.png",
            ],
            "result": "li_A p95 improves from A_unclipped to clip20, but decoder downstream remains near control; clip5 produces a positive mean step0 loss delta despite proxy improvement.",
            "conclusion": "This rules out the simple mechanism 'reduce A proxy tail and downstream improves' for the current surrogate.",
            "table": ("a_case_causal_report/a_case_proxy_downstream_intervention_summary.csv", ["variant", "li_A_p95", "test_mean_loss_delta_mean", "test_mean_loss_delta_median", "test_mean_loss_delta_ci_low", "test_mean_loss_delta_ci_high", "test_mean_loss_delta_worse_count", "test_mean_loss_delta_better_count", "step0_test_loss_delta_mean", "post0_test_loss_mean_delta_mean"]),
        },
        {
            "title": "Causal Pair Validity and Paired Downstream Metrics",
            "question": "For final c=0.001 margin-huber variants, are paired comparisons valid and what do train/test/step0/post0 deltas show?",
            "runs": "scripts/analyze_variant_a_causal_pair.py outputs under causal_pair_analysis for marginhuber c0p001, clip5, clip20.",
            "control": "Matched control/A run pair with start_bank_audit, completion_audit, selected_lrs, curve_completeness. Raw method is a zero-delta check.",
            "setup": f"{base_setup} Block recon lam=0.03 plus head_margin_drop_huber coefficient 0.001, margin=0, huber_delta=0.05, toprec=1, topex=0.1; A cap=0.25 with optional A-loss clip.",
            "measured": "Completion, selected LR, start-bank equality, curve completeness, spike summaries, pareto summary, downstream paired deltas.",
            "metrics": "Paired delta rows include train_mean_aulc_delta, test_mean_aulc_delta, test_trapz_aulc_delta, step0_test_loss_delta, post0_test_loss_mean_delta, exact sign p, bootstrap CI, leave-one-out range, trim1 mean.",
            "sources": [
                "causal_pair_analysis/marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_completion_audit.csv",
                "causal_pair_analysis/marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_start_bank_audit.csv",
                "causal_pair_analysis/marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_downstream_summary.csv",
                "causal_pair_analysis/marginhuber_c0p001_clip5_downstream_summary.csv",
                "causal_pair_analysis/marginhuber_c0p001_clip20_downstream_summary.csv",
                "causal_pair_analysis/marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_paired_downstream_deltas.png",
                "causal_pair_analysis/marginhuber_c0p001_clip20_paired_downstream_deltas.png",
            ],
            "result": "Raw deltas are zero; decoder deltas for c0p001 variants are small and often slightly negative in 16-start pair summaries, but CIs/sign tests do not prove a robust downstream win.",
            "conclusion": "The paired harness is valid for these artifacts; evidence is near-control, not a strong positive A result.",
            "table": ("__causal_pair_combined__", [],),
        },
        {
            "title": "Downstream Path Decomposition",
            "question": "Is downstream loss gap caused by decoded-start quality or by latent-only optimizer path after the decoded start?",
            "runs": "a_case_causal_report path summaries; clean_harness_analysis/weight_paths_anchor_sweep; causal_pair_analysis/downstream_weight_paths_*.",
            "control": "Compare raw Adam from w0, raw Adam from decoded D(E(w0)), and latent Adam from the same decoded start.",
            "setup": f"{downstream_setup} Path diagnostics reuse matched eval starts and evaluate curves for raw_from_w0, decoded_raw_from_dec0, latent_adam.",
            "measured": "Decoded-start AUC penalty, latent-only penalty, total latent-minus-raw penalty, tortuosity/change-cos metrics, correlations to step0/post0 downstream deltas.",
            "metrics": "decoded_start_component = AULC(raw Adam from decoded start) - AULC(raw Adam from original w0). latent_only_component = AULC(latent Adam) - AULC(raw Adam from decoded start).",
            "sources": [
                "a_case_causal_report/path_component_absolute_summary.csv",
                "a_case_causal_report/path_component_paired_delta_summary.csv",
                "arch_invariant_causal_analysis/path_component_summary.csv",
                "arch_invariant_causal_analysis/path_component_joined_downstream.csv",
                "arch_invariant_causal_analysis/path_decomposition_vs_downstream.png",
                "clean_harness_analysis/causal_mechanism/path_decomposition.png",
                "clean_harness_analysis/causal_mechanism/path_decomposition_tail.png",
            ],
            "result": "A-control changes in decoded-start and latent-only components are tiny in the current c=0.001 artifacts; damaging anchor variants showed larger decoded-start damage.",
            "conclusion": "Path decomposition supports localization to downstream-relevant components, but current A did not materially improve those components.",
            "table": ("arch_invariant_causal_analysis/path_component_summary.csv", ["metric", "n", "mean", "median", "p90", "p95", "corr_vs_step0_test_loss_delta", "corr_vs_test_mean_aulc_delta", "corr_vs_post0_test_loss_mean_delta"]),
        },
        {
            "title": "Latent Metric and Optimizer-Geometry Probes",
            "question": "Is latent Adam failing because the decoder tangent metric/projection is mismatched to the raw gradient direction?",
            "runs": "scripts/analyze_latent_metric_optimizer_confounds.py outputs in causal_pair_analysis/latent_metric_*, mechanism_debug_outliers, and arch_invariant_causal_analysis.",
            "control": "Control and A starts are probed with the same selected raw and latent LRs; natural/projected candidates are one-step probes, not full downstream replacements.",
            "setup": "For each start, compute decoder Jacobian J, metric G=J^T J, theta-gradient g, tangent projection/natural direction solve (G+dI) dz = -J^T g, raw Adam projection, and actual latent/raw Adam steps.",
            "measured": "normal_residual_fraction, projection_norm_fraction, projection_cos_with_neg_grad, metric_condition/effective rank, actual step cosines, one-step loss deltas.",
            "metrics": "normal_residual_fraction = ||-g - J dz_nat|| / ||g||. metric_condition is eigmax(G)/eigmin(G). cosines are weight-space cosine similarities against -g or raw decoded Adam step.",
            "sources": [
                "a_case_causal_report/latent_metric_key_summary.csv",
                "arch_invariant_causal_analysis/latent_metric_selected_summary.csv",
                "causal_pair_analysis/latent_metric_marginhuber_c0p001/latent_metric_projection.csv",
                "causal_pair_analysis/latent_metric_marginhuber_c0p001/latent_metric_one_step.csv",
                "mechanism_debug_outliers/latent_metric_projection_summary.png",
                "arch_invariant_causal_analysis/path_decomposition_vs_downstream.png",
            ],
            "result": "Normal residual is about 0.9 and metric condition is very high for both A and control; natural/projected probes improve one-step directions for both rather than exposing an A-specific rescue.",
            "conclusion": "Shared chart/optimizer bottleneck remains viable, but current evidence does not prove it is uniquely caused by A.",
            "table": ("arch_invariant_causal_analysis/latent_metric_selected_summary.csv", ["variant_label", "direction_method", "scale_anchor", "starts", "best_batch_loss_delta_median", "best_test_loss_delta_median", "step_rel_w0_median", "cos_step_with_neg_grad_median", "normal_residual_fraction_median", "metric_condition_median"]),
        },
        {
            "title": "Estimator Variance and HVP Tail Diagnostics",
            "question": "Are A estimator samples heavy-tailed, and does increasing pairs or comparing control explain the downstream failure?",
            "runs": "estimator_variance/current_A_pairs4, current_A_pairs16, current_control_pairs4 plus arch_invariant_causal_analysis estimator summary.",
            "control": "Compare A pairs=4, A pairs=16, and control pairs=4 with same sampled indices where available.",
            "setup": f"{precond_setup} Offline estimator sampling recomputes precond_a_paired_loss/global_audit_loss and trace terms without full VAE rerun.",
            "measured": "precond_a_paired_loss, precond_a_global_audit_loss, precond_trace_m_per_dim distributions, gradient component norms/cosines.",
            "metrics": "paired and global audit losses use the same two-sample formulas as training/probing. Summaries report mean, median, std, quantiles, max.",
            "sources": [
                "estimator_variance/current_A_pairs4/a_estimator_samples.csv",
                "estimator_variance/current_A_pairs16/a_estimator_samples.csv",
                "estimator_variance/current_control_pairs4/a_estimator_samples.csv",
                "arch_invariant_causal_analysis/estimator_variance_summary.csv",
                "arch_invariant_causal_analysis/estimator_variance_tails.png",
            ],
            "result": "A estimator tails are real and remain heavy under pairs=16; control also has tails. Tail control alone did not produce a robust downstream win in clip experiments.",
            "conclusion": "Estimator tails are a mediator/risk, not by themselves the found causal mechanism.",
            "table": ("arch_invariant_causal_analysis/estimator_variance_summary.csv", ["estimator_run", "metric", "n", "mean", "median", "std", "p90", "p95", "p99", "max"]),
        },
        {
            "title": "Block Reconstruction Lambda and Cap Guards",
            "question": "Can directly preserving fc2.weight reconstruction remove decoded-start/head damage while retaining A proxy gains?",
            "runs": "scripts/run_variant_a_block_recon_guard.py; block_recon_runs summaries for lam0.03, lam0.06, lam0.03_cap0.25, margin/head variants; lam_sweep and lam_cap_sweep analyses.",
            "control": "Matched control with fc2 block recon vs A with same block recon plus A. Interventions vary lambda, cap, A-loss clip, head guard.",
            "setup": f"{base_setup} Block recon loss selects fc2.weight in normalized space: MSE(recon_block, target_block) with ramp over 1000 steps and coeff lambda.",
            "measured": "Block recon loss/effective loss, block/full MSE ratio, block rel-L2, grad ratios, exact A p95, decoder AULC, step0 and curve deltas.",
            "metrics": "block_recon_to_full_mse = block_mse/full_mse. block_recon_rel_l2 = mean ||recon_block-target_block||/||target_block||. Lambda sweep deltas are A-control within matched lambda.",
            "sources": [
                "block_recon_runs/block_recon_summary_lam0p03.csv",
                "block_recon_runs/block_recon_summary_lam0p03_cap0p25.csv",
                "block_recon_runs/block_recon_summary_lam0p06.csv",
                "block_recon_analysis/lam_sweep/lam_tradeoff_summary.csv",
                "block_recon_analysis/lam_cap_sweep/lam_tradeoff_summary.csv",
                "block_recon_analysis/lam_sweep/lam_tradeoff.png",
                "block_recon_analysis/lam_sweep/step0_vs_aulc_lam_sweep.png",
            ],
            "result": "Block recon improves absolute decoded quality, but A-control residuals remain small/near-control and step0 still explains damaging cases.",
            "conclusion": "fc2 reconstruction is an important guard/localization, but not a standalone fix proving A is downstream-beneficial.",
            "table": ("__block_recon_combined__", []),
        },
        {
            "title": "Head Guards, Margin Mechanism, and Row Direction",
            "question": "Is residual damage caused by classifier-head functional margins, head CE outliers, or row direction/scale errors?",
            "runs": "head_guard_runs, block_recon headce/headhuber/marginhuber summaries, margin_mechanism analyses.",
            "control": "Matched control/A pairs with same fc2 block recon and head/function guard settings; interventions include CE gap hinge/Huber, margin drop Huber, row-direction penalty.",
            "setup": "Function anchor can splice decoded head into raw body and penalize CE gap or margin drop. Row-direction guard uses per-row cosine loss 1-cos(recon_row,target_row) on fc2.weight.",
            "measured": "Head CE gap, margin p05/p01, flip rates, top CE tails, fc2 direction/scale rescue, head guard loss and grad ratios, downstream curves.",
            "metrics": "margin_drop = raw_true_class_margin - decoded_true_class_margin. margin_hinge = relu(margin_drop - margin). Huber/top-k over examples or records depending config. Direction loss is mean clamp(1-row_cos, min=0).",
            "sources": [
                "head_guard_runs/head_guard_summary_c0p003.csv",
                "block_recon_runs/block_recon_headhuber_summary_c0p003_m0p02_d0p05_top1.csv",
                "block_recon_analysis/margin_mechanism/margin_mechanism_summary.csv",
                "block_recon_analysis/margin_mechanism/head_ce_batch_gap_distribution.png",
                "block_recon_analysis/margin_mechanism/margin_tail_vs_step0_gap.png",
                "block_recon_analysis/margin_mechanism/fc2_scale_direction_rescue.png",
            ],
            "result": "Head/margin diagnostics correlate with step0 decoded damage and fc2 direction/scale interventions can rescue much of the step0 gap in selected variants.",
            "conclusion": "This is a strong localization of decoded-start function damage, but it does not prove the upstream A-loss mechanism by itself.",
            "table": ("block_recon_analysis/margin_mechanism/margin_mechanism_summary.csv", ["section", "group", "starts", "mean_step0_test_gap", "median_step0_test_gap", "corr_gap_vs_delta_margin_p05", "corr_gap_vs_bad_flip_rate", "mean_rescue_fraction", "median_rescue_fraction", "mean_gap_after_candidate"]),
        },
        {
            "title": "Raw-Block and Cross-Variant Head Splice Interventions",
            "question": "Which decoded weight blocks account for decoded-start loss gaps?",
            "runs": "scripts/analyze_decoder_block_splice_interventions.py and scripts/analyze_block_recon_head_splice.py; outputs under block_splice_fixed_nocap_full16, lam*/head_splice, clean_harness/cross_variant_head_splice.",
            "control": "For raw-decoded splice, raw w0 and decoded D(E(w0)) are compared; for cross-variant splice, control decoded and A decoded are swapped on identical starts.",
            "setup": "Block groups include individual tensors, classifier_head=(fc2.weight, fc2.bias), input_hidden_block, all_weight_tensors, all_bias_tensors, all_tensors.",
            "measured": "Candidate train/test loss/acc after swapping or rescuing blocks, rescue_removed_fraction, injection fraction, recompute-vs-logged validation.",
            "metrics": "raw rescue fraction = (decoded_loss - candidate_loss)/(decoded_loss - raw_loss). Cross-variant control-into-A rescue fraction = (A_decoded_loss - candidate_loss)/(A_decoded_loss - control_decoded_loss).",
            "sources": [
                "block_splice_fixed_nocap_full16/decoder_block_splice_summary.csv",
                "block_recon_analysis/lam0p03/head_splice/block_lam0p03_head_splice_summary.csv",
                "block_recon_analysis/lam0p03_cap0p25/head_splice/block_lam0p03_cap0p25_head_splice_summary.csv",
                "block_recon_analysis/lam0p06/head_splice/block_lam0p06_head_splice_summary.csv",
                "clean_harness_analysis/cross_variant_head_splice/cross_variant_head_splice_notes.md",
                "frontier_analysis/variant_a_block_splice_rescue.png",
                "block_recon_analysis/lam0p03/head_splice/block_lam0p03_control_into_A_rescue.png",
            ],
            "result": "Classifier-head/fc2 blocks often remove a large fraction of decoded-start gaps, especially in damaging/tail variants.",
            "conclusion": "Splice interventions localize where decoded function damage lives; they do not alone identify why A training puts damage there.",
            "table": ("__head_splice_combined__", []),
        },
        {
            "title": "Block-Recon Optimizer Residual",
            "question": "After fc2 reconstruction, is the remaining A-control gap an optimizer/chart mismatch or decoded-start artifact?",
            "runs": "scripts/analyze_block_recon_optimizer_residual.py; outputs in block_recon_analysis/lam0p03/optimizer_residual.",
            "control": "clean control/A and block_lam0.03 control/A; raw paths provide identity check; latent metric worst-start probe optionally joins optimizer evidence.",
            "setup": "Decomposes decoder A-control AULC into step0 train offset and path-after-step0; joins latent metric projections on worst starts.",
            "measured": "Selected LRs, run validity, per-start AULC deltas, curve decomposition, optimizer projection summaries and actual one-step summaries.",
            "metrics": "path_after_step0_aulc = decoder_latent_a_minus_control_aulc - step0_train_loss_delta. step0_abs_share = |step0_delta|/|AULC_delta|.",
            "sources": [
                "block_recon_analysis/lam0p03/optimizer_residual/curve_decomposition_summary.csv",
                "block_recon_analysis/lam0p03/optimizer_residual/per_start_aulc_deltas.csv",
                "block_recon_analysis/lam0p03/optimizer_residual/optimizer_projection_summary.csv",
                "block_recon_analysis/lam0p03/optimizer_residual/optimizer_joined_worst_starts.csv",
                "block_recon_analysis/lam0p03/optimizer_residual/optimizer_residual_report.md",
                "block_recon_analysis/lam0p03/optimizer_residual/block_aulc_gap_decomposition.png",
                "block_recon_analysis/lam0p03/optimizer_residual/optimizer_direction_metrics_worst_starts.png",
            ],
            "result": "Worst-start optimizer geometry is poor for both variants; A is not materially worse in latent-vs-raw alignment. Residual block gap is mostly present at step0 while post-step path closes part of it.",
            "conclusion": "For block_lam0.03 residuals, supported localization is decoded-start/head quality rather than an A-specific latent Adam bug.",
            "table": ("block_recon_analysis/lam0p03/optimizer_residual/curve_decomposition_summary.csv", ["regime", "starts", "mean_a_minus_control_aulc", "median_a_minus_control_aulc", "frac_a_worse_aulc", "mean_step0_train_offset", "mean_path_after_step0_aulc", "corr_step0_train_vs_aulc", "raw_mean_a_minus_control_aulc"]),
        },
        {
            "title": "Outlier Mechanism Debug",
            "question": "Which per-start diagnostics predict downstream outliers: reconstruction error, logit saturation, task alignment, latent metric, or A proxy?",
            "runs": "mechanism_debug_outliers outputs and top-level outlier diagnostic CSVs/PNGs.",
            "control": "Compares control, fixed cap, local no-cap, old global and related variants on outlier starts and diagnostic joins.",
            "setup": "Focuses on selected tail starts, reconstructs decoded/raw logits, measures projection/task alignment, and correlates predictors with downstream delta AULC.",
            "measured": "Reconstruction rel-L2, decoded-minus-raw loss/acc, entropy/confidence/correct probability, reconstruction error dot/cos raw gradient, predictor correlations.",
            "metrics": "Correlations are Spearman/Pearson against delta_aulc and are treated as localization evidence, not causal proof. First-order loss increase per error norm uses recon_error dot raw_grad normalized by error norm.",
            "sources": [
                "mechanism_debug_outliers/reconstruction_error_task_alignment_summary.csv",
                "mechanism_debug_outliers/downstream_predictor_correlations.csv",
                "mechanism_debug_outliers/latent_metric_summary.csv",
                "mechanism_debug_outliers/reconstruction_error_task_alignment.png",
                "mechanism_debug_outliers/decoded_logit_saturation_metrics.png",
                "mechanism_debug_outliers/downstream_delta_predictor_scatter.png",
            ],
            "result": "Damaging variants show larger decoded-minus-raw test loss and reconstruction/task alignment signals on outlier starts; correlations are high for some predictors in no-cap/global-damaged variants.",
            "conclusion": "Outlier diagnostics support decoded function damage as a mediator/localization, not a final causal mechanism without discriminating interventions.",
            "table": ("mechanism_debug_outliers/reconstruction_error_task_alignment_summary.csv", ["variant_label", "starts", "recon_rel_l2_median", "decoded_minus_raw_test_loss_median", "decoded_minus_raw_test_acc_median", "recon_error_cos_raw_grad_median", "first_order_raw_loss_increase_per_err_norm_median", "decoded_test_entropy_median"]),
        },
        {
            "title": "Arch-Invariant Causal Synthesis",
            "question": "When joined across downstream, path components, latent metric, function tails, and estimator variance, which mechanisms remain viable?",
            "runs": "scripts/analyze_variant_a_arch_invariant_causal.py; outputs in arch_invariant_causal_analysis.",
            "control": "Uses paired current c0p001 artifacts and estimator variance controls.",
            "setup": "Reads causal pair downstream summaries, path component joined downstream, latent metric selected summaries, margin mechanism summaries, and estimator variance summaries.",
            "measured": "Downstream paired deltas, path component correlations, latent metric conditions, function-tail margins, estimator tails.",
            "metrics": "Summary tables aggregate mean/median/std/quantiles and correlations against step0, test mean AULC, and post0 deltas.",
            "sources": [
                "arch_invariant_causal_analysis/downstream_summary.csv",
                "arch_invariant_causal_analysis/path_component_summary.csv",
                "arch_invariant_causal_analysis/latent_metric_selected_summary.csv",
                "arch_invariant_causal_analysis/function_tail_summary.csv",
                "arch_invariant_causal_analysis/estimator_variance_summary.csv",
                "arch_invariant_causal_analysis/path_decomposition_vs_downstream.png",
                "arch_invariant_causal_analysis/estimator_variance_tails.png",
            ],
            "result": "Current c0p001 A-control component deltas are tiny and estimator/latent bottlenecks are largely shared; no single artifact class proves a broad cause.",
            "conclusion": "Supported conclusion stays narrow: current A surrogate did not realize downstream-improving Li/PSGD-style behavior in this setup.",
            "table": ("arch_invariant_causal_analysis/downstream_summary.csv", ["method", "metric", "n", "mean", "median", "worse_count_A_gt_control", "better_count_A_lt_control", "bootstrap95_low", "bootstrap95_high"]),
        },
    ]


def add_catalog_entry(pdf: PdfPages, entry: dict[str, Any]) -> None:
    sources = [src for src in entry["sources"] if exists(src)]
    missing = [src for src in entry["sources"] if not exists(src)]
    body = [
        "## Question / Hypothesis",
        f"- {entry['question']}",
        "## Run Dirs / Labels / Scripts",
        f"- {entry['runs']}",
        "## Control vs Intervention",
        f"- {entry['control']}",
        "## Exact Setup",
        f"- {entry['setup']}",
        "## What Was Measured",
        f"- {entry['measured']}",
        "## Metric Computation / Aggregation / Sign",
        f"- {entry['metrics']}",
        "## Source CSV / PNG / MD Paths",
    ]
    body.extend([f"- {src}" for src in sources[:18]])
    if len(sources) > 18:
        body.append(f"- plus {len(sources) - 18} more paths in the inventory appendix")
    if missing:
        body.append(f"- Missing at build time: {', '.join(missing[:6])}")
    body.extend(
        [
            "## Result",
            f"- {entry['result']}",
            "## Narrow Conclusion",
            f"- {entry['conclusion']}",
        ]
    )
    add_text_pages(pdf, entry["title"], body)

    table_ref = entry.get("table")
    if not table_ref:
        return
    if table_ref[0] == "__causal_pair_combined__":
        df = causal_pair_combined_summary()
        add_table_pages(pdf, f"Key Table: {entry['title']}", df, max_rows=26)
    elif table_ref[0] == "__block_recon_combined__":
        df = block_recon_combined_summary()
        add_table_pages(pdf, f"Key Table: {entry['title']}", df, max_rows=18, max_chars=54)
    elif table_ref[0] == "__head_splice_combined__":
        df = head_splice_combined_summary()
        add_table_pages(pdf, f"Key Table: {entry['title']}", df, max_rows=22, max_chars=56)
    else:
        relative, cols = table_ref
        df = compact_table(relative, cols)
        add_table_pages(pdf, f"Key Table: {entry['title']}", df, max_rows=24, max_chars=58)


MAIN_FIGURES: list[tuple[str, str, str]] = [
    ("a_case_causal_report/a_case_proxy_downstream_intervention_summary.png", "Current c=0.001 proxy/downstream intervention summary. Shows proxy tail changes versus paired downstream deltas.", "Main figure"),
    ("a_case_causal_report/a_case_path_latent_metric_summary.png", "Current c=0.001 path and latent metric mechanism summary.", "Main figure"),
    ("a_downstream_delta_with_ablation.png", "Top-level ablation downstream delta by run family.", "Alpha/ablation"),
    ("a_alpha_recon_li_a_by_alpha.png", "Alpha/reconstruction/exact A summary across early runs.", "Alpha/ablation"),
    ("a_fixed_alpha_cap_mechanics.png", "Fixed-alpha cap mechanics showing gradient scaling and proxy behavior.", "Alpha/ablation"),
    ("a_gradient_domination_with_ablation.png", "Gradient domination/tail diagnostics for A variants.", "Alpha/ablation"),
    ("frontier_analysis_current_control/variant_a_step0_vs_aulc_delta.png", "Step0 decoded test-loss delta against downstream AULC delta in current-control frontier.", "Frontier"),
    ("frontier_analysis_current_control/variant_a_gap_closure_curves.png", "Curve gap closure after step0 for frontier variants.", "Frontier"),
    ("frontier_analysis_current_control/variant_a_quality_frontier.png", "Quality frontier: reconstruction/decoded quality vs downstream.", "Frontier"),
    ("extended_downstream/c0p001_unclipped_fixedlr_64eval/paired_downstream_deltas.png", "64-start fixed-LR downstream deltas for unclipped A.", "Extended downstream"),
    ("extended_downstream/c0p001_clip5_fixedlr_64eval/paired_downstream_deltas.png", "64-start fixed-LR downstream deltas for A clip5.", "Extended downstream"),
    ("extended_downstream/c0p001_clip20_fixedlr_64eval/paired_downstream_deltas.png", "64-start fixed-LR downstream deltas for A clip20.", "Extended downstream"),
    ("clean_harness_analysis/robustness/robustness_task_centered_step0_vs_aulc.png", "Clean harness task-centered step0-vs-AULC robustness check.", "Clean harness"),
    ("clean_harness_analysis/robustness/robustness_test_curve_decomposition.png", "Clean harness test-curve AULC decomposition into step0 and post-step components.", "Clean harness"),
    ("clean_harness_analysis/anchor_sweep/anchor_sweep_pair_deltas.png", "Function-anchor paired deltas by coefficient.", "Function anchor"),
    ("clean_harness_analysis/causal_mechanism/path_decomposition.png", "Clean/anchor path decomposition into decoded-start and latent-only components.", "Path decomposition"),
    ("clean_harness_analysis/causal_mechanism/block_splice_rescue_fractions.png", "Clean/anchor block-splice rescue fractions.", "Block splice"),
    ("arch_invariant_causal_analysis/path_decomposition_vs_downstream.png", "Path component deltas joined against downstream deltas.", "Arch synthesis"),
    ("arch_invariant_causal_analysis/estimator_variance_tails.png", "Estimator tail distributions for A/control and pairs count.", "Estimator variance"),
    ("block_recon_analysis/lam_sweep/lam_tradeoff.png", "Block reconstruction lambda tradeoff.", "Block recon"),
    ("block_recon_analysis/lam_sweep/step0_vs_aulc_lam_sweep.png", "Lambda sweep step0 vs downstream AULC.", "Block recon"),
    ("block_recon_analysis/lam0p03/head_splice/block_lam0p03_control_into_A_rescue.png", "Control classifier-head blocks spliced into A decoded starts for lam0.03.", "Head splice"),
    ("block_recon_analysis/lam0p03/head_splice/block_lam0p03_classifier_head_absolute_gap.png", "Absolute classifier-head rescue gaps for lam0.03.", "Head splice"),
    ("block_recon_analysis/lam0p03_cap0p25/head_splice/block_lam0p03_cap0p25_control_into_A_rescue.png", "Control head into A decoded starts for capped lam0.03.", "Head splice"),
    ("block_recon_analysis/lam0p06/head_splice/block_lam0p06_control_into_A_rescue.png", "Control head into A decoded starts for lam0.06.", "Head splice"),
    ("block_recon_analysis/margin_mechanism/margin_tail_vs_step0_gap.png", "Margin-tail statistics against step0 gap.", "Margin mechanism"),
    ("block_recon_analysis/margin_mechanism/fc2_scale_direction_rescue.png", "fc2 direction/scale rescue interventions.", "Margin mechanism"),
    ("block_recon_analysis/margin_mechanism/head_ce_batch_gap_distribution.png", "Head CE batch gap distribution.", "Margin mechanism"),
    ("block_recon_analysis/lam0p03/optimizer_residual/block_aulc_gap_decomposition.png", "Block lam0.03 AULC gap split into step0 and post-step residual.", "Optimizer residual"),
    ("block_recon_analysis/lam0p03/optimizer_residual/mean_curve_deltas.png", "Block/clean mean downstream curve deltas.", "Optimizer residual"),
    ("block_recon_analysis/lam0p03/optimizer_residual/optimizer_direction_metrics_worst_starts.png", "Worst-start optimizer direction metrics.", "Optimizer residual"),
    ("causal_pair_analysis/marginhuber_c0p001_m0_d0p05_toprec1_topex0p1_paired_downstream_deltas.png", "Causal pair c0p001 marginhuber paired downstream deltas.", "Causal pair"),
    ("causal_pair_analysis/marginhuber_c0p001_clip20_paired_downstream_deltas.png", "Causal pair c0p001 clip20 paired downstream deltas.", "Causal pair"),
    ("mechanism_debug_outliers/reconstruction_error_task_alignment.png", "Outlier reconstruction error and task alignment diagnostics.", "Outlier debug"),
    ("mechanism_debug_outliers/decoded_logit_saturation_metrics.png", "Decoded logit saturation metrics for outlier debug.", "Outlier debug"),
    ("mechanism_debug_outliers/downstream_delta_predictor_scatter.png", "Predictor scatter plots against downstream delta.", "Outlier debug"),
    ("mechanism_debug_outliers/latent_metric_projection_summary.png", "Latent metric projection summary for outlier starts.", "Outlier debug"),
]


def image_inventory_relatives() -> list[str]:
    important = {relative for relative, _caption, _section in MAIN_FIGURES}
    all_pngs = sorted(str(path.relative_to(BASE)) for path in BASE.rglob("*.png"))
    return [relative for relative in all_pngs if relative not in important]


def build_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory()

    with PdfPages(OUT_PDF) as pdf:
        add_text_pages(
            pdf,
            "Variant A Protocol Review Packet",
            [
                "Date: 2026-07-06. Scope: protocol-heavy review packet for Variant A causal debug artifacts under docs/reparam_preconditioning_experiments/variant_A_causal_debug.",
                "This packet is designed for a reader or larger LLM with no repository access. It includes implementation-derived setup, formulas, run labels/scripts, source artifact paths, result tables, and a full artifact inventory.",
                "Sign convention throughout: A - control. Positive loss deltas are worse. Negative loss deltas are better. Accuracy deltas use the natural sign where positive is better.",
                "Causal-status discipline: cache/config/start/LR checks are validity checks, not mechanisms. Step0 gaps, bad plots, larger gradients, or lower downstream AULC are symptoms/localizations unless a discriminating intervention excludes competing mechanisms.",
            ],
            subtitle=f"Output PDF: {rel(OUT_PDF)}",
        )

        add_text_pages(
            pdf,
            "Executive Protocol Summary",
            [
                "## Failure Definition",
                "- Current Variant A means the implemented VAE latent HVP quadratic-majorant surrogate (li_a_hvp) trained offline on CELO weight vectors and evaluated with latent Adam downstream.",
                "- The failure under review is not 'Li/Kron fails'. It is: current A changes the intended proxy and tails but does not robustly improve downstream loss curves against matched controls.",
                "## Most Important Validity Checks",
                "- Extended downstream uses exact paired 64 heldout-final starts, skip_starts=8, fixed selected LRs, and per-label cache metadata signatures for run dir, checkpoint, config, weight pool, selected LR, method set, steps, and start bank.",
                "- Causal pair analyses include completion audits, start-bank audits, selected LR tables, and curve-completeness checks.",
                "- Raw method deltas are exactly zero in paired extended/cpair summaries, validating the paired delta direction and shared start bank.",
                "## Evidence Pattern",
                "- A/HVP and clipping change proxy/tail metrics. Extended downstream does not follow the proxy monotonically.",
                "- Step0 decoded-start loss explains damaging runs much better than post-step optimization alone.",
                "- Latent metric probes show a large shared normal residual and ill-conditioned J^T J for both A and control.",
                "- Block/head splices localize decoded function damage to fc2/classifier-head blocks in damaging variants.",
                "## Narrow Supported Conclusion",
                "- The current VAE Variant A surrogate, trained offline and evaluated with latent Adam, did not realize a robust downstream-improving mechanism in the CELO setup. The evidence supports objective/representation/optimizer mismatch as a family of remaining mechanisms, not a broad negative result about Li/PSGD/Kron.",
            ],
        )

        add_table_pages(pdf, "Implementation Files Inspected", code_reference_table(), max_rows=20)

        add_text_pages(
            pdf,
            "Implementation-Derived Shared Setup",
            [
                "## Weight Task and Weight Dimension",
                "- The CELO task model is CeloMetaMLP: input image shape (1,16,16), fc1: 256 -> 64, fc2: 64 -> 10. Flat weight dimension is 64*256 + 64 + 10*64 + 10 = 17098.",
                "- Source tasks are MNIST and Fashion-MNIST in the current CELO meta config. Images are resized/grayscaled to 16x16. train_subset=16384, test_subset=4096.",
                "- Weight pool generation trains 384 source Adam runs for 2000 steps, snapshots every 50 steps. Source LR grid is (3e-4, 1e-3, 3e-3). tau is sampled log-uniformly in [0.5, 2.0], and logits use effective_flat = flat * tau.",
                "## VAE Setup",
                "- Baseline current setup uses tiny_big_vae/direct, latent_dim=512, hidden_dim=2048, vae_train_fraction=0.70, vae_steps=150000 for baseline pretraining, vae_lr=3e-4, vae_batch_size=128, beta_kl=1e-6, base loss kind mse.",
                "- Variant A and controls usually finetune from the baseline weight_pool_source_dir and vae_init_checkpoint for vae_steps=5000, vae_lr=3e-5, batch=128.",
                "## Downstream Protocol",
                "- raw method optimizes the flat weight vector directly with Adam.",
                "- decoder_latent method encodes w0 to z0, decodes D(z0), and optimizes z with Adam through the decoder. reconstruction_rel_l2 = ||D(E(w0))-w0||/||w0||.",
                "- Standard downstream uses tune_starts=8 and eval_starts=16, downstream_steps=300, eval_every=25, downstream_batch_size=128. Extended downstream fixes selected LRs and uses 64 starts.",
            ],
        )

        add_text_pages(
            pdf,
            "Metric and Formula Reference",
            [
                "## A/HVP Surrogate",
                "- For HVP A loss, random latent probes u are passed through _latent_hvp. The stopped_composite HVP includes a J^T H_x J term plus decoder-curvature term with stopped x-gradient.",
                "- Two-sample A majorant: L_A = mean((h1 dot h2)^2 - ||h1||^2 - ||h2||^2 + dim) / dim.",
                "- Global audit uses all cross-sample dot products: (mean(dot^2) - mean(norm_a) - mean(norm_b) + dim) / dim.",
                "- Exact latent diagnostic li_A_full_per_dim = mean((eig(H_z)^2 - 1)^2). li_gap = mean((abs(eig(H_z))-1)^2).",
                "## Training Caps and Guards",
                "- precond effective loss is alpha * clipped_or_unclipped_A, then optionally gradient-scaled by grad_ratio cap. grad_scale = min(1, cap * ||base_grad|| / ||precond_grad||).",
                "- Block recon: MSE over selected block, in normalized or denormalized space. block_recon_to_full_mse = block_mse/full_mse; rel_l2 is block error norm over block target norm.",
                "- Row direction: mean clamp(1 - cosine(recon row, target row), min=0) over selected 2D block rows.",
                "- Function/head anchor: logit_mse, CE gap hinge/Huber, or margin_drop Huber. margin_drop = raw true-class margin - decoded true-class margin; top-k selectors focus on largest losses.",
                "## Downstream Aggregations",
                "- downstream_results.aulc is mean train_loss over recorded rows, after finite penalty clipping.",
                "- Extended paired test_mean_loss_delta = mean(test_loss_A over curve rows) - mean(test_loss_control). test_trapz uses trapezoid integration over step divided by final-start step span.",
                "- step0_test_loss_delta is first recorded curve row. post0_test_loss_mean_delta is mean over recorded rows after step 0.",
                "- Bootstrap CIs resample starts. Exact sign p enumerates all sign flips for n<=20 in causal-pair/robustness scripts.",
            ],
        )

        add_text_pages(
            pdf,
            "Causal Mechanism Standard Applied",
            [
                "## Competing Mechanisms",
                "- H1 surrogate-faithfulness gap: latent Psi_A(M)=||M-I||_F^2 improves while exact Li full-space criterion with inverse/barrier term does not.",
                "- H2 rank/tangent bottleneck: decoder tangent space misses downstream useful raw-gradient directions; normal residual dominates.",
                "- H3 optimizer mismatch: latent Adam in z is not the optimizer implied by Li/Kron geometry.",
                "- H4 offline amortization/state-consistency gap: frozen offline VAE chart does not adapt along downstream trajectories.",
                "- H5 estimator/training instability as mediator: quartic HVP estimator tails damage decoded starts, but clipping may not repair a wrong surrogate.",
                "- H6 noise-floor/damping mismatch: clipping/batching is not the same as Li criterion's inverse/noise damping behavior.",
                "## Predictions Used",
                "- If H5 alone were root, clipping should reduce proxy tails and robustly improve downstream. It did not.",
                "- If H2/H3 dominate, normal residual/metric condition should remain poor and metric-aware updates should help both A and control. Current probes support a shared bottleneck.",
                "- If decoded-start/head damage mediates harmful variants, step0 gaps and head splice rescue should explain tail starts. Current splice/path evidence supports this localization.",
                "## Status",
                "- Excluded as sufficient: cache/start/LR mismatch; 'just reduce A proxy tail'; raw baseline mismatch; a broad Li/Kron negative claim.",
                "- Remaining viable: surrogate-faithfulness gap, shared rank/tangent bottleneck, optimizer mismatch, offline amortization mismatch, estimator instability as mediator.",
            ],
        )

        add_text_pages(
            pdf,
            "Why This Is Not a Negative Claim About Li/Kron/PSGD",
            [
                "The current result is intentionally scoped to the implemented Variant A surrogate. It should not be presented as evidence that Li's PSGD criterion, Kron, or PSGD optimizers fail.",
                "## What Li/Kron/PSGD Fits",
                "- Li PSGD fits a positive-definite preconditioner P=Q^T Q with objective c(P)=E[delta_g^T P delta_g + delta_theta^T P^{-1} delta_theta].",
                "- The inverse/barrier term is part of the anti-collapse and noise-damping mechanism.",
                "- Practical implementations update Q multiplicatively/on a Lie-group-like structure and apply Q^T(Qg) directly to downstream gradients.",
                "## What Current Variant A Tests",
                "- Current A trains an offline VAE/latent quadratic-majorant proxy. It is a latent surrogate, not a direct full-space preconditioner fit.",
                "- The decoder is rectangular from latent dimension 512 into weight dimension 17098. In weight space, J J^T is singular for a generic J with latent_dim < weight_dim.",
                "- Downstream evaluation uses latent Adam in z, not Li/Kron's apply-P-to-gradient update.",
                "## Correct Scope Statement",
                "- Supported: the implemented VAE Variant A surrogate, under this offline training and latent Adam downstream harness, did not robustly improve downstream in the current CELO setup.",
                "- Unsupported: Li/Kron preconditioner fitting is not downstream-causal.",
                "Source: agent_journals/li_psgd_kron_literature_notes.md.",
            ],
        )

        for entry in catalog_entries():
            add_catalog_entry(pdf, entry)

        add_table_pages(
            pdf,
            "Extended Downstream Combined Summary",
            downstream_combined_summary(),
            note="Combined from three 64-start fixed-LR downstream runs. Positive loss deltas mean A is worse than control.",
            max_rows=28,
        )

        add_table_pages(
            pdf,
            "Artifact Inventory By Group",
            inventory[["group", "path", "shape", "key columns", "supports"]],
            note="All CSV/PNG/MD/JSON artifacts found under variant_A_causal_debug at build time. Paths are relative to repository root.",
            max_rows=22,
            max_chars=58,
        )

        add_text_pages(
            pdf,
            "Known Gaps and Next Discriminating Experiments",
            [
                "## Required Before Broader Cause Claims",
                "- Exact-CG full-space c3 audit: evaluate exact Li-style first term and inverse/barrier term for paired A/control starts, including damping grid, rank/tangent residual, and proxy-vs-exact rank correlation.",
                "- Metric-aware/projected downstream: full curves for raw_from_w0, decoded_raw_from_dec0, latent_adam, natural_latent, projected_raw_adam, with CG/JVP/VJP matvecs rather than full Jacobian where needed.",
                "- Direct PSGD/Kron positive-control optimizer: run a fair downstream optimizer baseline applying Kron/PSGD-style P=Q^TQ updates directly on the CELO MLP, matched for starts, data, LR budget, and reporting.",
                "## Useful Diagnostics To Add",
                "- Track exact proxy and metric residual along downstream trajectories, not only at step 0.",
                "- Separate decoded-start, latent-only, and full-space inverse-term changes per start and task.",
                "- Add multi-training-seed VAE variance for current best candidates after exact-CG/optimizer positive controls clarify mechanism.",
            ],
        )

        for relative, caption, section in MAIN_FIGURES:
            add_image_page(pdf, relative, caption=caption, section=section)

        remaining_pngs = image_inventory_relatives()
        add_text_pages(
            pdf,
            "Figure Appendix Map",
            [
                f"The previous section included curated protocol figures. The remaining PNG artifacts are shown as a thumbnail appendix for visual provenance. Count: {len(remaining_pngs)}.",
                "The full file-level table earlier lists shape and support question for every PNG/CSV/MD/JSON artifact.",
            ],
        )
        add_image_grid(pdf, "PNG Artifact Thumbnail Appendix", remaining_pngs)

    print(f"wrote {OUT_PDF}")
    print(f"size_bytes={OUT_PDF.stat().st_size}")
    print(f"inventory_rows={len(inventory)}")


def main() -> None:
    build_pdf()


if __name__ == "__main__":
    main()
