from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.analyze_variant_a_margin_mechanism import (
    KEY_COLS,
    _decode_start,
    _fc2_direction_stats,
    _fc2_scale_direction_candidates,
    _fc2_weight_shape,
    _load_run,
    _logits,
    _margin_stats,
    _prediction_disagreement,
    _safe_corr,
    _safe_fraction,
    _safe_unit,
    _spec_slices,
    _split_tensors,
    _splice,
    _task_tensor_set,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
DEFAULT_OUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/rank_repair_m2048_h4096/margin_function_audit"
)
DEFAULT_CONTROL_RUN = (
    "sage_cnn_vae_smoothing_celo_meta_control_m2048_fc2recon_lam0p03_marginhuber_c0p001_rank_repair_v1_seed0"
)
DEFAULT_A_RUN = (
    "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_m2048_cap0p25_clip20_fc2recon_lam0p03_marginhuber_c0p001_rank_repair_v1_seed0"
)
DEFAULT_START_BANK = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/rank_repair_m2048_h4096/trajectory_discriminator/selected_16_start_bank.csv"
)

JOIN_KEYS = ["source_weight_index"]
BLOCK_GROUPS = {
    "fc2.weight": ("fc2.weight",),
    "fc2.bias": ("fc2.bias",),
    "classifier_head": ("fc2.weight", "fc2.bias"),
    "all_weight_tensors": (),
    "all_bias_tensors": (),
}

REQUIRED_RUN_FILES = [
    "artifact_manifest.json",
    "config.json",
    "weight_pool.pt",
    "vae_checkpoint.pt",
    "downstream_results.csv",
    "downstream_curves.csv",
    "vae_metrics.csv",
    "preconditioning_diagnostics.csv",
]


def _log(message: str) -> None:
    print(f"[margin_function_current] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_dir(run_name: str) -> Path:
    path = Path(str(run_name)).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / str(run_name)).resolve()


def _manifest_status(run_dir: Path) -> str:
    path = run_dir / "artifact_manifest.json"
    if not path.is_file():
        return "missing_manifest"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("summary", {}).get("status", ""))


def _require_complete_run(run_name: str) -> None:
    run_dir = _run_dir(run_name)
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    status = _manifest_status(run_dir)
    if missing or status != "complete":
        raise RuntimeError(
            f"run {run_name} is not audit-ready: status={status!r} missing={missing} run_dir={run_dir}"
        )


def _load_start_bank(path: Path, *, samples: int) -> pd.DataFrame:
    rows = pd.read_csv(path)
    if "source_weight_index" not in rows.columns:
        raise ValueError(f"{path} missing source_weight_index")
    duplicate_count = int(rows.duplicated(subset=["source_weight_index"]).sum())
    if duplicate_count:
        raise ValueError(
            f"{path} has {duplicate_count} duplicate source_weight_index rows; "
            "this audit joins downstream and margin rows by source_weight_index"
        )
    rows = rows.reset_index(drop=True)
    if int(samples) > 0:
        rows = rows.iloc[: int(samples)].copy()
    rows["audit_order"] = np.arange(len(rows), dtype=int)
    rows["source_weight_index"] = pd.to_numeric(rows["source_weight_index"], errors="raise").astype(int)
    return rows


def _method_curves(run: dict[str, Any], *, method: str, start_bank: pd.DataFrame) -> pd.DataFrame:
    curves = run["curves"].copy()
    curves = curves[
        (curves["split"].astype(str) == "eval")
        & (curves["method"].astype(str) == str(method))
        & (curves["source_weight_index"].astype(int).isin(set(start_bank["source_weight_index"].astype(int))))
    ].copy()
    if curves.empty:
        raise ValueError(f"no eval/{method} curves for {run['run_name']}")
    for col in ["source_weight_index", "start_index", "step"]:
        if col in curves.columns:
            curves[col] = pd.to_numeric(curves[col], errors="raise").astype(int)
    for col in ["train_loss", "test_loss", "train_acc", "test_acc", "tau", "lr"]:
        if col in curves.columns:
            curves[col] = pd.to_numeric(curves[col], errors="coerce")
    return curves


def _curve_metric(group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("step").copy()
    test_loss = group["test_loss"].to_numpy(dtype=np.float64)
    train_loss = group["train_loss"].to_numpy(dtype=np.float64)
    steps = group["step"].to_numpy(dtype=np.float64)
    post0 = group[group["step"] > 0]
    if test_loss.size == 0:
        raise ValueError("empty curve group")
    trapz = float(np.trapezoid(test_loss) / max(1, test_loss.size - 1))
    if steps.size > 1 and float(steps[-1] - steps[0]) > 0.0:
        trapz_by_step = float(np.trapezoid(test_loss, x=steps) / float(steps[-1] - steps[0]))
    else:
        trapz_by_step = float(test_loss[0])
    step0 = group[group["step"] == 0]
    if len(step0) != 1:
        raise ValueError(f"expected one step0 row, got {len(step0)}")
    return {
        "source_weight_index": int(group["source_weight_index"].iloc[0]),
        "start_index": int(group["start_index"].iloc[0]) if "start_index" in group else -1,
        "task_name": str(group["task_name"].iloc[0]) if "task_name" in group else "",
        "tau": float(group["tau"].iloc[0]) if "tau" in group else float("nan"),
        "lr": float(group["lr"].iloc[0]) if "lr" in group else float("nan"),
        "test_mean_loss": float(np.mean(test_loss)),
        "test_trapz_loss": trapz,
        "test_trapz_by_step_loss": trapz_by_step,
        "step0_test_loss": float(step0["test_loss"].iloc[0]),
        "post0_test_loss_mean": float(post0["test_loss"].mean()) if not post0.empty else float("nan"),
        "train_mean_loss": float(np.mean(train_loss)),
        "final_test_loss": float(group["test_loss"].iloc[-1]),
        "final_test_acc": float(group["test_acc"].iloc[-1]) if "test_acc" in group else float("nan"),
    }


def _paired_downstream(control: dict[str, Any], a_run: dict[str, Any], *, method: str, start_bank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, run in [("control", control), ("A", a_run)]:
        curves = _method_curves(run, method=method, start_bank=start_bank)
        metrics = pd.DataFrame([_curve_metric(group) for _, group in curves.groupby("source_weight_index", sort=False)])
        metrics["label"] = label
        rows.append(metrics)
    control_metrics, a_metrics = rows
    merged = control_metrics.merge(a_metrics, on=JOIN_KEYS, suffixes=("_control", "_A"))
    expected = set(start_bank["source_weight_index"].astype(int))
    got = set(merged["source_weight_index"].astype(int))
    if got != expected:
        raise ValueError(f"paired downstream missing starts: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    for col in [
        "test_mean_loss",
        "test_trapz_loss",
        "test_trapz_by_step_loss",
        "step0_test_loss",
        "post0_test_loss_mean",
        "train_mean_loss",
        "final_test_loss",
        "final_test_acc",
    ]:
        merged[f"{col}_delta"] = merged[f"{col}_A"] - merged[f"{col}_control"]
    return merged


def _block_delta_metrics(
    *,
    decoded_a: torch.Tensor,
    decoded_control: torch.Tensor,
    raw: torch.Tensor,
    slices: dict[str, slice],
) -> dict[str, float]:
    out: dict[str, float] = {}
    block_groups = dict(BLOCK_GROUPS)
    block_groups["all_weight_tensors"] = tuple(key for key in slices if str(key).endswith(".weight"))
    block_groups["all_bias_tensors"] = tuple(key for key in slices if str(key).endswith(".bias"))
    pairs = {
        "A_minus_control": decoded_a - decoded_control,
        "A_minus_raw": decoded_a - raw,
        "control_minus_raw": decoded_control - raw,
    }
    for pair_name, delta in pairs.items():
        total_norm2 = float(delta.pow(2).sum().detach().cpu().item())
        out[f"{pair_name}_full_l2"] = math.sqrt(max(total_norm2, 0.0))
        for group_name, raw_keys in block_groups.items():
            keys = tuple(key for key in raw_keys if key in slices)
            if not keys:
                continue
            block_norm2 = float(sum(delta[slices[key]].pow(2).sum() for key in keys).detach().cpu().item())
            param_count = int(sum(slices[key].stop - slices[key].start for key in keys))
            out[f"{pair_name}_{group_name}_l2"] = math.sqrt(max(block_norm2, 0.0))
            out[f"{pair_name}_{group_name}_norm2_share"] = _safe_fraction(block_norm2, total_norm2)
            out[f"{pair_name}_{group_name}_param_fraction"] = _safe_fraction(param_count, int(delta.numel()))
            out[f"{pair_name}_{group_name}_enrichment_param"] = _safe_fraction(
                _safe_fraction(block_norm2, total_norm2),
                _safe_fraction(param_count, int(delta.numel())),
            )
    return out


def _fc2_a_vs_control_stats(
    *,
    decoded_a: torch.Tensor,
    decoded_control: torch.Tensor,
    slices: dict[str, slice],
    spec: Any,
) -> dict[str, float]:
    fc2_slice = slices["fc2.weight"]
    shape = _fc2_weight_shape(spec)
    a_weight = decoded_a[fc2_slice].reshape(shape)
    control_weight = decoded_control[fc2_slice].reshape(shape)
    a_unit = _safe_unit(a_weight, dim=1)
    control_unit = _safe_unit(control_weight, dim=1)
    row_cos = (a_unit * control_unit).sum(dim=1)
    a_row_norm = a_weight.norm(dim=1)
    control_row_norm = control_weight.norm(dim=1)
    return {
        "A_vs_control_fc2_global_cos": float(torch.dot(_safe_unit(a_weight.reshape(-1)), _safe_unit(control_weight.reshape(-1))).detach().cpu().item()),
        "A_vs_control_fc2_row_cos_mean": float(row_cos.mean().detach().cpu().item()),
        "A_vs_control_fc2_row_cos_min": float(row_cos.min().detach().cpu().item()),
        "A_vs_control_fc2_row_direction_error_mean": float((1.0 - row_cos).mean().detach().cpu().item()),
        "A_vs_control_fc2_row_direction_error_max": float((1.0 - row_cos).max().detach().cpu().item()),
        "A_vs_control_fc2_norm_ratio": float((a_weight.norm() / control_weight.norm().clamp_min(1e-12)).detach().cpu().item()),
        "A_vs_control_fc2_row_norm_rel_l1": float(((a_row_norm - control_row_norm).abs() / control_row_norm.clamp_min(1e-12)).mean().detach().cpu().item()),
    }


def _candidate_flats(
    *,
    raw: torch.Tensor,
    decoded_control: torch.Tensor,
    decoded_a: torch.Tensor,
    slices: dict[str, slice],
    spec: Any,
) -> dict[str, torch.Tensor]:
    candidates: dict[str, torch.Tensor] = {
        "A_decoded": decoded_a.detach().clone(),
        "control_decoded": decoded_control.detach().clone(),
        "full_control_fc2_weight": _splice(decoded_a, decoded_control, slices, ("fc2.weight",)),
        "full_control_classifier_head": _splice(decoded_a, decoded_control, slices, ("fc2.weight", "fc2.bias")),
        "full_control_fc2_bias": _splice(decoded_a, decoded_control, slices, ("fc2.bias",)),
        "raw_fc2_weight": _splice(decoded_a, raw, slices, ("fc2.weight",)),
        "raw_classifier_head": _splice(decoded_a, raw, slices, ("fc2.weight", "fc2.bias")),
        "raw_fc2_bias": _splice(decoded_a, raw, slices, ("fc2.bias",)),
    }
    candidates.update(_fc2_scale_direction_candidates(a_decoded=decoded_a, control_decoded=decoded_control, slices=slices, spec=spec))
    return candidates


def _eval_candidate(
    run: dict[str, Any],
    flat: torch.Tensor,
    task_set: Any,
    *,
    tau: float,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for split in ["train", "test"]:
        images, labels = _split_tensors(task_set, split)
        with torch.no_grad():
            logits = _logits(run, flat, images, tau=tau)
        row.update(_margin_stats(logits, labels, prefix=split))
    return row


def _analyze_starts(
    *,
    control: dict[str, Any],
    a_run: dict[str, Any],
    start_bank: pd.DataFrame,
    downstream: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slices = _spec_slices(control["spec"])
    margin_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    logged = downstream[["source_weight_index", "step0_test_loss_control", "step0_test_loss_A"]].copy()
    for audit_pos, start_row in start_bank.reset_index(drop=True).iterrows():
        source_weight_index = int(start_row["source_weight_index"])
        record = control["records"].iloc[source_weight_index].to_dict()
        task_name = str(start_row.get("task_name", record.get("task_name", "tiny_cnn")))
        tau = float(start_row.get("tau", record.get("tau", 1.0)))
        task_set = _task_tensor_set(control["task_tensors"], task_name)
        raw, decoded_control = _decode_start(control, source_weight_index)
        raw_a, decoded_a = _decode_start(a_run, source_weight_index)
        raw_diff = float((raw - raw_a).abs().max().detach().cpu().item())
        _log(
            "stage=start "
            f"{audit_pos + 1}/{len(start_bank)} source={source_weight_index} task={task_name} tau={tau:.6g} raw_diff={raw_diff:.3g}"
        )
        row: dict[str, Any] = {
            "source_weight_index": source_weight_index,
            "audit_order": int(start_row.get("audit_order", audit_pos)),
            "start_bank_position": int(start_row.get("start_bank_position", -1)) if not pd.isna(start_row.get("start_bank_position", np.nan)) else -1,
            "task_name": task_name,
            "tau": tau,
            "raw_max_abs_diff_control_A": raw_diff,
            "discriminator_selection": str(start_row.get("discriminator_selection", start_row.get("selection", ""))),
            "start_role": str(start_row.get("start_role", "")),
        }
        row.update(_block_delta_metrics(decoded_a=decoded_a, decoded_control=decoded_control, raw=raw, slices=slices))
        row.update(_fc2_direction_stats(raw=raw, decoded=decoded_control, slices=slices, spec=control["spec"], prefix="control_vs_raw"))
        row.update(_fc2_direction_stats(raw=raw, decoded=decoded_a, slices=slices, spec=control["spec"], prefix="A_vs_raw"))
        row.update(_fc2_a_vs_control_stats(decoded_a=decoded_a, decoded_control=decoded_control, slices=slices, spec=control["spec"]))
        for metric in [
            "fc2_global_cos",
            "fc2_row_cos_mean",
            "fc2_row_cos_min",
            "fc2_row_direction_error_mean",
            "fc2_row_direction_error_max",
            "fc2_norm_ratio",
            "fc2_row_norm_rel_l1",
        ]:
            row[f"A_minus_control_vs_raw_{metric}"] = row[f"A_vs_raw_{metric}"] - row[f"control_vs_raw_{metric}"]
        raw_eval = _eval_candidate(control, raw, task_set, tau=tau)
        control_eval = _eval_candidate(control, decoded_control, task_set, tau=tau)
        a_eval = _eval_candidate(control, decoded_a, task_set, tau=tau)
        for key, value in raw_eval.items():
            row[f"raw_{key}"] = value
        for key, value in control_eval.items():
            row[f"control_{key}"] = value
        for key, value in a_eval.items():
            row[f"A_{key}"] = value
        for split in ["train", "test"]:
            images, labels = _split_tensors(task_set, split)
            with torch.no_grad():
                logits_control = _logits(control, decoded_control, images, tau=tau)
                logits_a = _logits(control, decoded_a, images, tau=tau)
            row.update({f"{split}_{key}": value for key, value in _prediction_disagreement(logits_control, logits_a, labels).items()})
        for metric in [
            "loss",
            "acc",
            "margin_mean",
            "margin_median",
            "margin_p10",
            "margin_p05",
            "margin_p01",
            "ce_p95",
            "ce_p99",
            "ce_top5_mean",
            "ce_top1_mean",
        ]:
            for split in ["train", "test"]:
                row[f"A_minus_control_{split}_{metric}"] = row[f"A_{split}_{metric}"] - row[f"control_{split}_{metric}"]
                row[f"A_minus_raw_{split}_{metric}"] = row[f"A_{split}_{metric}"] - row[f"raw_{split}_{metric}"]
                row[f"control_minus_raw_{split}_{metric}"] = row[f"control_{split}_{metric}"] - row[f"raw_{split}_{metric}"]
        logged_match = logged[logged["source_weight_index"].astype(int) == source_weight_index]
        if len(logged_match) == 1:
            lm = logged_match.iloc[0]
            validation_rows.append(
                {
                    "source_weight_index": source_weight_index,
                    "control_step0_test_absdiff": abs(row["control_test_loss"] - float(lm["step0_test_loss_control"])),
                    "A_step0_test_absdiff": abs(row["A_test_loss"] - float(lm["step0_test_loss_A"])),
                    "raw_max_abs_diff_control_A": raw_diff,
                }
            )
        gap_test = row["A_test_loss"] - row["control_test_loss"]
        for candidate_name, candidate in _candidate_flats(
            raw=raw,
            decoded_control=decoded_control,
            decoded_a=decoded_a,
            slices=slices,
            spec=control["spec"],
        ).items():
            eval_row = _eval_candidate(control, candidate, task_set, tau=tau)
            out: dict[str, Any] = {
                "source_weight_index": source_weight_index,
                "audit_order": int(row["audit_order"]),
                "start_bank_position": int(row["start_bank_position"]),
                "task_name": task_name,
                "tau": tau,
                "candidate": candidate_name,
                "control_test_loss": row["control_test_loss"],
                "A_test_loss": row["A_test_loss"],
                "A_minus_control_test_loss": gap_test,
                "candidate_test_loss": eval_row["test_loss"],
                "candidate_train_loss": eval_row["train_loss"],
                "candidate_test_acc": eval_row["test_acc"],
                "candidate_test_margin_p05": eval_row["test_margin_p05"],
                "candidate_test_ce_top5_mean": eval_row["test_ce_top5_mean"],
                "gap_after_candidate_vs_control": eval_row["test_loss"] - row["control_test_loss"],
                "rescue_fraction": _safe_fraction(row["A_test_loss"] - eval_row["test_loss"], gap_test),
                "discriminator_selection": row["discriminator_selection"],
            }
            intervention_rows.append(out)
        margin_rows.append(row)
    return pd.DataFrame(margin_rows), pd.DataFrame(intervention_rows), pd.DataFrame(validation_rows)


def _add_downstream_join(margin: pd.DataFrame, downstream: pd.DataFrame) -> pd.DataFrame:
    join = margin.merge(downstream, on=JOIN_KEYS, how="left", suffixes=("", "_downstream"))
    return join


def _subset_masks(join: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {
        "all": pd.Series(True, index=join.index),
        "positive_step0": pd.to_numeric(join["step0_test_loss_delta"], errors="coerce") > 0.0,
        "negative_or_zero_step0": pd.to_numeric(join["step0_test_loss_delta"], errors="coerce") <= 0.0,
    }
    if "discriminator_selection" in join.columns:
        for value in sorted(str(v) for v in join["discriminator_selection"].dropna().unique() if str(v)):
            masks[f"selection:{value}"] = join["discriminator_selection"].astype(str) == value
    return masks


def _summarize_margins(join: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, mask in _subset_masks(join).items():
        group = join[mask].copy()
        if group.empty:
            continue
        target = pd.to_numeric(group["step0_test_loss_delta"], errors="coerce")
        rows.append(
            {
                "subset": name,
                "starts": int(len(group)),
                "mean_step0_test_loss_delta": float(target.mean()),
                "median_step0_test_loss_delta": float(target.median()),
                "mean_test_mean_loss_delta": float(pd.to_numeric(group["test_mean_loss_delta"], errors="coerce").mean()),
                "median_test_mean_loss_delta": float(pd.to_numeric(group["test_mean_loss_delta"], errors="coerce").median()),
                "mean_A_minus_control_test_margin_p05": float(group["A_minus_control_test_margin_p05"].mean()),
                "mean_A_minus_control_test_ce_top5": float(group["A_minus_control_test_ce_top5_mean"].mean()),
                "mean_test_bad_flip_rate": float(group["test_control_correct_A_wrong_rate"].mean()),
                "mean_A_vs_control_fc2_row_direction_error": float(group["A_vs_control_fc2_row_direction_error_mean"].mean()),
                "mean_A_minus_control_fc2_weight_share": float(group["A_minus_control_fc2.weight_norm2_share"].mean()),
                "mean_A_minus_control_classifier_head_share": float(group["A_minus_control_classifier_head_norm2_share"].mean()),
                "corr_step0_vs_margin_p05_delta": _safe_corr(target, group["A_minus_control_test_margin_p05"]),
                "corr_step0_vs_bad_flip_rate": _safe_corr(target, group["test_control_correct_A_wrong_rate"]),
                "corr_step0_vs_ce_top5_delta": _safe_corr(target, group["A_minus_control_test_ce_top5_mean"]),
                "corr_step0_vs_fc2_row_direction_error": _safe_corr(target, group["A_vs_control_fc2_row_direction_error_mean"]),
                "corr_step0_vs_A_minus_raw_test_loss": _safe_corr(target, group["A_minus_raw_test_loss"]),
            }
        )
    return pd.DataFrame(rows)


def _summarize_interventions(interventions: pd.DataFrame, join: pd.DataFrame) -> pd.DataFrame:
    annotated = interventions.merge(
        join[["source_weight_index", "step0_test_loss_delta", "test_mean_loss_delta"]],
        on="source_weight_index",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for subset, mask in _subset_masks(annotated).items():
        group = annotated[mask].copy()
        if group.empty:
            continue
        for candidate, cg in group.groupby("candidate", sort=True):
            rows.append(
                {
                    "subset": subset,
                    "candidate": candidate,
                    "starts": int(len(cg)),
                    "mean_A_minus_control_test_loss": float(cg["A_minus_control_test_loss"].mean()),
                    "mean_gap_after_candidate": float(cg["gap_after_candidate_vs_control"].mean()),
                    "median_gap_after_candidate": float(cg["gap_after_candidate_vs_control"].median()),
                    "mean_rescue_fraction": float(cg["rescue_fraction"].replace([np.inf, -np.inf], np.nan).mean()),
                    "median_rescue_fraction": float(cg["rescue_fraction"].replace([np.inf, -np.inf], np.nan).median()),
                    "worse_after_count": int((pd.to_numeric(cg["gap_after_candidate_vs_control"], errors="coerce") > 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _correlations(join: pd.DataFrame) -> pd.DataFrame:
    targets = ["step0_test_loss_delta", "test_mean_loss_delta", "post0_test_loss_mean_delta"]
    features = [
        "A_minus_control_test_margin_p05",
        "A_minus_control_test_margin_p01",
        "A_minus_control_test_ce_top5_mean",
        "A_minus_control_test_ce_top1_mean",
        "test_control_to_A_flip_rate",
        "test_control_correct_A_wrong_rate",
        "A_vs_control_fc2_row_direction_error_mean",
        "A_vs_control_fc2_row_norm_rel_l1",
        "A_minus_control_vs_raw_fc2_row_direction_error_mean",
        "A_minus_control_vs_raw_fc2_norm_ratio",
        "A_minus_control_fc2.weight_norm2_share",
        "A_minus_control_classifier_head_norm2_share",
        "A_minus_raw_test_loss",
        "control_minus_raw_test_loss",
    ]
    rows: list[dict[str, Any]] = []
    for target in targets:
        for feature in features:
            if target not in join.columns or feature not in join.columns:
                continue
            x = pd.to_numeric(join[feature], errors="coerce")
            y = pd.to_numeric(join[target], errors="coerce")
            mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
            if int(mask.sum()) < 3:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = _safe_corr(y[mask], x[mask])
                spearman = _safe_corr(y[mask].rank(), x[mask].rank())
            rows.append({"target": target, "feature": feature, "n": int(mask.sum()), "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows)


def _write_review(
    *,
    out_dir: Path,
    validation: dict[str, Any],
    margin_summary: pd.DataFrame,
    intervention_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    focus_candidates = [
        "full_control_fc2_weight",
        "full_control_classifier_head",
        "full_control_fc2_bias",
        "raw_fc2_weight",
        "raw_classifier_head",
        "raw_fc2_bias",
        "control_row_direction_A_row_scale",
        "A_row_direction_control_row_scale",
    ]
    pos = intervention_summary[
        (intervention_summary["subset"].astype(str) == "positive_step0")
        & (intervention_summary["candidate"].astype(str).isin(focus_candidates))
    ].copy()
    all_margin = margin_summary[margin_summary["subset"].astype(str) == "all"].copy()
    top_corr = correlations[correlations["target"].astype(str) == "step0_test_loss_delta"].copy()
    top_corr = top_corr.reindex(top_corr["pearson"].abs().sort_values(ascending=False).index).head(8)
    lines = [
        "# Current Rank-Repair Margin/Function Audit Review",
        "",
        "## Validity",
        f"- Accepted: `{validation.get('accepted')}`.",
        f"- Starts: `{validation.get('starts')}`; method: `{validation.get('method')}`.",
        f"- Max recompute step0 absdiff: control `{validation.get('control_step0_test_absdiff_max'):.6g}`, A `{validation.get('A_step0_test_absdiff_max'):.6g}`.",
        f"- Max raw pool diff: `{validation.get('raw_max_abs_diff_control_A_max'):.6g}`.",
        "",
        "## Main Margin Summary",
        all_margin.to_markdown(index=False) if not all_margin.empty else "_missing_",
        "",
        "## Positive-Step0 Candidate Rescue",
        pos.to_markdown(index=False) if not pos.empty else "_missing_",
        "",
        "## Strongest Step0 Correlations",
        top_corr.to_markdown(index=False) if not top_corr.empty else "_missing_",
        "",
        "## Interpretation Boundary",
        "This audit can support or falsify the functional `fc2/head` damage mechanism for step0 harm. It does not by itself prove the training-time source of that damage; CH3 tail pressure was tested separately and is not assumed here.",
    ]
    (out_dir / "margin_function_audit_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Current m2048/h4096 margin/function-damage audit for Variant A.")
    parser.add_argument("--control-run", default=DEFAULT_CONTROL_RUN)
    parser.add_argument("--a-run", default=DEFAULT_A_RUN)
    parser.add_argument("--start-bank-csv", type=Path, default=DEFAULT_START_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--method", default="decoder_latent")
    parser.add_argument("--step0-absdiff-gate", type=float, default=2e-5)
    args = parser.parse_args()

    t0 = time.perf_counter()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "startup "
        f"device={args.device} dtype=float32 method={args.method} samples={args.samples} "
        f"control_run={args.control_run} a_run={args.a_run} start_bank={args.start_bank_csv} output_dir={out_dir}"
    )
    _log("stage=load_start_bank")
    start_bank = _load_start_bank(args.start_bank_csv, samples=int(args.samples))
    _log(f"start_bank_rows={len(start_bank)} sha256={_sha256_file(args.start_bank_csv)}")

    _log("stage=load_runs")
    _require_complete_run(str(args.control_run))
    _require_complete_run(str(args.a_run))
    control = _load_run(str(args.control_run), device=str(args.device))
    a_run = _load_run(str(args.a_run), device=str(args.device))
    if list(control["spec"].keys) != list(a_run["spec"].keys):
        raise ValueError("spec mismatch between control and A")
    weight_pool_max_diff = float((control["weights"] - a_run["weights"]).abs().max().detach().cpu().item())
    _log(f"validity weight_pool_max_abs_diff={weight_pool_max_diff:.6g}")
    if weight_pool_max_diff > 1e-6:
        raise ValueError(f"weight pool mismatch: {weight_pool_max_diff}")

    _log("stage=paired_downstream")
    downstream = _paired_downstream(control, a_run, method=str(args.method), start_bank=start_bank)
    downstream.to_csv(out_dir / "paired_downstream_deltas.csv", index=False)

    _log("stage=margin_function_interventions")
    margin, interventions, recompute_validation = _analyze_starts(
        control=control,
        a_run=a_run,
        start_bank=start_bank,
        downstream=downstream,
    )
    join = _add_downstream_join(margin, downstream)
    margin_summary = _summarize_margins(join)
    intervention_summary = _summarize_interventions(interventions, join)
    corrs = _correlations(join)

    _log("stage=write_outputs")
    margin.to_csv(out_dir / "margin_function_rows.csv", index=False)
    interventions.to_csv(out_dir / "intervention_rows.csv", index=False)
    recompute_validation.to_csv(out_dir / "recompute_validation.csv", index=False)
    join.to_csv(out_dir / "margin_downstream_join.csv", index=False)
    margin_summary.to_csv(out_dir / "margin_summary.csv", index=False)
    intervention_summary.to_csv(out_dir / "intervention_summary.csv", index=False)
    corrs.to_csv(out_dir / "margin_downstream_correlations.csv", index=False)

    numeric_frames = [margin, interventions, recompute_validation, join, margin_summary, intervention_summary, corrs]
    optional_nan_substrings = ("rescue_fraction", "corr", "pearson", "spearman")
    all_required_finite = True
    optional_nan_count = 0
    for frame in numeric_frames:
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            continue
        optional_cols = [col for col in numeric.columns if any(token in str(col) for token in optional_nan_substrings)]
        required = numeric.drop(columns=optional_cols, errors="ignore")
        optional = numeric[optional_cols] if optional_cols else pd.DataFrame(index=numeric.index)
        if not required.empty and not np.isfinite(required.to_numpy(dtype=np.float64)).all():
            all_required_finite = False
            break
        if not optional.empty:
            optional_nan_count += int(np.isnan(optional.to_numpy(dtype=np.float64)).sum())
    validation = {
        "accepted": False,
        "control_run": str(args.control_run),
        "a_run": str(args.a_run),
        "method": str(args.method),
        "output_dir": str(out_dir),
        "start_bank_csv": str(args.start_bank_csv),
        "start_bank_sha256": _sha256_file(args.start_bank_csv),
        "starts": int(len(start_bank)),
        "weight_pool_max_abs_diff": weight_pool_max_diff,
        "control_step0_test_absdiff_max": float(recompute_validation["control_step0_test_absdiff"].max()) if not recompute_validation.empty else float("nan"),
        "A_step0_test_absdiff_max": float(recompute_validation["A_step0_test_absdiff"].max()) if not recompute_validation.empty else float("nan"),
        "raw_max_abs_diff_control_A_max": float(recompute_validation["raw_max_abs_diff_control_A"].max()) if not recompute_validation.empty else float("nan"),
        "all_required_numeric_finite": bool(all_required_finite),
        "optional_nan_count": int(optional_nan_count),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    validation["accepted"] = bool(
        validation["starts"] == int(len(recompute_validation))
        and validation["weight_pool_max_abs_diff"] <= 1e-6
        and validation["control_step0_test_absdiff_max"] <= float(args.step0_absdiff_gate)
        and validation["A_step0_test_absdiff_max"] <= float(args.step0_absdiff_gate)
        and validation["raw_max_abs_diff_control_A_max"] <= 1e-6
        and validation["all_required_numeric_finite"]
    )
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "script": "scripts/audit_variant_a_margin_function_current.py",
        "validation": validation,
        "artifacts": [
            "paired_downstream_deltas.csv",
            "margin_function_rows.csv",
            "intervention_rows.csv",
            "recompute_validation.csv",
            "margin_downstream_join.csv",
            "margin_summary.csv",
            "intervention_summary.csv",
            "margin_downstream_correlations.csv",
            "validation.json",
            "margin_function_audit_review.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_review(out_dir=out_dir, validation=validation, margin_summary=margin_summary, intervention_summary=intervention_summary, correlations=corrs)

    _log(
        "done "
        f"accepted={validation['accepted']} elapsed_sec={validation['elapsed_sec']:.1f} "
        f"review={out_dir / 'margin_function_audit_review.md'}"
    )
    print(margin_summary.to_string(index=False), flush=True)
    print(intervention_summary[intervention_summary["subset"].astype(str).eq("positive_step0")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
