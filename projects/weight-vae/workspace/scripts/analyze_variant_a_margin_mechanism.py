from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import functional_call

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    config_hash,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    _model_from_spec,
    build_weight_vae,
    decode_weights,
    encode_weights,
    flat_to_state_dict,
    load_celo_meta_task_tensors,
    load_torch_cache,
    move_task_tensors,
    spec_from_payload,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
OUT_DIR = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_analysis/margin_mechanism"
)

BASE_CONTROL_RUN = "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0"
BASE_A_RUN = "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_clean_harness_v1_seed0"
HEADCE_CONTROL_RUN = "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_headce_c0p0003_m0_clean_harness_v1_seed0"
HEADCE_A_RUN = "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_headce_c0p0003_m0_clean_harness_v1_seed0"

KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]
MARGINS = (0.0, 0.01, 0.02, 0.05)
HUBER_DELTAS = (0.02, 0.05)


def _log(message: str) -> None:
    print(f"[variant_a_margin] {message}", flush=True)


def _load_cfg(output_dir: Path, *, device: str) -> ExperimentConfig:
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{output_dir / 'config.json'} does not contain a config mapping")
    values = dict(raw_cfg)
    values["device"] = str(device)
    values["dtype"] = "float32"
    return ExperimentConfig(**values)


def _load_run(run_name: str, *, device: str) -> dict[str, Any]:
    output_dir = ARTIFACT_ROOT / run_name
    cfg = _load_cfg(output_dir, device=device)
    dev = torch.device(cfg.device)
    _log(
        "load_run "
        f"run={run_name} hash={config_hash(cfg)} device={cfg.device} dtype={cfg.dtype} "
        f"seed={cfg.seed} output_dir={output_dir}"
    )
    for filename in [
        "weight_pool.pt",
        "vae_checkpoint.pt",
        "downstream_results.csv",
        "downstream_curves.csv",
        "vae_metrics.csv",
    ]:
        _log(f"cache_hit run={run_name} file={output_dir / filename}")
    weight_payload = load_torch_cache(output_dir / "weight_pool.pt")
    vae_payload = load_torch_cache(output_dir / "vae_checkpoint.pt")
    if weight_payload is None or vae_payload is None:
        raise FileNotFoundError(f"{output_dir} missing weight_pool.pt or vae_checkpoint.pt")
    weights = weight_payload["weights"].to(device=dev, dtype=torch.float32)
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=dev, dtype=torch.float32).eval()
    vae.load_state_dict(vae_payload["model_state"])
    task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=dev, dtype=torch.float32)
    model = _model_from_spec(spec).to(device=dev, dtype=torch.float32).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return {
        "cfg": cfg,
        "device": dev,
        "run_name": run_name,
        "output_dir": output_dir,
        "weights": weights,
        "records": pd.DataFrame(weight_payload["records"]),
        "spec": spec,
        "normalizer": normalizer,
        "vae": vae,
        "task_tensors": task_tensors,
        "model": model,
        "results": pd.read_csv(output_dir / "downstream_results.csv"),
        "curves": pd.read_csv(output_dir / "downstream_curves.csv"),
        "vae_metrics": pd.read_csv(output_dir / "vae_metrics.csv"),
        "geometry": pd.read_csv(output_dir / "preconditioning_diagnostics.csv"),
    }


def _spec_slices(spec) -> dict[str, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    for key, size in zip(spec.keys, spec.sizes, strict=True):
        slices[str(key)] = slice(offset, offset + int(size))
        offset += int(size)
    return slices


def _task_tensor_set(task_tensors: dict[str, Any], task_name: str):
    if task_name in task_tensors:
        return task_tensors[task_name]
    if len(task_tensors) == 1:
        return next(iter(task_tensors.values()))
    raise KeyError(f"task {task_name!r} not found")


def _logits(run: dict[str, Any], flat: torch.Tensor, images: torch.Tensor, *, tau: float) -> torch.Tensor:
    effective_flat = flat.to(device=images.device, dtype=images.dtype) * float(tau)
    state = flat_to_state_dict(effective_flat, run["spec"])
    return functional_call(run["model"], state, (images,))


def _split_tensors(task_set, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        return task_set.train_images, task_set.train_labels
    if split == "test":
        return task_set.test_images, task_set.test_labels
    raise ValueError(split)


def _margin_stats(logits: torch.Tensor, labels: torch.Tensor, *, prefix: str) -> dict[str, float]:
    ce = F.cross_entropy(logits, labels, reduction="none")
    pred = logits.argmax(dim=-1)
    label_logits = logits.gather(1, labels.reshape(-1, 1)).squeeze(1)
    masked = logits.detach().clone()
    masked.scatter_(1, labels.reshape(-1, 1), -float("inf"))
    other = masked.max(dim=1).values
    margin = label_logits - other
    return {
        f"{prefix}_loss": float(ce.mean().detach().cpu().item()),
        f"{prefix}_acc": float((pred == labels).float().mean().detach().cpu().item()),
        f"{prefix}_margin_mean": float(margin.mean().detach().cpu().item()),
        f"{prefix}_margin_median": float(margin.quantile(0.50).detach().cpu().item()),
        f"{prefix}_margin_p10": float(margin.quantile(0.10).detach().cpu().item()),
        f"{prefix}_margin_p05": float(margin.quantile(0.05).detach().cpu().item()),
        f"{prefix}_margin_p01": float(margin.quantile(0.01).detach().cpu().item()),
        f"{prefix}_ce_p95": float(ce.quantile(0.95).detach().cpu().item()),
        f"{prefix}_ce_p99": float(ce.quantile(0.99).detach().cpu().item()),
        f"{prefix}_ce_top5_mean": float(torch.topk(ce, k=max(1, int(math.ceil(0.05 * ce.numel())))).values.mean().detach().cpu().item()),
        f"{prefix}_ce_top1_mean": float(torch.topk(ce, k=max(1, int(math.ceil(0.01 * ce.numel())))).values.mean().detach().cpu().item()),
    }


def _prediction_disagreement(
    control_logits: torch.Tensor,
    a_logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    pred_c = control_logits.argmax(dim=-1)
    pred_a = a_logits.argmax(dim=-1)
    return {
        "control_to_A_flip_rate": float((pred_c != pred_a).float().mean().detach().cpu().item()),
        "control_correct_A_wrong_rate": float(((pred_c == labels) & (pred_a != labels)).float().mean().detach().cpu().item()),
        "control_wrong_A_correct_rate": float(((pred_c != labels) & (pred_a == labels)).float().mean().detach().cpu().item()),
    }


def _eval_starts(run: dict[str, Any], *, samples: int) -> pd.DataFrame:
    starts = run["results"][
        (run["results"]["split"].astype(str) == "eval")
        & (run["results"]["method"].astype(str) == "decoder_latent")
    ].copy()
    starts = starts.sort_values("start_index").reset_index(drop=True)
    if int(samples) > 0:
        starts = starts.iloc[: int(samples)].copy()
    return starts


def _assert_common_start_bank(control_starts: pd.DataFrame, a_starts: pd.DataFrame) -> None:
    left = control_starts[KEY_COLS].reset_index(drop=True)
    right = a_starts[KEY_COLS].reset_index(drop=True)
    if not left.equals(right):
        merged = pd.concat({"control": left, "A": right}, axis=1)
        raise ValueError(f"start bank mismatch\n{merged}")


def _decode_start(run: dict[str, Any], source_weight_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = run["weights"][int(source_weight_index)].detach()
    with torch.no_grad():
        z = encode_weights(run["vae"], run["normalizer"], raw.reshape(1, -1)).squeeze(0)
        decoded = decode_weights(run["vae"], run["normalizer"], z.reshape(1, -1)).squeeze(0).detach()
    return raw, decoded


def _splice(base: torch.Tensor, donor: torch.Tensor, slices: dict[str, slice], keys: tuple[str, ...]) -> torch.Tensor:
    result = base.detach().clone()
    for key in keys:
        result[slices[key]] = donor.detach()[slices[key]]
    return result


def _fc2_weight_shape(spec) -> tuple[int, ...]:
    return tuple(spec.shapes[list(spec.keys).index("fc2.weight")])


def _safe_unit(value: torch.Tensor, *, dim: int | None = None) -> torch.Tensor:
    if dim is None:
        return value / value.norm().clamp_min(1e-12)
    return value / value.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def _fc2_direction_stats(
    *,
    raw: torch.Tensor,
    decoded: torch.Tensor,
    slices: dict[str, slice],
    spec,
    prefix: str,
) -> dict[str, float]:
    fc2_slice = slices["fc2.weight"]
    shape = _fc2_weight_shape(spec)
    raw_weight = raw[fc2_slice].reshape(shape)
    decoded_weight = decoded[fc2_slice].reshape(shape)
    raw_unit = _safe_unit(raw_weight, dim=1)
    decoded_unit = _safe_unit(decoded_weight, dim=1)
    row_cos = (raw_unit * decoded_unit).sum(dim=1)
    return {
        f"{prefix}_fc2_global_cos": float(torch.dot(_safe_unit(raw_weight.reshape(-1)), _safe_unit(decoded_weight.reshape(-1))).detach().cpu().item()),
        f"{prefix}_fc2_row_cos_mean": float(row_cos.mean().detach().cpu().item()),
        f"{prefix}_fc2_row_cos_min": float(row_cos.min().detach().cpu().item()),
        f"{prefix}_fc2_row_direction_error_mean": float((1.0 - row_cos).mean().detach().cpu().item()),
        f"{prefix}_fc2_row_direction_error_max": float((1.0 - row_cos).max().detach().cpu().item()),
        f"{prefix}_fc2_norm_ratio": float((decoded_weight.norm() / raw_weight.norm().clamp_min(1e-12)).detach().cpu().item()),
        f"{prefix}_fc2_row_norm_rel_l1": float(
            ((decoded_weight.norm(dim=1) - raw_weight.norm(dim=1)).abs() / raw_weight.norm(dim=1).clamp_min(1e-12)).mean().detach().cpu().item()
        ),
    }


def _fc2_scale_direction_candidates(
    *,
    a_decoded: torch.Tensor,
    control_decoded: torch.Tensor,
    slices: dict[str, slice],
    spec,
) -> dict[str, torch.Tensor]:
    fc2_slice = slices["fc2.weight"]
    shape = _fc2_weight_shape(spec)
    a_weight = a_decoded[fc2_slice].reshape(shape)
    c_weight = control_decoded[fc2_slice].reshape(shape)
    candidates = {"full_control_fc2": c_weight}
    candidates["control_global_direction_A_scale"] = _safe_unit(c_weight) * a_weight.norm().clamp_min(1e-12)
    candidates["A_global_direction_control_scale"] = _safe_unit(a_weight) * c_weight.norm().clamp_min(1e-12)
    candidates["control_row_direction_A_row_scale"] = _safe_unit(c_weight, dim=1) * a_weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    candidates["A_row_direction_control_row_scale"] = _safe_unit(a_weight, dim=1) * c_weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    result: dict[str, torch.Tensor] = {}
    for name, weight in candidates.items():
        flat = a_decoded.detach().clone()
        flat[fc2_slice] = weight.reshape(-1)
        result[name] = flat
    return result


def _safe_fraction(num: float, den: float) -> float:
    if abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _analyze_margin_and_splice(
    *,
    control: dict[str, Any],
    a_run: dict[str, Any],
    samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    control_starts = _eval_starts(control, samples=samples)
    a_starts = _eval_starts(a_run, samples=samples)
    _assert_common_start_bank(control_starts, a_starts)
    slices = _spec_slices(control["spec"])
    margin_rows: list[dict[str, Any]] = []
    splice_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    logged_c = control["curves"][
        (control["curves"]["split"].astype(str) == "eval")
        & (control["curves"]["method"].astype(str) == "decoder_latent")
        & (pd.to_numeric(control["curves"]["step"], errors="coerce") == 0)
    ].copy()
    logged_a = a_run["curves"][
        (a_run["curves"]["split"].astype(str) == "eval")
        & (a_run["curves"]["method"].astype(str) == "decoder_latent")
        & (pd.to_numeric(a_run["curves"]["step"], errors="coerce") == 0)
    ].copy()
    logged = logged_c[KEY_COLS + ["train_loss", "test_loss"]].merge(
        logged_a[KEY_COLS + ["train_loss", "test_loss"]],
        on=KEY_COLS,
        suffixes=("_control_logged", "_A_logged"),
    )
    for idx, start_row in control_starts.iterrows():
        source_weight_index = int(start_row["source_weight_index"])
        record = control["records"].iloc[source_weight_index].to_dict()
        task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
        tau = float(record.get("tau", start_row.get("tau", 1.0)))
        task_set = _task_tensor_set(control["task_tensors"], task_name)
        raw, dec_c = _decode_start(control, source_weight_index)
        _raw_a, dec_a = _decode_start(a_run, source_weight_index)
        _log(f"stage=margin_splice start={idx + 1}/{len(control_starts)} source={source_weight_index} task={task_name} tau={tau:.6g}")
        row: dict[str, Any] = {
            "source_weight_index": source_weight_index,
            "start_index": int(start_row["start_index"]),
            "task_name": task_name,
            "tau": tau,
        }
        row.update(_fc2_direction_stats(raw=raw, decoded=dec_c, slices=slices, spec=control["spec"], prefix="control"))
        row.update(_fc2_direction_stats(raw=raw, decoded=dec_a, slices=slices, spec=control["spec"], prefix="A"))
        for metric in [
            "fc2_global_cos",
            "fc2_row_cos_mean",
            "fc2_row_cos_min",
            "fc2_row_direction_error_mean",
            "fc2_row_direction_error_max",
            "fc2_norm_ratio",
            "fc2_row_norm_rel_l1",
        ]:
            row[f"A_minus_control_{metric}"] = row[f"A_{metric}"] - row[f"control_{metric}"]
        for split in ["train", "test"]:
            images, labels = _split_tensors(task_set, split)
            with torch.no_grad():
                logits_c = _logits(control, dec_c, images, tau=tau)
                logits_a = _logits(control, dec_a, images, tau=tau)
            row.update(_margin_stats(logits_c, labels, prefix=f"control_{split}"))
            row.update(_margin_stats(logits_a, labels, prefix=f"A_{split}"))
            row.update({f"{split}_{key}": value for key, value in _prediction_disagreement(logits_c, logits_a, labels).items()})
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
        margin_rows.append(row)

        match = logged[
            (logged["source_weight_index"].astype(int) == source_weight_index)
            & (logged["start_index"].astype(int) == int(start_row["start_index"]))
        ]
        if len(match) == 1:
            m = match.iloc[0]
            validation_rows.append(
                {
                    "source_weight_index": source_weight_index,
                    "start_index": int(start_row["start_index"]),
                    "control_train_absdiff": abs(row["control_train_loss"] - float(m["train_loss_control_logged"])),
                    "control_test_absdiff": abs(row["control_test_loss"] - float(m["test_loss_control_logged"])),
                    "A_train_absdiff": abs(row["A_train_loss"] - float(m["train_loss_A_logged"])),
                    "A_test_absdiff": abs(row["A_test_loss"] - float(m["test_loss_A_logged"])),
                }
            )

        gap_test = row["A_test_loss"] - row["control_test_loss"]
        for candidate_name, candidate in _fc2_scale_direction_candidates(
            a_decoded=dec_a,
            control_decoded=dec_c,
            slices=slices,
            spec=control["spec"],
        ).items():
            with torch.no_grad():
                logits = _logits(control, candidate, task_set.test_images, tau=tau)
                candidate_loss = float(F.cross_entropy(logits, task_set.test_labels).detach().cpu().item())
                candidate_acc = float((logits.argmax(dim=-1) == task_set.test_labels).float().mean().detach().cpu().item())
            splice_rows.append(
                {
                    "source_weight_index": source_weight_index,
                    "start_index": int(start_row["start_index"]),
                    "task_name": task_name,
                    "tau": tau,
                    "candidate": candidate_name,
                    "control_test_loss": row["control_test_loss"],
                    "A_test_loss": row["A_test_loss"],
                    "candidate_test_loss": candidate_loss,
                    "candidate_test_acc": candidate_acc,
                    "A_minus_control_test_loss": gap_test,
                    "gap_after_candidate_vs_control": candidate_loss - row["control_test_loss"],
                    "rescue_fraction": _safe_fraction(row["A_test_loss"] - candidate_loss, gap_test),
                }
            )
    margin_df = pd.DataFrame(margin_rows)
    splice_df = pd.DataFrame(splice_rows)
    validation_df = pd.DataFrame(validation_rows)
    return margin_df, splice_df, validation_df


def _deterministic_indices(*, total: int, count: int, seed: int, start_index: int, batch_id: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    mixed = int(seed) * 1_000_003 + int(start_index) * 1009 + int(batch_id) * 9176 + int(total)
    generator.manual_seed(mixed % (2**63 - 1))
    return torch.randperm(total, generator=generator, device="cpu")[: min(count, total)].to(device=device, dtype=torch.long)


def _head_ce_gap_rows(
    *,
    control: dict[str, Any],
    a_run: dict[str, Any],
    samples: int,
    batch_samples: int,
    batch_size: int,
) -> pd.DataFrame:
    starts = _eval_starts(control, samples=samples)
    _assert_common_start_bank(starts, _eval_starts(a_run, samples=samples))
    slices = _spec_slices(control["spec"])
    rows: list[dict[str, Any]] = []
    for run_kind, run in [("control", control), ("A", a_run)]:
        for idx, start_row in starts.iterrows():
            source_weight_index = int(start_row["source_weight_index"])
            record = run["records"].iloc[source_weight_index].to_dict()
            task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
            tau = float(record.get("tau", start_row.get("tau", 1.0)))
            task_set = _task_tensor_set(run["task_tensors"], task_name)
            images_all, labels_all = _split_tensors(task_set, "train")
            raw, decoded = _decode_start(run, source_weight_index)
            head_decoded = _splice(raw, decoded, slices, ("fc2.weight",))
            _log(
                "stage=head_ce_batches "
                f"variant={run_kind} start={idx + 1}/{len(starts)} source={source_weight_index} "
                f"batches={batch_samples} batch_size={batch_size}"
            )
            for batch_id in range(int(batch_samples)):
                batch_indices = _deterministic_indices(
                    total=int(images_all.shape[0]),
                    count=int(batch_size),
                    seed=int(run["cfg"].seed),
                    start_index=int(start_row["start_index"]),
                    batch_id=batch_id,
                    device=images_all.device,
                )
                images = images_all.index_select(0, batch_indices)
                labels = labels_all.index_select(0, batch_indices)
                with torch.no_grad():
                    raw_ce = F.cross_entropy(_logits(run, raw, images, tau=tau), labels)
                flat = head_decoded.detach().clone().requires_grad_(True)
                decoded_ce = F.cross_entropy(_logits(run, flat, images, tau=tau), labels)
                grad = torch.autograd.grad(decoded_ce, flat, retain_graph=False, create_graph=False)[0]
                fc2_grad_norm = float(grad[slices["fc2.weight"]].detach().norm().cpu().item())
                gap = float((decoded_ce.detach() - raw_ce.detach()).cpu().item())
                rows.append(
                    {
                        "variant": run_kind,
                        "source_weight_index": source_weight_index,
                        "start_index": int(start_row["start_index"]),
                        "task_name": task_name,
                        "tau": tau,
                        "batch_id": batch_id,
                        "raw_ce": float(raw_ce.detach().cpu().item()),
                        "decoded_head_ce": float(decoded_ce.detach().cpu().item()),
                        "ce_gap": gap,
                        "decoded_head_fc2_grad_norm": fc2_grad_norm,
                    }
                )
    return pd.DataFrame(rows)


def _summarize_head_ce_gaps(rows: pd.DataFrame, *, coeff: float = 3e-4) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for variant, group in rows.groupby("variant"):
        gap = pd.to_numeric(group["ce_gap"], errors="coerce").to_numpy(dtype=float)
        grad = pd.to_numeric(group["decoded_head_fc2_grad_norm"], errors="coerce").to_numpy(dtype=float)
        for margin in MARGINS:
            hinge = np.maximum(gap - float(margin), 0.0)
            square = hinge**2
            summaries.append(
                {
                    "variant": variant,
                    "loss_variant": "square",
                    "margin": float(margin),
                    "huber_delta": np.nan,
                    "active_fraction": float(np.mean(hinge > 0.0)),
                    "ce_gap_mean": float(np.mean(gap)),
                    "ce_gap_p95": float(np.quantile(gap, 0.95)),
                    "ce_gap_p99": float(np.quantile(gap, 0.99)),
                    "ce_gap_max": float(np.max(gap)),
                    "anchor_loss_mean": float(np.mean(square)),
                    "anchor_loss_p95": float(np.quantile(square, 0.95)),
                    "anchor_loss_p99": float(np.quantile(square, 0.99)),
                    "anchor_loss_max": float(np.max(square)),
                    "coeff": float(coeff),
                    "effective_loss_p99": float(coeff * np.quantile(square, 0.99)),
                    "grad_proxy_p95": float(np.quantile(2.0 * hinge * grad, 0.95)),
                    "grad_proxy_p99": float(np.quantile(2.0 * hinge * grad, 0.99)),
                    "grad_proxy_max": float(np.max(2.0 * hinge * grad)),
                }
            )
            for delta in HUBER_DELTAS:
                huber = np.where(hinge <= delta, 0.5 * hinge**2, float(delta) * (hinge - 0.5 * float(delta)))
                grad_proxy = np.minimum(hinge, float(delta)) * grad
                summaries.append(
                    {
                        "variant": variant,
                        "loss_variant": "huber",
                        "margin": float(margin),
                        "huber_delta": float(delta),
                        "active_fraction": float(np.mean(hinge > 0.0)),
                        "ce_gap_mean": float(np.mean(gap)),
                        "ce_gap_p95": float(np.quantile(gap, 0.95)),
                        "ce_gap_p99": float(np.quantile(gap, 0.99)),
                        "ce_gap_max": float(np.max(gap)),
                        "anchor_loss_mean": float(np.mean(huber)),
                        "anchor_loss_p95": float(np.quantile(huber, 0.95)),
                        "anchor_loss_p99": float(np.quantile(huber, 0.99)),
                        "anchor_loss_max": float(np.max(huber)),
                        "coeff": float(coeff),
                        "effective_loss_p99": float(coeff * np.quantile(huber, 0.99)),
                        "grad_proxy_p95": float(np.quantile(grad_proxy, 0.95)),
                        "grad_proxy_p99": float(np.quantile(grad_proxy, 0.99)),
                        "grad_proxy_max": float(np.max(grad_proxy)),
                    }
                )
    return pd.DataFrame(summaries)


def _headce_run_summary(control_headce: dict[str, Any], a_headce: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    for variant, run in [("control", control_headce), ("A", a_headce)]:
        rows = run["vae_metrics"].copy()
        record_type = rows.get("record_type", pd.Series("", index=rows.index)).fillna("").astype(str)
        rows = rows[record_type != "vae_quality"].copy()
        rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
        rows = rows[rows["step"].notna()].copy()
        for col in [
            "train_function_anchor_effective_loss",
            "train_function_anchor_ce_delta",
            "train_function_anchor_acc_delta",
            "train_block_recon_effective_loss",
            "train_block_recon_grad_ratio",
            "train_precond_effective_loss",
            "train_precond_base_grad_norm",
            "train_precond_grad_norm",
            "train_precond_grad_scale",
            "val_loss",
            "val_recon_mse",
        ]:
            if col in rows:
                rows[col] = pd.to_numeric(rows[col], errors="coerce")
        train_rows.append(rows.assign(variant=variant))
    train = pd.concat(train_rows, ignore_index=True)
    result_rows = []
    for run_kind, run in [("control", control_headce), ("A", a_headce)]:
        dec = run["results"][
            (run["results"]["split"].astype(str) == "eval")
            & (run["results"]["method"].astype(str) == "decoder_latent")
        ].copy()
        dec["aulc"] = pd.to_numeric(dec["aulc"], errors="coerce")
        result_rows.append(
            {
                "variant": run_kind,
                "decoder_aulc_mean": float(dec["aulc"].mean()),
                "decoder_aulc_median": float(dec["aulc"].median()),
                "decoder_starts": int(len(dec)),
                "anchor_eff_p95": float(train[train["variant"] == run_kind]["train_function_anchor_effective_loss"].quantile(0.95)),
                "anchor_eff_p99": float(train[train["variant"] == run_kind]["train_function_anchor_effective_loss"].quantile(0.99)),
                "anchor_eff_max": float(train[train["variant"] == run_kind]["train_function_anchor_effective_loss"].max()),
                "ce_delta_p95": float(train[train["variant"] == run_kind]["train_function_anchor_ce_delta"].quantile(0.95)),
                "ce_delta_p99": float(train[train["variant"] == run_kind]["train_function_anchor_ce_delta"].quantile(0.99)),
                "ce_delta_max": float(train[train["variant"] == run_kind]["train_function_anchor_ce_delta"].max()),
                "precond_eff_p50": float(train[train["variant"] == run_kind]["train_precond_effective_loss"].quantile(0.50)),
                "precond_eff_p95": float(train[train["variant"] == run_kind]["train_precond_effective_loss"].quantile(0.95)),
                "precond_grad_scale_p50": float(train[train["variant"] == run_kind]["train_precond_grad_scale"].quantile(0.50)),
                "precond_grad_scale_p95": float(train[train["variant"] == run_kind]["train_precond_grad_scale"].quantile(0.95)),
                "precond_base_grad_norm_p50": float(train[train["variant"] == run_kind]["train_precond_base_grad_norm"].quantile(0.50)),
                "precond_grad_norm_p50": float(train[train["variant"] == run_kind]["train_precond_grad_norm"].quantile(0.50)),
                "block_eff_p50": float(train[train["variant"] == run_kind]["train_block_recon_effective_loss"].quantile(0.50)),
                "block_eff_p95": float(train[train["variant"] == run_kind]["train_block_recon_effective_loss"].quantile(0.95)),
                "block_grad_ratio_p50": float(train[train["variant"] == run_kind]["train_block_recon_grad_ratio"].quantile(0.50)),
                "block_grad_ratio_p95": float(train[train["variant"] == run_kind]["train_block_recon_grad_ratio"].quantile(0.95)),
                "val_loss_max": float(train[train["variant"] == run_kind]["val_loss"].max()),
            }
        )
    return train, pd.DataFrame(result_rows)


def _headce_pair_delta(control_headce: dict[str, Any], a_headce: dict[str, Any]) -> pd.DataFrame:
    control = control_headce["curves"][
        (control_headce["curves"]["split"].astype(str) == "eval")
        & (control_headce["curves"]["method"].astype(str) == "decoder_latent")
    ].copy()
    variant = a_headce["curves"][
        (a_headce["curves"]["split"].astype(str) == "eval")
        & (a_headce["curves"]["method"].astype(str) == "decoder_latent")
    ].copy()
    for frame in [control, variant]:
        for col in ["step", "train_loss", "test_loss", "train_acc", "test_acc"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    merged = control[KEY_COLS + ["step", "train_loss", "test_loss", "train_acc", "test_acc"]].merge(
        variant[KEY_COLS + ["step", "train_loss", "test_loss", "train_acc", "test_acc"]],
        on=KEY_COLS + ["step"],
        suffixes=("_control", "_A"),
    )
    for col in ["train_loss", "test_loss", "train_acc", "test_acc"]:
        merged[f"delta_{col}"] = merged[f"{col}_A"] - merged[f"{col}_control"]
    return merged


def _summarize_margin(margin: pd.DataFrame, splice: pd.DataFrame, headce_delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name, group in [("all", margin), *list(margin.groupby("task_name"))]:
        target = group["A_minus_control_test_loss"]
        rows.append(
            {
                "section": "margin",
                "group": task_name,
                "starts": int(len(group)),
                "mean_step0_test_gap": float(target.mean()),
                "median_step0_test_gap": float(target.median()),
                "corr_gap_vs_delta_margin_p05": _safe_corr(target, group["A_minus_control_test_margin_p05"]),
                "corr_gap_vs_delta_margin_p01": _safe_corr(target, group["A_minus_control_test_margin_p01"]),
                "corr_gap_vs_flip_rate": _safe_corr(target, group["test_control_to_A_flip_rate"]),
                "corr_gap_vs_bad_flip_rate": _safe_corr(target, group["test_control_correct_A_wrong_rate"]),
                "corr_gap_vs_delta_ce_top5": _safe_corr(target, group["A_minus_control_test_ce_top5_mean"]),
                "corr_gap_vs_delta_fc2_row_direction_error": _safe_corr(target, group["A_minus_control_fc2_row_direction_error_mean"]),
                "mean_A_minus_control_fc2_row_direction_error": float(group["A_minus_control_fc2_row_direction_error_mean"].mean()),
            }
        )
    for candidate, group in splice.groupby("candidate"):
        rows.append(
            {
                "section": "splice",
                "group": candidate,
                "starts": int(len(group)),
                "mean_step0_test_gap": float(group["A_minus_control_test_loss"].mean()),
                "median_step0_test_gap": float(group["A_minus_control_test_loss"].median()),
                "mean_rescue_fraction": float(group["rescue_fraction"].mean()),
                "median_rescue_fraction": float(group["rescue_fraction"].median()),
                "mean_gap_after_candidate": float(group["gap_after_candidate_vs_control"].mean()),
                "median_gap_after_candidate": float(group["gap_after_candidate_vs_control"].median()),
            }
        )
    step0 = headce_delta[headce_delta["step"] == 0].copy()
    post0 = headce_delta[headce_delta["step"] > 0].copy()
    rows.append(
        {
            "section": "headce_pair",
            "group": "all",
            "starts": int(step0["source_weight_index"].nunique()),
            "mean_step0_test_gap": float(step0["delta_test_loss"].mean()),
            "median_step0_test_gap": float(step0["delta_test_loss"].median()),
            "mean_post0_test_gap": float(post0["delta_test_loss"].mean()),
            "median_post0_test_gap": float(post0["delta_test_loss"].median()),
        }
    )
    return pd.DataFrame(rows)


def _plot_margin_scatter(margin: pd.DataFrame, *, prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    colors = {"mnist": "#4c78a8", "fashion_mnist": "#f58518"}
    specs = [
        ("A_minus_control_test_margin_p05", "A - control margin p5", False),
        ("test_control_correct_A_wrong_rate", "control correct, A wrong rate", True),
        ("A_minus_control_test_ce_top5_mean", "A - control top 5% CE", True),
    ]
    for ax, (xcol, xlabel, positive_bad) in zip(axes, specs, strict=True):
        for task_name, group in margin.groupby("task_name"):
            ax.scatter(
                group[xcol],
                group["A_minus_control_test_loss"],
                s=58,
                color=colors.get(str(task_name), "#666666"),
                label=str(task_name),
                alpha=0.85,
            )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        corr = _safe_corr(margin[xcol], margin["A_minus_control_test_loss"])
        ax.set_title(f"{xlabel}\nr={corr:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("A - control step0 test CE")
        ax.grid(alpha=0.25)
        if not positive_bad:
            ax.invert_xaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(OUT_DIR / f"{prefix}_margin_tail_vs_step0_gap.png", dpi=190)
    plt.close(fig)


def _plot_splice(splice: pd.DataFrame, *, prefix: str) -> None:
    order = [
        "full_control_fc2",
        "control_global_direction_A_scale",
        "A_global_direction_control_scale",
        "control_row_direction_A_row_scale",
        "A_row_direction_control_row_scale",
    ]
    summary = (
        splice.groupby("candidate", as_index=False)
        .agg(
            mean_rescue=("rescue_fraction", "mean"),
            median_rescue=("rescue_fraction", "median"),
            mean_gap_after=("gap_after_candidate_vs_control", "mean"),
        )
        .set_index("candidate")
        .reindex(order)
        .reset_index()
    )
    labels = [
        "full\ncontrol fc2",
        "control dir\nA scale",
        "A dir\ncontrol scale",
        "control row-dir\nA row-scale",
        "A row-dir\ncontrol row-scale",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(summary))
    axes[0].bar(x, summary["mean_rescue"], color="#4c78a8", label="mean")
    axes[0].scatter(x, summary["median_rescue"], color="#f58518", zorder=3, label="median")
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_title("fc2.weight component rescue")
    axes[0].set_ylabel("fraction of A-control step0 gap removed")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(x, summary["mean_gap_after"], color="#72b7b2")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_title("remaining gap after candidate")
    axes[1].set_ylabel("candidate - control step0 test CE")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(OUT_DIR / f"{prefix}_fc2_scale_direction_rescue.png", dpi=190)
    plt.close(fig)


def _plot_head_ce_distribution(summary: pd.DataFrame, *, prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {"control": "#4c78a8", "A": "#f58518"}
    square = summary[summary["loss_variant"] == "square"].copy()
    for variant, group in square.groupby("variant"):
        axes[0].plot(group["margin"], group["active_fraction"], marker="o", color=colors.get(str(variant)), label=str(variant))
        axes[1].plot(group["margin"], group["anchor_loss_p99"], marker="o", color=colors.get(str(variant)), label=str(variant))
        axes[2].plot(group["margin"], group["grad_proxy_p99"], marker="o", color=colors.get(str(variant)), label=f"{variant} square")
    for delta, style in [(0.02, "--"), (0.05, ":")]:
        huber = summary[(summary["loss_variant"] == "huber") & (np.isclose(summary["huber_delta"], delta))].copy()
        for variant, group in huber.groupby("variant"):
            axes[2].plot(group["margin"], group["grad_proxy_p99"], marker="s", linestyle=style, color=colors.get(str(variant)), label=f"{variant} huber {delta:g}")
    axes[0].set_title("head CE hinge active fraction")
    axes[0].set_ylabel("fraction of sampled train batches")
    axes[1].set_title("square hinge p99 loss")
    axes[1].set_ylabel("p99 anchor loss")
    axes[2].set_title("p99 gradient proxy")
    axes[2].set_ylabel("p99 proxy norm")
    for ax in axes:
        ax.set_xlabel("CE margin")
        ax.grid(alpha=0.25)
    axes[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(OUT_DIR / f"{prefix}_head_ce_batch_gap_distribution.png", dpi=190)
    plt.close(fig)


def _plot_headce_training(train: pd.DataFrame, headce_delta: pd.DataFrame, *, prefix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    colors = {"control": "#4c78a8", "A": "#f58518"}
    for variant, group in train.groupby("variant"):
        group = group.sort_values("step")
        axes[0, 0].plot(group["step"], group["train_function_anchor_effective_loss"], color=colors.get(str(variant)), label=str(variant))
        axes[0, 1].plot(group["step"], group["train_function_anchor_ce_delta"], color=colors.get(str(variant)), label=str(variant))
        axes[1, 0].plot(group["step"], group["val_loss"], color=colors.get(str(variant)), label=str(variant))
    mean_delta = headce_delta.groupby("step", as_index=False).agg(delta_test_loss=("delta_test_loss", "mean"))
    axes[1, 1].plot(mean_delta["step"], mean_delta["delta_test_loss"], color="#7b6bbd", marker="o")
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_title(f"{prefix}: effective anchor loss")
    axes[0, 1].set_title(f"{prefix}: decoded CE - raw CE")
    axes[1, 0].set_title(f"{prefix}: validation loss")
    axes[1, 1].set_title(f"{prefix}: A - control eval curve")
    axes[1, 1].set_ylabel("test CE delta")
    for ax in axes.flat:
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUT_DIR / f"{prefix}_training_and_downstream.png", dpi=190)
    plt.close(fig)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Posthoc causal diagnostics for Variant A margin/head mechanism.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--control-run", default=BASE_CONTROL_RUN)
    parser.add_argument("--a-run", default=BASE_A_RUN)
    parser.add_argument("--paired-control-run", default=HEADCE_CONTROL_RUN)
    parser.add_argument("--paired-a-run", default=HEADCE_A_RUN)
    parser.add_argument("--prefix", default="cap0p25")
    parser.add_argument("--paired-prefix", default="headce_m0")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)
    paired_prefix = str(args.paired_prefix)
    t0 = time.perf_counter()
    _log(
        "startup "
        f"device={args.device} dtype=float32 samples={args.samples} batch_samples={args.batch_samples} "
        f"batch_size={args.batch_size} control_run={args.control_run} a_run={args.a_run} "
        f"paired_control_run={args.paired_control_run} paired_a_run={args.paired_a_run} "
        f"output_dir={OUT_DIR}"
    )
    _log("stage=load_base_runs")
    control = _load_run(str(args.control_run), device=str(args.device))
    a_run = _load_run(str(args.a_run), device=str(args.device))
    if list(control["spec"].keys) != list(a_run["spec"].keys):
        raise ValueError("base pair spec mismatch")
    max_weight_diff = float((control["weights"] - a_run["weights"]).abs().max().detach().cpu().item())
    _log(f"validity base_weight_pool_max_abs_diff={max_weight_diff:.6g}")
    if max_weight_diff > 1e-6:
        raise ValueError(f"base pair weight pool mismatch: {max_weight_diff}")

    _log("stage=margin_and_scale_direction_splice")
    margin, splice, validation = _analyze_margin_and_splice(control=control, a_run=a_run, samples=int(args.samples))
    margin.to_csv(OUT_DIR / f"{prefix}_margin_tail_rows.csv", index=False)
    splice.to_csv(OUT_DIR / f"{prefix}_fc2_scale_direction_splice_rows.csv", index=False)
    validation.to_csv(OUT_DIR / f"{prefix}_recompute_validation.csv", index=False)

    _log("stage=head_ce_gap_distribution")
    head_gap_rows = _head_ce_gap_rows(
        control=control,
        a_run=a_run,
        samples=int(args.samples),
        batch_samples=int(args.batch_samples),
        batch_size=int(args.batch_size),
    )
    control_anchor_coeff = float(getattr(control["cfg"], "vae_function_anchor_coeff", 0.0))
    a_anchor_coeff = float(getattr(a_run["cfg"], "vae_function_anchor_coeff", 0.0))
    if abs(control_anchor_coeff - a_anchor_coeff) > 1e-12:
        raise ValueError(
            "head CE gap summary requires matched function-anchor coeffs, "
            f"got control={control_anchor_coeff} A={a_anchor_coeff}"
        )
    _log(f"head_ce_gap_summary_coeff={control_anchor_coeff:.6g}")
    head_gap_summary = _summarize_head_ce_gaps(head_gap_rows, coeff=control_anchor_coeff)
    head_gap_rows.to_csv(OUT_DIR / f"{prefix}_head_ce_batch_gaps.csv", index=False)
    head_gap_summary.to_csv(OUT_DIR / f"{prefix}_head_ce_gap_summary.csv", index=False)

    _log("stage=load_paired_runs")
    control_headce = _load_run(str(args.paired_control_run), device=str(args.device))
    a_headce = _load_run(str(args.paired_a_run), device=str(args.device))
    headce_train, headce_train_summary = _headce_run_summary(control_headce, a_headce)
    headce_delta = _headce_pair_delta(control_headce, a_headce)
    headce_train.to_csv(OUT_DIR / f"{paired_prefix}_train_rows.csv", index=False)
    headce_train_summary.to_csv(OUT_DIR / f"{paired_prefix}_train_summary.csv", index=False)
    headce_delta.to_csv(OUT_DIR / f"{paired_prefix}_curve_deltas.csv", index=False)

    summary = _summarize_margin(margin, splice, headce_delta)
    summary.to_csv(OUT_DIR / f"{prefix}_margin_mechanism_summary.csv", index=False)

    _log("stage=plots")
    _plot_margin_scatter(margin, prefix=prefix)
    _plot_splice(splice, prefix=prefix)
    _plot_head_ce_distribution(head_gap_summary, prefix=prefix)
    _plot_headce_training(headce_train, headce_delta, prefix=paired_prefix)

    max_validation = float(validation.drop(columns=["source_weight_index", "start_index"]).max().max()) if not validation.empty else float("nan")
    elapsed = time.perf_counter() - t0
    _log(
        "done "
        f"elapsed_sec={elapsed:.2f} max_recompute_absdiff={max_validation:.6g} "
        f"rows={OUT_DIR / f'{prefix}_margin_tail_rows.csv'} "
        f"summary={OUT_DIR / f'{prefix}_margin_mechanism_summary.csv'}"
    )
    _log("summary")
    print(summary.to_string(index=False), flush=True)
    _log("head_ce_gap_summary")
    print(head_gap_summary.to_string(index=False), flush=True)
    _log("headce_train_summary")
    print(headce_train_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
