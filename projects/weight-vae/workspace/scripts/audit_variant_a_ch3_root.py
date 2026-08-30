#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    config_hash,
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    FlatSpec,
    WeightNormalizer,
    decode_weights,
    encode_weights,
    load_celo_meta_task_tensors,
    move_task_tensors,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    _batch_indices as _precond_batch_indices,
    _hessian_via_batched_hvp,
    _task_loss_from_flat,
)
from scripts.audit_reparam_cg import (
    DecoderMatvec,
    _hvp_x,
    _load_cfg,
    _load_start_bank,
    _tensor_sha256,
)
from scripts.audit_variant_a_clamped_cg import (
    _flat_spec_slices,
    _identity_validation,
    _load_run_artifacts,
    _run_dir,
    _select_flat_blocks,
    _selected_run_identity,
)


DEFAULT_START_BANK = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "rank_repair_m2048_h4096/trajectory_discriminator/selected_4_stress_start_bank.csv"
)
DEFAULT_DOWNSTREAM_DELTAS = Path(
    "docs/reparam_preconditioning_experiments/variant_A_causal_debug/"
    "rank_repair_m2048_h4096/latent_head_clamp/full64_raw_w0_clamp_adam/"
    "full64_latent_head_clamp_per_start_deltas.csv"
)


def _log(message: str) -> None:
    print(f"[ch3_root_audit] {message}", flush=True)


def _parse_source_indices(value: str) -> set[int]:
    result: set[int] = set()
    for item in str(value).split(","):
        item = item.strip()
        if item:
            result.add(int(item))
    return result


def _stable_seed(*values: Any) -> int:
    payload = ":".join(str(v) for v in values).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _block_groups(spec: FlatSpec) -> dict[str, tuple[str, ...]]:
    keys = tuple(str(key) for key in spec.keys)
    groups: dict[str, tuple[str, ...]] = {}
    if "fc2.weight" in keys:
        groups["fc2_weight"] = ("fc2.weight",)
    if "fc2.bias" in keys:
        groups["fc2_bias"] = ("fc2.bias",)
    head = tuple(key for key in keys if key.startswith("fc2."))
    if head:
        groups["classifier_head"] = head
    weights = tuple(key for key in keys if key.endswith(".weight"))
    if weights:
        groups["all_weight_tensors"] = weights
    biases = tuple(key for key in keys if key.endswith(".bias"))
    if biases:
        groups["all_bias_tensors"] = biases
    return groups


def _block_dim(spec: FlatSpec, block_keys: tuple[str, ...]) -> int:
    sizes = {str(key): int(size) for key, size in zip(spec.keys, spec.sizes, strict=True)}
    return int(sum(sizes[key] for key in block_keys))


def _safe_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.detach().float()
    right_f = right.detach().float()
    denom = left_f.norm().clamp_min(1e-30) * right_f.norm().clamp_min(1e-30)
    return _safe_float((left_f * right_f).sum() / denom)


def _block_metrics(
    vector: torch.Tensor,
    *,
    spec: FlatSpec,
    groups: dict[str, tuple[str, ...]],
    prefix: str,
) -> dict[str, float]:
    total_norm2 = _safe_float(vector.detach().float().square().sum())
    row: dict[str, float] = {f"{prefix}_norm2": float(total_norm2)}
    x_dim = int(spec.dim)
    for name, keys in groups.items():
        selected = _select_flat_blocks(vector, spec, keys)
        norm2 = _safe_float(selected.float().square().sum())
        share = norm2 / max(total_norm2, 1e-30)
        dim_frac = float(_block_dim(spec, keys)) / float(max(1, x_dim))
        row[f"{prefix}_{name}_norm2"] = float(norm2)
        row[f"{prefix}_{name}_share"] = float(share)
        row[f"{prefix}_{name}_param_fraction"] = float(dim_frac)
        row[f"{prefix}_{name}_enrichment_param"] = float(share / max(dim_frac, 1e-30))
    return row


def _block_cosines(
    vector: torch.Tensor,
    target: torch.Tensor,
    *,
    spec: FlatSpec,
    groups: dict[str, tuple[str, ...]],
    prefix: str,
) -> dict[str, float]:
    row = {
        f"{prefix}_full_cos": _cosine(vector, target),
        f"{prefix}_full_abs_cos": abs(_cosine(vector, target)),
    }
    for name, keys in groups.items():
        left = _select_flat_blocks(vector, spec, keys)
        right = _select_flat_blocks(target, spec, keys)
        cos = _cosine(left, right)
        row[f"{prefix}_{name}_cos"] = float(cos)
        row[f"{prefix}_{name}_abs_cos"] = float(abs(cos))
    return row


def _gradient_from_flat(
    flat: torch.Tensor,
    *,
    task_set: Any,
    spec: FlatSpec,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    flat_req = flat.detach().clone().requires_grad_(True)
    loss = _task_loss_from_flat(flat_req, task_set=task_set, spec=spec, tau=float(tau), batch_indices=batch_indices)
    (grad,) = torch.autograd.grad(loss, flat_req, create_graph=False, retain_graph=False)
    return grad.detach()


def _gini(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    finite = np.abs(finite)
    total = float(finite.sum())
    if total <= 0.0:
        return 0.0
    sorted_values = np.sort(finite)
    n = sorted_values.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * sorted_values) / (n * total)) - (n + 1.0) / n)


def _entropy_effective_rank(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = np.abs(finite)
    total = float(finite.sum())
    if total <= 0.0:
        return 0.0
    probs = finite / total
    probs = probs[probs > 0.0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def _top_share(values: np.ndarray, count: int) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = np.abs(finite)
    total = float(finite.sum())
    if total <= 0.0:
        return 0.0
    count = min(max(1, int(count)), finite.size)
    return float(np.sort(finite)[::-1][:count].sum() / total)


def _spectral_summary(eigvals: np.ndarray) -> dict[str, float]:
    eig = np.asarray(eigvals, dtype=np.float64)
    abs_eig = np.abs(eig)
    eig2 = eig * eig
    loss = (eig2 - 1.0) ** 2
    pressure_signed = 4.0 * eig * (eig2 - 1.0)
    pressure_abs = np.abs(pressure_signed)
    pressure_energy = pressure_signed * pressure_signed
    q = np.quantile(abs_eig, [0.01, 0.10, 0.50, 0.90, 0.95, 0.99])
    return {
        "hessian_eig_min": float(eig.min()),
        "hessian_eig_max": float(eig.max()),
        "hessian_abs_eig_min": float(abs_eig.min()),
        "hessian_abs_eig_max": float(abs_eig.max()),
        "hessian_abs_eig_p01": float(q[0]),
        "hessian_abs_eig_p10": float(q[1]),
        "hessian_abs_eig_p50": float(q[2]),
        "hessian_abs_eig_p90": float(q[3]),
        "hessian_abs_eig_p95": float(q[4]),
        "hessian_abs_eig_p99": float(q[5]),
        "hessian_negative_fraction": float(np.mean(eig < 0.0)),
        "li_A_full_per_dim": float(np.mean(loss)),
        "li_gap_per_dim": float(np.mean((abs_eig - 1.0) ** 2)),
        "li_trace_h2_per_dim": float(np.mean(eig2)),
        "bulk_abs_eig_lt_0p01_fraction": float(np.mean(abs_eig < 0.01)),
        "bulk_abs_eig_lt_0p1_fraction": float(np.mean(abs_eig < 0.1)),
        "bulk_abs_eig_lt_1_fraction": float(np.mean(abs_eig < 1.0)),
        "tail_abs_eig_gt_1_fraction": float(np.mean(abs_eig > 1.0)),
        "tail_abs_eig_gt_10_fraction": float(np.mean(abs_eig > 10.0)),
        "a_loss_top1_share": _top_share(loss, 1),
        "a_loss_top5_share": _top_share(loss, 5),
        "a_loss_top10_share": _top_share(loss, 10),
        "a_pressure_abs_top1_share": _top_share(pressure_abs, 1),
        "a_pressure_abs_top5_share": _top_share(pressure_abs, 5),
        "a_pressure_abs_top10_share": _top_share(pressure_abs, 10),
        "a_pressure_energy_top1_share": _top_share(pressure_energy, 1),
        "a_pressure_energy_top5_share": _top_share(pressure_energy, 5),
        "a_pressure_energy_top10_share": _top_share(pressure_energy, 10),
        "a_pressure_abs_effective_rank": _entropy_effective_rank(pressure_abs),
        "a_pressure_energy_effective_rank": _entropy_effective_rank(pressure_energy),
        "a_pressure_abs_gini": _gini(pressure_abs),
        "a_pressure_energy_gini": _gini(pressure_energy),
    }


def _mode_scalars(eigvals: np.ndarray) -> pd.DataFrame:
    eig = np.asarray(eigvals, dtype=np.float64)
    eig2 = eig * eig
    loss = (eig2 - 1.0) ** 2
    pressure_signed = 4.0 * eig * (eig2 - 1.0)
    pressure_abs = np.abs(pressure_signed)
    pressure_energy = pressure_signed * pressure_signed
    order = np.argsort(-pressure_abs)
    pressure_total = float(pressure_abs.sum())
    loss_total = float(loss.sum())
    energy_total = float(pressure_energy.sum())
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    cumulative = np.cumsum(pressure_abs[order]) / max(pressure_total, 1e-30)
    cumulative_by_index = np.empty_like(cumulative)
    cumulative_by_index[order] = cumulative
    return pd.DataFrame(
        {
            "mode_index": np.arange(len(eig), dtype=np.int64),
            "pressure_rank": ranks.astype(np.int64),
            "eigenvalue": eig,
            "abs_eigenvalue": np.abs(eig),
            "a_loss_contrib": loss,
            "a_loss_share": loss / max(loss_total, 1e-30),
            "a_pressure_signed": pressure_signed,
            "a_pressure_abs": pressure_abs,
            "a_pressure_abs_share": pressure_abs / max(pressure_total, 1e-30),
            "a_pressure_abs_cumulative_share": cumulative_by_index,
            "a_pressure_energy": pressure_energy,
            "a_pressure_energy_share": pressure_energy / max(energy_total, 1e-30),
        }
    )


def _weighted_average(rows: list[dict[str, float]], weights: np.ndarray, column: str) -> float:
    if not rows:
        return float("nan")
    values = np.asarray([float(row.get(column, float("nan"))) for row in rows], dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not bool(mask.any()):
        return float("nan")
    selected_weights = np.abs(weights[mask])
    denom = float(selected_weights.sum())
    if denom <= 0.0:
        return float(np.nanmean(values[mask]))
    return float(np.sum(values[mask] * selected_weights) / denom)


def _span_fraction(directions: list[torch.Tensor], target: torch.Tensor) -> float:
    if not directions:
        return float("nan")
    target_f = target.detach().float().reshape(-1)
    denom = target_f.square().sum().clamp_min(1e-30)
    matrix = torch.stack([direction.detach().float().reshape(-1) for direction in directions], dim=1)
    if int(matrix.numel()) == 0 or float(matrix.float().norm().detach().cpu().item()) <= 0.0:
        return float("nan")
    q, _r = torch.linalg.qr(matrix, mode="reduced")
    projection = q @ (q.transpose(0, 1) @ target_f)
    return _safe_float(projection.square().sum() / denom)


def _block_span_fractions(
    directions: list[torch.Tensor],
    target: torch.Tensor,
    *,
    spec: FlatSpec,
    groups: dict[str, tuple[str, ...]],
    prefix: str,
) -> dict[str, float]:
    row = {f"{prefix}_full_span_fraction": _span_fraction(directions, target)}
    for name, keys in groups.items():
        block_dirs = [_select_flat_blocks(direction, spec, keys) for direction in directions]
        block_target = _select_flat_blocks(target, spec, keys)
        row[f"{prefix}_{name}_span_fraction"] = _span_fraction(block_dirs, block_target)
    return row


def _random_latent_probe(z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    probe = torch.randn(tuple(z.shape), generator=generator, device="cpu", dtype=torch.float32).to(device=z.device, dtype=z.dtype)
    return probe * (math.sqrt(float(z.numel())) / probe.float().norm().clamp_min(1e-12))


def _corruption_rows(
    *,
    source_weight_index: int,
    task_name: str,
    tau: float,
    raw: torch.Tensor,
    decoded_control: torch.Tensor,
    decoded_a: torch.Tensor,
    spec: FlatSpec,
    groups: dict[str, tuple[str, ...]],
    start_row: pd.Series,
) -> dict[str, Any]:
    delta = decoded_a - decoded_control
    row: dict[str, Any] = {
        "source_weight_index": int(source_weight_index),
        "task_name": str(task_name),
        "tau": float(tau),
        "start_bank_position": int(start_row.get("start_bank_position", -1)),
        "selection": str(start_row.get("selection", "")),
        "discriminator_selection": str(start_row.get("discriminator_selection", "")),
        "start_delta_aulc": float(start_row.get("delta_aulc", float("nan"))),
        "start_delta_step0_test_loss": float(start_row.get("delta_step0_test_loss", float("nan"))),
        "start_delta_final_test_loss": float(start_row.get("delta_final_test_loss", float("nan"))),
        "decoded_A_minus_control_norm": _safe_float(delta.float().norm()),
        "decoded_control_minus_raw_rel_l2": _safe_float((decoded_control - raw).float().norm() / raw.float().norm().clamp_min(1e-30)),
        "decoded_A_minus_raw_rel_l2": _safe_float((decoded_a - raw).float().norm() / raw.float().norm().clamp_min(1e-30)),
    }
    row.update(_block_metrics(delta, spec=spec, groups=groups, prefix="delta_A_minus_control"))
    for name, keys in groups.items():
        control_block = _select_flat_blocks(decoded_control - raw, spec, keys)
        a_block = _select_flat_blocks(decoded_a - raw, spec, keys)
        raw_block = _select_flat_blocks(raw, spec, keys)
        row[f"decoded_control_minus_raw_{name}_rel_l2"] = _safe_float(
            control_block.float().norm() / raw_block.float().norm().clamp_min(1e-30)
        )
        row[f"decoded_A_minus_raw_{name}_rel_l2"] = _safe_float(a_block.float().norm() / raw_block.float().norm().clamp_min(1e-30))
    return row


def _load_downstream(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "method" in frame.columns:
        frame = frame[frame["method"].astype(str) == "decoder_latent"].copy()
    keep = [
        col
        for col in [
            "source_weight_index",
            "test_mean_loss_delta",
            "test_trapz_loss_delta",
            "step0_test_loss_delta",
            "post0_test_loss_mean_delta",
            "train_mean_loss_delta",
        ]
        if col in frame.columns
    ]
    return frame[keep].drop_duplicates(subset=["source_weight_index"]).copy() if keep else pd.DataFrame()


def _run_label_audit(
    *,
    label: str,
    label_index: int,
    artifacts: dict[str, Any],
    cfg: Any,
    start_bank: pd.DataFrame,
    decoded_by_source: dict[int, dict[str, torch.Tensor]],
    args: argparse.Namespace,
    groups: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    weights: torch.Tensor = artifacts["weights"]
    records: pd.DataFrame = artifacts["weight_records"]
    spec: FlatSpec = artifacts["spec"]
    vae: torch.nn.Module = artifacts["vae"]
    normalizer: WeightNormalizer = artifacts["normalizer"]
    task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=device, dtype=dtype)

    mode_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    label_start = time.perf_counter()
    _log(
        "label_start "
        f"label={label} config_hash={config_hash(cfg)} device={device} dtype={dtype} "
        f"starts={len(start_bank)} batch_size={int(args.batch_size)} top_k={int(args.top_k)} "
        f"random_probes={int(args.random_probes)} hessian_chunk_size={int(args.hessian_chunk_size)}"
    )
    _log(f"label_config label={label} resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}")

    for start_pos, start_row in start_bank.iterrows():
        source_weight_index = int(start_row["source_weight_index"])
        record = records.iloc[source_weight_index].to_dict()
        task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
        tau = float(record.get("tau", start_row.get("tau", 1.0)))
        task_set = task_tensors[task_name]
        batch_indices = _precond_batch_indices(
            task_set,
            batch_size=int(args.batch_size),
            step=int(args.hessian_batch_step),
            sample_key=source_weight_index,
            pair_key=int(args.hessian_batch_pair_key),
        )
        if batch_indices is not None:
            batch_indices = batch_indices.to(device=device)
        raw = weights[source_weight_index].detach()
        decoded_control = decoded_by_source[source_weight_index]["control"]
        decoded_a = decoded_by_source[source_weight_index]["A"]
        delta_a_control = decoded_a - decoded_control

        with torch.no_grad():
            z = encode_weights(vae, normalizer, raw.reshape(1, -1)).squeeze(0).detach()
            decoded = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0).detach()
        decoder_ops = DecoderMatvec(vae=vae, normalizer=normalizer, z=z)
        grad_x = _gradient_from_flat(decoded, task_set=task_set, spec=spec, tau=tau, batch_indices=batch_indices)

        def loss_from_z(latent: torch.Tensor) -> torch.Tensor:
            candidate = decode_weights(vae, normalizer, latent.reshape(1, -1)).squeeze(0)
            return _task_loss_from_flat(candidate, task_set=task_set, spec=spec, tau=tau, batch_indices=batch_indices)

        _log(
            "hessian_start "
            f"label={label} source={source_weight_index} pos={start_pos + 1}/{len(start_bank)} "
            f"task={task_name} tau={tau:.6g} batch_hash={_tensor_sha256(batch_indices)}"
        )
        hessian_start = time.perf_counter()
        hessian = _hessian_via_batched_hvp(loss_from_z, z, chunk_size=int(args.hessian_chunk_size))
        hessian = 0.5 * (hessian + hessian.transpose(0, 1))
        symmetry_rel = _safe_float((hessian - hessian.transpose(0, 1)).float().norm() / hessian.float().norm().clamp_min(1e-30))
        eigvals_t, eigvecs_t = torch.linalg.eigh(hessian.detach().double())
        eigvals = eigvals_t.detach().cpu().numpy().astype(np.float64)
        scalar_frame = _mode_scalars(eigvals)
        spectral = _spectral_summary(eigvals)
        hessian_elapsed = time.perf_counter() - hessian_start

        random_generator = torch.Generator(device="cpu")
        random_generator.manual_seed(_stable_seed(int(args.seed), label, source_weight_index, "random_latent"))
        random_jv_rows: list[dict[str, float]] = []
        for probe_id in range(int(args.random_probes)):
            u = _random_latent_probe(z, generator=random_generator)
            jv = decoder_ops.j_vec(u)
            hx = _hvp_x(
                decoded,
                jv,
                images=task_set.train_images if batch_indices is None else task_set.train_images.index_select(0, batch_indices),
                labels=task_set.train_labels if batch_indices is None else task_set.train_labels.index_select(0, batch_indices),
                spec=spec,
                tau=tau,
            )
            row: dict[str, Any] = {
                "label": label,
                "source_weight_index": int(source_weight_index),
                "task_name": task_name,
                "tau": float(tau),
                "probe_id": int(probe_id),
            }
            row.update(_block_metrics(jv, spec=spec, groups=groups, prefix="random_jv"))
            row.update(_block_metrics(hx, spec=spec, groups=groups, prefix="random_hx"))
            row.update(_block_cosines(jv, delta_a_control, spec=spec, groups=groups, prefix="random_jv_vs_delta"))
            row.update(_block_cosines(jv, grad_x, spec=spec, groups=groups, prefix="random_jv_vs_grad"))
            random_rows.append(row)
            random_jv_rows.append(row)

        top_k = min(max(1, int(args.top_k)), int(len(scalar_frame)))
        top = scalar_frame.sort_values("pressure_rank", ascending=True).head(top_k).copy()
        selected_mode_rows: list[dict[str, Any]] = []
        top_directions: list[torch.Tensor] = []
        top_pressure = top["a_pressure_abs"].to_numpy(dtype=np.float64)
        images = task_set.train_images if batch_indices is None else task_set.train_images.index_select(0, batch_indices)
        labels = task_set.train_labels if batch_indices is None else task_set.train_labels.index_select(0, batch_indices)
        for _, mode_scalar in top.iterrows():
            mode_index = int(mode_scalar["mode_index"])
            q = eigvecs_t[:, mode_index].to(device=device, dtype=dtype)
            jv = decoder_ops.j_vec(q)
            hx = _hvp_x(decoded, jv, images=images, labels=labels, spec=spec, tau=tau)
            top_directions.append(jv.detach())
            row = {
                "label": label,
                "source_weight_index": int(source_weight_index),
                "start_bank_position": int(start_row.get("start_bank_position", start_pos)),
                "selection": str(start_row.get("selection", "")),
                "discriminator_selection": str(start_row.get("discriminator_selection", "")),
                "task_name": task_name,
                "tau": float(tau),
                "batch_indices_sha256": _tensor_sha256(batch_indices),
                "mode_index": mode_index,
                "pressure_rank": int(mode_scalar["pressure_rank"]),
                "eigenvalue": float(mode_scalar["eigenvalue"]),
                "abs_eigenvalue": float(mode_scalar["abs_eigenvalue"]),
                "a_loss_contrib": float(mode_scalar["a_loss_contrib"]),
                "a_loss_share": float(mode_scalar["a_loss_share"]),
                "a_pressure_signed": float(mode_scalar["a_pressure_signed"]),
                "a_pressure_abs": float(mode_scalar["a_pressure_abs"]),
                "a_pressure_abs_share": float(mode_scalar["a_pressure_abs_share"]),
                "a_pressure_abs_cumulative_share": float(mode_scalar["a_pressure_abs_cumulative_share"]),
                "a_pressure_energy": float(mode_scalar["a_pressure_energy"]),
                "a_pressure_energy_share": float(mode_scalar["a_pressure_energy_share"]),
            }
            row.update(_block_metrics(jv, spec=spec, groups=groups, prefix="jv"))
            row.update(_block_metrics(hx, spec=spec, groups=groups, prefix="hx"))
            row.update(_block_cosines(jv, delta_a_control, spec=spec, groups=groups, prefix="jv_vs_delta"))
            row.update(_block_cosines(jv, grad_x, spec=spec, groups=groups, prefix="jv_vs_grad"))
            mode_rows.append(row)
            selected_mode_rows.append(row)

        start_summary: dict[str, Any] = {
            "label": label,
            "source_weight_index": int(source_weight_index),
            "start_bank_position": int(start_row.get("start_bank_position", start_pos)),
            "selection": str(start_row.get("selection", "")),
            "discriminator_selection": str(start_row.get("discriminator_selection", "")),
            "task_name": task_name,
            "tau": float(tau),
            "batch_indices_sha256": _tensor_sha256(batch_indices),
            "hessian_elapsed_sec": float(hessian_elapsed),
            "hessian_symmetry_rel": float(symmetry_rel),
            "decoded_reconstruction_rel_l2": _safe_float((decoded - raw).float().norm() / raw.float().norm().clamp_min(1e-30)),
            "grad_x_norm": _safe_float(grad_x.float().norm()),
            **spectral,
            **_block_span_fractions(top_directions, delta_a_control, spec=spec, groups=groups, prefix=f"top{top_k}_jv_vs_delta"),
        }
        for k in [1, 5, 10, top_k]:
            if k > top_k:
                continue
            sub_rows = selected_mode_rows[:k]
            sub_weights = top_pressure[:k]
            start_summary[f"top{k}_pressure_abs_share"] = float(np.sum(sub_weights) / max(float(scalar_frame["a_pressure_abs"].sum()), 1e-30))
            for group_name in groups:
                for metric_prefix in ["jv", "hx"]:
                    share_col = f"{metric_prefix}_{group_name}_share"
                    enrich_col = f"{metric_prefix}_{group_name}_enrichment_param"
                    start_summary[f"top{k}_{metric_prefix}_{group_name}_pressure_weighted_share"] = _weighted_average(
                        sub_rows, sub_weights, share_col
                    )
                    start_summary[f"top{k}_{metric_prefix}_{group_name}_pressure_weighted_enrichment_param"] = _weighted_average(
                        sub_rows, sub_weights, enrich_col
                    )
                start_summary[f"top{k}_jv_vs_delta_{group_name}_pressure_weighted_abs_cos"] = _weighted_average(
                    sub_rows, sub_weights, f"jv_vs_delta_{group_name}_abs_cos"
                )
                start_summary[f"top{k}_jv_vs_grad_{group_name}_pressure_weighted_abs_cos"] = _weighted_average(
                    sub_rows, sub_weights, f"jv_vs_grad_{group_name}_abs_cos"
                )
        if random_jv_rows:
            for group_name in groups:
                for metric_prefix in ["random_jv", "random_hx"]:
                    values = np.asarray([float(row[f"{metric_prefix}_{group_name}_share"]) for row in random_jv_rows], dtype=np.float64)
                    start_summary[f"{metric_prefix}_{group_name}_share_mean"] = float(np.mean(values))
                    start_summary[f"{metric_prefix}_{group_name}_share_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                    top_col = f"top{top_k}_{'jv' if metric_prefix == 'random_jv' else 'hx'}_{group_name}_pressure_weighted_share"
                    if top_col in start_summary:
                        start_summary[f"top{top_k}_{'jv' if metric_prefix == 'random_jv' else 'hx'}_{group_name}_enrichment_over_random"] = float(
                            start_summary[top_col] / max(start_summary[f"{metric_prefix}_{group_name}_share_mean"], 1e-30)
                        )
        start_rows.append(start_summary)
        validation_rows.append(
            {
                "label": label,
                "source_weight_index": int(source_weight_index),
                "hessian_symmetry_rel": float(symmetry_rel),
                "hessian_elapsed_sec": float(hessian_elapsed),
                "mode_rows": int(len(selected_mode_rows)),
                "random_probe_rows": int(len(random_jv_rows)),
                "all_finite_hessian": bool(np.isfinite(eigvals).all()),
            }
        )
        _log(
            "hessian_done "
            f"label={label} source={source_weight_index} elapsed_sec={hessian_elapsed:.1f} "
            f"li_A={spectral['li_A_full_per_dim']:.6g} "
            f"pressure_top5={spectral['a_pressure_abs_top5_share']:.3f} "
            f"top{top_k}_fc2w_enrich={start_summary.get(f'top{top_k}_jv_fc2_weight_pressure_weighted_enrichment_param', float('nan')):.3g}"
        )

    _log(f"label_done label={label} elapsed_sec={time.perf_counter() - label_start:.1f}")
    return pd.DataFrame(mode_rows), pd.DataFrame(start_rows), pd.DataFrame(random_rows), validation_rows


def _paired_start_summary(start_summary: pd.DataFrame, *, control_label: str, a_label: str) -> pd.DataFrame:
    if start_summary.empty:
        return pd.DataFrame()
    key_cols = ["source_weight_index", "task_name", "tau", "batch_indices_sha256"]
    left = start_summary[start_summary["label"].astype(str) == str(control_label)].copy()
    right = start_summary[start_summary["label"].astype(str) == str(a_label)].copy()
    paired = left.merge(right, on=key_cols, suffixes=("_control", "_A"), how="outer", indicator=True, validate="one_to_one")
    numeric_cols = sorted(
        set(left.select_dtypes(include=[np.number]).columns).intersection(set(right.select_dtypes(include=[np.number]).columns))
        - {"source_weight_index", "tau"}
    )
    delta_cols: dict[str, pd.Series] = {}
    for col in numeric_cols:
        c_col = f"{col}_control"
        a_col = f"{col}_A"
        if c_col in paired.columns and a_col in paired.columns:
            delta_cols[f"{col}_delta_A_minus_control"] = paired[a_col] - paired[c_col]
    if delta_cols:
        paired = pd.concat([paired, pd.DataFrame(delta_cols)], axis=1)
    return paired.copy()


def _correlations(join: pd.DataFrame) -> pd.DataFrame:
    if join.empty:
        return pd.DataFrame()
    target_cols = [
        col
        for col in ["step0_test_loss_delta", "test_mean_loss_delta", "test_trapz_loss_delta", "post0_test_loss_mean_delta"]
        if col in join.columns
    ]
    candidate_cols = [
        col
        for col in join.columns
        if (
            col.endswith("_delta_A_minus_control")
            or "top" in col
            or col in ["a_pressure_abs_top5_share_A", "a_pressure_abs_effective_rank_A"]
        )
        and pd.api.types.is_numeric_dtype(join[col])
    ]
    rows: list[dict[str, Any]] = []
    for target in target_cols:
        y = pd.to_numeric(join[target], errors="coerce")
        for candidate in candidate_cols:
            x = pd.to_numeric(join[candidate], errors="coerce")
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 3:
                continue
            if float(x[mask].std(ddof=0)) <= 0.0 or float(y[mask].std(ddof=0)) <= 0.0:
                continue
            rows.append(
                {
                    "target": target,
                    "metric": candidate,
                    "n": int(mask.sum()),
                    "pearson": float(x[mask].corr(y[mask], method="pearson")),
                    "spearman": float(x[mask].corr(y[mask], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    control_dir = _run_dir(str(args.control_run))
    a_dir = _run_dir(str(args.a_run))
    cfg_control = _load_cfg(control_dir, device=str(args.device), batch_size=int(args.batch_size))
    cfg_a = _load_cfg(a_dir, device=str(args.device), batch_size=int(args.batch_size))
    device = torch.device(cfg_control.device)
    if device.type == "cuda" and not bool(args.allow_gpu):
        raise RuntimeError("GPU device requested without --allow-gpu.")

    _log(
        "start "
        f"control_run={control_dir} a_run={a_dir} output_dir={out_dir} device={device} "
        f"seed={int(args.seed)} samples={int(args.samples)} batch_size={int(args.batch_size)} "
        f"top_k={int(args.top_k)} random_probes={int(args.random_probes)}"
    )
    _log(f"control_config_hash={config_hash(cfg_control)} A_config_hash={config_hash(cfg_a)}")

    control_artifacts = _load_run_artifacts(control_dir, device=device, dtype=torch_dtype(cfg_control), cfg=cfg_control)
    a_artifacts = _load_run_artifacts(a_dir, device=device, dtype=torch_dtype(cfg_a), cfg=cfg_a)
    start_bank, start_bank_path = _load_start_bank(
        control_dir,
        weight_records=control_artifacts["weight_records"],
        role=str(args.start_role),
        samples=0,
        start_bank_csv=Path(args.start_bank_csv) if str(args.start_bank_csv).strip() else None,
    )
    source_indices = _parse_source_indices(str(args.source_indices))
    if source_indices:
        start_bank = start_bank[start_bank["source_weight_index"].astype(int).isin(source_indices)].copy()
    if int(args.samples) > 0:
        start_bank = start_bank.iloc[: int(args.samples)].copy()
    start_bank = start_bank.reset_index(drop=True)
    if start_bank.empty:
        raise RuntimeError("empty start bank after filtering")
    start_bank_hash = hashlib.sha256(start_bank.to_csv(index=False).encode("utf-8")).hexdigest()
    _log(
        "start_bank "
        f"path={start_bank_path} rows={len(start_bank)} hash={start_bank_hash[:16]} "
        f"indices={start_bank['source_weight_index'].astype(int).tolist()}"
    )

    identity = _identity_validation(
        [
            _selected_run_identity(control_dir, label=str(args.control_label), start_bank=start_bank),
            _selected_run_identity(a_dir, label=str(args.a_label), start_bank=start_bank),
        ]
    )
    (out_dir / "identity_validation.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    if not bool(identity.get("accepted", False)):
        raise RuntimeError(f"selected run identity mismatch: {identity.get('mismatches')}")

    spec: FlatSpec = control_artifacts["spec"]
    if list(spec.keys) != list(a_artifacts["spec"].keys):
        raise RuntimeError("spec key mismatch between control and A")
    groups = _block_groups(spec)
    decoded_by_source: dict[int, dict[str, torch.Tensor]] = {}
    corruption: list[dict[str, Any]] = []
    for start_pos, start_row in start_bank.iterrows():
        source_weight_index = int(start_row["source_weight_index"])
        raw = control_artifacts["weights"][source_weight_index].detach()
        with torch.no_grad():
            z_control = encode_weights(
                control_artifacts["vae"],
                control_artifacts["normalizer"],
                raw.reshape(1, -1),
            ).squeeze(0)
            dec_control = decode_weights(
                control_artifacts["vae"],
                control_artifacts["normalizer"],
                z_control.reshape(1, -1),
            ).squeeze(0).detach()
            z_a = encode_weights(
                a_artifacts["vae"],
                a_artifacts["normalizer"],
                raw.reshape(1, -1),
            ).squeeze(0)
            dec_a = decode_weights(a_artifacts["vae"], a_artifacts["normalizer"], z_a.reshape(1, -1)).squeeze(0).detach()
        record = control_artifacts["weight_records"].iloc[source_weight_index].to_dict()
        task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
        tau = float(record.get("tau", start_row.get("tau", 1.0)))
        decoded_by_source[source_weight_index] = {"control": dec_control, "A": dec_a}
        corruption.append(
            _corruption_rows(
                source_weight_index=source_weight_index,
                task_name=task_name,
                tau=tau,
                raw=raw,
                decoded_control=dec_control,
                decoded_a=dec_a,
                spec=spec,
                groups=groups,
                start_row=start_row,
            )
        )
    corruption_frame = pd.DataFrame(corruption)
    corruption_frame.to_csv(out_dir / "decoded_corruption_rows.csv", index=False)

    control_modes, control_starts, control_random, control_validation = _run_label_audit(
        label=str(args.control_label),
        label_index=0,
        artifacts=control_artifacts,
        cfg=cfg_control,
        start_bank=start_bank,
        decoded_by_source=decoded_by_source,
        args=args,
        groups=groups,
    )
    a_modes, a_starts, a_random, a_validation = _run_label_audit(
        label=str(args.a_label),
        label_index=1,
        artifacts=a_artifacts,
        cfg=cfg_a,
        start_bank=start_bank,
        decoded_by_source=decoded_by_source,
        args=args,
        groups=groups,
    )
    mode_rows = pd.concat([control_modes, a_modes], ignore_index=True, sort=False)
    start_summary = pd.concat([control_starts, a_starts], ignore_index=True, sort=False)
    random_rows = pd.concat([control_random, a_random], ignore_index=True, sort=False)

    mode_rows.to_csv(out_dir / "spectral_mode_rows.csv", index=False)
    start_summary.to_csv(out_dir / "start_summary.csv", index=False)
    random_rows.to_csv(out_dir / "random_probe_block_rows.csv", index=False)
    paired = _paired_start_summary(start_summary, control_label=str(args.control_label), a_label=str(args.a_label))
    paired.to_csv(out_dir / "paired_start_summary.csv", index=False)
    downstream = _load_downstream(Path(args.downstream_deltas_csv))
    join = paired.copy()
    if not downstream.empty:
        join = join.merge(downstream, on="source_weight_index", how="left", validate="one_to_one")
    join = join.merge(corruption_frame, on=["source_weight_index", "task_name", "tau"], how="left", validate="one_to_one")
    join.to_csv(out_dir / "tail_downstream_join.csv", index=False)
    correlations = _correlations(join)
    correlations.to_csv(out_dir / "tail_downstream_correlations.csv", index=False)

    validation_rows = control_validation + a_validation
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(out_dir / "validation_rows.csv", index=False)
    numeric_frames = [mode_rows, start_summary, random_rows, paired, join]
    finite_ok = True
    for frame in numeric_frames:
        if frame.empty:
            continue
        numeric = frame.select_dtypes(include=[np.number])
        if not bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all()):
            finite_ok = False
            break
    validation = {
        "accepted": bool(
            bool(identity.get("accepted", False))
            and finite_ok
            and not validation_frame.empty
            and bool(validation_frame["all_finite_hessian"].all())
            and float(pd.to_numeric(validation_frame["hessian_symmetry_rel"], errors="coerce").max()) <= float(args.hessian_symmetry_gate)
            and int(mode_rows.shape[0]) == int(2 * len(start_bank) * min(max(1, int(args.top_k)), int(cfg_control.latent_dim)))
        ),
        "identity_accepted": bool(identity.get("accepted", False)),
        "identity_mismatches": list(identity.get("mismatches", [])),
        "start_bank_path": str(start_bank_path),
        "start_bank_sha256": start_bank_hash,
        "source_indices": start_bank["source_weight_index"].astype(int).tolist(),
        "labels": [str(args.control_label), str(args.a_label)],
        "mode_rows": int(mode_rows.shape[0]),
        "start_summary_rows": int(start_summary.shape[0]),
        "random_probe_rows": int(random_rows.shape[0]),
        "paired_rows": int(paired.shape[0]),
        "all_numeric_finite": bool(finite_ok),
        "hessian_symmetry_rel_max": float(pd.to_numeric(validation_frame["hessian_symmetry_rel"], errors="coerce").max()),
        "hessian_symmetry_gate": float(args.hessian_symmetry_gate),
        "outputs": {
            "identity_validation": str(out_dir / "identity_validation.json"),
            "decoded_corruption_rows": str(out_dir / "decoded_corruption_rows.csv"),
            "spectral_mode_rows": str(out_dir / "spectral_mode_rows.csv"),
            "start_summary": str(out_dir / "start_summary.csv"),
            "random_probe_block_rows": str(out_dir / "random_probe_block_rows.csv"),
            "paired_start_summary": str(out_dir / "paired_start_summary.csv"),
            "tail_downstream_join": str(out_dir / "tail_downstream_join.csv"),
            "tail_downstream_correlations": str(out_dir / "tail_downstream_correlations.csv"),
            "validation_rows": str(out_dir / "validation_rows.csv"),
            "validation": str(out_dir / "validation.json"),
            "manifest": str(out_dir / "manifest.json"),
        },
    }
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "args": vars(args),
        "control_run_dir": str(control_dir),
        "a_run_dir": str(a_dir),
        "control_config_hash": config_hash(cfg_control),
        "a_config_hash": config_hash(cfg_a),
        "control_config": asdict(cfg_control),
        "a_config": asdict(cfg_a),
        "block_groups": {key: list(value) for key, value in groups.items()},
        "validation": validation,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _log(f"done accepted={validation['accepted']} outputs={json.dumps(validation['outputs'], sort_keys=True)}")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="CH3 root-test audit for Variant A tail pressure and head damage.")
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--a-run", required=True)
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--a-label", default="A_clip20")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--start-bank-csv", default=str(DEFAULT_START_BANK))
    parser.add_argument("--start-role", default="all")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--source-indices", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hessian-batch-step", type=int, default=0)
    parser.add_argument("--hessian-batch-pair-key", type=int, default=0)
    parser.add_argument("--hessian-chunk-size", type=int, default=32)
    parser.add_argument("--hessian-symmetry-gate", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--random-probes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--downstream-deltas-csv", default=str(DEFAULT_DOWNSTREAM_DELTAS))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
