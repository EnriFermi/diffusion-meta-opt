from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    config_hash,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    decode_weights,
    decoder_jacobians,
    encode_weights,
    load_celo_meta_task_tensors,
    load_torch_cache,
    logits_from_flat,
    move_task_tensors,
    spec_from_payload,
)


def _load_cfg(output_dir: Path, *, device: str) -> ExperimentConfig:
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{output_dir / 'config.json'} does not contain a config mapping")
    values = dict(raw_cfg)
    values["device"] = str(device)
    values["dtype"] = "float32"
    return ExperimentConfig(**values)


def _selected_lrs(output_dir: Path) -> dict[str, float]:
    selected = pd.read_csv(output_dir / "selected_lrs.csv")
    rows = selected[pd.to_numeric(selected["selected"], errors="coerce").fillna(0).astype(int) == 1]
    result = {str(row["method"]): float(row["candidate_lr"]) for _, row in rows.iterrows()}
    missing = {"raw", "decoder_latent"} - set(result)
    if missing:
        raise ValueError(f"{output_dir} selected_lrs.csv missing selected methods: {sorted(missing)}")
    return result


def _task_tensor_set(task_tensors: dict[str, Any], task_name: str):
    if task_name in task_tensors:
        return task_tensors[task_name]
    if len(task_tensors) == 1:
        return next(iter(task_tensors.values()))
    raise KeyError(f"task {task_name!r} not found")


def _batch_indices(task_set, *, batch_size: int, step: int, start_index: int) -> torch.Tensor | None:
    train_count = int(task_set.train_labels.shape[0])
    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size >= train_count:
        return None
    offset = (int(start_index) * 1009 + int(step) * batch_size) % train_count
    return (torch.arange(batch_size, device=task_set.train_labels.device) + offset).remainder(train_count).long()


def _loss_acc(flat: torch.Tensor, *, task_set, spec, split: str, tau: float, batch_indices: torch.Tensor | None = None):
    if split == "train":
        images, labels = task_set.train_images, task_set.train_labels
        if batch_indices is not None:
            images = images.index_select(0, batch_indices)
            labels = labels.index_select(0, batch_indices)
    elif split == "test":
        images, labels = task_set.test_images, task_set.test_labels
    else:
        raise ValueError(split)
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc


def _norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(float(den), 1e-30))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().double().flatten()
    b64 = b.detach().double().flatten()
    denom = a64.norm() * b64.norm()
    if float(denom.item()) <= 1e-30:
        return float("nan")
    return float((torch.dot(a64, b64) / denom).detach().cpu().item())


def _finite(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    return float(value) if math.isfinite(float(value)) else float("nan")


def _eval_candidate(
    *,
    theta0: torch.Tensor,
    theta_candidate: torch.Tensor,
    batch_loss0: float,
    test_loss0: float,
    test_acc0: float,
    neg_grad_theta: torch.Tensor,
    w0_norm: float,
    task_set,
    spec,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> dict[str, float]:
    update = theta_candidate.detach() - theta0.detach()
    with torch.no_grad():
        batch_loss, batch_acc = _loss_acc(
            theta_candidate.detach(),
            task_set=task_set,
            spec=spec,
            split="train",
            tau=tau,
            batch_indices=batch_indices,
        )
        test_loss, test_acc = _loss_acc(theta_candidate.detach(), task_set=task_set, spec=spec, split="test", tau=tau)
    update_norm = _norm(update)
    return {
        "weight_step_norm": update_norm,
        "weight_step_rel_w0": _safe_ratio(update_norm, w0_norm),
        "cos_step_with_neg_grad": _cosine(update, neg_grad_theta),
        "batch_loss_after": _finite(batch_loss),
        "batch_acc_after": _finite(batch_acc),
        "batch_loss_delta": _finite(batch_loss) - float(batch_loss0),
        "test_loss_after": _finite(test_loss),
        "test_acc_after": _finite(test_acc),
        "test_loss_delta": _finite(test_loss) - float(test_loss0),
        "test_acc_delta": _finite(test_acc) - float(test_acc0),
    }


def _adam_weight_step_from_decoded(
    *,
    theta0: torch.Tensor,
    raw_lr: float,
    task_set,
    spec,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> torch.Tensor:
    value = theta0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([value], lr=float(raw_lr))
    optimizer.zero_grad(set_to_none=True)
    loss, _ = _loss_acc(value, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    loss.backward()
    optimizer.step()
    return value.detach() - theta0.detach()


def _adam_latent_step(
    *,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    z0: torch.Tensor,
    latent_lr: float,
    task_set,
    spec,
    tau: float,
    batch_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = z0.detach().clone().requires_grad_(True)
    theta0 = decode_weights(vae, normalizer, value.reshape(1, -1)).squeeze(0).detach()
    optimizer = torch.optim.Adam([value], lr=float(latent_lr))
    optimizer.zero_grad(set_to_none=True)
    theta = decode_weights(vae, normalizer, value.reshape(1, -1)).squeeze(0)
    loss, _ = _loss_acc(theta, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    loss.backward()
    optimizer.step()
    theta1 = decode_weights(vae, normalizer, value.reshape(1, -1)).squeeze(0).detach()
    return value.detach() - z0.detach(), theta1 - theta0


def _scale_latent_direction(
    *,
    jacobian: torch.Tensor,
    direction_z: torch.Tensor,
    target_weight_norm: float,
) -> tuple[torch.Tensor, float, float]:
    j64 = jacobian.detach().double()
    dz64 = direction_z.detach().double()
    linear_update = j64 @ dz64
    linear_norm = float(linear_update.norm().detach().cpu().item())
    if linear_norm <= 1e-30 or not math.isfinite(linear_norm):
        return direction_z.detach().clone().zero_(), linear_norm, 0.0
    scale = float(target_weight_norm) / linear_norm
    return (direction_z.detach() * scale).to(dtype=direction_z.dtype), linear_norm, scale


def _metric_rows_for_start(
    *,
    cfg: ExperimentConfig,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    weights_device: torch.Tensor,
    weight_records: pd.DataFrame,
    task_tensors: dict[str, Any],
    spec,
    start_row: pd.Series,
    raw_lr: float,
    latent_lr: float,
    label: str,
    run_name: str,
    damping_rel: tuple[float, ...],
    multipliers: tuple[float, ...],
    jacobian_chunk_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_weight_index = int(start_row["source_weight_index"])
    start_index = int(start_row["start_index"])
    record = weight_records.iloc[source_weight_index].to_dict()
    task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
    tau = float(record.get("tau", start_row.get("tau", 1.0)))
    task_set = _task_tensor_set(task_tensors, task_name)
    batch_size = int(getattr(cfg, "downstream_batch_size", 0))
    batch_indices = _batch_indices(task_set, batch_size=batch_size, step=0, start_index=start_index)

    w0 = weights_device[source_weight_index].detach()
    w0_norm = _norm(w0)
    with torch.no_grad():
        z0 = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0).detach()
        theta0 = decode_weights(vae, normalizer, z0.reshape(1, -1)).squeeze(0).detach()
        raw_batch_loss, raw_batch_acc = _loss_acc(w0, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
        raw_test_loss, raw_test_acc = _loss_acc(w0, task_set=task_set, spec=spec, split="test", tau=tau)
        dec_batch_loss, dec_batch_acc = _loss_acc(theta0, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
        dec_test_loss, dec_test_acc = _loss_acc(theta0, task_set=task_set, spec=spec, split="test", tau=tau)

    theta_req = theta0.detach().clone().requires_grad_(True)
    batch_loss_req, _ = _loss_acc(theta_req, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
    grad_theta = torch.autograd.grad(batch_loss_req, theta_req, retain_graph=False, create_graph=False)[0].detach()
    neg_grad_theta = -grad_theta
    grad_norm = _norm(grad_theta)

    jacobian = decoder_jacobians(
        vae,
        normalizer,
        None,
        z0.reshape(1, -1),
        create_graph=False,
        chunk_size=int(jacobian_chunk_size),
    ).squeeze(0).detach()
    j64 = jacobian.double()
    g64 = grad_theta.detach().double()
    metric = j64.T @ j64
    grad_z = (j64.T @ g64).to(dtype=z0.dtype)
    eig = torch.linalg.eigvalsh(metric.float()).detach()
    eig_min = float(eig.min().detach().cpu().item())
    eig_max = float(eig.max().detach().cpu().item())
    metric_trace = float(torch.trace(metric).detach().cpu().item())
    metric_trace2 = float((metric * metric).sum().detach().cpu().item())
    latent_dim = int(metric.shape[0])
    eig_mean = metric_trace / max(float(latent_dim), 1.0)
    condition = eig_max / max(eig_min, 1e-30)
    effective_rank = metric_trace * metric_trace / max(metric_trace2, 1e-30)

    raw_decoded_step = _adam_weight_step_from_decoded(
        theta0=theta0,
        raw_lr=raw_lr,
        task_set=task_set,
        spec=spec,
        tau=tau,
        batch_indices=batch_indices,
    )
    adam_dz, latent_adam_step = _adam_latent_step(
        vae=vae,
        normalizer=normalizer,
        z0=z0,
        latent_lr=latent_lr,
        task_set=task_set,
        spec=spec,
        tau=tau,
        batch_indices=batch_indices,
    )
    raw_decoded_step_norm = _norm(raw_decoded_step)
    latent_adam_step_norm = _norm(latent_adam_step)

    common = {
        "run_name": run_name,
        "variant_label": label,
        "source_weight_index": source_weight_index,
        "start_index": start_index,
        "task_name": task_name,
        "tau": tau,
        "raw_lr": float(raw_lr),
        "latent_lr": float(latent_lr),
        "w0_norm": w0_norm,
        "reconstruction_rel_l2": _safe_ratio(_norm(theta0 - w0), w0_norm),
        "raw_batch_loss": _finite(raw_batch_loss),
        "decoded_batch_loss": _finite(dec_batch_loss),
        "decoded_minus_raw_batch_loss": _finite(dec_batch_loss) - _finite(raw_batch_loss),
        "raw_batch_acc": _finite(raw_batch_acc),
        "decoded_batch_acc": _finite(dec_batch_acc),
        "raw_test_loss": _finite(raw_test_loss),
        "decoded_test_loss": _finite(dec_test_loss),
        "decoded_minus_raw_test_loss": _finite(dec_test_loss) - _finite(raw_test_loss),
        "raw_test_acc": _finite(raw_test_acc),
        "decoded_test_acc": _finite(dec_test_acc),
        "decoded_minus_raw_test_acc": _finite(dec_test_acc) - _finite(raw_test_acc),
        "grad_theta_norm": grad_norm,
        "grad_z_norm": _norm(grad_z),
        "metric_trace": metric_trace,
        "metric_trace_per_latent_dim": metric_trace / max(float(latent_dim), 1.0),
        "metric_effective_rank": effective_rank,
        "metric_eig_min": eig_min,
        "metric_eig_max": eig_max,
        "metric_condition": condition,
        "metric_log_eig_spread": math.log(max(eig_max, 1e-30)) - math.log(max(eig_min, 1e-30)),
        "raw_decoded_adam_step_norm": raw_decoded_step_norm,
        "raw_decoded_adam_step_rel_w0": _safe_ratio(raw_decoded_step_norm, w0_norm),
        "latent_adam_step_norm": latent_adam_step_norm,
        "latent_adam_step_rel_w0": _safe_ratio(latent_adam_step_norm, w0_norm),
        "cos_raw_decoded_adam_with_neg_grad": _cosine(raw_decoded_step, neg_grad_theta),
        "cos_latent_adam_with_neg_grad": _cosine(latent_adam_step, neg_grad_theta),
        "cos_latent_adam_with_raw_decoded_adam": _cosine(latent_adam_step, raw_decoded_step),
    }

    one_step_rows: list[dict[str, Any]] = []
    for method, theta_candidate in (
        ("raw_decoded_adam_actual", theta0 + raw_decoded_step),
        ("latent_adam_actual", theta0 + latent_adam_step),
    ):
        row = {
            **common,
            "direction_method": method,
            "damping_rel": 0.0,
            "damping_abs": 0.0,
            "scale_anchor": "actual_optimizer_step",
            "multiplier": 1.0,
            "latent_direction_linear_norm_unscaled": float("nan"),
            "latent_direction_scale": 1.0,
            "projection_cos_with_neg_grad": float("nan"),
            "projection_norm_fraction": float("nan"),
            "normal_residual_fraction": float("nan"),
        }
        row.update(
            _eval_candidate(
                theta0=theta0,
                theta_candidate=theta_candidate,
                batch_loss0=_finite(dec_batch_loss),
                test_loss0=_finite(dec_test_loss),
                test_acc0=_finite(dec_test_acc),
                neg_grad_theta=neg_grad_theta,
                w0_norm=w0_norm,
                task_set=task_set,
                spec=spec,
                tau=tau,
                batch_indices=batch_indices,
            )
        )
        one_step_rows.append(row)

    projection_rows: list[dict[str, Any]] = []
    eye64 = torch.eye(latent_dim, device=metric.device, dtype=torch.float64)
    for rel in damping_rel:
        damping_abs = float(rel) * max(eig_mean, 1e-30)
        solved = torch.linalg.solve(metric + damping_abs * eye64, -(j64.T @ g64))
        natural_dz = solved.to(device=z0.device, dtype=z0.dtype)
        natural_linear = j64 @ solved
        raw_adam_projected_dz64 = torch.linalg.solve(metric + damping_abs * eye64, j64.T @ raw_decoded_step.detach().double())
        raw_adam_projected_dz = raw_adam_projected_dz64.to(device=z0.device, dtype=z0.dtype)
        raw_adam_projected_linear = j64 @ raw_adam_projected_dz64
        projection_cos = _cosine(natural_linear, neg_grad_theta)
        projection_norm = float(natural_linear.norm().detach().cpu().item())
        residual_norm = float(((-g64) - natural_linear).norm().detach().cpu().item())
        raw_adam_projection_norm = float(raw_adam_projected_linear.norm().detach().cpu().item())
        raw_adam_projection_residual_norm = float((raw_decoded_step.detach().double() - raw_adam_projected_linear).norm().detach().cpu().item())
        projection_row = {
            **common,
            "damping_rel": float(rel),
            "damping_abs": damping_abs,
            "projection_cos_with_neg_grad": projection_cos,
            "projection_norm_fraction": _safe_ratio(projection_norm, grad_norm),
            "normal_residual_fraction": _safe_ratio(residual_norm, grad_norm),
            "raw_adam_projection_cos": _cosine(raw_adam_projected_linear, raw_decoded_step),
            "raw_adam_projection_norm_fraction": _safe_ratio(raw_adam_projection_norm, raw_decoded_step_norm),
            "raw_adam_projection_residual_fraction": _safe_ratio(raw_adam_projection_residual_norm, raw_decoded_step_norm),
            "natural_dz_norm": _norm(natural_dz),
            "natural_linear_step_norm": projection_norm,
            "cos_natural_linear_with_latent_adam": _cosine(natural_linear, latent_adam_step),
            "cos_natural_linear_with_raw_decoded_adam": _cosine(natural_linear, raw_decoded_step),
        }
        projection_rows.append(projection_row)

        for base_name, base_direction in (
            ("latent_euclidean_grad", -grad_z.detach()),
            ("latent_pullback_natural_grad", natural_dz.detach()),
            ("latent_projected_raw_adam_step", raw_adam_projected_dz.detach()),
        ):
            for anchor_name, target_norm in (
                ("match_raw_decoded_adam_step", raw_decoded_step_norm),
                ("match_latent_adam_step", latent_adam_step_norm),
            ):
                scaled_dz, linear_norm_unscaled, scale = _scale_latent_direction(
                    jacobian=jacobian,
                    direction_z=base_direction,
                    target_weight_norm=target_norm,
                )
                for multiplier in multipliers:
                    dz = scaled_dz * float(multiplier)
                    with torch.no_grad():
                        theta_candidate = decode_weights(vae, normalizer, (z0 + dz).reshape(1, -1)).squeeze(0).detach()
                    row = {
                        **common,
                        "direction_method": base_name,
                        "damping_rel": float(rel),
                        "damping_abs": damping_abs,
                        "scale_anchor": anchor_name,
                        "multiplier": float(multiplier),
                        "latent_direction_linear_norm_unscaled": linear_norm_unscaled,
                        "latent_direction_scale": scale,
                        "projection_cos_with_neg_grad": projection_cos,
                        "projection_norm_fraction": _safe_ratio(projection_norm, grad_norm),
                        "normal_residual_fraction": _safe_ratio(residual_norm, grad_norm),
                    }
                    row.update(
                        _eval_candidate(
                            theta0=theta0,
                            theta_candidate=theta_candidate,
                            batch_loss0=_finite(dec_batch_loss),
                            test_loss0=_finite(dec_test_loss),
                            test_acc0=_finite(dec_test_acc),
                            neg_grad_theta=neg_grad_theta,
                            w0_norm=w0_norm,
                            task_set=task_set,
                            spec=spec,
                            tau=tau,
                            batch_indices=batch_indices,
                        )
                    )
                    one_step_rows.append(row)

    return projection_rows, one_step_rows


def _summarize(one_step: pd.DataFrame, projection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if one_step.empty:
        return pd.DataFrame()
    group_cols = ["variant_label", "direction_method", "scale_anchor"]
    for keys, group in one_step.groupby(group_cols, dropna=False):
        variant_label, direction_method, scale_anchor = keys
        best = group.sort_values(["batch_loss_after", "test_loss_after"]).groupby(["source_weight_index"], as_index=False).first()
        rows.append(
            {
                "variant_label": variant_label,
                "direction_method": direction_method,
                "scale_anchor": scale_anchor,
                "starts": int(best["source_weight_index"].nunique()),
                "best_batch_loss_delta_median": float(best["batch_loss_delta"].median()),
                "best_test_loss_delta_median": float(best["test_loss_delta"].median()),
                "best_test_acc_delta_median": float(best["test_acc_delta"].median()),
                "step_rel_w0_median": float(best["weight_step_rel_w0"].median()),
                "cos_step_with_neg_grad_median": float(best["cos_step_with_neg_grad"].median()),
            }
        )
    summary = pd.DataFrame(rows)
    if not projection.empty:
        proj = (
            projection.groupby("variant_label", as_index=False)
            .agg(
                projection_cos_with_neg_grad_median=("projection_cos_with_neg_grad", "median"),
                projection_norm_fraction_median=("projection_norm_fraction", "median"),
                normal_residual_fraction_median=("normal_residual_fraction", "median"),
                metric_condition_median=("metric_condition", "median"),
                metric_effective_rank_median=("metric_effective_rank", "median"),
            )
        )
        summary = summary.merge(proj, how="left", on="variant_label")
    return summary


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    projection_rows: list[dict[str, Any]] = []
    one_step_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    print(
        "[latent_metric_confound] start "
        f"runs={len(args.run_dir)} device={args.device} samples={args.samples} "
        f"damping_rel={args.damping_rel} multipliers={args.multiplier} "
        f"jacobian_chunk_size={args.jacobian_chunk_size} output_root={output_root}",
        flush=True,
    )
    for run_dir_value in args.run_dir:
        run_dir_path = Path(run_dir_value).expanduser().resolve()
        run_name = run_dir_path.name
        label = str(args.label.get(run_name, run_name)) if isinstance(args.label, dict) else run_name
        cfg = _load_cfg(run_dir_path, device=str(args.device))
        device = torch.device(cfg.device)
        print(
            "[latent_metric_confound] run "
            f"name={run_name} label={label} dir={run_dir_path} config_hash={config_hash(cfg)} "
            f"device={device} dtype=float32",
            flush=True,
        )
        weight_payload = load_torch_cache(run_dir_path / "weight_pool.pt")
        vae_payload = load_torch_cache(run_dir_path / "vae_checkpoint.pt")
        if weight_payload is None or vae_payload is None:
            raise FileNotFoundError(f"{run_dir_path} missing weight_pool.pt or vae_checkpoint.pt")
        weights = weight_payload["weights"]
        weight_records = pd.DataFrame(weight_payload["records"])
        spec = spec_from_payload(weight_payload["spec"])
        normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
        vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=torch.float32).eval()
        vae.load_state_dict(vae_payload["model_state"])
        weights_device = weights.to(device=device, dtype=torch.float32)
        task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=device, dtype=torch.float32)
        selected_lrs = _selected_lrs(run_dir_path)
        results = pd.read_csv(run_dir_path / "downstream_results.csv")
        starts = results[(results["split"].astype(str) == "eval") & (results["method"].astype(str) == "decoder_latent")]
        starts = starts.sort_values("start_index").reset_index(drop=True)
        if args.source_index:
            allowed = {int(v) for v in args.source_index}
            starts = starts[starts["source_weight_index"].astype(int).isin(allowed)].reset_index(drop=True)
        elif int(args.samples) > 0:
            starts = starts.iloc[: int(args.samples)].copy()
        print(
            "[latent_metric_confound] starts "
            f"count={len(starts)} raw_lr={selected_lrs['raw']:.6g} latent_lr={selected_lrs['decoder_latent']:.6g} "
            f"indices={starts['source_weight_index'].astype(int).tolist()}",
            flush=True,
        )
        for pos, start_row in starts.iterrows():
            print(
                "[latent_metric_confound] probe "
                f"run={label} start={pos + 1}/{len(starts)} source={int(start_row['source_weight_index'])} "
                f"task={start_row['task_name']} tau={float(start_row['tau']):.6g}",
                flush=True,
            )
            projection, one_step = _metric_rows_for_start(
                cfg=cfg,
                vae=vae,
                normalizer=normalizer,
                weights_device=weights_device,
                weight_records=weight_records,
                task_tensors=task_tensors,
                spec=spec,
                start_row=start_row,
                raw_lr=selected_lrs["raw"],
                latent_lr=selected_lrs["decoder_latent"],
                label=label,
                run_name=run_name,
                damping_rel=tuple(float(v) for v in args.damping_rel),
                multipliers=tuple(float(v) for v in args.multiplier),
                jacobian_chunk_size=int(args.jacobian_chunk_size),
            )
            projection_rows.extend(projection)
            one_step_rows.extend(one_step)
            latest = pd.DataFrame(projection)
            if not latest.empty:
                row = latest.iloc[0]
                print(
                    "[latent_metric_confound] projection "
                    f"source={int(start_row['source_weight_index'])} "
                    f"proj_cos={float(row['projection_cos_with_neg_grad']):.4f} "
                    f"normal_frac={float(row['normal_residual_fraction']):.4f} "
                    f"cond={float(row['metric_condition']):.3e}",
                    flush=True,
                )
    projection_frame = pd.DataFrame(projection_rows)
    one_step_frame = pd.DataFrame(one_step_rows)
    summary_frame = _summarize(one_step_frame, projection_frame)
    projection_path = output_root / "latent_metric_projection.csv"
    one_step_path = output_root / "latent_metric_one_step.csv"
    summary_path = output_root / "latent_metric_summary.csv"
    projection_frame.to_csv(projection_path, index=False)
    one_step_frame.to_csv(one_step_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    elapsed = time.perf_counter() - start_time
    print(
        "[latent_metric_confound] wrote "
        f"projection={projection_path} rows={len(projection_frame)} "
        f"one_step={one_step_path} rows={len(one_step_frame)} "
        f"summary={summary_path} rows={len(summary_frame)} elapsed_sec={elapsed:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether decoder-latent Adam fails due to coordinate metric mismatch or tangent geometry."
    )
    parser.add_argument("--run-dir", action="append", required=True, help="Experiment output directory. Repeat for multiple runs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16, help="Use first N eval starts when --source-index is not set. <=0 means all.")
    parser.add_argument("--source-index", action="append", type=int, default=[], help="Restrict to specific source_weight_index values.")
    parser.add_argument("--damping-rel", action="append", type=float, default=None)
    parser.add_argument("--multiplier", action="append", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--label-json", default="")
    args = parser.parse_args()
    if args.damping_rel is None:
        args.damping_rel = [1e-4]
    args.label = json.loads(args.label_json) if args.label_json else {}
    run(args)


if __name__ == "__main__":
    main()
