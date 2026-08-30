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
    torch_dtype,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    decode_weights,
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
    return ExperimentConfig(**values)


def _selected_lrs(output_dir: Path) -> dict[str, float]:
    selected = pd.read_csv(output_dir / "selected_lrs.csv")
    rows = selected[pd.to_numeric(selected["selected"], errors="coerce").fillna(0).astype(int) == 1]
    result = {str(row["method"]): float(row["candidate_lr"]) for _, row in rows.iterrows()}
    missing = {"raw", "decoder_latent"} - set(result)
    if missing:
        raise ValueError(f"{output_dir} selected_lrs.csv missing selected methods: {sorted(missing)}")
    return result


def _task_tensor_set(task_tensors, task_name: str):
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().double().flatten()
    b64 = b.detach().double().flatten()
    denom = a64.norm() * b64.norm()
    if float(denom.item()) <= 1e-30:
        return float("nan")
    return float((torch.dot(a64, b64) / denom).detach().cpu().item())


def _norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(float(den), 1e-30))


def _eval_losses(
    theta_raw: torch.Tensor,
    theta_latent: torch.Tensor,
    *,
    theta_decoded_raw: torch.Tensor | None = None,
    task_set,
    spec,
    tau: float,
) -> dict[str, float]:
    with torch.no_grad():
        raw_train_loss, raw_train_acc = _loss_acc(theta_raw, task_set=task_set, spec=spec, split="train", tau=tau)
        raw_test_loss, raw_test_acc = _loss_acc(theta_raw, task_set=task_set, spec=spec, split="test", tau=tau)
        lat_train_loss, lat_train_acc = _loss_acc(theta_latent, task_set=task_set, spec=spec, split="train", tau=tau)
        lat_test_loss, lat_test_acc = _loss_acc(theta_latent, task_set=task_set, spec=spec, split="test", tau=tau)
        if theta_decoded_raw is not None:
            dec_train_loss, dec_train_acc = _loss_acc(
                theta_decoded_raw,
                task_set=task_set,
                spec=spec,
                split="train",
                tau=tau,
            )
            dec_test_loss, dec_test_acc = _loss_acc(
                theta_decoded_raw,
                task_set=task_set,
                spec=spec,
                split="test",
                tau=tau,
            )
        else:
            dec_train_loss = dec_train_acc = dec_test_loss = dec_test_acc = None
    result = {
        "raw_train_loss": float(raw_train_loss.detach().cpu().item()),
        "raw_train_acc": float(raw_train_acc.detach().cpu().item()),
        "raw_test_loss": float(raw_test_loss.detach().cpu().item()),
        "raw_test_acc": float(raw_test_acc.detach().cpu().item()),
        "latent_train_loss": float(lat_train_loss.detach().cpu().item()),
        "latent_train_acc": float(lat_train_acc.detach().cpu().item()),
        "latent_test_loss": float(lat_test_loss.detach().cpu().item()),
        "latent_test_acc": float(lat_test_acc.detach().cpu().item()),
    }
    if theta_decoded_raw is not None:
        assert dec_train_loss is not None
        assert dec_train_acc is not None
        assert dec_test_loss is not None
        assert dec_test_acc is not None
        result.update(
            {
                "decoded_raw_train_loss": float(dec_train_loss.detach().cpu().item()),
                "decoded_raw_train_acc": float(dec_train_acc.detach().cpu().item()),
                "decoded_raw_test_loss": float(dec_test_loss.detach().cpu().item()),
                "decoded_raw_test_acc": float(dec_test_acc.detach().cpu().item()),
            }
        )
    return result


def _trace_start(
    *,
    cfg: ExperimentConfig,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    weights_device: torch.Tensor,
    weight_records: pd.DataFrame,
    task_tensors,
    spec,
    start_row: pd.Series,
    raw_lr: float,
    latent_lr: float,
    steps: int,
    record_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_weight_index = int(start_row["source_weight_index"])
    start_index = int(start_row["start_index"])
    record = weight_records.iloc[source_weight_index].to_dict()
    task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
    tau = float(record.get("tau", start_row.get("tau", 1.0)))
    task_set = _task_tensor_set(task_tensors, task_name)
    batch_size = int(getattr(cfg, "downstream_batch_size", 0))

    w0 = weights_device[source_weight_index].detach()
    w0_norm = _norm(w0)
    with torch.no_grad():
        z0 = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0).detach()
        theta_lat0 = decode_weights(vae, normalizer, z0.reshape(1, -1)).squeeze(0).detach()
    theta_raw_value = w0.detach().clone().requires_grad_(True)
    theta_decoded_raw_value = theta_lat0.detach().clone().requires_grad_(True)
    z_value = z0.detach().clone().requires_grad_(True)
    opt_raw = torch.optim.Adam([theta_raw_value], lr=float(raw_lr))
    opt_decoded_raw = torch.optim.Adam([theta_decoded_raw_value], lr=float(raw_lr))
    opt_latent = torch.optim.Adam([z_value], lr=float(latent_lr))

    prev_raw = theta_raw_value.detach().clone()
    prev_decoded_raw = theta_decoded_raw_value.detach().clone()
    prev_latent = theta_lat0.detach().clone()
    path_len_raw = 0.0
    path_len_decoded_raw = 0.0
    path_len_latent = 0.0
    raw_step_sq_sum = 0.0
    latent_step_sq_sum = 0.0
    step_dot_sum = 0.0
    velocity_cosines: list[float] = []
    rows: list[dict[str, Any]] = []

    def current_latent_theta() -> torch.Tensor:
        return decode_weights(vae, normalizer, z_value.reshape(1, -1)).squeeze(0)

    for step in range(int(steps) + 1):
        should_record = step == 0 or step % max(1, int(record_every)) == 0 or step == int(steps)
        if should_record:
            theta_raw = theta_raw_value.detach()
            theta_decoded_raw = theta_decoded_raw_value.detach()
            with torch.no_grad():
                theta_latent = current_latent_theta().detach()
            raw_disp = theta_raw - w0
            decoded_raw_disp = theta_decoded_raw - theta_lat0
            latent_disp_from_dec0 = theta_latent - theta_lat0
            latent_disp_from_w0 = theta_latent - w0
            raw_disp_norm = _norm(raw_disp)
            decoded_raw_disp_norm = _norm(decoded_raw_disp)
            latent_disp_norm = _norm(latent_disp_from_dec0)
            losses = _eval_losses(
                theta_raw,
                theta_latent,
                theta_decoded_raw=theta_decoded_raw,
                task_set=task_set,
                spec=spec,
                tau=tau,
            )
            rows.append(
                {
                    "source_weight_index": source_weight_index,
                    "start_index": start_index,
                    "task_name": task_name,
                    "tau": tau,
                    "step": int(step),
                    "raw_lr": float(raw_lr),
                    "latent_lr": float(latent_lr),
                    "w0_norm": w0_norm,
                    "raw_path_len": path_len_raw,
                    "decoded_raw_path_len": path_len_decoded_raw,
                    "latent_path_len": path_len_latent,
                    "raw_disp_norm": raw_disp_norm,
                    "decoded_raw_disp_from_dec0_norm": decoded_raw_disp_norm,
                    "decoded_raw_disp_from_w0_norm": _norm(theta_decoded_raw - w0),
                    "latent_disp_from_dec0_norm": latent_disp_norm,
                    "latent_disp_from_w0_norm": _norm(latent_disp_from_w0),
                    "raw_disp_rel_w0": _safe_ratio(raw_disp_norm, w0_norm),
                    "decoded_raw_disp_from_dec0_rel_w0": _safe_ratio(decoded_raw_disp_norm, w0_norm),
                    "decoded_raw_disp_from_w0_rel_w0": _safe_ratio(_norm(theta_decoded_raw - w0), w0_norm),
                    "latent_disp_from_dec0_rel_w0": _safe_ratio(latent_disp_norm, w0_norm),
                    "latent_disp_from_w0_rel_w0": _safe_ratio(_norm(latent_disp_from_w0), w0_norm),
                    "raw_decoded_raw_distance_norm": _norm(theta_decoded_raw - theta_raw),
                    "raw_decoded_raw_distance_rel_w0": _safe_ratio(_norm(theta_decoded_raw - theta_raw), w0_norm),
                    "raw_latent_distance_norm": _norm(theta_latent - theta_raw),
                    "raw_latent_distance_rel_w0": _safe_ratio(_norm(theta_latent - theta_raw), w0_norm),
                    "decoded_raw_latent_distance_norm": _norm(theta_latent - theta_decoded_raw),
                    "decoded_raw_latent_distance_rel_w0": _safe_ratio(_norm(theta_latent - theta_decoded_raw), w0_norm),
                    "start_reconstruction_rel_l2": _safe_ratio(_norm(theta_lat0 - w0), w0_norm),
                    "change_cos_raw_vs_decoded_raw": _cosine(raw_disp, decoded_raw_disp),
                    "change_cos_raw_vs_latent_from_dec0": _cosine(raw_disp, latent_disp_from_dec0),
                    "change_cos_decoded_raw_vs_latent_from_dec0": _cosine(decoded_raw_disp, latent_disp_from_dec0),
                    "position_cos_raw_vs_latent_from_w0": _cosine(raw_disp, latent_disp_from_w0),
                    "latent_change_projection_on_raw": float(
                        torch.dot(latent_disp_from_dec0.detach().double(), raw_disp.detach().double()).cpu().item()
                    ),
                    "latent_change_projection_ratio_on_raw": _safe_ratio(
                        float(torch.dot(latent_disp_from_dec0.detach().double(), raw_disp.detach().double()).cpu().item()),
                        raw_disp_norm * raw_disp_norm,
                    ),
                    "raw_tortuosity": _safe_ratio(path_len_raw, raw_disp_norm),
                    "latent_tortuosity": _safe_ratio(path_len_latent, latent_disp_norm),
                    "velocity_cosine_mean_so_far": float(pd.Series(velocity_cosines).mean()) if velocity_cosines else float("nan"),
                    "velocity_global_cosine_so_far": _safe_ratio(
                        step_dot_sum,
                        math.sqrt(max(raw_step_sq_sum, 0.0)) * math.sqrt(max(latent_step_sq_sum, 0.0)),
                    ),
                    **losses,
                    "delta_train_loss_latent_minus_raw": losses["latent_train_loss"] - losses["raw_train_loss"],
                    "delta_test_loss_latent_minus_raw": losses["latent_test_loss"] - losses["raw_test_loss"],
                    "delta_test_acc_latent_minus_raw": losses["latent_test_acc"] - losses["raw_test_acc"],
                    "delta_train_loss_decoded_raw_minus_raw": losses["decoded_raw_train_loss"] - losses["raw_train_loss"],
                    "delta_test_loss_decoded_raw_minus_raw": losses["decoded_raw_test_loss"] - losses["raw_test_loss"],
                    "delta_train_loss_latent_minus_decoded_raw": losses["latent_train_loss"]
                    - losses["decoded_raw_train_loss"],
                    "delta_test_loss_latent_minus_decoded_raw": losses["latent_test_loss"]
                    - losses["decoded_raw_test_loss"],
                }
            )
        if step == int(steps):
            break

        batch_indices = _batch_indices(task_set, batch_size=batch_size, step=step, start_index=start_index)
        opt_raw.zero_grad(set_to_none=True)
        raw_loss, _ = _loss_acc(theta_raw_value, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
        raw_loss.backward()
        opt_raw.step()

        opt_decoded_raw.zero_grad(set_to_none=True)
        decoded_raw_loss, _ = _loss_acc(
            theta_decoded_raw_value,
            task_set=task_set,
            spec=spec,
            split="train",
            tau=tau,
            batch_indices=batch_indices,
        )
        decoded_raw_loss.backward()
        opt_decoded_raw.step()

        opt_latent.zero_grad(set_to_none=True)
        theta_latent_train = current_latent_theta()
        latent_loss, _ = _loss_acc(theta_latent_train, task_set=task_set, spec=spec, split="train", tau=tau, batch_indices=batch_indices)
        latent_loss.backward()
        opt_latent.step()

        with torch.no_grad():
            new_raw = theta_raw_value.detach().clone()
            new_decoded_raw = theta_decoded_raw_value.detach().clone()
            new_latent = current_latent_theta().detach().clone()
        raw_step = new_raw - prev_raw
        decoded_raw_step = new_decoded_raw - prev_decoded_raw
        latent_step = new_latent - prev_latent
        raw_step_norm = _norm(raw_step)
        decoded_raw_step_norm = _norm(decoded_raw_step)
        latent_step_norm = _norm(latent_step)
        path_len_raw += raw_step_norm
        path_len_decoded_raw += decoded_raw_step_norm
        path_len_latent += latent_step_norm
        raw_step_sq_sum += raw_step_norm * raw_step_norm
        latent_step_sq_sum += latent_step_norm * latent_step_norm
        step_dot = float(torch.dot(raw_step.detach().double(), latent_step.detach().double()).cpu().item())
        step_dot_sum += step_dot
        if raw_step_norm > 1e-30 and latent_step_norm > 1e-30:
            velocity_cosines.append(step_dot / (raw_step_norm * latent_step_norm))
        prev_raw = new_raw
        prev_decoded_raw = new_decoded_raw
        prev_latent = new_latent

    final = rows[-1].copy()
    return rows, final


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    step_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    print(
        "[weight_path] start "
        f"runs={len(args.run_dir)} device={args.device} steps={args.steps} samples={args.samples} "
        f"record_every={args.record_every} output_root={output_root}",
        flush=True,
    )
    for run_dir_value in args.run_dir:
        run_dir_path = Path(run_dir_value).expanduser().resolve()
        run_name = run_dir_path.name
        cfg = _load_cfg(run_dir_path, device=str(args.device))
        device = torch.device(cfg.device)
        dtype = torch_dtype(cfg)
        print(
            "[weight_path] run "
            f"name={run_name} dir={run_dir_path} config_hash={config_hash(cfg)} device={device} dtype={dtype}",
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
        vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=dtype).eval()
        vae.load_state_dict(vae_payload["model_state"])
        weights_device = weights.to(device=device, dtype=dtype)
        task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=device, dtype=dtype)
        selected_lrs = _selected_lrs(run_dir_path)
        results = pd.read_csv(run_dir_path / "downstream_results.csv")
        starts = results[(results["split"].astype(str) == "eval") & (results["method"].astype(str) == "decoder_latent")]
        starts = starts.sort_values("start_index").reset_index(drop=True)
        if int(args.samples) > 0:
            starts = starts.iloc[: int(args.samples)].copy()
        print(
            "[weight_path] starts "
            f"count={len(starts)} raw_lr={selected_lrs['raw']:.6g} latent_lr={selected_lrs['decoder_latent']:.6g} "
            f"indices={starts['source_weight_index'].astype(int).tolist()}",
            flush=True,
        )
        for pos, start_row in starts.iterrows():
            print(
                "[weight_path] trace "
                f"run={run_name} start={pos + 1}/{len(starts)} source={int(start_row['source_weight_index'])} "
                f"task={start_row['task_name']} tau={float(start_row['tau']):.6g}",
                flush=True,
            )
            rows, final = _trace_start(
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
                steps=int(args.steps),
                record_every=int(args.record_every),
            )
            for row in rows:
                row["run_name"] = run_name
                row["variant_label"] = str(args.label.get(run_name, run_name)) if isinstance(args.label, dict) else run_name
            final["run_name"] = run_name
            final["variant_label"] = str(args.label.get(run_name, run_name)) if isinstance(args.label, dict) else run_name
            step_rows.extend(rows)
            summary_rows.append(final)
    steps_frame = pd.DataFrame(step_rows)
    summary_frame = pd.DataFrame(summary_rows)
    steps_path = output_root / "downstream_weight_path_steps.csv"
    summary_path = output_root / "downstream_weight_path_summary.csv"
    steps_frame.to_csv(steps_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    elapsed = time.perf_counter() - start_time
    print(
        "[weight_path] wrote "
        f"steps={steps_path} rows={len(steps_frame)} summary={summary_path} rows={len(summary_frame)} "
        f"elapsed_sec={elapsed:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw Adam and decoder-latent Adam trajectories in weight space.")
    parser.add_argument("--run-dir", action="append", required=True, help="Experiment output directory. Repeat for multiple runs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--record-every", type=int, default=25)
    parser.add_argument("--label-json", default="")
    args = parser.parse_args()
    args.label = json.loads(args.label_json) if args.label_json else {}
    run(args)


if __name__ == "__main__":
    main()
