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
    build_weight_vae,
    decode_weights,
    encode_weights,
    load_celo_meta_task_tensors,
    load_torch_cache,
    move_task_tensors,
    spec_from_payload,
)
from scripts.audit_reparam_cg import (
    _batch_indices,
    _cg_solve,
    _hvp_x,
    _load_cfg,
    _load_start_bank,
    _tensor_sha256,
    _x_probe,
    _z_probe_like,
)


ARTIFACT_ROOT = Path("artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing").resolve()

SUPPORTED_METHODS = (
    "decoder_latent",
    "decoder_latent_raw_classifier_head_clamp",
    "decoder_latent_raw_fc2_weight_clamp",
    "decoder_latent_raw_fc2_bias_clamp",
)


def _log(message: str) -> None:
    print(f"[clamped_cg_audit] {message}", flush=True)


def _run_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    return (ARTIFACT_ROOT / str(value)).resolve()


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _flat_spec_slices(spec: FlatSpec) -> dict[str, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    for key, size in zip(spec.keys, spec.sizes, strict=True):
        slices[str(key)] = slice(offset, offset + int(size))
        offset += int(size)
    return slices


def _method_block_keys(method: str, spec: FlatSpec) -> tuple[str, ...]:
    method = str(method)
    keys = tuple(str(key) for key in spec.keys)
    if method == "decoder_latent":
        return ()
    if method == "decoder_latent_raw_classifier_head_clamp":
        selected = tuple(key for key in keys if key.startswith("fc2."))
    elif method == "decoder_latent_raw_fc2_weight_clamp":
        selected = tuple(key for key in keys if key == "fc2.weight")
    elif method == "decoder_latent_raw_fc2_bias_clamp":
        selected = tuple(key for key in keys if key == "fc2.bias")
    else:
        raise ValueError(f"unknown method={method!r}; supported={SUPPORTED_METHODS}")
    if not selected:
        raise ValueError(f"method={method!r} selected no keys from spec keys={keys}")
    return selected


def _method_block_name(method: str) -> str:
    if method == "decoder_latent_raw_classifier_head_clamp":
        return "classifier_head"
    if method == "decoder_latent_raw_fc2_weight_clamp":
        return "fc2.weight"
    if method == "decoder_latent_raw_fc2_bias_clamp":
        return "fc2.bias"
    return ""


def _splice_flat_blocks(base: torch.Tensor, donor: torch.Tensor, spec: FlatSpec, block_keys: tuple[str, ...]) -> torch.Tensor:
    if not block_keys:
        return base
    result = base.clone()
    slices = _flat_spec_slices(spec)
    donor_detached = donor.detach()
    for key in block_keys:
        result[slices[key]] = donor_detached[slices[key]]
    return result


def _select_flat_blocks(values: torch.Tensor, spec: FlatSpec, block_keys: tuple[str, ...]) -> torch.Tensor:
    slices = _flat_spec_slices(spec)
    return torch.cat([values[slices[key]] for key in block_keys], dim=0)


def _zero_flat_blocks(values: torch.Tensor, spec: FlatSpec, block_keys: tuple[str, ...]) -> torch.Tensor:
    if not block_keys:
        return values
    result = values.clone()
    slices = _flat_spec_slices(spec)
    for key in block_keys:
        result[slices[key]] = 0.0
    return result


def _spec_signature(spec: FlatSpec) -> dict[str, Any]:
    return {
        "dim": int(spec.dim),
        "keys": [str(v) for v in spec.keys],
        "sizes": [int(v) for v in spec.sizes],
        "shapes": [[int(x) for x in shape] for shape in spec.shapes],
    }


def _records_signature(records: pd.DataFrame, source_indices: list[int]) -> tuple[str, list[dict[str, Any]]]:
    columns = ["source_weight_index", "task_name", "tau", "run", "step", "source_lr", "optimizer", "weight_distribution"]
    selected = records.iloc[source_indices].copy().reset_index(drop=True)
    frame = pd.DataFrame({"source_weight_index": [int(v) for v in source_indices]})
    for column in columns[1:]:
        if column in selected.columns:
            frame[column] = selected[column].tolist()
        else:
            frame[column] = ""
    text = frame.to_json(orient="records", double_precision=15)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), json.loads(text)


def _selected_run_identity(run_dir: Path, *, label: str, start_bank: pd.DataFrame) -> dict[str, Any]:
    weight_payload = load_torch_cache(run_dir / "weight_pool.pt")
    if weight_payload is None or not isinstance(weight_payload.get("weights"), torch.Tensor):
        raise FileNotFoundError(run_dir / "weight_pool.pt")
    weights = weight_payload["weights"].detach().cpu()
    records = pd.DataFrame(weight_payload.get("records", []))
    records_path = run_dir / "weight_pool_records.csv"
    if records.empty and records_path.is_file():
        records = pd.read_csv(records_path)
    if records.empty:
        raise RuntimeError(f"empty weight records for {run_dir}")
    spec = spec_from_payload(weight_payload["spec"])
    source_indices = [int(v) for v in start_bank["source_weight_index"].astype(int).tolist()]
    selected = weights.index_select(0, torch.as_tensor(source_indices, dtype=torch.long)).contiguous()
    records_hash, records_preview = _records_signature(records, source_indices)
    return {
        "label": str(label),
        "run_dir": str(run_dir),
        "weight_cache_key": str(weight_payload.get("cache_key", "")),
        "spec_signature": _spec_signature(spec),
        "selected_weight_sha256": _tensor_sha256(selected),
        "selected_records_sha256": records_hash,
        "selected_records_preview": records_preview,
    }


def _identity_validation(identities: list[dict[str, Any]]) -> dict[str, Any]:
    if not identities:
        return {"accepted": False, "mismatches": ["no identities"], "by_label": {}}
    base = identities[0]
    mismatches: list[str] = []
    for item in identities[1:]:
        label = str(item["label"])
        for key in ["weight_cache_key", "spec_signature", "selected_weight_sha256", "selected_records_sha256"]:
            if item.get(key) != base.get(key):
                mismatches.append(f"{label}:{key}")
    return {
        "accepted": len(mismatches) == 0,
        "mismatches": mismatches,
        "by_label": {str(item["label"]): item for item in identities},
    }


def _parse_probe_spaces(value: str) -> list[str]:
    normalized = str(value).strip().lower()
    if normalized == "both":
        return ["full", "active"]
    if normalized in {"full", "active"}:
        return [normalized]
    raise ValueError("--probe-space must be one of full, active, both")


def _stable_generator(*, seed: int, source_weight_index: int, probe_id: int, purpose: str) -> torch.Generator:
    payload = f"{int(seed)}:{int(source_weight_index)}:{int(probe_id)}:{purpose}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(value)
    return generator


def _bootstrap_ci(values: np.ndarray, *, seed: int = 0, reps: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class EffectiveDecoderMatvec:
    def __init__(
        self,
        *,
        vae: torch.nn.Module,
        normalizer: WeightNormalizer,
        z: torch.Tensor,
        donor: torch.Tensor,
        spec: FlatSpec,
        block_keys: tuple[str, ...],
    ) -> None:
        self.vae = vae
        self.normalizer = normalizer
        self.z = z.detach()
        self.donor = donor.detach()
        self.spec = spec
        self.block_keys = tuple(block_keys)

    def decode_single(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = decode_weights(self.vae, self.normalizer, latent.reshape(1, -1)).squeeze(0)
        return _splice_flat_blocks(decoded, self.donor, self.spec, self.block_keys)

    def jt_vec(self, q: torch.Tensor) -> torch.Tensor:
        z_req = self.z.detach().clone().requires_grad_(True)
        decoded = self.decode_single(z_req)
        (jtq,) = torch.autograd.grad(
            (decoded * q.detach()).sum(),
            z_req,
            retain_graph=False,
            create_graph=False,
        )
        return jtq.detach()

    def j_vec(self, a: torch.Tensor) -> torch.Tensor:
        _decoded, ja = torch.autograd.functional.jvp(
            self.decode_single,
            self.z.detach(),
            v=a.detach(),
            create_graph=False,
            strict=False,
        )
        return ja.detach()

    def a_lambda(self, q: torch.Tensor, lambda_abs: float) -> torch.Tensor:
        jtq = self.jt_vec(q)
        return self.j_vec(jtq) + float(lambda_abs) * q

    def trace_jtj_per_dim(self, *, generator: torch.Generator, probes: int) -> float:
        values: list[float] = []
        for _idx in range(max(1, int(probes))):
            u = _z_probe_like(self.z, generator=generator)
            ju = self.j_vec(u)
            values.append(float((ju.float().square().sum() / float(max(1, int(u.numel())))).detach().cpu().item()))
        return float(sum(values) / max(1, len(values)))


def _load_run_artifacts(run_dir: Path, *, device: torch.device, dtype: torch.dtype, cfg: Any) -> dict[str, Any]:
    weight_payload = load_torch_cache(run_dir / "weight_pool.pt")
    if weight_payload is None or not isinstance(weight_payload.get("weights"), torch.Tensor):
        raise FileNotFoundError(run_dir / "weight_pool.pt")
    vae_payload = load_torch_cache(run_dir / "vae_checkpoint.pt")
    if vae_payload is None:
        raise FileNotFoundError(run_dir / "vae_checkpoint.pt")
    weights = weight_payload["weights"].detach().cpu()
    weight_records = pd.DataFrame(weight_payload.get("records", []))
    records_path = run_dir / "weight_pool_records.csv"
    if weight_records.empty and records_path.is_file():
        weight_records = pd.read_csv(records_path)
    if weight_records.empty:
        raise RuntimeError(f"empty weight records for {run_dir}")
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=dtype).eval()
    vae.load_state_dict(vae_payload["model_state"])
    return {
        "weights": weights.to(device=device, dtype=dtype),
        "weight_records": weight_records.reset_index(drop=True),
        "spec": spec,
        "normalizer": normalizer,
        "vae": vae,
        "weight_cache_key": str(weight_payload.get("cache_key", "")),
        "vae_cache_key": str(vae_payload.get("cache_key", "")),
    }


def _parse_methods(values: list[str]) -> list[str]:
    methods: list[str] = []
    for value in values:
        for item in str(value).split(","):
            method = item.strip()
            if method:
                methods.append(method)
    if not methods:
        methods = list(SUPPORTED_METHODS)
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"unknown methods={unknown}; supported={list(SUPPORTED_METHODS)}")
    return list(dict.fromkeys(methods))


def _lambda_values(args: argparse.Namespace, *, trace_jtj_per_dim: float) -> list[float]:
    if str(args.lambdas).strip():
        return [float(v) for v in str(args.lambdas).split(",") if str(v).strip()]
    rel_values = [float(v) for v in str(args.lambda_rel).split(",") if str(v).strip()]
    return [max(float(args.min_lambda), rel * max(float(trace_jtj_per_dim), float(args.min_lambda))) for rel in rel_values]


def _audit_one_label(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    label: str,
    methods: list[str],
    start_bank: pd.DataFrame,
    start_bank_path: Path,
    start_bank_hash: str,
) -> pd.DataFrame:
    cfg = _load_cfg(run_dir, device=str(args.device), batch_size=int(args.batch_size))
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    artifacts = _load_run_artifacts(run_dir, device=device, dtype=dtype, cfg=cfg)
    weights: torch.Tensor = artifacts["weights"]
    weight_records: pd.DataFrame = artifacts["weight_records"]
    spec: FlatSpec = artifacts["spec"]
    vae: torch.nn.Module = artifacts["vae"]
    normalizer: WeightNormalizer = artifacts["normalizer"]
    task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=device, dtype=dtype)

    _log(
        "label_start "
        f"label={label} run_dir={run_dir} config_hash={config_hash(cfg)} "
        f"device={device} dtype={dtype} methods={methods} starts={len(start_bank)} "
        f"probes={int(args.probes)} batch_size={int(args.batch_size)}"
    )
    _log(f"label_config label={label} resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}")
    _log(
        "label_artifacts "
        f"label={label} weights_shape={tuple(weights.shape)} spec_dim={int(spec.dim)} "
        f"latent_dim={int(cfg.latent_dim)} weight_cache_key={artifacts['weight_cache_key']} "
        f"vae_cache_key={artifacts['vae_cache_key']}"
    )

    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    probe_spaces = _parse_probe_spaces(str(args.probe_space))
    total = len(methods) * len(start_bank) * int(args.probes) * len(probe_spaces)
    completed = 0
    for method in methods:
        block_keys = _method_block_keys(method, spec)
        block_name = _method_block_name(method)
        for start_pos, start_row in start_bank.iterrows():
            source_weight_index = int(start_row["source_weight_index"])
            record = weight_records.iloc[source_weight_index].to_dict()
            task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
            tau = float(record.get("tau", start_row.get("tau", 1.0)))
            task_set = task_tensors[task_name]
            w0 = weights[source_weight_index].detach()
            with torch.no_grad():
                z = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0)
                decoded_plain = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0).detach()
                theta_bar = _splice_flat_blocks(decoded_plain, w0, spec, block_keys).detach()
            decoder_ops = EffectiveDecoderMatvec(
                vae=vae,
                normalizer=normalizer,
                z=z,
                donor=w0,
                spec=spec,
                block_keys=block_keys,
            )
            scale_generator = _stable_generator(seed=int(args.seed), source_weight_index=source_weight_index, probe_id=0, purpose=f"scale:{method}")
            trace_jtj_per_dim = decoder_ops.trace_jtj_per_dim(generator=scale_generator, probes=int(args.scale_probes))
            lambdas = _lambda_values(args, trace_jtj_per_dim=trace_jtj_per_dim)
            preserved_dim = 0
            preserved_rel_to_donor = float("nan")
            splice_delta_rel_l2_to_decoded = float("nan")
            preserved_jvp_block_norm = float("nan")
            preserved_jt_block_norm = float("nan")
            if block_keys:
                donor_block = _select_flat_blocks(w0, spec, block_keys)
                theta_block = _select_flat_blocks(theta_bar, spec, block_keys)
                preserved_dim = int(donor_block.numel())
                preserved_rel_to_donor = float(
                    ((theta_block - donor_block).float().norm() / donor_block.float().norm().clamp_min(1e-12)).detach().cpu().item()
                )
                splice_delta_rel_l2_to_decoded = float(
                    ((theta_bar - decoded_plain).float().norm() / decoded_plain.float().norm().clamp_min(1e-12)).detach().cpu().item()
                )
                if bool(args.jacobian_check):
                    check_generator = _stable_generator(
                        seed=int(args.seed),
                        source_weight_index=source_weight_index,
                        probe_id=0,
                        purpose=f"jacobian_check:{method}",
                    )
                    u = _z_probe_like(z, generator=check_generator)
                    ju = decoder_ops.j_vec(u)
                    preserved_jvp_block_norm = float(_select_flat_blocks(ju, spec, block_keys).float().norm().detach().cpu().item())
                    q = torch.zeros_like(w0)
                    slices = _flat_spec_slices(spec)
                    for key in block_keys:
                        block_noise = torch.randn(
                            (slices[key].stop - slices[key].start,),
                            generator=check_generator,
                            device="cpu",
                            dtype=torch.float32,
                        ).to(device=w0.device, dtype=w0.dtype)
                        q[slices[key]] = block_noise
                    preserved_jt_block_norm = float(decoder_ops.jt_vec(q).float().norm().detach().cpu().item())
            reconstruction_rel_l2 = float(((theta_bar - w0).float().norm() / w0.float().norm().clamp_min(1e-12)).detach().cpu().item())
            _log(
                "start "
                f"label={label} method={method} source={source_weight_index} "
                f"task={task_name} tau={tau:.6g} block={block_name or 'none'} "
                f"preserved_dim={preserved_dim} trace_jtj_per_dim={trace_jtj_per_dim:.6g} "
                f"lambdas={','.join(f'{v:.6g}' for v in lambdas)}"
            )
            for probe_id in range(int(args.probes)):
                batch_idx = _batch_indices(
                    int(task_set.train_labels.shape[0]),
                    batch_size=int(args.batch_size),
                    source_weight_index=source_weight_index,
                    probe_id=probe_id,
                )
                if batch_idx is None:
                    images = task_set.train_images
                    labels = task_set.train_labels
                else:
                    batch_idx = batch_idx.to(device=device)
                    images = task_set.train_images.index_select(0, batch_idx)
                    labels = task_set.train_labels.index_select(0, batch_idx)
                probe_generator = _stable_generator(
                    seed=int(args.seed),
                    source_weight_index=source_weight_index,
                    probe_id=probe_id,
                    purpose="x_probe",
                )
                base_v = _x_probe(int(spec.dim), generator=probe_generator, device=device, dtype=dtype, kind=str(args.probe_kind))
                base_probe_hash = _tensor_sha256(base_v)
                batch_hash = _tensor_sha256(batch_idx)
                for probe_space in probe_spaces:
                    if probe_space == "active":
                        v = _zero_flat_blocks(base_v, spec, block_keys)
                    else:
                        v = base_v
                    probe_hash = _tensor_sha256(v)
                    h_x = _hvp_x(theta_bar, v, images=images, labels=labels, spec=spec, tau=tau)
                    jt_h = decoder_ops.jt_vec(h_x)
                    c3_forward = float(jt_h.float().square().sum().detach().cpu().item())
                    probe_norm2 = float(v.float().square().sum().detach().cpu().item())
                    base_probe_norm2 = float(base_v.float().square().sum().detach().cpu().item())
                    if block_keys:
                        preserved_probe_norm2 = float(_select_flat_blocks(v, spec, block_keys).float().square().sum().detach().cpu().item())
                    else:
                        preserved_probe_norm2 = 0.0
                    active_probe_norm2 = float(max(0.0, probe_norm2 - preserved_probe_norm2))
                    for lambda_abs in lambdas:
                        cg = _cg_solve(
                            decoder_ops.a_lambda,
                            v,
                            lambda_abs=float(lambda_abs),
                            max_iters=int(args.cg_max_iters),
                            tol=float(args.cg_tol),
                            init=str(args.cg_init),
                        )
                        w = cg.pop("w")
                        jt_w = decoder_ops.jt_vec(w)
                        inverse_preserved_vtw = float(preserved_probe_norm2 / max(float(lambda_abs), 1e-30))
                        inverse_active_vtw = float(float(cg["vTw"]) - inverse_preserved_vtw)
                        effective_rank_bound = int(min(max(0, int(spec.dim) - int(preserved_dim)), int(cfg.latent_dim)))
                        row = {
                            "label": str(label),
                            "run_dir": str(run_dir),
                            "method": str(method),
                            "block_preserve_block": str(block_name),
                            "block_preserve_keys": ";".join(block_keys),
                            "block_preserve_donor": "raw_w0" if block_keys else "",
                            "source_weight_index": int(source_weight_index),
                            "start_bank_position": int(start_row.get("start_bank_position", start_pos)),
                            "start_role": str(start_row.get("start_role", "")),
                            "selection": str(start_row.get("selection", "")),
                            "audit_seed": int(args.seed),
                            "start_bank_path": str(start_bank_path),
                            "start_bank_sha256": str(start_bank_hash),
                            "source_run": int(record.get("run", -1)),
                            "source_step": int(record.get("step", -1)),
                            "source_lr": float(record.get("source_lr", float("nan"))),
                            "task_name": task_name,
                            "tau": float(tau),
                            "probe_id": int(probe_id),
                            "probe_kind": str(args.probe_kind),
                            "probe_space": str(probe_space),
                            "base_probe_sha256": str(base_probe_hash),
                            "probe_sha256": str(probe_hash),
                            "batch_indices_sha256": str(batch_hash),
                            "batch_size": int(images.shape[0]),
                            "lambda_abs": float(lambda_abs),
                            "lambda_rel_to_trace_jtj": float(lambda_abs / max(trace_jtj_per_dim, 1e-30)),
                            "trace_jtj_per_dim": float(trace_jtj_per_dim),
                            "rank_bound": int(min(int(spec.dim), int(cfg.latent_dim))),
                            "x_dim": int(spec.dim),
                            "z_dim": int(cfg.latent_dim),
                            "preserved_dim": int(preserved_dim),
                            "effective_rank_bound": int(effective_rank_bound),
                            "expected_null_fraction": float(max(0.0, 1.0 - float(cfg.latent_dim) / float(spec.dim))),
                            "effective_expected_null_fraction": float(max(0.0, 1.0 - float(effective_rank_bound) / float(spec.dim))),
                            "decoded_reconstruction_rel_l2": float(
                                ((decoded_plain - w0).float().norm() / w0.float().norm().clamp_min(1e-12)).detach().cpu().item()
                            ),
                            "post_splice_reconstruction_rel_l2": float(reconstruction_rel_l2),
                            "preserved_block_rel_l2_to_donor": float(preserved_rel_to_donor),
                            "splice_delta_rel_l2_to_decoded": float(splice_delta_rel_l2_to_decoded),
                            "preserved_jvp_block_norm": float(preserved_jvp_block_norm),
                            "preserved_jt_block_norm": float(preserved_jt_block_norm),
                            "base_probe_norm2": float(base_probe_norm2),
                            "probe_norm2": float(probe_norm2),
                            "preserved_probe_norm2": float(preserved_probe_norm2),
                            "active_probe_norm2": float(active_probe_norm2),
                            "hvp_norm2": float(h_x.float().square().sum().detach().cpu().item()),
                            "jt_h_norm2": float(c3_forward),
                            "jt_w_norm2": float(jt_w.float().square().sum().detach().cpu().item()),
                            "c3_forward": float(c3_forward),
                            "c3_inverse": float(cg["vTw"]),
                            "c3_inverse_preserved_analytic": float(inverse_preserved_vtw),
                            "c3_inverse_active": float(inverse_active_vtw),
                            "c3_total": float(c3_forward + float(cg["vTw"])),
                            "lambda_vtw_ratio": float(float(lambda_abs) * float(cg["vTw"]) / max(probe_norm2, 1e-30)),
                            "lambda_inverse_preserved_ratio": float(float(lambda_abs) * inverse_preserved_vtw / max(probe_norm2, 1e-30)),
                            "lambda_inverse_active_ratio": float(float(lambda_abs) * inverse_active_vtw / max(probe_norm2, 1e-30)),
                            **cg,
                        }
                        rows.append(row)
                        _log(
                            "metric "
                            f"label={label} method={method} space={probe_space} source={source_weight_index} probe={probe_id} "
                            f"lambda={float(lambda_abs):.6g} iters={int(row['cg_iterations'])} "
                            f"rel_res={float(row['cg_final_rel_residual']):.3g} "
                            f"c3_forward={c3_forward:.6g} c3_inverse={float(cg['vTw']):.6g}"
                        )
                    completed += 1
                    if completed % max(1, min(10, total)) == 0:
                        elapsed = time.perf_counter() - start_time
                        _log(f"progress label={label} completed_probe_units={completed}/{total} elapsed_sec={elapsed:.1f}")
    frame = pd.DataFrame(rows)
    elapsed = time.perf_counter() - start_time
    _log(f"label_done label={label} rows={len(frame)} elapsed_sec={elapsed:.1f}")
    return frame


def _paired_outputs(
    combined: pd.DataFrame,
    *,
    labels: list[str],
    out_dir: Path,
    identity: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(labels) != 2:
        return pd.DataFrame(), pd.DataFrame(), {}
    control_label, a_label = labels
    key_cols = [
        "method",
        "source_weight_index",
        "task_name",
        "tau",
        "probe_id",
        "probe_kind",
        "probe_space",
        "probe_sha256",
        "batch_indices_sha256",
        "lambda_abs",
    ]
    control = combined[combined["label"].astype(str) == str(control_label)].copy()
    a_rows = combined[combined["label"].astype(str) == str(a_label)].copy()
    outer = control.merge(a_rows, on=key_cols, suffixes=("_control", "_a"), how="outer", validate="one_to_one", indicator=True)
    unpaired = outer[outer["_merge"].astype(str) != "both"].copy()
    unpaired.to_csv(out_dir / "unpaired_clamped_cg_keys.csv", index=False)
    paired = outer[outer["_merge"].astype(str) == "both"].drop(columns=["_merge"]).copy()
    metric_cols = [
        "trace_jtj_per_dim",
        "decoded_reconstruction_rel_l2",
        "post_splice_reconstruction_rel_l2",
        "preserved_jvp_block_norm",
        "preserved_jt_block_norm",
        "base_probe_norm2",
        "probe_norm2",
        "preserved_probe_norm2",
        "active_probe_norm2",
        "hvp_norm2",
        "jt_h_norm2",
        "jt_w_norm2",
        "c3_forward",
        "c3_inverse",
        "c3_inverse_preserved_analytic",
        "c3_inverse_active",
        "c3_total",
        "lambda_vtw_ratio",
        "lambda_inverse_preserved_ratio",
        "lambda_inverse_active_ratio",
        "cg_final_rel_residual",
        "cg_iterations",
    ]
    summary_rows: list[dict[str, Any]] = []
    for metric in metric_cols:
        a_col = f"{metric}_a"
        c_col = f"{metric}_control"
        if a_col not in paired.columns or c_col not in paired.columns:
            continue
        delta_col = f"{metric}_delta"
        rel_col = f"{metric}_rel_delta"
        paired[delta_col] = paired[a_col] - paired[c_col]
        paired[rel_col] = paired[delta_col] / paired[c_col].abs().clip(lower=1e-30)
        for (method, probe_space), sub in paired.groupby(["method", "probe_space"], dropna=False):
            values = sub[delta_col].to_numpy(dtype=np.float64)
            rel_values = sub[rel_col].to_numpy(dtype=np.float64)
            lo, hi = _bootstrap_ci(values, seed=101 + len(summary_rows))
            rel_lo, rel_hi = _bootstrap_ci(rel_values, seed=701 + len(summary_rows))
            summary_rows.append(
                {
                    "method": str(method),
                    "probe_space": str(probe_space),
                    "metric": metric,
                    "n": int(values.size),
                    "mean_delta": float(np.mean(values)),
                    "median_delta": float(np.median(values)),
                    "mean_rel_delta": float(np.mean(rel_values)),
                    "median_rel_delta": float(np.median(rel_values)),
                    "better_count_A_lt_control": int(np.sum(values < 0.0)),
                    "worse_count_A_gt_control": int(np.sum(values > 0.0)),
                    "bootstrap95_delta_low": float(lo),
                    "bootstrap95_delta_high": float(hi),
                    "bootstrap95_rel_low": float(rel_lo),
                    "bootstrap95_rel_high": float(rel_hi),
                    "min_delta": float(np.min(values)),
                    "max_delta": float(np.max(values)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    paired_path = out_dir / "paired_clamped_cg_deltas.csv"
    summary_path = out_dir / "paired_clamped_cg_summary.csv"
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    validation = _validation(combined, paired, unpaired, labels=labels, identity=identity, args=args, key_cols=key_cols)
    (out_dir / "clamped_cg_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    return paired, summary, validation


def _validation(
    combined: pd.DataFrame,
    paired: pd.DataFrame,
    unpaired: pd.DataFrame,
    *,
    labels: list[str],
    identity: dict[str, Any],
    args: argparse.Namespace,
    key_cols: list[str],
) -> dict[str, Any]:
    finite_cols = ["c3_forward", "c3_inverse", "c3_total", "trace_jtj_per_dim", "cg_final_rel_residual"]
    numeric = combined[finite_cols].apply(pd.to_numeric, errors="coerce") if not combined.empty else pd.DataFrame()
    residuals = pd.to_numeric(combined.get("cg_final_rel_residual", pd.Series(dtype=float)), errors="coerce")
    preserve = combined[combined["block_preserve_keys"].fillna("").astype(str) != ""].copy() if "block_preserve_keys" in combined.columns else pd.DataFrame()
    preserve_errors = pd.to_numeric(preserve.get("preserved_block_rel_l2_to_donor", pd.Series(dtype=float)), errors="coerce")
    jvp_errors = pd.to_numeric(preserve.get("preserved_jvp_block_norm", pd.Series(dtype=float)), errors="coerce")
    jt_errors = pd.to_numeric(preserve.get("preserved_jt_block_norm", pd.Series(dtype=float)), errors="coerce")
    same_base_probe = True
    same_effective_probe_within_method = True
    same_batch = True
    if not combined.empty:
        for _keys, sub in combined.groupby(["source_weight_index", "probe_id"], dropna=False):
            if "base_probe_sha256" in sub and sub["base_probe_sha256"].astype(str).nunique(dropna=False) != 1:
                same_base_probe = False
            if sub["batch_indices_sha256"].astype(str).nunique(dropna=False) != 1:
                same_batch = False
        for _keys, sub in combined.groupby(["method", "source_weight_index", "probe_id", "probe_space"], dropna=False):
            if sub["probe_sha256"].astype(str).nunique(dropna=False) != 1:
                same_effective_probe_within_method = False
    label_key_counts: dict[str, int] = {}
    if len(labels) == 2 and not combined.empty:
        for label in labels:
            sub = combined[combined["label"].astype(str) == str(label)].copy()
            label_key_counts[str(label)] = int(sub[key_cols].drop_duplicates().shape[0])
    expected_paired = min(label_key_counts.values()) if label_key_counts else 0
    residual_gate = float(args.cg_residual_gate)
    preserve_gate = float(args.preserved_gate)
    jacobian_gate = float(args.jacobian_gate)
    residual_ok = bool(not residuals.empty and float(residuals.max()) <= residual_gate)
    preserve_ok = bool(preserve_errors.empty or float(preserve_errors.max()) <= preserve_gate)
    jacobian_ok = bool(
        (jvp_errors.empty or float(jvp_errors.fillna(0.0).max()) <= jacobian_gate)
        and (jt_errors.empty or float(jt_errors.fillna(0.0).max()) <= jacobian_gate)
    )
    paired_ok = bool(len(unpaired) == 0 and int(len(paired)) == int(expected_paired) and len(set(label_key_counts.values())) <= 1)
    accepted = bool(
        bool(identity.get("accepted", False))
        and paired_ok
        and (bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all()) if not numeric.empty else False)
        and residual_ok
        and preserve_ok
        and jacobian_ok
        and same_base_probe
        and same_effective_probe_within_method
        and same_batch
    )
    return {
        "accepted": bool(accepted),
        "labels": [str(v) for v in labels],
        "combined_rows": int(len(combined)),
        "paired_rows": int(len(paired)),
        "unpaired_rows": int(len(unpaired)),
        "label_key_counts": label_key_counts,
        "expected_paired_rows_if_two_labels": int(expected_paired),
        "all_core_metrics_finite": bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all()) if not numeric.empty else False,
        "identity_accepted": bool(identity.get("accepted", False)),
        "identity_mismatches": list(identity.get("mismatches", [])),
        "cg_residual_p50": float(residuals.median()) if not residuals.empty else float("nan"),
        "cg_residual_p90": float(residuals.quantile(0.90)) if not residuals.empty else float("nan"),
        "cg_residual_max": float(residuals.max()) if not residuals.empty else float("nan"),
        "cg_residual_gate": float(residual_gate),
        "cg_residual_gate_ok": bool(residual_ok),
        "cg_converged_fraction": float(pd.to_numeric(combined.get("cg_converged", pd.Series(dtype=float)), errors="coerce").mean())
        if "cg_converged" in combined.columns
        else float("nan"),
        "same_base_probe_sha256_all_methods_labels": bool(same_base_probe),
        "same_effective_probe_sha256_within_method_labels": bool(same_effective_probe_within_method),
        "same_batch_indices_sha256_all_methods_labels": bool(same_batch),
        "preserved_block_rel_l2_to_donor_max": float(preserve_errors.max()) if not preserve_errors.empty else float("nan"),
        "preserved_block_rel_l2_to_donor_p95": float(preserve_errors.quantile(0.95)) if not preserve_errors.empty else float("nan"),
        "preserved_gate": float(preserve_gate),
        "preserved_gate_ok": bool(preserve_ok),
        "preserved_jvp_block_norm_max": float(jvp_errors.max()) if not jvp_errors.empty else float("nan"),
        "preserved_jt_block_norm_max": float(jt_errors.max()) if not jt_errors.empty else float("nan"),
        "jacobian_gate": float(jacobian_gate),
        "jacobian_gate_ok": bool(jacobian_ok),
        "methods": sorted(combined["method"].astype(str).unique().tolist()) if "method" in combined.columns else [],
        "probe_spaces": sorted(combined["probe_space"].astype(str).unique().tolist()) if "probe_space" in combined.columns else [],
        "sources": sorted(int(v) for v in combined["source_weight_index"].dropna().astype(int).unique().tolist())
        if "source_weight_index" in combined.columns
        else [],
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [_run_dir(v) for v in args.run]
    labels = [str(v) for v in args.label]
    if len(run_dirs) != len(labels):
        raise ValueError(f"--run count {len(run_dirs)} must match --label count {len(labels)}")
    if len(run_dirs) < 1:
        raise ValueError("at least one --run is required")
    methods = _parse_methods(args.methods)
    first_cfg = _load_cfg(run_dirs[0], device=str(args.device), batch_size=int(args.batch_size))
    if torch.device(first_cfg.device).type == "cuda" and not bool(args.allow_gpu):
        raise RuntimeError("GPU device requested without --allow-gpu")
    first_weight_payload = load_torch_cache(run_dirs[0] / "weight_pool.pt")
    if first_weight_payload is None:
        raise FileNotFoundError(run_dirs[0] / "weight_pool.pt")
    first_records = pd.DataFrame(first_weight_payload.get("records", []))
    if first_records.empty and (run_dirs[0] / "weight_pool_records.csv").is_file():
        first_records = pd.read_csv(run_dirs[0] / "weight_pool_records.csv")
    start_bank, start_bank_path = _load_start_bank(
        run_dirs[0],
        weight_records=first_records,
        role=str(args.start_role),
        samples=int(args.samples),
        start_bank_csv=Path(args.start_bank_csv) if str(args.start_bank_csv).strip() else None,
    )
    start_bank_hash = hashlib.sha256(start_bank.to_csv(index=False).encode("utf-8")).hexdigest()
    identities = [
        _selected_run_identity(run_dir, label=label, start_bank=start_bank)
        for run_dir, label in zip(run_dirs, labels, strict=True)
    ]
    identity = _identity_validation(identities)
    (out_dir / "identity_validation.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    if not bool(identity.get("accepted", False)):
        raise RuntimeError(f"run identity validation failed: {identity.get('mismatches', [])}")
    _log(
        "start "
        f"runs={run_dirs} labels={labels} methods={methods} output_dir={out_dir} "
        f"device={args.device} seed={int(args.seed)} start_bank={start_bank_path} "
        f"start_bank_hash={start_bank_hash[:16]} samples={len(start_bank)}"
    )
    manifest = {
        "runs": [str(v) for v in run_dirs],
        "labels": labels,
        "methods": methods,
        "output_dir": str(out_dir),
        "device": str(args.device),
        "seed": int(args.seed),
        "probe_kind": str(args.probe_kind),
        "probe_space": str(args.probe_space),
        "probes": int(args.probes),
        "batch_size": int(args.batch_size),
        "lambdas": str(args.lambdas),
        "lambda_rel": str(args.lambda_rel),
        "damping_mode": "absolute" if str(args.lambdas).strip() else "relative_to_trace_jtj",
        "cg_max_iters": int(args.cg_max_iters),
        "cg_tol": float(args.cg_tol),
        "cg_residual_gate": float(args.cg_residual_gate),
        "cg_init": str(args.cg_init),
        "preserved_gate": float(args.preserved_gate),
        "jacobian_check": bool(args.jacobian_check),
        "jacobian_gate": float(args.jacobian_gate),
        "start_bank_path": str(start_bank_path),
        "start_bank_sha256": str(start_bank_hash),
        "start_source_weight_indices": [int(v) for v in start_bank["source_weight_index"].astype(int).tolist()],
        "identity_validation": identity,
        "run_signatures": {
            str(label): {
                "config": _file_signature(run_dir / "config.json"),
                "checkpoint": _file_signature(run_dir / "vae_checkpoint.pt"),
                "weight_pool": _file_signature(run_dir / "weight_pool.pt"),
            }
            for label, run_dir in zip(labels, run_dirs, strict=True)
        },
        "effective_map": "theta=decode(z) for decoder_latent; theta=splice(decode(z), raw_w0, block) for clamp methods",
        "effective_jacobian": "preserved block rows are constant donor values and have zero Jacobian",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    frames: list[pd.DataFrame] = []
    for run_dir, label in zip(run_dirs, labels, strict=True):
        frame = _audit_one_label(
            args=args,
            run_dir=run_dir,
            label=label,
            methods=methods,
            start_bank=start_bank,
            start_bank_path=start_bank_path,
            start_bank_hash=start_bank_hash,
        )
        frame.to_csv(out_dir / f"clamped_cg_{label}.csv", index=False)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    combined.to_csv(out_dir / "clamped_cg_combined.csv", index=False)
    paired, summary, validation = _paired_outputs(combined, labels=labels, out_dir=out_dir, identity=identity, args=args)
    _log(
        "wrote "
        f"combined={out_dir / 'clamped_cg_combined.csv'} paired={out_dir / 'paired_clamped_cg_deltas.csv'} "
        f"summary={out_dir / 'paired_clamped_cg_summary.csv'} validation={out_dir / 'clamped_cg_validation.json'} "
        f"validation={json.dumps(validation, sort_keys=True)}"
    )
    if not bool(validation.get("accepted", False)):
        raise RuntimeError(f"clamped CG validation failed: {json.dumps(validation, sort_keys=True)}")
    return {
        "combined_rows": int(len(combined)),
        "paired_rows": int(len(paired)),
        "summary_rows": int(len(summary)),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clamp-aware exact-CG c3 audit for Variant A downstream repairs.")
    parser.add_argument("--run", action="append", required=True, help="Run name or artifact path. Pass one or two.")
    parser.add_argument("--label", action="append", required=True, help="Label matching --run. Pass one per --run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--start-role", default="eval", choices=("tune", "eval", "all", "any"))
    parser.add_argument("--start-bank-csv", type=str, default="")
    parser.add_argument("--methods", nargs="*", default=list(SUPPORTED_METHODS))
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--probe-kind", default="rademacher", choices=("rademacher", "gaussian"))
    parser.add_argument("--probe-space", default="full", choices=("full", "active", "both"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lambda-rel", default="1e-1")
    parser.add_argument("--lambdas", default="0.1")
    parser.add_argument("--min-lambda", type=float, default=1e-6)
    parser.add_argument("--scale-probes", type=int, default=1)
    parser.add_argument("--cg-max-iters", type=int, default=40)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--cg-residual-gate", type=float, default=1e-4)
    parser.add_argument("--cg-init", default="zero", choices=("zero", "lambda", "v_over_lambda", "null"))
    parser.add_argument("--preserved-gate", type=float, default=1e-7)
    parser.add_argument("--jacobian-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jacobian-gate", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
