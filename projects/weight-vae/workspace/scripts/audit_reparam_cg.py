from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
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


def _load_cfg(output_dir: Path, *, device: str, batch_size: int) -> ExperimentConfig:
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"config.json in {output_dir} does not contain a config mapping")
    values = dict(raw_cfg)
    values["device"] = str(device)
    values["vae_precond_batch_size"] = int(batch_size)
    return ExperimentConfig(**values)


def _load_start_bank(
    output_dir: Path,
    *,
    weight_records: pd.DataFrame,
    role: str,
    samples: int,
    start_bank_csv: Path | None,
) -> tuple[pd.DataFrame, Path]:
    bank_path = Path(start_bank_csv).expanduser().resolve() if start_bank_csv is not None else output_dir / "downstream_start_bank.csv"
    if bank_path.is_file():
        bank = pd.read_csv(bank_path)
    else:
        if start_bank_csv is not None:
            raise FileNotFoundError(bank_path)
        final_step = int(pd.to_numeric(weight_records["step"], errors="coerce").max())
        bank = weight_records[pd.to_numeric(weight_records["step"], errors="coerce") == final_step].copy()
        bank = bank.reset_index(names="source_weight_index")
        bank.insert(0, "start_bank_position", list(range(len(bank))))
        bank["start_role"] = "eval"
        bank["selection"] = "final_all_fallback"
    if "source_weight_index" not in bank.columns:
        raise ValueError(f"{bank_path} has no source_weight_index column")
    role_value = str(role).strip().lower()
    if role_value not in {"all", "any"} and "start_role" in bank.columns:
        selected = bank[bank["start_role"].astype(str).str.lower() == role_value].copy()
        if selected.empty:
            raise ValueError(f"start bank has no rows for start_role={role_value!r}")
    else:
        selected = bank.copy()
    selected = selected.reset_index(drop=True)
    if int(samples) > 0:
        selected = selected.iloc[: int(samples)].copy()
    return selected, bank_path


def _batch_indices(train_count: int, *, batch_size: int, source_weight_index: int, probe_id: int) -> torch.Tensor | None:
    if int(batch_size) <= 0 or int(batch_size) >= int(train_count):
        return None
    offset = (int(source_weight_index) * 1009 + int(probe_id) * 7919) % int(train_count)
    return (torch.arange(int(batch_size)) + offset).remainder(int(train_count)).long()


def _task_loss_from_flat(
    flat: torch.Tensor,
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    spec,
    tau: float,
) -> torch.Tensor:
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    return F.cross_entropy(logits, labels)


def _x_probe(dim: int, *, generator: torch.Generator, device: torch.device, dtype: torch.dtype, kind: str) -> torch.Tensor:
    kind_value = str(kind).strip().lower()
    if kind_value in {"rademacher", "rad", "sign"}:
        probe = torch.randint(0, 2, (int(dim),), generator=generator, device="cpu", dtype=torch.int64)
        probe = probe.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
    elif kind_value in {"gaussian", "normal"}:
        probe = torch.randn((int(dim),), generator=generator, device="cpu", dtype=torch.float32)
        probe = probe * (math.sqrt(float(dim)) / probe.norm().clamp_min(1e-12))
    else:
        raise ValueError(f"unsupported probe kind {kind!r}")
    return probe.to(device=device, dtype=dtype)


def _z_probe_like(z: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    probe = torch.randn(tuple(z.shape), generator=generator, device="cpu", dtype=torch.float32).to(device=z.device, dtype=z.dtype)
    return probe * (math.sqrt(float(z.numel())) / probe.norm().clamp_min(1e-12))


class DecoderMatvec:
    def __init__(self, *, vae: torch.nn.Module, normalizer: WeightNormalizer, z: torch.Tensor) -> None:
        self.vae = vae
        self.normalizer = normalizer
        self.z = z.detach()

    def decode_single(self, latent: torch.Tensor) -> torch.Tensor:
        return decode_weights(self.vae, self.normalizer, latent.reshape(1, -1)).squeeze(0)

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

    def scale_trace_jtj_per_dim(self, *, generator: torch.Generator, probes: int) -> float:
        values: list[float] = []
        for _idx in range(max(1, int(probes))):
            u = _z_probe_like(self.z, generator=generator)
            ju = self.j_vec(u)
            values.append(float((ju.float().square().sum() / float(max(1, int(u.numel())))).detach().cpu().item()))
        return float(sum(values) / max(1, len(values)))


def _dot64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a.detach().double(), b.detach().double())


def _tensor_sha256(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "full_batch"
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _cg_solve(
    matvec,
    v: torch.Tensor,
    *,
    lambda_abs: float,
    max_iters: int,
    tol: float,
    init: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    if str(init).strip().lower() in {"lambda", "v_over_lambda", "null"}:
        w = v.detach().clone() / float(lambda_abs)
    else:
        w = torch.zeros_like(v)
    aw = matvec(w, float(lambda_abs))
    r = v.detach() - aw
    p = r.clone()
    v_norm = float(v.float().norm().detach().cpu().item())
    r_norm = float(r.float().norm().detach().cpu().item())
    initial_rel = r_norm / max(v_norm, 1e-30)
    rel = initial_rel
    rs_old = _dot64(r, r)
    residual_history = [float(rel)]
    converged = bool(rel <= float(tol))
    iters = 0
    breakdown = ""
    for idx in range(int(max_iters)):
        if converged:
            break
        ap = matvec(p, float(lambda_abs))
        denom = _dot64(p, ap)
        denom_value = float(denom.detach().cpu().item())
        if (not math.isfinite(denom_value)) or denom_value <= 0.0:
            breakdown = f"non_positive_denom:{denom_value:.6g}"
            break
        alpha = (rs_old / denom).to(device=v.device, dtype=v.dtype)
        w = w + alpha * p
        r = r - alpha * ap
        rs_new = _dot64(r, r)
        rel = math.sqrt(max(0.0, float(rs_new.detach().cpu().item()))) / max(v_norm, 1e-30)
        residual_history.append(float(rel))
        iters = idx + 1
        if rel <= float(tol):
            converged = True
            rs_old = rs_new
            break
        beta = (rs_new / rs_old.clamp_min(1e-300)).to(device=v.device, dtype=v.dtype)
        p = r + beta * p
        rs_old = rs_new
    elapsed = time.perf_counter() - start
    aw_final = matvec(w, float(lambda_abs))
    final_residual = aw_final - v.detach()
    final_rel = float(final_residual.float().norm().detach().cpu().item()) / max(v_norm, 1e-30)
    vtw = float(_dot64(v, w).detach().cpu().item())
    return {
        "w": w.detach(),
        "cg_iterations": int(iters),
        "cg_converged": bool(converged),
        "cg_breakdown": breakdown,
        "cg_initial_rel_residual": float(initial_rel),
        "cg_final_rel_residual": float(final_rel),
        "cg_elapsed_sec": float(elapsed),
        "cg_residual_history_json": json.dumps(residual_history),
        "vTw": float(vtw),
    }


def _hvp_x(
    theta_bar: torch.Tensor,
    v: torch.Tensor,
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    spec,
    tau: float,
) -> torch.Tensor:
    theta_req = theta_bar.detach().clone().requires_grad_(True)
    loss = _task_loss_from_flat(theta_req, images=images, labels=labels, spec=spec, tau=float(tau))
    grad_theta = torch.autograd.grad(loss, theta_req, create_graph=True, retain_graph=True)[0]
    hvp = torch.autograd.grad((grad_theta * v.detach()).sum(), theta_req, create_graph=False, retain_graph=False)[0]
    return hvp.detach()


def _finite_difference_hvp_check(
    theta_bar: torch.Tensor,
    v: torch.Tensor,
    h_x: torch.Tensor,
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    spec,
    tau: float,
    eps: float,
) -> dict[str, float]:
    def grad_at(theta: torch.Tensor) -> torch.Tensor:
        theta_req = theta.detach().clone().requires_grad_(True)
        loss = _task_loss_from_flat(theta_req, images=images, labels=labels, spec=spec, tau=float(tau))
        grad = torch.autograd.grad(loss, theta_req, create_graph=False, retain_graph=False)[0]
        return grad.detach()

    g_plus = grad_at(theta_bar + float(eps) * v)
    g_base = grad_at(theta_bar)
    fd = (g_plus - g_base) / float(eps)
    denom = fd.float().norm().clamp_min(1e-30) * h_x.float().norm().clamp_min(1e-30)
    cosine = float(((fd.float() * h_x.float()).sum() / denom).detach().cpu().item())
    rel = float(((fd - h_x).float().norm() / h_x.float().norm().clamp_min(1e-30)).detach().cpu().item())
    return {"fd_hvp_cosine": cosine, "fd_hvp_rel_error": rel}


def run_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else output_csv.with_suffix(".summary.json")
    cfg = _load_cfg(output_dir, device=str(args.device), batch_size=int(args.batch_size))
    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    if device.type == "cuda" and not bool(args.allow_gpu):
        raise RuntimeError("GPU device requested without --allow-gpu. Use CPU or wrap command with flock and pass --allow-gpu.")

    print(
        "[reparam_cg_audit] start "
        f"output_dir={output_dir} output_csv={output_csv} device={device} dtype={dtype} "
        f"seed={int(args.seed)} batch_size={int(args.batch_size)} samples={int(args.samples)} "
        f"probes={int(args.probes)} cg_max_iters={int(args.cg_max_iters)} cg_tol={float(args.cg_tol)}",
        flush=True,
    )
    print(f"[reparam_cg_audit] resolved_config_hash={config_hash(cfg)}", flush=True)
    print(f"[reparam_cg_audit] resolved_config={json.dumps(asdict(cfg), sort_keys=True, default=str)}", flush=True)

    print("[reparam_cg_audit] stage=load artifacts", flush=True)
    weight_payload = load_torch_cache(output_dir / "weight_pool.pt")
    if weight_payload is None:
        raise FileNotFoundError(output_dir / "weight_pool.pt")
    vae_payload = load_torch_cache(output_dir / "vae_checkpoint.pt")
    if vae_payload is None:
        raise FileNotFoundError(output_dir / "vae_checkpoint.pt")
    weights = weight_payload["weights"]
    weight_records = pd.DataFrame(weight_payload["records"])
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=dtype).eval()
    vae.load_state_dict(vae_payload["model_state"])
    weights_device = weights.to(device=device, dtype=dtype)
    print(
        "[reparam_cg_audit] artifacts "
        f"weights_shape={tuple(weights.shape)} spec_dim={int(spec.dim)} latent_dim={int(cfg.latent_dim)} "
        f"weight_cache_key={weight_payload.get('cache_key')} vae_cache_key={vae_payload.get('cache_key')}",
        flush=True,
    )

    print("[reparam_cg_audit] stage=load task tensors", flush=True)
    task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=device, dtype=dtype)
    for task_name, task_set in task_tensors.items():
        print(
            "[reparam_cg_audit] task "
            f"{task_name}: train={tuple(task_set.train_images.shape)} test={tuple(task_set.test_images.shape)}",
            flush=True,
        )

    start_bank, start_bank_path = _load_start_bank(
        output_dir,
        weight_records=weight_records,
        role=str(args.start_role),
        samples=int(args.samples),
        start_bank_csv=Path(args.start_bank_csv) if str(args.start_bank_csv).strip() else None,
    )
    start_bank_hash = hashlib.sha256(start_bank.to_csv(index=False).encode("utf-8")).hexdigest()
    print(
        "[reparam_cg_audit] stage=start bank "
        f"rows={len(start_bank)} role={args.start_role} "
        f"path={start_bank_path} hash={start_bank_hash[:16]} "
        f"indices={start_bank['source_weight_index'].astype(int).tolist()}",
        flush=True,
    )

    lambda_rel = [float(v) for v in str(args.lambda_rel).split(",") if str(v).strip()]
    lambda_abs_values = [float(v) for v in str(args.lambdas).split(",") if str(v).strip()] if args.lambdas else []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for start_pos, start_row in start_bank.iterrows():
        source_weight_index = int(start_row["source_weight_index"])
        record = weight_records.iloc[source_weight_index].to_dict()
        task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
        tau = float(record.get("tau", start_row.get("tau", 1.0)))
        task_set = task_tensors[task_name]
        w0 = weights_device[source_weight_index]
        with torch.no_grad():
            z = encode_weights(vae, normalizer, w0.reshape(1, -1).to(device=device, dtype=dtype)).squeeze(0)
            theta_bar = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0).detach()
        decoder_ops = DecoderMatvec(vae=vae, normalizer=normalizer, z=z)
        scale = decoder_ops.scale_trace_jtj_per_dim(generator=generator, probes=int(args.scale_probes))
        lambdas = lambda_abs_values if lambda_abs_values else [max(float(args.min_lambda), rel * max(scale, float(args.min_lambda))) for rel in lambda_rel]
        print(
            "[reparam_cg_audit] start "
            f"{start_pos + 1}/{len(start_bank)} source_weight_index={source_weight_index} "
            f"task={task_name} tau={tau:.6g} z_dim={int(z.numel())} theta_dim={int(theta_bar.numel())} "
            f"trace_jtj_per_dim={scale:.6g} lambdas={','.join(f'{v:.6g}' for v in lambdas)}",
            flush=True,
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
            v = _x_probe(int(spec.dim), generator=generator, device=device, dtype=dtype, kind=str(args.probe_kind))
            probe_hash = _tensor_sha256(v)
            batch_hash = _tensor_sha256(batch_idx)
            h_x = _hvp_x(theta_bar, v, images=images, labels=labels, spec=spec, tau=tau)
            jt_h = decoder_ops.jt_vec(h_x)
            fd_metrics: dict[str, float] = {"fd_hvp_cosine": float("nan"), "fd_hvp_rel_error": float("nan")}
            if bool(args.fd_check) and probe_id == 0 and start_pos == 0:
                fd_metrics = _finite_difference_hvp_check(
                    theta_bar,
                    v,
                    h_x,
                    images=images,
                    labels=labels,
                    spec=spec,
                    tau=tau,
                    eps=float(args.fd_eps),
                )
                print(
                    "[reparam_cg_audit] fd_check "
                    f"eps={float(args.fd_eps):.3g} cosine={fd_metrics['fd_hvp_cosine']:.6g} "
                    f"rel_error={fd_metrics['fd_hvp_rel_error']:.6g}",
                    flush=True,
                )
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
                c3_forward = float(jt_h.float().square().sum().detach().cpu().item())
                c3_inverse = float(cg["vTw"])
                probe_norm2 = float(v.float().square().sum().detach().cpu().item())
                row = {
                    "output_dir": str(output_dir),
                    "source_weight_index": int(source_weight_index),
                    "start_bank_position": int(start_row.get("start_bank_position", start_pos)),
                    "start_role": str(start_row.get("start_role", "")),
                    "selection": str(start_row.get("selection", "")),
                    "audit_seed": int(args.seed),
                    "start_bank_path": str(start_bank_path),
                    "start_bank_sha256": start_bank_hash,
                    "source_run": int(record.get("run", -1)),
                    "source_step": int(record.get("step", -1)),
                    "source_lr": float(record.get("source_lr", float("nan"))),
                    "task_name": task_name,
                    "tau": float(tau),
                    "probe_id": int(probe_id),
                    "probe_kind": str(args.probe_kind),
                    "probe_sha256": probe_hash,
                    "batch_indices_sha256": batch_hash,
                    "batch_size": int(images.shape[0]),
                    "lambda_abs": float(lambda_abs),
                    "lambda_rel_to_trace_jtj": float(lambda_abs / max(scale, 1e-30)),
                    "trace_jtj_per_dim": float(scale),
                    "rank_bound": int(min(int(spec.dim), int(cfg.latent_dim))),
                    "x_dim": int(spec.dim),
                    "z_dim": int(cfg.latent_dim),
                    "expected_null_fraction": float(max(0.0, 1.0 - float(cfg.latent_dim) / float(spec.dim))),
                    "probe_norm2": probe_norm2,
                    "hvp_norm2": float(h_x.float().square().sum().detach().cpu().item()),
                    "jt_h_norm2": c3_forward,
                    "jt_w_norm2": float(jt_w.float().square().sum().detach().cpu().item()),
                    "c3_forward": c3_forward,
                    "c3_inverse": c3_inverse,
                    "c3_total": c3_forward + c3_inverse,
                    "lambda_vtw_ratio": float(lambda_abs * c3_inverse / max(probe_norm2, 1e-30)),
                    **cg,
                    **fd_metrics,
                }
                rows.append(row)
                print(
                    "[reparam_cg_audit] metric "
                    f"start={source_weight_index} probe={probe_id} lambda={lambda_abs:.6g} "
                    f"iters={row['cg_iterations']} rel_res={row['cg_final_rel_residual']:.3g} "
                    f"lambda_vtw_ratio={row['lambda_vtw_ratio']:.6g} "
                    f"c3_forward={c3_forward:.6g} c3_inverse={c3_inverse:.6g}",
                    flush=True,
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    elapsed = time.perf_counter() - start_time
    summary = {
        "output_csv": str(output_csv),
        "rows": int(len(frame)),
        "elapsed_sec": float(elapsed),
        "device": str(device),
        "dtype": str(dtype),
        "samples": int(args.samples),
        "probes": int(args.probes),
        "cg_max_iters": int(args.cg_max_iters),
        "seed": int(args.seed),
        "start_bank_path": str(start_bank_path),
        "start_bank_sha256": start_bank_hash,
        "median_cg_final_rel_residual": float(frame["cg_final_rel_residual"].median()) if not frame.empty else float("nan"),
        "median_cg_iterations": float(frame["cg_iterations"].median()) if not frame.empty else float("nan"),
        "median_lambda_vtw_ratio": float(frame["lambda_vtw_ratio"].median()) if not frame.empty else float("nan"),
        "median_c3_forward": float(frame["c3_forward"].median()) if not frame.empty else float("nan"),
        "median_c3_inverse": float(frame["c3_inverse"].median()) if not frame.empty else float("nan"),
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[reparam_cg_audit] wrote "
        f"csv={output_csv} json={output_json} rows={len(frame)} summary={json.dumps(summary, sort_keys=True)}",
        flush=True,
    )
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc Variant C damped CG audit for CELO TinyBigVAE artifacts.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--start-role", default="eval", choices=("tune", "eval", "all", "any"))
    parser.add_argument("--start-bank-csv", type=str, default="")
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--probe-kind", default="rademacher", choices=("rademacher", "gaussian"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lambda-rel", default="1e-2")
    parser.add_argument("--lambdas", default="")
    parser.add_argument("--min-lambda", type=float, default=1e-6)
    parser.add_argument("--scale-probes", type=int, default=1)
    parser.add_argument("--cg-max-iters", type=int, default=5)
    parser.add_argument("--cg-tol", type=float, default=1e-3)
    parser.add_argument("--cg-init", default="zero", choices=("zero", "lambda", "v_over_lambda", "null"))
    parser.add_argument("--fd-check", action="store_true")
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
