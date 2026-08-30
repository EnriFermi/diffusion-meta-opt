from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from big_vae.models import build_weight_quantile_vae
from training.big_vae.model_config import build_big_vae_model_config
from training.big_vae.operator_set_overfit import CanonicalOperatorSetTrainingDataset
from training.big_vae.two_operator_overfit import _loss_metrics
from training.weightclip_benchmark.analyze_ae_v9a_final_causal_bundle import (
    _amp,
    _capture_forward,
    _decode_latent_with_boundaries,
    _exact_v8_routing,
    _loss_cfg,
    _move,
    _pair_batch,
)


SCHEMA = "weightclip_ae_v9a_basis_aliasing_v1"
RIDGE_RELATIVE_TO_MEAN_ROW_ENERGY = 1.0e-6
PINV_RTOL = 1.0e-4
CAPACITY_ROUTE_ROWS = (384, 512, 640, 768)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-update V9-A discriminator: apply the same X-only A0 synthesis "
            "to carrier, adaptive, and final latent coordinates."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pair-count", type=int, default=4)
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("/mnt/shared/weightclip_benchmark/diagnostics"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty results CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = torch.sqrt(
        0.5 * (left.float().square().mean() + right.float().square().mean())
    ).clamp_min(1.0e-24)
    return float(delta / scale)


def _latent_to_a0_measurements(
    latent: torch.Tensor,
    inverse_out_weight: torch.Tensor,
    *,
    heads: int,
    patch: int,
) -> torch.Tensor:
    """Express any decoder-visible latent in the clean carrier measurement basis."""
    batch, latents, _ = latent.shape
    unpacked = F.linear(latent.float(), inverse_out_weight)
    return (
        unpacked.view(batch, latents, heads, patch)
        .permute(0, 2, 1, 3)
        .reshape(batch, heads * latents, patch)
    )


def _latin_cycle_route_order(heads: int, latents: int) -> torch.Tensor:
    """Balanced deterministic prefix order over the [head, latent] route grid.

    With the exact64 dimensions (24 heads, 32 slots), prefixes of 384, 512,
    640, and 768 contain exactly 12, 16, 20, and 24 heads per slot.  Stride 7
    is coprime to 24 and also keeps the head counts within two of one another.
    """
    if math.gcd(7, heads) != 1:
        raise ValueError("Latin-cycle subset requires stride 7 coprime to head count")
    order = [
        ((wave + 7 * latent) % heads) * latents + latent
        for wave in range(heads)
        for latent in range(latents)
    ]
    if len(set(order)) != heads * latents:
        raise RuntimeError("route subset order is not a permutation")
    return torch.tensor(order, dtype=torch.long)


def _ridge_factor(design: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    gram = design @ design.transpose(0, 1)
    mean_row_energy = gram.diagonal().mean()
    ridge = RIDGE_RELATIVE_TO_MEAN_ROW_ENERGY * mean_row_energy
    regularized = gram + ridge * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    factor = torch.linalg.cholesky(regularized)
    eigenvalues = torch.linalg.eigvalsh(gram)
    if float(eigenvalues[0]) <= 0.0:
        raise RuntimeError("A0 route subset is not full row rank")
    row_norm = gram.diagonal().sqrt().clamp_min(1.0e-24)
    normalized_gram = gram / (row_norm[:, None] * row_norm[None, :])
    off_diagonal = normalized_gram - torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    off_diagonal_count = max(gram.shape[0] * (gram.shape[0] - 1), 1)
    coverage = design.square().sum(dim=0)
    coverage_mean = coverage.mean().clamp_min(1.0e-30)
    return factor, {
        "ridge_absolute": float(ridge),
        "ridge_relative_to_mean_row_energy": RIDGE_RELATIVE_TO_MEAN_ROW_ENERGY,
        "condition_A": float(torch.sqrt(eigenvalues[-1] / eigenvalues[0])),
        "minimum_gram_eigenvalue": float(eigenvalues[0]),
        "maximum_gram_eigenvalue": float(eigenvalues[-1]),
        "normalized_row_gram_offdiag_rms": float(
            torch.sqrt(off_diagonal.square().sum() / off_diagonal_count)
        ),
        "normalized_row_gram_offdiag_max_abs": float(off_diagonal.abs().amax()),
        "gram_eigenvalue_coefficient_of_variation": float(
            eigenvalues.std() / eigenvalues.mean().clamp_min(1.0e-24)
        ),
        "column_coverage_minimum": float(coverage.min()),
        "column_coverage_median": float(coverage.median()),
        "column_coverage_mean": float(coverage_mean),
        "column_coverage_maximum": float(coverage.max()),
        "column_coverage_below_1e_8_mean_fraction": float(
            (coverage < 1.0e-8 * coverage_mean).float().mean()
        ),
    }


def _synthesize_ridge(
    design: torch.Tensor,
    factor: torch.Tensor,
    measurements: torch.Tensor,
) -> torch.Tensor:
    # Solve all tiles and patch channels in one Cholesky call.  The synthesis
    # is A^T (A A^T + lambda I)^-1 M and depends only on A(X,masks) and M.
    samples, routes, patch = measurements.shape
    rhs = measurements.permute(1, 0, 2).reshape(routes, samples * patch)
    dual_measurements = torch.cholesky_solve(rhs, factor)
    recovered = design.transpose(0, 1) @ dual_measurements
    return recovered.view(design.shape[1], samples, patch).permute(1, 0, 2)


def _synthesize_transpose(design: torch.Tensor, measurements: torch.Tensor) -> torch.Tensor:
    recovered = torch.einsum("rs,nrp->nsp", design, measurements)
    coverage = design.square().sum(dim=0).view(1, -1, 1).clamp_min(1.0e-12)
    return recovered / coverage


def _synthesize_pinv(design: torch.Tensor, measurements: torch.Tensor) -> torch.Tensor:
    inverse = torch.linalg.pinv(design, rtol=PINV_RTOL)
    return torch.einsum("sr,nrp->nsp", inverse, measurements)


def _patches_to_matrix(patches: torch.Tensor, *, d_in: int, d_out: int) -> torch.Tensor:
    return patches.reshape(patches.shape[0], d_out, d_in).transpose(1, 2).contiguous()


def _quality(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    loss_cfg: Mapping[str, Any],
) -> dict[str, float]:
    structural = _loss_metrics(target, prediction, **loss_cfg)
    patch = int(loss_cfg["patch_size"])
    target_patch = target.transpose(1, 2).reshape(-1, patch).float()
    prediction_patch = prediction.transpose(1, 2).reshape(-1, patch).float()
    cosine = F.cosine_similarity(target_patch, prediction_patch, dim=-1)
    error_sum_sq = (prediction.float() - target.float()).square().sum()
    target_sum_sq = target.float().square().sum()
    prediction_sum_sq = prediction.float().square().sum()
    return {
        "weighted_direction_loss": structural["dir"],
        "plain_direction_loss": float((1.0 - cosine).mean()),
        "log_scale_huber": structural["scale"],
        "structural_total": structural["total"],
        "nrmse": float(torch.sqrt(error_sum_sq / target_sum_sq.clamp_min(1.0e-24))),
        "target_rms": float(torch.sqrt(target_sum_sq / target.numel())),
        "prediction_rms": float(torch.sqrt(prediction_sum_sq / prediction.numel())),
        "error_sum_sq": float(error_sum_sq),
        "target_sum_sq": float(target_sum_sq),
        "prediction_sum_sq": float(prediction_sum_sq),
        "element_count": int(target.numel()),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    keys = ("scope", "variant", "method", "route_rows", "family_left_out")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    mean_fields = (
        "weighted_direction_loss",
        "plain_direction_loss",
        "log_scale_huber",
        "structural_total",
    )
    for group, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        error = sum(float(row["error_sum_sq"]) for row in values)
        target = sum(float(row["target_sum_sq"]) for row in values)
        prediction = sum(float(row["prediction_sum_sq"]) for row in values)
        count = sum(int(row["element_count"]) for row in values)
        row = {key: value for key, value in zip(keys, group, strict=True)}
        row.update(
            {
                "operator_count": len(values),
                **{
                    field: sum(float(value[field]) for value in values) / len(values)
                    for field in mean_fields
                },
                "nrmse": math.sqrt(error / max(target, 1.0e-24)),
                "target_rms": math.sqrt(target / count),
                "prediction_rms": math.sqrt(prediction / count),
            }
        )
        result.append(row)
    return result


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    resolved_path = run_root / "resolved_run_config.json"
    selection_path = run_root / "operator_set_selection.json"
    if not 1 <= args.pair_count <= 4:
        raise ValueError("this bounded discriminator requires 1 <= pair-count <= 4")
    for path in (resolved_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    launch = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint_path),
        "run_root": str(run_root),
        "device": args.device,
        "pair_count": args.pair_count,
        "optimizer_steps": 0,
        "source": str(Path(__file__).resolve()),
    }
    print(json.dumps(launch, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("stage=dry_run_complete no_files_written=true", flush=True)
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = args.output_parent.expanduser().resolve() / f"v9a_basis_aliasing_{stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    report_path = output_root / "report.json"
    csv_path = output_root / "readout_results.csv"
    manifest_path = output_root / "artifact_manifest.json"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "preflight": launch,
        "artifacts": {
            "report": str(report_path),
            "readout_csv": str(csv_path),
            "manifest": str(manifest_path),
        },
        "partial_pairs": [],
    }
    _atomic_json(report_path, report)

    try:
        print(f"stage=checkpoint_load output={output_root}", flush=True)
        cfg_dict = json.loads(resolved_path.read_text(encoding="utf-8"))
        expected_checkpoint = (
            Path(str(cfg_dict["train"]["checkpoint_dir"])).expanduser().resolve()
            / "stage_1"
            / "step_0001984.pt"
        )
        if checkpoint_path != expected_checkpoint:
            raise RuntimeError(
                f"checkpoint/run-root binding failed: got={checkpoint_path} expected={expected_checkpoint}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        required = {"step", "stage", "model_state", "config"}
        if not required.issubset(checkpoint):
            raise RuntimeError(f"checkpoint lacks keys {sorted(required - set(checkpoint))}")
        if int(checkpoint["step"]) != 1984 or int(checkpoint["stage"]) != 1:
            raise RuntimeError("basis discriminator requires final stage=1 step=1984")
        cfg = OmegaConf.create(checkpoint["config"])
        embedded = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(embedded, dict)
        critical_paths = (
            ("data", "seed"),
            ("model", "patch_size"),
            ("model", "big_vae"),
            ("train", "operator_bank", "pair_manifest"),
            ("train", "operator_bank", "pair_manifest_sha256"),
            ("train", "operator_bank", "operator_set_overfit", "selected_operators"),
            ("train", "operator_bank", "operator_set_overfit", "selection_sha256"),
            ("train", "operator_bank", "operator_set_overfit", "schedule_sha256"),
            ("train", "struct_loss"),
            ("train", "max_steps"),
            ("train", "lr"),
            ("train", "eps"),
        )
        mismatches = [
            ".".join(path)
            for path in critical_paths
            if _nested(embedded, path) != _nested(cfg_dict, path)
        ]
        if mismatches:
            raise RuntimeError(f"embedded/resolved config mismatch: {mismatches}")
        operator_cfg = cfg_dict["train"]["operator_bank"]
        overfit = operator_cfg["operator_set_overfit"]
        if (
            selection["selection_sha256"] != overfit["selection_sha256"]
            or selection["schedule_sha256"] != overfit["schedule_sha256"]
        ):
            raise RuntimeError("selection/config seal mismatch")

        device = torch.device(args.device)
        model = build_weight_quantile_vae(build_big_vae_model_config(cfg))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        del checkpoint
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        refiner = model.hybrid_content_readout_v9
        if (
            str(cfg.model.big_vae.architecture_version)
            != "carrier_mean_content_posfilm_v9a"
            or refiner is None
            or not refiner.carrier_mean_content
            or refiner.carrier_content_aggregation != "sqrt_depth_sum"
        ):
            raise RuntimeError("checkpoint is not exact V9-A sqrt-depth")
        if refiner.carrier_mix != 0.1 or len(refiner.blocks) != 40:
            raise RuntimeError("expected frozen carrier_mix=0.1 and 40 refinement blocks")

        source = CanonicalOperatorSetTrainingDataset(
            operator_cfg["pair_manifest"],
            selected_operators=overfit["selected_operators"],
            expected_operator_count=64,
            seed=int(cfg_dict["data"]["seed"]),
            hot_shards=int(operator_cfg["hot_shards"]),
            expected_pair_manifest_sha256=operator_cfg["pair_manifest_sha256"],
            expected_selection_sha256=overfit["selection_sha256"],
            expected_schedule_sha256=overfit["schedule_sha256"],
        )
        heldout_pairs = list(source._rounds[62])[: args.pair_count]
        loss_cfg = _loss_cfg(cfg)
        readout = model.clean_content_readout_v8
        if readout is None:
            raise RuntimeError("clean V8 carrier is missing")
        heads = int(readout.n_heads)
        latents = int(readout.num_latents)
        patch = int(readout.patch_size)
        route_order_cpu = _latin_cycle_route_order(heads, latents)
        if tuple(CAPACITY_ROUTE_ROWS)[-1] != heads * latents:
            raise RuntimeError("capacity route counts do not end at the full A0 route count")
        out_weight = readout.out_proj.weight.float()
        inverse_out = torch.linalg.pinv(out_weight, rtol=1.0e-5)

        report["preflight"].update(
            {
                "checkpoint_stat": {
                    "size": checkpoint_path.stat().st_size,
                    "mtime_ns": checkpoint_path.stat().st_mtime_ns,
                },
                "checkpoint_step": 1984,
                "checkpoint_stage": 1,
                "checkpoint_run_root_binding": "pass",
                "embedded_vs_resolved_config_binding": "pass",
                "selection_sha256": selection["selection_sha256"],
                "schedule_sha256": selection["schedule_sha256"],
                "source_sha256": _sha256(Path(__file__).resolve()),
                "dtype": "bfloat16-autocast forward; FP32 synthesis",
                "out_proj_inverse_condition": float(torch.linalg.cond(out_weight)),
                "a0_dependency_contract": [
                    "distribution context X",
                    "patch mask",
                    "output mask",
                ],
                "adaptive_scaling": (
                    "debug adaptive is the decoder-visible 0.1*a contribution; "
                    "adaptive_unscaled divides by 0.1; final_z is native 0.9*c+0.1*a"
                ),
                "common_basis_rule": (
                    "all latent variants are first mapped through pinv(clean out_proj), then "
                    "the identical X-only A0 synthesis is applied"
                ),
                "capacity_subset_rule": {
                    "name": "latin_cycle_wave_prefix_stride_7",
                    "order": "wave outer, slot inner, head=(wave+7*slot) mod 24",
                    "route_rows": list(CAPACITY_ROUTE_ROWS),
                    "coordinates": [count * patch for count in CAPACITY_ROUTE_ROWS],
                    "prospective_Hc20_rule": (
                        "separate exact slice heads 0..19 x all 32 slots; "
                        "R=640, 10,240 protected scalar coordinates, four complete heads reserved"
                    ),
                },
                "family_leaveout_semantics": (
                    "frozen-route additive leaveout from captured block writes; later routing "
                    "is not recomputed, so this is not a recurrent counterfactual"
                ),
            }
        )
        _atomic_json(report_path, report)

        result_rows: list[dict[str, Any]] = []
        parity_rows: list[dict[str, Any]] = []
        design_rows: list[dict[str, Any]] = []
        first_design: torch.Tensor | None = None
        family_names = ("local", "medium", "broad", "global")

        with torch.inference_mode():
            for pair_index, pair in enumerate(heldout_pairs):
                started = time.time()
                print(
                    f"stage=pair pair={pair_index + 1}/{len(heldout_pairs)} operators={pair[0][0][:8]},{pair[1][0][:8]}",
                    flush=True,
                )
                cpu_batch, assignments, swap_cpu = _pair_batch(source, pair)
                batch = _move(cpu_batch, device)
                swap = swap_cpu.to(device)
                w_patches = batch["W"].transpose(1, 2).reshape(
                    batch["W"].shape[0], batch["W"].shape[2], -1, patch
                )
                with _amp(device):
                    native_prediction, debug, writes = _capture_forward(model, batch)
                    a0 = _exact_v8_routing(
                        readout,
                        debug["dist_patch_by_patch"],
                        debug["patch_mask"],
                        batch["d_out_mask"],
                    )
                    native_repeat_carrier = readout(
                        w_patches,
                        debug["dist_patch_by_patch"],
                        patch_mask=debug["patch_mask"],
                        output_mask=batch["d_out_mask"],
                    )

                carrier = refiner.last_carrier_state
                adaptive_scaled = refiner.last_adaptive_state
                if carrier is None or adaptive_scaled is None:
                    raise RuntimeError("V9-A carrier/adaptive capture missing")
                carrier = carrier.detach()
                adaptive_scaled = adaptive_scaled.detach()
                native_z = debug["latent_decoder_z"].view_as(carrier).detach()
                content_sum = torch.stack([write.float() for write in writes]).sum(dim=0)
                adaptive_manual = (content_sum / math.sqrt(float(len(writes)))).to(carrier.dtype)
                z_manual = (0.9 * carrier + 0.1 * adaptive_manual).to(carrier.dtype)

                target_flat = w_patches.float().reshape(w_patches.shape[0], -1, patch)
                design = a0.float().reshape(a0.shape[0], heads * latents, -1)
                with _amp(device):
                    routed = torch.einsum(
                        "bhls,bsp->blhp",
                        a0,
                        target_flat,
                    ).contiguous()
                    carrier_manual = readout.out_proj(
                        routed.view(routed.shape[0], latents, heads * patch)
                    )
                    manual_prediction, _boundaries = _decode_latent_with_boundaries(
                        model, native_z, batch, debug
                    )

                parity = {
                    "pair_index": pair_index,
                    "native_carrier_repeat_max_abs": float(
                        (native_repeat_carrier.float() - carrier.float()).abs().amax()
                    ),
                    "manual_A0_carrier_max_abs": float(
                        (carrier_manual.float() - carrier.float()).abs().amax()
                    ),
                    "manual_adaptive_scaled_max_abs": float(
                        ((0.1 * adaptive_manual).float() - adaptive_scaled.float()).abs().amax()
                    ),
                    "manual_final_z_max_abs": float(
                        (z_manual.float() - native_z.float()).abs().amax()
                    ),
                    "captured_components_final_z_max_abs": float(
                        (
                            (0.9 * carrier.float() + adaptive_scaled.float())
                            - native_z.float()
                        )
                        .abs()
                        .amax()
                    ),
                    "captured_components_final_z_relative_rms": _relative_rms(
                        0.9 * carrier.float() + adaptive_scaled.float(), native_z
                    ),
                    "manual_decode_max_abs": float(
                        (manual_prediction.float() - native_prediction.float()).abs().amax()
                    ),
                    "manual_decode_relative_rms": _relative_rms(
                        manual_prediction, native_prediction
                    ),
                }

                # One explicit W-shuffle test is enough: X and masks are held fixed,
                # while W is swapped within every held-out operator pair.
                if pair_index == 0:
                    shuffled_batch = dict(batch)
                    shuffled_batch["W"] = batch["W"].index_select(0, swap)
                    with _amp(device):
                        _p, _m, _l, _d, shuffled_debug = model.forward_debug(
                            shuffled_batch["W"],
                            shuffled_batch["x"],
                            x_mask=shuffled_batch["x_mask"],
                            d_in_mask=shuffled_batch["d_in_mask"],
                            d_out_mask=shuffled_batch["d_out_mask"],
                            disable_z_shortcut=True,
                        )
                        shuffled_a0 = _exact_v8_routing(
                            readout,
                            shuffled_debug["dist_patch_by_patch"],
                            shuffled_debug["patch_mask"],
                            shuffled_batch["d_out_mask"],
                        )
                    parity["W_shuffle_dist_context_max_abs"] = float(
                        (
                            shuffled_debug["dist_patch_by_patch"].float()
                            - debug["dist_patch_by_patch"].float()
                        )
                        .abs()
                        .amax()
                    )
                    parity["W_shuffle_A0_max_abs"] = float(
                        (shuffled_a0.float() - a0.float()).abs().amax()
                    )
                    if parity["W_shuffle_A0_max_abs"] != 0.0:
                        raise RuntimeError("A0 changed under W-only shuffle")

                measurements = {
                    "carrier_unscaled": _latent_to_a0_measurements(
                        carrier, inverse_out, heads=heads, patch=patch
                    ),
                    "adaptive_unscaled": _latent_to_a0_measurements(
                        adaptive_scaled.float() / 0.1,
                        inverse_out,
                        heads=heads,
                        patch=patch,
                    ),
                    "carrier_contribution_0.9": _latent_to_a0_measurements(
                        0.9 * carrier.float(), inverse_out, heads=heads, patch=patch
                    ),
                    "adaptive_contribution_0.1": _latent_to_a0_measurements(
                        adaptive_scaled, inverse_out, heads=heads, patch=patch
                    ),
                    "final_z_native": _latent_to_a0_measurements(
                        native_z, inverse_out, heads=heads, patch=patch
                    ),
                }
                family_raw: dict[str, torch.Tensor] = {}
                for family in family_names:
                    selected = [
                        write.float()
                        for block, write in zip(refiner.blocks, writes, strict=True)
                        if block.anchor_family == family
                    ]
                    if len(selected) != 10:
                        raise RuntimeError(f"expected ten {family} blocks, got {len(selected)}")
                    family_raw[family] = (
                        torch.stack(selected).sum(dim=0) / math.sqrt(float(len(writes)))
                    )
                    leaveout = native_z.float() - 0.1 * family_raw[family]
                    measurements[f"final_without_{family}"] = _latent_to_a0_measurements(
                        leaveout, inverse_out, heads=heads, patch=patch
                    )

                # Confirm that pinv(out_proj) recovers the true clean A0
                # measurements before using the transform on any other branch.
                clean_measurement = torch.bmm(design, target_flat)
                recovered_clean_measurement = measurements["carrier_unscaled"]
                parity["out_proj_measurement_relative_rms"] = _relative_rms(
                    recovered_clean_measurement, clean_measurement
                )
                parity_rows.append(parity)

                for operator_parity, operator_key in enumerate(pair):
                    indices = torch.arange(
                        operator_parity, design.shape[0], 2, device=device
                    )
                    operator_designs = design.index_select(0, indices)
                    repeat_error = float(
                        (operator_designs - operator_designs[0]).abs().amax()
                    )
                    if repeat_error > 1.0e-6:
                        raise RuntimeError(
                            f"A0 does not repeat over tiles for operator parity {operator_parity}: {repeat_error}"
                        )
                    operator_design = operator_designs[0]
                    if first_design is None:
                        first_design = operator_design.detach().clone()
                    cross_reference_max_abs = float(
                        (operator_design - first_design).abs().amax()
                    )
                    target = batch["W"].index_select(0, indices)
                    d_in, d_out = int(target.shape[1]), int(target.shape[2])
                    operator_id = operator_key[0]
                    design_entry: dict[str, Any] = {
                        "pair_index": pair_index,
                        "operator_parity": operator_parity,
                        "operator_id": operator_id,
                        "A0_tile_repeat_max_abs": repeat_error,
                        "A0_vs_first_operator_max_abs": cross_reference_max_abs,
                        "subsets": [],
                    }

                    route_order = route_order_cpu.to(device)
                    factors: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                    for route_count in CAPACITY_ROUTE_ROWS:
                        route_index = route_order[:route_count]
                        subset_design = operator_design.index_select(0, route_index)
                        factor, factor_stats = _ridge_factor(subset_design)
                        head_counts = torch.bincount(
                            torch.div(route_index, latents, rounding_mode="floor"),
                            minlength=heads,
                        )
                        slot_counts = torch.bincount(route_index % latents, minlength=latents)
                        factor_stats.update(
                            {
                                "route_rows": route_count,
                                "scalar_coordinates": route_count * patch,
                                "isotropic_projection_nrmse_floor": math.sqrt(
                                    1.0 - route_count / operator_design.shape[1]
                                ),
                                "head_count_min": int(head_counts.min()),
                                "head_count_max": int(head_counts.max()),
                                "slot_count_min": int(slot_counts.min()),
                                "slot_count_max": int(slot_counts.max()),
                            }
                        )
                        design_entry["subsets"].append(factor_stats)
                        factors[route_count] = (subset_design, factor, route_index)
                    design_rows.append(design_entry)

                    def add_result(
                        *,
                        scope: str,
                        variant: str,
                        method: str,
                        route_count: int,
                        family_left_out: str,
                        recovered: torch.Tensor,
                    ) -> None:
                        prediction = _patches_to_matrix(
                            recovered, d_in=d_in, d_out=d_out
                        )
                        metrics = _quality(target, prediction, loss_cfg=loss_cfg)
                        result_rows.append(
                            {
                                "scope": scope,
                                "variant": variant,
                                "method": method,
                                "route_rows": route_count,
                                "scalar_coordinates": route_count * patch,
                                "family_left_out": family_left_out,
                                "pair_index": pair_index,
                                "operator_parity": operator_parity,
                                "operator_id": operator_id,
                                **metrics,
                            }
                        )

                    full_design, full_factor, full_index = factors[heads * latents]
                    main_variants = (
                        "carrier_unscaled",
                        "adaptive_unscaled",
                        "carrier_contribution_0.9",
                        "adaptive_contribution_0.1",
                        "final_z_native",
                    )
                    for variant in main_variants:
                        operator_measurement = measurements[variant].index_select(
                            0, indices
                        ).index_select(1, full_index)
                        add_result(
                            scope="main_arm",
                            variant=variant,
                            method="ridge_dual",
                            route_count=heads * latents,
                            family_left_out="",
                            recovered=_synthesize_ridge(
                                full_design, full_factor, operator_measurement
                            ),
                        )
                        add_result(
                            scope="main_arm",
                            variant=variant,
                            method="coverage_normalized_transpose",
                            route_count=heads * latents,
                            family_left_out="",
                            recovered=_synthesize_transpose(
                                full_design, operator_measurement
                            ),
                        )
                    # SVD pinv is retained for a direct regression against the
                    # prior report; the scalable candidate remains ridge/transpose.
                    carrier_measurement = measurements["carrier_unscaled"].index_select(
                        0, indices
                    ).index_select(1, full_index)
                    add_result(
                        scope="main_arm_validation",
                        variant="carrier_unscaled",
                        method="svd_pinv",
                        route_count=heads * latents,
                        family_left_out="",
                        recovered=_synthesize_pinv(full_design, carrier_measurement),
                    )

                    for family in family_names:
                        variant = f"final_without_{family}"
                        operator_measurement = measurements[variant].index_select(
                            0, indices
                        ).index_select(1, full_index)
                        for method, recovered in (
                            (
                                "ridge_dual",
                                _synthesize_ridge(
                                    full_design, full_factor, operator_measurement
                                ),
                            ),
                            (
                                "coverage_normalized_transpose",
                                _synthesize_transpose(full_design, operator_measurement),
                            ),
                        ):
                            add_result(
                                scope="frozen_route_family_leaveout",
                                variant=variant,
                                method=method,
                                route_count=heads * latents,
                                family_left_out=family,
                                recovered=recovered,
                            )

                    full_carrier_measurement = measurements[
                        "carrier_unscaled"
                    ].index_select(0, indices)
                    for route_count in CAPACITY_ROUTE_ROWS:
                        subset_design, factor, route_index = factors[route_count]
                        subset_measurement = full_carrier_measurement.index_select(
                            1, route_index
                        )
                        for method, recovered in (
                            (
                                "ridge_dual",
                                _synthesize_ridge(
                                    subset_design, factor, subset_measurement
                                ),
                            ),
                            (
                                "coverage_normalized_transpose",
                                _synthesize_transpose(
                                    subset_design, subset_measurement
                                ),
                            ),
                        ):
                            add_result(
                                scope="A0_subset_capacity",
                                variant="carrier_unscaled",
                                method=method,
                                route_count=route_count,
                                family_left_out="",
                                recovered=recovered,
                            )

                    # Exact prospective V10 protected slice: 20 complete head
                    # channels x all 32 slots = 640 A0 rows (10,240 scalar
                    # coordinates), leaving four complete head channels free.
                    hc20_index = torch.arange(20 * latents, device=device)
                    hc20_design = operator_design.index_select(0, hc20_index)
                    hc20_factor, hc20_stats = _ridge_factor(hc20_design)
                    hc20_stats.update(
                        {
                            "route_rows": 20 * latents,
                            "scalar_coordinates": 20 * latents * patch,
                            "isotropic_projection_nrmse_floor": math.sqrt(
                                1.0 - (20 * latents) / operator_design.shape[1]
                            ),
                            "head_count_min": 0,
                            "head_count_max": latents,
                            "slot_count_min": 20,
                            "slot_count_max": 20,
                            "special_rule": "contiguous_heads_0_through_19_all_32_slots",
                        }
                    )
                    design_entry["prospective_Hc20_contiguous"] = hc20_stats
                    hc20_measurement = full_carrier_measurement.index_select(1, hc20_index)
                    for method, recovered in (
                        (
                            "ridge_dual",
                            _synthesize_ridge(
                                hc20_design, hc20_factor, hc20_measurement
                            ),
                        ),
                        (
                            "coverage_normalized_transpose",
                            _synthesize_transpose(hc20_design, hc20_measurement),
                        ),
                    ):
                        add_result(
                            scope="A0_Hc20_contiguous_capacity",
                            variant="carrier_unscaled",
                            method=method,
                            route_count=20 * latents,
                            family_left_out="",
                            recovered=recovered,
                        )
                    print(
                        f"stage=operator_done pair={pair_index} parity={operator_parity} "
                        f"rows={len(result_rows)}",
                        flush=True,
                    )

                pair_record = {
                    "pair_index": pair_index,
                    "operator_ids": [pair[0][0], pair[1][0]],
                    "elapsed_seconds": time.time() - started,
                    "parity": parity,
                    "result_row_count": len(result_rows),
                }
                report["partial_pairs"].append(pair_record)
                report["partial_result_rows"] = result_rows
                _atomic_json(report_path, report)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Strong integration gates.  BF16 manual routing can differ by one ULP;
        # native repeat/final/decode must be exact for this deterministic model.
        if max(row["native_carrier_repeat_max_abs"] for row in parity_rows) != 0.0:
            raise RuntimeError("native clean-carrier replay was not exact")
        if max(row["manual_final_z_max_abs"] for row in parity_rows) > 2.0e-4:
            raise RuntimeError("manual 0.9*c+0.1*a latent parity failed")
        # The separately captured 0.1*a tensor has one additional BF16 rounding
        # relative to the native expression; three BF16 ULPs are acceptable.
        if max(row["captured_components_final_z_max_abs"] for row in parity_rows) > 1.0e-3:
            raise RuntimeError("captured carrier/adaptive components do not reproduce final z")
        if max(row["manual_decode_max_abs"] for row in parity_rows) != 0.0:
            raise RuntimeError("manual/native decoder parity failed")

        aggregate = _aggregate(result_rows)
        report.pop("partial_result_rows", None)
        report["status"] = "complete_valid"
        report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["sections"] = {
            "native_manual_parity": parity_rows,
            "A0_design_and_capacity": design_rows,
            "readout_rows": result_rows,
            "aggregate_results": aggregate,
        }
        _atomic_csv(csv_path, result_rows)
        _atomic_json(report_path, report)
        manifest = {
            "schema": "weightclip_small_artifact_manifest_v1",
            "created_utc": report["completed_utc"],
            "files": {
                "report.json": {
                    "size": report_path.stat().st_size,
                    "sha256": _sha256(report_path),
                },
                "readout_results.csv": {
                    "size": csv_path.stat().st_size,
                    "sha256": _sha256(csv_path),
                },
            },
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "size": checkpoint_path.stat().st_size,
                "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            },
        }
        _atomic_json(manifest_path, manifest)
        print(
            f"stage=complete report={report_path} csv={csv_path} manifest={manifest_path}",
            flush=True,
        )
    except Exception as error:
        report["status"] = "failed_invalid"
        report["exception"] = repr(error)
        report["traceback"] = traceback.format_exc()
        _atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
