from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from big_vae.datasets.offline import OfflineBigVAEDataset
from big_vae.models import build_weight_quantile_vae
from training.big_vae.checkpointing import _normalize_model_state_dict_keys
from training.big_vae.runtime import _autocast_context, _build_model_cfg, _resolve_amp


DEFAULT_CHECKPOINT = Path(
    "/home/coder/project/projects/shared/storage/artifacts/training/checkpoints/"
    "weight_quantile_vae_gpu0_square/stage_1/latest.pt"
)
DEFAULT_SOURCE_ROOT = Path(
    "/home/coder/project/projects/shared/storage/artifacts/training/checkpoints/"
    "weight_quantile_vae/stage_1/offline_dataset"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/coder/project/artifacts/crossmodal_united_structure/"
    "source_full_matrix_gate_20260816"
)
ROLES = (
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_output",
    "ffn_up",
    "ffn_down",
)


@dataclass
class MatrixRecord:
    split: str
    depth: int
    role: str
    layer_name: str
    source_key: str
    context_ref: dict[str, Any]
    score_ref: dict[str, Any]
    W: torch.Tensor
    X_context: torch.Tensor
    X_score: torch.Tensor


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"expected a non-empty integer list, got {raw!r}")
    return values


def parse_str_list(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"expected a non-empty string list, got {raw!r}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def corrected_role(layer_name: str) -> str | None:
    name = layer_name.lower()
    if "attention.attention.query" in name or "attention.self.query" in name or name.endswith(".q_proj"):
        return "attn_query"
    if "attention.attention.key" in name or "attention.self.key" in name or name.endswith(".k_proj"):
        return "attn_key"
    if "attention.attention.value" in name or "attention.self.value" in name or name.endswith(".v_proj"):
        return "attn_value"
    if "attention.output.dense" in name or name.endswith(".out_proj") or name.endswith(".projection"):
        return "attn_output"
    if "intermediate.dense" in name or name.endswith(".mlp.fc1") or "feed_forward.intermediate_dense" in name:
        return "ffn_up"
    if (
        (name.endswith(".output.dense") and "attention.output.dense" not in name)
        or name.endswith(".mlp.fc2")
        or "feed_forward.output_dense" in name
    ):
        return "ffn_down"
    return None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scan_source_refs(
    *,
    source_root: Path,
    model_name: str,
    depths: Iterable[int],
    roles: Iterable[str],
    context_dataset: str,
    score_dataset: str,
) -> list[dict[str, Any]]:
    needed = {(int(depth), str(role)) for depth in depths for role in roles}
    refs_by_cell: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    index_paths = sorted((source_root / "chunk_index").glob("*.json"))
    if not index_paths:
        raise FileNotFoundError(f"no chunk indexes under {source_root / 'chunk_index'}")
    for chunk_idx, path in enumerate(index_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            if str(record.get("model_name", "")) != model_name:
                continue
            depth = int(record.get("layer_depth", -1))
            role = corrected_role(str(record.get("layer_name", "")))
            if (depth, role) not in needed:
                continue
            refs_by_cell[(depth, str(role))].append(
                {
                    **record,
                    "chunk_idx": int(chunk_idx),
                    "chunk_index_path": str(path),
                    "corrected_role": role,
                }
            )

    selected: list[dict[str, Any]] = []
    for depth, role in sorted(needed):
        refs = refs_by_cell.get((depth, role), [])
        if not refs:
            raise RuntimeError(f"missing source cell depth={depth} role={role}")
        identities = {
            (str(ref["source_key"]), str(ref["layer_name"]), tuple(ref["weight_shape"]))
            for ref in refs
        }
        if len(identities) != 1:
            raise RuntimeError(f"ambiguous matrix identity depth={depth} role={role}: {sorted(identities)}")
        contexts = sorted(
            (ref for ref in refs if str(ref.get("primary_dataset", "")) == context_dataset),
            key=lambda ref: (int(ref["chunk_idx"]), int(ref["record_idx"])),
        )
        scores = sorted(
            (ref for ref in refs if str(ref.get("primary_dataset", "")) == score_dataset),
            key=lambda ref: (int(ref["chunk_idx"]), int(ref["record_idx"])),
        )
        if not contexts or not scores:
            available = sorted({str(ref.get("primary_dataset", "")) for ref in refs})
            raise RuntimeError(
                f"missing context/score record depth={depth} role={role}; "
                f"context={context_dataset} score={score_dataset} available={available}"
            )
        context_ref = contexts[0]
        score_ref = scores[0]
        if (context_ref["chunk_idx"], context_ref["record_idx"]) == (
            score_ref["chunk_idx"],
            score_ref["record_idx"],
        ):
            raise RuntimeError(f"context and score refs unexpectedly coincide: depth={depth} role={role}")
        selected.append(
            {
                "depth": int(depth),
                "role": role,
                "layer_name": str(context_ref["layer_name"]),
                "source_key": str(context_ref["source_key"]),
                "weight_shape": list(context_ref["weight_shape"]),
                "context_ref": context_ref,
                "score_ref": score_ref,
            }
        )
    return selected


def load_matrix_records(
    *,
    dataset: OfflineBigVAEDataset,
    selected: list[dict[str, Any]],
    fit_depths: set[int],
    logger: logging.Logger,
) -> list[MatrixRecord]:
    result: list[MatrixRecord] = []
    for idx, cell in enumerate(selected):
        context_ref = cell["context_ref"]
        score_ref = cell["score_ref"]
        for ref in (context_ref, score_ref):
            runtime_record = dataset._load_x_chunk(int(ref["chunk_idx"]))["records"][int(ref["record_idx"])]
            if str(runtime_record.get("source_key", "")) != str(ref["source_key"]):
                raise RuntimeError(f"chunk-index/runtime source_key mismatch: {ref}")
        context_sample = dataset._shared_sample_from_record_ref(
            chunk_idx=int(context_ref["chunk_idx"]), record_idx=int(context_ref["record_idx"])
        )
        score_sample = dataset._shared_sample_from_record_ref(
            chunk_idx=int(score_ref["chunk_idx"]), record_idx=int(score_ref["record_idx"])
        )
        W_context = context_sample.weight.detach().cpu().to(torch.float32).contiguous()
        W_score = score_sample.weight.detach().cpu().to(torch.float32).contiguous()
        X_context = context_sample.x.detach().cpu().to(torch.float32).contiguous()
        X_score = score_sample.x.detach().cpu().to(torch.float32).contiguous()
        if not torch.equal(W_context, W_score):
            raise RuntimeError(
                f"context/score W mismatch depth={cell['depth']} role={cell['role']} "
                f"context_sha={tensor_sha256(W_context)} score_sha={tensor_sha256(W_score)}"
            )
        if tensor_sha256(X_context) == tensor_sha256(X_score):
            raise RuntimeError(f"context and score X are byte-identical: depth={cell['depth']} role={cell['role']}")
        if W_context.ndim != 2 or X_context.ndim != 2 or X_score.ndim != 2:
            raise ValueError(f"invalid ranks W={W_context.shape} Xc={X_context.shape} Xs={X_score.shape}")
        if X_context.shape[1] != W_context.shape[0] or X_score.shape[1] != W_context.shape[0]:
            raise ValueError(f"W/X dimension mismatch W={W_context.shape} Xc={X_context.shape} Xs={X_score.shape}")
        if W_context.shape[0] % 64 or W_context.shape[1] % 64:
            raise ValueError(f"gate requires complete 64x64 tiling, got {tuple(W_context.shape)}")
        record = MatrixRecord(
            split="fit" if int(cell["depth"]) in fit_depths else "holdout",
            depth=int(cell["depth"]),
            role=str(cell["role"]),
            layer_name=str(cell["layer_name"]),
            source_key=str(cell["source_key"]),
            context_ref=context_ref,
            score_ref=score_ref,
            W=W_context,
            X_context=X_context,
            X_score=X_score,
        )
        result.append(record)
        logger.info(
            "stage=data_loading matrix=%s/%s split=%s depth=%s role=%s W=%s X_context=%s X_score=%s refs=(%s:%s,%s:%s)",
            idx + 1,
            len(selected),
            record.split,
            record.depth,
            record.role,
            tuple(record.W.shape),
            tuple(record.X_context.shape),
            tuple(record.X_score.shape),
            context_ref["chunk_idx"],
            context_ref["record_idx"],
            score_ref["chunk_idx"],
            score_ref["record_idx"],
        )
    return result


def tile_matrix(W: torch.Tensor, tile_size: int = 64) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    if W.ndim != 2 or W.shape[0] % tile_size or W.shape[1] % tile_size:
        raise ValueError(f"cannot completely tile W={tuple(W.shape)} with tile_size={tile_size}")
    tiles: list[torch.Tensor] = []
    coords: list[tuple[int, int]] = []
    for row_start in range(0, int(W.shape[0]), tile_size):
        for col_start in range(0, int(W.shape[1]), tile_size):
            tiles.append(W[row_start : row_start + tile_size, col_start : col_start + tile_size])
            coords.append((row_start, col_start))
    return torch.stack(tiles), coords


def reassemble_tiles(
    tiles: torch.Tensor,
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    tile_size: int = 64,
) -> torch.Tensor:
    if int(tiles.shape[0]) != len(coords):
        raise ValueError(f"tiles/coords mismatch: {tiles.shape} vs {len(coords)}")
    result = torch.empty(shape, dtype=torch.float32)
    coverage = torch.zeros(shape, dtype=torch.int16)
    for tile, (row_start, col_start) in zip(tiles, coords, strict=True):
        result[row_start : row_start + tile_size, col_start : col_start + tile_size] = tile.float().cpu()
        coverage[row_start : row_start + tile_size, col_start : col_start + tile_size] += 1
    if not torch.all(coverage == 1):
        values, counts = torch.unique(coverage, return_counts=True)
        raise RuntimeError(f"invalid reassembly coverage: {list(zip(values.tolist(), counts.tolist(), strict=True))}")
    return result


@torch.inference_mode()
def compute_exchangeable_c0(
    *,
    model: torch.nn.Module,
    fit_records: list[MatrixRecord],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> dict[str, torch.Tensor]:
    role_vectors: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for record_idx, record in enumerate(fit_records):
        row_vectors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for row_start in range(0, int(record.W.shape[0]), 64):
            X_tile = record.X_context[:, row_start : row_start + 64].unsqueeze(0).to(device)
            x_mask = torch.ones((1, int(X_tile.shape[1])), dtype=torch.bool, device=device)
            d_in_mask = torch.ones((1, 64), dtype=torch.bool, device=device)
            with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                outputs = model._encode_distribution_context(
                    X_tile,
                    x_mask=x_mask,
                    d_in_mask=d_in_mask,
                )
            _T, _d_in_pad, _patch_mask, _struct_mask, c_var, c_patch, c_pooled = outputs
            if c_var is None or c_patch is None or c_pooled is None:
                raise RuntimeError("distribution encoder unexpectedly returned None")
            row_vectors.append(
                (
                    c_var.float().mean(dim=(0, 1, 2)).cpu(),
                    c_patch.float().mean(dim=(0, 1)).cpu(),
                    c_pooled.float().mean(dim=(0, 1)).cpu(),
                )
            )
        role_vectors[record.role].append(
            tuple(torch.stack([values[i] for values in row_vectors]).mean(dim=0) for i in range(3))
        )
        logger.info(
            "stage=c0_extraction matrix=%s/%s role=%s depth=%s row_tiles=%s elapsed_complete=true",
            record_idx + 1,
            len(fit_records),
            record.role,
            record.depth,
            len(row_vectors),
        )
    role_means: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for role, values in role_vectors.items():
        role_means[role] = tuple(torch.stack([value[i] for value in values]).mean(dim=0) for i in range(3))
    missing_roles = sorted(set(record.role for record in fit_records) - set(role_means))
    if missing_roles:
        raise RuntimeError(f"missing C0 role means: {missing_roles}")
    # Equal role weighting prevents large FFN-down matrices from dominating the
    # source template merely because they have more input-coordinate tiles.
    ordered_roles = sorted(role_means)
    result = {
        "c_var": torch.stack([role_means[role][0] for role in ordered_roles]).mean(dim=0),
        "c_patch": torch.stack([role_means[role][1] for role in ordered_roles]).mean(dim=0),
        "c_pooled": torch.stack([role_means[role][2] for role in ordered_roles]).mean(dim=0),
    }
    for key, value in result.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite {key}")
    return result


@torch.inference_mode()
def fixed_c_forward(
    *,
    model: torch.nn.Module,
    W: torch.Tensor,
    c0: dict[str, torch.Tensor],
) -> torch.Tensor:
    if W.ndim != 3 or tuple(W.shape[1:]) != (64, 64):
        raise ValueError(f"fixed gate expects [B,64,64], got {tuple(W.shape)}")
    B = int(W.shape[0])
    d_in_mask = torch.ones((B, 64), dtype=torch.bool, device=W.device)
    d_out_mask = torch.ones((B, 64), dtype=torch.bool, device=W.device)
    _patch_idx, patch_mask, _structural_patch_mask, _valid_d_in, T, d_in_pad = model._build_batched_patch_indices(
        d_in_mask=d_in_mask,
        patch_size=int(model.cfg.patch_size),
    )
    p = int(model.cfg.patch_size)
    c_var = c0["c_var"].to(device=W.device, dtype=W.dtype).view(1, 1, 1, -1).expand(B, T, p, -1)
    c_patch = c0["c_patch"].to(device=W.device, dtype=W.dtype).view(1, 1, -1).expand(B, T, -1)
    c_pooled = c0["c_pooled"].to(device=W.device, dtype=W.dtype).view(1, 1, -1).expand(B, T, -1)
    latent_slots, debug = model._encode_latent_slots(
        W,
        T=T,
        d_in_pad=d_in_pad,
        patch_mask=patch_mask,
        d_out_mask=d_out_mask,
        dist_var_by_patch=c_var,
        dist_patch_by_patch=c_patch,
        dist_var_pooled=c_pooled,
        return_debug_info=True,
    )
    W_hat, _mu, _logvar, _pred_dirs = model._decode_from_latent_slots(
        latent_slots,
        dist_patch_by_patch=c_patch,
        patch_mask=patch_mask,
        d_in_mask=d_in_mask,
        d_out_mask=d_out_mask,
        d_in=64,
        d_out=64,
        d_in_pad=d_in_pad,
        T=T,
        encoder_patch_tokens=debug["encoder_patch_tokens"],
    )
    return W_hat


@torch.inference_mode()
def decode_full_matrix(
    *,
    model: torch.nn.Module,
    record: MatrixRecord,
    condition: str,
    c0: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: logging.Logger,
) -> torch.Tensor:
    tiles, coords = tile_matrix(record.W)
    predictions: list[torch.Tensor] = []
    total_batches = math.ceil(int(tiles.shape[0]) / batch_size)
    start_time = time.monotonic()
    for batch_idx, begin in enumerate(range(0, int(tiles.shape[0]), batch_size)):
        end = min(begin + batch_size, int(tiles.shape[0]))
        W_batch = tiles[begin:end].to(device=device, non_blocking=True)
        with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
            if condition == "source_mean_c0":
                W_hat = fixed_c_forward(model=model, W=W_batch, c0=c0)
            elif condition == "native_c":
                X_batch = torch.stack(
                    [
                        record.X_context[:, row_start : row_start + 64]
                        for row_start, _col_start in coords[begin:end]
                    ]
                ).to(device=device, non_blocking=True)
                x_mask = torch.ones(
                    (int(X_batch.shape[0]), int(X_batch.shape[1])), dtype=torch.bool, device=device
                )
                d_in_mask = torch.ones((int(X_batch.shape[0]), 64), dtype=torch.bool, device=device)
                d_out_mask = torch.ones((int(X_batch.shape[0]), 64), dtype=torch.bool, device=device)
                W_hat, _mu, _logvar, _pred_dirs = model(
                    W_batch,
                    X_batch,
                    x_mask=x_mask,
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                )
            else:
                raise ValueError(f"unknown condition: {condition}")
        predictions.append(W_hat.float().cpu())
        if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
            elapsed = time.monotonic() - start_time
            logger.info(
                "stage=ae_decode condition=%s split=%s depth=%s role=%s batch=%s/%s tiles=%s/%s elapsed_s=%.2f rate_tiles_s=%.2f",
                condition,
                record.split,
                record.depth,
                record.role,
                batch_idx + 1,
                total_batches,
                end,
                len(coords),
                elapsed,
                end / max(elapsed, 1e-6),
            )
    result = reassemble_tiles(torch.cat(predictions, dim=0), coords, tuple(record.W.shape))
    if result.shape != record.W.shape or not torch.isfinite(result).all():
        raise RuntimeError(f"invalid reassembled prediction shape={result.shape} finite={torch.isfinite(result).all()}")
    return result


def fit_fast_baselines(
    fit_records: list[MatrixRecord], *, code_dim: int, seed: int
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for role in sorted({record.role for record in fit_records}):
        role_tiles = torch.cat([tile_matrix(record.W)[0].reshape(-1, 64 * 64) for record in fit_records if record.role == role])
        mean = role_tiles.mean(dim=0)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + sum(ord(char) for char in role))
        coordinate_indices = torch.randperm(64 * 64, generator=generator)[:code_dim].sort().values
        result[role] = {"mean": mean, "coordinate_indices": coordinate_indices}
    return result


def baseline_prediction(
    *, record: MatrixRecord, method: str, baseline_state: dict[str, dict[str, torch.Tensor]]
) -> torch.Tensor:
    if method == "zero":
        return torch.zeros_like(record.W)
    tiles, coords = tile_matrix(record.W)
    flat = tiles.reshape(int(tiles.shape[0]), -1)
    role_state = baseline_state[record.role]
    mean = role_state["mean"].view(1, -1)
    if method == "role_mean":
        pred_flat = mean.expand_as(flat).clone()
    elif method.startswith("random_coordinate_"):
        indices = role_state["coordinate_indices"]
        pred_flat = mean.expand_as(flat).clone()
        pred_flat[:, indices] = flat[:, indices]
    else:
        raise ValueError(f"unknown baseline method: {method}")
    return reassemble_tiles(pred_flat.view(-1, 64, 64), coords, tuple(record.W.shape))


def raw_sufficient_stats(W: torch.Tensor, W_hat: torch.Tensor, X_score: torch.Tensor) -> dict[str, float]:
    W64 = W.to(torch.float64)
    P64 = W_hat.to(torch.float64)
    X64 = X_score.to(torch.float64)
    target = X64 @ W64
    pred = X64 @ P64
    return {
        "weight_sq_error": float((W64 - P64).square().sum()),
        "weight_target_energy": float(W64.square().sum()),
        "weight_pred_energy": float(P64.square().sum()),
        "weight_dot": float((W64 * P64).sum()),
        "operator_sq_error": float((target - pred).square().sum()),
        "operator_target_energy": float(target.square().sum()),
        "operator_pred_energy": float(pred.square().sum()),
        "operator_dot": float((target * pred).sum()),
    }


def fit_role_gains(
    fit_predictions: list[tuple[MatrixRecord, dict[str, torch.Tensor]]]
) -> dict[tuple[str, str], float]:
    dots: dict[tuple[str, str], float] = defaultdict(float)
    energies: dict[tuple[str, str], float] = defaultdict(float)
    for record, predictions in fit_predictions:
        for method, prediction in predictions.items():
            stats = raw_sufficient_stats(record.W, prediction, record.X_score)
            key = (method, record.role)
            dots[key] += stats["operator_dot"]
            energies[key] += stats["operator_pred_energy"]
    return {
        key: (dots[key] / energies[key] if energies[key] > 1e-30 else 0.0)
        for key in sorted(dots)
    }


def metric_row(
    *,
    record: MatrixRecord,
    method: str,
    prediction: torch.Tensor,
    gain: float,
    code_dim: int,
) -> dict[str, Any]:
    raw = raw_sufficient_stats(record.W, prediction, record.X_score)
    scaled_prediction = float(gain) * prediction
    scaled = raw_sufficient_stats(record.W, scaled_prediction, record.X_score)
    eps = 1e-30
    return {
        "split": record.split,
        "method": method,
        "depth": record.depth,
        "role": record.role,
        "layer_name": record.layer_name,
        "source_key": record.source_key,
        "weight_shape": "x".join(str(int(v)) for v in record.W.shape),
        "context_dataset": record.context_ref["primary_dataset"],
        "score_dataset": record.score_ref["primary_dataset"],
        "num_tiles": (int(record.W.shape[0]) // 64) * (int(record.W.shape[1]) // 64),
        "code_dim_per_tile": int(code_dim),
        "source_fit_role_gain": float(gain),
        "raw_weight_relative_mse": raw["weight_sq_error"] / max(raw["weight_target_energy"], eps),
        "raw_weight_cosine": raw["weight_dot"] / math.sqrt(max(raw["weight_target_energy"] * raw["weight_pred_energy"], eps)),
        "raw_weight_norm_ratio": math.sqrt(raw["weight_pred_energy"] / max(raw["weight_target_energy"], eps)),
        "raw_operator_relative_error": raw["operator_sq_error"] / max(raw["operator_target_energy"], eps),
        "raw_operator_cosine": raw["operator_dot"] / math.sqrt(max(raw["operator_target_energy"] * raw["operator_pred_energy"], eps)),
        "raw_operator_norm_ratio": math.sqrt(raw["operator_pred_energy"] / max(raw["operator_target_energy"], eps)),
        "scaled_weight_relative_mse": scaled["weight_sq_error"] / max(scaled["weight_target_energy"], eps),
        "scaled_operator_relative_error": scaled["operator_sq_error"] / max(scaled["operator_target_energy"], eps),
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"scaled_{key}": value for key, value in scaled.items()},
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["method"]), "__all__")].append(row)
        groups[(str(row["split"]), str(row["method"]), str(row["role"]))].append(row)
    result: list[dict[str, Any]] = []
    for (split, method, role), group in sorted(groups.items()):
        raw_w_num = sum(float(row["raw_weight_sq_error"]) for row in group)
        raw_w_den = sum(float(row["raw_weight_target_energy"]) for row in group)
        raw_w_pred = sum(float(row["raw_weight_pred_energy"]) for row in group)
        raw_w_dot = sum(float(row["raw_weight_dot"]) for row in group)
        raw_o_num = sum(float(row["raw_operator_sq_error"]) for row in group)
        raw_o_den = sum(float(row["raw_operator_target_energy"]) for row in group)
        raw_o_pred = sum(float(row["raw_operator_pred_energy"]) for row in group)
        raw_o_dot = sum(float(row["raw_operator_dot"]) for row in group)
        scaled_w_num = sum(float(row["scaled_weight_sq_error"]) for row in group)
        scaled_o_num = sum(float(row["scaled_operator_sq_error"]) for row in group)
        result.append(
            {
                "split": split,
                "method": method,
                "role": role,
                "num_matrices": len(group),
                "num_tiles": sum(int(row["num_tiles"]) for row in group),
                "raw_weight_relative_mse": raw_w_num / max(raw_w_den, 1e-30),
                "raw_weight_cosine": raw_w_dot / math.sqrt(max(raw_w_den * raw_w_pred, 1e-30)),
                "raw_weight_norm_ratio": math.sqrt(raw_w_pred / max(raw_w_den, 1e-30)),
                "raw_operator_relative_error": raw_o_num / max(raw_o_den, 1e-30),
                "raw_operator_cosine": raw_o_dot / math.sqrt(max(raw_o_den * raw_o_pred, 1e-30)),
                "raw_operator_norm_ratio": math.sqrt(raw_o_pred / max(raw_o_den, 1e-30)),
                "scaled_weight_relative_mse": scaled_w_num / max(raw_w_den, 1e-30),
                "scaled_operator_relative_error": scaled_o_num / max(raw_o_den, 1e-30),
            }
        )
    return result


def plot_holdout(aggregate_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in aggregate_rows if row["split"] == "holdout" and row["role"] == "__all__"]
    rows.sort(key=lambda row: float(row["scaled_operator_relative_error"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [str(row["method"]) for row in rows]
    values = [float(row["scaled_operator_relative_error"]) for row in rows]
    bars = ax.bar(labels, values, color=["#2878b5" if label.startswith("ae_") else "#b8b8b8" for label in labels])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="zero reconstruction")
    ax.set_ylabel("Held-out native relative operator error (source-fit role gain)")
    ax.set_title("Source full-matrix gate: depth holdout")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="best")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_raw_holdout(aggregate_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot gain-free endpoints so invalid calibration cannot dominate review."""
    import matplotlib.pyplot as plt

    rows = [row for row in aggregate_rows if row["split"] == "holdout" and row["role"] == "__all__"]
    rows.sort(key=lambda row: float(row["raw_operator_relative_error"]))
    labels = [str(row["method"]) for row in rows]
    errors = [float(row["raw_operator_relative_error"]) for row in rows]
    cosines = [float(row["raw_operator_cosine"]) for row in rows]
    colors = ["#2878b5" if label.startswith("ae_") else "#b8b8b8" for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    error_bars = axes[0].bar(labels, errors, color=colors)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="zero reconstruction")
    axes[0].set_ylabel("Raw relative operator error")
    axes[0].set_title("Gain-free native functional error")
    axes[0].legend(loc="best")
    for bar, value in zip(error_bars, errors, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    cosine_bars = axes[1].bar(labels, cosines, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("Raw operator cosine")
    axes[1].set_title("Gain-free operator alignment")
    for bar, value in zip(cosine_bars, cosines, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("BEiT contiguous full-matrix engineering probe: odd-depth holdout")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy-safe source-only full-matrix Weight-AE validity gate")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="beit_base")
    parser.add_argument("--fit-depths", default="0")
    parser.add_argument("--holdout-depths", default="1")
    parser.add_argument("--roles", default=",".join(ROLES))
    parser.add_argument("--context-dataset", default="dtd_textures")
    parser.add_argument("--score-dataset", default="cc12m")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--random-code-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=260816)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    fit_depths = parse_int_list(args.fit_depths)
    holdout_depths = parse_int_list(args.holdout_depths)
    roles = parse_str_list(args.roles)
    if set(fit_depths) & set(holdout_depths):
        raise ValueError(f"fit/holdout depths overlap: {fit_depths} vs {holdout_depths}")
    if not set(roles).issubset(ROLES):
        raise ValueError(f"unknown roles: {sorted(set(roles) - set(ROLES))}")
    if not 0 < int(args.random_code_dim) <= 64 * 64:
        raise ValueError(f"random-code-dim must be in [1,4096], got {args.random_code_dim}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
        force=True,
    )
    logger = logging.getLogger("source_full_matrix_validity_gate")
    start_time = time.monotonic()
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    resolved = {
        "checkpoint": str(args.checkpoint.resolve()),
        "source_root": str(args.source_root.resolve()),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "fit_depths": list(fit_depths),
        "holdout_depths": list(holdout_depths),
        "roles": list(roles),
        "context_dataset": args.context_dataset,
        "score_dataset": args.score_dataset,
        "batch_size": int(args.batch_size),
        "random_code_dim": int(args.random_code_dim),
        "seed": int(args.seed),
        "device": str(device),
        "dtype": "checkpoint-resolved AMP unless --no-amp",
        "cache_mode": "offline chunk cache; deterministic unshuffled exact refs",
        "tiling": "complete contiguous non-overlapping 64x64; exact one-hit reassembly",
        "conditioning": ["source_mean_c0", "native_c"],
        "gain_fit": "per method and corrected role on fit depths, native operator least squares",
        "target_metrics": "sealed; no target data2vec code path in this script",
    }
    write_json(output_dir / "resolved_config.json", resolved)
    logger.info("resolved_config=%s", json.dumps(resolved, sort_keys=True))

    logger.info("stage=checkpoint_load")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise TypeError(f"invalid checkpoint payload: {args.checkpoint}")
    ckpt_cfg = OmegaConf.create(payload["config"])
    model_cfg = _build_model_cfg(ckpt_cfg)
    model_cfg.big_vae.use_latent_sampling = False
    model_cfg.big_vae.rope_2d_coord_kind = "raw"
    model = build_weight_quantile_vae(model_cfg).to(device)
    state = _normalize_model_state_dict_keys(payload["model_state"])
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    amp_enabled, amp_dtype = _resolve_amp(ckpt_cfg, device)
    amp_enabled = bool(amp_enabled and not args.no_amp)
    logger.info(
        "checkpoint path=%s sha256=%s step=%s stage=%s params=%s rope=%s deterministic_ae=%s amp_enabled=%s amp_dtype=%s",
        args.checkpoint,
        sha256_file(args.checkpoint),
        payload.get("step"),
        payload.get("stage"),
        sum(parameter.numel() for parameter in model.parameters()),
        model.cfg.big_vae.rope_2d_coord_kind,
        not bool(model.cfg.big_vae.use_latent_sampling),
        amp_enabled,
        amp_dtype,
    )

    logger.info("stage=source_manifest_scan")
    selected = scan_source_refs(
        source_root=args.source_root,
        model_name=args.model_name,
        depths=tuple(fit_depths) + tuple(holdout_depths),
        roles=roles,
        context_dataset=args.context_dataset,
        score_dataset=args.score_dataset,
    )
    write_json(output_dir / "selected_records.json", selected)
    dataset = OfflineBigVAEDataset(
        root_dir=args.source_root,
        shuffle_chunks=False,
        shuffle_records_within_chunk=False,
        repeat=False,
        seed=args.seed,
        weight_cache_size=64,
        sampling_mode="balanced",
        sampling_group_keys=("dataset", "model"),
        sampling_window_size=2048,
        sampling_max_records_per_chunk_round=8,
        x_chunk_cache_size=16,
    )
    records = load_matrix_records(
        dataset=dataset,
        selected=selected,
        fit_depths=set(fit_depths),
        logger=logger,
    )
    fit_records = [record for record in records if record.split == "fit"]
    holdout_records = [record for record in records if record.split == "holdout"]
    if not fit_records or not holdout_records:
        raise RuntimeError(f"empty fit/holdout split: fit={len(fit_records)} holdout={len(holdout_records)}")

    logger.info("stage=c0_fit fit_matrices=%s", len(fit_records))
    c0 = compute_exchangeable_c0(
        model=model,
        fit_records=fit_records,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        logger=logger,
    )
    torch.save({key: value.cpu() for key, value in c0.items()}, output_dir / "source_mean_c0.pt")
    c0_summary = {
        key: {
            "shape": list(value.shape),
            "mean": float(value.mean()),
            "std": float(value.std()),
            "norm": float(value.norm()),
            "sha256": tensor_sha256(value),
        }
        for key, value in c0.items()
    }
    write_json(output_dir / "source_mean_c0_summary.json", c0_summary)
    logger.info("c0_summary=%s artifact=%s", c0_summary, output_dir / "source_mean_c0.pt")

    logger.info("stage=baseline_fit")
    baseline_state = fit_fast_baselines(fit_records, code_dim=args.random_code_dim, seed=args.seed)
    torch.save(baseline_state, output_dir / "fast_baseline_state.pt")
    methods = (
        "ae_source_mean_c0",
        "ae_native_c",
        "role_mean",
        f"random_coordinate_{args.random_code_dim}",
        "zero",
    )
    code_dims = {
        "ae_source_mean_c0": int(model.cfg.big_vae.num_latents) * int(model.cfg.big_vae.d_lat),
        "ae_native_c": int(model.cfg.big_vae.num_latents) * int(model.cfg.big_vae.d_lat),
        "role_mean": 0,
        f"random_coordinate_{args.random_code_dim}": int(args.random_code_dim),
        "zero": 0,
    }

    fit_predictions: list[tuple[MatrixRecord, dict[str, torch.Tensor]]] = []
    for record_idx, record in enumerate(fit_records):
        logger.info(
            "stage=fit_predictions matrix=%s/%s depth=%s role=%s",
            record_idx + 1,
            len(fit_records),
            record.depth,
            record.role,
        )
        predictions = {
            "ae_source_mean_c0": decode_full_matrix(
                model=model,
                record=record,
                condition="source_mean_c0",
                c0=c0,
                device=device,
                batch_size=args.batch_size,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                logger=logger,
            ),
            "ae_native_c": decode_full_matrix(
                model=model,
                record=record,
                condition="native_c",
                c0=c0,
                device=device,
                batch_size=args.batch_size,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                logger=logger,
            ),
            "role_mean": baseline_prediction(record=record, method="role_mean", baseline_state=baseline_state),
            f"random_coordinate_{args.random_code_dim}": baseline_prediction(
                record=record,
                method=f"random_coordinate_{args.random_code_dim}",
                baseline_state=baseline_state,
            ),
            "zero": torch.zeros_like(record.W),
        }
        fit_predictions.append((record, predictions))

    logger.info("stage=gain_fit")
    gains = fit_role_gains(fit_predictions)
    gain_rows = [
        {"method": method, "role": role, "operator_least_squares_gain": gain}
        for (method, role), gain in sorted(gains.items())
    ]
    write_csv(output_dir / "source_fit_role_gains.csv", gain_rows)
    logger.info("source_fit_role_gains=%s", gain_rows)

    metric_rows: list[dict[str, Any]] = []
    for record, predictions in fit_predictions:
        for method, prediction in predictions.items():
            metric_rows.append(
                metric_row(
                    record=record,
                    method=method,
                    prediction=prediction,
                    gain=gains[(method, record.role)],
                    code_dim=code_dims[method],
                )
            )
    del fit_predictions

    logger.info("stage=holdout_evaluation holdout_matrices=%s", len(holdout_records))
    for record_idx, record in enumerate(holdout_records):
        logger.info(
            "stage=holdout_predictions matrix=%s/%s depth=%s role=%s",
            record_idx + 1,
            len(holdout_records),
            record.depth,
            record.role,
        )
        predictions = {
            "ae_source_mean_c0": decode_full_matrix(
                model=model,
                record=record,
                condition="source_mean_c0",
                c0=c0,
                device=device,
                batch_size=args.batch_size,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                logger=logger,
            ),
            "ae_native_c": decode_full_matrix(
                model=model,
                record=record,
                condition="native_c",
                c0=c0,
                device=device,
                batch_size=args.batch_size,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                logger=logger,
            ),
            "role_mean": baseline_prediction(record=record, method="role_mean", baseline_state=baseline_state),
            f"random_coordinate_{args.random_code_dim}": baseline_prediction(
                record=record,
                method=f"random_coordinate_{args.random_code_dim}",
                baseline_state=baseline_state,
            ),
            "zero": torch.zeros_like(record.W),
        }
        for method, prediction in predictions.items():
            row = metric_row(
                record=record,
                method=method,
                prediction=prediction,
                gain=gains[(method, record.role)],
                code_dim=code_dims[method],
            )
            metric_rows.append(row)
            logger.info(
                "metric split=holdout method=%s depth=%s role=%s raw_e=%.6f scaled_e=%.6f op_cos=%.6f raw_weight_rel=%.6f scaled_weight_rel=%.6f gain=%.6f",
                method,
                record.depth,
                record.role,
                row["raw_operator_relative_error"],
                row["scaled_operator_relative_error"],
                row["raw_operator_cosine"],
                row["raw_weight_relative_mse"],
                row["scaled_weight_relative_mse"],
                row["source_fit_role_gain"],
            )

    aggregate_rows = aggregate_metrics(metric_rows)
    write_csv(output_dir / "matrix_metrics.csv", metric_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    plot_holdout(aggregate_rows, output_dir / "holdout_operator_error.png")
    plot_raw_holdout(aggregate_rows, output_dir / "holdout_raw_operator_metrics.png")
    overall_holdout = {
        row["method"]: row
        for row in aggregate_rows
        if row["split"] == "holdout" and row["role"] == "__all__"
    }
    fixed_e = float(overall_holdout["ae_source_mean_c0"]["scaled_operator_relative_error"])
    native_e = float(overall_holdout["ae_native_c"]["scaled_operator_relative_error"])
    random_e = float(overall_holdout[f"random_coordinate_{args.random_code_dim}"]["scaled_operator_relative_error"])
    mean_e = float(overall_holdout["role_mean"]["scaled_operator_relative_error"])
    decision = {
        "status": "ENGINEERING_PROBE_ONLY",
        "criterion": "No scientific pass/fail: contiguous BEiT probe is not the preregistered held-out-model source gate.",
        "ae_source_mean_c0_scaled_operator_error": fixed_e,
        "ae_native_c_scaled_operator_error": native_e,
        "random_coordinate_scaled_operator_error": random_e,
        "role_mean_scaled_operator_error": mean_e,
        "caveat": (
            f"Fit depths={list(fit_depths)} and holdout depths={list(holdout_depths)} are within the same BEiT checkpoint. "
            "Contiguous tiling, single-model C0 fitting, random-coordinate rather than random-orthoprojector, and no PCA mean this run cannot unseal targets."
        ),
    }
    manifest = {
        "resolved_config": resolved,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "step": int(payload.get("step", 0) or 0),
            "stage": int(payload.get("stage", 0) or 0),
            "rope_2d_coord_kind": str(model.cfg.big_vae.rope_2d_coord_kind),
            "use_latent_sampling": bool(model.cfg.big_vae.use_latent_sampling),
        },
        "counts": {
            "fit_matrices": len(fit_records),
            "holdout_matrices": len(holdout_records),
            "metric_rows": len(metric_rows),
            "aggregate_rows": len(aggregate_rows),
        },
        "decision": decision,
        "elapsed_seconds": time.monotonic() - start_time,
        "artifacts": {
            "log": str(log_path),
            "resolved_config": str(output_dir / "resolved_config.json"),
            "selected_records": str(output_dir / "selected_records.json"),
            "c0": str(output_dir / "source_mean_c0.pt"),
            "c0_summary": str(output_dir / "source_mean_c0_summary.json"),
            "baseline_state": str(output_dir / "fast_baseline_state.pt"),
            "gains": str(output_dir / "source_fit_role_gains.csv"),
            "matrix_metrics": str(output_dir / "matrix_metrics.csv"),
            "aggregate_metrics": str(output_dir / "aggregate_metrics.csv"),
            "plot": str(output_dir / "holdout_operator_error.png"),
            "raw_plot": str(output_dir / "holdout_raw_operator_metrics.png"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    logger.info("stage=output_writing decision=%s", decision)
    logger.info("artifacts=%s", manifest["artifacts"])
    logger.info("completed elapsed_s=%.2f", manifest["elapsed_seconds"])


if __name__ == "__main__":
    main()
