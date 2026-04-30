from __future__ import annotations

import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

from hydra import compose, initialize_config_dir
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    GroupedMetrics,
    apply_heldout_log_dir,
    coverage_report,
    env_bool,
    env_int,
    env_path,
    finite_metrics,
    primary_dataset_name,
    promote_run_profile_to_root,
    sanitize_programmatic_hydra_logging,
    tensor_to_float,
    write_csv,
    write_json,
)
from dataset.big_vae_offline import OfflineBigVAEDataset, infer_layer_depth, infer_layer_type
from dataset.logging_utils import configure_process_logging
from experiments.train_big_vae import (
    SourceSampleRecord,
    _autocast_context,
    _build_model_cfg,
    _build_training_batch_from_source_states,
    _compute_curriculum_slice_sizes,
    _make_source_slice_state,
    _normalize_model_state_dict_keys,
    _remaining_source_state_slices,
    _resolve_amp,
    _stable_batch_shape_targets,
)
from models.weight_quantile_vae import WeightQuantileVAE, build_weight_quantile_vae
from training.runtime import resolve_device as runtime_resolve_device


LOGGER = logging.getLogger("evaluate_big_vae_heldout")
_LATENT_DUMP_BALANCE_KEY_DEFAULT = "dataset,model,layer_type,depth_label"


def _compute_model_latent_kl(model: torch.nn.Module, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    latent_kl = getattr(model, "latent_kl_loss", None)
    if callable(latent_kl):
        return latent_kl(mu, logvar)
    return WeightQuantileVAE.kl_loss(mu, logvar)


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, DictConfig, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload)!r}: {checkpoint_path}")
    cfg_payload = payload.get("config")
    if not isinstance(cfg_payload, dict):
        raise KeyError(f"Checkpoint does not contain a dict 'config': {checkpoint_path}")
    ckpt_cfg = OmegaConf.create(cfg_payload)
    model = build_weight_quantile_vae(_build_model_cfg(ckpt_cfg)).to(device)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint does not contain a dict 'model_state': {checkpoint_path}")
    model.load_state_dict(_normalize_model_state_dict_keys(state), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt_cfg, payload


def _resolve_eval_device(runtime_cfg: DictConfig) -> torch.device:
    env_device = str(os.environ.get("EVAL_DEVICE", "")).strip()
    if env_device:
        return torch.device(env_device)
    return runtime_resolve_device(runtime_cfg, rank=0, world_size=1, section="train")


def _source_record_from_sample(sample: Any, device: torch.device) -> SourceSampleRecord:
    x = getattr(sample, "x", None)
    W = getattr(sample, "weight", None)
    if not (
        torch.is_tensor(x)
        and torch.is_tensor(W)
        and x.ndim == 2
        and W.ndim == 2
        and int(x.shape[1]) == int(W.shape[0])
    ):
        raise ValueError(
            "offline sample has invalid x/W shapes: "
            f"x={tuple(x.shape) if torch.is_tensor(x) else type(x)} "
            f"W={tuple(W.shape) if torch.is_tensor(W) else type(W)}"
        )
    return SourceSampleRecord(
        x=x.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        W=W.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        model_name=str(getattr(sample, "model_name", "") or "").strip(),
        layer_name=str(getattr(sample, "layer_name", "") or "").strip(),
    )


def _loss_coefficients(cfg: DictConfig) -> dict[str, float]:
    train_cfg = cfg.get("train", {})
    behavioral_cfg = train_cfg.get("behavioral_loss", {})
    if behavioral_cfg is None:
        behavioral_cfg = {}
    struct_cfg = train_cfg.get("struct_loss", {})
    if struct_cfg is None:
        struct_cfg = {}
    return {
        "behavioral_coef": float(train_cfg.get("behavioral_coef", 1.0)),
        "structural_coef": float(train_cfg.get("structural_coef", 0.5)),
        "kl_beta": float(train_cfg.get("kl_beta", 0.0)),
        "behavioral_lambda_operator": float(behavioral_cfg.get("lambda_operator", 1.0)),
        "behavioral_lambda_dir": float(behavioral_cfg.get("lambda_dir", 0.0)),
        "behavioral_lambda_scale": float(behavioral_cfg.get("lambda_scale", 0.0)),
        "behavioral_gamma": float(behavioral_cfg.get("gamma", 0.5)),
        "behavioral_huber_delta": float(behavioral_cfg.get("huber_delta", 0.1)),
        "struct_lambda_dir": float(struct_cfg.get("lambda_dir", 1.0)),
        "struct_lambda_scale": float(struct_cfg.get("lambda_scale", 0.25)),
        "struct_lambda_rec": float(struct_cfg.get("lambda_rec", 0.5)),
        "struct_lambda_rel": float(struct_cfg.get("lambda_rel", 0.1)),
        "struct_gamma": float(struct_cfg.get("gamma", 0.5)),
        "struct_huber_delta": float(struct_cfg.get("huber_delta", 0.1)),
    }


@torch.no_grad()
def _compute_loss_metrics(
    *,
    model: torch.nn.Module,
    cfg: DictConfig,
    W_s: torch.Tensor,
    x_s: torch.Tensor,
    x_mask_s: torch.Tensor,
    d_in_mask_s: torch.Tensor,
    d_out_mask_s: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    coeffs = _loss_coefficients(cfg)
    patch_size = int(cfg.model.get("patch_size", 16))
    use_latent_sampling = bool(getattr(model.cfg.big_vae, "use_latent_sampling", False))

    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
        W_hat, mu, logvar, pred_dirs = model(
            W_s,
            x_s,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        behavioral_operator = WeightQuantileVAE.operator_recon_loss(
            x_s,
            W_s,
            W_hat,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        behavioral_dir, behavioral_scale = WeightQuantileVAE.operator_direction_scale_loss(
            x_s,
            W_s,
            W_hat,
            x_mask=x_mask_s,
            d_out_mask=d_out_mask_s,
            gamma=coeffs["behavioral_gamma"],
            huber_delta=coeffs["behavioral_huber_delta"],
        )
        _struct_all, struct_details = WeightQuantileVAE.patch_structure_loss(
            W_s,
            W_hat,
            patch_size=patch_size,
            gamma=coeffs["struct_gamma"],
            lambda_dir=1.0,
            lambda_scale=1.0,
            lambda_rec=1.0,
            lambda_rel=1.0,
            huber_delta=coeffs["struct_huber_delta"],
            pred_dirs=pred_dirs,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        struct_dir = struct_details["L_dir"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        struct_scale = struct_details["L_scale"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        struct_rec = struct_details["L_rec"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        struct_rel = struct_details["L_rel"].to(device=behavioral_operator.device, dtype=behavioral_operator.dtype)
        behavioral_loss = (
            coeffs["behavioral_lambda_operator"] * behavioral_operator
            + coeffs["behavioral_lambda_dir"] * behavioral_dir
            + coeffs["behavioral_lambda_scale"] * behavioral_scale
        )
        structural_loss = (
            coeffs["struct_lambda_dir"] * struct_dir
            + coeffs["struct_lambda_scale"] * struct_scale
            + coeffs["struct_lambda_rec"] * struct_rec
            + coeffs["struct_lambda_rel"] * struct_rel
        )
        kl_loss = _compute_model_latent_kl(model, mu, logvar) if use_latent_sampling else behavioral_operator.new_zeros(())
        total_loss = (
            coeffs["behavioral_coef"] * behavioral_loss
            + coeffs["structural_coef"] * structural_loss
            + coeffs["kl_beta"] * kl_loss
        )

    return {
        "total_loss": tensor_to_float(total_loss),
        "behavioral_loss": tensor_to_float(behavioral_loss),
        "behavioral_operator": tensor_to_float(behavioral_operator),
        "behavioral_dir": tensor_to_float(behavioral_dir),
        "behavioral_scale": tensor_to_float(behavioral_scale),
        "structural_loss": tensor_to_float(structural_loss),
        "struct_dir": tensor_to_float(struct_dir),
        "struct_scale": tensor_to_float(struct_scale),
        "struct_rec": tensor_to_float(struct_rec),
        "struct_rel": tensor_to_float(struct_rel),
        "kl_loss": tensor_to_float(kl_loss),
    }


def _record_metrics_writer(path: Path) -> tuple[csv.DictWriter, Any] | tuple[None, None]:
    if not env_bool("EVAL_SAVE_RECORD_METRICS", True):
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", encoding="utf-8", newline="")
    fieldnames = [
        "record_index",
        "dataset",
        "model",
        "layer",
        "weight_shape",
        "x_shape",
        "batch_slices",
        "source_slices_total",
        "source_batch_index",
        "finite",
        "total_loss",
        "behavioral_loss",
        "behavioral_operator",
        "behavioral_dir",
        "behavioral_scale",
        "structural_loss",
        "struct_dir",
        "struct_scale",
        "struct_rec",
        "struct_rel",
        "kl_loss",
    ]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return writer, fh


@torch.no_grad()
def _extract_latents_for_batch(
    *,
    model: torch.nn.Module,
    W_s: torch.Tensor,
    x_s: torch.Tensor,
    x_mask_s: torch.Tensor,
    d_in_mask_s: torch.Tensor,
    d_out_mask_s: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
        _W_hat, mu, logvar, _pred_dirs = model(
            W_s,
            x_s,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
    return mu.detach().to(device="cpu", dtype=torch.float32), logvar.detach().to(device="cpu", dtype=torch.float32)


def _depth_label(depth: int | None) -> str:
    return f"depth_{int(depth):03d}" if depth is not None else "<unknown_depth>"


def _plot_safe_label(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    return text if text else "<unknown>"


def _latent_dump_metadata_row(
    *,
    entry_index: int,
    record_index: int,
    slice_index: int,
    source_batch_index: int,
    dataset_name: str,
    model_name: str,
    layer_name: str,
    source: SourceSampleRecord,
) -> dict[str, Any]:
    depth = infer_layer_depth(layer_name)
    return {
        "entry_index": int(entry_index),
        "record_index": int(record_index),
        "slice_index": int(slice_index),
        "source_batch_index": int(source_batch_index),
        "dataset": dataset_name,
        "model": model_name,
        "layer": layer_name,
        "layer_type": infer_layer_type(layer_name),
        "layer_depth": "" if depth is None else int(depth),
        "depth_label": _depth_label(depth),
        "weight_shape": "x".join(str(int(dim)) for dim in source.W.shape),
        "x_shape": "x".join(str(int(dim)) for dim in source.x.shape),
    }


def _write_latent_dump(
    *,
    output_dir: Path,
    latents: list[torch.Tensor],
    logvars: list[torch.Tensor],
    rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        return {
            "enabled": bool(config.get("enabled", False)),
            "count": 0,
            "path": "",
            "metadata_csv": "",
        }

    dump_path = env_path("EVAL_LATENT_DUMP_PATH", output_dir / "latent_dump.pt")
    metadata_csv_path = dump_path.with_suffix(".metadata.csv")
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    latent_tensor = torch.cat(latents, dim=0).contiguous()
    logvar_tensor = torch.cat(logvars, dim=0).contiguous()
    payload = {
        "latents": latent_tensor,
        "logvars": logvar_tensor,
        "rows": rows,
        "config": dict(config),
    }
    torch.save(payload, dump_path)
    write_csv(metadata_csv_path, rows)
    LOGGER.info("Latent dump written: path=%s entries=%s dim=%s", dump_path, len(rows), int(latent_tensor.shape[1]))
    return {
        "enabled": True,
        "count": int(len(rows)),
        "latent_dim": int(latent_tensor.shape[1]),
        "path": str(dump_path),
        "metadata_csv": str(metadata_csv_path),
    }


def _parse_latent_dump_balance_keys() -> tuple[str, ...]:
    raw = str(os.environ.get("EVAL_LATENT_DUMP_BALANCE_KEYS", _LATENT_DUMP_BALANCE_KEY_DEFAULT)).strip()
    keys = tuple(key.strip() for key in raw.split(",") if key.strip())
    allowed = {
        "dataset",
        "model",
        "layer",
        "layer_type",
        "layer_depth",
        "depth_label",
        "weight_shape",
        "x_shape",
    }
    unknown = [key for key in keys if key not in allowed]
    if unknown:
        raise ValueError(
            "EVAL_LATENT_DUMP_BALANCE_KEYS contains unsupported keys: "
            f"{unknown}. Supported keys: {sorted(allowed)}"
        )
    return keys


def _latent_dump_group_key(row: Mapping[str, Any], balance_keys: tuple[str, ...]) -> tuple[str, ...]:
    if not balance_keys:
        return ("__all__",)
    return tuple(_plot_safe_label(row.get(key)) for key in balance_keys)


def _latent_dump_replacement_index(
    *,
    buckets: dict[tuple[str, ...], list[dict[str, Any]]],
    seen_by_group: dict[tuple[str, ...], int],
    group_key: tuple[str, ...],
    max_per_group: int,
    rng: random.Random,
) -> int | None:
    seen = int(seen_by_group.get(group_key, 0)) + 1
    seen_by_group[group_key] = seen

    bucket = buckets.setdefault(group_key, [])
    if len(bucket) < max_per_group:
        return len(bucket)

    replacement_idx = rng.randrange(seen)
    if replacement_idx < max_per_group:
        return replacement_idx
    return None


def _store_latent_dump_entry(
    *,
    buckets: dict[tuple[str, ...], list[dict[str, Any]]],
    group_key: tuple[str, ...],
    replacement_idx: int,
    row: dict[str, Any],
    latent: torch.Tensor,
    logvar: torch.Tensor,
) -> None:
    entry = {"row": dict(row), "latent": latent.contiguous(), "logvar": logvar.contiguous()}
    bucket = buckets.setdefault(group_key, [])
    if replacement_idx == len(bucket):
        bucket.append(entry)
    else:
        bucket[replacement_idx] = entry


def _select_balanced_latent_dump_entries(
    *,
    buckets: Mapping[tuple[str, ...], list[dict[str, Any]]],
    max_entries: int,
    balance_keys: tuple[str, ...],
    seed: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group_key in sorted(buckets.keys()):
        candidates.extend(buckets[group_key])
    if max_entries <= 0 or not candidates:
        return []
    if len(candidates) <= max_entries:
        return candidates

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    counts_by_key: dict[str, dict[str, int]] = {key: {} for key in balance_keys}

    for _ in range(min(max_entries, len(candidates))):
        best_idx = -1
        best_score: tuple[int, int] | None = None
        for idx, entry in enumerate(candidates):
            if idx in selected_indices:
                continue
            row = entry["row"]
            marginal_score = sum(
                counts_by_key[key].get(_plot_safe_label(row.get(key)), 0)
                for key in balance_keys
            )
            score = (int(marginal_score), idx)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        entry = candidates[best_idx]
        selected.append(entry)
        selected_indices.add(best_idx)
        for key in balance_keys:
            value = _plot_safe_label(entry["row"].get(key))
            counts_by_key[key][value] = counts_by_key[key].get(value, 0) + 1

    return selected


def _load_latent_dump(dump_path: Path) -> tuple[Any, list[dict[str, Any]]]:
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Latent dump must be a dict, got {type(payload)!r}: {dump_path}")
    latents = payload.get("latents")
    rows = payload.get("rows")
    if not torch.is_tensor(latents) or latents.ndim != 2:
        raise ValueError(f"Latent dump is missing a 2D tensor 'latents': {dump_path}")
    if not isinstance(rows, list) or len(rows) != int(latents.shape[0]):
        raise ValueError(f"Latent dump rows must be a list with len={int(latents.shape[0])}: {dump_path}")
    return latents.to(dtype=torch.float32), [dict(row) for row in rows]


def _standardize_latents(latents: Any) -> Any:
    import numpy as np

    x = latents.detach().cpu().numpy().astype("float32", copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    x = x / np.maximum(scale, 1e-6)
    return x


def _pca_embedding(latents: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    x = _standardize_latents(latents)
    n = int(x.shape[0])
    if n < 2:
        coords = np.zeros((n, 2), dtype="float32")
        return coords, {"explained_variance_ratio": [0.0, 0.0]}

    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    if int(coords.shape[1]) < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - int(coords.shape[1]))), mode="constant")
    denom = float(np.square(s).sum())
    explained = (np.square(s[:2]) / denom).tolist() if denom > 0.0 else [0.0, 0.0]
    while len(explained) < 2:
        explained.append(0.0)
    return coords.astype("float32", copy=False), {"explained_variance_ratio": explained[:2]}


def _tsne_embedding(latents: Any, *, seed: int, perplexity: int) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from sklearn.manifold import TSNE

    x = _standardize_latents(latents)
    n = int(x.shape[0])
    if n < 3:
        coords = np.zeros((n, 2), dtype="float32")
        return coords, {"perplexity": 0, "skipped": "need at least 3 points"}

    safe_perplexity = max(1, min(int(perplexity), n - 1, max(1, (n - 1) // 3)))
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=safe_perplexity,
        random_state=int(seed),
    ).fit_transform(x)
    return coords.astype("float32", copy=False), {"perplexity": int(safe_perplexity)}


def _embedding_rows(
    *,
    rows: list[dict[str, Any]],
    coords: Any,
    x_name: str,
    y_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            x_name: float(coords[idx, 0]),
            y_name: float(coords[idx, 1]),
        }
        for idx, row in enumerate(rows)
    ]


def _sorted_group_values(rows: list[dict[str, Any]], color_by: str) -> list[str]:
    values = {_plot_safe_label(row.get(color_by)) for row in rows}
    if color_by in {"layer_depth", "depth_label"}:
        def depth_key(value: str) -> tuple[int, str]:
            if value.startswith("depth_"):
                try:
                    return (0, f"{int(value.split('_', 1)[1]):09d}")
                except Exception:
                    pass
            return (1, value)

        return sorted(values, key=depth_key)
    return sorted(values)


def _plot_embedding_by_metadata(
    *,
    coords: Any,
    rows: list[dict[str, Any]],
    method: str,
    color_by: str,
    output_path: Path,
    title: str,
    max_legend_labels: int,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    groups = _sorted_group_values(rows, color_by)
    cmap = plt.get_cmap("tab20", max(1, min(len(groups), 20)))
    for group_idx, group_value in enumerate(groups):
        indices = [idx for idx, row in enumerate(rows) if _plot_safe_label(row.get(color_by)) == group_value]
        if not indices:
            continue
        label = group_value if group_idx < max_legend_labels else "_nolegend_"
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            s=18,
            alpha=0.78,
            linewidths=0.0,
            color=cmap(group_idx % max(1, min(len(groups), 20))),
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.grid(True, alpha=0.2)
    if len(groups) <= max_legend_labels:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    elif max_legend_labels > 0:
        ax.legend(
            title=f"first {max_legend_labels} of {len(groups)}",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=False,
        )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_latent_dump(
    dump_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    run_tsne: bool | None = None,
) -> dict[str, Any]:
    dump_path = Path(dump_path).expanduser()
    if not dump_path.is_absolute():
        dump_path = PROJECT_ROOT / dump_path
    if not dump_path.exists():
        raise FileNotFoundError(f"Latent dump not found: {dump_path}")

    output_path = Path(output_dir).expanduser() if output_dir is not None else dump_path.parent / "latent_plots"
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
    except Exception as exc:
        LOGGER.warning("Skipping latent plots because matplotlib is unavailable: %r", exc)
        return {"enabled": False, "reason": "matplotlib_unavailable", "output_dir": str(output_path)}

    latents, rows = _load_latent_dump(dump_path)
    if int(latents.shape[0]) <= 0:
        return {"enabled": False, "reason": "empty_dump", "output_dir": str(output_path)}

    color_specs = [
        ("dataset", "dataset"),
        ("model", "model"),
        ("layer_type", "layer type"),
        ("depth_label", "layer depth"),
    ]
    max_legend_labels = max(0, env_int("EVAL_LATENT_PLOT_MAX_LEGEND_LABELS", 40))
    seed_value = int(seed if seed is not None else env_int("EVAL_SEED", 42))
    methods: dict[str, dict[str, Any]] = {}
    files: list[str] = []
    count_files = []
    for color_by, _label in color_specs:
        counts: dict[str, int] = {}
        for row in rows:
            value = _plot_safe_label(row.get(color_by))
            counts[value] = counts.get(value, 0) + 1
        count_csv = output_path / f"latent_counts_by_{color_by}.csv"
        write_csv(
            count_csv,
            [
                {color_by: value, "count": int(count)}
                for value, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
            ],
        )
        count_files.append(str(count_csv))
        files.append(str(count_csv))

    pca_coords, pca_info = _pca_embedding(latents)
    pca_csv = output_path / "latent_embedding_pca.csv"
    write_csv(pca_csv, _embedding_rows(rows=rows, coords=pca_coords, x_name="pca_x", y_name="pca_y"))
    files.append(str(pca_csv))
    pca_files = []
    for color_by, label in color_specs:
        target = output_path / f"latent_pca_by_{color_by}.png"
        _plot_embedding_by_metadata(
            coords=pca_coords,
            rows=rows,
            method="pca",
            color_by=color_by,
            output_path=target,
            title=f"Latent PCA by {label}",
            max_legend_labels=max_legend_labels,
        )
        pca_files.append(str(target))
        files.append(str(target))
    methods["pca"] = {**pca_info, "csv": str(pca_csv), "plots": pca_files}

    should_run_tsne = env_bool("EVAL_LATENT_PLOT_TSNE", True) if run_tsne is None else bool(run_tsne)
    if should_run_tsne:
        try:
            perplexity = max(1, env_int("EVAL_LATENT_TSNE_PERPLEXITY", 30))
            tsne_coords, tsne_info = _tsne_embedding(latents, seed=seed_value, perplexity=perplexity)
            tsne_csv = output_path / "latent_embedding_tsne.csv"
            write_csv(tsne_csv, _embedding_rows(rows=rows, coords=tsne_coords, x_name="tsne_x", y_name="tsne_y"))
            files.append(str(tsne_csv))
            tsne_files = []
            for color_by, label in color_specs:
                target = output_path / f"latent_tsne_by_{color_by}.png"
                _plot_embedding_by_metadata(
                    coords=tsne_coords,
                    rows=rows,
                    method="tsne",
                    color_by=color_by,
                    output_path=target,
                    title=f"Latent t-SNE by {label}",
                    max_legend_labels=max_legend_labels,
                )
                tsne_files.append(str(target))
                files.append(str(target))
            methods["tsne"] = {**tsne_info, "csv": str(tsne_csv), "plots": tsne_files}
        except Exception as exc:
            LOGGER.warning("Skipping latent t-SNE plots: %r", exc)
            methods["tsne"] = {"skipped": repr(exc)}

    info = {
        "enabled": True,
        "dump_path": str(dump_path),
        "output_dir": str(output_path),
        "count": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "methods": methods,
        "count_files": count_files,
        "files": files,
    }
    write_json(output_path / "latent_plot_summary.json", info)
    LOGGER.info("Latent plots written: output_dir=%s files=%s", output_path, len(files))
    return info


def _evaluate_dataset(
    *,
    model: torch.nn.Module,
    ckpt_cfg: DictConfig,
    dataset: OfflineBigVAEDataset,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    eval_batch_size = max(1, env_int("EVAL_BATCH_SIZE", int(ckpt_cfg.train.get("slice_batch_size", 1))))
    max_records = max(0, env_int("EVAL_MAX_RECORDS", 0))
    max_slices_per_source = max(0, env_int("EVAL_MAX_SLICES_PER_SOURCE", 0))
    log_every_records = max(1, env_int("EVAL_LOG_EVERY_RECORDS", 100))
    seed = env_int("EVAL_SEED", int(ckpt_cfg.data.get("seed", 42)) if "data" in ckpt_cfg else 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if env_int("EVAL_STAGE", 0) > 0:
        with open_dict(ckpt_cfg):
            ckpt_cfg.train.stage = int(env_int("EVAL_STAGE", 0))

    max_T_patches, max_d_out = _compute_curriculum_slice_sizes(ckpt_cfg)
    patch_size = int(ckpt_cfg.model.get("patch_size", 16))
    max_x_rows = max(0, int(ckpt_cfg.train.get("max_x_rows", 0)))
    target_x_rows, target_d_in, target_d_out = _stable_batch_shape_targets(
        cfg=ckpt_cfg,
        patch_size=patch_size,
        max_T_patches=max_T_patches,
        max_d_out=max_d_out,
        max_x_rows=max_x_rows,
    )
    amp_enabled, amp_dtype = _resolve_amp(ckpt_cfg, device)
    amp_enabled = bool(env_bool("EVAL_AMP", amp_enabled))

    grouped = GroupedMetrics()
    skipped: dict[str, int] = {"invalid": 0, "incompatible": 0, "non_finite": 0}
    record_writer, record_fh = _record_metrics_writer(output_dir / "record_metrics.csv")
    records_seen = 0
    records_evaluated = 0
    slices_evaluated = 0
    latent_dump_enabled = env_bool("EVAL_LATENT_DUMP_ENABLED", True)
    latent_dump_max_entries = max(0, env_int("EVAL_LATENT_DUMP_MAX_ENTRIES", 256))
    latent_dump_max_slices_per_source = max(1, env_int("EVAL_LATENT_DUMP_MAX_SLICES_PER_SOURCE", 1))
    latent_dump_balance_enabled = env_bool("EVAL_LATENT_DUMP_BALANCE_ENABLED", True)
    latent_dump_balance_keys = _parse_latent_dump_balance_keys() if latent_dump_balance_enabled else ()
    default_max_per_group = 1 if latent_dump_balance_enabled else max(1, latent_dump_max_entries)
    latent_dump_max_per_group = max(1, env_int("EVAL_LATENT_DUMP_MAX_PER_GROUP", default_max_per_group))
    latent_dump_buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    latent_dump_seen_by_group: dict[tuple[str, ...], int] = {}
    latent_dump_rng = random.Random(seed)
    latent_dump_config = {
        "enabled": bool(latent_dump_enabled),
        "max_entries": int(latent_dump_max_entries),
        "max_slices_per_source": int(latent_dump_max_slices_per_source),
        "balance_enabled": bool(latent_dump_balance_enabled),
        "balance_keys": list(latent_dump_balance_keys),
        "max_per_group": int(latent_dump_max_per_group),
        "source": "posterior_mu_when_sampling_enabled_else_base_z",
        "point_unit": "slice",
    }

    try:
        for sample in dataset:
            if max_records > 0 and records_seen >= max_records:
                break
            records_seen += 1
            dataset_name = primary_dataset_name(getattr(sample, "meta", {}) or {})
            model_name = str(getattr(sample, "model_name", "") or "").strip() or "<unknown_model>"
            layer_name = str(getattr(sample, "layer_name", "") or "").strip() or "<unknown_layer>"
            try:
                source = _source_record_from_sample(sample, device)
                state = _make_source_slice_state(
                    source,
                    max_T_patches=max_T_patches,
                    max_d_out=max_d_out,
                    patch_size=patch_size,
                )
            except Exception:
                skipped["invalid"] += 1
                continue

            total_source_slices = _remaining_source_state_slices(state)
            if total_source_slices <= 0:
                skipped["incompatible"] += 1
                continue
            if max_slices_per_source > 0:
                total_to_eval = min(total_source_slices, max_slices_per_source)
            else:
                total_to_eval = total_source_slices

            source_batch_idx = 0
            evaluated_for_source = 0
            dumped_for_source = 0
            while evaluated_for_source < total_to_eval:
                remaining = _remaining_source_state_slices(state)
                if remaining <= 0:
                    break
                batch_slices = min(eval_batch_size, remaining, total_to_eval - evaluated_for_source)
                batch_payload = _build_training_batch_from_source_states(
                    [state],
                    batch_size=batch_slices,
                    target_x_rows=target_x_rows,
                    target_d_in=target_d_in,
                    target_d_out=target_d_out,
                )
                W_batch = batch_payload.W.to(device=device, non_blocking=True)
                x_batch = batch_payload.x.to(device=device, non_blocking=True)
                x_mask_batch = batch_payload.x_mask.to(device=device, non_blocking=True)
                d_in_mask_batch = batch_payload.d_in_mask.to(device=device, non_blocking=True)
                d_out_mask_batch = batch_payload.d_out_mask.to(device=device, non_blocking=True)
                metrics = _compute_loss_metrics(
                    model=model,
                    cfg=ckpt_cfg,
                    W_s=W_batch,
                    x_s=x_batch,
                    x_mask_s=x_mask_batch,
                    d_in_mask_s=d_in_mask_batch,
                    d_out_mask_s=d_out_mask_batch,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                source_dump_remaining = latent_dump_max_slices_per_source - dumped_for_source
                if latent_dump_enabled and latent_dump_max_entries > 0 and source_dump_remaining > 0:
                    candidate_take = min(int(source_dump_remaining), int(batch_slices))
                    for local_idx in range(candidate_take):
                        row = _latent_dump_metadata_row(
                            entry_index=0,
                            record_index=records_seen,
                            slice_index=evaluated_for_source + local_idx,
                            source_batch_index=source_batch_idx,
                            dataset_name=dataset_name,
                            model_name=model_name,
                            layer_name=layer_name,
                            source=source,
                        )
                        group_key = _latent_dump_group_key(row, latent_dump_balance_keys)
                        row["balance_group"] = " | ".join(group_key)
                        replacement_idx = _latent_dump_replacement_index(
                            buckets=latent_dump_buckets,
                            seen_by_group=latent_dump_seen_by_group,
                            group_key=group_key,
                            max_per_group=latent_dump_max_per_group,
                            rng=latent_dump_rng,
                        )
                        if replacement_idx is None:
                            continue
                        mu_cpu, logvar_cpu = _extract_latents_for_batch(
                            model=model,
                            W_s=W_batch[local_idx : local_idx + 1],
                            x_s=x_batch[local_idx : local_idx + 1],
                            x_mask_s=x_mask_batch[local_idx : local_idx + 1],
                            d_in_mask_s=d_in_mask_batch[local_idx : local_idx + 1],
                            d_out_mask_s=d_out_mask_batch[local_idx : local_idx + 1],
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                        )
                        _store_latent_dump_entry(
                            buckets=latent_dump_buckets,
                            group_key=group_key,
                            replacement_idx=int(replacement_idx),
                            row=row,
                            latent=mu_cpu[:1],
                            logvar=logvar_cpu[:1],
                        )
                    dumped_for_source += int(candidate_take)
                if not finite_metrics(metrics):
                    skipped["non_finite"] += 1
                else:
                    grouped.update(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        layer_name=layer_name,
                        metrics=metrics,
                        weight=batch_slices,
                    )
                    slices_evaluated += int(batch_slices)

                if record_writer is not None:
                    record_writer.writerow(
                        {
                            "record_index": int(records_seen),
                            "dataset": dataset_name,
                            "model": model_name,
                            "layer": layer_name,
                            "weight_shape": "x".join(str(int(dim)) for dim in source.W.shape),
                            "x_shape": "x".join(str(int(dim)) for dim in source.x.shape),
                            "batch_slices": int(batch_slices),
                            "source_slices_total": int(total_source_slices),
                            "source_batch_index": int(source_batch_idx),
                            "finite": bool(finite_metrics(metrics)),
                            **metrics,
                        }
                    )

                evaluated_for_source += int(batch_slices)
                source_batch_idx += 1

            records_evaluated += 1
            if records_evaluated % log_every_records == 0:
                LOGGER.info(
                    "Eval progress: records_seen=%s records_evaluated=%s slices_evaluated=%s skipped=%s",
                    records_seen,
                    records_evaluated,
                    slices_evaluated,
                    skipped,
                )
    finally:
        if record_fh is not None:
            record_fh.close()

    payload = grouped.payload()
    payload["coverage"] = coverage_report(
        observed_datasets=grouped.by_dataset.keys(),
        observed_models=grouped.by_model.keys(),
        observed_pairs=grouped.by_pair.keys(),
        skipped=skipped,
    )
    payload["run"] = {
        "records_seen": int(records_seen),
        "records_evaluated": int(records_evaluated),
        "slices_evaluated": int(slices_evaluated),
        "skipped": {key: int(value) for key, value in skipped.items()},
        "eval_batch_size": int(eval_batch_size),
        "max_records": int(max_records),
        "max_slices_per_source": int(max_slices_per_source),
        "stage": int(ckpt_cfg.train.get("stage", 1)),
        "max_T_patches": int(max_T_patches),
        "max_d_out": int(max_d_out),
        "patch_size": int(patch_size),
        "amp_enabled": bool(amp_enabled),
        "amp_dtype": str(amp_dtype),
        "device": str(device),
    }
    candidate_entries = sum(len(bucket) for bucket in latent_dump_buckets.values())
    latent_dump_config["candidate_groups"] = int(len(latent_dump_buckets))
    latent_dump_config["candidate_entries_before_global_cap"] = int(candidate_entries)
    selected_latent_entries = _select_balanced_latent_dump_entries(
        buckets=latent_dump_buckets,
        max_entries=latent_dump_max_entries,
        balance_keys=latent_dump_balance_keys,
        seed=seed,
    )
    latent_dump_rows = []
    latent_dump_latents = []
    latent_dump_logvars = []
    for entry_index, entry in enumerate(selected_latent_entries):
        row = dict(entry["row"])
        row["entry_index"] = int(entry_index)
        latent_dump_rows.append(row)
        latent_dump_latents.append(entry["latent"])
        latent_dump_logvars.append(entry["logvar"])
    payload["latent_dump"] = _write_latent_dump(
        output_dir=output_dir,
        latents=latent_dump_latents,
        logvars=latent_dump_logvars,
        rows=latent_dump_rows,
        config=latent_dump_config,
    )
    return payload


def _load_base_config() -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        return compose(config_name="config")


def main() -> None:
    cfg = _load_base_config()
    promote_run_profile_to_root(cfg)
    sanitize_programmatic_hydra_logging(cfg, role="evaluate_big_vae_heldout")
    raw_checkpoint = str(os.environ.get("BIG_VAE_CHECKPOINT", "")).strip()
    if not raw_checkpoint:
        raise ValueError("Set BIG_VAE_CHECKPOINT=/path/to/big_vae_checkpoint.pt")
    checkpoint_path = env_path("BIG_VAE_CHECKPOINT", raw_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BIG_VAE_CHECKPOINT does not exist: {checkpoint_path}")

    offline_root = env_path(
        "HELDOUT_ROOT",
        "post_train_research/big_vae_heldout_eval/artifacts/offline_dataset",
    )
    log_dir = apply_heldout_log_dir(cfg, root_dir=offline_root)
    if not (offline_root / "manifest.json").exists():
        raise FileNotFoundError(f"Held-out offline dataset manifest not found: {offline_root / 'manifest.json'}")

    default_output = offline_root / "eval" / f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
    output_dir = env_path("EVAL_OUTPUT_DIR", default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = configure_process_logging(cfg=cfg, role="evaluate_big_vae_heldout", rank=0, force=True)
    LOGGER.info("Run log file: %s", log_path)
    LOGGER.info("Held-out log dir: %s", log_dir)
    LOGGER.info("Loading BigVAE checkpoint: %s", checkpoint_path)
    device = _resolve_eval_device(cfg)
    model, ckpt_cfg, ckpt_payload = _load_checkpoint_model(checkpoint_path, device)

    dataset = OfflineBigVAEDataset(
        root_dir=offline_root,
        shuffle_chunks=env_bool("EVAL_SHUFFLE_CHUNKS", False),
        shuffle_records_within_chunk=env_bool("EVAL_SHUFFLE_RECORDS_WITHIN_CHUNK", False),
        repeat=False,
        seed=env_int("EVAL_SEED", int(cfg.data.get("seed", 42))),
        weight_cache_size=env_int("EVAL_WEIGHT_CACHE_SIZE", 64),
        sampling_mode="balanced",
        sampling_group_keys=("dataset", "model"),
        sampling_window_size=env_int("EVAL_SAMPLING_WINDOW_SIZE_RECORDS", 2048),
        sampling_max_records_per_chunk_round=env_int("EVAL_MAX_RECORDS_PER_CHUNK_ROUND", 8),
        x_chunk_cache_size=env_int("EVAL_X_CHUNK_CACHE_SIZE", 4),
    )
    LOGGER.info("Held-out dataset summary: %s", dataset.summary())

    payload = _evaluate_dataset(
        model=model,
        ckpt_cfg=ckpt_cfg,
        dataset=dataset,
        device=device,
        output_dir=output_dir,
    )
    payload["checkpoint"] = {
        "path": str(checkpoint_path),
        "step": int(ckpt_payload.get("step", 0) or 0),
        "stage": int(ckpt_payload.get("stage", 0) or 0),
    }
    payload["offline_dataset"] = dataset.summary()
    latent_dump_path = str(payload.get("latent_dump", {}).get("path", "")).strip()
    if latent_dump_path and env_bool("EVAL_LATENT_PLOT_ENABLED", True):
        payload["latent_plots"] = plot_latent_dump(
            latent_dump_path,
            output_dir=env_path("EVAL_LATENT_PLOT_DIR", output_dir / "latent_plots"),
            seed=env_int("EVAL_SEED", int(cfg.data.get("seed", 42))),
        )
    else:
        payload["latent_plots"] = {"enabled": False}

    write_json(output_dir / "metrics_summary.json", payload)
    write_json(output_dir / "coverage.json", payload["coverage"])
    write_json(output_dir / "metrics_global.json", payload["global"])
    write_json(output_dir / "metrics_macro.json", payload["macro"])
    write_csv(output_dir / "metrics_by_model.csv", payload["by_model"])
    write_csv(output_dir / "metrics_by_dataset.csv", payload["by_dataset"])
    write_csv(output_dir / "metrics_by_dataset_model_pair.csv", payload["by_dataset_model_pair"])
    write_csv(output_dir / "metrics_by_layer.csv", payload["by_layer"])
    LOGGER.info(
        "Held-out eval complete: output_dir=%s coverage_ok=%s global=%s macro=%s",
        output_dir,
        bool(payload["coverage"].get("ok", False)),
        payload["global"],
        payload["macro"],
    )
    if not bool(payload["coverage"].get("ok", False)):
        LOGGER.error("Held-out coverage check failed: %s", payload["coverage"])
        if env_bool("EVAL_REQUIRE_FULL_COVERAGE", env_int("EVAL_MAX_RECORDS", 0) <= 0):
            raise RuntimeError(f"Held-out coverage check failed; see {output_dir / 'coverage.json'}")


if __name__ == "__main__":
    main()
