from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from dataset.shared.collector_service import CollectorService
from dataset.shared.compatibility_index import CompatibilityIndex
from dataset.shared.shared_dataset import SharedModelDataset
from dataset.shared.types import SharedSample


LOGGER = logging.getLogger("patch_pca_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect random weight patches from selected models and compute PCA/t-SNE grouped by "
            "architecture and layer type."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names from conf/data/models/*.yaml",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names. If omitted, uses config default enabled_datasets.",
    )
    parser.add_argument("--config-name", default="mini_vae_train", help="Hydra config name from conf/")
    parser.add_argument("--patch-size", type=int, default=64, help="Input-dimension patch size")
    parser.add_argument("--patches-per-sample", type=int, default=16, help="How many random patches per SharedSample")
    parser.add_argument("--components", type=int, default=8, help="Number of principal components to keep")
    parser.add_argument(
        "--tsne-max-points",
        type=int,
        default=5000,
        help="Maximum number of points for t-SNE (uniformly sampled from collected patches)",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0, help="t-SNE learning rate")
    parser.add_argument("--tsne-n-iter", type=int, default=1000, help="t-SNE optimization iterations")
    parser.add_argument("--max-samples-per-model", type=int, default=64, help="Max SharedSample items per model")
    parser.add_argument("--max-total-samples", type=int, default=1024, help="Global SharedSample cap")
    parser.add_argument("--max-patches-per-group", type=int, default=8192, help="Max patch vectors retained per group")
    parser.add_argument("--time-limit-seconds", type=float, default=1800.0, help="Hard timeout for collection loop")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--collector-mode", default=None, help="Optional override for collector.mode")
    parser.add_argument("--collector-device", default=None, help="Optional override for collector.device")
    parser.add_argument("--streaming-mode", default="none", help="Override for streaming.mode (recommended: none)")
    parser.add_argument("--predownload-models", action="store_true", help="Predownload model artifacts before start")
    parser.add_argument(
        "--output-dir",
        default="data/reports/pca_patches",
        help="Output directory for JSON/PT reports",
    )
    return parser.parse_args()


def compose_cfg(config_name: str) -> DictConfig:
    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name=config_name)
    # Standalone scripts don't populate HydraConfig; avoid `${hydra:...}` in logging config.
    with open_dict(cfg):
        if "logging" in cfg:
            cfg.logging.file_name = "patch_pca_analysis.log"
            cfg.logging.file_path = f"{cfg.logging.dir}/{cfg.logging.file_name}"
    return cfg


def infer_layer_type(layer_name: str) -> str:
    name = str(layer_name).lower()

    if "attention" in name:
        if any(token in name for token in ("query", "q_proj", ".q.", "self.q", ".qkv")):
            return "attn_query"
        if any(token in name for token in ("key", "k_proj", ".k.", "self.k")):
            return "attn_key"
        if any(token in name for token in ("value", "v_proj", ".v.", "self.v")):
            return "attn_value"
        if any(token in name for token in ("out_proj", "output.dense", "attention.output", "proj")):
            return "attn_output"
        return "attn_other"

    if any(token in name for token in ("intermediate", "fc1", "mlp.fc1", "gate_proj", "up_proj")):
        return "ffn_up"
    if any(token in name for token in ("output.dense", "fc2", "mlp.fc2", "down_proj")):
        return "ffn_down"

    if "pooler" in name:
        return "pooler"
    if any(token in name for token in ("embed", "embedding")):
        return "embedding"
    if "conv" in name:
        return "conv"
    if any(token in name for token in ("lm_head", "classifier", "score", "head")):
        return "head"
    return "other_linear"


def model_architecture_label(model_cfg: dict[str, Any]) -> str:
    for key in ("architecture", "model_family", "runner", "provider"):
        value = model_cfg.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def sample_weight_patches(
    weight: torch.Tensor,
    *,
    patch_size: int,
    patches_per_sample: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2 [d_in,d_out], got {tuple(weight.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    if patches_per_sample <= 0:
        raise ValueError(f"patches_per_sample must be > 0, got {patches_per_sample}")

    d_in, d_out = weight.shape
    if d_in <= 0 or d_out <= 0:
        raise ValueError(f"weight shape must be positive, got {tuple(weight.shape)}")

    num_patch_slots = max(1, (int(d_in) + patch_size - 1) // patch_size)
    out_idx = torch.randint(0, int(d_out), (patches_per_sample,), generator=generator)
    patch_t = torch.randint(0, num_patch_slots, (patches_per_sample,), generator=generator)

    offsets = torch.arange(patch_size).unsqueeze(0)  # [1, p]
    patch_idx = patch_t.unsqueeze(1) * patch_size + offsets  # [B, p]
    patch_idx = patch_idx.clamp(max=max(0, int(d_in) - 1))

    # [B, d_in]
    w_col = weight.transpose(0, 1).index_select(0, out_idx).contiguous()
    # [B, p]
    return w_col.gather(dim=1, index=patch_idx)


def _stack_or_empty(chunks: list[torch.Tensor], patch_size: int) -> torch.Tensor:
    if not chunks:
        return torch.empty((0, patch_size), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def compute_group_pca(
    patches: torch.Tensor,
    *,
    n_components: int,
) -> dict[str, Any]:
    if patches.ndim != 2:
        raise ValueError(f"patches must be rank-2, got {tuple(patches.shape)}")
    n, dim = patches.shape
    if n < 2:
        return {
            "num_patches": int(n),
            "dim": int(dim),
            "num_components": 0,
            "explained_variance_ratio": [],
            "explained_variance_cumulative": [],
            "total_variance": 0.0,
            "mean_l2_norm": 0.0,
        }

    x = patches.to(dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean

    # Full SVD is stable enough here because patch_size is typically small (e.g. 64).
    _, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
    r = int(min(max(1, n_components), s.numel(), dim))
    s = s[:r]
    components = vh[:r, :]

    denom = max(1.0, float(n - 1))
    explained_variance = (s * s) / denom
    total_variance = float((x_centered * x_centered).sum().item() / denom)
    total_variance = max(total_variance, 1e-12)
    ratio = explained_variance / total_variance
    ratio_cum = torch.cumsum(ratio, dim=0)

    return {
        "num_patches": int(n),
        "dim": int(dim),
        "num_components": int(r),
        "explained_variance_ratio": [float(v) for v in ratio.tolist()],
        "explained_variance_cumulative": [float(v) for v in ratio_cum.tolist()],
        "total_variance": float(total_variance),
        "mean_l2_norm": float(x.norm(dim=1).mean().item()),
        "mean": [float(v) for v in mean.squeeze(0).tolist()],
        "components": [[float(v) for v in row] for row in components.tolist()],
    }


def compute_projection_2d(patches: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """
    Project patch matrix [N, D] to PCA-2D.

    Returns:
    - coords: [N, 2]
    - explained_ratio: [pc1_ratio, pc2_ratio] (missing components are padded with 0.0)
    """
    if patches.ndim != 2:
        raise ValueError(f"patches must be rank-2, got {tuple(patches.shape)}")
    n, d = patches.shape
    if n == 0:
        return torch.empty((0, 2), dtype=torch.float32), [0.0, 0.0]

    x = patches.to(dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean

    if n < 2 or d < 1:
        return torch.zeros((n, 2), dtype=torch.float32), [0.0, 0.0]

    _, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
    rank = int(min(2, vh.shape[0], vh.shape[1]))
    components = vh[:rank, :]  # [rank, D]
    coords = x_centered @ components.transpose(0, 1)  # [N, rank]

    if rank == 1:
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)

    denom = max(1.0, float(n - 1))
    total_variance = float((x_centered * x_centered).sum().item() / denom)
    total_variance = max(total_variance, 1e-12)
    explained_variance = (s[:rank] * s[:rank]) / denom
    ratios = [float(v) for v in (explained_variance / total_variance).tolist()]
    while len(ratios) < 2:
        ratios.append(0.0)
    return coords, ratios[:2]


def compute_tsne_projection_2d(
    patches: torch.Tensor,
    *,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    seed: int,
) -> torch.Tensor:
    if patches.ndim != 2:
        raise ValueError(f"patches must be rank-2, got {tuple(patches.shape)}")
    n, _ = patches.shape
    if n < 3:
        return torch.zeros((n, 2), dtype=torch.float32)

    # Local import: keep script usable even without sklearn when t-SNE is not needed.
    from sklearn.manifold import TSNE

    x_np = patches.to(dtype=torch.float32, device="cpu").numpy()
    safe_perplexity = max(1.0, min(float(perplexity), float(n - 1)))
    common_kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": safe_perplexity,
        "learning_rate": float(learning_rate),
        "init": "pca",
        "random_state": int(seed),
    }

    # sklearn API changed from n_iter -> max_iter in newer releases.
    try:
        tsne = TSNE(n_iter=int(n_iter), **common_kwargs)
    except TypeError:
        tsne = TSNE(max_iter=int(n_iter), **common_kwargs)

    coords = tsne.fit_transform(x_np)
    return torch.from_numpy(coords).to(dtype=torch.float32)


def _normalize_model_list(models: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in models:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        output.append(name)
    return output


def _prepare_runtime_cfg(cfg: DictConfig, args: argparse.Namespace, models: list[str]) -> DictConfig:
    with open_dict(cfg):
        if args.datasets:
            cfg.data.enabled_datasets = [str(name) for name in args.datasets]
        if args.collector_mode is not None:
            cfg.collector.mode = str(args.collector_mode)
        if args.collector_device is not None:
            cfg.collector.device = str(args.collector_device)
        if args.streaming_mode is not None:
            cfg.streaming.mode = str(args.streaming_mode)

        # Keep this script lean: avoid extra side effects from training-oriented config.
        if "mini_train" in cfg:
            cfg.mini_train.predownload_models = bool(args.predownload_models)

    index = CompatibilityIndex(cfg)
    enabled_datasets = [str(name) for name in cfg.data.enabled_datasets]
    model_set = set(models)

    dataset_overrides_raw = cfg.data.get("dataset_overrides")
    if dataset_overrides_raw is None:
        dataset_overrides: dict[str, Any] = {}
    else:
        plain = OmegaConf.to_container(dataset_overrides_raw, resolve=False)
        dataset_overrides = dict(plain) if isinstance(plain, dict) else {}

    filtered_datasets: list[str] = []
    for dataset_name in enabled_datasets:
        try:
            ds_cfg = index.get_dataset_cfg(dataset_name)
        except KeyError:
            continue
        ds_models = [str(name) for name in ds_cfg.get("models", [])]
        kept = [name for name in ds_models if name in model_set]
        if not kept:
            continue
        filtered_datasets.append(dataset_name)
        current_override = dataset_overrides.get(dataset_name, {})
        if not isinstance(current_override, dict):
            current_override = {}
        dataset_overrides[dataset_name] = {
            **current_override,
            "models": kept,
        }

    if not filtered_datasets:
        raise RuntimeError(
            "No enabled datasets remain after restricting to selected models. "
            "Check --models/--datasets compatibility."
        )

    with open_dict(cfg):
        cfg.data.enabled_datasets = filtered_datasets
        cfg.data.dataset_overrides = dataset_overrides

    return cfg


def _extract_valid_weight(sample: SharedSample) -> torch.Tensor | None:
    w = sample.weight
    if not torch.is_tensor(w):
        return None
    if w.ndim != 2:
        return None
    if w.shape[0] <= 0 or w.shape[1] <= 0:
        return None
    return w.detach().to(dtype=torch.float32, device="cpu", copy=True).contiguous()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    models = _normalize_model_list(args.models)
    if not models:
        raise SystemExit("No valid --models provided")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    cfg = compose_cfg(args.config_name)
    cfg = _prepare_runtime_cfg(cfg, args, models)
    index = CompatibilityIndex(cfg)

    unknown = [name for name in models if name not in index.get_model_cfgs()]
    if unknown:
        raise RuntimeError(f"Unknown model configs: {unknown}")
    incompatible = [name for name in models if not index.get_datasets_for_model(name)]
    if incompatible:
        raise RuntimeError(
            "Selected model(s) are not compatible with currently enabled datasets after filtering: "
            f"{incompatible}"
        )

    model_to_arch: dict[str, str] = {}
    for model_name in models:
        model_cfg = index.get_model_cfg(model_name)
        model_to_arch[model_name] = model_architecture_label(model_cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_counts: Counter[str] = Counter()
    layer_type_counts: Counter[str] = Counter()
    arch_counts: Counter[str] = Counter()
    arch_layer_counts: Counter[str] = Counter()

    patches_by_arch: dict[str, list[torch.Tensor]] = defaultdict(list)
    patches_by_layer: dict[str, list[torch.Tensor]] = defaultdict(list)
    patches_by_arch_layer: dict[str, list[torch.Tensor]] = defaultdict(list)
    patches_by_model: dict[str, list[torch.Tensor]] = defaultdict(list)
    all_patch_chunks: list[torch.Tensor] = []
    all_patch_model_labels: list[str] = []
    all_patch_arch_labels: list[str] = []
    all_patch_layer_labels: list[str] = []

    patch_count_by_group: Counter[str] = Counter()
    total_seen = 0
    total_kept_samples = 0
    total_kept_patches = 0

    start_ts = time.time()
    deadline = start_ts + max(1.0, float(args.time_limit_seconds))

    collector = CollectorService(cfg)
    dataset = SharedModelDataset(collector)
    try:
        if args.predownload_models:
            LOGGER.info("Predownloading model artifacts")
            collector.predownload_models()
        collector.start()
        data_iter = iter(dataset)

        LOGGER.info(
            "Collecting patches: models=%s datasets=%s patch_size=%s patches_per_sample=%s max_samples_per_model=%s",
            models,
            list(cfg.data.enabled_datasets),
            args.patch_size,
            args.patches_per_sample,
            args.max_samples_per_model,
        )

        while True:
            if time.time() > deadline:
                LOGGER.warning("Stopping by time limit (%.1fs)", args.time_limit_seconds)
                break
            if total_kept_samples >= int(args.max_total_samples):
                break
            if all(model_counts[m] >= int(args.max_samples_per_model) for m in models):
                break

            sample = next(data_iter)
            total_seen += 1

            model_name = str(sample.model_name)
            if model_name not in model_to_arch:
                continue
            if model_counts[model_name] >= int(args.max_samples_per_model):
                continue

            w = _extract_valid_weight(sample)
            if w is None:
                continue

            try:
                patches = sample_weight_patches(
                    w,
                    patch_size=int(args.patch_size),
                    patches_per_sample=int(args.patches_per_sample),
                    generator=generator,
                )
            except Exception:
                continue

            arch = model_to_arch[model_name]
            layer_type = infer_layer_type(sample.layer_name)
            arch_layer = f"{arch}::{layer_type}"
            num_patches = int(patches.shape[0])

            all_patch_chunks.append(patches)
            all_patch_model_labels.extend([model_name] * num_patches)
            all_patch_arch_labels.extend([arch] * num_patches)
            all_patch_layer_labels.extend([layer_type] * num_patches)

            # Keep bounded number of vectors per group.
            def _append_if_room(group_key: str, bucket: list[torch.Tensor]) -> None:
                if patch_count_by_group[group_key] >= int(args.max_patches_per_group):
                    return
                room = int(args.max_patches_per_group) - int(patch_count_by_group[group_key])
                take = min(room, int(patches.shape[0]))
                if take <= 0:
                    return
                chunk = patches[:take].contiguous()
                bucket.append(chunk)
                patch_count_by_group[group_key] += take

            _append_if_room(f"arch::{arch}", patches_by_arch[arch])
            _append_if_room(f"layer::{layer_type}", patches_by_layer[layer_type])
            _append_if_room(f"arch_layer::{arch_layer}", patches_by_arch_layer[arch_layer])
            _append_if_room(f"model::{model_name}", patches_by_model[model_name])

            model_counts[model_name] += 1
            layer_type_counts[layer_type] += 1
            arch_counts[arch] += 1
            arch_layer_counts[arch_layer] += 1

            total_kept_samples += 1
            total_kept_patches += num_patches

            if total_kept_samples % 25 == 0:
                LOGGER.info(
                    "progress samples=%s seen=%s patches=%s models=%s",
                    total_kept_samples,
                    total_seen,
                    total_kept_patches,
                    {name: int(model_counts[name]) for name in models},
                )
    finally:
        dataset.close()
        collector.shutdown()

    LOGGER.info(
        "Collection done: kept_samples=%s seen=%s kept_patches=%s",
        total_kept_samples,
        total_seen,
        total_kept_patches,
    )

    def _compute_group_report(
        grouped: dict[str, list[torch.Tensor]],
        *,
        grouping: str,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        tensors_report: dict[str, dict[str, Any]] = {}
        summary_report: dict[str, Any] = {
            "grouping": grouping,
            "groups": {},
        }
        for key in sorted(grouped.keys()):
            stacked = _stack_or_empty(grouped[key], int(args.patch_size))
            result = compute_group_pca(stacked, n_components=int(args.components))
            summary_report["groups"][key] = {
                k: v
                for k, v in result.items()
                if k not in {"components", "mean"}
            }
            tensors_report[key] = {
                "mean": result.get("mean", []),
                "components": result.get("components", []),
                "explained_variance_ratio": result.get("explained_variance_ratio", []),
            }
        return summary_report, tensors_report

    report_arch, tensors_arch = _compute_group_report(patches_by_arch, grouping="architecture")
    report_layer, tensors_layer = _compute_group_report(patches_by_layer, grouping="layer_type")
    report_arch_layer, tensors_arch_layer = _compute_group_report(
        patches_by_arch_layer, grouping="architecture_layer_type"
    )
    report_model, tensors_model = _compute_group_report(patches_by_model, grouping="model")

    meta = {
        "timestamp": time.time(),
        "duration_s": time.time() - start_ts,
        "args": vars(args),
        "models": models,
        "datasets": [str(name) for name in cfg.data.enabled_datasets],
        "model_architecture": model_to_arch,
        "collector_mode": str(cfg.collector.mode),
        "collector_device": str(cfg.collector.get("device", "auto")),
        "streaming_mode": str(cfg.streaming.get("mode", "none")),
        "seen_samples": int(total_seen),
        "kept_samples": int(total_kept_samples),
        "kept_patches": int(total_kept_patches),
        "counts": {
            "by_model": {k: int(v) for k, v in model_counts.items()},
            "by_architecture": {k: int(v) for k, v in arch_counts.items()},
            "by_layer_type": {k: int(v) for k, v in layer_type_counts.items()},
            "by_architecture_layer_type": {k: int(v) for k, v in arch_layer_counts.items()},
        },
    }

    summary = {
        "meta": meta,
        "pca_by_architecture": report_arch,
        "pca_by_layer_type": report_layer,
        "pca_by_architecture_layer_type": report_arch_layer,
        "pca_by_model": report_model,
    }

    summary_path = output_dir / "summary.json"

    tensors_payload = {
        "meta": meta,
        "architecture": tensors_arch,
        "layer_type": tensors_layer,
        "architecture_layer_type": tensors_arch_layer,
        "model": tensors_model,
    }
    tensors_path = output_dir / "pca_tensors.pt"
    torch.save(tensors_payload, tensors_path)

    rows: list[dict[str, Any]] = []
    for grouping_key, grouping_payload in (
        ("architecture", report_arch),
        ("layer_type", report_layer),
        ("architecture_layer_type", report_arch_layer),
        ("model", report_model),
    ):
        groups = grouping_payload.get("groups", {})
        if not isinstance(groups, dict):
            continue
        for group_name, info in groups.items():
            if not isinstance(info, dict):
                continue
            ratios = info.get("explained_variance_ratio", [])
            cumulative = info.get("explained_variance_cumulative", [])
            for idx, ratio in enumerate(ratios):
                rows.append(
                    {
                        "grouping": grouping_key,
                        "group": group_name,
                        "num_patches": info.get("num_patches", 0),
                        "component_idx": idx + 1,
                        "explained_variance_ratio": ratio,
                        "explained_variance_cumulative": cumulative[idx] if idx < len(cumulative) else None,
                    }
                )

    csv_path = output_dir / "explained_variance.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "grouping",
                "group",
                "num_patches",
                "component_idx",
                "explained_variance_ratio",
                "explained_variance_cumulative",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    all_points = _stack_or_empty(all_patch_chunks, int(args.patch_size))

    # Optional plotting backend used by both PCA and t-SNE scatter plots.
    plt = None
    try:
        import matplotlib.pyplot as plt  # type: ignore[assignment]
    except Exception as exc:
        LOGGER.warning("Could not import matplotlib; scatter plots will be skipped: %s", exc)

    def _scatter_by_labels(
        coords: torch.Tensor,
        labels: list[str],
        *,
        title: str,
        out_path: Path,
        x_label: str,
        y_label: str,
    ) -> bool:
        if plt is None or int(coords.shape[0]) == 0:
            return False
        unique_labels = sorted(set(labels))
        if not unique_labels:
            return False

        coords_cpu = coords.cpu()
        fig, ax = plt.subplots(figsize=(12, 9))
        cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
        for color_idx, label in enumerate(unique_labels):
            indices = [i for i, value in enumerate(labels) if value == label]
            if not indices:
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            points = coords_cpu.index_select(0, idx_tensor)
            ax.scatter(
                points[:, 0].numpy(),
                points[:, 1].numpy(),
                s=8,
                alpha=0.35,
                color=cmap(color_idx),
                label=label,
                linewidths=0.0,
            )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
        if len(unique_labels) <= 20:
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return True

    # ------------------------------- PCA -------------------------------
    pca_arch_plot_path = output_dir / "all_points_pca_by_architecture.png"
    pca_layer_plot_path = output_dir / "all_points_pca_by_layer_type.png"
    pca_alias_plot_path = output_dir / "all_points_pca.png"
    pca_points_csv_path = output_dir / "all_points_pca2d.csv"
    pca_coords = torch.empty((0, 2), dtype=torch.float32)
    pca_explained = [0.0, 0.0]
    global_plot_meta: dict[str, Any] = {
        "num_points": int(all_points.shape[0]),
        "plot_path": str(pca_alias_plot_path),  # backward-compatible alias to architecture plot
        "architecture_plot_path": str(pca_arch_plot_path),
        "layer_type_plot_path": str(pca_layer_plot_path),
        "points_csv_path": str(pca_points_csv_path),
        "pc1_explained_variance_ratio": 0.0,
        "pc2_explained_variance_ratio": 0.0,
        "architecture_plot_saved": False,
        "layer_type_plot_saved": False,
    }
    if int(all_points.shape[0]) > 0:
        pca_coords, pca_explained = compute_projection_2d(all_points)
        global_plot_meta["pc1_explained_variance_ratio"] = float(pca_explained[0])
        global_plot_meta["pc2_explained_variance_ratio"] = float(pca_explained[1])

    with pca_points_csv_path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["point_idx", "pc1", "pc2", "model", "architecture", "layer_type"],
        )
        writer.writeheader()
        for idx in range(int(pca_coords.shape[0])):
            writer.writerow(
                {
                    "point_idx": idx,
                    "pc1": float(pca_coords[idx, 0].item()),
                    "pc2": float(pca_coords[idx, 1].item()),
                    "model": all_patch_model_labels[idx],
                    "architecture": all_patch_arch_labels[idx],
                    "layer_type": all_patch_layer_labels[idx],
                }
            )

    pca_arch_saved = _scatter_by_labels(
        pca_coords,
        all_patch_arch_labels,
        title="PCA of Weight Patches (Colored by Architecture)",
        out_path=pca_arch_plot_path,
        x_label=f"PC1 ({pca_explained[0] * 100.0:.2f}% variance)",
        y_label=f"PC2 ({pca_explained[1] * 100.0:.2f}% variance)",
    )
    pca_layer_saved = _scatter_by_labels(
        pca_coords,
        all_patch_layer_labels,
        title="PCA of Weight Patches (Colored by Layer Type)",
        out_path=pca_layer_plot_path,
        x_label=f"PC1 ({pca_explained[0] * 100.0:.2f}% variance)",
        y_label=f"PC2 ({pca_explained[1] * 100.0:.2f}% variance)",
    )
    global_plot_meta["architecture_plot_saved"] = pca_arch_saved
    global_plot_meta["layer_type_plot_saved"] = pca_layer_saved
    if pca_arch_saved:
        try:
            pca_alias_plot_path.write_bytes(pca_arch_plot_path.read_bytes())
        except Exception:
            pass

    # ------------------------------- t-SNE -------------------------------
    tsne_arch_plot_path = output_dir / "all_points_tsne_by_architecture.png"
    tsne_layer_plot_path = output_dir / "all_points_tsne_by_layer_type.png"
    tsne_points_csv_path = output_dir / "all_points_tsne2d.csv"
    tsne_coords = torch.empty((0, 2), dtype=torch.float32)
    tsne_model_labels: list[str] = []
    tsne_arch_labels: list[str] = []
    tsne_layer_labels: list[str] = []
    tsne_meta: dict[str, Any] = {
        "input_points": int(all_points.shape[0]),
        "used_points": 0,
        "max_points": int(args.tsne_max_points),
        "perplexity": float(args.tsne_perplexity),
        "learning_rate": float(args.tsne_learning_rate),
        "n_iter": int(args.tsne_n_iter),
        "points_csv_path": str(tsne_points_csv_path),
        "architecture_plot_path": str(tsne_arch_plot_path),
        "layer_type_plot_path": str(tsne_layer_plot_path),
        "computed": False,
        "architecture_plot_saved": False,
        "layer_type_plot_saved": False,
        "error": None,
        "matplotlib_available": bool(plt is not None),
    }

    num_all_points = int(all_points.shape[0])
    if num_all_points >= 3 and int(args.tsne_max_points) != 0:
        max_tsne_points = int(args.tsne_max_points)
        if max_tsne_points < 0:
            max_tsne_points = num_all_points
        if 0 < max_tsne_points < num_all_points:
            tsne_sample_gen = torch.Generator(device="cpu")
            tsne_sample_gen.manual_seed(int(args.seed) + 1001)
            sampled_idx = torch.randperm(num_all_points, generator=tsne_sample_gen)[:max_tsne_points]
            sampled_idx_list = sampled_idx.tolist()
            tsne_input = all_points.index_select(0, sampled_idx)
            tsne_model_labels = [all_patch_model_labels[i] for i in sampled_idx_list]
            tsne_arch_labels = [all_patch_arch_labels[i] for i in sampled_idx_list]
            tsne_layer_labels = [all_patch_layer_labels[i] for i in sampled_idx_list]
        else:
            tsne_input = all_points
            tsne_model_labels = list(all_patch_model_labels)
            tsne_arch_labels = list(all_patch_arch_labels)
            tsne_layer_labels = list(all_patch_layer_labels)

        tsne_meta["used_points"] = int(tsne_input.shape[0])
        try:
            tsne_coords = compute_tsne_projection_2d(
                tsne_input,
                perplexity=float(args.tsne_perplexity),
                learning_rate=float(args.tsne_learning_rate),
                n_iter=int(args.tsne_n_iter),
                seed=int(args.seed),
            )
            tsne_meta["computed"] = True
        except Exception as exc:
            tsne_meta["error"] = str(exc)
            LOGGER.warning("Could not compute t-SNE projection: %s", exc)
    elif num_all_points > 0:
        tsne_meta["used_points"] = int(min(num_all_points, max(0, int(args.tsne_max_points))))
        tsne_meta["error"] = "Not enough points for t-SNE (need at least 3)"
    else:
        tsne_meta["error"] = "No points collected"

    with tsne_points_csv_path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["point_idx", "tsne_1", "tsne_2", "model", "architecture", "layer_type"],
        )
        writer.writeheader()
        for idx in range(int(tsne_coords.shape[0])):
            writer.writerow(
                {
                    "point_idx": idx,
                    "tsne_1": float(tsne_coords[idx, 0].item()),
                    "tsne_2": float(tsne_coords[idx, 1].item()),
                    "model": tsne_model_labels[idx],
                    "architecture": tsne_arch_labels[idx],
                    "layer_type": tsne_layer_labels[idx],
                }
            )

    tsne_arch_saved = _scatter_by_labels(
        tsne_coords,
        tsne_arch_labels,
        title="t-SNE of Weight Patches (Colored by Architecture)",
        out_path=tsne_arch_plot_path,
        x_label="t-SNE 1",
        y_label="t-SNE 2",
    )
    tsne_layer_saved = _scatter_by_labels(
        tsne_coords,
        tsne_layer_labels,
        title="t-SNE of Weight Patches (Colored by Layer Type)",
        out_path=tsne_layer_plot_path,
        x_label="t-SNE 1",
        y_label="t-SNE 2",
    )
    tsne_meta["architecture_plot_saved"] = tsne_arch_saved
    tsne_meta["layer_type_plot_saved"] = tsne_layer_saved

    summary["global_all_points_plot"] = global_plot_meta
    summary["global_all_points_tsne"] = tsne_meta
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved PCA summary to %s", summary_path)
    LOGGER.info("Saved PCA tensors to %s", tensors_path)
    LOGGER.info("Saved explained variance table to %s", csv_path)
    LOGGER.info("Saved PCA point coordinates to %s", pca_points_csv_path)
    if global_plot_meta["architecture_plot_saved"]:
        LOGGER.info("Saved all-points PCA architecture plot to %s", pca_arch_plot_path)
    else:
        LOGGER.info("All-points PCA architecture plot was not saved")
    if global_plot_meta["layer_type_plot_saved"]:
        LOGGER.info("Saved all-points PCA layer-type plot to %s", pca_layer_plot_path)
    else:
        LOGGER.info("All-points PCA layer-type plot was not saved")
    LOGGER.info("Saved t-SNE point coordinates to %s", tsne_points_csv_path)
    if not tsne_meta["computed"]:
        LOGGER.info(
            "t-SNE embedding was not computed: error=%s input_points=%s used_points=%s",
            tsne_meta.get("error"),
            tsne_meta.get("input_points"),
            tsne_meta.get("used_points"),
        )
    if tsne_meta["architecture_plot_saved"]:
        LOGGER.info("Saved all-points t-SNE architecture plot to %s", tsne_arch_plot_path)
    else:
        LOGGER.info(
            "All-points t-SNE architecture plot was not saved (error=%s, matplotlib_available=%s)",
            tsne_meta.get("error"),
            tsne_meta.get("matplotlib_available"),
        )
    if tsne_meta["layer_type_plot_saved"]:
        LOGGER.info("Saved all-points t-SNE layer-type plot to %s", tsne_layer_plot_path)
    else:
        LOGGER.info(
            "All-points t-SNE layer-type plot was not saved (error=%s, matplotlib_available=%s)",
            tsne_meta.get("error"),
            tsne_meta.get("matplotlib_available"),
        )


if __name__ == "__main__":
    main()
