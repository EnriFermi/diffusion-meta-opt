from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Mapping

import torch

from ..common import env_bool, env_int, primary_dataset_name, write_csv, write_json
from big_vae.datasets.offline import infer_layer_depth, infer_layer_type

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("evaluate_big_vae_heldout")
_LATENT_DUMP_BALANCE_KEY_DEFAULT = "dataset,model,layer_type,depth_label"

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
