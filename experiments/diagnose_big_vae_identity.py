from __future__ import annotations

import copy
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from dataset import data_pipeline, setup_logging
from models.weight_quantile_vae import (
    BigVAEConfig,
    BigWeightVAE,
    DistributionConfig,
    EncoderConfig,
    MiniVAEConfig,
    ModelConfig,
)


EPS = 1e-8
LOSS_THRESHOLDS = (1e-2, 1e-3, 1e-4)


@dataclass(slots=True)
class FrozenTarget:
    W_s: torch.Tensor
    x_s: torch.Tensor
    X_patch_target: torch.Tensor
    u_target: torch.Tensor
    r_target: torch.Tensor
    w_dir: torch.Tensor
    meta: dict[str, Any]
    x_mask_s: torch.Tensor | None = None
    d_in_mask_s: torch.Tensor | None = None

    def to(self, device: torch.device) -> "FrozenTarget":
        return FrozenTarget(
            W_s=self.W_s.to(device),
            x_s=self.x_s.to(device),
            X_patch_target=self.X_patch_target.to(device),
            u_target=self.u_target.to(device),
            r_target=self.r_target.to(device),
            w_dir=self.w_dir.to(device),
            meta=dict(self.meta),
            x_mask_s=self.x_mask_s.to(device) if self.x_mask_s is not None else None,
            d_in_mask_s=self.d_in_mask_s.to(device) if self.d_in_mask_s is not None else None,
        )

    def save(self, path: Path) -> None:
        payload = {
            "W_s": self.W_s.detach().cpu(),
            "x_s": self.x_s.detach().cpu(),
            "X_patch_target": self.X_patch_target.detach().cpu(),
            "u_target": self.u_target.detach().cpu(),
            "r_target": self.r_target.detach().cpu(),
            "w_dir": self.w_dir.detach().cpu(),
            "meta": self.meta,
            "x_mask_s": self.x_mask_s.detach().cpu() if self.x_mask_s is not None else None,
            "d_in_mask_s": self.d_in_mask_s.detach().cpu() if self.d_in_mask_s is not None else None,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: Path) -> "FrozenTarget":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return FrozenTarget(
            W_s=payload["W_s"],
            x_s=payload["x_s"],
            X_patch_target=payload["X_patch_target"],
            u_target=payload["u_target"],
            r_target=payload["r_target"],
            w_dir=payload["w_dir"],
            meta=dict(payload["meta"]),
            x_mask_s=payload.get("x_mask_s"),
            d_in_mask_s=payload.get("d_in_mask_s"),
        )


def _logger() -> logging.Logger:
    return logging.getLogger("diagnose_big_vae_identity")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml_without_defaults(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        del cfg["defaults"]
    return cfg


def _compose_default_cfg() -> DictConfig:
    conf_root = _repo_root() / "conf"
    root_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "shared_runtime_environment" / "global_app_hydra_logging_and_model_registry.yaml"),
        _load_yaml_without_defaults(conf_root / "shared_runtime_environment" / "checkpoint_and_artifact_paths_for_mini_then_big.yaml"),
    )
    data_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "data_profiles" / "data_profile_common_settings.yaml"),
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "data_profiles" / "data_profile_for_big_vae_training.yaml"),
    )
    collector_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "collector_profiles" / "collector_common_settings.yaml"),
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "collector_profiles" / "collector_profile_async_gpu1.yaml"),
    )
    streaming_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "streaming_profiles" / "streaming_common_settings.yaml"),
        _load_yaml_without_defaults(conf_root / "data_collection_runtime" / "streaming_profiles" / "streaming_profile_none.yaml"),
    )
    train_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "shared_training_parameters" / "trainer_execution_common.yaml"),
        _load_yaml_without_defaults(conf_root / "big_vae_experiment" / "trainer_parameters_specific_to_big_vae_stage.yaml"),
    )
    model_cfg = OmegaConf.merge(
        _load_yaml_without_defaults(conf_root / "shared_training_parameters" / "model_backbone_distribution_and_mini_vae.yaml"),
        _load_yaml_without_defaults(conf_root / "big_vae_experiment" / "model_parameters_specific_to_big_vae_stage.yaml"),
    )
    run_profile_cfg = _load_yaml_without_defaults(conf_root / "run_profiles" / "train_big_vae.yaml")
    diagnostics_cfg = _load_yaml_without_defaults(conf_root / "big_vae_experiment" / "diagnostics_identity.yaml")

    cfg = OmegaConf.merge(
        root_cfg,
        OmegaConf.create(
            {
                "data": data_cfg,
                "collector": collector_cfg,
                "streaming": streaming_cfg,
                "train": train_cfg,
                "model": model_cfg,
            }
        ),
        run_profile_cfg,
        diagnostics_cfg,
    )
    return cfg


def _apply_cli_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    if not overrides:
        return cfg
    dotlist_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, dotlist_cfg)


def _build_model_cfg_local(cfg: DictConfig) -> ModelConfig:
    model_cfg = cfg.get("model", {})
    dist_cfg = model_cfg.get("distribution", {})
    mini_cfg = model_cfg.get("mini_vae", {})
    big_cfg = model_cfg.get("big_vae", {})
    enc_cfg = big_cfg.get("encoder", {})

    return ModelConfig(
        patch_size=int(model_cfg.get("patch_size", 16)),
        beta=float(model_cfg.get("beta", 1e-3)),
        distribution=DistributionConfig(
            k_s=int(dist_cfg.get("k_s", 16)),
            Kq=int(dist_cfg.get("Kq", 32)),
            d_var=int(dist_cfg.get("d_var", 128)),
            d_dist=int(dist_cfg.get("d_dist", 128)),
            num_var_attn_layers=int(dist_cfg.get("num_var_attn_layers", 2)),
            var_attn_heads=int(dist_cfg.get("var_attn_heads", 4)),
            dcn_num_cross_layers=int(dist_cfg.get("dcn_num_cross_layers", 3)),
            dcn_deep_hidden=int(dist_cfg.get("dcn_deep_hidden", 0)),
            dcn_deep_layers=int(dist_cfg.get("dcn_deep_layers", 0)),
            dropout=float(dist_cfg.get("dropout", 0.0)),
            use_covariance=bool(dist_cfg.get("use_covariance", True)),
            patch_size_for_cov=int(dist_cfg.get("patch_size_for_cov", int(model_cfg.get("patch_size", 16)))),
        ),
        mini_vae=MiniVAEConfig(
            z_dim=int(mini_cfg.get("z_dim", 64)),
            d_e=int(mini_cfg.get("d_e", 128)),
            pos_dim=int(mini_cfg.get("pos_dim", 32)),
            num_attn_layers_encoder=int(mini_cfg.get("num_attn_layers_encoder", 2)),
            num_layers_decoder=int(mini_cfg.get("num_layers_decoder", 2)),
            n_heads=int(mini_cfg.get("n_heads", 4)),
            d_patch=int(mini_cfg.get("d_patch", 64)),
            dropout=float(mini_cfg.get("dropout", 0.0)),
            mlp_stub_hidden_dim=int(mini_cfg.get("mlp_stub_hidden_dim", 256)),
        ),
        big_vae=BigVAEConfig(
            d_model=int(big_cfg.get("d_model", 256)),
            d_lat=int(big_cfg.get("d_lat", 256)),
            num_latents=int(big_cfg.get("num_latents", 32)),
            num_encoder_layers=int(big_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(big_cfg.get("num_decoder_layers", 4)),
            n_heads=int(big_cfg.get("n_heads", 8)),
            ffn_mult=float(big_cfg.get("ffn_mult", 4.0)),
            dropout=float(big_cfg.get("dropout", 0.0)),
            pos_fourier_dim=int(big_cfg.get("pos_fourier_dim", 64)),
            use_latent_sampling=bool(big_cfg.get("use_latent_sampling", True)),
            latent_sampling_min_std=float(big_cfg.get("latent_sampling_min_std", 1e-4)),
            latent_sampling_logvar_min=float(big_cfg.get("latent_sampling_logvar_min", -20.0)),
            latent_sampling_logvar_max=float(big_cfg.get("latent_sampling_logvar_max", 10.0)),
            disable_z_shortcut=bool(big_cfg.get("disable_z_shortcut", False)),
            disable_distribution_encoder=bool(big_cfg.get("disable_distribution_encoder", False)),
            patch_tokenizer_kind=str(big_cfg.get("patch_tokenizer_kind", "residual")),
            distribution_encoder_conditioning_kind=str(big_cfg.get("distribution_encoder_conditioning_kind", "legacy")),
            encoder=EncoderConfig(
                self_attn_mode=str(enc_cfg.get("self_attn_mode", "full")),
                cross_attend_only_cls=bool(enc_cfg.get("cross_attend_only_cls", True)),
            ),
        ),
    )


def _compute_curriculum_slice_sizes_local(cfg: DictConfig) -> tuple[int, int]:
    train_cfg = cfg.get("train", {})
    stage = max(1, int(train_cfg.get("stage", 1)))
    base_T = int(train_cfg.get("stage_base_T_patches", 4))
    base_d_out = int(train_cfg.get("stage_base_d_out", 16))
    scale = int(train_cfg.get("stage_scale_factor", 2))
    max_T = base_T * (scale ** (stage - 1))
    max_d_out = base_d_out * (scale ** (stage - 1))
    return max_T, max_d_out


def _load_checkpoint_local(model: BigWeightVAE, path: str, logger: logging.Logger) -> None:
    logger.info("Loading model weights from checkpoint: %s", path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state"]
    model.load_state_dict(state_dict, strict=True)


def _strip_ddp_prefix(name: str) -> str:
    out = str(name)
    changed = True
    while changed:
        changed = False
        if out.startswith("module."):
            out = out[len("module.") :]
            changed = True
        if out.startswith("_orig_mod."):
            out = out[len("_orig_mod.") :]
            changed = True
    return out


def _grad_stat_group_prefixes() -> dict[str, tuple[str, ...]]:
    return {
        "distribution_encoder": ("distribution_encoder.",),
        "patch_tokenizer": ("patch_tokenizer.", "patch_token_proj.", "cls_token"),
        "encoder": (
            "encoder_layers.",
            "latent_resampler_layers.",
            "enc_dist_inject_projs.",
            "enc_dist_to_latent_heads.",
            "encoder_conditioning_adapters.",
            "latent_base",
            "latent_norm.",
        ),
        "decoder": (
            "decoder_layers.",
            "latent_to_decoder.",
            "pos_proj.",
            "query_pos_proj.",
            "query_proj.",
            "direction_head.",
            "z_shortcut_proj.",
            "z_shortcut.",
            "scale_head.",
        ),
        "big_vae_other": tuple(),
    }


def _grad_group_name_for_param(name: str, groups: dict[str, tuple[str, ...]]) -> str:
    for group_name, prefixes in groups.items():
        if group_name == "big_vae_other":
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            return group_name
    return "big_vae_other"


def _compute_grad_stats_local(model: BigWeightVAE) -> dict[str, float]:
    groups = _grad_stat_group_prefixes()
    group_sums = {name: 0.0 for name in groups}
    group_numel = {name: 0 for name in groups}
    group_param_count = {name: 0 for name in groups}

    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0
    grad_numel = 0
    param_sum_sq = 0.0
    param_numel = 0

    for raw_name, param in model.named_parameters():
        name = _strip_ddp_prefix(raw_name)
        if not param.requires_grad:
            continue

        p = param.detach()
        param_sum_sq += float(p.pow(2).sum().item())
        param_numel += int(p.numel())

        grad = param.grad
        if grad is None:
            continue

        g = grad.detach()
        abs_g = g.abs()
        sum_sq += float((g * g).sum().item())
        sum_abs += float(abs_g.sum().item())
        max_abs = max(max_abs, float(abs_g.max().item()))
        numel = int(g.numel())
        grad_numel += numel

        group_name = _grad_group_name_for_param(name, groups)
        group_param_count[group_name] += 1
        group_sums[group_name] += float((g * g).sum().item())
        group_numel[group_name] += numel

    grad_rms = math.sqrt(sum_sq / max(1, grad_numel))
    param_rms = math.sqrt(param_sum_sq / max(1, param_numel))
    payload: dict[str, float] = {
        "grad/global_norm": math.sqrt(sum_sq),
        "grad/rms": grad_rms,
        "grad/abs_mean": (sum_abs / max(1, grad_numel)),
        "grad/max_abs": max_abs,
        "grad/numel": float(grad_numel),
        "param/rms": param_rms,
        "grad_to_param_rms_ratio": grad_rms / max(1e-12, param_rms),
    }
    for group_name in groups:
        payload[f"grad/{group_name}_rms"] = (
            math.sqrt(group_sums[group_name] / max(1, group_numel[group_name])) if group_numel[group_name] > 0 else 0.0
        )
        payload[f"grad/{group_name}_numel"] = float(group_numel[group_name])
        payload[f"grad/{group_name}_params_with_grad"] = float(group_param_count[group_name])
    return payload


def _resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(cfg.train.get("device", "cuda:0")).strip().lower()
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested if requested.startswith("cuda") else "cuda:0")


def _configure_reproducibility(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _artifact_root(cfg: DictConfig) -> Path:
    base = _repo_root() / str(cfg.diagnostics.get("output_root", "artifacts/big_vae_identity_diagnostics"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = base / timestamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_safe_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_scalar(item) for key, item in value.items()}
    return value


def _plot_grouped_lines(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_key: str,
    path: Path,
    title: str,
    ylabel: str,
    ylog: bool = False,
) -> str | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        note = f"matplotlib unavailable: {exc}"
        path.with_suffix(".txt").write_text(note, encoding="utf-8")
        return note

    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        group = str(row[group_key])
        xs, ys = grouped.setdefault(group, ([], []))
        xs.append(float(row[x_key]))
        ys.append(float(row[y_key]))

    plt.figure(figsize=(8, 4.5))
    for label, (xs, ys) in grouped.items():
        plt.plot(xs, ys, label=label, linewidth=1.6)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    if ylog:
        plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return None


def _plot_interpolation(rows: list[dict[str, Any]], path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        note = f"matplotlib unavailable: {exc}"
        path.with_suffix(".txt").write_text(note, encoding="utf-8")
        return note

    lambdas = [float(row["lambda"]) for row in rows]
    losses = [float(row["loss_dir"]) for row in rows]
    weighted_cos = [float(row["weighted_cos"]) for row in rows]
    mean_cos = [float(row["mean_cos"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(lambdas, losses, linewidth=1.8)
    axes[0].set_ylabel("loss_dir")
    axes[0].set_title("Experiment 2: interpolation geometry")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(lambdas, weighted_cos, label="weighted_cos", linewidth=1.6)
    axes[1].plot(lambdas, mean_cos, label="mean_cos", linewidth=1.6)
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("cosine")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return None


def _cpu_state_dict(model: BigWeightVAE) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _tensor_debug_stats_local(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"is_none": True}

    detached = tensor.detach()
    flat = detached.reshape(-1)
    numel = int(flat.numel())
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    payload: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": numel,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    if finite_count > 0:
        finite_values = flat[finite_mask]
        payload["min"] = float(finite_values.min().item())
        payload["max"] = float(finite_values.max().item())
        payload["abs_max"] = float(finite_values.abs().max().item())
        payload["mean"] = float(finite_values.mean().item())
        payload["std"] = float(finite_values.std(unbiased=False).item())
    return payload


def _build_model_state(
    cfg: DictConfig,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[Any, dict[str, torch.Tensor], str]:
    model_cfg = _build_model_cfg_local(cfg)
    model = BigWeightVAE(model_cfg).to(device)
    checkpoint_path = str(
        cfg.diagnostics.get("checkpoint_path", cfg.train.get("resume_checkpoint", ""))
    ).strip()
    if checkpoint_path:
        _load_checkpoint_local(model, checkpoint_path, logger)
    state = _cpu_state_dict(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model_cfg, state, checkpoint_path


def _instantiate_model(
    model_cfg: Any,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> BigWeightVAE:
    model = BigWeightVAE(model_cfg).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


def _next_valid_sample_deterministic(
    dataset_iter: Any,
    *,
    max_x_rows: int,
    generator: torch.Generator,
    logger: logging.Logger,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    attempts = 0
    while True:
        sample = next(dataset_iter)
        x = getattr(sample, "x", None)
        W = getattr(sample, "weight", None)
        valid = (
            torch.is_tensor(x)
            and torch.is_tensor(W)
            and x.ndim == 2
            and W.ndim == 2
            and x.shape[1] == W.shape[0]
            and x.shape[0] > 0
            and W.shape[0] > 0
            and W.shape[1] > 0
        )
        if not valid:
            attempts += 1
            if attempts % 50 == 0:
                logger.warning("Skipping invalid sample repeatedly; attempts=%s", attempts)
            continue

        x_cpu = x.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        W_cpu = W.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        x_row_indices: list[int] | None = None
        if max_x_rows > 0 and x_cpu.shape[0] > max_x_rows:
            keep = torch.randperm(x_cpu.shape[0], generator=generator)[:max_x_rows].sort().values
            x_row_indices = keep.tolist()
            x_cpu = x_cpu[keep]

        sample_meta = {
            "model_name": str(getattr(sample, "model_name", "")),
            "layer_name": str(getattr(sample, "layer_name", "")),
            "sample_meta": _safe_scalar(getattr(sample, "meta", {})),
            "x_row_indices": x_row_indices,
            "original_x_shape": list(x.shape),
            "original_W_shape": list(W.shape),
        }
        return x_cpu, W_cpu, sample_meta


def _sample_synthetic_deterministic(
    cfg: DictConfig,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    syn_cfg = cfg.diagnostics.frozen_target.synthetic
    n_rows = max(1, int(syn_cfg.get("n_rows", 256)))
    d_in = max(1, int(syn_cfg.get("d_in", 256)))
    d_out = max(1, int(syn_cfg.get("d_out", 32)))
    x_std = float(syn_cfg.get("x_std", 1.0))
    w_std = float(syn_cfg.get("w_std", 1.0))
    x = torch.randn((n_rows, d_in), generator=generator, dtype=torch.float32) * x_std
    W = torch.randn((d_in, d_out), generator=generator, dtype=torch.float32) * w_std
    meta = {
        "kind": "synthetic",
        "n_rows": n_rows,
        "d_in": d_in,
        "d_out": d_out,
        "x_std": x_std,
        "w_std": w_std,
    }
    return x, W, meta


def _deterministic_slice_sample(
    W: torch.Tensor,
    x: torch.Tensor,
    *,
    max_T_patches: int,
    max_d_out: int,
    patch_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    d_in, d_out = W.shape
    T_total = d_in // patch_size
    T_use = min(max_T_patches, T_total) if T_total > 0 else 0
    d_out_use = min(max_d_out, d_out)
    need_row_slice = T_total > 0 and T_use < T_total
    need_col_slice = d_out_use < d_out
    offsets = torch.arange(patch_size, dtype=torch.long) if need_row_slice else None

    W_slices: list[torch.Tensor] = []
    x_slices: list[torch.Tensor] = []
    row_patch_indices_all: list[list[int] | None] = []
    col_indices_all: list[list[int] | None] = []

    for _ in range(batch_size):
        W_i = W
        x_i = x
        row_patch_indices: list[int] | None = None
        col_indices: list[int] | None = None

        if need_row_slice:
            patch_idx = torch.randperm(T_total, generator=generator)[:T_use].sort().values
            row_patch_indices = patch_idx.tolist()
            row_idx = (patch_idx.unsqueeze(1) * patch_size + offsets.unsqueeze(0)).flatten()
            W_i = W_i[row_idx, :]
            x_i = x_i[:, row_idx]

        if need_col_slice:
            col_idx = torch.randperm(d_out, generator=generator)[:d_out_use].sort().values
            col_indices = col_idx.tolist()
            W_i = W_i[:, col_idx]

        W_slices.append(W_i.clone())
        x_slices.append(x_i.clone())
        row_patch_indices_all.append(row_patch_indices)
        col_indices_all.append(col_indices)

    meta = {
        "batch_size": batch_size,
        "max_T_patches": int(max_T_patches),
        "max_d_out": int(max_d_out),
        "row_patch_indices": row_patch_indices_all,
        "col_indices": col_indices_all,
        "T_total_floor": int(T_total),
        "T_use_floor": int(T_use),
        "d_out_total": int(d_out),
        "d_out_use": int(d_out_use),
    }
    return torch.stack(W_slices), torch.stack(x_slices), meta


def _derive_frozen_target(
    W_s: torch.Tensor,
    x_s: torch.Tensor,
    *,
    patch_size: int,
    gamma: float,
) -> FrozenTarget:
    if W_s.ndim != 3 or x_s.ndim != 3:
        raise ValueError(f"W_s and x_s must be batched rank-3 tensors, got {tuple(W_s.shape)} and {tuple(x_s.shape)}")

    B, d_in, d_out = W_s.shape
    T_loss = d_in // patch_size
    if T_loss <= 0:
        raise ValueError(f"d_in={d_in} with patch_size={patch_size} gives T_loss={T_loss}")

    used = T_loss * patch_size
    X_patch_target = W_s[:, :used, :].transpose(1, 2).contiguous().view(B, d_out, T_loss, patch_size)
    r_target = X_patch_target.norm(dim=-1)
    u_target = X_patch_target / (r_target.unsqueeze(-1) + EPS)
    w_dir = (r_target + EPS).pow(float(gamma))
    w_dir = w_dir / (w_dir.sum(dim=-1, keepdim=True) + EPS)
    T_forward = (d_in + patch_size - 1) // patch_size

    meta = {
        "B": int(B),
        "d_in": int(d_in),
        "d_out": int(d_out),
        "patch_size": int(patch_size),
        "T": int(T_loss),
        "T_loss": int(T_loss),
        "T_forward": int(T_forward),
        "used_T_in_loss": int(T_loss),
        "used_T_in_forward": int(T_forward),
        "used_rows_in_loss": int(used),
        "gamma": float(gamma),
        "target_patch_shape": list(X_patch_target.shape),
        "pred_dirs_shape_forward": [int(B), int(d_out), int(T_forward), int(patch_size)],
    }
    return FrozenTarget(
        W_s=W_s.detach().cpu(),
        x_s=x_s.detach().cpu(),
        X_patch_target=X_patch_target.detach().cpu(),
        u_target=u_target.detach().cpu(),
        r_target=r_target.detach().cpu(),
        w_dir=w_dir.detach().cpu(),
        meta=meta,
    )


def _build_frozen_target(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
    artifact_root: Path,
) -> tuple[FrozenTarget, dict[str, Any]]:
    frozen_cfg = cfg.diagnostics.frozen_target
    source_kind = str(frozen_cfg.get("source_kind", "real")).strip().lower()
    seed = int(cfg.diagnostics.get("seed", 0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if bool(frozen_cfg.get("use_curriculum_slice", True)):
        max_T_patches, max_d_out = _compute_curriculum_slice_sizes_local(cfg)
    else:
        max_T_patches = int(frozen_cfg.get("max_T_patches", 1))
        max_d_out = int(frozen_cfg.get("max_d_out", 1))

    max_T_override = frozen_cfg.get("max_T_patches")
    max_d_out_override = frozen_cfg.get("max_d_out")
    if max_T_override is not None:
        max_T_patches = int(max_T_override)
    if max_d_out_override is not None:
        max_d_out = int(max_d_out_override)

    batch_size = max(1, int(frozen_cfg.get("batch_size", 1)))
    max_x_rows = int(frozen_cfg.get("max_x_rows", cfg.train.get("max_x_rows", 0)))
    patch_size = int(cfg.model.get("patch_size", 16))
    struct_cfg = cfg.train.get("struct_loss", {})
    gamma = float(struct_cfg.get("gamma", 0.5))

    if source_kind == "synthetic":
        x_cpu, W_cpu, source_meta = _sample_synthetic_deterministic(cfg, generator=generator)
    else:
        with data_pipeline(
            cfg,
            logger=logger,
            emit_run_report=False,
            rank=0,
        ) as (dataset, _collector):
            dataset_iter = iter(dataset)
            x_cpu, W_cpu, source_meta = _next_valid_sample_deterministic(
                dataset_iter,
                max_x_rows=max_x_rows,
                generator=generator,
                logger=logger,
            )

    W_s, x_s, slice_meta = _deterministic_slice_sample(
        W_cpu,
        x_cpu,
        max_T_patches=max_T_patches,
        max_d_out=max_d_out,
        patch_size=patch_size,
        batch_size=batch_size,
        generator=generator,
    )
    frozen = _derive_frozen_target(W_s, x_s, patch_size=patch_size, gamma=gamma)
    frozen.meta["source_kind"] = source_kind
    frozen.meta["source_meta"] = source_meta
    frozen.meta["slice_meta"] = slice_meta

    dump_path = artifact_root / "frozen_target_dump.pt"
    frozen.save(dump_path)
    immutability_checks: list[dict[str, Any]] = []
    for step_idx in range(5):
        reloaded = FrozenTarget.load(dump_path)
        step_payload = {
            "step": step_idx,
            "W_s_equal": bool(torch.equal(reloaded.W_s, frozen.W_s)),
            "x_s_equal": bool(torch.equal(reloaded.x_s, frozen.x_s)),
            "X_patch_target_equal": bool(torch.equal(reloaded.X_patch_target, frozen.X_patch_target)),
            "u_target_equal": bool(torch.equal(reloaded.u_target, frozen.u_target)),
            "r_target_equal": bool(torch.equal(reloaded.r_target, frozen.r_target)),
            "w_dir_equal": bool(torch.equal(reloaded.w_dir, frozen.w_dir)),
        }
        step_payload["all_equal"] = all(bool(value) for key, value in step_payload.items() if key.endswith("_equal"))
        immutability_checks.append(step_payload)

    meta_payload = {
        **frozen.meta,
        "immutability_checks": immutability_checks,
    }
    _write_json(artifact_root / "frozen_target_meta.json", _safe_scalar(meta_payload))
    return frozen, meta_payload


def _exact_dir_loss(
    frozen: FrozenTarget,
    pred_dirs: torch.Tensor,
    *,
    W_hat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dummy = torch.zeros_like(frozen.W_s) if W_hat is None else W_hat
    return BigWeightVAE.patch_structure_loss(
        frozen.W_s,
        dummy,
        patch_size=int(frozen.meta["patch_size"]),
        eps=EPS,
        gamma=float(frozen.meta["gamma"]),
        lambda_dir=1.0,
        lambda_scale=0.0,
        lambda_rec=0.0,
        lambda_rel=0.0,
        pred_dirs=pred_dirs,
        d_in_mask=frozen.d_in_mask_s,
    )


def _direction_metrics(
    frozen: FrozenTarget,
    pred_dirs: torch.Tensor,
) -> dict[str, float]:
    T_loss = int(frozen.meta["used_T_in_loss"])
    pred_dirs_used = pred_dirs[:, :, :T_loss, :]
    cos = (pred_dirs_used * frozen.u_target).sum(dim=-1)
    weighted_cos = (frozen.w_dir * cos).sum(dim=-1).mean()
    return {
        "mean_cos": float(cos.mean().item()),
        "weighted_cos": float(weighted_cos.item()),
        "max_cos": float(cos.max().item()),
        "min_cos": float(cos.min().item()),
    }


def _threshold_steps(rows: list[dict[str, Any]], *, key: str = "loss") -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for threshold in LOSS_THRESHOLDS:
        found = next((int(row["step"]) for row in rows if float(row[key]) <= threshold), None)
        result[f"{threshold:.0e}"] = found
    return result


def _generic_free_tensor_run(
    frozen: FrozenTarget,
    *,
    optimizer_name: str,
    lr: float,
    steps: int,
    seed: int,
    loss_name: str,
    loss_fn: Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]],
    init_tensor: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]:
    device = frozen.W_s.device
    generator = torch.Generator(device=device.type if device.type == "cpu" else "cpu")
    generator.manual_seed(seed)
    A = torch.nn.Parameter(
        init_tensor.clone() if init_tensor is not None else torch.randn_like(frozen.u_target, device=device)
    )
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD([A], lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam([A], lr=lr, weight_decay=0.0)
    else:
        raise ValueError(f"Unsupported optimizer_name={optimizer_name}")

    rows: list[dict[str, Any]] = []
    best_loss = math.inf
    best_bundle: dict[str, torch.Tensor] = {}

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, bundle = loss_fn(A)
        row = {
            "mode": loss_name,
            "optimizer": optimizer_name,
            "step": step,
            "loss": float(loss.detach().item()),
            "grad_norm": math.nan,
            **metrics,
        }
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_bundle = {
                key: value.detach().cpu().clone()
                for key, value in bundle.items()
            }

        if step < steps:
            loss.backward()
            grad_norm = float(A.grad.detach().norm().item()) if A.grad is not None else 0.0
            row["grad_norm"] = grad_norm
            optimizer.step()

        rows.append(row)

    summary = {
        "mode": loss_name,
        "optimizer": optimizer_name,
        "initial_loss": float(rows[0]["loss"]),
        "final_loss": float(rows[-1]["loss"]),
        "best_loss": float(min(float(row["loss"]) for row in rows)),
        "steps_to_threshold": _threshold_steps(rows),
        "final_weighted_cos": float(rows[-1]["weighted_cos"]),
        "best_weighted_cos": float(max(float(row["weighted_cos"]) for row in rows)),
    }
    return rows, summary, best_bundle


def _experiment_1(
    frozen: FrozenTarget,
    cfg: DictConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    logger = _logger()
    exp_cfg = cfg.diagnostics.experiments.loss_only
    steps = max(1, int(exp_cfg.get("steps", 1500)))
    base_seed = int(cfg.diagnostics.get("seed", 0)) + 101

    def loss_fn_builder(current_frozen: FrozenTarget) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]]:
        def _loss_fn(A: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
            pred_dirs = F.normalize(A, dim=-1, eps=EPS)
            loss, details = _exact_dir_loss(current_frozen, pred_dirs)
            metrics = {
                "loss_dir": float(details["L_dir"].item()),
                **_direction_metrics(current_frozen, pred_dirs),
            }
            bundle = {
                "A": A.detach(),
                "pred_dirs": pred_dirs.detach(),
            }
            return loss, metrics, bundle
        return _loss_fn

    results: dict[str, Any] = {"optimizers": {}}
    all_rows: list[dict[str, Any]] = []
    for offset, (optimizer_name, lr) in enumerate(
        (
            ("sgd", float(exp_cfg.get("lr_sgd", 0.5))),
            ("adam", float(exp_cfg.get("lr_adam", 0.05))),
        )
    ):
        logger.info("Experiment 1 [%s]: steps=%s lr=%s", optimizer_name, steps, lr)
        rows, summary, best_bundle = _generic_free_tensor_run(
            frozen,
            optimizer_name=optimizer_name,
            lr=lr,
            steps=steps,
            seed=base_seed + offset,
            loss_name="current_exact",
            loss_fn=loss_fn_builder(frozen),
        )
        all_rows.extend(rows)
        results["optimizers"][optimizer_name] = summary
        torch.save(best_bundle, artifact_dir / f"best_bundle_{optimizer_name}.pt")

    _write_csv(artifact_dir / "trajectory.csv", _safe_scalar(all_rows))
    plot_note = _plot_grouped_lines(
        all_rows,
        x_key="step",
        y_key="loss",
        group_key="optimizer",
        path=artifact_dir / "loss_vs_step.png",
        title="Experiment 1: loss-only free tensor",
        ylabel="loss",
        ylog=True,
    )
    results["plot_note"] = plot_note
    _write_json(artifact_dir / "metrics.json", _safe_scalar(results))
    return results


def _experiment_2(
    frozen: FrozenTarget,
    cfg: DictConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    steps = max(2, int(cfg.diagnostics.experiments.interpolation.get("num_points", 51)))
    seed = int(cfg.diagnostics.get("seed", 0)) + 202
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # Sample deterministically on CPU, then move to match the frozen target device.
    random_logits = torch.randn(
        frozen.u_target.shape,
        generator=generator,
        dtype=frozen.u_target.dtype,
    ).to(device=frozen.u_target.device)

    rows: list[dict[str, Any]] = []
    for idx in range(steps):
        lam = idx / float(steps - 1)
        A_lambda = (1.0 - lam) * random_logits + lam * frozen.u_target
        pred_dirs = F.normalize(A_lambda, dim=-1, eps=EPS)
        loss, details = _exact_dir_loss(frozen, pred_dirs)
        metrics = _direction_metrics(frozen, pred_dirs)
        rows.append(
            {
                "lambda": lam,
                "loss_dir": float(details["L_dir"].item()),
                "loss": float(loss.item()),
                **metrics,
            }
        )

    deltas = [float(rows[idx + 1]["loss_dir"]) - float(rows[idx]["loss_dir"]) for idx in range(len(rows) - 1)]
    monotonic_nonincreasing = all(delta <= 1e-6 for delta in deltas)
    result = {
        "num_points": steps,
        "loss_at_lambda_0": float(rows[0]["loss_dir"]),
        "loss_at_lambda_1": float(rows[-1]["loss_dir"]),
        "monotonic_nonincreasing": monotonic_nonincreasing,
        "max_positive_delta": float(max(deltas) if deltas else 0.0),
    }
    _write_csv(artifact_dir / "interpolation.csv", _safe_scalar(rows))
    plot_note = _plot_interpolation(rows, artifact_dir / "interpolation.png")
    result["plot_note"] = plot_note
    _write_json(artifact_dir / "metrics.json", _safe_scalar(result))
    return result


def _normalized_mse_loss(frozen: FrozenTarget, pred_dirs: torch.Tensor) -> torch.Tensor:
    diff_sq = (pred_dirs[:, :, : int(frozen.meta["used_T_in_loss"]), :] - frozen.u_target).pow(2).sum(dim=-1)
    return (frozen.w_dir * diff_sq).sum(dim=-1).mean()


def _teacher_forced_loss(frozen: FrozenTarget, pred_dirs: torch.Tensor) -> torch.Tensor:
    pred_patch = frozen.r_target.unsqueeze(-1) * pred_dirs[:, :, : int(frozen.meta["used_T_in_loss"]), :]
    rec_sq = (pred_patch - frozen.X_patch_target).pow(2).sum(dim=-1)
    normalized = rec_sq / (frozen.r_target.pow(2) + EPS)
    return (frozen.w_dir * normalized).sum(dim=-1).mean()


def _experiment_3(
    frozen: FrozenTarget,
    cfg: DictConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    exp_cfg = cfg.diagnostics.experiments.alt_losses
    steps = max(1, int(exp_cfg.get("steps", 1200)))
    lr = float(exp_cfg.get("lr", 0.05))
    optimizer_name = str(exp_cfg.get("optimizer", "adam")).strip().lower()
    base_seed = int(cfg.diagnostics.get("seed", 0)) + 303

    def current_exact(A: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        pred_dirs = F.normalize(A, dim=-1, eps=EPS)
        loss, details = _exact_dir_loss(frozen, pred_dirs)
        return loss, {"loss_dir": float(details["L_dir"].item()), **_direction_metrics(frozen, pred_dirs)}, {
            "A": A.detach(),
            "pred_dirs": pred_dirs.detach(),
        }

    def normalized_mse(A: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        pred_dirs = F.normalize(A, dim=-1, eps=EPS)
        loss = _normalized_mse_loss(frozen, pred_dirs)
        return loss, _direction_metrics(frozen, pred_dirs), {"A": A.detach(), "pred_dirs": pred_dirs.detach()}

    def teacher_forced(A: torch.Tensor) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
        pred_dirs = F.normalize(A, dim=-1, eps=EPS)
        loss = _teacher_forced_loss(frozen, pred_dirs)
        return loss, _direction_metrics(frozen, pred_dirs), {"A": A.detach(), "pred_dirs": pred_dirs.detach()}

    mode_fns = {
        "current_exact": current_exact,
        "normalized_mse": normalized_mse,
        "teacher_forced_reconstruction": teacher_forced,
    }

    rows_all: list[dict[str, Any]] = []
    result: dict[str, Any] = {"modes": {}}
    for offset, (mode_name, loss_fn) in enumerate(mode_fns.items()):
        rows, summary, best_bundle = _generic_free_tensor_run(
            frozen,
            optimizer_name=optimizer_name,
            lr=lr,
            steps=steps,
            seed=base_seed + offset,
            loss_name=mode_name,
            loss_fn=loss_fn,
        )
        initial_loss = max(float(rows[0]["loss"]), EPS)
        for row in rows:
            row["normalized_loss"] = float(row["loss"]) / initial_loss
        rows_all.extend(rows)
        result["modes"][mode_name] = summary
        torch.save(best_bundle, artifact_dir / f"best_bundle_{mode_name}.pt")

    _write_csv(artifact_dir / "trajectory.csv", _safe_scalar(rows_all))
    plot_note = _plot_grouped_lines(
        rows_all,
        x_key="step",
        y_key="normalized_loss",
        group_key="mode",
        path=artifact_dir / "normalized_comparison.png",
        title="Experiment 3: equivalent directional objectives",
        ylabel="normalized_loss",
    )
    result["plot_note"] = plot_note
    _write_json(artifact_dir / "metrics.json", _safe_scalar(result))
    return result


def _variant_seed(base_seed: int, text: str) -> int:
    return base_seed + sum(ord(ch) for ch in text)


def _experiment_4_or_5_common_row(
    step: int,
    loss: torch.Tensor,
    frozen: FrozenTarget,
    pred_dirs: torch.Tensor,
    *,
    mode: str,
    direction_pre_norms: torch.Tensor | None = None,
) -> dict[str, Any]:
    row = {
        "mode": mode,
        "step": int(step),
        "loss": float(loss.detach().item()),
        **_direction_metrics(frozen, pred_dirs),
    }
    if direction_pre_norms is not None:
        stats = _tensor_debug_stats_local(direction_pre_norms)
        row.update(
            {
                "direction_pre_norm_mean": float(stats.get("mean", 0.0)),
                "direction_pre_norm_std": float(stats.get("std", 0.0)),
                "direction_pre_norm_min": float(stats.get("min", 0.0)),
                "direction_pre_norm_max": float(stats.get("max", 0.0)),
            }
        )
    return row


def _experiment_4(
    frozen_cpu: FrozenTarget,
    cfg: DictConfig,
    artifact_dir: Path,
    *,
    model_cfg: Any,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    logger = _logger()
    exp_cfg = cfg.diagnostics.experiments.free_latent_decoder
    steps = max(1, int(exp_cfg.get("steps", 1000)))
    lr = float(exp_cfg.get("lr", 0.05))
    base_seed = int(cfg.diagnostics.get("seed", 0)) + 404
    result: dict[str, Any] = {"modes": {}}
    all_rows: list[dict[str, Any]] = []

    for mode_name, disable_z_shortcut in (("decoder_as_is", False), ("decoder_without_z_shortcut", True)):
        logger.info("Experiment 4 [%s]: steps=%s lr=%s", mode_name, steps, lr)
        model = _instantiate_model(model_cfg, base_state, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()

        frozen = frozen_cpu.to(device)
        with torch.no_grad():
            T, d_in_pad, patch_mask, _structural_patch_mask, _dist_var_by_patch, dist_patch_by_patch, _dist_var_pooled = model._encode_distribution_context(
                frozen.x_s,
                x_mask=frozen.x_mask_s,
                d_in_mask=frozen.d_in_mask_s,
            )

        init_slots = model.latent_base.detach().unsqueeze(0).expand(int(frozen.meta["B"]), -1, -1).clone()
        H = torch.nn.Parameter(init_slots)
        optimizer = torch.optim.Adam([H], lr=lr, weight_decay=0.0)
        best_loss = math.inf
        best_payload: dict[str, torch.Tensor] = {}
        rows: list[dict[str, Any]] = []

        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            W_hat, z, _logvar, pred_dirs, direction_pre_norms = model._decode_from_latent_slots(
                H,
                dist_patch_by_patch=dist_patch_by_patch,
                patch_mask=patch_mask,
                d_in_mask=frozen.d_in_mask_s if frozen.d_in_mask_s is not None else torch.ones_like(frozen.W_s[:, :, 0], dtype=torch.bool),
                d_out_mask=torch.ones_like(frozen.W_s[:, 0, :], dtype=torch.bool),
                d_in=int(frozen.meta["d_in"]),
                d_out=int(frozen.meta["d_out"]),
                d_in_pad=d_in_pad,
                T=T,
                return_direction_pre_norms=True,
                disable_z_shortcut=disable_z_shortcut,
            )
            loss, _details = _exact_dir_loss(frozen, pred_dirs, W_hat=W_hat)
            row = _experiment_4_or_5_common_row(
                step,
                loss,
                frozen,
                pred_dirs,
                mode=mode_name,
                direction_pre_norms=direction_pre_norms,
            )
            row["slot_norm_mean"] = float(H.detach().norm(dim=-1).mean().item())
            row["slot_norm_max"] = float(H.detach().norm(dim=-1).max().item())
            row["latent_norm_mean"] = float(z.detach().view(int(frozen.meta["B"]), model.dec_L_latents, -1).norm(dim=-1).mean().item())
            row["latent_norm_max"] = float(z.detach().view(int(frozen.meta["B"]), model.dec_L_latents, -1).norm(dim=-1).max().item())

            if row["loss"] < best_loss:
                best_loss = row["loss"]
                best_payload = {
                    "H": H.detach().cpu().clone(),
                    "W_hat": W_hat.detach().cpu().clone(),
                    "z": z.detach().cpu().clone(),
                    "pred_dirs": pred_dirs.detach().cpu().clone(),
                }

            if step < steps:
                loss.backward()
                row["grad_norm"] = float(H.grad.detach().norm().item()) if H.grad is not None else 0.0
                optimizer.step()
            else:
                row["grad_norm"] = math.nan

            rows.append(row)

        summary = {
            "initial_loss": float(rows[0]["loss"]),
            "final_loss": float(rows[-1]["loss"]),
            "best_loss": float(min(float(row["loss"]) for row in rows)),
            "best_weighted_cos": float(max(float(row["weighted_cos"]) for row in rows)),
            "steps_to_threshold": _threshold_steps(rows),
            "final_slot_norm_mean": float(rows[-1]["slot_norm_mean"]),
            "final_latent_norm_mean": float(rows[-1]["latent_norm_mean"]),
        }
        result["modes"][mode_name] = summary
        all_rows.extend(rows)
        torch.save(best_payload, artifact_dir / f"best_bundle_{mode_name}.pt")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(artifact_dir / "trajectory.csv", _safe_scalar(all_rows))
    plot_note = _plot_grouped_lines(
        all_rows,
        x_key="step",
        y_key="loss",
        group_key="mode",
        path=artifact_dir / "loss_vs_step.png",
        title="Experiment 4: free latent state -> current decoder",
        ylabel="loss",
        ylog=True,
    )
    result["plot_note"] = plot_note
    _write_json(artifact_dir / "metrics.json", _safe_scalar(result))
    return result


def _collect_first_nonfinite(
    *,
    step: int,
    loss: torch.Tensor,
    model: BigWeightVAE,
    W_hat: torch.Tensor,
    pred_dirs: torch.Tensor,
    direction_pre_norms: torch.Tensor,
) -> dict[str, Any] | None:
    if torch.isfinite(loss).all() and torch.isfinite(W_hat).all() and torch.isfinite(pred_dirs).all() and torch.isfinite(direction_pre_norms).all():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return {
                    "step": int(step),
                    "kind": "non_finite_grad",
                    "parameter": name,
                    "grad_stats": _tensor_debug_stats_local(param.grad),
                }
        return None
    return {
        "step": int(step),
        "kind": "non_finite_forward",
        "loss": _safe_scalar(loss),
        "W_hat": _tensor_debug_stats_local(W_hat),
        "pred_dirs": _tensor_debug_stats_local(pred_dirs),
        "direction_pre_norms": _tensor_debug_stats_local(direction_pre_norms),
    }


def _experiment_5(
    frozen_cpu: FrozenTarget,
    cfg: DictConfig,
    artifact_dir: Path,
    *,
    model_cfg: Any,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    logger = _logger()
    exp_cfg = cfg.diagnostics.experiments.clean_e2e
    steps = max(1, int(exp_cfg.get("steps", 500)))
    lr = float(exp_cfg.get("lr", float(cfg.train.get("lr", 5e-4))))
    result: dict[str, Any] = {"modes": {}}
    all_rows: list[dict[str, Any]] = []

    for mode_name, disable_z_shortcut in (("model_as_is", False), ("model_without_z_shortcut", True)):
        logger.info("Experiment 5 [%s]: steps=%s lr=%s", mode_name, steps, lr)
        frozen = frozen_cpu.to(device)
        model = _instantiate_model(model_cfg, base_state, device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        rows: list[dict[str, Any]] = []
        best_loss = math.inf
        first_nonfinite: dict[str, Any] | None = None
        best_payload: dict[str, torch.Tensor] = {}

        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            W_hat, _z, _logvar, pred_dirs, direction_pre_norms = model(
                frozen.W_s,
                frozen.x_s,
                x_mask=frozen.x_mask_s,
                d_in_mask=frozen.d_in_mask_s,
                return_direction_pre_norms=True,
                disable_z_shortcut=disable_z_shortcut,
            )
            loss, _details = _exact_dir_loss(frozen, pred_dirs, W_hat=W_hat)
            row = _experiment_4_or_5_common_row(
                step,
                loss,
                frozen,
                pred_dirs,
                mode=mode_name,
                direction_pre_norms=direction_pre_norms,
            )

            if row["loss"] < best_loss:
                best_loss = row["loss"]
                best_payload = {
                    "W_hat": W_hat.detach().cpu().clone(),
                    "pred_dirs": pred_dirs.detach().cpu().clone(),
                    "direction_pre_norms": direction_pre_norms.detach().cpu().clone(),
                }

            if step < steps:
                loss.backward()
                grad_stats = _compute_grad_stats_local(model)
                row.update(
                    {
                        "grad_global_norm": float(grad_stats.get("grad/global_norm", 0.0)),
                        "grad_rms": float(grad_stats.get("grad/rms", 0.0)),
                        "grad_distribution_encoder_rms": float(grad_stats.get("grad/distribution_encoder_rms", 0.0)),
                        "grad_patch_tokenizer_rms": float(grad_stats.get("grad/patch_tokenizer_rms", 0.0)),
                        "grad_encoder_rms": float(grad_stats.get("grad/encoder_rms", 0.0)),
                        "grad_decoder_rms": float(grad_stats.get("grad/decoder_rms", 0.0)),
                        "grad_big_vae_other_rms": float(grad_stats.get("grad/big_vae_other_rms", 0.0)),
                    }
                )
                if first_nonfinite is None:
                    first_nonfinite = _collect_first_nonfinite(
                        step=step,
                        loss=loss,
                        model=model,
                        W_hat=W_hat,
                        pred_dirs=pred_dirs,
                        direction_pre_norms=direction_pre_norms,
                    )
                optimizer.step()
            else:
                row["grad_global_norm"] = math.nan
                row["grad_rms"] = math.nan

            rows.append(row)

        summary = {
            "initial_loss": float(rows[0]["loss"]),
            "final_loss": float(rows[-1]["loss"]),
            "best_loss": float(min(float(row["loss"]) for row in rows)),
            "best_weighted_cos": float(max(float(row["weighted_cos"]) for row in rows)),
            "steps_to_threshold": _threshold_steps(rows),
            "first_nonfinite": _safe_scalar(first_nonfinite),
        }
        result["modes"][mode_name] = summary
        all_rows.extend(rows)
        torch.save(best_payload, artifact_dir / f"best_bundle_{mode_name}.pt")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(artifact_dir / "trajectory.csv", _safe_scalar(all_rows))
    plot_note = _plot_grouped_lines(
        all_rows,
        x_key="step",
        y_key="loss",
        group_key="mode",
        path=artifact_dir / "loss_vs_step.png",
        title="Experiment 5: clean end-to-end AE",
        ylabel="loss",
        ylog=True,
    )
    result["plot_note"] = plot_note
    _write_json(artifact_dir / "metrics.json", _safe_scalar(result))
    return result


def _run_sanity_checks(
    frozen: FrozenTarget,
    artifact_root: Path,
) -> dict[str, Any]:
    scaled_pred_dirs = F.normalize(3.7 * frozen.u_target, dim=-1, eps=EPS)
    identity_loss, identity_details = _exact_dir_loss(frozen, frozen.u_target)
    scaled_loss, scaled_details = _exact_dir_loss(frozen, scaled_pred_dirs)
    neg_loss, neg_details = _exact_dir_loss(frozen, -frozen.u_target)

    checks = {
        "identity_sanity": {
            "loss": float(identity_loss.item()),
            "L_dir": float(identity_details["L_dir"].item()),
        },
        "positive_scaling_sanity": {
            "loss": float(scaled_loss.item()),
            "L_dir": float(scaled_details["L_dir"].item()),
        },
        "negation_sanity": {
            "loss": float(neg_loss.item()),
            "L_dir": float(neg_details["L_dir"].item()),
        },
        "shape_patch_accounting": {
            "T_forward": int(frozen.meta["T_forward"]),
            "T_loss": int(frozen.meta["T_loss"]),
            "used_rows_in_loss": int(frozen.meta["used_rows_in_loss"]),
            "pred_dirs_shape_forward": list(frozen.meta["pred_dirs_shape_forward"]),
            "target_patch_shape": list(frozen.meta["target_patch_shape"]),
        },
        "immutability_checks": frozen.meta.get("immutability_checks", []),
    }
    _write_json(artifact_root / "sanity_checks.json", _safe_scalar(checks))
    return checks


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "inconclusive"
    return "yes" if value else "no"


def _interpret_results(
    sanity: dict[str, Any],
    exp1: dict[str, Any],
    exp2: dict[str, Any],
    exp3: dict[str, Any],
    exp4: dict[str, Any],
    exp5: dict[str, Any],
) -> dict[str, Any]:
    exp1_best = min(float(payload["best_loss"]) for payload in exp1["optimizers"].values())
    exp3_current = float(exp3["modes"]["current_exact"]["best_loss"])
    exp3_alt_best = min(
        float(payload["best_loss"])
        for name, payload in exp3["modes"].items()
        if name != "current_exact"
    )
    exp4_as_is = float(exp4["modes"]["decoder_as_is"]["best_loss"])
    exp4_no_shortcut = float(exp4["modes"]["decoder_without_z_shortcut"]["best_loss"])
    exp5_as_is = float(exp5["modes"]["model_as_is"]["best_loss"])
    exp5_no_shortcut = float(exp5["modes"]["model_without_z_shortcut"]["best_loss"])

    loss_likely_culprit = None
    if exp1_best < 1e-4 and exp3_current < 1e-4 and float(sanity["identity_sanity"]["L_dir"]) < 1e-6:
        loss_likely_culprit = False
    elif exp1_best > 1e-2 or exp3_current > 1e-2:
        loss_likely_culprit = True

    encoder_bottleneck = None
    if exp4_as_is < 1e-2 and exp5_as_is > 5e-2:
        encoder_bottleneck = True
    elif exp4_as_is > 5e-2:
        encoder_bottleneck = False

    decoder_bottleneck = None
    if exp1_best < 1e-4 and exp3_alt_best < 1e-4 and exp4_as_is > 5e-2:
        decoder_bottleneck = True
    elif exp4_as_is < 1e-2:
        decoder_bottleneck = False

    z_shortcut_harmful = None
    if exp4_no_shortcut < 0.8 * exp4_as_is or exp5_no_shortcut < 0.8 * exp5_as_is:
        z_shortcut_harmful = True
    elif exp4_no_shortcut >= 0.95 * exp4_as_is and exp5_no_shortcut >= 0.95 * exp5_as_is:
        z_shortcut_harmful = False

    patch_accounting_mismatch = bool(
        int(sanity["shape_patch_accounting"]["T_forward"]) != int(sanity["shape_patch_accounting"]["T_loss"])
    )

    conclusions = {
        "loss_likely_culprit": loss_likely_culprit,
        "encoder_bottleneck": encoder_bottleneck,
        "decoder_bottleneck": decoder_bottleneck,
        "z_shortcut_harmful": z_shortcut_harmful,
        "patch_accounting_mismatch_evidence": patch_accounting_mismatch,
        "supporting_numbers": {
            "exp1_best_loss": exp1_best,
            "exp2_monotonic_nonincreasing": bool(exp2["monotonic_nonincreasing"]),
            "exp3_current_best_loss": exp3_current,
            "exp3_alt_best_loss": exp3_alt_best,
            "exp4_decoder_as_is_best_loss": exp4_as_is,
            "exp4_decoder_without_z_shortcut_best_loss": exp4_no_shortcut,
            "exp5_model_as_is_best_loss": exp5_as_is,
            "exp5_model_without_z_shortcut_best_loss": exp5_no_shortcut,
        },
    }
    return conclusions


def _write_summary_markdown(
    path: Path,
    *,
    checkpoint_path: str,
    frozen_meta: dict[str, Any],
    sanity: dict[str, Any],
    exp1: dict[str, Any],
    exp2: dict[str, Any],
    exp3: dict[str, Any],
    exp4: dict[str, Any],
    exp5: dict[str, Any],
    conclusions: dict[str, Any],
) -> None:
    lines = [
        "# Big VAE Identity Diagnostics",
        "",
        f"- checkpoint: `{checkpoint_path or '<random_init>'}`",
        f"- source_kind: `{frozen_meta['source_kind']}`",
        f"- frozen batch: `B={frozen_meta['B']}`, `d_in={frozen_meta['d_in']}`, `d_out={frozen_meta['d_out']}`, `patch_size={frozen_meta['patch_size']}`",
        f"- patch accounting: `T_forward={frozen_meta['T_forward']}`, `T_loss={frozen_meta['T_loss']}`, `used_rows_in_loss={frozen_meta['used_rows_in_loss']}`",
        "",
        "## Sanity Checks",
        "",
        f"- identity: `L_dir={sanity['identity_sanity']['L_dir']:.6e}`",
        f"- positive scaling: `L_dir={sanity['positive_scaling_sanity']['L_dir']:.6e}`",
        f"- negation: `L_dir={sanity['negation_sanity']['L_dir']:.6f}`",
        f"- frozen immutability: `{all(item['all_equal'] for item in sanity['immutability_checks'])}`",
        "",
        "## Experiment 1: loss-only free tensor harness",
        "",
        f"- SGD best/final: `{exp1['optimizers']['sgd']['best_loss']:.6e}` / `{exp1['optimizers']['sgd']['final_loss']:.6e}`",
        f"- Adam best/final: `{exp1['optimizers']['adam']['best_loss']:.6e}` / `{exp1['optimizers']['adam']['final_loss']:.6e}`",
        "",
        "## Experiment 2: interpolation geometry",
        "",
        f"- loss at lambda=0: `{exp2['loss_at_lambda_0']:.6f}`",
        f"- loss at lambda=1: `{exp2['loss_at_lambda_1']:.6e}`",
        f"- monotonic non-increasing: `{exp2['monotonic_nonincreasing']}`",
        "",
        "## Experiment 3: equivalent directional objectives",
        "",
        f"- current exact best: `{exp3['modes']['current_exact']['best_loss']:.6e}`",
        f"- normalized MSE best: `{exp3['modes']['normalized_mse']['best_loss']:.6e}`",
        f"- teacher-forced best: `{exp3['modes']['teacher_forced_reconstruction']['best_loss']:.6e}`",
        "",
        "## Experiment 4: free latent state -> current decoder",
        "",
        f"- decoder as-is best/final: `{exp4['modes']['decoder_as_is']['best_loss']:.6e}` / `{exp4['modes']['decoder_as_is']['final_loss']:.6e}`",
        f"- decoder without z_shortcut best/final: `{exp4['modes']['decoder_without_z_shortcut']['best_loss']:.6e}` / `{exp4['modes']['decoder_without_z_shortcut']['final_loss']:.6e}`",
        "",
        "## Experiment 5: clean end-to-end AE",
        "",
        f"- model as-is best/final: `{exp5['modes']['model_as_is']['best_loss']:.6e}` / `{exp5['modes']['model_as_is']['final_loss']:.6e}`",
        f"- model without z_shortcut best/final: `{exp5['modes']['model_without_z_shortcut']['best_loss']:.6e}` / `{exp5['modes']['model_without_z_shortcut']['final_loss']:.6e}`",
        "",
        "## Conclusions",
        "",
        f"- loss likely culprit: `{_bool_text(conclusions['loss_likely_culprit'])}`",
        f"- encoder bottleneck: `{_bool_text(conclusions['encoder_bottleneck'])}`",
        f"- decoder bottleneck: `{_bool_text(conclusions['decoder_bottleneck'])}`",
        f"- z_shortcut harmful: `{_bool_text(conclusions['z_shortcut_harmful'])}`",
        f"- evidence for patch accounting mismatch: `{_bool_text(conclusions['patch_accounting_mismatch_evidence'])}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(cfg: DictConfig) -> None:
    artifact_root = _artifact_root(cfg)
    log_dir = artifact_root / "logs"
    cfg.logging.dir = str(log_dir)
    cfg.logging.file_name = "diagnose_big_vae_identity.log"
    cfg.logging.file_path = str(log_dir / cfg.logging.file_name)
    setup_logging(cfg, rank=0)
    logger = _logger()
    (artifact_root / "experiment_1_loss_only").mkdir(parents=True, exist_ok=True)
    (artifact_root / "experiment_2_interpolation").mkdir(parents=True, exist_ok=True)
    (artifact_root / "experiment_3_alt_losses").mkdir(parents=True, exist_ok=True)
    (artifact_root / "experiment_4_free_latent_decoder").mkdir(parents=True, exist_ok=True)
    (artifact_root / "experiment_5_clean_e2e").mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg)
    _configure_reproducibility(int(cfg.diagnostics.get("seed", 0)), device)
    logger.info("Diagnostics device=%s distributed=%s amp=%s compile=%s", device, cfg.train.get("distributed"), cfg.train.get("amp"), cfg.train.get("compile"))

    (artifact_root / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    frozen_cpu, frozen_meta = _build_frozen_target(cfg, logger=logger, artifact_root=artifact_root)
    frozen_cpu.meta["immutability_checks"] = frozen_meta["immutability_checks"]
    sanity = _run_sanity_checks(frozen_cpu, artifact_root)

    model_cfg, base_state, checkpoint_path = _build_model_state(cfg, device, logger)
    exp1 = _experiment_1(frozen_cpu.to(device), cfg, artifact_root / "experiment_1_loss_only")
    exp2 = _experiment_2(frozen_cpu.to(device), cfg, artifact_root / "experiment_2_interpolation")
    exp3 = _experiment_3(frozen_cpu.to(device), cfg, artifact_root / "experiment_3_alt_losses")
    exp4 = _experiment_4(
        frozen_cpu,
        cfg,
        artifact_root / "experiment_4_free_latent_decoder",
        model_cfg=model_cfg,
        base_state=base_state,
        device=device,
    )
    exp5 = _experiment_5(
        frozen_cpu,
        cfg,
        artifact_root / "experiment_5_clean_e2e",
        model_cfg=model_cfg,
        base_state=base_state,
        device=device,
    )

    conclusions = _interpret_results(sanity, exp1, exp2, exp3, exp4, exp5)
    summary = {
        "artifact_root": str(artifact_root),
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "sanity_checks": sanity,
        "frozen_target_meta": frozen_meta,
        "experiment_1_loss_only": exp1,
        "experiment_2_interpolation": exp2,
        "experiment_3_alt_losses": exp3,
        "experiment_4_free_latent_decoder": exp4,
        "experiment_5_clean_e2e": exp5,
        "conclusions": conclusions,
    }
    _write_json(artifact_root / "summary.json", _safe_scalar(summary))
    _write_summary_markdown(
        artifact_root / "summary.md",
        checkpoint_path=checkpoint_path,
        frozen_meta=frozen_meta,
        sanity=sanity,
        exp1=exp1,
        exp2=exp2,
        exp3=exp3,
        exp4=exp4,
        exp5=exp5,
        conclusions=conclusions,
    )
    logger.info("Diagnostics completed. Report: %s", artifact_root / "summary.md")


def main(argv: list[str] | None = None) -> None:
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = _compose_default_cfg()
    cfg = _apply_cli_overrides(cfg, overrides)
    _run(cfg)


if __name__ == "__main__":
    main()
