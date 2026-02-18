from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.weight_quantile_vae import DistributionConfig, InputDistributionEncodingModule, MiniPatchVAE, MiniVAEConfig


@dataclass(slots=True)
class PretrainConfig:
    # Training loop
    steps: int = 20000
    log_every: int = 100
    val_every_epochs: int = 1
    val_split: float = 0.1
    val_enabled: bool = False
    val_random_z_probe: bool = True
    ablation_steps: int = 0
    ablation_zero_dist_conditioning: bool = False
    ablation_deterministic_z: bool = True
    grad_monitor_enabled: bool = True
    grad_monitor_every: int = 10
    grad_monitor_weights_only: bool = True
    grad_monitor_topk_layers: int = 100
    grad_monitor_log_scale: bool = True
    grad_live_csv: bool = True
    grad_live_plot_every: int = 200
    grad_plot_path: str = "./checkpoints/mini_patch_vae_grad_rms.png"
    grad_csv_path: str = "./checkpoints/mini_patch_vae_grad_rms.csv"
    xavier_init_enabled: bool = True
    seed: int = 42
    batch_size: int = 64
    lr: float = 3e-4
    cosine_scheduler_enabled: bool = True
    cosine_warmup_steps: int = 500
    cosine_min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    recon_y_weight: float = 0.0
    recon_w_weight: float = 1.0
    beta: float = 0.0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_path: str = "./checkpoints/mini_patch_vae.pt"
    print_model_summary: bool = False

    # Fixed synthetic dataset
    dataset_path: str = "./checkpoints/mini_patch_dataset.pt"
    dataset_num_samples: int = 2048
    regenerate_dataset: bool = True
    shuffle_dataset_each_epoch: bool = True
    dataset_seed: int = 123
    dataset_only: bool = False

    # Synthetic tensor shapes
    n: int = 1024
    d_in: int = 512
    d_out: int = 128
    patch_size: int = 64

    # Synthetic input distribution (mixture of Gaussians)
    mixture_components: int = 4
    mixture_mean_scale: float = 1.0
    mixture_std_min: float = 0.15
    mixture_std_max: float = 1.0
    mixture_weight_eps: float = 1e-8

    # Synthetic weight distribution
    weight_mean: float = 0.0
    weight_std: float = 1.0
    weight_latent_dim: int = 4
    weight_projection_seed: int = 2027
    weight_projector_from_input_distribution: bool = True
    weight_projector_input_noise_std: float = 0.0

    # InputDistributionEncodingModule config
    dist_k_s: int = 32
    dist_Kq: int = 8
    dist_d_var: int = 128
    dist_d_dist: int = 128
    dist_num_var_attn_layers: int = 2
    dist_var_attn_heads: int = 8
    dist_dcn_num_cross_layers: int = 3
    dist_dcn_deep_hidden: int = 64
    dist_dcn_deep_layers: int = 2
    dist_dropout: float = 0.05

    # MiniPatchVAE config
    mini_z_dim: int = 16
    mini_d_e: int = 64
    mini_pos_dim: int = 16
    mini_num_attn_layers_encoder: int = 2
    mini_num_layers_decoder: int = 2
    mini_n_heads: int = 4
    mini_d_patch: int = 16
    mini_dropout: float = 0.05



def sample_random_patches(W: torch.Tensor, X: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one (output, patch) per batch item.

    Inputs:
    - W: [B, d_in, d_out]
    - X: [B, n, d_in]

    Outputs:
    - patch_idx: [B, p]
    - w_patch: [B, p]
    - X_I: [B, n, p]
    - out_idx: [B]
    """
    B, d_in, d_out = W.shape
    _, n, d_in_x = X.shape
    if d_in_x != d_in:
        raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

    p = patch_size
    T = (d_in + p - 1) // p

    # Random output-column index per sample.
    # out_idx: [B]
    out_idx = torch.randint(0, d_out, (B,), device=W.device)

    # Random patch index per sample.
    # patch_t: [B]
    patch_t = torch.randint(0, T, (B,), device=W.device)

    base = patch_t.unsqueeze(-1) * p + torch.arange(p, device=W.device).unsqueeze(0)  # [B, p]
    patch_idx = base.clamp(max=d_in - 1)  # [B, p]

    # Gather selected output column weights: W_col [B, d_in]
    W_col = W[torch.arange(B, device=W.device), :, out_idx]

    # w_patch: [B, p]
    w_patch = W_col.gather(dim=1, index=patch_idx)

    # X_I: [B, n, p]
    X_I = X.gather(dim=2, index=patch_idx.unsqueeze(1).expand(-1, n, -1))

    return patch_idx, w_patch, X_I, out_idx



def sample_mixture_inputs(
    batch_size: int,
    n: int,
    d_in: int,
    device: torch.device,
    n_components: int,
    mean_scale: float,
    std_min: float,
    std_max: float,
    weight_eps: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample X from a per-feature Gaussian mixture.

    Output:
    - X: [B, n, d_in]
    """
    if n_components < 2:
        raise ValueError(f"n_components must be >= 2, got {n_components}")
    if std_min <= 0.0 or std_max <= 0.0 or std_max < std_min:
        raise ValueError(f"Invalid std range: std_min={std_min}, std_max={std_max}")

    # Mixture params per (batch item, input feature, component).
    # means/stds/weights: [B, d_in, K]
    means = mean_scale * torch.randn(batch_size, d_in, n_components, device=device, generator=generator)
    stds = torch.empty(batch_size, d_in, n_components, device=device).uniform_(std_min, std_max, generator=generator)
    weights = torch.rand(batch_size, d_in, n_components, device=device, generator=generator)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(weight_eps)

    # Component ids sampled independently per (batch item, feature, sample row).
    # comp_idx: [B, d_in, n]
    comp_idx = torch.multinomial(
        weights.view(batch_size * d_in, n_components),
        num_samples=n,
        replacement=True,
        generator=generator,
    ).view(batch_size, d_in, n)

    # Select means/stds for sampled components.
    # mean_sel/std_sel: [B, n, d_in]
    mean_sel = means.gather(dim=2, index=comp_idx).permute(0, 2, 1).contiguous()
    std_sel = stds.gather(dim=2, index=comp_idx).permute(0, 2, 1).contiguous()

    eps = torch.randn(batch_size, n, d_in, device=device, generator=generator)
    return mean_sel + std_sel * eps


def sample_projected_weights(
    batch_size: int,
    d_in: int,
    d_out: int,
    latent_dim: int,
    device: torch.device,
    latent_generator: torch.Generator,
    projection_generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample W via per-matrix random projection:
    z_low ~ N(0, I) in latent_dim, W_col = P_b z_low, where each sample b has its own
    projection matrix P_b shared across that sample's output columns.

    Output:
    - W: [B, d_in, d_out]
    """
    if latent_dim <= 0:
        raise ValueError(f"weight_latent_dim must be > 0, got {latent_dim}")
    if latent_dim > d_in:
        raise ValueError(f"weight_latent_dim ({latent_dim}) must be <= d_in ({d_in})")

    # One projection matrix per sample.
    # Scale keeps projected variance O(1) before applying weight_std.
    proj = torch.randn(batch_size, latent_dim, d_in, device=device, generator=projection_generator)
    proj = proj / float(latent_dim) ** 0.5

    # Independent low-d latent code per (sample, output column).
    z_low = torch.randn(batch_size, d_out, latent_dim, device=device, generator=latent_generator)
    w_by_out = torch.einsum("bol,bld->bod", z_low, proj)  # [B, d_out, d_in]
    return w_by_out.transpose(1, 2).contiguous()  # [B, d_in, d_out]


def sample_input_conditioned_projected_weights(
    X: torch.Tensor,
    d_out: int,
    latent_dim: int,
    latent_generator: torch.Generator,
    projection_generator: torch.Generator,
    projector_noise_std: float = 0.0,
) -> torch.Tensor:
    """
    Sample W via input-conditioned per-matrix projection:
    1) Compute per-feature distribution stats from X.
    2) Build P_b deterministically from local stats of each coordinate d.
       No global normalization over d_in is used.
    3) Sample low-d z_low and project with P_b.

    Output:
    - W: [B, d_in, d_out]
    """
    if X.ndim != 3:
        raise ValueError(f"X must be [B, n, d_in], got {tuple(X.shape)}")
    if latent_dim <= 0:
        raise ValueError(f"weight_latent_dim must be > 0, got {latent_dim}")
    if projector_noise_std < 0.0:
        raise ValueError(f"projector_noise_std must be >= 0, got {projector_noise_std}")

    B, _, d_in = X.shape
    if latent_dim > d_in:
        raise ValueError(f"weight_latent_dim ({latent_dim}) must be <= d_in ({d_in})")

    # Per-feature distribution summaries from inputs.
    x_mean = X.mean(dim=1)  # [B, d_in]
    x_std = X.std(dim=1, unbiased=False).clamp_min(1e-6)  # [B, d_in]
    x_abs = X.abs().mean(dim=1)  # [B, d_in]
    x_rms = torch.sqrt(torch.mean(X * X, dim=1).clamp_min(1e-12))  # [B, d_in]

    # stats: [B, d_in, 4]. Each coordinate d is mapped independently,
    # so projector entries for d depend only on local statistics of d.
    stats = torch.stack([x_mean, torch.log(x_std), x_abs, x_rms], dim=-1)
    stats = torch.tanh(stats)

    # Shared local mapping from 4 local stats -> latent_dim projector channels.
    # No mixing across coordinates d_in.
    mixing = torch.randn(latent_dim, 4, device=X.device, generator=projection_generator) / (4.0 ** 0.5)
    bias = torch.randn(latent_dim, device=X.device, generator=projection_generator) / (latent_dim ** 0.5)
    proj = torch.einsum("lc,bdc->bld", mixing, stats) + bias.view(1, latent_dim, 1)  # [B, latent_dim, d_in]
    proj = torch.tanh(proj)
    if projector_noise_std > 0.0:
        proj = proj + projector_noise_std * torch.randn(
            proj.shape,
            device=X.device,
            generator=projection_generator,
        )

    # Keep global scale predictable without normalizing over d_in coordinates.
    proj = proj / float(latent_dim) ** 0.5

    z_low = torch.randn(B, d_out, latent_dim, device=X.device, generator=latent_generator)
    w_by_out = torch.einsum("bol,bld->bod", z_low, proj)  # [B, d_out, d_in]
    return w_by_out.transpose(1, 2).contiguous()  # [B, d_in, d_out]


def _meta_value_equal(got: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(got) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return got == expected


def _build_or_load_dataset(cfg: PretrainConfig) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Returns:
    - X_all: [N_ds, n, d_in] on CPU
    - W_all: [N_ds, d_in, d_out] on CPU
    - generated_now: bool
    """
    dataset_path = Path(cfg.dataset_path)

    expected_shape_x = (cfg.dataset_num_samples, cfg.n, cfg.d_in)
    expected_shape_w = (cfg.dataset_num_samples, cfg.d_in, cfg.d_out)
    weight_generation_mode = (
        "input_conditioned_local_feature_projection_v2"
        if cfg.weight_projector_from_input_distribution
        else "per_matrix_random_projection"
    )
    expected_meta = {
        "dataset_num_samples": cfg.dataset_num_samples,
        "n": cfg.n,
        "d_in": cfg.d_in,
        "d_out": cfg.d_out,
        "patch_size": cfg.patch_size,
        "dataset_seed": cfg.dataset_seed,
        "mixture_components": cfg.mixture_components,
        "mixture_mean_scale": cfg.mixture_mean_scale,
        "mixture_std_min": cfg.mixture_std_min,
        "mixture_std_max": cfg.mixture_std_max,
        "mixture_weight_eps": cfg.mixture_weight_eps,
        "weight_generation": weight_generation_mode,
        "weight_mean": cfg.weight_mean,
        "weight_std": cfg.weight_std,
        "weight_latent_dim": cfg.weight_latent_dim,
        "weight_projection_seed": cfg.weight_projection_seed,
        "weight_projector_from_input_distribution": cfg.weight_projector_from_input_distribution,
        "weight_projector_input_noise_std": cfg.weight_projector_input_noise_std,
    }

    if dataset_path.exists() and not cfg.regenerate_dataset:
        payload = torch.load(dataset_path, map_location="cpu")
        if not isinstance(payload, dict) or "X" not in payload or "W" not in payload:
            raise ValueError(
                f"Invalid dataset format in {dataset_path}. "
                "Set regenerate_dataset=True to recreate."
            )
        X_all = payload["X"]
        W_all = payload["W"]
        if tuple(X_all.shape) != expected_shape_x or tuple(W_all.shape) != expected_shape_w:
            raise ValueError(
                f"Dataset shape mismatch in {dataset_path}: "
                f"got X={tuple(X_all.shape)} W={tuple(W_all.shape)}, "
                f"expected X={expected_shape_x} W={expected_shape_w}. "
                "Set regenerate_dataset=True to recreate."
            )
        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(
                f"Dataset meta format mismatch in {dataset_path}. "
                "Set regenerate_dataset=True to recreate."
            )
        mismatches: list[str] = []
        for key, expected in expected_meta.items():
            if key not in meta:
                mismatches.append(f"{key}: missing (expected={expected})")
                continue
            got = meta[key]
            if not _meta_value_equal(got=got, expected=expected):
                mismatches.append(f"{key}: got={got} expected={expected}")
        if mismatches:
            preview = "; ".join(mismatches[:4])
            raise ValueError(
                f"Dataset meta mismatch in {dataset_path}: {preview}. "
                "Set regenerate_dataset=True to recreate."
            )
        return X_all.contiguous(), W_all.contiguous(), False

    if cfg.dataset_num_samples <= 0:
        raise ValueError(f"dataset_num_samples must be > 0, got {cfg.dataset_num_samples}")

    cpu = torch.device("cpu")
    latent_gen = torch.Generator(device="cpu")
    latent_gen.manual_seed(int(cfg.dataset_seed))
    proj_gen = torch.Generator(device="cpu")
    proj_gen.manual_seed(int(cfg.weight_projection_seed))

    X_all = sample_mixture_inputs(
        batch_size=cfg.dataset_num_samples,
        n=cfg.n,
        d_in=cfg.d_in,
        device=cpu,
        n_components=cfg.mixture_components,
        mean_scale=cfg.mixture_mean_scale,
        std_min=cfg.mixture_std_min,
        std_max=cfg.mixture_std_max,
        weight_eps=cfg.mixture_weight_eps,
        generator=latent_gen,
    )
    if cfg.weight_projector_from_input_distribution:
        W_proj = sample_input_conditioned_projected_weights(
            X=X_all,
            d_out=cfg.d_out,
            latent_dim=cfg.weight_latent_dim,
            latent_generator=latent_gen,
            projection_generator=proj_gen,
            projector_noise_std=cfg.weight_projector_input_noise_std,
        )
        weight_generation_mode = "input_conditioned_local_feature_projection_v2"
    else:
        W_proj = sample_projected_weights(
            batch_size=cfg.dataset_num_samples,
            d_in=cfg.d_in,
            d_out=cfg.d_out,
            latent_dim=cfg.weight_latent_dim,
            device=cpu,
            latent_generator=latent_gen,
            projection_generator=proj_gen,
        )
        weight_generation_mode = "per_matrix_random_projection"
    W_all = cfg.weight_mean + cfg.weight_std * W_proj

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X": X_all.contiguous(),
            "W": W_all.contiguous(),
            "meta": {
                "dataset_num_samples": cfg.dataset_num_samples,
                "n": cfg.n,
                "d_in": cfg.d_in,
                "d_out": cfg.d_out,
                "patch_size": cfg.patch_size,
                "dataset_seed": cfg.dataset_seed,
                "mixture_components": cfg.mixture_components,
                "mixture_mean_scale": cfg.mixture_mean_scale,
                "mixture_std_min": cfg.mixture_std_min,
                "mixture_std_max": cfg.mixture_std_max,
                "mixture_weight_eps": cfg.mixture_weight_eps,
                "weight_generation": weight_generation_mode,
                "weight_mean": cfg.weight_mean,
                "weight_std": cfg.weight_std,
                "weight_latent_dim": cfg.weight_latent_dim,
                "weight_projection_seed": cfg.weight_projection_seed,
                "weight_projector_from_input_distribution": cfg.weight_projector_from_input_distribution,
                "weight_projector_input_noise_std": cfg.weight_projector_input_noise_std,
            },
        },
        dataset_path,
    )
    return X_all.contiguous(), W_all.contiguous(), True


def _forward_patch_objective(
    cfg: PretrainConfig,
    dist_encoder: InputDistributionEncodingModule,
    mini_vae: MiniPatchVAE,
    W: torch.Tensor,
    X: torch.Tensor,
    randomize_latent: bool = False,
    zero_dist_conditioning: bool = False,
    deterministic_z: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_idx, w_patch, X_I, _ = sample_random_patches(W=W, X=X, patch_size=cfg.patch_size)

    # dist_var_tokens: [B, p, d_var]
    # dist_patch_embed: [B, d_dist] (unused in this minimal mini-VAE loop)
    dist_var_tokens, _ = dist_encoder(X=X, patch_idx=patch_idx)
    if zero_dist_conditioning:
        dist_var_tokens = torch.zeros_like(dist_var_tokens)

    if randomize_latent:
        # Probe latent dependence: decode with random z while keeping conditioning fixed.
        mu, logvar = mini_vae.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        z = torch.randn_like(mu)
        w_hat = mini_vae.decode(z=z, patch_size=w_patch.shape[1])
    elif deterministic_z:
        # Deterministic VAE pass: use z=mu without sampling noise.
        mu, logvar = mini_vae.encode(w_patch=w_patch, dist_var_tokens=dist_var_tokens)
        w_hat = mini_vae.decode(z=mu, patch_size=w_patch.shape[1])
    else:
        # w_hat: [B, p], mu/logvar: [B, z_dim]
        w_hat, mu, logvar, _ = mini_vae(w_patch=w_patch, dist_var_tokens=dist_var_tokens)

    # Functional patch-local objective.
    # y/y_hat: [B, n]
    y = torch.einsum("bnp,bp->bn", X_I, w_patch)
    y_hat = torch.einsum("bnp,bp->bn", X_I, w_hat)

    recon_y = F.mse_loss(y_hat, y, reduction='sum')
    recon_w = F.mse_loss(w_hat, w_patch, reduction='sum')
    kl = mini_vae.kl_loss(mu=mu, logvar=logvar)
    loss = cfg.recon_y_weight * recon_y + cfg.recon_w_weight * recon_w + cfg.beta * kl
    return loss, recon_y, recon_w, kl, w_patch, w_hat


@torch.no_grad()
def _validate_on_holdout(
    cfg: PretrainConfig,
    dist_encoder: InputDistributionEncodingModule,
    mini_vae: MiniPatchVAE,
    X_all: torch.Tensor,
    W_all: torch.Tensor,
    val_idx: torch.Tensor,
    batch_size: int,
    device: torch.device,
    randomize_latent: bool = False,
    zero_dist_conditioning: bool = False,
    deterministic_z: bool = False,
) -> tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
    if val_idx.numel() == 0:
        raise ValueError("Validation requested with empty holdout split")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 for validation, got {batch_size}")

    was_training_dist = dist_encoder.training
    was_training_mini = mini_vae.training
    dist_encoder.eval()
    mini_vae.eval()

    total = 0
    total_loss = 0.0
    total_recon_y = 0.0
    total_recon_w = 0.0
    total_kl = 0.0
    patch_true_sample: torch.Tensor | None = None
    patch_pred_sample: torch.Tensor | None = None
    try:
        for start in range(0, int(val_idx.numel()), batch_size):
            batch_idx = val_idx[start : start + batch_size]
            W = W_all.index_select(0, batch_idx).to(device=device, non_blocking=True)
            X = X_all.index_select(0, batch_idx).to(device=device, non_blocking=True)

            loss, recon_y, recon_w, kl, w_patch, w_hat = _forward_patch_objective(
                cfg=cfg,
                dist_encoder=dist_encoder,
                mini_vae=mini_vae,
                W=W,
                X=X,
                randomize_latent=randomize_latent,
                zero_dist_conditioning=zero_dist_conditioning,
                deterministic_z=deterministic_z,
            )
            bsz = int(batch_idx.numel())
            total += bsz
            total_loss += float(loss.item()) * bsz
            total_recon_y += float(recon_y.item()) * bsz
            total_recon_w += float(recon_w.item()) * bsz
            total_kl += float(kl.item()) * bsz
            if patch_true_sample is None or patch_pred_sample is None:
                patch_true_sample = w_patch[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
                patch_pred_sample = w_hat[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
    finally:
        dist_encoder.train(was_training_dist)
        mini_vae.train(was_training_mini)

    if patch_true_sample is None or patch_pred_sample is None:
        raise RuntimeError("Validation produced no patch sample to print")

    denom = max(1, total)
    return (
        total_loss / denom,
        total_recon_y / denom,
        total_recon_w / denom,
        total_kl / denom,
        patch_true_sample,
        patch_pred_sample,
    )


def _format_patch_for_log(patch: torch.Tensor) -> str:
    values = patch.detach().to(device="cpu", dtype=torch.float32).tolist()
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def _layer_key_from_param_name(name: str) -> str:
    parts = name.split(".")
    if not parts:
        return name
    last = parts[-1]
    if last in {"weight", "bias"} or last.endswith("_weight") or last.endswith("_bias"):
        if len(parts) > 1:
            return ".".join(parts[:-1])
    return name


def _collect_grad_rms_per_layer(
    named_params: list[tuple[str, torch.nn.Parameter]],
    *,
    weights_only: bool,
) -> dict[str, float]:
    sumsq_by_layer: dict[str, float] = {}
    count_by_layer: dict[str, int] = {}

    global_sumsq = 0.0
    global_count = 0

    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        if weights_only and param.ndim < 2:
            continue

        g = grad.detach().to(dtype=torch.float32)
        if g.numel() == 0:
            continue

        sumsq = float(torch.sum(g * g).item())
        count = int(g.numel())
        layer = _layer_key_from_param_name(name)

        sumsq_by_layer[layer] = sumsq_by_layer.get(layer, 0.0) + sumsq
        count_by_layer[layer] = count_by_layer.get(layer, 0) + count

        global_sumsq += sumsq
        global_count += count

    out: dict[str, float] = {}
    for layer, sumsq in sumsq_by_layer.items():
        count = max(1, count_by_layer[layer])
        out[layer] = math.sqrt(sumsq / float(count))

    if global_count > 0:
        out["__global__"] = math.sqrt(global_sumsq / float(global_count))

    return out


def _renormalize_gradients(
    params: list[torch.nn.Parameter],
    target_norm: float,
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Renormalize global gradient norm to `target_norm` (not clipping).

    Returns:
    - total_norm_before: global L2 norm before scaling
    - applied_scale: multiplicative factor applied to all grads
    """
    if target_norm <= 0.0:
        return 0.0, 1.0

    total_sumsq = 0.0
    has_grad = False
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        has_grad = True
        g = grad.detach().to(dtype=torch.float32)
        total_sumsq += float(torch.sum(g * g).item())

    if (not has_grad) or total_sumsq <= 0.0:
        return 0.0, 1.0

    total_norm = math.sqrt(total_sumsq)
    scale = float(target_norm) / max(total_norm, eps)

    for param in params:
        if param.grad is not None:
            param.grad.mul_(scale)

    return total_norm, scale


def _save_grad_rms_csv(history: dict[str, list[tuple[int, float]]], save_path: Path) -> None:
    lines = ["step,layer,grad_rms_per_weight"]
    for layer, points in history.items():
        for step, value in points:
            lines.append(f"{step},{layer},{value:.12e}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_grad_rms_csv(step: int, stats: dict[str, float], save_path: Path) -> None:
    if not stats:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("a", encoding="utf-8") as f:
        for layer, value in stats.items():
            f.write(f"{step},{layer},{float(value):.12e}\n")


def _save_grad_rms_plot(
    history: dict[str, list[tuple[int, float]]],
    save_path: Path,
    *,
    topk_layers: int,
    log_scale: bool,
) -> bool:
    if not history:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"grad_plot skipped: matplotlib import failed: {exc}")
        return False

    layers = [k for k in history.keys() if k != "__global__"]
    if layers:
        layers.sort(
            key=lambda key: sum(v for _, v in history[key]) / max(1, len(history[key])),
            reverse=True,
        )
        layers = layers[: max(1, int(topk_layers))]

    plt.figure(figsize=(12, 6))
    if "__global__" in history:
        steps = [s for s, _ in history["__global__"]]
        vals = [v for _, v in history["__global__"]]
        plt.plot(steps, vals, label="__global__", linewidth=2.5, color="black")

    for layer in layers:
        steps = [s for s, _ in history[layer]]
        vals = [v for _, v in history[layer]]
        plt.plot(steps, vals, label=layer, linewidth=1.2, alpha=0.85)

    plt.title("Gradient RMS Per Weight By Layer")
    plt.xlabel("Step")
    plt.ylabel("Grad RMS Per Weight")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.2)
    if layers or "__global__" in history:
        plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140)
    plt.close()
    return True


def _apply_xavier_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)


def _build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        progress = (step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_pretrain_stub(cfg: PretrainConfig) -> None:
    device = torch.device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    dist_cfg = DistributionConfig(
        k_s=cfg.dist_k_s,
        Kq=cfg.dist_Kq,
        d_var=cfg.dist_d_var,
        d_dist=cfg.dist_d_dist,
        num_var_attn_layers=cfg.dist_num_var_attn_layers,
        var_attn_heads=cfg.dist_var_attn_heads,
        dcn_num_cross_layers=cfg.dist_dcn_num_cross_layers,
        dcn_deep_hidden=cfg.dist_dcn_deep_hidden,
        dcn_deep_layers=cfg.dist_dcn_deep_layers,
        dropout=cfg.dist_dropout,
    )
    dist_encoder = InputDistributionEncodingModule(dist_cfg).to(device)

    mini_cfg = MiniVAEConfig(
        z_dim=cfg.mini_z_dim,
        d_e=cfg.mini_d_e,
        pos_dim=cfg.mini_pos_dim,
        num_attn_layers_encoder=cfg.mini_num_attn_layers_encoder,
        num_layers_decoder=cfg.mini_num_layers_decoder,
        n_heads=cfg.mini_n_heads,
        d_patch=cfg.mini_d_patch,
        dropout=cfg.mini_dropout,
    )
    mini_vae = MiniPatchVAE(d_var=dist_cfg.d_var, cfg=mini_cfg).to(device)
    if cfg.xavier_init_enabled:
        _apply_xavier_init(dist_encoder)
        _apply_xavier_init(mini_vae)
    if cfg.print_model_summary:
        print(mini_vae)

    params = list(dist_encoder.parameters()) + list(mini_vae.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    if cfg.recon_y_weight < 0.0 or cfg.recon_w_weight < 0.0:
        raise ValueError(
            f"recon_y_weight/recon_w_weight must be >= 0, got "
            f"{cfg.recon_y_weight} and {cfg.recon_w_weight}"
        )
    if cfg.recon_y_weight == 0.0 and cfg.recon_w_weight == 0.0:
        raise ValueError("At least one of recon_y_weight/recon_w_weight must be > 0")
    if cfg.ablation_steps < 0:
        raise ValueError(f"ablation_steps must be >= 0, got {cfg.ablation_steps}")
    if cfg.grad_clip_norm < 0.0:
        raise ValueError(f"grad_clip_norm must be >= 0, got {cfg.grad_clip_norm}")
    if cfg.cosine_warmup_steps < 0:
        raise ValueError(f"cosine_warmup_steps must be >= 0, got {cfg.cosine_warmup_steps}")
    if not (0.0 <= cfg.cosine_min_lr_ratio <= 1.0):
        raise ValueError(f"cosine_min_lr_ratio must be in [0, 1], got {cfg.cosine_min_lr_ratio}")
    if cfg.grad_monitor_every <= 0:
        raise ValueError(f"grad_monitor_every must be > 0, got {cfg.grad_monitor_every}")
    if cfg.grad_live_plot_every < 0:
        raise ValueError(f"grad_live_plot_every must be >= 0, got {cfg.grad_live_plot_every}")

    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if cfg.cosine_scheduler_enabled:
        scheduler = _build_cosine_scheduler(
            optimizer=optimizer,
            total_steps=cfg.steps,
            warmup_steps=cfg.cosine_warmup_steps,
            min_lr_ratio=cfg.cosine_min_lr_ratio,
        )

    log_every = max(1, int(cfg.log_every))
    val_every_epochs = max(1, int(cfg.val_every_epochs))
    X_all, W_all, generated_now = _build_or_load_dataset(cfg)
    dataset_size = int(X_all.shape[0])
    if dataset_size < 2:
        raise ValueError(
            f"dataset_num_samples must be >= 2 to build a holdout split, got {dataset_size}"
        )

    val_idx = torch.empty(0, dtype=torch.long)
    if cfg.val_enabled:
        if not (0.0 < float(cfg.val_split) < 1.0):
            raise ValueError(f"val_split must be in (0, 1), got {cfg.val_split}")
        split_gen = torch.Generator(device="cpu")
        split_gen.manual_seed(int(cfg.seed) + 777)
        perm = torch.randperm(dataset_size, generator=split_gen)
        val_size = int(round(dataset_size * float(cfg.val_split)))
        val_size = min(max(1, val_size), dataset_size - 1)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
    else:
        train_idx = torch.arange(dataset_size, dtype=torch.long)

    train_size = int(train_idx.numel())
    val_size = int(val_idx.numel())
    if train_size <= 0:
        raise ValueError("Train split is empty; increase dataset_num_samples or reduce val_split")

    train_batch_size = min(int(cfg.batch_size), train_size)
    val_batch_size = min(int(cfg.batch_size), val_size) if val_size > 0 else 0
    steps_per_epoch = (train_size + train_batch_size - 1) // train_batch_size

    print(
        f"dataset: path={cfg.dataset_path} size={dataset_size} train={train_size} val={val_size} "
        f"generated_now={generated_now} shuffle_each_epoch={cfg.shuffle_dataset_each_epoch} "
        f"train_batch={train_batch_size} steps_per_epoch={steps_per_epoch}"
    )
    print(
        "ablation: "
        f"steps={cfg.ablation_steps} "
        f"zero_dist_conditioning={cfg.ablation_zero_dist_conditioning} "
        f"deterministic_z={cfg.ablation_deterministic_z}"
    )
    print(f"optimizer: grad_renorm_norm={cfg.grad_clip_norm}")
    print(
        "scheduler: "
        f"cosine_enabled={cfg.cosine_scheduler_enabled} "
        f"warmup_steps={cfg.cosine_warmup_steps} "
        f"min_lr_ratio={cfg.cosine_min_lr_ratio}"
    )
    print(f"init: xavier_init_enabled={cfg.xavier_init_enabled}")
    if cfg.grad_monitor_enabled:
        print(
            "grad_monitor: "
            f"every={cfg.grad_monitor_every} "
            f"weights_only={cfg.grad_monitor_weights_only} "
            f"topk_layers={cfg.grad_monitor_topk_layers} "
            f"log_scale={cfg.grad_monitor_log_scale}"
        )
    if cfg.dataset_only:
        print("dataset_only=True: dataset prepared, training skipped")
        return

    order = torch.arange(train_size, dtype=torch.long)
    order_gen = torch.Generator(device="cpu")
    order_gen.manual_seed(int(cfg.seed) + 1)
    cursor = 0
    epoch = 0
    named_params: list[tuple[str, torch.nn.Parameter]] = [
        *[(f"dist_encoder.{name}", param) for name, param in dist_encoder.named_parameters()],
        *[(f"mini_vae.{name}", param) for name, param in mini_vae.named_parameters()],
    ]
    grad_history: dict[str, list[tuple[int, float]]] = {}
    latest_grad_global: float | None = None
    grad_csv_path = Path(cfg.grad_csv_path)
    grad_plot_path = Path(cfg.grad_plot_path)
    if cfg.grad_monitor_enabled and cfg.grad_live_csv:
        grad_csv_path.parent.mkdir(parents=True, exist_ok=True)
        grad_csv_path.write_text("step,layer,grad_rms_per_weight\n", encoding="utf-8")

    for step in range(1, cfg.steps + 1):
        # Build batch indices from train split.
        if cursor == 0:
            if cfg.shuffle_dataset_each_epoch:
                order = torch.randperm(train_size, generator=order_gen)
            else:
                order = torch.arange(train_size, dtype=torch.long)

        end = min(cursor + train_batch_size, train_size)
        batch_local_idx = order[cursor:end]
        cursor = end
        epoch_completed = False
        if cursor >= train_size:
            cursor = 0
            epoch += 1
            epoch_completed = True
        batch_idx = train_idx.index_select(0, batch_local_idx)

        # W: [B, d_in, d_out], X: [B, n, d_in]
        W = W_all.index_select(0, batch_idx).to(device=device, non_blocking=True)
        X = X_all.index_select(0, batch_idx).to(device=device, non_blocking=True)

        ablation_active = step <= int(cfg.ablation_steps)
        zero_dist = bool(cfg.ablation_zero_dist_conditioning and ablation_active)
        deterministic_z = bool(cfg.ablation_deterministic_z and ablation_active)

        loss, recon_y, recon_w, kl, _, _ = _forward_patch_objective(
            cfg=cfg,
            dist_encoder=dist_encoder,
            mini_vae=mini_vae,
            W=W,
            X=X,
            zero_dist_conditioning=zero_dist,
            deterministic_z=deterministic_z,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_monitor_enabled and (step % int(cfg.grad_monitor_every) == 0 or step == 1):
            grad_stats = _collect_grad_rms_per_layer(
                named_params=named_params,
                weights_only=bool(cfg.grad_monitor_weights_only),
            )
            for layer, value in grad_stats.items():
                grad_history.setdefault(layer, []).append((step, float(value)))
            latest_grad_global = grad_stats.get("__global__", latest_grad_global)
            if cfg.grad_live_csv:
                _append_grad_rms_csv(step=step, stats=grad_stats, save_path=grad_csv_path)
            if cfg.grad_live_plot_every > 0 and (step % int(cfg.grad_live_plot_every) == 0 or step == 1):
                _save_grad_rms_plot(
                    history=grad_history,
                    save_path=grad_plot_path,
                    topk_layers=int(cfg.grad_monitor_topk_layers),
                    log_scale=bool(cfg.grad_monitor_log_scale),
                )

        if cfg.grad_clip_norm > 0.0:
            _renormalize_gradients(params=params, target_norm=float(cfg.grad_clip_norm))

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0 or step == 1:
            grad_suffix = ""
            if latest_grad_global is not None:
                grad_suffix = f" grad_global={latest_grad_global:.3e}"
            print(
                f"step={step:04d} epoch={epoch:04d} loss={loss.item():.6f} "
                f"recon_y={recon_y.item():.6f} recon_w={recon_w.item():.6f} kl={kl.item():.6f} "
                f"ablation={'on' if ablation_active else 'off'}"
                f" lr={optimizer.param_groups[0]['lr']:.3e}"
                f"{grad_suffix}"
            )

        if val_size > 0 and epoch_completed and (epoch % val_every_epochs == 0):
            val_ablation_active = step <= int(cfg.ablation_steps)
            val_zero_dist = bool(cfg.ablation_zero_dist_conditioning and val_ablation_active)
            val_deterministic_z = bool(cfg.ablation_deterministic_z and val_ablation_active)
            val_loss, val_recon_y, val_recon_w, val_kl, val_patch_true, val_patch_pred = _validate_on_holdout(
                cfg=cfg,
                dist_encoder=dist_encoder,
                mini_vae=mini_vae,
                X_all=X_all,
                W_all=W_all,
                val_idx=val_idx,
                batch_size=val_batch_size,
                device=device,
                zero_dist_conditioning=val_zero_dist,
                deterministic_z=val_deterministic_z,
            )
            print(
                f"val epoch={epoch:04d} loss={val_loss:.6f} "
                f"recon_y={val_recon_y:.6f} recon_w={val_recon_w:.6f} kl={val_kl:.6f} "
                f"ablation={'on' if val_ablation_active else 'off'}"
            )
            print(f"val_patch_true epoch={epoch:04d} w_patch={_format_patch_for_log(val_patch_true)}")
            print(f"val_patch_pred epoch={epoch:04d} w_hat={_format_patch_for_log(val_patch_pred)}")
            if cfg.val_random_z_probe:
                (
                    val_rand_loss,
                    val_rand_recon_y,
                    val_rand_recon_w,
                    val_rand_kl,
                    val_rand_patch_true,
                    val_rand_patch_pred,
                ) = _validate_on_holdout(
                    cfg=cfg,
                    dist_encoder=dist_encoder,
                    mini_vae=mini_vae,
                    X_all=X_all,
                    W_all=W_all,
                    val_idx=val_idx,
                    batch_size=val_batch_size,
                    device=device,
                    randomize_latent=True,
                    zero_dist_conditioning=val_zero_dist,
                )
                print(
                    f"val_randz epoch={epoch:04d} loss={val_rand_loss:.6f} "
                    f"recon_y={val_rand_recon_y:.6f} recon_w={val_rand_recon_w:.6f} kl={val_rand_kl:.6f} "
                    f"ablation={'on' if val_ablation_active else 'off'}"
                )
                print(
                    f"val_randz_patch_true epoch={epoch:04d} "
                    f"w_patch={_format_patch_for_log(val_rand_patch_true)}"
                )
                print(
                    f"val_randz_patch_pred epoch={epoch:04d} "
                    f"w_hat={_format_patch_for_log(val_rand_patch_pred)}"
                )

    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": mini_vae.state_dict(),
            "distribution_encoder_state": dist_encoder.state_dict(),
            "config": {
                "pretrain": asdict(cfg),
                "distribution": asdict(dist_cfg),
                "mini_vae": asdict(mini_cfg),
            },
        },
        save_path,
    )
    if cfg.grad_monitor_enabled and grad_history:
        _save_grad_rms_csv(grad_history, save_path=grad_csv_path)
        plot_saved = _save_grad_rms_plot(
            history=grad_history,
            save_path=grad_plot_path,
            topk_layers=int(cfg.grad_monitor_topk_layers),
            log_scale=bool(cfg.grad_monitor_log_scale),
        )
        print(f"saved grad stats csv: {grad_csv_path}")
        if plot_saved:
            print(f"saved grad plot: {grad_plot_path}")
    print(f"saved checkpoint: {save_path}")


if __name__ == "__main__":
    run_pretrain_stub(PretrainConfig())
