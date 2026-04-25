from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae_shared import _key_padding_to_attn_bias, sinusoidal_embedding


@dataclass(slots=True)
class DiffusionScheduleConfig:
    num_train_timesteps: int = 1000
    schedule_type: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 0.008
    default_sampling_steps: int = 50
    sampler_type: str = "ddim"
    ddim_eta: float = 0.0


@dataclass(slots=True)
class LayerLatentDiffusionPriorConfig:
    z_dim: int = 2048
    num_latent_tokens: int = 8
    cond_dim: int = 128
    cond_global_dim: int = 0
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: float = 4.0
    dropout: float = 0.0
    cond_pool_tokens: int = 8
    pos_fourier_dim: int = 64
    time_embed_dim: int = 256
    use_qk_norm: bool = True
    use_cross_conditioning: bool = False
    prediction_type: str = "v"
    latent_normalization: str = "diagonal"
    use_decoder_aux: bool = False
    decoder_aux_lambda: float = 0.0
    decoder_aux_max_sigma: float = 0.5
    decoder_aux_behavioral_coef: float = 1.0
    decoder_aux_structural_coef: float = 1.0
    decoder_aux_struct_gamma: float = 0.5
    decoder_aux_struct_lambda_dir: float = 1.0
    decoder_aux_struct_lambda_scale: float = 0.25
    decoder_aux_struct_lambda_rec: float = 0.5
    decoder_aux_struct_lambda_rel: float = 0.1
    decoder_aux_struct_huber_delta: float = 0.1
    schedule: DiffusionScheduleConfig = field(default_factory=DiffusionScheduleConfig)


def build_layer_latent_diffusion_prior_config(raw_cfg: Mapping[str, Any]) -> LayerLatentDiffusionPriorConfig:
    schedule_raw = raw_cfg.get("schedule", {})
    if schedule_raw is None:
        schedule_raw = {}
    if not isinstance(schedule_raw, Mapping):
        raise TypeError("model.latent_diffusion_prior.schedule must be a mapping")
    return LayerLatentDiffusionPriorConfig(
        z_dim=int(raw_cfg.get("z_dim", 2048)),
        num_latent_tokens=int(raw_cfg.get("num_latent_tokens", 8)),
        cond_dim=int(raw_cfg.get("cond_dim", 128)),
        cond_global_dim=int(raw_cfg.get("cond_global_dim", 0)),
        d_model=int(raw_cfg.get("d_model", 256)),
        n_layers=int(raw_cfg.get("n_layers", 6)),
        n_heads=int(raw_cfg.get("n_heads", 8)),
        ffn_mult=float(raw_cfg.get("ffn_mult", 4.0)),
        dropout=float(raw_cfg.get("dropout", 0.0)),
        cond_pool_tokens=int(raw_cfg.get("cond_pool_tokens", 8)),
        pos_fourier_dim=int(raw_cfg.get("pos_fourier_dim", 64)),
        time_embed_dim=int(raw_cfg.get("time_embed_dim", 256)),
        use_qk_norm=bool(raw_cfg.get("use_qk_norm", True)),
        use_cross_conditioning=bool(raw_cfg.get("use_cross_conditioning", False)),
        prediction_type=str(raw_cfg.get("prediction_type", "v")),
        latent_normalization=str(raw_cfg.get("latent_normalization", "diagonal")),
        use_decoder_aux=bool(raw_cfg.get("use_decoder_aux", False)),
        decoder_aux_lambda=float(raw_cfg.get("decoder_aux_lambda", 0.0)),
        decoder_aux_max_sigma=float(raw_cfg.get("decoder_aux_max_sigma", 0.5)),
        decoder_aux_behavioral_coef=float(raw_cfg.get("decoder_aux_behavioral_coef", 1.0)),
        decoder_aux_structural_coef=float(raw_cfg.get("decoder_aux_structural_coef", 1.0)),
        decoder_aux_struct_gamma=float(raw_cfg.get("decoder_aux_struct_gamma", 0.5)),
        decoder_aux_struct_lambda_dir=float(raw_cfg.get("decoder_aux_struct_lambda_dir", 1.0)),
        decoder_aux_struct_lambda_scale=float(raw_cfg.get("decoder_aux_struct_lambda_scale", 0.25)),
        decoder_aux_struct_lambda_rec=float(raw_cfg.get("decoder_aux_struct_lambda_rec", 0.5)),
        decoder_aux_struct_lambda_rel=float(raw_cfg.get("decoder_aux_struct_lambda_rel", 0.1)),
        decoder_aux_struct_huber_delta=float(raw_cfg.get("decoder_aux_struct_huber_delta", 0.1)),
        schedule=DiffusionScheduleConfig(
            num_train_timesteps=int(schedule_raw.get("num_train_timesteps", 1000)),
            schedule_type=str(schedule_raw.get("schedule_type", "cosine")),
            beta_start=float(schedule_raw.get("beta_start", 1e-4)),
            beta_end=float(schedule_raw.get("beta_end", 0.02)),
            cosine_s=float(schedule_raw.get("cosine_s", 0.008)),
            default_sampling_steps=int(schedule_raw.get("default_sampling_steps", 50)),
            sampler_type=str(schedule_raw.get("sampler_type", "ddim")),
            ddim_eta=float(schedule_raw.get("ddim_eta", 0.0)),
        ),
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(f"RMSNorm expected last dim {self.weight.shape[0]}, got {x.shape[-1]}")
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight


class FiLMNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        if gamma.ndim != 2 or beta.ndim != 2:
            raise ValueError(f"gamma and beta must be [B,D], got {tuple(gamma.shape)} and {tuple(beta.shape)}")
        h = self.norm(x)
        return h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float, dropout: float) -> None:
        super().__init__()
        hidden = max(1, int(float(mult) * int(dim)))
        self.up_proj = nn.Linear(dim, hidden)
        self.gate_proj = nn.Linear(dim, hidden)
        self.down_proj = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        dropout: float,
        use_qk_norm: bool,
    ) -> None:
        super().__init__()
        if int(dim) % int(n_heads) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(dim // n_heads)
        self.use_qk_norm = bool(use_qk_norm)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        *,
        key_value: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError(f"query must be [B,T,D], got {tuple(query.shape)}")
        kv = query if key_value is None else key_value
        if kv.ndim != 3:
            raise ValueError(f"key_value must be [B,S,D], got {tuple(kv.shape)}")
        if int(query.shape[0]) != int(kv.shape[0]) or int(query.shape[2]) != self.dim or int(kv.shape[2]) != self.dim:
            raise ValueError(
                "query/key_value shapes must be [B,T,D] / [B,S,D] with aligned batch and model dims, "
                f"got {tuple(query.shape)} and {tuple(kv.shape)}"
            )

        q = self._reshape(self.q_proj(query))
        k = self._reshape(self.k_proj(kv))
        v = self._reshape(self.v_proj(kv))

        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * math.sqrt(float(self.head_dim))
            k = F.normalize(k, dim=-1)

        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.ndim != 2 or tuple(key_padding_mask.shape) != tuple(kv.shape[:2]):
                raise ValueError(
                    f"key_padding_mask must be {tuple(kv.shape[:2])}, got {tuple(key_padding_mask.shape)}"
                )
            attn_mask = _key_padding_to_attn_bias(key_padding_mask.to(dtype=torch.bool, device=kv.device), dtype=q.dtype)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.dim)
        return self.out_proj(out)


class VariableLengthConditionEncoder(nn.Module):
    def __init__(
        self,
        *,
        cond_dim: int,
        d_model: int,
        n_heads: int,
        pool_tokens: int,
        pos_fourier_dim: int,
        dropout: float,
        use_qk_norm: bool,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.d_model = int(d_model)
        self.pool_tokens = int(pool_tokens)
        self.pos_fourier_dim = int(pos_fourier_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.cond_dim + self.pos_fourier_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.d_model),
        )
        self.cond_queries = nn.Parameter(torch.randn(self.pool_tokens, self.d_model) * 0.02)
        self.pool_cross_attn = MultiHeadAttention(
            self.d_model,
            n_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
        )
        self.pool_ffn_norm = RMSNorm(self.d_model)
        self.pool_ffn = SwiGLUFeedForward(self.d_model, mult=2.0, dropout=dropout)
        self.global_proj = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        cond_patch: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond_patch.ndim != 3:
            raise ValueError(f"cond_patch must be [B,T,d_cond], got {tuple(cond_patch.shape)}")
        if int(cond_patch.shape[2]) != self.cond_dim:
            raise ValueError(f"cond_patch last dim must be {self.cond_dim}, got {int(cond_patch.shape[2])}")
        batch, seq_len, _ = cond_patch.shape
        if patch_mask is None:
            patch_mask = torch.ones((batch, seq_len), device=cond_patch.device, dtype=torch.bool)
        else:
            if patch_mask.ndim != 2 or tuple(patch_mask.shape) != (batch, seq_len):
                raise ValueError(f"patch_mask must be {(batch, seq_len)}, got {tuple(patch_mask.shape)}")
            patch_mask = patch_mask.to(device=cond_patch.device, dtype=torch.bool)

        patch_positions = self._patch_positions(batch=batch, seq_len=seq_len, patch_mask=patch_mask, device=cond_patch.device)
        patch_pos_emb = sinusoidal_embedding(patch_positions, self.pos_fourier_dim).to(dtype=cond_patch.dtype)
        cond_tokens = self.token_mlp(torch.cat([cond_patch, patch_pos_emb], dim=-1))
        cond_tokens = cond_tokens * patch_mask.unsqueeze(-1).to(dtype=cond_tokens.dtype)

        queries = self.cond_queries.unsqueeze(0).expand(batch, -1, -1)
        pooled = queries + self.pool_cross_attn(queries, key_value=cond_tokens, key_padding_mask=patch_mask)
        pooled = pooled + self.pool_ffn(self.pool_ffn_norm(pooled))
        global_ctx = self.global_proj(pooled.mean(dim=1))
        return global_ctx, pooled

    @staticmethod
    def _patch_positions(
        *,
        batch: int,
        seq_len: int,
        patch_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        position_idx = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len).expand(batch, -1)
        valid_counts = patch_mask.to(dtype=torch.long).sum(dim=1, keepdim=True).clamp_min(1).to(dtype=torch.float32)
        return (position_idx + 0.5) / valid_counts


class LatentDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        ffn_mult: float,
        dropout: float,
        use_qk_norm: bool,
        use_cross_conditioning: bool,
    ) -> None:
        super().__init__()
        self.use_cross_conditioning = bool(use_cross_conditioning)
        self.attn_norm = FiLMNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
        )
        self.ffn_norm = FiLMNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, mult=ffn_mult, dropout=dropout)
        self.attn_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))
        self.ffn_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))

        if self.use_cross_conditioning:
            self.cross_norm = FiLMNorm(d_model)
            self.cross_attn = MultiHeadAttention(
                d_model,
                n_heads,
                dropout=dropout,
                use_qk_norm=use_qk_norm,
            )
            self.cross_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))
        else:
            self.cross_norm = None
            self.cross_attn = None
            self.cross_modulation = None

        self._init_zero_modulation(self.attn_modulation)
        self._init_zero_modulation(self.ffn_modulation)
        if self.cross_modulation is not None:
            self._init_zero_modulation(self.cross_modulation)

    @staticmethod
    def _init_zero_modulation(module: nn.Sequential) -> None:
        linear = module[-1]
        if not isinstance(linear, nn.Linear):
            raise TypeError("expected final Linear in modulation MLP")
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        *,
        cond_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_gamma, attn_beta, attn_gate = self.attn_modulation(g).chunk(3, dim=-1)
        attn_hidden = self.attn_norm(x, attn_gamma, attn_beta)
        x = x + attn_gate.unsqueeze(1) * self.self_attn(attn_hidden)

        if self.use_cross_conditioning:
            if cond_memory is None:
                raise ValueError("cond_memory is required when use_cross_conditioning=True")
            assert self.cross_norm is not None
            assert self.cross_attn is not None
            assert self.cross_modulation is not None
            cross_gamma, cross_beta, cross_gate = self.cross_modulation(g).chunk(3, dim=-1)
            cross_hidden = self.cross_norm(x, cross_gamma, cross_beta)
            x = x + cross_gate.unsqueeze(1) * self.cross_attn(cross_hidden, key_value=cond_memory)

        ffn_gamma, ffn_beta, ffn_gate = self.ffn_modulation(g).chunk(3, dim=-1)
        ffn_hidden = self.ffn_norm(x, ffn_gamma, ffn_beta)
        x = x + ffn_gate.unsqueeze(1) * self.ffn(ffn_hidden)
        return x


class DiffusionSchedule(nn.Module):
    def __init__(self, cfg: DiffusionScheduleConfig) -> None:
        super().__init__()
        if int(cfg.num_train_timesteps) <= 0:
            raise ValueError(f"num_train_timesteps must be positive, got {cfg.num_train_timesteps}")
        self.cfg = cfg
        betas = self._build_betas(cfg)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-12)
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-12)
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars).clamp_min(1e-12)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev, persistent=False)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt((1.0 - alpha_bars).clamp_min(1e-12)), persistent=False)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20), persistent=False)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1, persistent=False)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2, persistent=False)

    @staticmethod
    def _build_betas(cfg: DiffusionScheduleConfig) -> torch.Tensor:
        schedule_type = str(cfg.schedule_type).strip().lower()
        steps = int(cfg.num_train_timesteps)
        if schedule_type == "linear":
            return torch.linspace(float(cfg.beta_start), float(cfg.beta_end), steps, dtype=torch.float32)
        if schedule_type == "cosine":
            s = float(cfg.cosine_s)
            idx = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
            t = idx / float(steps)
            alpha_bar = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5).pow(2)
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
            return betas.clamp(1e-5, 0.999).to(dtype=torch.float32)
        raise ValueError(f"Unsupported diffusion schedule_type: {cfg.schedule_type!r}")

    def sample_timesteps(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=int(self.cfg.num_train_timesteps),
            size=(int(batch_size),),
            device=device,
            dtype=torch.long,
        )

    def alpha_sigma(self, timesteps: torch.Tensor, *, x_ndim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be [B], got {tuple(timesteps.shape)}")
        alpha = self.sqrt_alpha_bars[timesteps]
        sigma = self.sqrt_one_minus_alpha_bars[timesteps]
        while alpha.ndim < x_ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha, sigma

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha, sigma = self.alpha_sigma(timesteps, x_ndim=x0.ndim)
        return alpha * x0 + sigma * noise, noise

    def prediction_target(
        self,
        *,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        pred_type = str(prediction_type).strip().lower()
        alpha, sigma = self.alpha_sigma(timesteps, x_ndim=x0.ndim)
        if pred_type == "eps":
            return noise
        if pred_type == "x0":
            return x0
        if pred_type == "v":
            return alpha * noise - sigma * x0
        raise ValueError(f"Unsupported prediction_type: {prediction_type!r}")

    def predict_x0(
        self,
        *,
        x_t: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        pred_type = str(prediction_type).strip().lower()
        alpha, sigma = self.alpha_sigma(timesteps, x_ndim=x_t.ndim)
        if pred_type == "x0":
            return model_pred
        if pred_type == "eps":
            return (x_t - sigma * model_pred) / alpha.clamp_min(1e-8)
        if pred_type == "v":
            return alpha * x_t - sigma * model_pred
        raise ValueError(f"Unsupported prediction_type: {prediction_type!r}")

    def predict_noise(
        self,
        *,
        x_t: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        pred_type = str(prediction_type).strip().lower()
        alpha, sigma = self.alpha_sigma(timesteps, x_ndim=x_t.ndim)
        if pred_type == "eps":
            return model_pred
        if pred_type == "x0":
            return (x_t - alpha * model_pred) / sigma.clamp_min(1e-8)
        if pred_type == "v":
            return sigma * x_t + alpha * model_pred
        raise ValueError(f"Unsupported prediction_type: {prediction_type!r}")

    def ddim_step(
        self,
        *,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        timesteps: torch.Tensor,
        next_timesteps: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        alpha_t, sigma_t = self.alpha_sigma(timesteps, x_ndim=x_t.ndim)
        eps_pred = (x_t - alpha_t * x0_pred) / sigma_t.clamp_min(1e-8)
        alpha_bar_t = self.alpha_bars[timesteps].view(-1, *([1] * (x_t.ndim - 1)))
        alpha_bar_next = self.alpha_bars[next_timesteps].view(-1, *([1] * (x_t.ndim - 1)))
        sigma_t = eta * torch.sqrt(
            ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t).clamp_min(1e-12))
            * (1.0 - (alpha_bar_t / alpha_bar_next).clamp_max(1.0))
        ).clamp_min(0.0)
        dir_scale = torch.sqrt((1.0 - alpha_bar_next - sigma_t.pow(2)).clamp_min(0.0))
        noise = torch.randn_like(x_t) if float(eta) > 0.0 else torch.zeros_like(x_t)
        return torch.sqrt(alpha_bar_next) * x0_pred + dir_scale * eps_pred + sigma_t * noise

    def ddpm_step(
        self,
        *,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        coef1 = self.posterior_mean_coef1[timesteps].view(-1, *([1] * (x_t.ndim - 1)))
        coef2 = self.posterior_mean_coef2[timesteps].view(-1, *([1] * (x_t.ndim - 1)))
        mean = coef1 * x0_pred + coef2 * x_t
        variance = self.posterior_variance[timesteps].view(-1, *([1] * (x_t.ndim - 1)))
        nonzero_mask = (timesteps > 0).to(device=x_t.device, dtype=x_t.dtype).view(-1, *([1] * (x_t.ndim - 1)))
        noise = torch.randn_like(x_t)
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    def make_inference_timesteps(self, num_steps: int, *, device: torch.device) -> torch.Tensor:
        requested = max(1, int(num_steps))
        max_t = int(self.cfg.num_train_timesteps) - 1
        values = torch.linspace(float(max_t), 0.0, requested, device=device)
        timesteps = values.round().to(dtype=torch.long)
        timesteps = torch.unique_consecutive(timesteps)
        if int(timesteps[-1].item()) != 0:
            timesteps = torch.cat([timesteps, timesteps.new_zeros(1)], dim=0)
        return timesteps


class LayerLatentDiffusionPrior(nn.Module):
    def __init__(self, cfg: LayerLatentDiffusionPriorConfig) -> None:
        super().__init__()
        if int(cfg.z_dim) <= 0:
            raise ValueError(f"z_dim must be positive, got {cfg.z_dim}")
        if int(cfg.num_latent_tokens) <= 0:
            raise ValueError(f"num_latent_tokens must be positive, got {cfg.num_latent_tokens}")
        if int(cfg.z_dim) % int(cfg.num_latent_tokens) != 0:
            raise ValueError(
                f"z_dim ({cfg.z_dim}) must be divisible by num_latent_tokens ({cfg.num_latent_tokens})"
            )
        if str(cfg.prediction_type).strip().lower() not in {"v", "eps", "x0"}:
            raise ValueError(f"Unsupported prediction_type: {cfg.prediction_type!r}")
        if str(cfg.latent_normalization).strip().lower() != "diagonal":
            raise ValueError(
                "Only diagonal latent normalization is currently supported; "
                f"got {cfg.latent_normalization!r}"
            )
        if float(cfg.decoder_aux_lambda) < 0.0:
            raise ValueError(f"decoder_aux_lambda must be >= 0, got {cfg.decoder_aux_lambda}")
        if float(cfg.decoder_aux_max_sigma) <= 0.0:
            raise ValueError(f"decoder_aux_max_sigma must be > 0, got {cfg.decoder_aux_max_sigma}")

        self.cfg = cfg
        self.z_dim = int(cfg.z_dim)
        self.num_latent_tokens = int(cfg.num_latent_tokens)
        self.latent_token_dim = int(cfg.z_dim // cfg.num_latent_tokens)
        self.prediction_type = str(cfg.prediction_type).strip().lower()
        self.schedule = DiffusionSchedule(cfg.schedule)

        self.cond_encoder = VariableLengthConditionEncoder(
            cond_dim=int(cfg.cond_dim),
            d_model=int(cfg.d_model),
            n_heads=int(cfg.n_heads),
            pool_tokens=int(cfg.cond_pool_tokens),
            pos_fourier_dim=int(cfg.pos_fourier_dim),
            dropout=float(cfg.dropout),
            use_qk_norm=bool(cfg.use_qk_norm),
        )
        self.latent_in_proj = nn.Linear(self.latent_token_dim, int(cfg.d_model))
        self.latent_out_proj = nn.Linear(int(cfg.d_model), self.latent_token_dim)
        self.slot_pos_proj = nn.Linear(int(cfg.pos_fourier_dim), int(cfg.d_model))
        self.time_embed = nn.Sequential(
            nn.Linear(int(cfg.pos_fourier_dim), int(cfg.time_embed_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.time_embed_dim), int(cfg.d_model)),
        )
        global_input_dim = int(cfg.d_model) + int(cfg.d_model) + max(0, int(cfg.cond_global_dim))
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, int(cfg.d_model)),
            nn.SiLU(),
            nn.Linear(int(cfg.d_model), int(cfg.d_model)),
        )
        self.blocks = nn.ModuleList(
            [
                LatentDiTBlock(
                    d_model=int(cfg.d_model),
                    n_heads=int(cfg.n_heads),
                    ffn_mult=float(cfg.ffn_mult),
                    dropout=float(cfg.dropout),
                    use_qk_norm=bool(cfg.use_qk_norm),
                    use_cross_conditioning=bool(cfg.use_cross_conditioning),
                )
                for _ in range(max(1, int(cfg.n_layers)))
            ]
        )
        self.out_norm = RMSNorm(int(cfg.d_model))
        self.register_buffer("latent_mean", torch.zeros(self.z_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("latent_std", torch.ones(self.z_dim, dtype=torch.float32), persistent=True)

    def set_latent_normalization_stats(self, *, latent_mean: torch.Tensor, latent_std: torch.Tensor) -> None:
        mean = latent_mean.detach().to(device=self.latent_mean.device, dtype=self.latent_mean.dtype).view(-1)
        std = latent_std.detach().to(device=self.latent_std.device, dtype=self.latent_std.dtype).view(-1)
        if int(mean.numel()) != self.z_dim or int(std.numel()) != self.z_dim:
            raise ValueError(
                f"latent normalization stats must have {self.z_dim} elements, got {mean.numel()} and {std.numel()}"
            )
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std.clamp_min(1e-6))

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latent_mean.to(device=latents.device, dtype=latents.dtype)) / self.latent_std.to(
            device=latents.device,
            dtype=latents.dtype,
        )

    def unnormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latent_std.to(device=latents.device, dtype=latents.dtype) + self.latent_mean.to(
            device=latents.device,
            dtype=latents.dtype,
        )

    def tokens_from_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if latents.ndim == 2:
            if int(latents.shape[1]) != self.z_dim:
                raise ValueError(f"flat latents must be [B,{self.z_dim}], got {tuple(latents.shape)}")
            return latents.view(latents.shape[0], self.num_latent_tokens, self.latent_token_dim), True
        if latents.ndim == 3:
            expected = (self.num_latent_tokens, self.latent_token_dim)
            if tuple(latents.shape[1:]) != expected:
                raise ValueError(f"token latents must be [B,{expected[0]},{expected[1]}], got {tuple(latents.shape)}")
            return latents, False
        raise ValueError(f"latents must be [B,z_dim] or [B,L,d], got {tuple(latents.shape)}")

    def latents_from_tokens(self, latents: torch.Tensor, *, flatten: bool) -> torch.Tensor:
        if flatten:
            return latents.reshape(latents.shape[0], self.z_dim)
        return latents

    def forward(
        self,
        u_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        cond_patch: torch.Tensor,
        cond_global: torch.Tensor | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent_tokens, flatten_output = self.tokens_from_latents(u_noisy)
        if timesteps.ndim != 1 or int(timesteps.shape[0]) != int(latent_tokens.shape[0]):
            raise ValueError(
                f"timesteps must be [B] matching batch size {latent_tokens.shape[0]}, got {tuple(timesteps.shape)}"
            )

        x = self.latent_in_proj(latent_tokens)
        x = x + self._slot_positional_encoding(batch_size=int(x.shape[0]), device=x.device, dtype=x.dtype)

        cond_summary, cond_memory = self.cond_encoder(cond_patch, patch_mask=patch_mask)
        time_emb = self._time_embedding(timesteps, device=x.device, dtype=x.dtype)
        cond_global_input = self._resolve_cond_global(
            cond_global,
            batch_size=int(x.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        global_ctx = self.global_mlp(torch.cat([time_emb, cond_summary, cond_global_input], dim=-1))

        for block in self.blocks:
            x = block(
                x,
                global_ctx,
                cond_memory=cond_memory if bool(self.cfg.use_cross_conditioning) else None,
            )

        pred_tokens = self.latent_out_proj(self.out_norm(x))
        return self.latents_from_tokens(pred_tokens, flatten=flatten_output)

    def compute_training_targets(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_tokens, _ = self.tokens_from_latents(self.normalize_latents(latents))
        noisy_tokens, used_noise = self.schedule.q_sample(clean_tokens, timesteps, noise=noise)
        target_tokens = self.schedule.prediction_target(
            x0=clean_tokens,
            noise=used_noise,
            timesteps=timesteps,
            prediction_type=self.prediction_type,
        )
        return noisy_tokens, target_tokens, clean_tokens

    @torch.no_grad()
    def sample_latents(
        self,
        *,
        cond_patch: torch.Tensor,
        cond_global: torch.Tensor | None = None,
        patch_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        sampler_type: str | None = None,
        eta: float | None = None,
    ) -> torch.Tensor:
        batch = int(cond_patch.shape[0])
        device = cond_patch.device
        dtype = cond_patch.dtype
        x_t = torch.randn(
            batch,
            self.num_latent_tokens,
            self.latent_token_dim,
            device=device,
            dtype=dtype,
        )
        steps = int(num_steps) if num_steps is not None else int(self.cfg.schedule.default_sampling_steps)
        sampler = str(sampler_type or self.cfg.schedule.sampler_type).strip().lower()
        eta_value = float(self.cfg.schedule.ddim_eta if eta is None else eta)
        timesteps = self.schedule.make_inference_timesteps(steps, device=device)

        for idx, timestep in enumerate(timesteps.tolist()):
            t = torch.full((batch,), int(timestep), device=device, dtype=torch.long)
            model_pred = self.forward(
                x_t,
                t,
                cond_patch,
                cond_global=cond_global,
                patch_mask=patch_mask,
            )
            x0_pred = self.schedule.predict_x0(
                x_t=x_t,
                model_pred=model_pred,
                timesteps=t,
                prediction_type=self.prediction_type,
            )
            if idx == len(timesteps) - 1:
                x_t = x0_pred
                break

            next_t = torch.full((batch,), int(timesteps[idx + 1].item()), device=device, dtype=torch.long)
            if sampler == "ddpm":
                x_t = self.schedule.ddpm_step(x_t=x_t, x0_pred=x0_pred, timesteps=t)
            elif sampler == "ddim":
                x_t = self.schedule.ddim_step(
                    x_t=x_t,
                    x0_pred=x0_pred,
                    timesteps=t,
                    next_timesteps=next_t,
                    eta=eta_value,
                )
            else:
                raise ValueError(f"Unsupported sampler_type: {sampler!r}")

        sampled = self.latents_from_tokens(x_t, flatten=True)
        return self.unnormalize_latents(sampled)

    def _slot_positional_encoding(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        slot_pos = (torch.arange(self.num_latent_tokens, device=device, dtype=torch.float32) + 0.5) / float(
            self.num_latent_tokens
        )
        emb = sinusoidal_embedding(slot_pos, self.cfg.pos_fourier_dim).to(dtype=dtype)
        return self.slot_pos_proj(emb).unsqueeze(0).expand(batch_size, -1, -1)

    def _time_embedding(
        self,
        timesteps: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        time_emb = sinusoidal_embedding(timesteps.to(device=device, dtype=torch.float32), self.cfg.pos_fourier_dim)
        return self.time_embed(time_emb.to(dtype=dtype))

    def _resolve_cond_global(
        self,
        cond_global: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cond_global_dim = max(0, int(self.cfg.cond_global_dim))
        if cond_global_dim == 0:
            return torch.zeros((batch_size, 0), device=device, dtype=dtype)
        if cond_global is None:
            return torch.zeros((batch_size, cond_global_dim), device=device, dtype=dtype)
        if cond_global.ndim != 2 or tuple(cond_global.shape) != (batch_size, cond_global_dim):
            raise ValueError(
                f"cond_global must be {(batch_size, cond_global_dim)} when provided, got {tuple(cond_global.shape)}"
            )
        return cond_global.to(device=device, dtype=dtype)


def compute_layer_latent_diffusion_loss(
    model: LayerLatentDiffusionPrior,
    *,
    clean_latents: torch.Tensor,
    cond_patch: torch.Tensor,
    timesteps: torch.Tensor,
    patch_mask: torch.Tensor | None = None,
    cond_global: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    decoder_aux_model: Any | None = None,
    decoder_aux_W: torch.Tensor | None = None,
    decoder_aux_X: torch.Tensor | None = None,
    decoder_aux_x_mask: torch.Tensor | None = None,
    decoder_aux_d_in_mask: torch.Tensor | None = None,
    decoder_aux_d_out_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    noisy_tokens, target_tokens, clean_tokens = model.compute_training_targets(
        clean_latents,
        timesteps,
        noise=noise,
    )
    pred_tokens = model(
        noisy_tokens,
        timesteps,
        cond_patch,
        cond_global=cond_global,
        patch_mask=patch_mask,
    )
    pred_tokens_shaped, _ = model.tokens_from_latents(pred_tokens)
    diffusion_loss = F.mse_loss(pred_tokens_shaped, target_tokens)
    x0_pred_tokens = model.schedule.predict_x0(
        x_t=noisy_tokens,
        model_pred=pred_tokens_shaped,
        timesteps=timesteps,
        prediction_type=model.prediction_type,
    )
    zero = diffusion_loss.new_zeros(())
    decoder_aux_loss = zero
    decoder_aux_behavioral_loss = zero
    decoder_aux_structural_loss = zero
    decoder_aux_applied_fraction = zero

    if bool(model.cfg.use_decoder_aux) and float(model.cfg.decoder_aux_lambda) > 0.0:
        missing: list[str] = []
        if decoder_aux_model is None:
            missing.append("decoder_aux_model")
        if decoder_aux_W is None:
            missing.append("decoder_aux_W")
        if decoder_aux_X is None:
            missing.append("decoder_aux_X")
        if decoder_aux_x_mask is None:
            missing.append("decoder_aux_x_mask")
        if decoder_aux_d_in_mask is None:
            missing.append("decoder_aux_d_in_mask")
        if decoder_aux_d_out_mask is None:
            missing.append("decoder_aux_d_out_mask")
        if patch_mask is None:
            missing.append("patch_mask")
        if missing:
            raise ValueError(
                "decoder auxiliary loss requires additional tensors; missing="
                + ",".join(missing)
            )

        sigma_t = model.schedule.alpha_sigma(timesteps, x_ndim=1)[1]
        active_mask = sigma_t <= float(model.cfg.decoder_aux_max_sigma)
        decoder_aux_applied_fraction = active_mask.to(dtype=diffusion_loss.dtype).mean()
        if bool(active_mask.any().item()):
            active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            decoder_z = model.unnormalize_latents(model.latents_from_tokens(x0_pred_tokens, flatten=True))
            decoder_z_active = decoder_z.index_select(dim=0, index=active_idx)
            cond_patch_active = cond_patch.index_select(dim=0, index=active_idx)
            patch_mask_active = patch_mask.index_select(dim=0, index=active_idx)
            target_W_active = decoder_aux_W.index_select(dim=0, index=active_idx)
            target_X_active = decoder_aux_X.index_select(dim=0, index=active_idx)
            x_mask_active = decoder_aux_x_mask.index_select(dim=0, index=active_idx)
            d_in_mask_active = decoder_aux_d_in_mask.index_select(dim=0, index=active_idx)
            d_out_mask_active = decoder_aux_d_out_mask.index_select(dim=0, index=active_idx)
            d_in = int(target_W_active.shape[1])
            d_out = int(target_W_active.shape[2])
            T = int(cond_patch_active.shape[1])
            d_in_pad = int(T) * int(decoder_aux_model.cfg.patch_size)
            if d_in != d_in_pad:
                raise ValueError(
                    "decoder auxiliary expected W slices with d_in == T*patch_size, "
                    f"got d_in={d_in} T={T} patch_size={decoder_aux_model.cfg.patch_size}"
                )
            decode_outputs = decoder_aux_model._decode_from_decoder_latent(
                decoder_z_active,
                dist_patch_by_patch=cond_patch_active,
                patch_mask=patch_mask_active,
                d_in_mask=d_in_mask_active,
                d_out_mask=d_out_mask_active,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
            )
            target_W_hat = decode_outputs[0]
            pred_dirs = decode_outputs[3]
            from models.big_weight_vae import BigWeightVAE

            decoder_aux_behavioral_loss = BigWeightVAE.operator_recon_loss(
                target_X_active,
                target_W_active,
                target_W_hat,
                x_mask=x_mask_active,
                d_in_mask=d_in_mask_active,
                d_out_mask=d_out_mask_active,
            )
            decoder_aux_structural_loss, _decoder_struct_details = BigWeightVAE.patch_structure_loss(
                target_W_active,
                target_W_hat,
                patch_size=int(decoder_aux_model.cfg.patch_size),
                gamma=float(model.cfg.decoder_aux_struct_gamma),
                lambda_dir=float(model.cfg.decoder_aux_struct_lambda_dir),
                lambda_scale=float(model.cfg.decoder_aux_struct_lambda_scale),
                lambda_rec=float(model.cfg.decoder_aux_struct_lambda_rec),
                lambda_rel=float(model.cfg.decoder_aux_struct_lambda_rel),
                huber_delta=float(model.cfg.decoder_aux_struct_huber_delta),
                pred_dirs=pred_dirs,
                d_in_mask=d_in_mask_active,
                d_out_mask=d_out_mask_active,
            )
            decoder_aux_loss = (
                float(model.cfg.decoder_aux_behavioral_coef) * decoder_aux_behavioral_loss
                + float(model.cfg.decoder_aux_structural_coef) * decoder_aux_structural_loss
            )

    loss = diffusion_loss + float(model.cfg.decoder_aux_lambda) * decoder_aux_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "decoder_aux_loss": decoder_aux_loss,
        "decoder_aux_behavioral_loss": decoder_aux_behavioral_loss,
        "decoder_aux_structural_loss": decoder_aux_structural_loss,
        "decoder_aux_applied_fraction": decoder_aux_applied_fraction,
        "pred_tokens": pred_tokens_shaped,
        "target_tokens": target_tokens,
        "clean_tokens": clean_tokens,
        "noisy_tokens": noisy_tokens,
        "x0_pred_tokens": x0_pred_tokens,
    }


__all__ = [
    "DiffusionSchedule",
    "DiffusionScheduleConfig",
    "FiLMNorm",
    "LayerLatentDiffusionPrior",
    "LayerLatentDiffusionPriorConfig",
    "LatentDiTBlock",
    "RMSNorm",
    "VariableLengthConditionEncoder",
    "build_layer_latent_diffusion_prior_config",
    "compute_layer_latent_diffusion_loss",
]
