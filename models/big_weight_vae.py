from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.distribution_encoder import DistributionConfig, InputDistributionEncodingModule
from models.mini_patch_vae import MiniVAEConfig
from models.patch_tokenizers import MixerPatchTokenizer, PatchConditionedMLPTokenizer, ResidualPatchTokenizer
from models.vae_shared import (
    CrossAttnBlock,
    MLP,
    PerceiverResamplerBlock,
    _decode_direction_and_logscale,
    _rope_attention,
    sinusoidal_embedding,
)


@dataclass(slots=True)
class EncoderConfig:
    self_attn_mode: str = "full"
    cross_attend_only_cls: bool = True


@dataclass(slots=True)
class TTMMemoryConfig:
    proc_tokens: int = 8
    process_depth: int = 2
    summarizer_mode: str = "mlp"
    summarizer_hidden_mult: float = 2.0
    num_blocks: int = 1
    share_weights: bool = False
    dropout: float = 0.0
    use_type_embeddings: bool = True
    use_positional_embeddings: bool = True
    memory_init: str = "learned"
    return_aux: bool = False


@dataclass(slots=True)
class BigVAEConfig:
    d_model: int = 256
    d_lat: int = 256
    num_latents: int = 32
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    n_heads: int = 8
    ffn_mult: float = 4.0
    dropout: float = 0.0
    pos_fourier_dim: int = 64
    use_latent_sampling: bool = True
    disable_z_shortcut: bool = False
    disable_distribution_encoder: bool = False
    patch_tokenizer_kind: str = "residual"
    distribution_encoder_conditioning_kind: str = "legacy"
    latent_bottleneck_kind: str = "perceiver_resampler"
    ttm: TTMMemoryConfig = field(default_factory=TTMMemoryConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass(slots=True)
class ResamplerConfig:
    n_layers: int = 0


@dataclass(slots=True)
class ModelConfig:
    patch_size: int = 16
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    mini_vae: MiniVAEConfig = field(default_factory=MiniVAEConfig)
    big_vae: BigVAEConfig = field(default_factory=BigVAEConfig)
    beta: float = 1e-3
    variant: str = "full"


class LocalOutputSelfAttentionBlock(nn.Module):
    """Local attention within one output-column token group, with RoPE on patch positions."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: float, dropout: float, self_attn_mode: str) -> None:
        super().__init__()
        if self_attn_mode not in {"full", "cls_only"}:
            raise ValueError(f"encoder.self_attn_mode must be 'full' or 'cls_only', got {self_attn_mode}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.self_attn_mode = self_attn_mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_prob_dropout_p = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        hidden = max(1, int(d_model * ffn_mult))
        self.ffn = MLP(d_model, hidden, d_model, dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S, _ = tokens.shape
        h = self.norm_attn(tokens)
        pos = torch.arange(S, device=tokens.device, dtype=torch.float32)

        q_all = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k_all = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v_all = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if self.self_attn_mode == "full":
            attn_out = _rope_attention(
                q=q_all,
                k=k_all,
                v=v_all,
                q_pos=pos,
                k_pos=pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
        else:
            cls_out = _rope_attention(
                q=q_all[:, :, 0:1, :],
                k=k_all,
                v=v_all,
                q_pos=pos[0:1],
                k_pos=pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
            patch_out = _rope_attention(
                q=q_all[:, :, 1:, :],
                k=k_all[:, :, 0:1, :],
                v=v_all[:, :, 0:1, :],
                q_pos=pos[1:],
                k_pos=pos[0:1],
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
            )
            attn_out = torch.cat([cls_out, patch_out], dim=2)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        tokens = tokens + self.dropout(self.out_proj(attn_out))
        tokens = tokens + self.dropout(self.ffn(self.norm_ffn(tokens)))
        return tokens


class TokenConditioningAdapter(nn.Module):
    """Token-wise residual conditioning for encoder patch tokens."""

    def __init__(
        self,
        *,
        d_model: int,
        d_y: int,
        dropout: float,
        d_hidden: int | None = None,
        d_gate: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_hidden) if d_hidden is not None else int(4 * d_model)
        gate_dim = int(d_gate) if d_gate is not None else int(d_model)

        self.y_proj = nn.Linear(d_y, d_model)
        self.h_norm = nn.LayerNorm(d_model)
        self.c_norm = nn.LayerNorm(d_model)

        self.mix_h = nn.Linear(d_model, hidden_dim)
        self.mix_c = nn.Linear(d_model, hidden_dim)
        self.mix_out = nn.Linear(hidden_dim, d_model)

        self.gate_h = nn.Linear(d_model, gate_dim)
        self.gate_c = nn.Linear(d_model, gate_dim)
        self.gate_out = nn.Linear(gate_dim, d_model)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.conditioning_dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.mix_out.bias)
        nn.init.constant_(self.gate_out.bias, -2.0)

    def forward(
        self,
        h: torch.Tensor,
        y_ctx: torch.Tensor,
        map_idx: torch.Tensor,
    ) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must be [B, T_x, d_model], got {tuple(h.shape)}")
        if y_ctx.ndim != 3:
            raise ValueError(f"y_ctx must be [B, T_y, d_y], got {tuple(y_ctx.shape)}")
        if map_idx.ndim != 2:
            raise ValueError(f"map_idx must be [B, T_x], got {tuple(map_idx.shape)}")
        if int(h.shape[0]) != int(y_ctx.shape[0]) or int(h.shape[0]) != int(map_idx.shape[0]):
            raise ValueError(
                "Batch size mismatch between h, y_ctx, and map_idx: "
                f"{tuple(h.shape)}, {tuple(y_ctx.shape)}, {tuple(map_idx.shape)}"
            )
        if int(h.shape[1]) != int(map_idx.shape[1]):
            raise ValueError(
                f"map_idx token length must match h token length, got {tuple(map_idx.shape)} vs {tuple(h.shape)}"
            )

        gather_idx = map_idx.to(device=y_ctx.device, dtype=torch.long).clamp(min=0, max=max(0, int(y_ctx.shape[1]) - 1))
        y_match = y_ctx.gather(dim=1, index=gather_idx.unsqueeze(-1).expand(-1, -1, int(y_ctx.shape[2])))

        c = self.y_proj(y_match)
        c = self.conditioning_dropout(c)

        h_n = self.h_norm(h)
        c_n = self.c_norm(c)

        mixed = F.gelu(self.mix_h(h_n) + self.mix_c(c_n))
        residual = self.mix_out(mixed)

        gate_in = F.gelu(self.gate_h(h_n) + self.gate_c(c_n))
        gate = torch.sigmoid(self.gate_out(gate_in))
        return h + self.alpha * gate * residual


class LatentEncoderLayer(nn.Module):
    """
    One big-encoder block:
    1) local per-output attention over [CLS_o, patches_o]
    2) Perceiver Resampler over the resulting tokens
    """

    def __init__(
        self,
        d_model: int,
        d_lat: int,
        n_heads: int,
        ffn_mult: float,
        dropout: float,
        self_attn_mode: str,
        use_rope_2d: bool = False,
    ) -> None:
        super().__init__()
        if d_lat % n_heads != 0:
            raise ValueError(f"d_lat ({d_lat}) must be divisible by n_heads ({n_heads})")

        self.local_block = LocalOutputSelfAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            self_attn_mode=self_attn_mode,
        )
        self.perceiver_block = PerceiverResamplerBlock(
            d_latent=d_lat,
            d_token=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_rope_2d=use_rope_2d,
        )

    def forward(
        self,
        tokens_by_output: torch.Tensor,
        latents: torch.Tensor,
        cross_attend_only_cls: bool,
        patch_conditioner: TokenConditioningAdapter | None = None,
        patch_conditioning_ctx: torch.Tensor | None = None,
        patch_conditioning_map_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, d_out, L_local, d_model = tokens_by_output.shape

        local_in = tokens_by_output.reshape(B * d_out, L_local, d_model)
        local_out = self.local_block(local_in)
        tokens = local_out.reshape(B, d_out, L_local, d_model)
        if patch_conditioner is not None:
            if patch_conditioning_ctx is None or patch_conditioning_map_idx is None:
                raise ValueError("patch conditioning context and map_idx are required when patch_conditioner is set")
            patch_tokens = tokens[:, :, 1:, :].reshape(B * d_out, L_local - 1, d_model)
            patch_tokens = patch_conditioner(
                h=patch_tokens,
                y_ctx=patch_conditioning_ctx,
                map_idx=patch_conditioning_map_idx,
            )
            tokens = torch.cat(
                [
                    tokens[:, :, :1, :],
                    patch_tokens.reshape(B, d_out, L_local - 1, d_model),
                ],
                dim=2,
            )

        device = tokens.device
        if cross_attend_only_cls:
            kv = tokens[:, :, 0, :]
            token_pos_o = torch.arange(d_out, device=device, dtype=torch.float32)
            token_pos_t = torch.zeros(d_out, device=device, dtype=torch.float32)
        else:
            kv = tokens.reshape(B, d_out * L_local, d_model)
            token_pos_o = torch.arange(d_out, device=device, dtype=torch.float32).repeat_interleave(L_local)
            token_pos_t = torch.arange(L_local, device=device, dtype=torch.float32).repeat(d_out)

        L = latents.shape[1]
        latent_pos = (torch.arange(L, device=device, dtype=torch.float32) + 0.5) / max(float(L), 1.0)
        latents = self.perceiver_block(
            latents=latents,
            tokens=kv,
            latent_pos=latent_pos,
            token_pos=token_pos_o,
            token_pos2=token_pos_t,
        )
        return tokens, latents


class DecoderCrossBlock(nn.Module):
    """Decoder block with query-to-latent cross-attention only."""

    def __init__(self, d_model: int, d_lat: int, n_heads: int, ffn_mult: float, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_lat)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=d_lat,
            vdim=d_lat,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = max(1, int(d_model * ffn_mult))
        self.ffn = MLP(d_model, hidden, d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_tokens: torch.Tensor, z_latents: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(z_latents)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q_tokens = q_tokens + self.dropout(cross_out)
        q_tokens = q_tokens + self.dropout(self.ffn(self.norm_ffn(q_tokens)))
        return q_tokens


class BigWeightVAE(nn.Module):
    """
    Full matrix VAE.

    Input:
    - W: [B, d_in, d_out] or [d_in, d_out]
    - X: [B, n, d_in] or [n, d_in]
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        p = cfg.patch_size
        d_var = cfg.distribution.d_var
        d_dist = cfg.distribution.d_dist
        d_model = cfg.big_vae.d_model
        d_lat = cfg.big_vae.d_lat
        n_heads = cfg.big_vae.n_heads
        num_latents = cfg.big_vae.num_latents
        num_enc_layers = max(1, cfg.big_vae.num_encoder_layers)
        num_dec_layers = max(1, cfg.big_vae.num_decoder_layers)
        dropout = cfg.big_vae.dropout

        self.use_distribution_encoder = not bool(cfg.big_vae.disable_distribution_encoder)
        self.patch_tokenizer_kind = self._normalize_patch_tokenizer_kind(cfg.big_vae.patch_tokenizer_kind)
        self.distribution_encoder_conditioning_kind = self._normalize_distribution_encoder_conditioning_kind(
            cfg.big_vae.distribution_encoder_conditioning_kind
        )
        self.use_legacy_distribution_encoder_conditioning = bool(
            self.use_distribution_encoder and self.distribution_encoder_conditioning_kind == "legacy"
        )
        self.use_token_adapter_distribution_encoder_conditioning = bool(
            self.use_distribution_encoder and self.distribution_encoder_conditioning_kind == "token_adapter"
        )
        if self.use_distribution_encoder:
            self.distribution_encoder: InputDistributionEncodingModule | None = InputDistributionEncodingModule(
                cfg.distribution
            )
            patch_tokenizer_d_dist = d_dist
            query_in_dim = d_model + d_dist
        else:
            self.distribution_encoder = None
            patch_tokenizer_d_dist = 0
            query_in_dim = d_model
        d_patch = cfg.mini_vae.d_patch
        if self.patch_tokenizer_kind == "conditioned_mlp":
            if not self.use_distribution_encoder:
                raise ValueError(
                    "big_vae.patch_tokenizer_kind='conditioned_mlp' requires disable_distribution_encoder=false"
                )
            self.patch_tokenizer = PatchConditionedMLPTokenizer(
                p=p,
                d_var=d_var,
                d_patch=d_patch,
                dropout=dropout,
            )
        else:
            # self.patch_tokenizer = MixerPatchTokenizer(
            #     p=p,
            #     d_var=d_var,
            #     d_dist=d_dist,
            #     d_patch=d_patch,
            #     d_hidden=d_var,
            #     num_mixer_layers=3,
            #     dropout=cfg.big_vae.dropout,
            # )
            self.patch_tokenizer = ResidualPatchTokenizer(
                p=p,
                d_dist=patch_tokenizer_d_dist,
                d_patch=d_patch,
                hidden_dim=256,
                num_layers=3,
                dropout=0.0,
            )

        if d_model % n_heads != 0:
            raise ValueError(f"big_vae.d_model ({d_model}) must be divisible by big_vae.n_heads ({n_heads})")
        if d_lat % n_heads != 0:
            raise ValueError(f"big_vae.d_lat ({d_lat}) must be divisible by big_vae.n_heads ({n_heads})")

        self.patch_token_proj = nn.Linear(d_patch, d_model)
        self.cls_token = nn.Parameter(torch.zeros(d_model))

        self.encoder_layers = nn.ModuleList(
            [
                LatentEncoderLayer(
                    d_model=d_model,
                    d_lat=d_lat,
                    n_heads=n_heads,
                    ffn_mult=cfg.big_vae.ffn_mult,
                    dropout=dropout,
                    self_attn_mode=cfg.big_vae.encoder.self_attn_mode,
                    use_rope_2d=True,
                )
                for _ in range(num_enc_layers)
            ]
        )
        if self.use_legacy_distribution_encoder_conditioning:
            self.enc_dist_inject_projs: nn.ModuleList | None = nn.ModuleList(
                [nn.Linear(d_model + d_var, d_model) for _ in range(num_enc_layers)]
            )
        else:
            self.enc_dist_inject_projs = None
        self.flat_lat_dim = num_latents * d_lat
        if self.use_legacy_distribution_encoder_conditioning:
            self.enc_dist_to_latent_heads: nn.ModuleList | None = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_var, d_var * 2),
                        nn.GELU(),
                        nn.Linear(d_var * 2, d_var * 2),
                        nn.GELU(),
                        nn.Linear(d_var * 2, self.flat_lat_dim),
                    )
                    for _ in range(num_enc_layers)
                ]
            )
        else:
            self.enc_dist_to_latent_heads = None
        if self.use_token_adapter_distribution_encoder_conditioning:
            self.encoder_conditioning_adapters: nn.ModuleList | None = nn.ModuleList(
                [
                    TokenConditioningAdapter(
                        d_model=d_model,
                        d_y=d_dist,
                        dropout=dropout,
                    )
                    for _ in range(num_enc_layers)
                ]
            )
        else:
            self.encoder_conditioning_adapters = None

        self.latent_base = nn.Parameter(torch.randn(num_latents, d_lat) * 0.02)
        self.z_dim = self.flat_lat_dim
        self.latent_norm = nn.LayerNorm(self.flat_lat_dim)

        self.dec_L_latents = num_latents
        self.latent_to_decoder = nn.Linear(d_lat, d_model)
        self.pos_proj = nn.Linear(2 * cfg.big_vae.pos_fourier_dim, d_model)
        self.query_proj = nn.Linear(query_in_dim, d_model)
        self.query_pos_proj = nn.Linear(d_model, d_model)
        self.decoder_layers = nn.ModuleList(
            [CrossAttnBlock(d_model=d_model, n_heads=n_heads, dropout=dropout, use_rope_2d=True) for _ in range(num_dec_layers)]
        )

        self.q_tokens_norm = nn.LayerNorm(d_model)
        self.direction_head = nn.Linear(d_model, p)
        self.scale_head = nn.Linear(d_model, 1)
        self.debug_query_hint_proj = nn.Linear(d_model, d_model)
        self.debug_encoder_direct_direction_head = nn.Linear(d_model, p)
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0
        self.direction_seq_norm_eps = 1e-6

        self.z_shortcut_proj = nn.Linear(self.flat_lat_dim, d_model)
        self.z_shortcut = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, p),
        )
        if bool(cfg.big_vae.disable_z_shortcut):
            self.z_shortcut_proj.requires_grad_(False)
            self.z_shortcut.requires_grad_(False)

    @staticmethod
    def _normalize_patch_tokenizer_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value in {"residual", "default", "legacy"}:
            return "residual"
        if value in {"conditioned_mlp", "patch_conditioned_mlp"}:
            return "conditioned_mlp"
        raise ValueError(
            "big_vae.patch_tokenizer_kind must be one of "
            "'residual', 'conditioned_mlp', "
            f"got {kind!r}"
        )

    @staticmethod
    def _normalize_distribution_encoder_conditioning_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value not in {"legacy", "token_adapter"}:
            raise ValueError(
                "big_vae.distribution_encoder_conditioning_kind must be one of "
                "'legacy', 'token_adapter', "
                f"got {kind!r}"
            )
        return value

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu.new_zeros(())

    @staticmethod
    def patch_structure_loss(
        W: torch.Tensor,
        W_hat: torch.Tensor,
        patch_size: int,
        eps: float = 1e-8,
        gamma: float = 0.5,
        lambda_dir: float = 1.0,
        lambda_scale: float = 0.25,
        lambda_rec: float = 0.5,
        lambda_rel: float = 0.1,
        huber_delta: float = 0.1,
        pred_dirs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if W.ndim == 2:
            W = W.unsqueeze(0)
            W_hat = W_hat.unsqueeze(0)

        B, d_in, d_out = W.shape
        p = int(patch_size)
        T = d_in // p
        if T <= 0:
            zero = W.new_zeros(())
            return zero, {"L_dir": zero, "L_scale": zero, "L_rec": zero, "L_rel": zero}

        used = T * p
        X = W[:, :used, :].transpose(1, 2).contiguous().view(B, d_out, T, p)
        X_hat = W_hat[:, :used, :].transpose(1, 2).contiguous().view(B, d_out, T, p)

        N = B * d_out
        X = X.reshape(N, T, p)
        X_hat = X_hat.reshape(N, T, p)

        r = X.norm(dim=-1)
        u = X / (r.unsqueeze(-1) + eps)
        log_r = torch.log(r + eps)

        use_dir = float(lambda_dir) != 0.0
        use_scale = float(lambda_scale) != 0.0
        use_rec = float(lambda_rec) != 0.0
        use_rel = float(lambda_rel) != 0.0

        zero = X.new_zeros(())
        L_dir = zero
        L_scale = zero
        L_rec = zero
        L_rel = zero

        r_hat: torch.Tensor | None = None
        u_hat: torch.Tensor | None = None
        log_r_hat: torch.Tensor | None = None

        def _ensure_r_hat() -> torch.Tensor:
            nonlocal r_hat
            if r_hat is None:
                r_hat = X_hat.norm(dim=-1)
            return r_hat

        def _ensure_u_hat() -> torch.Tensor:
            nonlocal u_hat
            if u_hat is None:
                current_r_hat = _ensure_r_hat()
                u_hat = X_hat / (current_r_hat.unsqueeze(-1) + eps)
            return u_hat

        def _ensure_log_r_hat() -> torch.Tensor:
            nonlocal log_r_hat
            if log_r_hat is None:
                log_r_hat = torch.log(_ensure_r_hat() + eps)
            return log_r_hat

        if use_dir:
            if pred_dirs is not None:
                u_hat_dir = pred_dirs[:, :, :T, :].reshape(N, T, p)
            else:
                u_hat_dir = _ensure_u_hat()

            w = (r + eps) ** gamma
            w = w / (w.sum(dim=1, keepdim=True) + eps)
            cos = (u_hat_dir * u).sum(dim=-1)
            L_dir = ((1.0 - cos)).mean(dim=1).mean()

        if use_scale:
            d = _ensure_log_r_hat() - log_r
            abs_d = d.abs()
            huber = torch.where(
                abs_d <= huber_delta,
                0.5 * d.pow(2),
                huber_delta * (abs_d - 0.5 * huber_delta),
            )
            L_scale = huber.mean()

        if use_rec:
            rec_num = (X_hat - X).pow(2).sum(dim=-1)
            rec_den = r.pow(2) + eps
            L_rec = (rec_num / rec_den).mean()

        if use_rel:
            current_u_hat = _ensure_u_hat()
            G = torch.bmm(u, u.transpose(1, 2))
            G_hat = torch.bmm(current_u_hat, current_u_hat.transpose(1, 2))
            L_rel = (G_hat - G).pow(2).mean()

        total = lambda_dir * L_dir + lambda_scale * L_scale + lambda_rec * L_rec + lambda_rel * L_rel
        details = {
            "L_dir": L_dir.detach(),
            "L_scale": L_scale.detach(),
            "L_rec": L_rec.detach(),
            "L_rel": L_rel.detach(),
        }
        return total, details

    @staticmethod
    def operator_recon_loss(X: torch.Tensor, W: torch.Tensor, W_hat: torch.Tensor) -> torch.Tensor:
        d_in = W.shape[-2]
        if X.ndim == 2:
            return (F.mse_loss(X @ W_hat, X @ W) / d_in).sqrt()
        return (F.mse_loss(torch.matmul(X, W_hat), torch.matmul(X, W)) / d_in).sqrt()

    @staticmethod
    def _channel_norm_over_sequence(x: torch.Tensor, eps: float) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,C], got {tuple(x.shape)}")
        if int(x.shape[1]) <= 1:
            return x
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + float(eps))

    @staticmethod
    def _build_patch_indices(d_in: int, patch_size: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
        T = (d_in + patch_size - 1) // patch_size
        d_in_pad = T * patch_size
        patch_idx_t = torch.arange(d_in_pad, device=device, dtype=torch.long).view(T, patch_size)
        patch_idx_t = patch_idx_t.clamp(max=max(0, d_in - 1))
        return patch_idx_t, T, d_in_pad

    def _encode_distribution_context(
        self,
        X: torch.Tensor,
        *,
        d_in: int,
    ) -> tuple[int, int, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if X.ndim != 3:
            raise ValueError(f"X must be rank-3 [B, n, d_in], got {tuple(X.shape)}")

        B, n, d_in_x = X.shape
        if d_in_x != d_in:
            raise ValueError(f"X last dim ({d_in_x}) must match d_in ({d_in})")

        device = X.device
        p = self.cfg.patch_size
        patch_idx_t, T, d_in_pad = self._build_patch_indices(d_in=d_in, patch_size=p, device=device)
        if not self.use_distribution_encoder:
            return T, d_in_pad, None, None, None
        if self.distribution_encoder is None:
            raise RuntimeError("distribution_encoder is not initialized")

        patch_idx_bt = patch_idx_t.unsqueeze(0).expand(B, -1, -1).contiguous()
        X_rep = X.unsqueeze(1).expand(B, T, n, d_in).reshape(B * T, n, d_in)
        patch_idx_flat = patch_idx_bt.reshape(B * T, p)

        dist_var_flat, dist_patch_flat = self.distribution_encoder(X_rep, patch_idx_flat)

        d_var = self.cfg.distribution.d_var
        d_dist = self.cfg.distribution.d_dist
        dist_var_by_patch = dist_var_flat.view(B, T, p, d_var)
        dist_patch_by_patch = dist_patch_flat.view(B, T, d_dist)
        dist_var_pooled = dist_var_by_patch.mean(dim=2)
        return T, d_in_pad, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled

    def _encode_latent_slots(
        self,
        W: torch.Tensor,
        *,
        T: int,
        d_in_pad: int,
        dist_var_by_patch: torch.Tensor | None,
        dist_patch_by_patch: torch.Tensor | None,
        dist_var_pooled: torch.Tensor | None,
        return_debug_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if W.ndim != 3:
            raise ValueError(f"W must be rank-3 [B, d_in, d_out], got {tuple(W.shape)}")

        B, d_in, d_out = W.shape
        p = self.cfg.patch_size
        d_var = self.cfg.distribution.d_var
        d_dist = self.cfg.distribution.d_dist

        if self.use_distribution_encoder:
            if dist_var_by_patch is None or dist_patch_by_patch is None or dist_var_pooled is None:
                raise ValueError("distribution tensors are required when distribution encoder is enabled")
            if tuple(dist_var_by_patch.shape[:3]) != (B, T, p):
                raise ValueError(
                    "dist_var_by_patch must be [B, T, p, d_var], got "
                    f"{tuple(dist_var_by_patch.shape)} for expected {(B, T, p, d_var)}"
                )
            if int(dist_var_by_patch.shape[3]) != d_var:
                raise ValueError(f"dist_var_by_patch last dim must be {d_var}, got {int(dist_var_by_patch.shape[3])}")
            if tuple(dist_patch_by_patch.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_patch_by_patch must be [B, T, d_dist], got "
                    f"{tuple(dist_patch_by_patch.shape)} for expected {(B, T, d_dist)}"
                )
            if int(dist_patch_by_patch.shape[2]) != d_dist:
                raise ValueError(
                    f"dist_patch_by_patch last dim must be {d_dist}, got {int(dist_patch_by_patch.shape[2])}"
                )
            if tuple(dist_var_pooled.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_var_pooled must be [B, T, d_var], got "
                    f"{tuple(dist_var_pooled.shape)} for expected {(B, T, d_var)}"
                )
        elif dist_var_by_patch is not None or dist_patch_by_patch is not None or dist_var_pooled is not None:
            raise ValueError("distribution tensors must be None when distribution encoder is disabled")

        W_pad = torch.zeros(B, d_in_pad, d_out, device=W.device, dtype=W.dtype)
        W_pad[:, :d_in, :] = W
        w_patches = W_pad.transpose(1, 2).contiguous().view(B, d_out, T, p)

        w_flat = w_patches.reshape(B * d_out * T, p)
        if self.use_distribution_encoder:
            assert dist_var_by_patch is not None
            assert dist_patch_by_patch is not None
            assert dist_var_pooled is not None
            dist_patch_expanded = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1)
            dist_var_expanded = dist_var_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1, -1)
            dist_patch_flat_expanded = dist_patch_expanded.reshape(B * d_out * T, d_dist)
            dist_var_flat_expanded: torch.Tensor | None = dist_var_expanded.reshape(B * d_out * T, p, d_var)
            dist_var_for_inject = dist_var_pooled.unsqueeze(1).expand(-1, d_out, -1, -1)
            dist_var_global = dist_var_pooled.mean(dim=1)
        else:
            dist_patch_flat_expanded = w_flat.new_zeros((B * d_out * T, 0))
            dist_var_flat_expanded = None
            dist_var_for_inject = None
            dist_var_global = None

        d_patch = self.cfg.mini_vae.d_patch
        patch_token_raw_flat = self.patch_tokenizer(
            w_patch=w_flat,
            dist_var_tokens=dist_var_flat_expanded,
            dist_patch_embed=dist_patch_flat_expanded,
        )
        patch_token_raw = patch_token_raw_flat.view(B, d_out, T, d_patch)
        patch_tokens = self.patch_token_proj(patch_token_raw)

        cls_tokens = self.cls_token.view(1, 1, 1, -1).expand(B, d_out, 1, -1)
        tokens_by_output = torch.cat([cls_tokens, patch_tokens], dim=2)
        if self.use_token_adapter_distribution_encoder_conditioning:
            assert dist_patch_by_patch is not None
            assert self.encoder_conditioning_adapters is not None
            dist_patch_ctx_by_output = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1).reshape(B * d_out, T, d_dist)
            patch_map_idx = torch.arange(T, device=W.device, dtype=torch.long).view(1, T).expand(B * d_out, -1)
        else:
            dist_patch_ctx_by_output = None
            patch_map_idx = None

        latents = self.latent_base.unsqueeze(0).expand(B, -1, -1)
        num_latents = self.cfg.big_vae.num_latents
        d_lat = self.cfg.big_vae.d_lat

        for layer_idx, enc_layer in enumerate(self.encoder_layers):
            if self.use_legacy_distribution_encoder_conditioning:
                assert dist_var_for_inject is not None
                assert dist_var_global is not None
                assert self.enc_dist_inject_projs is not None
                assert self.enc_dist_to_latent_heads is not None
                cls_part = tokens_by_output[:, :, :1, :]
                patch_part = tokens_by_output[:, :, 1:, :]
                patch_with_dist = torch.cat([patch_part, dist_var_for_inject], dim=-1)
                patch_part = self.enc_dist_inject_projs[layer_idx](patch_with_dist)
                tokens_by_output = torch.cat([cls_part, patch_part], dim=2)

                dist_lat_delta = self.enc_dist_to_latent_heads[layer_idx](dist_var_global)
                latents = latents + dist_lat_delta.view(B, num_latents, d_lat)

            tokens_by_output, latents = enc_layer(
                tokens_by_output=tokens_by_output,
                latents=latents,
                cross_attend_only_cls=self.cfg.big_vae.encoder.cross_attend_only_cls,
                patch_conditioner=(
                    self.encoder_conditioning_adapters[layer_idx]
                    if self.use_token_adapter_distribution_encoder_conditioning and self.encoder_conditioning_adapters is not None
                    else None
                ),
                patch_conditioning_ctx=dist_patch_ctx_by_output,
                patch_conditioning_map_idx=patch_map_idx,
            )

        if return_debug_info:
            return latents, {
                "debug_patch_tokenizer_kind": self.patch_tokenizer_kind,
                "debug_encoder_conditioning_kind": self.distribution_encoder_conditioning_kind,
                "encoder_tokens_by_output": tokens_by_output,
                "encoder_cls_tokens": tokens_by_output[:, :, :1, :],
                "encoder_patch_tokens": tokens_by_output[:, :, 1:, :],
            }
        return latents

    @staticmethod
    def _normalize_debug_decoder_kv_source(source: str) -> str:
        value = str(source).strip().lower()
        if value not in {"latents", "encoder_patch_tokens"}:
            raise ValueError(
                "debug_decoder_kv_source must be 'latents' or 'encoder_patch_tokens', "
                f"got {source!r}"
            )
        return value

    @staticmethod
    def _normalize_debug_query_hint(source: str) -> str:
        value = str(source).strip().lower()
        if value not in {"none", "aligned_encoder_token"}:
            raise ValueError(
                "debug_query_hint must be 'none' or 'aligned_encoder_token', "
                f"got {source!r}"
            )
        return value

    def _build_decoder_query_state(
        self,
        *,
        batch_size: int,
        dist_patch_by_patch: torch.Tensor | None,
        d_out: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_model = int(self.cfg.big_vae.d_model)
        if self.use_distribution_encoder:
            if dist_patch_by_patch is None:
                raise ValueError("dist_patch_by_patch is required when distribution encoder is enabled")
            B = int(dist_patch_by_patch.shape[0])
            device = dist_patch_by_patch.device
        else:
            B = int(batch_size)
            device = self.cls_token.device

        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")

        pos_dim = self.cfg.big_vae.pos_fourier_dim
        pos_o = sinusoidal_embedding(o_grid.to(torch.float32), pos_dim)
        pos_t = sinusoidal_embedding(t_grid.to(torch.float32), pos_dim)
        pos_ot = torch.cat([pos_o, pos_t], dim=-1)

        q_base = self.pos_proj(pos_ot)
        q_pos_emb = self.query_pos_proj(q_base)
        q_base_expanded = q_base.unsqueeze(0).expand(B, -1, -1, -1)
        if self.use_distribution_encoder:
            assert dist_patch_by_patch is not None
            dist_patch_expanded = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1)
            q_inputs = torch.cat([q_base_expanded, dist_patch_expanded], dim=-1)
        else:
            q_inputs = q_base_expanded
        q_tokens = self.query_proj(q_inputs).reshape(B, d_out * T, d_model)
        q_pos_emb_flat = q_pos_emb.reshape(1, d_out * T, d_model).expand(B, -1, -1)
        q_pos_o = o_grid.flatten().to(dtype=torch.float32)
        q_pos_t = t_grid.flatten().to(dtype=torch.float32)
        return q_tokens, q_pos_emb_flat, q_pos_o, q_pos_t

    def _apply_debug_query_hint(
        self,
        q_tokens: torch.Tensor,
        *,
        encoder_patch_tokens: torch.Tensor | None,
        debug_query_hint: str,
    ) -> torch.Tensor:
        hint_source = self._normalize_debug_query_hint(debug_query_hint)
        if hint_source == "none":
            return q_tokens
        if encoder_patch_tokens is None:
            raise ValueError("encoder_patch_tokens are required when debug_query_hint='aligned_encoder_token'")

        B, Q, d_model = q_tokens.shape
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if int(encoder_patch_tokens.shape[0]) != B or int(encoder_patch_tokens.shape[3]) != d_model:
            raise ValueError(
                "encoder_patch_tokens batch/model dims must match q_tokens, got "
                f"{tuple(encoder_patch_tokens.shape)} vs {tuple(q_tokens.shape)}"
            )

        encoder_patch_flat = encoder_patch_tokens.reshape(B, -1, d_model)
        if int(encoder_patch_flat.shape[1]) != Q:
            raise ValueError(
                "encoder_patch_tokens flattened length must match decoder query length, got "
                f"{tuple(encoder_patch_flat.shape)} vs {tuple(q_tokens.shape)}"
            )
        return q_tokens + self.debug_query_hint_proj(encoder_patch_flat)

    def _build_decoder_kv_state(
        self,
        *,
        lat: torch.Tensor,
        encoder_patch_tokens: torch.Tensor | None,
        d_out: int,
        T: int,
        debug_decoder_kv_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv_source = self._normalize_debug_decoder_kv_source(debug_decoder_kv_source)
        if kv_source == "latents":
            kv = lat
            kv_pos = torch.arange(lat.shape[1], device=lat.device, dtype=torch.float32)
            return kv, kv_pos, kv_pos

        if encoder_patch_tokens is None:
            raise ValueError("encoder_patch_tokens are required when debug_decoder_kv_source='encoder_patch_tokens'")
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if tuple(encoder_patch_tokens.shape[1:3]) != (d_out, T):
            raise ValueError(
                "encoder_patch_tokens must match decoder grid [d_out, T], got "
                f"{tuple(encoder_patch_tokens.shape)} vs expected (*, {d_out}, {T}, {self.cfg.big_vae.d_model})"
            )

        device = encoder_patch_tokens.device
        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")
        kv = encoder_patch_tokens.reshape(encoder_patch_tokens.shape[0], d_out * T, encoder_patch_tokens.shape[-1])
        return kv, o_grid.flatten().to(dtype=torch.float32), t_grid.flatten().to(dtype=torch.float32)

    def _decode_query_tokens_to_output(
        self,
        q_tokens: torch.Tensor,
        *,
        z: torch.Tensor,
        q_pos_emb: torch.Tensor,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Q, d_model = q_tokens.shape
        p = self.cfg.patch_size
        expected_d_in_pad = T * p
        if d_in_pad != expected_d_in_pad:
            raise ValueError(f"d_in_pad must equal T*patch_size={expected_d_in_pad}, got {d_in_pad}")

        shortcut_disabled = bool(disable_z_shortcut or self.cfg.big_vae.disable_z_shortcut)
        q_dir = self.q_tokens_norm(q_tokens)
        u_hat_attn = self.direction_head(q_dir)

        if shortcut_disabled:
            u_hat_shortcut = torch.zeros_like(u_hat_attn)
        else:
            z_proj = self.z_shortcut_proj(z)
            z_exp = z_proj.unsqueeze(1).expand(B, Q, -1)
            shortcut_in = torch.cat([z_exp, q_pos_emb], dim=-1)
            u_hat_shortcut = self.z_shortcut(shortcut_in)

        u_hat = (u_hat_attn + u_hat_shortcut).reshape(B * Q, p)
        direction_pre_norms = u_hat.norm(dim=-1)
        s_hat = self.scale_head(q_tokens).reshape(B * Q)

        w_hat_flat = _decode_direction_and_logscale(
            u_hat=u_hat,
            s_hat=s_hat,
            eps=self.output_eps,
            s_min=self.output_s_min,
            s_max=self.output_s_max,
        )

        u_hat_norm = u_hat / (u_hat.norm(dim=-1, keepdim=True) + self.output_eps)
        pred_dirs = u_hat_norm.view(B, d_out, T, p)

        w_hat_patches = w_hat_flat.view(B, d_out, T, p)
        W_hat_pad = w_hat_patches.view(B, d_out, d_in_pad).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]

        outputs = (W_hat, z, pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs

    def _decode_direct_from_encoder_patch_tokens(
        self,
        encoder_patch_tokens: torch.Tensor,
        *,
        z: torch.Tensor,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        return_direction_pre_norms: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if tuple(encoder_patch_tokens.shape[1:3]) != (d_out, T):
            raise ValueError(
                "encoder_patch_tokens must match decoder grid [d_out, T], got "
                f"{tuple(encoder_patch_tokens.shape)} vs expected (*, {d_out}, {T}, {self.cfg.big_vae.d_model})"
            )

        B = int(encoder_patch_tokens.shape[0])
        Q = d_out * T
        p = self.cfg.patch_size
        direct_hidden = encoder_patch_tokens.reshape(B, Q, encoder_patch_tokens.shape[-1])
        u_hat = self.debug_encoder_direct_direction_head(direct_hidden).reshape(B * Q, p)
        direction_pre_norms = u_hat.norm(dim=-1)
        s_hat = torch.zeros(B * Q, device=u_hat.device, dtype=u_hat.dtype)

        w_hat_flat = _decode_direction_and_logscale(
            u_hat=u_hat,
            s_hat=s_hat,
            eps=self.output_eps,
            s_min=self.output_s_min,
            s_max=self.output_s_max,
        )

        u_hat_norm = u_hat / (u_hat.norm(dim=-1, keepdim=True) + self.output_eps)
        pred_dirs = u_hat_norm.view(B, d_out, T, p)

        w_hat_patches = w_hat_flat.view(B, d_out, T, p)
        W_hat_pad = w_hat_patches.view(B, d_out, d_in_pad).transpose(1, 2)
        W_hat = W_hat_pad[:, :d_in, :]

        outputs = (W_hat, z, pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs

    def _decode_from_latent_slots(
        self,
        latent_slots: torch.Tensor,
        *,
        dist_patch_by_patch: torch.Tensor | None,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
        encoder_patch_tokens: torch.Tensor | None = None,
        debug_decoder_kv_source: str = "latents",
        debug_query_hint: str = "none",
        debug_direct_from_encoder_tokens: bool = False,
        return_debug_info: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if latent_slots.ndim != 3:
            raise ValueError(f"latent_slots must be [B, num_latents, d_lat], got {tuple(latent_slots.shape)}")

        B, num_latents, d_lat = latent_slots.shape
        expected_num_latents = int(self.cfg.big_vae.num_latents)
        expected_d_lat = int(self.cfg.big_vae.d_lat)
        if num_latents != expected_num_latents or d_lat != expected_d_lat:
            raise ValueError(
                "latent_slots shape mismatch: "
                f"got {(B, num_latents, d_lat)}, expected (*, {expected_num_latents}, {expected_d_lat})"
            )
        if self.use_distribution_encoder:
            if dist_patch_by_patch is None:
                raise ValueError("dist_patch_by_patch is required when distribution encoder is enabled")
            if tuple(dist_patch_by_patch.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_patch_by_patch must be [B, T, d_dist], got "
                    f"{tuple(dist_patch_by_patch.shape)} for expected {(B, T, self.cfg.distribution.d_dist)}"
                )
        elif dist_patch_by_patch is not None:
            raise ValueError("dist_patch_by_patch must be None when distribution encoder is disabled")

        p = self.cfg.patch_size
        expected_d_in_pad = T * p
        if d_in_pad != expected_d_in_pad:
            raise ValueError(f"d_in_pad must equal T*patch_size={expected_d_in_pad}, got {d_in_pad}")

        z = self.latent_norm(latent_slots.reshape(B, self.flat_lat_dim))
        lat = self.latent_to_decoder(z.view(B, num_latents, d_lat))

        q_tokens_base, q_pos_emb, q_pos_o, q_pos_t = self._build_decoder_query_state(
            batch_size=B,
            dist_patch_by_patch=dist_patch_by_patch,
            d_out=d_out,
            T=T,
        )
        q_tokens = self._apply_debug_query_hint(
            q_tokens_base,
            encoder_patch_tokens=encoder_patch_tokens,
            debug_query_hint=debug_query_hint,
        )

        #DEBUG
        # debug_direct_from_encoder_tokens = True
        if bool(debug_direct_from_encoder_tokens):
            if encoder_patch_tokens is None:
                raise ValueError("encoder_patch_tokens are required when debug_direct_from_encoder_tokens=True")
            outputs = self._decode_direct_from_encoder_patch_tokens(
                encoder_patch_tokens,
                z=z,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
                return_direction_pre_norms=return_direction_pre_norms,
            )
            decoder_kv = None
            kv_pos_o = None
            kv_pos_t = None
        else:
            decoder_kv, kv_pos_o, kv_pos_t = self._build_decoder_kv_state(
                lat=lat,
                encoder_patch_tokens=encoder_patch_tokens,
                d_out=d_out,
                T=T,
                debug_decoder_kv_source=debug_decoder_kv_source,
            )
            q_hidden = q_tokens
            for dec_layer in self.decoder_layers:
                q_hidden = dec_layer(
                    q=q_hidden,
                    kv=decoder_kv,
                    q_pos=q_pos_o,
                    kv_pos=kv_pos_o,
                    q_pos2=q_pos_t,
                    kv_pos2=kv_pos_t,
                )
            outputs = self._decode_query_tokens_to_output(
                q_hidden,
                z=z,
                q_pos_emb=q_pos_emb,
                d_in=d_in,
                d_out=d_out,
                d_in_pad=d_in_pad,
                T=T,
                return_direction_pre_norms=return_direction_pre_norms,
                disable_z_shortcut=disable_z_shortcut,
            )

        if return_debug_info:
            pred_dirs = outputs[2]
            debug_info: dict[str, object] = {
                "debug_patch_tokenizer_kind": self.patch_tokenizer_kind,
                "debug_encoder_conditioning_kind": self.distribution_encoder_conditioning_kind,
                "debug_decoder_kv_source": self._normalize_debug_decoder_kv_source(debug_decoder_kv_source),
                "debug_query_hint": self._normalize_debug_query_hint(debug_query_hint),
                "debug_direct_from_encoder_tokens": bool(debug_direct_from_encoder_tokens),
                "decoder_queries_base": q_tokens_base,
                "decoder_queries_init": q_tokens,
                "decoder_query_pos_emb": q_pos_emb,
                "decoder_kv": decoder_kv,
                "decoder_kv_pos_o": kv_pos_o,
                "decoder_kv_pos_t": kv_pos_t,
                "pred_dirs": pred_dirs,
            }
            return outputs + (debug_info,)
        return outputs

    def forward_debug(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        *,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
        debug_decoder_kv_source: str = "latents",
        debug_query_hint: str = "none",
        debug_direct_from_encoder_tokens: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if W.ndim not in {2, 3}:
            raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
        if X.ndim not in {2, 3}:
            raise ValueError(f"X must be rank-2 or rank-3, got {tuple(X.shape)}")

        squeeze_batch = W.ndim == 2
        if squeeze_batch != (X.ndim == 2):
            raise ValueError(f"W and X must be both batched or both unbatched, got W={tuple(W.shape)}, X={tuple(X.shape)}")

        if squeeze_batch:
            W = W.unsqueeze(0)
            X = X.unsqueeze(0)
        B, d_in, d_out = W.shape
        Bx, _, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        T, d_in_pad, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled = self._encode_distribution_context(
            X,
            d_in=d_in,
        )
        latents, encode_debug = self._encode_latent_slots(
            W,
            T=T,
            d_in_pad=d_in_pad,
            dist_var_by_patch=dist_var_by_patch,
            dist_patch_by_patch=dist_patch_by_patch,
            dist_var_pooled=dist_var_pooled,
            return_debug_info=True,
        )
        decode_outputs = self._decode_from_latent_slots(
            latents,
            dist_patch_by_patch=dist_patch_by_patch,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            T=T,
            return_direction_pre_norms=return_direction_pre_norms,
            disable_z_shortcut=disable_z_shortcut,
            encoder_patch_tokens=encode_debug["encoder_patch_tokens"],
            debug_decoder_kv_source=debug_decoder_kv_source,
            debug_query_hint=debug_query_hint,
            debug_direct_from_encoder_tokens=debug_direct_from_encoder_tokens,
            return_debug_info=True,
        )
        if return_direction_pre_norms:
            W_hat, z, pred_dirs, direction_pre_norms, decode_debug = decode_outputs
        else:
            W_hat, z, pred_dirs, decode_debug = decode_outputs

        debug_info = {
            "T": int(T),
            "d_in_pad": int(d_in_pad),
            "dist_patch_by_patch": dist_patch_by_patch,
            "encoder_tokens_by_output": encode_debug["encoder_tokens_by_output"],
            "encoder_cls_tokens": encode_debug["encoder_cls_tokens"],
            "encoder_patch_tokens": encode_debug["encoder_patch_tokens"],
            **decode_debug,
        }

        if squeeze_batch:
            outputs = (
                W_hat.squeeze(0),
                z.squeeze(0),
                z.new_zeros(self.z_dim),
                pred_dirs.squeeze(0),
            )
            if return_direction_pre_norms:
                return outputs + (direction_pre_norms, debug_info)
            return outputs + (debug_info,)

        outputs = (W_hat, z, z.new_zeros(B, self.z_dim), pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms, debug_info)
        return outputs + (debug_info,)

    def forward(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        *,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if W.ndim not in {2, 3}:
            raise ValueError(f"W must be rank-2 or rank-3, got {tuple(W.shape)}")
        if X.ndim not in {2, 3}:
            raise ValueError(f"X must be rank-2 or rank-3, got {tuple(X.shape)}")

        squeeze_batch = W.ndim == 2
        if squeeze_batch != (X.ndim == 2):
            raise ValueError(f"W and X must be both batched or both unbatched, got W={tuple(W.shape)}, X={tuple(X.shape)}")

        # W = W[0].repeat(W.shape[0], 1, 1)
        # X = X[0].repeat(X.shape[0], 1, 1)

        if squeeze_batch:
            W = W.unsqueeze(0)
            X = X.unsqueeze(0)
        B, d_in, d_out = W.shape
        Bx, _, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")

        T, d_in_pad, dist_var_by_patch, dist_patch_by_patch, dist_var_pooled = self._encode_distribution_context(
            X,
            d_in=d_in,
        )
        latents, encode_debug = self._encode_latent_slots(
            W,
            T=T,
            d_in_pad=d_in_pad,
            dist_var_by_patch=dist_var_by_patch,
            dist_patch_by_patch=dist_patch_by_patch,
            dist_var_pooled=dist_var_pooled,
            return_debug_info=True
        )
        decode_outputs = self._decode_from_latent_slots(
            latents,
            dist_patch_by_patch=dist_patch_by_patch,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            encoder_patch_tokens=encode_debug["encoder_patch_tokens"],
            T=T,
            return_direction_pre_norms=return_direction_pre_norms,
            disable_z_shortcut=disable_z_shortcut,
        )
        if return_direction_pre_norms:
            W_hat, z, pred_dirs, direction_pre_norms = decode_outputs
        else:
            W_hat, z, pred_dirs = decode_outputs

        if squeeze_batch:
            outputs = (
                W_hat.squeeze(0),
                z.squeeze(0),
                z.new_zeros(self.z_dim),
                pred_dirs.squeeze(0),
            )
            if return_direction_pre_norms:
                return outputs + (direction_pre_norms,)
            return outputs

        outputs = (W_hat, z, z.new_zeros(B, self.z_dim), pred_dirs)
        if return_direction_pre_norms:
            return outputs + (direction_pre_norms,)
        return outputs


WeightQuantileVAE = BigWeightVAE


def smoke_test_big_weight_vae() -> None:
    torch.manual_seed(0)

    p = 16
    num_latents = 16
    d_lat = 256
    z_dim_expected = num_latents * d_lat

    cfg = ModelConfig(
        patch_size=p,
        distribution=DistributionConfig(
            k_s=16,
            Kq=32,
            d_var=128,
            d_dist=128,
            use_covariance=True,
            patch_size_for_cov=p,
        ),
        mini_vae=MiniVAEConfig(
            z_dim=64,
            d_e=128,
            encoder_latent_dim=8,
            num_attn_layers_encoder=2,
            num_layers_decoder=2,
            n_heads=4,
            d_patch=64,
            mlp_stub_hidden_dim=256,
        ),
        big_vae=BigVAEConfig(
            d_model=256,
            d_lat=d_lat,
            num_latents=num_latents,
            num_encoder_layers=2,
            num_decoder_layers=2,
            n_heads=8,
            use_latent_sampling=False,
            encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
        ),
    )

    model = BigWeightVAE(cfg)
    B, n, d_in, d_out = 2, 32, 65, 7
    X = torch.randn(B, n, d_in)
    W = torch.randn(B, d_in, d_out)

    W_hat, z, logvar_dummy, pred_dirs = model(W, X)

    assert tuple(W_hat.shape) == (B, d_in, d_out), f"W_hat shape mismatch: {tuple(W_hat.shape)}"
    assert tuple(z.shape) == (B, z_dim_expected), f"z shape mismatch: {tuple(z.shape)}, expected {(B, z_dim_expected)}"
    assert (logvar_dummy == 0).all(), "logvar dummy should be all zeros"

    behavioral_loss = BigWeightVAE.operator_recon_loss(X, W, W_hat)
    structural_loss, struct_details = BigWeightVAE.patch_structure_loss(W, W_hat, patch_size=cfg.patch_size, pred_dirs=pred_dirs)
    kl_loss = BigWeightVAE.kl_loss(z, logvar_dummy)
    assert kl_loss.item() == 0.0, f"kl_loss should be 0, got {kl_loss.item()}"
    total_loss = behavioral_loss + 0.5 * structural_loss
    total_loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

    print("smoke_test_big_weight_vae PASSED")
    print(
        f"  W_hat: {tuple(W_hat.shape)}, z: {tuple(z.shape)}, "
        f"behavioral={behavioral_loss.item():.4f}, structural={structural_loss.item():.4f}"
    )
    print(f"  struct details: " + ", ".join(f"{k}={v.item():.4f}" for k, v in struct_details.items()))


__all__ = [
    "BigVAEConfig",
    "BigWeightVAE",
    "DecoderCrossBlock",
    "EncoderConfig",
    "LatentEncoderLayer",
    "LocalOutputSelfAttentionBlock",
    "ModelConfig",
    "ResamplerConfig",
    "TTMMemoryConfig",
    "WeightQuantileVAE",
    "smoke_test_big_weight_vae",
    "MixerPatchTokenizer",
    "ResidualPatchTokenizer",
]
