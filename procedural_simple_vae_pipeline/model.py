from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.big_weight_vae import LocalOutputSelfAttentionBlock, ModelConfig
from models.distribution_encoder import InputDistributionEncodingModule
from models.vae_shared import (
    CrossAttnBlock,
    PerceiverResamplerBlock,
    TTMMemoryBlock,
    TTMMemoryStack,
    _decode_direction_and_logscale,
    sinusoidal_embedding,
)

PROCEDURAL_SIMPLE_USE_DISTRIBUTION_CONDITIONING = False
PROCEDURAL_SIMPLE_DEFAULT_ENCODER_CONDITIONING_KIND = "token_adapter"
PROCEDURAL_SIMPLE_DEFAULT_PATCH_TOKENIZER_KIND = "linear"
PROCEDURAL_SIMPLE_DEFAULT_LATENT_BOTTLENECK_KIND = "ttm"


class SlotAttentionBottleneckBlock(nn.Module):
    """Slot-attention refinement over encoder patch tokens."""

    def __init__(self, d_slot: int, d_token: int, dropout: float) -> None:
        super().__init__()
        self.d_slot = int(d_slot)
        self.norm_tokens = nn.LayerNorm(d_token)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_mlp = nn.LayerNorm(d_slot)
        self.to_k = nn.Linear(d_token, d_slot, bias=False)
        self.to_v = nn.Linear(d_token, d_slot, bias=False)
        self.to_q = nn.Linear(d_slot, d_slot, bias=False)
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, 4 * d_slot),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_slot, d_slot),
            nn.Dropout(dropout),
        )
        self.scale = float(d_slot) ** -0.5
        self.eps = 1e-8

    def forward(self, slots: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3:
            raise ValueError(f"slots must be [B, num_slots, d_slot], got {tuple(slots.shape)}")
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B, num_tokens, d_token], got {tuple(tokens.shape)}")

        B, num_slots, d_slot = slots.shape
        if d_slot != self.d_slot:
            raise ValueError(f"slots last dim ({d_slot}) must equal d_slot ({self.d_slot})")
        if int(tokens.shape[0]) != B:
            raise ValueError(f"slots batch ({B}) must match tokens batch ({int(tokens.shape[0])})")

        token_inputs = self.norm_tokens(tokens)
        slot_inputs = self.norm_slots(slots)
        k = self.to_k(token_inputs)
        v = self.to_v(token_inputs)
        q = self.to_q(slot_inputs)

        attn_logits = torch.einsum("bnd,bkd->bnk", k, q) * self.scale
        attn = torch.softmax(attn_logits, dim=-1) + self.eps
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(self.eps)
        updates = torch.einsum("bnk,bnd->bkd", attn, v)

        slots = self.gru(
            updates.reshape(B * num_slots, self.d_slot),
            slots.reshape(B * num_slots, self.d_slot),
        ).view(B, num_slots, self.d_slot)
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


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


def _zero_init_last_linear(module: nn.Module) -> None:
    last_linear: nn.Linear | None = None
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            last_linear = submodule
            break
    if last_linear is None:
        return
    nn.init.zeros_(last_linear.weight)
    if last_linear.bias is not None:
        nn.init.zeros_(last_linear.bias)


class ConditionedMLPBlock(nn.Module):
    """Conditioned residual MLP block for patch-token refinement."""

    def __init__(
        self,
        *,
        d_model: int,
        d_hidden: int,
        d_cond: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        cond_hidden = max(int(d_cond), int(d_model))
        self.cond = nn.Sequential(
            nn.Linear(d_cond, cond_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cond_hidden, 3 * d_model),
        )
        self.alpha = nn.Parameter(torch.zeros(d_model))
        _zero_init_last_linear(self.cond)

    def forward(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if u.ndim != 2:
            raise ValueError(f"u must be [B, D], got {tuple(u.shape)}")
        if c.ndim != 2:
            raise ValueError(f"c must be [B, Dc], got {tuple(c.shape)}")
        if int(u.shape[0]) != int(c.shape[0]):
            raise ValueError(f"u and c batch must match, got {tuple(u.shape)} vs {tuple(c.shape)}")

        h = self.norm(u)
        abg = self.cond(c)
        a, b, g = abg.chunk(3, dim=-1)
        h_mod = (1.0 + a) * h + b
        delta = self.fc2(F.gelu(self.fc1(h_mod)))
        return u + self.alpha * torch.sigmoid(g) * delta


class PatchConditionedTokenizer(nn.Module):
    """Patch token encoder with a Y-only base path and X-conditioned MLP refinement."""

    def __init__(
        self,
        *,
        p: int,
        d_model: int,
        d_x: int,
        dropout: float,
        d_hidden: int | None = None,
        d_c_proj: int | None = None,
        d_c: int | None = None,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_hidden) if d_hidden is not None else int(4 * d_model)
        cond_proj_dim = int(d_c_proj) if d_c_proj is not None else min(64, max(16, d_model // 8))
        cond_dim = int(d_c) if d_c is not None else max(32, d_model // 4)

        self.y_encoder = nn.Sequential(
            nn.Linear(p, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.x_proj = nn.Linear(d_x, cond_proj_dim)
        x_hidden_dim = max(hidden_dim, cond_dim)
        self.x_encoder = nn.Sequential(
            nn.Linear(p * cond_proj_dim, x_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(x_hidden_dim, cond_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ConditionedMLPBlock(
                    d_model=d_model,
                    d_hidden=hidden_dim,
                    d_cond=cond_dim,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_blocks)))
            ]
        )

    def forward(
        self,
        y_patch: torch.Tensor,
        x_patch: torch.Tensor,
    ) -> torch.Tensor:
        if y_patch.ndim != 2:
            raise ValueError(f"y_patch must be [B, P], got {tuple(y_patch.shape)}")
        if x_patch.ndim != 3:
            raise ValueError(f"x_patch must be [B, P, Dx], got {tuple(x_patch.shape)}")
        if int(y_patch.shape[0]) != int(x_patch.shape[0]) or int(y_patch.shape[1]) != int(x_patch.shape[1]):
            raise ValueError(
                "y_patch and x_patch must agree on [B, P], got "
                f"{tuple(y_patch.shape)} vs {tuple(x_patch.shape)}"
            )

        u = self.y_encoder(y_patch)
        x_small = self.x_proj(x_patch)
        c = self.x_encoder(x_small.flatten(start_dim=1))
        for block in self.blocks:
            u = block(u, c)
        return u


class ProceduralSimpleBigWeightVAE(nn.Module):
    """
    Isolated clone of SimpleDirectBigWeightVAE for procedural-weight experiments.

    This starts as a direct clone so it can diverge safely without touching the
    shared model used by the existing big VAE pipeline.
    """

    @staticmethod
    def _normalize_latent_bottleneck_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value not in {"disabled", "perceiver_resampler", "slot_attention", "ttm"}:
            raise ValueError(
                "big_vae.latent_bottleneck_kind must be one of "
                "'disabled', 'perceiver_resampler', 'slot_attention', 'ttm', "
                f"got {kind!r}"
            )
        return value

    @staticmethod
    def _normalize_encoder_conditioning_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value not in {"concat", "token_adapter"}:
            raise ValueError(
                "encoder_conditioning_kind must be one of "
                "'concat', 'token_adapter', "
                f"got {kind!r}"
            )
        return value

    @staticmethod
    def _normalize_patch_tokenizer_kind(kind: str) -> str:
        value = str(kind).strip().lower()
        if value not in {"linear", "patch_conditioner"}:
            raise ValueError(
                "patch_tokenizer_kind must be one of "
                "'linear', 'patch_conditioner', "
                f"got {kind!r}"
            )
        return value

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        p = int(cfg.patch_size)
        d_model = int(cfg.big_vae.d_model)
        d_lat = int(cfg.big_vae.d_lat)
        n_heads = int(cfg.big_vae.n_heads)
        num_enc_layers = max(1, int(cfg.big_vae.num_encoder_layers))
        num_dec_layers = max(1, int(cfg.big_vae.num_decoder_layers))
        num_latents = int(cfg.big_vae.num_latents)
        dropout = float(cfg.big_vae.dropout)
        d_dist = int(cfg.distribution.d_dist)

        if d_model % n_heads != 0:
            raise ValueError(f"big_vae.d_model ({d_model}) must be divisible by big_vae.n_heads ({n_heads})")

        self.use_distribution_conditioning = bool(PROCEDURAL_SIMPLE_USE_DISTRIBUTION_CONDITIONING)
        self.encoder_conditioning_kind = self._normalize_encoder_conditioning_kind(
            getattr(cfg.big_vae, "encoder_conditioning_kind", PROCEDURAL_SIMPLE_DEFAULT_ENCODER_CONDITIONING_KIND)
        )
        self.patch_tokenizer_kind = self._normalize_patch_tokenizer_kind(
            getattr(cfg.big_vae, "patch_tokenizer_kind", PROCEDURAL_SIMPLE_DEFAULT_PATCH_TOKENIZER_KIND)
        )
        self.use_concat_encoder_conditioning = bool(
            self.use_distribution_conditioning and self.encoder_conditioning_kind == "concat"
        )
        self.use_token_adapter_encoder_conditioning = bool(
            self.use_distribution_conditioning and self.encoder_conditioning_kind == "token_adapter"
        )
        self.use_patch_conditioner_tokenizer = self.patch_tokenizer_kind == "patch_conditioner"
        self.latent_bottleneck_kind = self._normalize_latent_bottleneck_kind(
            getattr(cfg.big_vae, "latent_bottleneck_kind", PROCEDURAL_SIMPLE_DEFAULT_LATENT_BOTTLENECK_KIND)
        )
        self.use_latent_bottleneck = self.latent_bottleneck_kind != "disabled"
        self.use_slot_attention_latent_bottleneck = self.latent_bottleneck_kind == "slot_attention"
        self.use_ttm_latent_bottleneck = self.latent_bottleneck_kind == "ttm"
        if self.use_latent_bottleneck and d_lat % n_heads != 0:
            raise ValueError(f"big_vae.d_lat ({d_lat}) must be divisible by big_vae.n_heads ({n_heads})")

        if self.use_patch_conditioner_tokenizer and not self.use_distribution_conditioning:
            raise ValueError(
                "patch_tokenizer_kind='patch_conditioner' requires "
                "PROCEDURAL_SIMPLE_USE_DISTRIBUTION_CONDITIONING=True"
            )
        if self.use_patch_conditioner_tokenizer and self.use_concat_encoder_conditioning:
            raise ValueError(
                "patch_tokenizer_kind='patch_conditioner' is incompatible with "
                "encoder_conditioning_kind='concat'; use 'token_adapter' instead"
            )

        if self.use_patch_conditioner_tokenizer:
            self.patch_tokenizer = PatchConditionedTokenizer(
                p=p,
                d_model=d_model,
                d_x=int(cfg.distribution.d_var),
                dropout=dropout,
            )
        else:
            self.patch_tokenizer = nn.Linear(p + (d_dist if self.use_concat_encoder_conditioning else 0), d_model)
        if self.use_distribution_conditioning:
            self.distribution_encoder = InputDistributionEncodingModule(cfg.distribution)
            query_in_dim = d_model + d_dist
        else:
            self.distribution_encoder = None
            query_in_dim = d_model
        if self.use_token_adapter_encoder_conditioning:
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
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.encoder_layers = nn.ModuleList(
            [
                LocalOutputSelfAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_mult=float(cfg.big_vae.ffn_mult),
                    dropout=dropout,
                    self_attn_mode=cfg.big_vae.encoder.self_attn_mode,
                )
                for _ in range(num_enc_layers)
            ]
        )

        self.dec_L_latents = num_latents
        self.z_dim = int(num_latents * d_lat)
        self.flat_lat_dim = self.z_dim
        if self.use_latent_bottleneck:
            if self.use_slot_attention_latent_bottleneck:
                self.latent_base = nn.Parameter(torch.randn(num_latents, d_lat) * 0.02)
                self.latent_resampler_layers = None
                self.slot_attention_token_pos_proj: nn.Linear | None = nn.Linear(
                    2 * int(cfg.big_vae.pos_fourier_dim),
                    d_model,
                )
                self.latent_slot_attention_layers: nn.ModuleList | None = nn.ModuleList(
                    [
                        SlotAttentionBottleneckBlock(
                            d_slot=d_lat,
                            d_token=d_model,
                            dropout=dropout,
                        )
                        for _ in range(num_enc_layers)
                    ]
                )
                self.latent_ttm_input_proj = None
                self.latent_ttm_token_pos_proj = None
                self.latent_ttm_stack = None
            else:
                self.slot_attention_token_pos_proj = None
                self.latent_slot_attention_layers = None
                if self.use_ttm_latent_bottleneck:
                    ttm_cfg = cfg.big_vae.ttm
                    self.register_parameter("latent_base", None)
                    self.latent_resampler_layers = None
                    self.latent_ttm_input_proj: nn.Linear | None = nn.Linear(d_model, d_lat)
                    self.latent_ttm_token_pos_proj: nn.Linear | None
                    if bool(ttm_cfg.use_positional_embeddings):
                        self.latent_ttm_token_pos_proj = nn.Linear(2 * int(cfg.big_vae.pos_fourier_dim), d_lat)
                    else:
                        self.latent_ttm_token_pos_proj = None
                    ttm_block = TTMMemoryBlock(
                        dim=d_lat,
                        mem_tokens=num_latents,
                        proc_tokens=int(ttm_cfg.proc_tokens),
                        process_depth=int(ttm_cfg.process_depth),
                        num_heads=n_heads,
                        summarizer_mode=str(ttm_cfg.summarizer_mode),
                        summarizer_hidden_mult=float(ttm_cfg.summarizer_hidden_mult),
                        dropout=float(ttm_cfg.dropout),
                        use_type_embeddings=bool(ttm_cfg.use_type_embeddings),
                        use_positional_embeddings=bool(ttm_cfg.use_positional_embeddings),
                        memory_init=str(ttm_cfg.memory_init),
                        return_aux=bool(ttm_cfg.return_aux),
                    )
                    self.latent_ttm_stack: TTMMemoryStack | None = TTMMemoryStack(
                        num_blocks=max(1, int(ttm_cfg.num_blocks)),
                        block=ttm_block,
                        share_weights=bool(ttm_cfg.share_weights),
                    )
                else:
                    self.latent_base = nn.Parameter(torch.randn(num_latents, d_lat) * 0.02)
                    self.latent_resampler_layers = nn.ModuleList(
                        [
                            PerceiverResamplerBlock(
                                d_latent=d_lat,
                                d_token=d_model,
                                n_heads=n_heads,
                                dropout=dropout,
                                use_rope_2d=True,
                            )
                            for _ in range(num_enc_layers)
                        ]
                    )
                    self.latent_ttm_input_proj = None
                    self.latent_ttm_token_pos_proj = None
                    self.latent_ttm_stack = None
            self.latent_norm: nn.LayerNorm | None = nn.LayerNorm(self.flat_lat_dim)
        else:
            self.register_parameter("latent_base", None)
            self.latent_resampler_layers = None
            self.slot_attention_token_pos_proj = None
            self.latent_slot_attention_layers = None
            self.latent_ttm_input_proj = None
            self.latent_ttm_token_pos_proj = None
            self.latent_ttm_stack = None
            self.latent_norm = None

        self.latent_to_decoder = nn.Linear(d_lat, d_model)
        self.pos_proj = nn.Linear(2 * int(cfg.big_vae.pos_fourier_dim), d_model)
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
    def _build_patch_geometry(
        d_in: int,
        patch_size: int,
    ) -> tuple[int, int]:
        T = (int(d_in) + int(patch_size) - 1) // int(patch_size)
        d_in_pad = T * int(patch_size)
        return T, d_in_pad

    def _encode_distribution_context(
        self,
        X: torch.Tensor,
        *,
        d_in: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_distribution_conditioning or self.distribution_encoder is None:
            raise RuntimeError("distribution conditioning is disabled for ProceduralSimpleBigWeightVAE")
        if X.ndim != 3:
            raise ValueError(f"X must be rank-3 [B, n, d_in], got {tuple(X.shape)}")

        B, n, d_in_x = X.shape
        if d_in_x != d_in:
            raise ValueError(f"X last dim ({d_in_x}) must match d_in ({d_in})")

        p = int(self.cfg.patch_size)
        T, d_in_pad = self._build_patch_geometry(d_in=d_in, patch_size=p)
        del d_in_pad
        patch_idx_t = torch.arange(T * p, device=X.device, dtype=torch.long).view(T, p)
        patch_idx_t = patch_idx_t.clamp(max=max(0, d_in - 1))
        patch_idx_bt = patch_idx_t.unsqueeze(0).expand(B, -1, -1).contiguous()
        X_rep = X.unsqueeze(1).expand(B, T, n, d_in).reshape(B * T, n, d_in)
        patch_idx_flat = patch_idx_bt.reshape(B * T, p)

        dist_var_flat, dist_patch_flat = self.distribution_encoder(X_rep, patch_idx_flat)
        return dist_var_flat.view(B, T, p, -1), dist_patch_flat.view(B, T, -1)

    def _encode_patch_tokens(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
        if W.ndim != 3:
            raise ValueError(f"W must be rank-3 [B, d_in, d_out], got {tuple(W.shape)}")
        if X.ndim != 3:
            raise ValueError(f"X must be rank-3 [B, n, d_in], got {tuple(X.shape)}")

        B, d_in, d_out = W.shape
        Bx, _, d_in_x = X.shape
        if Bx != B or d_in_x != d_in:
            raise ValueError(f"Shape mismatch: W={tuple(W.shape)}, X={tuple(X.shape)}")
        p = int(self.cfg.patch_size)
        T, d_in_pad = self._build_patch_geometry(d_in=d_in, patch_size=p)

        W_pad = W.new_zeros((B, d_in_pad, d_out))
        W_pad[:, :d_in, :] = W
        w_patches = W_pad.transpose(1, 2).contiguous().view(B, d_out, T, p)

        dist_var_by_patch: torch.Tensor | None = None
        dist_patch_by_patch: torch.Tensor | None = None
        patch_token_inputs = w_patches
        if self.use_distribution_conditioning:
            dist_var_by_patch, dist_patch_by_patch = self._encode_distribution_context(X, d_in=d_in)
            if self.use_concat_encoder_conditioning:
                dist_patch_expanded = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1)
                patch_token_inputs = torch.cat([w_patches, dist_patch_expanded], dim=-1)

        if self.use_patch_conditioner_tokenizer:
            if dist_var_by_patch is None:
                raise RuntimeError("patch_conditioner patch tokenizer requires dist_var_by_patch")
            dist_var_expanded = dist_var_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1, -1)
            patch_tokens = self.patch_tokenizer(
                y_patch=w_patches.reshape(B * d_out * T, p),
                x_patch=dist_var_expanded.reshape(B * d_out * T, p, dist_var_by_patch.shape[-1]),
            )
        else:
            patch_tokens = self.patch_tokenizer(patch_token_inputs.reshape(B * d_out * T, patch_token_inputs.shape[-1]))
        patch_tokens = patch_tokens.view(B, d_out, T, -1)
        cls_tokens = self.cls_token.view(1, 1, 1, -1).expand(B, d_out, 1, -1)
        tokens_by_output = torch.cat([cls_tokens, patch_tokens], dim=2)
        if self.use_token_adapter_encoder_conditioning:
            if dist_patch_by_patch is None or self.encoder_conditioning_adapters is None:
                raise RuntimeError("token-adapter conditioning requires distribution context and adapter modules")
            dist_ctx_by_output = dist_patch_by_patch.unsqueeze(1).expand(-1, d_out, -1, -1).reshape(B * d_out, T, -1)
            map_idx = torch.arange(T, device=W.device, dtype=torch.long).view(1, T).expand(B * d_out, -1)
        else:
            dist_ctx_by_output = None
            map_idx = None

        for layer_idx, enc_layer in enumerate(self.encoder_layers):
            local_in = tokens_by_output.reshape(B * d_out, T + 1, tokens_by_output.shape[-1])
            local_out = enc_layer(local_in)
            if self.use_token_adapter_encoder_conditioning:
                assert dist_ctx_by_output is not None
                assert map_idx is not None
                assert self.encoder_conditioning_adapters is not None
                patch_out = self.encoder_conditioning_adapters[layer_idx](
                    h=local_out[:, 1:, :],
                    y_ctx=dist_ctx_by_output,
                    map_idx=map_idx,
                )
                local_out = torch.cat([local_out[:, :1, :], patch_out], dim=1)
            tokens_by_output = local_out.reshape(B, d_out, T + 1, tokens_by_output.shape[-1])

        encoder_patch_tokens = tokens_by_output[:, :, 1:, :]
        return tokens_by_output, encoder_patch_tokens, T, d_in_pad, dist_patch_by_patch

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

    def _resolve_decoder_kv_source(self, source: str) -> str:
        kv_source = self._normalize_debug_decoder_kv_source(source)
        if kv_source == "latents" and not self.use_latent_bottleneck:
            return "encoder_patch_tokens"
        return kv_source

    def _flatten_encoder_patch_tokens(
        self,
        encoder_patch_tokens: torch.Tensor,
        *,
        d_out: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_patch_tokens.ndim != 4:
            raise ValueError(
                "encoder_patch_tokens must be [B, d_out, T, d_model], got "
                f"{tuple(encoder_patch_tokens.shape)}"
            )
        if tuple(encoder_patch_tokens.shape[1:3]) != (d_out, T):
            raise ValueError(
                "encoder_patch_tokens must match decoder grid [d_out, T], got "
                f"{tuple(encoder_patch_tokens.shape)} vs expected (*, {d_out}, {T}, {encoder_patch_tokens.shape[-1]})"
            )

        device = encoder_patch_tokens.device
        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")
        patch_flat = encoder_patch_tokens.reshape(encoder_patch_tokens.shape[0], d_out * T, encoder_patch_tokens.shape[-1])
        return patch_flat, o_grid.flatten().to(dtype=torch.float32), t_grid.flatten().to(dtype=torch.float32)

    def _encode_latent_slots_from_patch_tokens(
        self,
        encoder_patch_tokens: torch.Tensor,
        *,
        d_out: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object] | None]:
        if not self.use_latent_bottleneck:
            raise RuntimeError("latent bottleneck is disabled for ProceduralSimpleBigWeightVAE")
        if self.latent_norm is None:
            raise RuntimeError("latent bottleneck modules are not initialized")

        B = int(encoder_patch_tokens.shape[0])
        patch_flat, token_pos_o, token_pos_t = self._flatten_encoder_patch_tokens(
            encoder_patch_tokens,
            d_out=d_out,
            T=T,
        )
        ttm_aux: dict[str, object] | None = None
        if self.use_ttm_latent_bottleneck:
            # TTM sees only the final encoder patch embeddings, not intermediate encoder states.
            if self.latent_ttm_input_proj is None or self.latent_ttm_stack is None:
                raise RuntimeError("TTM bottleneck modules are not initialized")
            pos_dim = int(self.cfg.big_vae.pos_fourier_dim)
            ttm_inputs = self.latent_ttm_input_proj(patch_flat)
            input_pos: torch.Tensor | None = None
            if self.latent_ttm_token_pos_proj is not None:
                pos_o = sinusoidal_embedding(token_pos_o, pos_dim)
                pos_t = sinusoidal_embedding(token_pos_t, pos_dim)
                token_pos_features = torch.cat([pos_o, pos_t], dim=-1)
                input_pos = self.latent_ttm_token_pos_proj(token_pos_features).unsqueeze(0)
            ttm_out = self.latent_ttm_stack(
                ttm_inputs,
                M=None,
                input_pos=input_pos,
            )
            if isinstance(ttm_out, tuple):
                latents, ttm_aux = ttm_out
            else:
                latents = ttm_out
        else:
            if self.latent_base is None:
                raise RuntimeError("latent bottleneck seed is not initialized")
            latents = self.latent_base.unsqueeze(0).expand(B, -1, -1)
            latent_pos = (
                (torch.arange(self.dec_L_latents, device=encoder_patch_tokens.device, dtype=torch.float32) + 0.5)
                / max(float(self.dec_L_latents), 1.0)
            )
            if self.use_slot_attention_latent_bottleneck:
                if self.latent_slot_attention_layers is None or self.slot_attention_token_pos_proj is None:
                    raise RuntimeError("slot attention bottleneck modules are not initialized")
                pos_dim = int(self.cfg.big_vae.pos_fourier_dim)
                pos_o = sinusoidal_embedding(token_pos_o, pos_dim)
                pos_t = sinusoidal_embedding(token_pos_t, pos_dim)
                token_pos_features = torch.cat([pos_o, pos_t], dim=-1)
                slot_tokens = patch_flat + self.slot_attention_token_pos_proj(token_pos_features).unsqueeze(0)
                for bottleneck_layer in self.latent_slot_attention_layers:
                    latents = bottleneck_layer(
                        slots=latents,
                        tokens=slot_tokens,
                    )
            else:
                if self.latent_resampler_layers is None:
                    raise RuntimeError("perceiver bottleneck modules are not initialized")
                for bottleneck_layer in self.latent_resampler_layers:
                    latents = bottleneck_layer(
                        latents=latents,
                        tokens=patch_flat,
                        latent_pos=latent_pos,
                        token_pos=token_pos_o,
                        token_pos2=token_pos_t,
                    )

        z = self.latent_norm(latents.reshape(B, self.flat_lat_dim))
        decoder_latents = self.latent_to_decoder(z.view(B, self.dec_L_latents, int(self.cfg.big_vae.d_lat)))
        return latents, z, decoder_latents, ttm_aux

    def _build_decoder_query_state(
        self,
        *,
        batch_size: int,
        dist_patch_by_patch: torch.Tensor | None,
        d_out: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_model = int(self.cfg.big_vae.d_model)
        if self.use_distribution_conditioning:
            if dist_patch_by_patch is None:
                raise ValueError("dist_patch_by_patch is required when distribution conditioning is enabled")
            B = int(dist_patch_by_patch.shape[0])
            device = dist_patch_by_patch.device
        else:
            B = int(batch_size)
            device = self.cls_token.device

        o_idx = torch.arange(d_out, device=device)
        t_idx = torch.arange(T, device=device)
        o_grid, t_grid = torch.meshgrid(o_idx, t_idx, indexing="ij")

        pos_dim = int(self.cfg.big_vae.pos_fourier_dim)
        pos_o = sinusoidal_embedding(o_grid.to(torch.float32), pos_dim)
        pos_t = sinusoidal_embedding(t_grid.to(torch.float32), pos_dim)
        pos_ot = torch.cat([pos_o, pos_t], dim=-1)

        q_base = self.pos_proj(pos_ot)
        q_pos_emb = self.query_pos_proj(q_base)
        q_base_expanded = q_base.unsqueeze(0).expand(B, -1, -1, -1)
        if self.use_distribution_conditioning:
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
        encoder_patch_tokens: torch.Tensor,
        debug_query_hint: str,
    ) -> torch.Tensor:
        hint_source = self._normalize_debug_query_hint(debug_query_hint)
        if hint_source == "none":
            return q_tokens

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
        decoder_latents: torch.Tensor | None,
        encoder_patch_tokens: torch.Tensor,
        d_out: int,
        T: int,
        debug_decoder_kv_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        kv_source = self._resolve_decoder_kv_source(debug_decoder_kv_source)
        if kv_source == "latents":
            if decoder_latents is None:
                raise ValueError("decoder_latents are required when debug_decoder_kv_source='latents'")
            kv_pos = torch.arange(decoder_latents.shape[1], device=decoder_latents.device, dtype=torch.float32)
            return decoder_latents, kv_pos, kv_pos, kv_source

        encoder_patch_flat, kv_pos_o, kv_pos_t = self._flatten_encoder_patch_tokens(
            encoder_patch_tokens,
            d_out=d_out,
            T=T,
        )
        return encoder_patch_flat, kv_pos_o, kv_pos_t, kv_source

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
        B, Q, _ = q_tokens.shape
        p = int(self.cfg.patch_size)
        expected_d_in_pad = T * p
        if d_in_pad != expected_d_in_pad:
            raise ValueError(f"d_in_pad must equal T*patch_size={expected_d_in_pad}, got {d_in_pad}")

        shortcut_disabled = bool(
            disable_z_shortcut or self.cfg.big_vae.disable_z_shortcut or not self.use_latent_bottleneck
        )
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

    def _decode_direct_from_encoder_tokens(
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
                f"{tuple(encoder_patch_tokens.shape)} vs expected (*, {d_out}, {T}, {encoder_patch_tokens.shape[-1]})"
            )

        B = int(encoder_patch_tokens.shape[0])
        p = int(self.cfg.patch_size)
        Q = d_out * T

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

    def _decode_from_encoder_outputs(
        self,
        encoder_patch_tokens: torch.Tensor,
        *,
        dist_patch_by_patch: torch.Tensor | None,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
        return_direction_pre_norms: bool = False,
        disable_z_shortcut: bool = False,
        debug_decoder_kv_source: str = "latents",
        debug_query_hint: str = "none",
        debug_direct_from_encoder_tokens: bool = False,
        return_debug_info: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        B = int(encoder_patch_tokens.shape[0])
        if self.use_distribution_conditioning:
            if dist_patch_by_patch is None:
                raise ValueError("dist_patch_by_patch is required when distribution conditioning is enabled")
            if tuple(dist_patch_by_patch.shape[:2]) != (B, T):
                raise ValueError(
                    "dist_patch_by_patch must be [B, T, d_dist], got "
                    f"{tuple(dist_patch_by_patch.shape)} for expected {(B, T, self.cfg.distribution.d_dist)}"
                )
        elif dist_patch_by_patch is not None:
            raise ValueError("dist_patch_by_patch must be None when distribution conditioning is disabled")

        if self.use_latent_bottleneck:
            latent_slots, z, decoder_latents, latent_bottleneck_debug = self._encode_latent_slots_from_patch_tokens(
                encoder_patch_tokens,
                d_out=d_out,
                T=T,
            )
        else:
            latent_slots = None
            z = encoder_patch_tokens.new_zeros((B, self.z_dim))
            decoder_latents = None
            latent_bottleneck_debug = None

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

        if bool(debug_direct_from_encoder_tokens):
            outputs = self._decode_direct_from_encoder_tokens(
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
            resolved_kv_source = "encoder_patch_tokens"
        else:
            decoder_kv, kv_pos_o, kv_pos_t, resolved_kv_source = self._build_decoder_kv_state(
                decoder_latents=decoder_latents,
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
                "debug_latent_bottleneck_kind": self.latent_bottleneck_kind,
                "debug_encoder_conditioning_kind": self.encoder_conditioning_kind,
                "debug_patch_tokenizer_kind": self.patch_tokenizer_kind,
                "debug_decoder_kv_source": resolved_kv_source,
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
            if latent_slots is not None:
                debug_info["latent_slots"] = latent_slots
            if latent_bottleneck_debug is not None:
                debug_info["latent_bottleneck_debug"] = latent_bottleneck_debug
            return outputs + (debug_info,)
        return outputs

    def _forward_debug(
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

        tokens_by_output, encoder_patch_tokens, T, d_in_pad, dist_patch_by_patch = self._encode_patch_tokens(W, X)
        decode_outputs = self._decode_from_encoder_outputs(
            encoder_patch_tokens,
            dist_patch_by_patch=dist_patch_by_patch,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            T=T,
            return_direction_pre_norms=return_direction_pre_norms,
            disable_z_shortcut=disable_z_shortcut,
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
            "encoder_tokens_by_output": tokens_by_output,
            "encoder_cls_tokens": tokens_by_output[:, :, :1, :],
            "encoder_patch_tokens": encoder_patch_tokens,
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

        _tokens_by_output, encoder_patch_tokens, T, d_in_pad, dist_patch_by_patch = self._encode_patch_tokens(W, X)
        decode_outputs = self._decode_from_encoder_outputs(
            encoder_patch_tokens,
            dist_patch_by_patch=dist_patch_by_patch,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
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


def build_procedural_simple_vae(model_cfg: ModelConfig) -> nn.Module:
    return ProceduralSimpleBigWeightVAE(model_cfg)


__all__ = [
    "ProceduralSimpleBigWeightVAE",
    "build_procedural_simple_vae",
]
