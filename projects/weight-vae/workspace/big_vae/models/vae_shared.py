from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quantile_over_samples(X_q: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compile-friendly quantile over sample axis.

    X_q: [B, n, p]
    q: [k]
    returns: [k, B, p]
    """
    if X_q.ndim != 3:
        raise ValueError(f"X_q must be [B,n,p], got {tuple(X_q.shape)}")
    if q.ndim != 1:
        raise ValueError(f"q must be rank-1, got {tuple(q.shape)}")

    sorted_vals, _ = torch.sort(X_q, dim=1)
    n = sorted_vals.shape[1]
    k = q.shape[0]

    q = q.to(device=X_q.device, dtype=X_q.dtype).clamp(0.0, 1.0)
    pos = q * (n - 1)

    lower = torch.floor(pos).to(dtype=torch.long)
    upper = torch.ceil(pos).to(dtype=torch.long)
    alpha = (pos - lower.to(dtype=X_q.dtype)).view(1, k, 1)

    B, _, p = sorted_vals.shape
    lower_idx = lower.view(1, k, 1).expand(B, k, p)
    upper_idx = upper.view(1, k, 1).expand(B, k, p)

    lower_vals = sorted_vals.gather(dim=1, index=lower_idx)
    upper_vals = sorted_vals.gather(dim=1, index=upper_idx)
    q_vals = lower_vals + (upper_vals - lower_vals) * alpha
    return q_vals.permute(1, 0, 2).contiguous()


def _masked_quantile_over_samples(X_q: torch.Tensor, q: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    """
    Mask-aware quantile over sample axis.

    X_q: [B, n, p]
    sample_mask: [B, n] with True for valid rows
    q: [k]
    returns: [k, B, p]
    """
    if X_q.ndim != 3:
        raise ValueError(f"X_q must be [B,n,p], got {tuple(X_q.shape)}")
    if q.ndim != 1:
        raise ValueError(f"q must be rank-1, got {tuple(q.shape)}")
    if sample_mask.ndim != 2:
        raise ValueError(f"sample_mask must be [B,n], got {tuple(sample_mask.shape)}")
    if tuple(sample_mask.shape) != tuple(X_q.shape[:2]):
        raise ValueError(
            f"sample_mask shape must match X_q sample axes, got {tuple(sample_mask.shape)} vs {tuple(X_q.shape[:2])}"
        )

    mask = sample_mask.to(device=X_q.device, dtype=torch.bool)
    valid_counts = mask.sum(dim=1).clamp_min(1)
    fill_value = torch.full((), float("inf"), device=X_q.device, dtype=X_q.dtype)
    masked_values = torch.where(mask.unsqueeze(-1), X_q, fill_value)
    sorted_vals, _ = torch.sort(masked_values, dim=1)

    q = q.to(device=X_q.device, dtype=X_q.dtype).clamp(0.0, 1.0)
    pos = q.unsqueeze(0) * (valid_counts.to(dtype=X_q.dtype).unsqueeze(1) - 1.0)

    lower = torch.floor(pos).to(dtype=torch.long)
    upper = torch.ceil(pos).to(dtype=torch.long)
    alpha = (pos - lower.to(dtype=X_q.dtype)).unsqueeze(-1)

    B, _, p = sorted_vals.shape
    k = q.shape[0]
    lower_idx = lower.unsqueeze(-1).expand(B, k, p)
    upper_idx = upper.unsqueeze(-1).expand(B, k, p)

    lower_vals = sorted_vals.gather(dim=1, index=lower_idx)
    upper_vals = sorted_vals.gather(dim=1, index=upper_idx)
    q_vals = lower_vals + (upper_vals - lower_vals) * alpha
    return q_vals.permute(1, 0, 2).contiguous()


def sinusoidal_embedding(indices: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Deterministic sinusoidal embedding for arbitrary-sized integer grids."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    x = indices.to(dtype=torch.float32)
    half = dim // 2
    if half == 0:
        return x.unsqueeze(-1)

    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / max(half - 1, 1))
    angles = x.unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class CleanContentReadoutV8(nn.Module):
    """Single bias-free W-value readout with X/position-only routing.

    For fixed ``x_context`` and masks the returned latent is exactly linear in
    ``w_patches`` up to floating-point arithmetic.  There is deliberately no
    residual, normalization, gate, or dropout on the value/return path.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        d_context: int,
        d_latent: int,
        num_latents: int,
        position_bands: int = 8,
        init_seed: int = 8008,
    ) -> None:
        super().__init__()
        if min(patch_size, d_context, d_latent, num_latents, position_bands) <= 0:
            raise ValueError("clean content readout dimensions must all be positive")
        if int(d_latent) % int(patch_size) != 0:
            raise ValueError("clean content readout requires d_latent divisible by patch_size")
        self.patch_size = int(patch_size)
        self.d_context = int(d_context)
        self.d_latent = int(d_latent)
        self.num_latents = int(num_latents)
        self.position_bands = int(position_bands)
        self.n_heads = self.d_latent // self.patch_size
        self.head_dim = self.patch_size
        self.anchor_sigma_keys = 0.6
        self.q_proj = nn.Linear(self.d_context, self.d_latent, bias=False)
        self.k_proj = nn.Linear(self.d_context, self.d_latent, bias=False)
        self.out_proj = nn.Linear(self.d_latent, self.d_latent, bias=False)
        self.capture_routing_diagnostics = False
        self.last_routing_diagnostics: dict[str, torch.Tensor] | None = None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(init_seed))
        # Q/K are a deliberately small learned X-only perturbation around the
        # fixed routing below.  Values remain the raw signed p-vectors.
        for projection in (self.q_proj, self.k_proj):
            nn.init.normal_(
                projection.weight,
                mean=0.0,
                # The distribution context has a much larger empirical scale
                # than a unit-normal test input.  Keep the learned X-only
                # perturbation small relative to the fixed anchor routing.
                std=0.1 * math.sqrt(0.05 * math.sqrt(8.0) / float(self.d_context)),
                generator=generator,
            )
        with torch.no_grad():
            self.out_proj.weight.copy_(
                torch.eye(self.d_latent, dtype=self.out_proj.weight.dtype)
            )

    def _fixed_anchor_scores(self, key_valid: torch.Tensor) -> torch.Tensor:
        B, key_count = key_valid.shape
        route_count = self.num_latents * self.n_heads
        key_order = key_valid.long().cumsum(dim=-1).sub(1).clamp_min(0).to(torch.float32)
        valid_counts = key_valid.sum(dim=-1).to(torch.float32).clamp_min(1.0)
        routes = torch.arange(route_count, device=key_valid.device, dtype=torch.float32)
        anchors = (
            (routes.view(1, route_count, 1) + 0.5)
            * valid_counts.view(B, 1, 1)
            / float(route_count)
            - 0.5
        )
        distance = (key_order.view(B, 1, key_count) - anchors) / self.anchor_sigma_keys
        scores = -0.5 * distance.square()
        return scores.masked_fill(~key_valid.unsqueeze(1), torch.finfo(scores.dtype).min)

    def forward(
        self,
        w_patches: torch.Tensor,
        x_context: torch.Tensor,
        *,
        patch_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        if w_patches.ndim != 4:
            raise ValueError(f"w_patches must be [B,d_out,T,p], got {tuple(w_patches.shape)}")
        B, d_out, T, p = w_patches.shape
        if p != self.patch_size:
            raise ValueError(f"w_patches patch width must be {self.patch_size}, got {p}")
        if tuple(x_context.shape) != (B, T, self.d_context):
            raise ValueError(
                f"x_context must be {(B, T, self.d_context)}, got {tuple(x_context.shape)}"
            )
        if tuple(patch_mask.shape) != (B, T) or tuple(output_mask.shape) != (B, d_out):
            raise ValueError("clean content readout mask shapes do not match W patches")
        patch_valid = patch_mask.to(device=w_patches.device, dtype=torch.bool)
        output_valid = output_mask.to(device=w_patches.device, dtype=torch.bool)
        key_valid = (output_valid.unsqueeze(-1) & patch_valid.unsqueeze(1)).reshape(B, d_out * T)
        if not bool(key_valid.any(dim=1).all().item()):
            raise ValueError("every sample must have at least one valid content key")

        patch_weight = patch_valid.to(dtype=x_context.dtype).unsqueeze(-1)
        x_global = (x_context * patch_weight).sum(dim=1) / patch_weight.sum(dim=1).clamp_min(1.0)
        x_keys = x_context.unsqueeze(1).expand(-1, d_out, -1, -1)
        x_keys = x_keys.reshape(B, d_out * T, self.d_context)
        q = self.q_proj(x_global).view(B, self.n_heads, self.head_dim)
        k = self.k_proj(x_keys).view(B, d_out * T, self.n_heads, self.head_dim).transpose(1, 2)
        raw_learned_scores = torch.einsum("bhd,bhsd->bhs", q.float(), k.float()) / math.sqrt(
            float(self.head_dim)
        )
        learned_scores = 0.1 * torch.tanh(raw_learned_scores / 0.1)
        learned_scores = learned_scores.unsqueeze(2).expand(-1, -1, self.num_latents, -1)
        fixed_scores = self._fixed_anchor_scores(key_valid).view(
            B, self.num_latents, self.n_heads, d_out * T
        ).transpose(1, 2)
        routing = torch.softmax(fixed_scores + learned_scores, dim=-1).to(dtype=w_patches.dtype)
        raw_values = w_patches.reshape(B, d_out * T, p)
        raw_values = raw_values * key_valid.to(dtype=raw_values.dtype).unsqueeze(-1)
        attended = torch.einsum("bhls,bsp->blhp", routing, raw_values).contiguous()
        if self.capture_routing_diagnostics:
            with torch.no_grad():
                entropy = -(routing.float() * routing.float().clamp_min(1.0e-30).log()).sum(dim=-1)
                self.last_routing_diagnostics = {
                    "entropy_mean": entropy.mean().detach(),
                    "effective_keys_median": entropy.exp().median().detach(),
                    "max_mass_median": routing.float().amax(dim=-1).median().detach(),
                    "raw_score_rms": raw_learned_scores.square().mean().sqrt().detach(),
                    "bounded_score_rms": learned_scores.square().mean().sqrt().detach(),
                    "bounded_score_max_abs": learned_scores.abs().amax().detach(),
                    "bounded_score_saturation_fraction": (
                        learned_scores.abs() >= 0.099
                    ).float().mean().detach(),
                    "unique_argmax_min": torch.tensor(
                        min(int(torch.unique(sample.argmax(dim=-1)).numel()) for sample in routing),
                        device=routing.device,
                        dtype=torch.int64,
                    ),
                }
        else:
            self.last_routing_diagnostics = None
        return self.out_proj(attended.view(B, self.num_latents, self.d_latent))


class NonlinearContentRefinementBlockV9(nn.Module):
    """One zero-preserving W-conditioned refinement of a protected V8 carrier.

    The state and raw signed W patches affect routing, while the attention
    values remain the raw signed patches.  The only write to the latent state
    is a fixed, non-zero residual scale times a bias-free output projection.
    There is no post-write normalization, learned gate, or dropout.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        d_context: int,
        d_latent: int,
        num_latents: int,
        router_width: int,
        router_ffn_width: int,
        score_cap: float,
        residual_scale: float,
        anchor_sigma_keys: float,
        anchor_family: str,
        anchor_offset: float,
        init_seed: int,
        carrier_mean_content: bool = False,
        protected_anchor_floor: float = 0.0,
        global_fixed_anchor_sigma_keys: float = 32.0,
    ) -> None:
        super().__init__()
        if int(d_latent) % int(patch_size) != 0:
            raise ValueError("V9 requires d_latent divisible by patch_size")
        self.patch_size = int(patch_size)
        self.d_context = int(d_context)
        self.d_latent = int(d_latent)
        self.num_latents = int(num_latents)
        self.n_heads = self.d_latent // self.patch_size
        if int(router_width) <= 0 or int(router_width) % self.n_heads != 0:
            raise ValueError(
                "V9 router_width must be positive and divisible by d_latent/patch_size"
            )
        if int(router_ffn_width) <= 0:
            raise ValueError("V9 router_ffn_width must be positive")
        if not 0.0 < float(score_cap) <= 8.0:
            raise ValueError("V9 score_cap must lie in (0,8]")
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("V9 residual_scale must lie in (0,1]")
        if float(anchor_sigma_keys) <= 0.0:
            raise ValueError("V9 anchor_sigma_keys must be positive")
        self.router_width = int(router_width)
        self.router_head_dim = self.router_width // self.n_heads
        self.router_ffn_width = int(router_ffn_width)
        self.score_cap = float(score_cap)
        self.residual_scale = float(residual_scale)
        self.anchor_sigma_keys = float(anchor_sigma_keys)
        self.anchor_family = str(anchor_family)
        if self.anchor_family not in {"local", "medium", "broad", "global"}:
            raise ValueError("V9 anchor_family must be local, medium, broad, or global")
        self.anchor_offset = float(anchor_offset)
        self.carrier_mean_content = bool(carrier_mean_content)
        self.protected_anchor_floor = float(protected_anchor_floor)
        self.global_fixed_anchor_sigma_keys = float(global_fixed_anchor_sigma_keys)
        if not 0.0 <= self.protected_anchor_floor < 1.0:
            raise ValueError("V9 protected_anchor_floor must lie in [0,1)")
        if self.global_fixed_anchor_sigma_keys <= 0.0:
            raise ValueError("V9 global fixed-anchor sigma must be positive")

        self.state_norm = NonAffineRMSNorm(self.d_latent)
        self.weight_key_norm = NonAffineRMSNorm(self.patch_size)
        self.context_key_norm = NonAffineRMSNorm(self.d_context)
        self.query_proj = nn.Linear(self.d_latent, self.router_width, bias=False)
        self.weight_key_proj = nn.Linear(self.patch_size, self.router_width, bias=False)
        self.weight_norm_key_proj = nn.Linear(1, self.router_width, bias=False)
        self.context_key_proj = nn.Linear(self.d_context, self.router_width, bias=False)
        self.out_proj = nn.Linear(self.d_latent, self.d_latent, bias=False)
        self.local_ffn_norm = NonAffineRMSNorm(self.d_latent)
        self.local_ffn = nn.Sequential(
            nn.Linear(self.d_latent, self.router_ffn_width, bias=False),
            nn.GELU(),
            nn.Linear(self.router_ffn_width, self.d_latent, bias=False),
        )
        self.capture_routing_diagnostics = False
        self.last_routing_diagnostics: dict[str, torch.Tensor] | None = None

        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        for projection, scale in (
            (self.query_proj, 0.25),
            (self.weight_key_proj, 0.25),
            (self.weight_norm_key_proj, 0.05),
            (self.context_key_proj, 0.25),
            (self.local_ffn[0], 1.0),
            (self.local_ffn[2], 0.25),
        ):
            nn.init.normal_(
                projection.weight,
                mean=0.0,
                std=float(scale) / math.sqrt(float(projection.in_features)),
                generator=generator,
            )
        with torch.no_grad():
            if self.carrier_mean_content:
                # V9-A has a structural identity write from raw routed W; this
                # output projection starts at zero without gating that path.
                # The FFN remains nonzero at initialization and its energy is
                # reported separately from the raw write.
                self.out_proj.weight.zero_()
            else:
                self.out_proj.weight.copy_(
                    torch.eye(self.d_latent, dtype=self.out_proj.weight.dtype)
                )

    @staticmethod
    def _relative_log_rms_feature(
        raw_values: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = key_valid.to(device=raw_values.device, dtype=torch.float32)
        log_rms = 0.5 * torch.log(raw_values.float().square().mean(dim=-1) + 1.0e-24)
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (log_rms * valid).sum(dim=-1, keepdim=True) / count
        centered = (log_rms - mean) * valid
        std = torch.sqrt(centered.square().sum(dim=-1, keepdim=True) / count + 1.0e-8)
        return torch.tanh(centered / std).unsqueeze(-1) * valid.unsqueeze(-1)

    def _fixed_anchor_scores(
        self,
        key_valid: torch.Tensor,
        *,
        protect_global_diversity: bool = False,
    ) -> torch.Tensor:
        B, key_count = key_valid.shape
        if self.anchor_family == "global" and not protect_global_diversity:
            return torch.zeros(
                B,
                self.num_latents * self.n_heads,
                key_count,
                device=key_valid.device,
                dtype=torch.float32,
            ).masked_fill(~key_valid.unsqueeze(1), torch.finfo(torch.float32).min)
        route_count = self.num_latents * self.n_heads
        key_order = key_valid.long().cumsum(dim=-1).sub(1).clamp_min(0).to(torch.float32)
        valid_counts = key_valid.sum(dim=-1).to(torch.float32).clamp_min(1.0)
        routes = torch.arange(route_count, device=key_valid.device, dtype=torch.float32)
        anchors = (
            (routes.view(1, route_count, 1) + 0.5 + self.anchor_offset)
            * valid_counts.view(B, 1, 1)
            / float(route_count)
            - 0.5
        )
        sigma = (
            self.global_fixed_anchor_sigma_keys
            if self.anchor_family == "global"
            else self.anchor_sigma_keys
        )
        distance = (key_order.view(B, 1, key_count) - anchors) / sigma
        scores = -0.5 * distance.square()
        return scores.masked_fill(~key_valid.unsqueeze(1), torch.finfo(scores.dtype).min)

    def forward(
        self,
        state: torch.Tensor,
        w_patches: torch.Tensor,
        x_context: torch.Tensor,
        *,
        patch_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, d_out, T, p = w_patches.shape
        if tuple(state.shape) != (B, self.num_latents, self.d_latent):
            raise ValueError("V9 latent state shape does not match its configured carrier")
        if p != self.patch_size or tuple(x_context.shape) != (B, T, self.d_context):
            raise ValueError("V9 W/context dimensions do not match its configured carrier")
        if tuple(patch_mask.shape) != (B, T) or tuple(output_mask.shape) != (B, d_out):
            raise ValueError("V9 mask shapes do not match W patches")
        patch_valid = patch_mask.to(device=w_patches.device, dtype=torch.bool)
        output_valid = output_mask.to(device=w_patches.device, dtype=torch.bool)
        key_valid = (output_valid.unsqueeze(-1) & patch_valid.unsqueeze(1)).reshape(B, d_out * T)
        if not bool(key_valid.any(dim=1).all().item()):
            raise ValueError("every V9 sample must have at least one valid content key")

        query = self.query_proj(self.state_norm(state)).view(
            B, self.num_latents, self.n_heads, self.router_head_dim
        ).transpose(1, 2)
        query = F.normalize(query.float(), dim=-1, eps=1.0e-6)
        raw_values = w_patches.reshape(B, d_out * T, p)
        raw_values = raw_values * key_valid.to(dtype=raw_values.dtype).unsqueeze(-1)
        x_keys = x_context.unsqueeze(1).expand(-1, d_out, -1, -1).reshape(
            B, d_out * T, self.d_context
        )
        weight_keys = self.weight_key_proj(self.weight_key_norm(raw_values))
        norm_feature = self._relative_log_rms_feature(raw_values, key_valid)
        norm_key_contribution = self.weight_norm_key_proj(
            norm_feature.to(dtype=self.weight_norm_key_proj.weight.dtype)
        )
        context_keys = self.context_key_proj(self.context_key_norm(x_keys))
        keys = (weight_keys + norm_key_contribution + context_keys).view(
            B, d_out * T, self.n_heads, self.router_head_dim
        ).transpose(1, 2)
        keys = F.normalize(keys.float(), dim=-1, eps=1.0e-6)
        cosine_scores = torch.einsum("bhld,bhsd->bhls", query, keys)
        adaptive_scores = self.score_cap * cosine_scores
        fixed_scores = self._fixed_anchor_scores(key_valid).view(
            B, self.num_latents, self.n_heads, d_out * T
        ).transpose(1, 2)
        adaptive_routing = torch.softmax(fixed_scores + adaptive_scores, dim=-1)
        if self.protected_anchor_floor > 0.0:
            protected_scores = self._fixed_anchor_scores(
                key_valid,
                protect_global_diversity=True,
            ).view(B, self.num_latents, self.n_heads, d_out * T).transpose(1, 2)
            protected_routing = torch.softmax(protected_scores, dim=-1)
            routing = (
                self.protected_anchor_floor * protected_routing
                + (1.0 - self.protected_anchor_floor) * adaptive_routing
            ).to(dtype=w_patches.dtype)
        else:
            routing = adaptive_routing.to(dtype=w_patches.dtype)
        attended = torch.einsum("bhls,bsp->blhp", routing, raw_values).contiguous()
        raw_content = attended.view(B, self.num_latents, self.d_latent)
        routed_return = self.out_proj(raw_content)
        if self.carrier_mean_content:
            # The decoder-visible write is a zero-preserving transform of the
            # current raw-W read only.  The recurrent routing state may feed
            # later queries, but it cannot write a state-only/common value.
            ffn_return = self.local_ffn(self.local_ffn_norm(raw_content))
            branch = raw_content + self.residual_scale * (
                routed_return + ffn_return
            )
        else:
            local_state = state + self.residual_scale * routed_return
            ffn_return = self.local_ffn(self.local_ffn_norm(local_state))
            branch = routed_return + ffn_return

        if self.capture_routing_diagnostics:
            with torch.no_grad():
                entropy = -(routing.float() * routing.float().clamp_min(1.0e-30).log()).sum(dim=-1)
                state_rms = state.float().square().mean().sqrt()
                delta_rms = branch.float().square().mean().sqrt()
                self.last_routing_diagnostics = {
                    "state_rms": state_rms.detach(),
                    "delta_rms": delta_rms.detach(),
                    "raw_content_rms": raw_content.float().square().mean().sqrt().detach(),
                    "out_correction_rms": routed_return.float().square().mean().sqrt().detach(),
                    "ffn_correction_rms": ffn_return.float().square().mean().sqrt().detach(),
                    "scaled_delta_to_state_rms": (
                        self.residual_scale * delta_rms / state_rms.clamp_min(1.0e-24)
                    ).detach(),
                    "entropy_mean": entropy.mean().detach(),
                    "effective_keys_median": entropy.exp().median().detach(),
                    "max_mass_median": routing.float().amax(dim=-1).median().detach(),
                    "cosine_score_rms": cosine_scores.square().mean().sqrt().detach(),
                    "adaptive_score_rms": adaptive_scores.square().mean().sqrt().detach(),
                    "adaptive_score_max_abs": adaptive_scores.abs().amax().detach(),
                    "adaptive_score_near_bound_fraction": (
                        adaptive_scores.abs() >= (0.99 * self.score_cap)
                    ).float().mean().detach(),
                    "relative_log_rms_feature_rms": norm_feature.square().mean().sqrt().detach(),
                    "norm_key_contribution_rms": norm_key_contribution.float().square().mean().sqrt().detach(),
                    "unique_argmax_min": torch.tensor(
                        min(int(torch.unique(sample.argmax(dim=-1)).numel()) for sample in routing),
                        device=routing.device,
                        dtype=torch.int64,
                    ),
                    "anchor_family_id": torch.tensor(
                        {"local": 0, "medium": 1, "broad": 2, "global": 3}[
                            self.anchor_family
                        ],
                        device=routing.device,
                        dtype=torch.int64,
                    ),
                    "protected_anchor_floor": torch.tensor(
                        self.protected_anchor_floor,
                        device=routing.device,
                        dtype=torch.float32,
                    ),
                }
        else:
            self.last_routing_diagnostics = None
        return branch


class HybridNonlinearContentReadoutV9(nn.Module):
    """Scalable nonlinear refinements around an exact protected V8 carrier."""

    def __init__(
        self,
        *,
        patch_size: int,
        d_context: int,
        d_latent: int,
        num_latents: int,
        num_blocks: int,
        router_width: int,
        router_ffn_width: int,
        score_cap: float,
        residual_scale: float,
        anchor_sigma_keys: float,
        bypass_refinement: bool,
        init_seed: int = 9009,
        carrier_mean_content: bool = False,
        carrier_mix: float = 0.1,
        protected_anchor_floor: float = 0.25,
        carrier_content_aggregation: str = "sqrt_depth_sum",
    ) -> None:
        super().__init__()
        if int(num_blocks) <= 0:
            raise ValueError("V9 num_blocks must be positive")
        self.bypass_refinement = bool(bypass_refinement)
        self.residual_scale = float(residual_scale)
        self.carrier_mean_content = bool(carrier_mean_content)
        self.carrier_mix = float(carrier_mix)
        self.carrier_content_aggregation = str(carrier_content_aggregation)
        if self.carrier_mean_content and self.carrier_mix != 0.1:
            raise ValueError("V9-A carrier_mix is frozen to exactly 0.1")
        if (
            self.carrier_mean_content
            and self.carrier_content_aggregation != "sqrt_depth_sum"
        ):
            raise ValueError("V9-A aggregation must be exactly sqrt_depth_sum")
        self.blocks = nn.ModuleList(
            [
                NonlinearContentRefinementBlockV9(
                    patch_size=int(patch_size),
                    d_context=int(d_context),
                    d_latent=int(d_latent),
                    num_latents=int(num_latents),
                    router_width=int(router_width),
                    router_ffn_width=int(router_ffn_width),
                    score_cap=(
                        float(score_cap)
                        * {"local": 1.0, "medium": 5.0, "broad": 20.0, "global": 40.0}[
                            ("local", "medium", "broad", "global")[block_index % 4]
                        ]
                    ),
                    residual_scale=float(residual_scale),
                    anchor_sigma_keys=(
                        {"local": float(anchor_sigma_keys), "medium": 4.0, "broad": 32.0, "global": 1.0}[
                            ("local", "medium", "broad", "global")[block_index % 4]
                        ]
                    ),
                    anchor_family=("local", "medium", "broad", "global")[block_index % 4],
                    anchor_offset=((block_index * 0.6180339887498949) % 1.0) - 0.5,
                    init_seed=int(init_seed) + block_index,
                    carrier_mean_content=self.carrier_mean_content,
                    protected_anchor_floor=(
                        float(protected_anchor_floor) if self.carrier_mean_content else 0.0
                    ),
                    global_fixed_anchor_sigma_keys=32.0,
                )
                for block_index in range(int(num_blocks))
            ]
        )
        self.last_stage_states: tuple[torch.Tensor, ...] = ()
        self.last_carrier_state: torch.Tensor | None = None
        self.last_adaptive_state: torch.Tensor | None = None
        self.capture_stage_states = False

    def forward(
        self,
        carrier_state: torch.Tensor,
        w_patches: torch.Tensor,
        x_context: torch.Tensor,
        *,
        patch_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = carrier_state
        self.last_carrier_state = carrier_state.detach() if self.capture_stage_states else None
        if self.bypass_refinement:
            self.last_stage_states = ()
            self.last_adaptive_state = None
            return state
        states: list[torch.Tensor] = []
        if self.carrier_mean_content:
            routing_state = carrier_state
            # FP32 accumulation followed by 1/sqrt(k) preserves the RMS of
            # independent writes across depth.  Aligned writes intentionally
            # grow as sqrt(k); each final direct-write coefficient is 0.1/sqrt(N).
            content_sum = torch.zeros_like(carrier_state, dtype=torch.float32)
            for block_index, block in enumerate(self.blocks):
                content_write = block(
                    routing_state,
                    w_patches,
                    x_context,
                    patch_mask=patch_mask,
                    output_mask=output_mask,
                )
                content_sum = content_sum + content_write.float()
                depth = float(block_index + 1)
                content_mean = content_sum / depth
                depth_normalized = content_sum / math.sqrt(depth)
                # The private controller remains bounded by the online mean;
                # only the decoder-visible adaptive arm uses the variance-
                # preserving sqrt-depth sum.
                routing_state = (
                    carrier_state.float() + self.residual_scale * content_mean
                ).to(dtype=carrier_state.dtype)
                adaptive = depth_normalized.to(dtype=carrier_state.dtype)
                state = (
                    (1.0 - self.carrier_mix) * carrier_state
                    + self.carrier_mix * adaptive
                )
                if self.capture_stage_states:
                    states.append(state.detach())
            self.last_stage_states = tuple(states)
            self.last_adaptive_state = (
                (self.carrier_mix * adaptive).detach()
                if self.capture_stage_states
                else None
            )
            return state
        self.last_adaptive_state = None
        accumulated = torch.zeros_like(state)
        carrier_state = state
        for block in self.blocks:
            branch = block(
                state,
                w_patches,
                x_context,
                patch_mask=patch_mask,
                output_mask=output_mask,
            )
            accumulated = accumulated + self.residual_scale * branch
            state = carrier_state + accumulated
            if self.capture_stage_states:
                states.append(state.detach())
        self.last_stage_states = tuple(states)
        return state


class FixedOrthonormalRoutingBasisV10(nn.Module):
    """Fixed position/mask-only orthonormal analysis basis for V10.

    The Gaussian anchor bank is orthonormalized in FP32.  Equal masks share a
    cached basis, while different masks are never aliased.  The module owns no
    trainable state and never observes W or distribution/context features.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        num_latents: int,
        protected_dim: int,
        anchor_sigma_keys: float = 0.6,
        max_cache_entries: int = 8,
    ) -> None:
        super().__init__()
        if protected_dim % patch_size != 0:
            raise ValueError("V10 protected_dim must be divisible by patch_size")
        if anchor_sigma_keys <= 0.0:
            raise ValueError("V10 anchor_sigma_keys must be positive")
        self.patch_size = int(patch_size)
        self.num_latents = int(num_latents)
        self.protected_dim = int(protected_dim)
        self.n_heads = self.protected_dim // self.patch_size
        self.route_count = self.num_latents * self.n_heads
        self.anchor_sigma_keys = float(anchor_sigma_keys)
        self.max_cache_entries = int(max_cache_entries)
        self._basis_cache: dict[tuple[object, ...], torch.Tensor] = {}

    def _cache_key(self, key_valid: torch.Tensor) -> tuple[object, ...]:
        packed = key_valid.detach().to(device="cpu", dtype=torch.uint8).numpy().tobytes()
        return (
            key_valid.device.type,
            key_valid.device.index,
            int(key_valid.numel()),
            packed,
        )

    def _build_one(self, key_valid: torch.Tensor) -> torch.Tensor:
        if key_valid.ndim != 1:
            raise ValueError("V10 basis key mask must be one-dimensional")
        valid_indices = torch.nonzero(key_valid, as_tuple=False).flatten()
        valid_count = int(valid_indices.numel())
        if valid_count < self.route_count:
            raise ValueError(
                f"V10 needs at least {self.route_count} valid W patches, got {valid_count}"
            )
        with torch.autocast(device_type=key_valid.device.type, enabled=False):
            ordinal = torch.arange(valid_count, device=key_valid.device, dtype=torch.float32)
            routes = torch.arange(self.route_count, device=key_valid.device, dtype=torch.float32)
            anchors = (routes + 0.5) * float(valid_count) / float(self.route_count) - 0.5
            scores = -0.5 * (
                (ordinal.unsqueeze(0) - anchors.unsqueeze(1)) / self.anchor_sigma_keys
            ).square()
            anchored = torch.softmax(scores, dim=-1)
            # QR(A^T) constructs an orthonormal row basis spanning the fixed anchor
            # bank.  Canonical signs make reconstruction deterministic across calls.
            q, r = torch.linalg.qr(anchored.transpose(0, 1), mode="reduced")
            signs = torch.where(
                torch.diagonal(r) < 0,
                -torch.ones((), device=r.device, dtype=r.dtype),
                torch.ones((), device=r.device, dtype=r.dtype),
            )
            q = q * signs.unsqueeze(0)
            basis = torch.zeros(
                self.route_count,
                int(key_valid.numel()),
                device=key_valid.device,
                dtype=torch.float32,
            )
            basis[:, valid_indices] = q.transpose(0, 1)
        return basis

    def forward(
        self,
        patch_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        if patch_mask.ndim != 2 or output_mask.ndim != 2:
            raise ValueError("V10 basis masks must be [B,T] and [B,d_out]")
        if int(patch_mask.shape[0]) != int(output_mask.shape[0]):
            raise ValueError("V10 basis masks must have the same batch size")
        patch_valid = patch_mask.to(dtype=torch.bool)
        output_valid = output_mask.to(device=patch_mask.device, dtype=torch.bool)
        key_valid = (
            output_valid.unsqueeze(-1) & patch_valid.unsqueeze(1)
        ).reshape(int(patch_mask.shape[0]), -1)
        if int(key_valid.shape[0]) > 1 and torch.equal(
            key_valid,
            key_valid[:1].expand_as(key_valid),
        ):
            sample_mask = key_valid[0]
            cache_key = self._cache_key(sample_mask)
            basis = self._basis_cache.get(cache_key)
            if basis is None or basis.device != sample_mask.device:
                basis = self._build_one(sample_mask)
                if len(self._basis_cache) >= self.max_cache_entries:
                    self._basis_cache.pop(next(iter(self._basis_cache)))
                self._basis_cache[cache_key] = basis
            return basis.unsqueeze(0).expand(int(key_valid.shape[0]), -1, -1)
        bases: list[torch.Tensor] = []
        for sample_mask in key_valid:
            cache_key = self._cache_key(sample_mask)
            basis = self._basis_cache.get(cache_key)
            if basis is None or basis.device != sample_mask.device:
                basis = self._build_one(sample_mask)
                if len(self._basis_cache) >= self.max_cache_entries:
                    self._basis_cache.pop(next(iter(self._basis_cache)))
                self._basis_cache[cache_key] = basis
            bases.append(basis)
        return torch.stack(bases, dim=0)


class OrthogonalComplementEncoderV10(nn.Module):
    """Protected fixed analysis plus independent direct-W adaptive experts."""

    def __init__(
        self,
        *,
        patch_size: int,
        d_context: int,
        d_latent: int,
        num_latents: int,
        protected_dim: int,
        adaptive_dim: int,
        num_experts: int,
        router_width: int,
        router_ffn_width: int,
        score_cap: float,
        residual_scale: float,
        anchor_sigma_keys: float,
        protected_anchor_floor: float,
        init_seed: int = 10101,
    ) -> None:
        super().__init__()
        if d_latent != protected_dim + adaptive_dim:
            raise ValueError("V10 protected_dim + adaptive_dim must equal d_latent")
        if protected_dim % patch_size != 0 or d_latent % patch_size != 0:
            raise ValueError("V10 protected and expert widths must be divisible by patch_size")
        if num_experts <= 0:
            raise ValueError("V10 num_experts must be positive")
        self.patch_size = int(patch_size)
        self.d_latent = int(d_latent)
        self.num_latents = int(num_latents)
        self.protected_dim = int(protected_dim)
        self.adaptive_dim = int(adaptive_dim)
        self.num_experts = int(num_experts)
        self.basis = FixedOrthonormalRoutingBasisV10(
            patch_size=patch_size,
            num_latents=num_latents,
            protected_dim=protected_dim,
            anchor_sigma_keys=anchor_sigma_keys,
        )
        families = ("local", "medium", "broad", "global")
        cap_multipliers = {"local": 1.0, "medium": 5.0, "broad": 20.0, "global": 40.0}
        sigmas = {"local": float(anchor_sigma_keys), "medium": 4.0, "broad": 32.0, "global": 1.0}
        self.experts = nn.ModuleList()
        self.slot_mixers = nn.ModuleList()
        self.code_projections = nn.ModuleList()
        generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
        for expert_index in range(self.num_experts):
            family = families[expert_index % len(families)]
            self.experts.append(
                NonlinearContentRefinementBlockV9(
                    patch_size=patch_size,
                    d_context=d_context,
                    d_latent=d_latent,
                    num_latents=num_latents,
                    router_width=router_width,
                    router_ffn_width=router_ffn_width,
                    score_cap=float(score_cap) * cap_multipliers[family],
                    residual_scale=residual_scale,
                    anchor_sigma_keys=sigmas[family],
                    anchor_family=family,
                    anchor_offset=((expert_index * 0.6180339887498949) % 1.0) - 0.5,
                    init_seed=int(init_seed) + expert_index,
                    carrier_mean_content=True,
                    protected_anchor_floor=protected_anchor_floor,
                    global_fixed_anchor_sigma_keys=32.0,
                )
            )
            slot_mixer = nn.Linear(num_latents, num_latents, bias=False)
            nn.init.eye_(slot_mixer.weight)
            self.slot_mixers.append(slot_mixer)
            code_projection = nn.Linear(d_latent, adaptive_dim, bias=False)
            nn.init.normal_(
                code_projection.weight,
                mean=0.0,
                std=0.1 / math.sqrt(float(d_latent)),
                generator=generator,
            )
            self.code_projections.append(code_projection)
        self.capture_expert_states = False
        self.last_expert_codes: tuple[torch.Tensor, ...] = ()
        self.last_basis: torch.Tensor | None = None
        self.last_protected: torch.Tensor | None = None
        self.last_adaptive: torch.Tensor | None = None

    def forward(
        self,
        w_patches: torch.Tensor,
        x_context: torch.Tensor,
        *,
        patch_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, d_out, T, p = w_patches.shape
        if p != self.patch_size:
            raise ValueError("V10 W patch size does not match its frozen contract")
        with torch.autocast(device_type=w_patches.device.type, enabled=False):
            basis = self.basis(patch_mask, output_mask).float()
            raw_values = w_patches.reshape(B, d_out * T, p).float()
            protected_routes = torch.einsum("brs,bsp->brp", basis, raw_values)
            protected = protected_routes.reshape(B, self.num_latents, self.protected_dim)
        protected_padded = F.pad(protected, (0, self.adaptive_dim)).to(dtype=w_patches.dtype)
        adaptive_sum = torch.zeros(
            B,
            self.num_latents,
            self.adaptive_dim,
            device=w_patches.device,
            dtype=torch.float32,
        )
        captured: list[torch.Tensor] = []
        for expert, slot_mixer, code_projection in zip(
            self.experts,
            self.slot_mixers,
            self.code_projections,
            strict=True,
        ):
            raw_write = expert(
                protected_padded,
                w_patches,
                x_context,
                patch_mask=patch_mask,
                output_mask=output_mask,
            )
            slot_mixed = slot_mixer(raw_write.transpose(1, 2)).transpose(1, 2)
            code = code_projection(slot_mixed)
            adaptive_sum = adaptive_sum + code.float()
            if self.capture_expert_states:
                captured.append(code.detach())
        adaptive = adaptive_sum / math.sqrt(float(self.num_experts))
        latent = torch.cat([protected, adaptive], dim=-1).to(dtype=w_patches.dtype)
        if self.capture_expert_states:
            self.last_expert_codes = tuple(captured)
            self.last_basis = basis.detach()
            self.last_protected = protected.detach()
            self.last_adaptive = adaptive.detach()
        else:
            self.last_expert_codes = ()
            self.last_basis = None
            self.last_protected = None
            self.last_adaptive = None
        return latent


class FullViewComplementTrunkV11(nn.Module):
    """One full-complement trunk conditioned by the protected coefficients."""

    def __init__(
        self,
        *,
        patch_size: int,
        d_context: int,
        d_latent: int,
        num_latents: int,
        output_dim: int,
        num_blocks: int,
        router_width: int,
        router_ffn_width: int,
        score_cap: float,
        residual_scale: float,
        anchor_sigma_keys: float,
        protected_anchor_floor: float,
        trunk_index: int,
        init_seed: int,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("V11 trunk depth must be positive")
        if output_dim <= 0:
            raise ValueError("V11 trunk output width must be positive")
        self.num_blocks = int(num_blocks)
        self.state_write_scale = 1.0 / math.sqrt(float(self.num_blocks))
        families = ("local", "medium", "broad", "global")
        cap_multipliers = {"local": 1.0, "medium": 5.0, "broad": 20.0, "global": 40.0}
        sigmas = {
            "local": float(anchor_sigma_keys),
            "medium": 4.0,
            "broad": 32.0,
            "global": 1.0,
        }
        self.blocks = nn.ModuleList()
        for block_index in range(self.num_blocks):
            global_index = int(trunk_index) * self.num_blocks + block_index
            family = families[block_index % len(families)]
            self.blocks.append(
                NonlinearContentRefinementBlockV9(
                    patch_size=patch_size,
                    d_context=d_context,
                    d_latent=d_latent,
                    num_latents=num_latents,
                    router_width=router_width,
                    router_ffn_width=router_ffn_width,
                    score_cap=float(score_cap) * cap_multipliers[family],
                    residual_scale=residual_scale,
                    anchor_sigma_keys=sigmas[family],
                    anchor_family=family,
                    anchor_offset=((global_index * 0.6180339887498949) % 1.0) - 0.5,
                    init_seed=int(init_seed) + global_index,
                    carrier_mean_content=True,
                    protected_anchor_floor=protected_anchor_floor,
                    global_fixed_anchor_sigma_keys=32.0,
                )
            )
        self.slot_mixer = nn.Linear(num_latents, num_latents, bias=False)
        nn.init.eye_(self.slot_mixer.weight)
        self.code_projection = nn.Linear(d_latent, output_dim, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(
            int(init_seed) + 10_000 + int(trunk_index)
        )
        nn.init.normal_(
            self.code_projection.weight,
            mean=0.0,
            std=0.1 / math.sqrt(float(d_latent)),
            generator=generator,
        )
        self.capture_states = False
        self.retain_code_gradient = False
        self.last_delta: torch.Tensor | None = None
        self.last_code: torch.Tensor | None = None
        self.last_code_for_gradient: torch.Tensor | None = None
        self.code_tensors_for_gradient: list[torch.Tensor] = []

    def forward(
        self,
        protected_state: torch.Tensor,
        w_patches: torch.Tensor,
        x_context: torch.Tensor,
        *,
        patch_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        # The protected carrier and ten direct writes can differ by much less
        # than one BF16 ulp. Keep the recurrent accumulator and subtraction in
        # FP32 so the serialized code cannot lose small but valid W writes.
        protected_state_f = protected_state.float()
        delta = torch.zeros_like(protected_state_f)
        for block in self.blocks:
            state = protected_state_f + delta
            raw_write = block(
                state,
                w_patches,
                x_context,
                patch_mask=patch_mask,
                output_mask=output_mask,
            )
            delta = delta + self.state_write_scale * raw_write.float()
        # Deliberately no terminal normalization: complement magnitude and
        # positive homogeneity must survive into the serialized code.
        with torch.autocast(device_type=delta.device.type, enabled=False):
            slot_mixed = self.slot_mixer(delta.transpose(1, 2)).transpose(1, 2)
            code = self.code_projection(slot_mixed)
        if self.retain_code_gradient and code.requires_grad:
            code.retain_grad()
            self.last_code_for_gradient = code
            self.code_tensors_for_gradient.append(code)
        elif not self.retain_code_gradient:
            self.last_code_for_gradient = None
            self.code_tensors_for_gradient.clear()
        if self.capture_states:
            self.last_delta = delta.detach()
            self.last_code = code.detach()
        else:
            self.last_delta = None
            self.last_code = None
        return code


class FourTrunkComplementEncoderV11(nn.Module):
    """Exact protected analysis plus four independent full-view trunks."""

    def __init__(
        self,
        *,
        patch_size: int,
        d_context: int,
        d_latent: int,
        num_latents: int,
        protected_dim: int,
        adaptive_dim: int,
        num_trunks: int,
        blocks_per_trunk: int,
        trunk_dim: int,
        router_width: int,
        router_ffn_width: int,
        score_cap: float,
        residual_scale: float,
        anchor_sigma_keys: float,
        protected_anchor_floor: float,
        init_seed: int = 11101,
    ) -> None:
        super().__init__()
        if d_latent != protected_dim + adaptive_dim:
            raise ValueError("V11 protected_dim + adaptive_dim must equal d_latent")
        if num_trunks * trunk_dim != adaptive_dim:
            raise ValueError("V11 trunk outputs must concatenate to adaptive_dim")
        if protected_dim % patch_size != 0 or d_latent % patch_size != 0:
            raise ValueError("V11 protected and latent widths must be divisible by patch_size")
        self.patch_size = int(patch_size)
        self.d_latent = int(d_latent)
        self.num_latents = int(num_latents)
        self.protected_dim = int(protected_dim)
        self.adaptive_dim = int(adaptive_dim)
        self.num_trunks = int(num_trunks)
        self.blocks_per_trunk = int(blocks_per_trunk)
        self.trunk_dim = int(trunk_dim)
        self.basis = FixedOrthonormalRoutingBasisV10(
            patch_size=patch_size,
            num_latents=num_latents,
            protected_dim=protected_dim,
            anchor_sigma_keys=anchor_sigma_keys,
        )
        self.trunks = nn.ModuleList(
            [
                FullViewComplementTrunkV11(
                    patch_size=patch_size,
                    d_context=d_context,
                    d_latent=d_latent,
                    num_latents=num_latents,
                    output_dim=trunk_dim,
                    num_blocks=blocks_per_trunk,
                    router_width=router_width,
                    router_ffn_width=router_ffn_width,
                    score_cap=score_cap,
                    residual_scale=residual_scale,
                    anchor_sigma_keys=anchor_sigma_keys,
                    protected_anchor_floor=protected_anchor_floor,
                    trunk_index=trunk_index,
                    init_seed=init_seed,
                )
                for trunk_index in range(num_trunks)
            ]
        )
        self.capture_trunk_states = False
        self.last_trunk_codes: tuple[torch.Tensor, ...] = ()
        self.last_trunk_deltas: tuple[torch.Tensor, ...] = ()
        self.last_basis: torch.Tensor | None = None
        self.last_protected: torch.Tensor | None = None
        self.last_complement_patches: torch.Tensor | None = None
        self.last_complement_feed: torch.Tensor | None = None
        self.last_complement_feed_dtype: torch.dtype | None = None
        self.last_adaptive: torch.Tensor | None = None

    @property
    def routing_blocks(self) -> tuple[NonlinearContentRefinementBlockV9, ...]:
        return tuple(block for trunk in self.trunks for block in trunk.blocks)

    def forward(
        self,
        w_patches: torch.Tensor,
        x_context: torch.Tensor,
        *,
        patch_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, d_out, token_count, patch_size = w_patches.shape
        if patch_size != self.patch_size:
            raise ValueError("V11 W patch size does not match its frozen contract")
        outer_autocast_enabled = torch.is_autocast_enabled(w_patches.device.type)
        compute_dtype = (
            torch.get_autocast_dtype(w_patches.device.type)
            if outer_autocast_enabled
            else w_patches.dtype
        )
        with torch.autocast(device_type=w_patches.device.type, enabled=False):
            basis = self.basis(patch_mask, output_mask).float()
            raw_values = w_patches.reshape(batch, d_out * token_count, patch_size).float()
            protected_routes = torch.einsum("brs,bsp->brp", basis, raw_values)
            floor_values = torch.einsum("brs,brp->bsp", basis, protected_routes)
            complement_values = raw_values - floor_values
            protected = protected_routes.reshape(
                batch, self.num_latents, self.protected_dim
            )
            complement_patches_fp32 = complement_values.reshape_as(w_patches)
        complement_patches = complement_patches_fp32.to(dtype=compute_dtype)
        protected_padded = F.pad(protected, (0, self.adaptive_dim)).to(
            dtype=w_patches.dtype
        )
        for trunk in self.trunks:
            trunk.capture_states = self.capture_trunk_states
        codes = tuple(
            trunk(
                protected_padded,
                complement_patches,
                x_context,
                patch_mask=patch_mask,
                output_mask=output_mask,
            )
            for trunk in self.trunks
        )
        adaptive = torch.cat(codes, dim=-1)
        latent = torch.cat([protected, adaptive], dim=-1).to(dtype=w_patches.dtype)
        if self.capture_trunk_states:
            if any(trunk.last_delta is None or trunk.last_code is None for trunk in self.trunks):
                raise RuntimeError("V11 trunk capture was incomplete")
            self.last_trunk_codes = tuple(code.detach() for code in codes)
            self.last_trunk_deltas = tuple(
                trunk.last_delta for trunk in self.trunks if trunk.last_delta is not None
            )
            self.last_basis = basis.detach()
            self.last_protected = protected.detach()
            self.last_complement_patches = complement_patches_fp32.detach()
            self.last_complement_feed = complement_patches.detach()
            self.last_complement_feed_dtype = complement_patches.dtype
            self.last_adaptive = adaptive.detach()
        else:
            self.last_trunk_codes = ()
            self.last_trunk_deltas = ()
            self.last_basis = None
            self.last_protected = None
            self.last_complement_patches = None
            self.last_complement_feed = None
            self.last_complement_feed_dtype = None
            self.last_adaptive = None
        return latent


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
    """
    Apply RoPE to attention projections.

    x: [B, H, T, D_h]
    positions: [T]
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B,H,T,D_h], got {tuple(x.shape)}")
    if positions.ndim != 1:
        raise ValueError(f"positions must be rank-1 [T], got {tuple(positions.shape)}")
    if int(positions.shape[0]) != int(x.shape[2]):
        raise ValueError(f"positions length ({int(positions.shape[0])}) must match sequence length ({int(x.shape[2])})")

    _, _, _, d_h = x.shape
    rope_dim = int(d_h) if int(d_h) % 2 == 0 else int(d_h) - 1
    if rope_dim <= 0:
        return x

    pos = positions.to(device=x.device, dtype=torch.float32)
    inv_freq = torch.exp(
        -math.log(max_period) * torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / float(rope_dim)
    )
    angles = pos[:, None] * inv_freq[None, :]
    sin = torch.sin(angles).to(dtype=x.dtype).view(1, 1, x.shape[2], rope_dim // 2)
    cos = torch.cos(angles).to(dtype=x.dtype).view(1, 1, x.shape[2], rope_dim // 2)

    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    x_rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    if x_pass.numel() == 0:
        return x_rotated
    return torch.cat([x_rotated, x_pass], dim=-1)


def _key_padding_to_attn_bias(key_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a [B, S] validity mask into an additive attention bias for SDPA.

    `key_mask=True` means a key/value position is valid and may be attended to.
    """
    if key_mask.ndim != 2:
        raise ValueError(f"key_mask must be [B,S], got {tuple(key_mask.shape)}")
    mask = key_mask.to(dtype=torch.bool)
    bias = torch.zeros((int(mask.shape[0]), 1, 1, int(mask.shape[1])), device=mask.device, dtype=dtype)
    neg_inf = torch.full_like(bias, torch.finfo(dtype).min)
    return torch.where(mask.unsqueeze(1).unsqueeze(1), bias, neg_inf)


def _apply_sequence_mask(x: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return x
    if x.ndim != 3:
        raise ValueError(f"x must be [B,S,D] when applying a token mask, got {tuple(x.shape)}")
    if token_mask.ndim != 2:
        raise ValueError(f"token_mask must be [B,S], got {tuple(token_mask.shape)}")
    if tuple(token_mask.shape) != tuple(x.shape[:2]):
        raise ValueError(f"token_mask shape must match x sample axes, got {tuple(token_mask.shape)} vs {tuple(x.shape[:2])}")
    return x * token_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)


def _rope_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    dropout_p: float,
    training: bool,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    RoPE + scaled dot-product attention.

    q/k/v: [B, H, T, D_h] and [B, H, S, D_h]
    q_pos: [T]
    k_pos: [S]
    returns: [B, H, T, D_h]
    """
    q_rot = _apply_rope(q, q_pos)
    k_rot = _apply_rope(k, k_pos)
    attn_dropout = float(dropout_p) if training else 0.0
    return F.scaled_dot_product_attention(
        q_rot,
        k_rot,
        v,
        attn_mask=attn_mask,
        dropout_p=attn_dropout,
        is_causal=False,
    )


class RoPeMixed2D(nn.Module):
    """
    Learned mixed 2D RoPE (arXiv:2403.13298, Eq. 14).

    angle[h, t] = freq1[h, t] * pos1 + freq2[h, t] * pos2

    Every head dimension encodes both positional axes simultaneously.
    """

    def __init__(self, n_heads: int, head_dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        half = head_dim // 2
        if half <= 0:
            raise ValueError(f"head_dim must be >= 2 for RoPE, got {head_dim}")
        self.half = half
        base_freq1 = torch.exp(
            -math.log(max_period) * torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
        )
        base_freq2 = torch.exp(
            -0.5 * math.log(max_period) * torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)
        )
        self.freq1 = nn.Parameter(base_freq1.unsqueeze(0).expand(n_heads, -1).clone())
        self.freq2 = nn.Parameter(base_freq2.unsqueeze(0).expand(n_heads, -1).clone())

    def compute_angles(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """
        pos1, pos2: [T]
        returns: [n_heads, T, half]
        """
        p1 = pos1.to(device=self.freq1.device, dtype=torch.float32)
        p2 = pos2.to(device=self.freq2.device, dtype=torch.float32)
        return p1[None, :, None] * self.freq1[:, None, :] + p2[None, :, None] * self.freq2[:, None, :]


def _apply_rope_with_angles(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE using precomputed per-head angles.

    x: [B, H, T, D_h]
    angles: [H, T, D_h/2]
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B,H,T,D_h], got {tuple(x.shape)}")
    D_h = x.shape[-1]
    rope_dim = int(D_h) if int(D_h) % 2 == 0 else int(D_h) - 1
    if rope_dim <= 0:
        return x

    sin = torch.sin(angles).to(dtype=x.dtype).unsqueeze(0)
    cos = torch.cos(angles).to(dtype=x.dtype).unsqueeze(0)

    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    x_rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
    if x_pass.numel() == 0:
        return x_rotated
    return torch.cat([x_rotated, x_pass], dim=-1)


def _rope_attention_with_angles(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_angles: torch.Tensor,
    k_angles: torch.Tensor,
    dropout_p: float,
    training: bool,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mixed 2D RoPE + scaled dot-product attention.

    q/k/v: [B, H, T, D_h] and [B, H, S, D_h]
    q_angles: [H, T, D_h/2]
    k_angles: [H, S, D_h/2]
    """
    q_rot = _apply_rope_with_angles(q, q_angles)
    k_rot = _apply_rope_with_angles(k, k_angles)
    attn_dropout = float(dropout_p) if training else 0.0
    return F.scaled_dot_product_attention(
        q_rot,
        k_rot,
        v,
        attn_mask=attn_mask,
        dropout_p=attn_dropout,
        is_causal=False,
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class NonAffineRMSNorm(nn.Module):
    """Parameter-free RMS normalization used on v4 residual returns and Q/K."""

    def __init__(self, normalized_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if int(normalized_dim) < 1:
            raise ValueError(f"normalized_dim must be positive, got {normalized_dim}")
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.normalized_dim = int(normalized_dim)
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if int(value.shape[-1]) != self.normalized_dim:
            raise ValueError(
                f"last dimension must be {self.normalized_dim}, got {tuple(value.shape)}"
            )
        value_float = value.to(dtype=torch.float32)
        inverse_rms = torch.rsqrt(value_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (value_float * inverse_rms).to(dtype=value.dtype)


def _find_ffn_down_projection(module: nn.Module) -> nn.Linear | None:
    ffn_down = getattr(module, "ffn_down", None)
    if isinstance(ffn_down, nn.Linear):
        return ffn_down

    ffn = getattr(module, "ffn", None)
    if isinstance(ffn, MLP):
        return ffn.fc2

    if isinstance(ffn, nn.Sequential):
        for child in reversed(ffn):
            if isinstance(child, nn.Linear):
                return child
    return None


def _init_vae_module_weights(module: nn.Module, base_std: float = 0.02, init_style: str = "llm") -> None:
    """Explicit initialization for mini-VAE modules (real and stub variants)."""
    style = str(init_style).strip().lower()
    if style not in {"xavier", "llm"}:
        raise ValueError(f"init_style must be 'xavier' or 'llm', got {init_style!r}")
    if float(base_std) <= 0.0:
        raise ValueError(f"base_std must be positive, got {base_std}")

    if isinstance(module, nn.Linear):
        if style == "xavier":
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=float(base_std))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            if style == "xavier":
                nn.init.xavier_uniform_(module.in_proj_weight)
            else:
                nn.init.normal_(module.in_proj_weight, mean=0.0, std=float(base_std))
        else:
            for attr in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
                proj = getattr(module, attr, None)
                if proj is None:
                    continue
                if style == "xavier":
                    nn.init.xavier_uniform_(proj)
                else:
                    nn.init.normal_(proj, mean=0.0, std=float(base_std))
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def _apply_residual_scaled_init(root_module: nn.Module, L_stack: int, base_std: float = 0.02) -> None:
    """
    Downscale residual-branch output projections:
    std_out = base_std / sqrt(2 * L_stack)
    """
    if int(L_stack) <= 0:
        raise ValueError(f"L_stack must be positive, got {L_stack}")
    if float(base_std) <= 0.0:
        raise ValueError(f"base_std must be positive, got {base_std}")
    std_out = float(base_std) / math.sqrt(2.0 * float(L_stack))

    with torch.no_grad():
        for module in root_module.modules():
            is_mha = isinstance(module, nn.MultiheadAttention)
            if is_mha:
                out_proj = getattr(module, "out_proj", None)
                if isinstance(out_proj, nn.Linear):
                    nn.init.normal_(out_proj.weight, mean=0.0, std=std_out)
                    if out_proj.bias is not None:
                        nn.init.zeros_(out_proj.bias)

            for attr in ("out_proj", "cross_out_proj", "self_out_proj"):
                if is_mha and attr == "out_proj":
                    continue
                proj = getattr(module, attr, None)
                if isinstance(proj, nn.Linear):
                    nn.init.normal_(proj.weight, mean=0.0, std=std_out)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

            down_proj = _find_ffn_down_projection(module)
            if isinstance(down_proj, nn.Linear):
                nn.init.normal_(down_proj.weight, mean=0.0, std=std_out)
                if down_proj.bias is not None:
                    nn.init.zeros_(down_proj.bias)


def _init_vae_latent_parameters(module: nn.Module) -> None:
    """Initialize standalone latent parameters that are not covered by module.apply()."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("resampler_latents"):
                nn.init.normal_(param, mean=0.0, std=0.005)


def _decode_direction_and_logscale(
    u_hat: torch.Tensor,
    s_hat: torch.Tensor,
    eps: float,
    s_min: float,
    s_max: float | None = None,
) -> torch.Tensor:
    """
    Convert decoder outputs into weights:
    U = u_hat / (||u_hat||_2 + eps), log_s = bounded(s_hat), s = exp(log_s), W_hat = s * U.

    The bounds are applied in log-space before exp(). This avoids masked overflow
    in the forward graph when s_hat becomes very large.
    """
    if u_hat.ndim != 2:
        raise ValueError(f"u_hat must be [B, p], got {tuple(u_hat.shape)}")
    if s_hat.ndim == 1:
        log_s = s_hat.unsqueeze(-1)
    elif s_hat.ndim == 2 and s_hat.shape[1] == 1:
        log_s = s_hat
    else:
        raise ValueError(f"s_hat must be [B] or [B,1], got {tuple(s_hat.shape)}")
    if tuple(log_s.shape[:1]) != tuple(u_hat.shape[:1]):
        raise ValueError(f"Batch mismatch: u_hat={tuple(u_hat.shape)}, s_hat={tuple(log_s.shape)}")

    u_norm = u_hat.norm(dim=1, keepdim=True).clamp_min(float(eps))
    u = u_hat / u_norm

    if s_min != 0:
        log_s = float(s_min) + F.softplus(log_s - float(s_min))
    if s_max is not None and s_max != 0:
        log_s = float(s_max) - F.softplus(float(s_max) - log_s)
    s = torch.exp(log_s)

    return s * u


class PerceiverResamplerBlock(nn.Module):
    """Perceiver-style resampler block over latent queries and input patch tokens."""

    def __init__(
        self,
        d_latent: int,
        d_token: int,
        n_heads: int,
        dropout: float,
        use_rope_2d: bool = False,
        *,
        qk_norm: bool = False,
        peri_rms_residual: bool = False,
        residual_scale: float = 1.0,
        mandatory_cross_refresh: bool = False,
        cross_refresh_rms: float = 1.0,
    ) -> None:
        super().__init__()
        if d_latent % n_heads != 0:
            raise ValueError(f"d_latent ({d_latent}) must be divisible by n_heads ({n_heads})")
        self.d_latent = int(d_latent)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_latent // n_heads)
        self.use_rope_2d = bool(use_rope_2d)
        self.norm_cross_q = nn.LayerNorm(d_latent)
        self.norm_cross_kv = nn.LayerNorm(d_token)
        self.cross_q_proj = nn.Linear(d_latent, d_latent)
        self.cross_k_proj = nn.Linear(d_token, d_latent)
        self.cross_v_proj = nn.Linear(d_token, d_latent)
        self.cross_out_proj = nn.Linear(d_latent, d_latent)
        self.norm_self = nn.LayerNorm(d_latent)
        self.self_q_proj = nn.Linear(d_latent, d_latent)
        self.self_k_proj = nn.Linear(d_latent, d_latent)
        self.self_v_proj = nn.Linear(d_latent, d_latent)
        self.self_out_proj = nn.Linear(d_latent, d_latent)
        self.norm_ffn = nn.LayerNorm(d_latent)
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, 4 * d_latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_latent, d_latent),
            nn.Dropout(dropout),
        )
        self.attn_prob_dropout_p = float(dropout)
        self.dropout = nn.Dropout(dropout)
        self.qk_norm_enabled = bool(qk_norm)
        self.peri_rms_residual = bool(peri_rms_residual)
        self.residual_scale = float(residual_scale)
        self.mandatory_cross_refresh = bool(mandatory_cross_refresh)
        self.cross_refresh_rms = float(cross_refresh_rms)
        if self.residual_scale <= 0.0:
            raise ValueError(f"residual_scale must be positive, got {residual_scale}")
        if not math.isfinite(self.cross_refresh_rms) or self.cross_refresh_rms <= 0.0:
            raise ValueError(f"cross_refresh_rms must be finite and positive, got {cross_refresh_rms}")
        self.cross_q_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.cross_k_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.self_q_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.self_k_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.cross_return_norm = NonAffineRMSNorm(d_latent) if self.peri_rms_residual else nn.Identity()
        self.self_return_norm = NonAffineRMSNorm(d_latent) if self.peri_rms_residual else nn.Identity()
        self.ffn_return_norm = NonAffineRMSNorm(d_latent) if self.peri_rms_residual else nn.Identity()
        if self.use_rope_2d:
            self.rope_2d_cross = RoPeMixed2D(n_heads, self.head_dim)
        self.mandatory_cross_refresh_frozen_parameter_names: tuple[str, ...] = ()
        if self.mandatory_cross_refresh:
            frozen: list[str] = []
            for name, parameter in (
                ("norm_cross_kv.bias", self.norm_cross_kv.bias),
                ("cross_v_proj.bias", self.cross_v_proj.bias),
                ("cross_out_proj.bias", self.cross_out_proj.bias),
            ):
                if parameter is None:
                    raise RuntimeError(f"mandatory cross refresh requires {name}")
                with torch.no_grad():
                    parameter.zero_()
                parameter.requires_grad_(False)
                frozen.append(name)
            for module_name, module in (
                ("norm_self", self.norm_self),
                ("self_q_proj", self.self_q_proj),
                ("self_k_proj", self.self_k_proj),
                ("self_v_proj", self.self_v_proj),
                ("self_out_proj", self.self_out_proj),
                ("norm_ffn", self.norm_ffn),
                ("ffn", self.ffn),
            ):
                for parameter_name, parameter in module.named_parameters():
                    parameter.requires_grad_(False)
                    frozen.append(f"{module_name}.{parameter_name}")
            self.mandatory_cross_refresh_frozen_parameter_names = tuple(sorted(frozen))

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        latent_pos: torch.Tensor,
        token_pos: torch.Tensor,
        token_pos2: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_2d = self.use_rope_2d and token_pos2 is not None
        q = self.norm_cross_q(latents)
        kv = self.norm_cross_kv(tokens)
        B, L_lat, _ = q.shape
        L_tok = kv.shape[1]
        cross_attn_mask = (
            _key_padding_to_attn_bias(token_mask.to(device=kv.device, dtype=torch.bool), dtype=q.dtype)
            if token_mask is not None
            else None
        )
        q_cross = self.cross_q_proj(q).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        k_cross = self.cross_k_proj(kv).view(B, L_tok, self.n_heads, self.head_dim).transpose(1, 2)
        v_cross = self.cross_v_proj(kv).view(B, L_tok, self.n_heads, self.head_dim).transpose(1, 2)
        q_cross = self.cross_q_norm(q_cross)
        k_cross = self.cross_k_norm(k_cross)

        if use_2d:
            q_angles = self.rope_2d_cross.compute_angles(latent_pos, latent_pos)
            k_angles = self.rope_2d_cross.compute_angles(token_pos, token_pos2)
            cross_out = _rope_attention_with_angles(
                q=q_cross,
                k=k_cross,
                v=v_cross,
                q_angles=q_angles,
                k_angles=k_angles,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=cross_attn_mask,
            )
        else:
            cross_out = _rope_attention(
                q=q_cross,
                k=k_cross,
                v=v_cross,
                q_pos=latent_pos,
                k_pos=token_pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=cross_attn_mask,
            )
        cross_out = cross_out.transpose(1, 2).contiguous().view(B, L_lat, self.d_latent)
        cross_out = self.cross_out_proj(cross_out)
        cross_return = self.dropout(cross_out)
        if self.mandatory_cross_refresh:
            return self.cross_return_norm(cross_return) * self.cross_refresh_rms
        if self.peri_rms_residual:
            cross_return = self.cross_return_norm(cross_return) * self.residual_scale
        latents = latents + cross_return

        lat_norm = self.norm_self(latents)
        q_self = self.self_q_proj(lat_norm).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        k_self = self.self_k_proj(lat_norm).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        v_self = self.self_v_proj(lat_norm).view(B, L_lat, self.n_heads, self.head_dim).transpose(1, 2)
        q_self = self.self_q_norm(q_self)
        k_self = self.self_k_norm(k_self)
        self_out = _rope_attention(
            q=q_self,
            k=k_self,
            v=v_self,
            q_pos=latent_pos,
            k_pos=latent_pos,
            dropout_p=self.attn_prob_dropout_p,
            training=self.training,
        )
        self_out = self_out.transpose(1, 2).contiguous().view(B, L_lat, self.d_latent)
        self_out = self.self_out_proj(self_out)
        self_return = self.dropout(self_out)
        if self.peri_rms_residual:
            self_return = self.self_return_norm(self_return) * self.residual_scale
        latents = latents + self_return

        ffn_return = self.dropout(self.ffn(self.norm_ffn(latents)))
        if self.peri_rms_residual:
            ffn_return = self.ffn_return_norm(ffn_return) * self.residual_scale
        latents = latents + ffn_return
        return latents


class TokenSummarizer(nn.Module):
    """Summarize a variable-length token set into a fixed number of output tokens."""

    def __init__(
        self,
        dim: int,
        out_tokens: int,
        mode: str = "mlp",
        hidden_mult: float = 2.0,
        dropout: float = 0.0,
        use_input_norm: bool = True,
    ) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if int(out_tokens) <= 0:
            raise ValueError(f"out_tokens must be positive, got {out_tokens}")

        mode_value = str(mode).strip().lower()
        if mode_value not in {"mlp", "latent_query"}:
            raise ValueError(f"mode must be 'mlp' or 'latent_query', got {mode!r}")

        self.dim = int(dim)
        self.out_tokens = int(out_tokens)
        self.mode = mode_value
        self.input_norm = nn.LayerNorm(self.dim) if bool(use_input_norm) else nn.Identity()

        if self.mode == "mlp":
            hidden_dim = max(1, int(float(hidden_mult) * self.dim))
            self.hidden = nn.Linear(self.dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, self.out_tokens)
            self.dropout = nn.Dropout(dropout)
            self.lat_queries = None
        else:
            self.hidden = None
            self.out_proj = None
            self.dropout = nn.Dropout(0.0)
            self.lat_queries = nn.Parameter(torch.empty(self.out_tokens, self.dim))
            nn.init.normal_(self.lat_queries, mean=0.0, std=0.02)

    def forward(self, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # V: [B, P, D]
        if V.ndim != 3:
            raise ValueError(f"V must be [B, P, D], got {tuple(V.shape)}")
        if int(V.shape[-1]) != self.dim:
            raise ValueError(f"V last dim ({int(V.shape[-1])}) must equal dim ({self.dim})")

        V_norm = self.input_norm(V)
        if self.mode == "mlp":
            assert self.hidden is not None and self.out_proj is not None
            hidden = self.dropout(F.gelu(self.hidden(V_norm)))
            logits = self.out_proj(hidden).permute(0, 2, 1)
        else:
            assert self.lat_queries is not None
            scale = float(self.dim) ** -0.5
            logits = torch.einsum("kd,bpd->bkp", self.lat_queries, V_norm) * scale

        A = torch.softmax(logits, dim=-1)
        Z = torch.matmul(A, V)
        return Z, A


class _TransformerProcessLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        hidden_dim = max(1, int(float(mlp_ratio) * int(dim)))
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B, R, D]
        attn_in = self.norm_attn(Z)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        Z = Z + self.attn_dropout(attn_out)
        Z = Z + self.mlp(self.norm_mlp(Z))
        return Z


class TransformerProcess(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerProcessLayer(
                    dim=int(dim),
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(max(0, int(depth)))
            ]
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B, R, D]
        for layer in self.layers:
            Z = layer(Z)
        return Z


class TTMMemoryBlock(nn.Module):
    """Token Turing Machine latent-memory block with summarization read/write."""

    def __init__(
        self,
        dim: int,
        mem_tokens: int,
        proc_tokens: int,
        process_depth: int,
        num_heads: int,
        summarizer_mode: str = "mlp",
        summarizer_hidden_mult: float = 2.0,
        dropout: float = 0.0,
        use_type_embeddings: bool = True,
        use_positional_embeddings: bool = True,
        memory_init: str = "learned",
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        memory_init_value = str(memory_init).strip().lower()
        if memory_init_value not in {"learned", "zeros"}:
            raise ValueError(f"memory_init must be 'learned' or 'zeros', got {memory_init!r}")

        self.dim = int(dim)
        self.mem_tokens = int(mem_tokens)
        self.proc_tokens = int(proc_tokens)
        self.use_type_embeddings = bool(use_type_embeddings)
        self.use_positional_embeddings = bool(use_positional_embeddings)
        self.memory_init = memory_init_value
        self.return_aux = bool(return_aux)
        self.max_input_tokens = 4096

        self.read_summarizer = TokenSummarizer(
            dim=self.dim,
            out_tokens=self.proc_tokens,
            mode=summarizer_mode,
            hidden_mult=float(summarizer_hidden_mult),
            dropout=float(dropout),
            use_input_norm=True,
        )
        self.process = TransformerProcess(
            dim=self.dim,
            depth=int(process_depth),
            num_heads=int(num_heads),
            mlp_ratio=4.0,
            dropout=float(dropout),
        )
        self.write_summarizer = TokenSummarizer(
            dim=self.dim,
            out_tokens=self.mem_tokens,
            mode=summarizer_mode,
            hidden_mult=float(summarizer_hidden_mult),
            dropout=float(dropout),
            use_input_norm=True,
        )

        if self.memory_init == "learned":
            self.memory_seed = nn.Parameter(torch.empty(1, self.mem_tokens, self.dim))
            nn.init.normal_(self.memory_seed, mean=0.0, std=0.02)
        else:
            self.register_parameter("memory_seed", None)

        if self.use_type_embeddings:
            self.type_mem = nn.Parameter(torch.empty(1, 1, self.dim))
            self.type_in = nn.Parameter(torch.empty(1, 1, self.dim))
            self.type_out = nn.Parameter(torch.empty(1, 1, self.dim))
            nn.init.normal_(self.type_mem, mean=0.0, std=0.02)
            nn.init.normal_(self.type_in, mean=0.0, std=0.02)
            nn.init.normal_(self.type_out, mean=0.0, std=0.02)
        else:
            self.register_parameter("type_mem", None)
            self.register_parameter("type_in", None)
            self.register_parameter("type_out", None)

        if self.use_positional_embeddings:
            self.mem_pos = nn.Parameter(torch.empty(1, self.mem_tokens, self.dim))
            self.input_pos_emb = nn.Parameter(torch.empty(1, self.max_input_tokens, self.dim))
            nn.init.normal_(self.mem_pos, mean=0.0, std=0.02)
            nn.init.normal_(self.input_pos_emb, mean=0.0, std=0.02)
        else:
            self.register_parameter("mem_pos", None)
            self.register_parameter("input_pos_emb", None)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if self.memory_init == "learned":
            assert self.memory_seed is not None
            memory = self.memory_seed.expand(int(batch_size), -1, -1)
            return memory.to(device=device if device is not None else memory.device, dtype=dtype or memory.dtype)

        if device is None:
            if self.mem_pos is not None:
                device = self.mem_pos.device
            else:
                device = torch.device("cpu")
        if dtype is None:
            if self.mem_pos is not None:
                dtype = self.mem_pos.dtype
            else:
                dtype = torch.float32
        return torch.zeros((int(batch_size), self.mem_tokens, self.dim), device=device, dtype=dtype)

    def _resolve_input_pos(self, X: torch.Tensor, input_pos: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_positional_embeddings:
            return None

        B, N, _ = X.shape
        if input_pos is not None:
            if input_pos.ndim != 3:
                raise ValueError(f"input_pos must be [1 or B, N, D], got {tuple(input_pos.shape)}")
            if int(input_pos.shape[-1]) != self.dim or int(input_pos.shape[1]) != N:
                raise ValueError(
                    f"input_pos must be [1 or B, {N}, {self.dim}], got {tuple(input_pos.shape)}"
                )
            if int(input_pos.shape[0]) == 1:
                return input_pos.expand(B, -1, -1)
            if int(input_pos.shape[0]) == B:
                return input_pos
            raise ValueError(f"input_pos batch must be 1 or {B}, got {int(input_pos.shape[0])}")

        if N > self.max_input_tokens:
            raise ValueError(
                f"input length ({N}) exceeds learned input_pos_emb capacity ({self.max_input_tokens})"
            )
        assert self.input_pos_emb is not None
        return self.input_pos_emb[:, :N, :].expand(B, -1, -1)

    def forward(
        self,
        X: torch.Tensor,
        M: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # X: [B, N, D], M: [B, K, D]
        if X.ndim != 3:
            raise ValueError(f"X must be [B, N, D], got {tuple(X.shape)}")
        if int(X.shape[-1]) != self.dim:
            raise ValueError(f"X last dim ({int(X.shape[-1])}) must equal dim ({self.dim})")

        B = int(X.shape[0])
        if M is None:
            M = self.init_memory(batch_size=B, device=X.device, dtype=X.dtype)
        if M.ndim != 3:
            raise ValueError(f"M must be [B, K, D], got {tuple(M.shape)}")
        if tuple(M.shape[1:]) != (self.mem_tokens, self.dim):
            raise ValueError(
                f"M must be [B, {self.mem_tokens}, {self.dim}], got {tuple(M.shape)}"
            )
        if int(M.shape[0]) != B:
            raise ValueError(f"M batch ({int(M.shape[0])}) must match X batch ({B})")

        resolved_input_pos = self._resolve_input_pos(X, input_pos)

        X_read = X
        M_read = M
        if self.use_positional_embeddings:
            assert self.mem_pos is not None
            M_read = M_read + self.mem_pos[:, : self.mem_tokens, :]
            if resolved_input_pos is not None:
                X_read = X_read + resolved_input_pos
        if self.use_type_embeddings:
            assert self.type_mem is not None and self.type_in is not None
            M_read = M_read + self.type_mem
            X_read = X_read + self.type_in

        V_read = torch.cat([M_read, X_read], dim=1)
        Z, A_read = self.read_summarizer(V_read)
        O = self.process(Z)

        X_write = X
        M_write = M
        O_write = O
        if self.use_positional_embeddings:
            assert self.mem_pos is not None
            M_write = M_write + self.mem_pos[:, : self.mem_tokens, :]
            if resolved_input_pos is not None:
                X_write = X_write + resolved_input_pos
        if self.use_type_embeddings:
            assert self.type_mem is not None and self.type_in is not None and self.type_out is not None
            M_write = M_write + self.type_mem
            X_write = X_write + self.type_in
            O_write = O_write + self.type_out

        V_write = torch.cat([M_write, O_write, X_write], dim=1)
        M_new, A_write = self.write_summarizer(V_write)

        if not self.return_aux:
            return M_new
        aux = {
            "read_tokens": Z,
            "process_tokens": O,
            "read_weights": A_read,
            "write_weights": A_write,
        }
        return M_new, aux


class TTMMemoryStack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        block: TTMMemoryBlock,
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        if int(num_blocks) <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self.num_blocks = int(num_blocks)
        self.share_weights = bool(share_weights)
        if self.share_weights:
            self.shared_block = block
            self.blocks = None
        else:
            self.shared_block = None
            self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(self.num_blocks)])

    def _get_block(self, index: int) -> TTMMemoryBlock:
        if self.share_weights:
            assert self.shared_block is not None
            return self.shared_block
        assert self.blocks is not None
        return self.blocks[index]

    def forward(
        self,
        X: torch.Tensor,
        M: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, list[dict[str, torch.Tensor]]]]:
        # X: [B, N, D], M: [B, K, D]
        block0 = self._get_block(0)
        if M is None:
            M = block0.init_memory(batch_size=int(X.shape[0]), device=X.device, dtype=X.dtype)

        aux_list: list[dict[str, torch.Tensor]] = []
        for block_idx in range(self.num_blocks):
            block = self._get_block(block_idx)
            block_out = block(X, M=M, input_pos=input_pos)
            if isinstance(block_out, tuple):
                M, aux = block_out
                aux_list.append(aux)
            else:
                M = block_out

        if not aux_list:
            return M
        return M, {"blocks": aux_list}


class MandatoryLatentBridge(nn.Module):
    """One mandatory X-query/z-value read with a direct value gradient to z.

    The X-conditioned query is used only to form routing weights.  It is never
    added to the returned decoder state, so there is no query residual that can
    bypass the encoder latents.
    """

    def __init__(
        self,
        *,
        d_query: int,
        d_latent: int,
        d_output: int,
        attn_dim: int,
        n_heads: int,
        dropout: float,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if int(attn_dim) <= 0 or int(attn_dim) % int(n_heads) != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be positive and divisible by n_heads ({n_heads})"
            )
        if int(attn_dim) < int(d_latent):
            raise ValueError(
                "attn_dim must be at least d_latent so the direct value projection "
                f"can be full-rank, got attn_dim={attn_dim} d_latent={d_latent}"
            )
        self.attn_dim = int(attn_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.attn_dim // self.n_heads
        self.query_norm = nn.LayerNorm(int(d_query))
        # Bias-free projections prevent a sample-common value template.  All
        # four matrices are deliberately non-zero at initialization.
        self.q_proj = nn.Linear(int(d_query), self.attn_dim, bias=False)
        self.k_proj = nn.Linear(int(d_latent), self.attn_dim, bias=False)
        self.v_proj = nn.Linear(int(d_latent), self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, int(d_output), bias=False)
        self.q_norm = NonAffineRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = NonAffineRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_dropout_p = float(dropout)
        self.output_dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(projection.weight)

    def forward(
        self,
        query: torch.Tensor,
        latent_slots: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or latent_slots.ndim != 3:
            raise ValueError(
                "mandatory bridge expects query and latent_slots to be rank-3, "
                f"got {tuple(query.shape)} and {tuple(latent_slots.shape)}"
            )
        if int(query.shape[0]) != int(latent_slots.shape[0]):
            raise ValueError("mandatory bridge query and latent batch sizes must match")
        B, Q, _ = query.shape
        L = int(latent_slots.shape[1])
        q = self.q_proj(self.query_norm(query)).view(B, Q, self.n_heads, self.head_dim)
        k = self.k_proj(latent_slots).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(latent_slots).view(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(B, Q, self.attn_dim)
        output = self.output_dropout(self.out_proj(attended))
        return _apply_sequence_mask(output, query_mask)


class FixedPositionOnlyFiLM(nn.Module):
    """Parameter-free, zero-preserving 2D position modulation.

    The modulation is a deterministic function of query coordinates only.  It
    cannot carry sample, X, W, or z content, and multiplication preserves an
    exactly zero mandatory-bridge output.
    """

    def __init__(
        self,
        d_model: int,
        *,
        strength: float = 0.1,
        clamp_value: float = 2.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if float(strength) != 0.1:
            raise ValueError("V6 fixed position FiLM strength is frozen to exactly 0.1")
        if float(clamp_value) != 2.0:
            raise ValueError("V6 fixed position FiLM clamp is frozen to exactly 2.0")
        if float(eps) <= 0.0:
            raise ValueError("V6 fixed position FiLM eps must be positive")
        self.d_model = int(d_model)
        self.strength = float(strength)
        self.clamp_value = float(clamp_value)
        self.eps = float(eps)

    def position_modulation(
        self,
        pos_o: torch.Tensor,
        pos_t: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if pos_o.ndim != 1 or pos_t.ndim != 1 or tuple(pos_o.shape) != tuple(pos_t.shape):
            raise ValueError(
                "V6 position FiLM expects matching rank-1 2D coordinate vectors, "
                f"got {tuple(pos_o.shape)} and {tuple(pos_t.shape)}"
            )
        bands = (self.d_model + 3) // 4
        frequencies = torch.arange(
            1,
            bands + 1,
            device=pos_o.device,
            dtype=torch.float32,
        )
        angles_o = (2.0 * math.pi) * pos_o.to(torch.float32).unsqueeze(-1) * frequencies
        angles_t = (2.0 * math.pi) * pos_t.to(torch.float32).unsqueeze(-1) * frequencies
        features = torch.cat(
            [angles_o.sin(), angles_o.cos(), angles_t.sin(), angles_t.cos()],
            dim=-1,
        )[:, : self.d_model]
        centered = features - features.mean(dim=0, keepdim=True)
        normalized = centered * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.clamp(min=-self.clamp_value, max=self.clamp_value).to(dtype=dtype)

    def forward(
        self,
        bridge_output: torch.Tensor,
        *,
        pos_o: torch.Tensor,
        pos_t: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bridge_output.ndim != 3 or int(bridge_output.shape[-1]) != self.d_model:
            raise ValueError(
                "V6 position FiLM expects bridge_output [B,Q,d_model], got "
                f"{tuple(bridge_output.shape)}"
            )
        if int(pos_o.numel()) != int(bridge_output.shape[1]):
            raise ValueError(
                "V6 position FiLM coordinate count must match query count, got "
                f"{int(pos_o.numel())} and {int(bridge_output.shape[1])}"
            )
        modulation = self.position_modulation(pos_o, pos_t, dtype=bridge_output.dtype)
        output = bridge_output * (1.0 + self.strength * modulation.unsqueeze(0))
        return _apply_sequence_mask(output, query_mask)


class DecoderSelfAttentionBlock(nn.Module):
    """Standard Pre-Norm decoder self-attention + FFN block for V5."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float,
        use_rope_2d: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.use_rope_2d = bool(use_rope_2d)
        self.self_attn_norm = nn.LayerNorm(self.d_model)
        self.self_q_proj = nn.Linear(self.d_model, self.d_model)
        self.self_k_proj = nn.Linear(self.d_model, self.d_model)
        self.self_v_proj = nn.Linear(self.d_model, self.d_model)
        self.self_out_proj = nn.Linear(self.d_model, self.d_model)
        self.self_attn_out_dropout = nn.Dropout(float(dropout))
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.d_model, self.d_model),
            nn.Dropout(float(dropout)),
        )
        self.attn_prob_dropout_p = float(dropout)
        self.self_q_norm = NonAffineRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.self_k_norm = NonAffineRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.rope_2d_self = RoPeMixed2D(self.n_heads, self.head_dim) if self.use_rope_2d else None
        self._init_nonreturn_projections()

    def _init_nonreturn_projections(self) -> None:
        for projection in (self.self_q_proj, self.self_k_proj, self.self_v_proj, self.ffn[0]):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(
        self,
        q: torch.Tensor,
        *,
        q_pos: torch.Tensor,
        q_pos2: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_self = self.self_attn_norm(q)
        B, Tq, _ = q_self.shape
        self_q = self.self_q_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_k = self.self_k_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_v = self.self_v_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_q = self.self_q_norm(self_q)
        self_k = self.self_k_norm(self_k)
        self_attn_mask = (
            _key_padding_to_attn_bias(q_mask.to(device=q.device, dtype=torch.bool), dtype=q.dtype)
            if q_mask is not None
            else None
        )
        if self.use_rope_2d and q_pos2 is not None:
            assert self.rope_2d_self is not None
            angles = self.rope_2d_self.compute_angles(q_pos, q_pos2)
            self_out = _rope_attention_with_angles(
                q=self_q,
                k=self_k,
                v=self_v,
                q_angles=angles,
                k_angles=angles,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=self_attn_mask,
            )
        else:
            self_out = _rope_attention(
                q=self_q,
                k=self_k,
                v=self_v,
                q_pos=q_pos,
                k_pos=q_pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=self_attn_mask,
            )
        self_out = self_out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        q = q + self.self_attn_out_dropout(self.self_out_proj(self_out))
        q = _apply_sequence_mask(q, q_mask)
        q = q + self.ffn(self.ffn_norm(q))
        return _apply_sequence_mask(q, q_mask)


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        use_rope_2d: bool = False,
        *,
        qk_norm: bool = False,
        peri_rms_residual: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.use_rope_2d = bool(use_rope_2d)
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_prob_dropout_p = float(dropout)
        self.attn_out_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_q_proj = nn.Linear(d_model, d_model)
        self.self_k_proj = nn.Linear(d_model, d_model)
        self.self_v_proj = nn.Linear(d_model, d_model)
        self.self_out_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.qk_norm_enabled = bool(qk_norm)
        self.peri_rms_residual = bool(peri_rms_residual)
        self.residual_scale = float(residual_scale)
        if self.residual_scale <= 0.0:
            raise ValueError(f"residual_scale must be positive, got {residual_scale}")
        self.cross_q_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.cross_k_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.self_q_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.self_k_norm = NonAffineRMSNorm(self.head_dim) if self.qk_norm_enabled else nn.Identity()
        self.cross_return_norm = NonAffineRMSNorm(d_model) if self.peri_rms_residual else nn.Identity()
        self.self_return_norm = NonAffineRMSNorm(d_model) if self.peri_rms_residual else nn.Identity()
        self.ffn_return_norm = NonAffineRMSNorm(d_model) if self.peri_rms_residual else nn.Identity()
        if self.use_rope_2d:
            self.rope_2d_cross = RoPeMixed2D(n_heads, self.head_dim)
            self.rope_2d_self = RoPeMixed2D(n_heads, self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_pos: torch.Tensor,
        kv_pos: torch.Tensor,
        q_pos2: torch.Tensor | None = None,
        kv_pos2: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_2d = self.use_rope_2d and q_pos2 is not None
        q_attn = self.q_norm(q)
        kv_attn = self.kv_norm(kv)

        B, Tq, _ = q_attn.shape
        Tk = kv_attn.shape[1]
        cross_attn_mask = (
            _key_padding_to_attn_bias(kv_mask.to(device=kv_attn.device, dtype=torch.bool), dtype=q_attn.dtype)
            if kv_mask is not None
            else None
        )

        q_proj = self.q_proj(q_attn).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(kv_attn).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(kv_attn).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        q_proj = self.cross_q_norm(q_proj)
        k_proj = self.cross_k_norm(k_proj)

        if use_2d:
            effective_kv_pos2 = kv_pos2 if kv_pos2 is not None else kv_pos
            q_angles = self.rope_2d_cross.compute_angles(q_pos, q_pos2)
            k_angles = self.rope_2d_cross.compute_angles(kv_pos, effective_kv_pos2)
            attn_out = _rope_attention_with_angles(
                q=q_proj,
                k=k_proj,
                v=v_proj,
                q_angles=q_angles,
                k_angles=k_angles,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=cross_attn_mask,
            )
        else:
            attn_out = _rope_attention(
                q=q_proj,
                k=k_proj,
                v=v_proj,
                q_pos=q_pos,
                k_pos=kv_pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=cross_attn_mask,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        attn_out = self.out_proj(attn_out)

        cross_return = self.attn_out_dropout(attn_out)
        if self.peri_rms_residual:
            cross_return = self.cross_return_norm(cross_return) * self.residual_scale
        q = q + cross_return
        q = _apply_sequence_mask(q, q_mask)
        q_self = self.self_attn_norm(q)
        self_q = self.self_q_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_k = self.self_k_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_v = self.self_v_proj(q_self).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        self_q = self.self_q_norm(self_q)
        self_k = self.self_k_norm(self_k)
        self_attn_mask = (
            _key_padding_to_attn_bias(q_mask.to(device=q.device, dtype=torch.bool), dtype=q.dtype)
            if q_mask is not None
            else None
        )

        if use_2d:
            self_angles = self.rope_2d_self.compute_angles(q_pos, q_pos2)
            self_out = _rope_attention_with_angles(
                q=self_q,
                k=self_k,
                v=self_v,
                q_angles=self_angles,
                k_angles=self_angles,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=self_attn_mask,
            )
        else:
            self_out = _rope_attention(
                q=self_q,
                k=self_k,
                v=self_v,
                q_pos=q_pos,
                k_pos=q_pos,
                dropout_p=self.attn_prob_dropout_p,
                training=self.training,
                attn_mask=self_attn_mask,
            )
        self_out = self_out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        self_out = self.self_out_proj(self_out)
        self_return = self.self_attn_out_dropout(self_out)
        if self.peri_rms_residual:
            self_return = self.self_return_norm(self_return) * self.residual_scale
        q = q + self_return
        q = _apply_sequence_mask(q, q_mask)
        ffn_return = self.ffn(self.ffn_norm(q))
        if self.peri_rms_residual:
            ffn_return = self.ffn_return_norm(ffn_return) * self.residual_scale
        q = q + ffn_return
        return _apply_sequence_mask(q, q_mask)


__all__ = [
    "CleanContentReadoutV8",
    "CrossAttnBlock",
    "DecoderSelfAttentionBlock",
    "HybridNonlinearContentReadoutV9",
    "MandatoryLatentBridge",
    "MLP",
    "NonAffineRMSNorm",
    "NonlinearContentRefinementBlockV9",
    "PerceiverResamplerBlock",
    "RoPeMixed2D",
    "TTMMemoryBlock",
    "TTMMemoryStack",
    "TokenSummarizer",
    "TransformerProcess",
    "sinusoidal_embedding",
]
