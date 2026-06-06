from __future__ import annotations

from .stores import *
from .types import *

class FunctionalViTTiny(nn.Module):
    def __init__(
        self,
        vit_cfg: ViTTinyConfig,
        initial_tensors: dict[str, torch.Tensor],
        *,
        parameter_mode: str,
        latent_rank: int = 8,
        latent_delta_scale: float = 1.0,
        latent_factor_init_std: float = 0.02,
        big_vae: BigWeightVAE | None = None,
        big_vae_latent_init: str = "random",
        big_vae_latent_space: str | None = None,
        big_vae_diffusion_prior: Any | None = None,
        big_vae_diffusion_prior_steps: int = 50,
        big_vae_diffusion_prior_sampler: str = "ddim",
        big_vae_diffusion_prior_eta: float = 0.0,
        big_vae_random_init_std: float = 0.02,
        big_vae_latent_noise_std: float = 0.0,
        big_vae_latent_parameterization: str = "euclidean",
        big_vae_encoder_context_rows: int = 64,
        big_vae_encoder_context_std: float = 1.0,
        big_vae_encoder_batch_size: int = 16,
        big_vae_decode: str = "weights",
        big_vae_tile_T_patches: int = 16,
        big_vae_tile_d_out: int = 8,
    ) -> None:
        super().__init__()
        self.cfg = vit_cfg
        if parameter_mode == "bigvae_latent":
            if big_vae is None:
                raise ValueError("big_vae is required when parameter_mode='bigvae_latent'")
            self.store = BigVAELatentTensorStore(
                initial_tensors,
                big_vae=big_vae,
                latent_init=big_vae_latent_init,
                latent_space=big_vae_latent_space,
                latent_parameterization=big_vae_latent_parameterization,
                random_init_std=big_vae_random_init_std,
                latent_noise_std=big_vae_latent_noise_std,
                decode_policy=big_vae_decode,
                tile_T_patches=int(big_vae_tile_T_patches),
                tile_d_out=int(big_vae_tile_d_out),
                latent_diffusion_prior=big_vae_diffusion_prior,
                latent_diffusion_prior_steps=int(big_vae_diffusion_prior_steps),
                latent_diffusion_prior_sampler=str(big_vae_diffusion_prior_sampler),
                latent_diffusion_prior_eta=float(big_vae_diffusion_prior_eta),
                encoder_context_rows=int(big_vae_encoder_context_rows),
                encoder_context_std=float(big_vae_encoder_context_std),
                encoder_batch_size=int(big_vae_encoder_batch_size),
            )
        else:
            store_mode = "latent" if parameter_mode == "lowrank_latent" else parameter_mode
            self.store = TensorStore(
                initial_tensors,
                mode=store_mode,
                latent_rank=latent_rank,
                latent_delta_scale=latent_delta_scale,
                latent_factor_init_std=latent_factor_init_std,
            )
        self._decoded_tensor_cache: dict[str, torch.Tensor] | None = None

    def w(self, name: str) -> torch.Tensor:
        if self._decoded_tensor_cache is not None and name in self._decoded_tensor_cache:
            return self._decoded_tensor_cache[name]
        return self.store.tensor(name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if isinstance(self.store, BigVAELatentTensorStore):
            self._decoded_tensor_cache = self.store.decode_all_tensors()
        else:
            self._decoded_tensor_cache = None
        try:
            return self._forward_with_cached_weights(images)
        finally:
            self._decoded_tensor_cache = None

    def _forward_with_cached_weights(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = F.conv2d(
            images,
            self.w("patch_embed.weight"),
            self.w("patch_embed.bias"),
            stride=cfg.patch_size,
        )
        x = x.flatten(2).transpose(1, 2).contiguous()
        batch = int(x.shape[0])
        cls = self.w("cls_token").expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.w("pos_embed")
        x = F.dropout(x, p=cfg.dropout, training=self.training)

        for layer_idx in range(cfg.depth):
            x = self._block(x, layer_idx)

        x = F.layer_norm(
            x,
            (cfg.hidden_dim,),
            weight=self.w("norm.weight"),
            bias=self.w("norm.bias"),
            eps=cfg.layer_norm_eps,
        )
        cls_out = x[:, 0]
        return F.linear(cls_out, self.w("head.weight"), self.w("head.bias"))

    def _block(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.cfg
        prefix = f"blocks.{layer_idx}"
        h = F.layer_norm(
            x,
            (cfg.hidden_dim,),
            weight=self.w(f"{prefix}.norm1.weight"),
            bias=self.w(f"{prefix}.norm1.bias"),
            eps=cfg.layer_norm_eps,
        )
        qkv = F.linear(h, self.w(f"{prefix}.attn.qkv.weight"), self.w(f"{prefix}.attn.qkv.bias"))
        batch, tokens, _ = qkv.shape
        head_dim = cfg.hidden_dim // cfg.num_heads
        qkv = qkv.view(batch, tokens, 3, cfg.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=cfg.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, tokens, cfg.hidden_dim)
        attn = F.linear(attn, self.w(f"{prefix}.attn.proj.weight"), self.w(f"{prefix}.attn.proj.bias"))
        attn = F.dropout(attn, p=cfg.dropout, training=self.training)
        x = x + attn

        h = F.layer_norm(
            x,
            (cfg.hidden_dim,),
            weight=self.w(f"{prefix}.norm2.weight"),
            bias=self.w(f"{prefix}.norm2.bias"),
            eps=cfg.layer_norm_eps,
        )
        h = F.linear(h, self.w(f"{prefix}.mlp.fc1.weight"), self.w(f"{prefix}.mlp.fc1.bias"))
        h = F.gelu(h)
        h = F.dropout(h, p=cfg.dropout, training=self.training)
        h = F.linear(h, self.w(f"{prefix}.mlp.fc2.weight"), self.w(f"{prefix}.mlp.fc2.bias"))
        h = F.dropout(h, p=cfg.dropout, training=self.training)
        return x + h

    @torch.no_grad()
    def initialize_bigvae_diffusion_prior(self, calibration_images: torch.Tensor) -> None:
        if isinstance(self.store, BigVAELatentTensorStore):
            self.store.initialize_from_diffusion_prior_autoregressive(
                self,
                calibration_images=calibration_images,
            )

    @torch.no_grad()
    def collect_parameter_input_matrix(self, tensor_name: str, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.store, BigVAELatentTensorStore):
            raise TypeError("parameter activation collection is only defined for BigVAELatentTensorStore-backed models")

        cfg = self.cfg
        device = next(self.parameters()).device
        images = images.to(device=device, non_blocking=False)
        hidden_dim = int(cfg.hidden_dim)
        mlp_dim = int(round(cfg.hidden_dim * cfg.mlp_ratio))
        target = str(tensor_name)

        was_training = self.training
        if was_training:
            self.eval()
        if isinstance(self.store, BigVAELatentTensorStore):
            self._decoded_tensor_cache = self.store.decode_all_tensors()
        else:
            self._decoded_tensor_cache = None

        try:
            patch_weight = self.w("patch_embed.weight")
            patch_bias = self.w("patch_embed.bias")
            patch_tokens = F.unfold(images, kernel_size=cfg.patch_size, stride=cfg.patch_size).transpose(1, 2).contiguous()
            if target == "patch_embed.weight":
                return patch_tokens.reshape(-1, patch_tokens.shape[-1]).contiguous()

            patch_pre = F.conv2d(images, patch_weight, None, stride=cfg.patch_size)
            x = patch_pre.flatten(2).transpose(1, 2).contiguous()
            if target == "patch_embed.bias":
                return x.reshape(-1, hidden_dim).contiguous()
            x = x + patch_bias.view(1, 1, -1)

            batch = int(x.shape[0])
            cls = self.w("cls_token").expand(batch, -1, -1)
            if target == "cls_token":
                return cls.reshape(batch, hidden_dim).contiguous()
            x = torch.cat((cls, x), dim=1)

            if target == "pos_embed":
                return x.reshape(batch, -1).contiguous()
            x = x + self.w("pos_embed")
            x = F.dropout(x, p=cfg.dropout, training=self.training)

            for layer_idx in range(cfg.depth):
                prefix = f"blocks.{layer_idx}"

                if target == f"{prefix}.norm1.weight" or target == f"{prefix}.norm1.bias":
                    return x.reshape(-1, hidden_dim).contiguous()
                h = F.layer_norm(
                    x,
                    (cfg.hidden_dim,),
                    weight=self.w(f"{prefix}.norm1.weight"),
                    bias=self.w(f"{prefix}.norm1.bias"),
                    eps=cfg.layer_norm_eps,
                )

                if target == f"{prefix}.attn.qkv.weight":
                    return h.reshape(-1, hidden_dim).contiguous()
                qkv_pre = F.linear(h, self.w(f"{prefix}.attn.qkv.weight"), None)
                if target == f"{prefix}.attn.qkv.bias":
                    return qkv_pre.reshape(-1, 3 * hidden_dim).contiguous()
                qkv = qkv_pre + self.w(f"{prefix}.attn.qkv.bias").view(1, 1, -1)
                tokens = int(qkv.shape[1])
                head_dim = hidden_dim // cfg.num_heads
                qkv = qkv.view(batch, tokens, 3, cfg.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(dim=0)
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=cfg.attention_dropout if self.training else 0.0,
                    is_causal=False,
                )
                attn = attn.transpose(1, 2).contiguous().view(batch, tokens, hidden_dim)

                if target == f"{prefix}.attn.proj.weight":
                    return attn.reshape(-1, hidden_dim).contiguous()
                proj_pre = F.linear(attn, self.w(f"{prefix}.attn.proj.weight"), None)
                if target == f"{prefix}.attn.proj.bias":
                    return proj_pre.reshape(-1, hidden_dim).contiguous()
                attn_out = proj_pre + self.w(f"{prefix}.attn.proj.bias").view(1, 1, -1)
                attn_out = F.dropout(attn_out, p=cfg.dropout, training=self.training)
                x = x + attn_out

                if target == f"{prefix}.norm2.weight" or target == f"{prefix}.norm2.bias":
                    return x.reshape(-1, hidden_dim).contiguous()
                h = F.layer_norm(
                    x,
                    (cfg.hidden_dim,),
                    weight=self.w(f"{prefix}.norm2.weight"),
                    bias=self.w(f"{prefix}.norm2.bias"),
                    eps=cfg.layer_norm_eps,
                )

                if target == f"{prefix}.mlp.fc1.weight":
                    return h.reshape(-1, hidden_dim).contiguous()
                fc1_pre = F.linear(h, self.w(f"{prefix}.mlp.fc1.weight"), None)
                if target == f"{prefix}.mlp.fc1.bias":
                    return fc1_pre.reshape(-1, mlp_dim).contiguous()
                h = fc1_pre + self.w(f"{prefix}.mlp.fc1.bias").view(1, 1, -1)
                h = F.gelu(h)
                h = F.dropout(h, p=cfg.dropout, training=self.training)

                if target == f"{prefix}.mlp.fc2.weight":
                    return h.reshape(-1, mlp_dim).contiguous()
                fc2_pre = F.linear(h, self.w(f"{prefix}.mlp.fc2.weight"), None)
                if target == f"{prefix}.mlp.fc2.bias":
                    return fc2_pre.reshape(-1, hidden_dim).contiguous()
                h = fc2_pre + self.w(f"{prefix}.mlp.fc2.bias").view(1, 1, -1)
                h = F.dropout(h, p=cfg.dropout, training=self.training)
                x = x + h

            if target == "norm.weight" or target == "norm.bias":
                return x.reshape(-1, hidden_dim).contiguous()
            x = F.layer_norm(
                x,
                (cfg.hidden_dim,),
                weight=self.w("norm.weight"),
                bias=self.w("norm.bias"),
                eps=cfg.layer_norm_eps,
            )
            cls_out = x[:, 0]
            if target == "head.weight":
                return cls_out.contiguous()
            logits_pre = F.linear(cls_out, self.w("head.weight"), None)
            if target == "head.bias":
                return logits_pre.contiguous()
        finally:
            self._decoded_tensor_cache = None
            if was_training:
                self.train(True)

        raise KeyError(f"Unsupported tensor name for activation-conditioned diffusion prior init: {tensor_name}")


