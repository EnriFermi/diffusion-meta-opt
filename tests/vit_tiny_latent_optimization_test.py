from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.compare_vit_tiny_latent_optimization import (
    ExperimentConfig,
    FunctionalViTTiny,
    ViTTinyConfig,
    build_optimizer,
    count_trainable_parameters,
    make_initial_tensors,
)
from models.layer_latent_diffusion_prior import LayerLatentDiffusionPrior, LayerLatentDiffusionPriorConfig
from models.weight_quantile_vae import BigVAEConfig, BigWeightVAE, DistributionConfig, EncoderConfig, MiniVAEConfig, ModelConfig
from training.big_vae_latent_diffusion import latent_diffusion_layer_metadata_cond_dim


def _small_vit_cfg() -> ViTTinyConfig:
    return ViTTinyConfig(
        image_size=32,
        patch_size=8,
        hidden_dim=24,
        depth=2,
        num_heads=3,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
    )


def test_latent_vit_initial_logits_match_direct_vit() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=123)
    direct = FunctionalViTTiny(cfg, initial, parameter_mode="direct")
    latent = FunctionalViTTiny(cfg, initial, parameter_mode="lowrank_latent", latent_rank=4)

    x = torch.randn(3, 3, 32, 32)
    direct_logits = direct(x)
    latent_logits = latent(x)

    assert direct_logits.shape == (3, 10)
    assert torch.allclose(direct_logits, latent_logits, atol=1e-6, rtol=1e-6)


def test_latent_vit_has_fewer_trainable_parameters_and_receives_gradients() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=456)
    direct = FunctionalViTTiny(cfg, initial, parameter_mode="direct")
    latent = FunctionalViTTiny(cfg, initial, parameter_mode="lowrank_latent", latent_rank=2)

    assert count_trainable_parameters(latent) < count_trainable_parameters(direct)

    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = F.cross_entropy(latent(x), y)
    loss.backward()

    grad_sum = 0.0
    for param in latent.parameters():
        if param.grad is not None:
            grad_sum += float(param.grad.detach().abs().sum().item())
    assert grad_sum > 0.0


def test_bigvae_latent_vit_decodes_weights_and_receives_latent_gradients() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=789)
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=False,
                disable_distribution_encoder=True,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )
    model = FunctionalViTTiny(cfg, initial, parameter_mode="bigvae_latent", big_vae=big_vae)

    assert model.store.decode_group_count() == 1
    assert model.store.tile_decode_shape() == (128, 8, 16)
    assert model.store.decoded_tile_count() > len(model.store.decoded_tensor_names())
    qkv_matrix = model.store.target_matrix("blocks.0.attn.qkv.weight", initial["blocks.0.attn.qkv.weight"])
    assert qkv_matrix.shape == (cfg.hidden_dim, 3 * cfg.hidden_dim)

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, 10)

    loss = F.cross_entropy(logits, y)
    loss.backward()
    grad_sum = 0.0
    frozen_decoder_grad_sum = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "latent_slots" in name:
            grad_sum += float(param.grad.detach().abs().sum().item())
        if "big_vae" in name:
            frozen_decoder_grad_sum += float(param.grad.detach().abs().sum().item())
    assert grad_sum > 0.0
    assert frozen_decoder_grad_sum == 0.0


def test_bigvae_latent_default_keeps_non_weight_tensors_direct() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=101)
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=False,
                disable_distribution_encoder=True,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )
    model = FunctionalViTTiny(cfg, initial, parameter_mode="bigvae_latent", big_vae=big_vae)

    assert "pos_embed" in model.store._direct_name_to_key
    assert "cls_token" in model.store._direct_name_to_key
    assert "patch_embed.weight" in model.store._direct_name_to_key
    assert "blocks.0.attn.qkv.weight" in model.store._name_to_key
    assert model.store.decode_group_count() < len(initial)
    assert model.store.decoded_tile_count() > len(model.store.decoded_tensor_names())

    diversity = model.store.latent_init_diversity()
    assert diversity["across_layer_std_mean"] > 0.0
    assert diversity["max_pair_delta"] > 0.0


def test_bigvae_latent_encoded_init_can_optimize_only_latent_slots() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=202)
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=False,
                disable_distribution_encoder=True,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )
    model = FunctionalViTTiny(
        cfg,
        initial,
        parameter_mode="bigvae_latent",
        big_vae=big_vae,
        big_vae_latent_init="encoded",
        big_vae_decode="all",
        big_vae_encoder_context_rows=4,
        big_vae_encoder_batch_size=2,
    )

    assert len(model.store.direct_tensors) == 0
    assert model.store.latent_numel() == sum(param.numel() for param in model.parameters() if param.requires_grad)
    first_latent = next(iter(model.store.latent_slots.values())).detach()
    assert not torch.allclose(first_latent, big_vae.latent_base.detach(), atol=1e-6, rtol=1e-6)

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)
    loss = F.cross_entropy(model(x), y)
    loss.backward()

    latent_grad_sum = 0.0
    frozen_decoder_grad_sum = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "latent_slots" in name:
            latent_grad_sum += float(param.grad.detach().abs().sum().item())
        if "big_vae" in name:
            frozen_decoder_grad_sum += float(param.grad.detach().abs().sum().item())
    assert latent_grad_sum > 0.0
    assert frozen_decoder_grad_sum == 0.0


def test_bigvae_encoder_mu_head_can_run_without_latent_sampling() -> None:
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=False,
                use_encoder_mu_head=True,
                disable_distribution_encoder=True,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )

    assert big_vae.to_mu is not None
    assert big_vae.to_logvar is None

    with torch.no_grad():
        big_vae.to_mu.weight.mul_(2.0)

    base_z = torch.randn(3, big_vae.z_dim)
    decoder_z, mu, logvar = big_vae._sample_latent_posterior(base_z)

    assert torch.allclose(decoder_z, mu)
    assert torch.allclose(logvar, torch.zeros_like(base_z))
    assert not torch.allclose(decoder_z, base_z)


def test_bigvae_vamp_prior_kl_is_finite() -> None:
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=True,
                latent_prior_kind="vamp",
                vamp_prior_K=5,
                disable_distribution_encoder=True,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )

    assert big_vae.vamp_prior_base is not None
    assert tuple(big_vae.vamp_prior_base.shape) == (5, 2, 12)

    mu = torch.randn(4, big_vae.z_dim)
    logvar = torch.randn(4, big_vae.z_dim).clamp(-2.0, 2.0)
    kl = big_vae.latent_kl_loss(mu, logvar)

    assert kl.ndim == 0
    assert torch.isfinite(kl)


def test_bigvae_decoder_query_conditioning_mlp_runs() -> None:
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=False,
                disable_distribution_encoder=False,
                decoder_query_conditioning_kind="mlp",
                decoder_query_conditioning_hidden_mult=1.5,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )

    assert isinstance(big_vae.query_proj, torch.nn.Sequential)

    W = torch.randn(16, 5)
    X = torch.randn(7, 16)
    W_hat, mu, logvar, pred_dirs = big_vae(W, X)

    assert W_hat.shape == W.shape
    assert mu.shape == (big_vae.z_dim,)
    assert logvar.shape == (big_vae.z_dim,)
    assert pred_dirs.ndim == 3


def test_bigvae_rope_2d_coord_kind_switches_between_normalized_and_raw() -> None:
    def _build_model(coord_kind: str) -> BigWeightVAE:
        return BigWeightVAE(
            ModelConfig(
                patch_size=8,
                distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
                mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
                big_vae=BigVAEConfig(
                    d_model=24,
                    d_lat=12,
                    num_latents=4,
                    num_encoder_layers=1,
                    num_decoder_layers=1,
                    n_heads=3,
                    ffn_mult=2.0,
                    pos_fourier_dim=12,
                    use_latent_sampling=False,
                    disable_distribution_encoder=True,
                    rope_2d_coord_kind=coord_kind,
                    disable_z_shortcut=True,
                    encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
                ),
            )
        )

    normalized = _build_model("normalized_center")
    raw = _build_model("raw")

    _, _, q_pos_o_norm, q_pos_t_norm = normalized._build_decoder_query_state(
        batch_size=2,
        dist_patch_by_patch=None,
        d_out=3,
        T=2,
    )
    _, _, q_pos_o_raw, q_pos_t_raw = raw._build_decoder_query_state(
        batch_size=2,
        dist_patch_by_patch=None,
        d_out=3,
        T=2,
    )

    expected_o_norm = torch.tensor([1.0 / 6.0, 1.0 / 6.0, 0.5, 0.5, 5.0 / 6.0, 5.0 / 6.0], dtype=torch.float32)
    expected_t_norm = torch.tensor([0.25, 0.75, 0.25, 0.75, 0.25, 0.75], dtype=torch.float32)
    expected_o_raw = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float32)
    expected_t_raw = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32)

    assert torch.allclose(q_pos_o_norm, expected_o_norm)
    assert torch.allclose(q_pos_t_norm, expected_t_norm)
    assert torch.allclose(q_pos_o_raw, expected_o_raw)
    assert torch.allclose(q_pos_t_raw, expected_t_raw)

    lat = torch.zeros(2, 4, int(normalized.cfg.big_vae.d_model))
    kv_norm, kv_pos1_norm, kv_pos2_norm, _ = normalized._build_decoder_kv_state(
        lat=lat,
        encoder_patch_tokens=None,
        patch_mask=torch.ones(2, 2, dtype=torch.bool),
        d_out_mask=torch.ones(2, 3, dtype=torch.bool),
        d_out=3,
        T=2,
        debug_decoder_kv_source="latents",
    )
    kv_raw, kv_pos1_raw, kv_pos2_raw, _ = raw._build_decoder_kv_state(
        lat=lat,
        encoder_patch_tokens=None,
        patch_mask=torch.ones(2, 2, dtype=torch.bool),
        d_out_mask=torch.ones(2, 3, dtype=torch.bool),
        d_out=3,
        T=2,
        debug_decoder_kv_source="latents",
    )

    assert tuple(kv_norm.shape) == tuple(kv_raw.shape)
    assert torch.allclose(kv_pos1_norm, torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=torch.float32))
    assert torch.allclose(kv_pos2_norm, kv_pos1_norm)
    assert torch.allclose(kv_pos1_raw, torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32))
    assert torch.allclose(kv_pos2_raw, kv_pos1_raw)


def test_bigvae_latent_decoder_z_space_can_run_without_diffusion_prior_model() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=303)
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=True,
                disable_distribution_encoder=False,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )
    model = FunctionalViTTiny(
        cfg,
        initial,
        parameter_mode="bigvae_latent",
        big_vae=big_vae,
        big_vae_latent_init="random",
        big_vae_latent_space="decoder_z",
        big_vae_decode="all",
    )

    assert model.store.latent_space == "decoder_z"
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)
    loss = F.cross_entropy(model(x), y)
    loss.backward()

    latent_grad_sum = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None and "latent_slots" in name:
            latent_grad_sum += float(param.grad.detach().abs().sum().item())
    assert latent_grad_sum > 0.0


def test_bigvae_latent_sphere_parameterization_preserves_materialized_norms_after_step() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=404)
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=True,
                disable_distribution_encoder=False,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )
    model = FunctionalViTTiny(
        cfg,
        initial,
        parameter_mode="bigvae_latent",
        big_vae=big_vae,
        big_vae_latent_init="random",
        big_vae_latent_space="decoder_z",
        big_vae_latent_parameterization="sphere",
        big_vae_decode="all",
    )

    initial_norms = {
        str(key): float(model.store.materialize_latent_slot(str(key)).detach().norm().item())
        for key in model.store.latent_slots.keys()
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

    final_norms = {
        str(key): float(model.store.materialize_latent_slot(str(key)).detach().norm().item())
        for key in model.store.latent_slots.keys()
    }
    for key, initial_norm in initial_norms.items():
        assert abs(final_norms[key] - initial_norm) < 1e-5


def test_build_optimizer_supports_adamw_and_sgd() -> None:
    cfg = ExperimentConfig()
    param = torch.nn.Parameter(torch.ones(2))

    adamw = build_optimizer([param], optimizer_name="adamw", lr=1e-3, weight_decay=0.01, cfg=cfg)
    assert isinstance(adamw, torch.optim.AdamW)

    sgd = build_optimizer([param], optimizer_name="sgd", lr=1e-2, weight_decay=0.0, cfg=cfg)
    assert isinstance(sgd, torch.optim.SGD)


def test_bigvae_latent_diffusion_prior_init_decodes_weights_and_keeps_prior_frozen() -> None:
    cfg = _small_vit_cfg()
    initial = make_initial_tensors(cfg, seed=303)
    big_vae = BigWeightVAE(
        ModelConfig(
            patch_size=8,
            distribution=DistributionConfig(k_s=4, Kq=4, d_var=16, d_dist=16, use_covariance=False),
            mini_vae=MiniVAEConfig(z_dim=8, d_e=16, num_attn_layers_encoder=1, num_layers_decoder=1, n_heads=2, d_patch=8),
            big_vae=BigVAEConfig(
                d_model=24,
                d_lat=12,
                num_latents=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                n_heads=3,
                ffn_mult=2.0,
                pos_fourier_dim=12,
                use_latent_sampling=False,
                disable_distribution_encoder=False,
                disable_z_shortcut=True,
                encoder=EncoderConfig(self_attn_mode="cls_only", cross_attend_only_cls=True),
            ),
        )
    )
    prior = LayerLatentDiffusionPrior(
        LayerLatentDiffusionPriorConfig(
            z_dim=24,
            num_latent_tokens=2,
            cond_dim=16,
            cond_global_dim=16
            + latent_diffusion_layer_metadata_cond_dim(
                use_layer_type_conditioning=True,
                use_layer_depth_conditioning=True,
                depth_fourier_dim=8,
            ),
            use_layer_type_conditioning=True,
            use_layer_depth_conditioning=True,
            layer_depth_fourier_dim=8,
            layer_depth_scale=32.0,
            d_model=16,
            n_layers=2,
            n_heads=2,
            ffn_mult=2.0,
            dropout=0.0,
            cond_pool_tokens=2,
            pos_fourier_dim=8,
            time_embed_dim=16,
            use_cross_conditioning=False,
            prediction_type="v",
        )
    )
    model = FunctionalViTTiny(
        cfg,
        initial,
        parameter_mode="bigvae_latent",
        big_vae=big_vae,
        big_vae_latent_init="diffusion_prior",
        big_vae_diffusion_prior=prior,
        big_vae_diffusion_prior_steps=4,
        big_vae_diffusion_prior_sampler="ddim",
        big_vae_diffusion_prior_eta=0.0,
        big_vae_decode="all",
        big_vae_encoder_context_rows=4,
        big_vae_encoder_batch_size=2,
    )

    assert model.store.latent_space == "decoder_z"
    assert len(model.store.direct_tensors) == 0
    assert all(not param.requires_grad for param in model.store.latent_diffusion_prior.parameters())

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1], dtype=torch.long)
    loss = F.cross_entropy(model(x), y)
    loss.backward()

    latent_grad_sum = 0.0
    frozen_decoder_grad_sum = 0.0
    frozen_prior_grad_sum = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "latent_slots" in name:
            latent_grad_sum += float(param.grad.detach().abs().sum().item())
        if "big_vae" in name:
            frozen_decoder_grad_sum += float(param.grad.detach().abs().sum().item())
        if "latent_diffusion_prior" in name:
            frozen_prior_grad_sum += float(param.grad.detach().abs().sum().item())
    assert latent_grad_sum > 0.0
    assert frozen_decoder_grad_sum == 0.0
    assert frozen_prior_grad_sum == 0.0
