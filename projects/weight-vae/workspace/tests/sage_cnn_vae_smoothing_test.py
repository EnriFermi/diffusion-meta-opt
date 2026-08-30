from __future__ import annotations

import json
from pathlib import Path

import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import config_hash, fast_config
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    CeloMetaMLP,
    ExperimentStopRequested,
    TaskTensorSet,
    TinyBigWeightVAE,
    WeightNormalizer,
    WeightVAE,
    build_weight_vae,
    celo_meta_mlp_spec,
    load_torch_cache,
    vae_loss,
    weight_pool_cache_key,
    decoder_jacobians,
    decode_weights,
    generate_weight_pool,
    logits_from_flat,
    flat_to_state_dict,
    geometry_metrics_from_jacobians,
    state_dict_to_flat,
    tiny_cnn_logits_from_flat,
    tiny_cnn_spec,
    TinyCNN,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.downstream import (
    DownstreamContext,
    run_downstream_curve,
)
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.preconditioning import (
    PreconditioningState,
    preconditioning_regularizer,
)


def test_config_hash_ignores_runtime_display_fields() -> None:
    base = fast_config(
        cache_first=True,
        force_rerun=False,
        save_figures=True,
        show_progress=False,
        progress_backend="text",
        live_vae_loss_curve=False,
        live_vae_loss_every=25,
    )
    display_only = fast_config(
        cache_first=False,
        force_rerun=True,
        save_figures=False,
        show_progress=True,
        progress_backend="notebook",
        live_vae_loss_curve=True,
        live_vae_loss_every=5,
        comet_enabled=True,
        comet_project_name="alternate-project",
        comet_workspace="workspace",
        comet_experiment_name="display-only",
        comet_tags=("notebook", "debug"),
        comet_log_every=7,
        comet_mode="offline",
        comet_offline_directory="/tmp/comet-offline",
        comet_log_code=True,
    )

    nf_only = fast_config(
        flow_steps=999,
        flow_batch_size=3,
        flow_lr=9e-4,
        flow_eta=0.7,
        mixup_alpha_min=0.0,
        mixup_alpha_max=1.0,
        nf_lrs=(1e-4,),
    )

    assert config_hash(base) == config_hash(display_only)
    assert config_hash(base) == config_hash(nf_only)
    assert config_hash(base) != config_hash(fast_config(vae_lr=2.0 * float(base.vae_lr)))
    assert config_hash(base) == config_hash(fast_config(vae_training_seed=-1))
    assert config_hash(base) != config_hash(fast_config(vae_training_seed=1))


def test_tiny_cnn_flat_state_roundtrip_and_functional_forward() -> None:
    spec = tiny_cnn_spec()
    model = TinyCNN()
    flat = state_dict_to_flat(model.state_dict(), spec)
    restored = flat_to_state_dict(flat, spec)
    images = torch.randn(3, 1, 28, 28)

    assert int(flat.numel()) == spec.dim
    assert all(torch.equal(model.state_dict()[key], restored[key]) for key in spec.keys)
    logits = tiny_cnn_logits_from_flat(flat, images, spec)
    assert tuple(logits.shape) == (3, 10)
    assert torch.isfinite(logits).all()


def test_celo_meta_mlp_tau_scaled_functional_forward() -> None:
    cfg = fast_config(celo_image_size=8, celo_hidden_dim=32)
    spec = celo_meta_mlp_spec(cfg)
    model = CeloMetaMLP(image_shape=(1, 8, 8), hidden_dim=32, num_classes=10)
    flat = state_dict_to_flat(model.state_dict(), spec)
    images = torch.randn(5, 1, 8, 8)
    tau = 7.0

    logits_reference = logits_from_flat(flat, images, spec, tau=1.0)
    logits_reparameterized = logits_from_flat(flat / tau, images, spec, tau=tau)
    logits_scaled = logits_from_flat(flat, images, spec, tau=tau)

    assert spec.model_kind == "celo_meta_mlp"
    assert spec.image_shape == (1, 8, 8)
    assert int(flat.numel()) == spec.dim
    assert tuple(logits_reference.shape) == (5, 10)
    assert torch.allclose(logits_reference, logits_reparameterized, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(logits_reference, logits_scaled)


def _fake_celo_task_tensors(*, task_names: tuple[str, ...] = ("mnist", "fashion_mnist")) -> dict[str, TaskTensorSet]:
    generator = torch.Generator().manual_seed(123)
    task_tensors: dict[str, TaskTensorSet] = {}
    for offset, task_name in enumerate(task_names):
        train_images = torch.randn(12, 1, 8, 8, generator=generator) + 0.1 * offset
        train_labels = torch.randint(0, 10, (12,), generator=generator)
        test_images = torch.randn(6, 1, 8, 8, generator=generator) + 0.1 * offset
        test_labels = torch.randint(0, 10, (6,), generator=generator)
        task_tensors[task_name] = TaskTensorSet(
            task_name=task_name,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
    return task_tensors


def test_celo_meta_weight_pool_samples_task_lr_and_tau(tmp_path) -> None:
    cfg = fast_config(
        device="cpu",
        cache_first=False,
        force_rerun=True,
        show_progress=False,
        weight_distribution="celo_meta_mlp",
        celo_tasks=("mnist", "fashion_mnist"),
        celo_adam_lrs=(1e-3, 3e-3),
        celo_tau_min=0.5,
        celo_tau_max=2.0,
        weight_runs=4,
        weight_train_steps=1,
        weight_snapshot_every=1,
        cnn_batch_size=4,
    )

    weights, records, spec = generate_weight_pool(
        cfg,
        task_tensors=_fake_celo_task_tensors(),
        output_path=tmp_path / "weight_pool.pt",
    )

    assert spec.model_kind == "celo_meta_mlp"
    assert tuple(weights.shape) == (8, spec.dim)
    assert set(records["task_name"]).issubset({"mnist", "fashion_mnist"})
    assert set(records["source_lr"]).issubset({1e-3, 3e-3})
    assert records["tau"].between(0.5, 2.0).all()
    assert set(records["optimizer"]) == {"adam"}
    final_records = records[records["step"] == int(cfg.weight_train_steps)]
    assert set(zip(final_records["task_name"], final_records["source_lr"], strict=False)) == {
        ("mnist", 1e-3),
        ("mnist", 3e-3),
        ("fashion_mnist", 1e-3),
        ("fashion_mnist", 3e-3),
    }
    payload = load_torch_cache(tmp_path / "weight_pool.pt")
    assert payload["spec"]["model_kind"] == "celo_meta_mlp"


def test_weight_pool_stop_file_cancels_without_kernel_restart(tmp_path) -> None:
    cfg = fast_config(
        device="cpu",
        cache_first=False,
        force_rerun=True,
        show_progress=False,
        weight_distribution="celo_meta_mlp",
        celo_tasks=("mnist",),
        weight_runs=1,
        weight_train_steps=1,
        weight_snapshot_every=1,
        cnn_batch_size=4,
    )
    output_path = tmp_path / "weight_pool.pt"
    (tmp_path / "STOP").write_text("stop", encoding="utf-8")

    try:
        generate_weight_pool(
            cfg,
            task_tensors=_fake_celo_task_tensors(task_names=("mnist",)),
            output_path=output_path,
        )
    except ExperimentStopRequested:
        pass
    else:
        raise AssertionError("STOP file should cancel weight-pool generation")

    assert not output_path.exists()


def test_celo_meta_weight_pool_runs_when_grad_mode_is_disabled(tmp_path) -> None:
    cfg = fast_config(
        device="cpu",
        cache_first=False,
        force_rerun=True,
        show_progress=False,
        weight_distribution="celo_meta_mlp",
        celo_tasks=("mnist", "fashion_mnist"),
        celo_adam_lrs=(1e-3,),
        celo_tau_min=0.5,
        celo_tau_max=2.0,
        weight_runs=1,
        weight_train_steps=1,
        weight_snapshot_every=1,
        cnn_batch_size=4,
    )

    previous_grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        weights, records, spec = generate_weight_pool(
            cfg,
            task_tensors=_fake_celo_task_tensors(),
            output_path=tmp_path / "weight_pool.pt",
        )
    finally:
        torch.set_grad_enabled(previous_grad_mode)

    assert spec.model_kind == "celo_meta_mlp"
    assert tuple(weights.shape) == (2, spec.dim)
    assert len(records) == 2


def test_weight_normalizer_roundtrip() -> None:
    values = torch.randn(5, 17)
    normalizer = WeightNormalizer.fit(values)
    restored = normalizer.denormalize(normalizer.normalize(values))

    assert torch.allclose(restored, values, atol=1e-6, rtol=1e-6)


def test_vae_decoder_weights_receive_latent_gradients() -> None:
    spec = tiny_cnn_spec()
    normalizer = WeightNormalizer.fit(torch.randn(4, spec.dim))
    vae = WeightVAE(weight_dim=spec.dim, latent_dim=4, hidden_dim=16)
    z = torch.randn(4, requires_grad=True)
    images = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([0, 1])

    decoded = decode_weights(vae, normalizer, z.reshape(1, -1)).squeeze(0)
    logits = tiny_cnn_logits_from_flat(decoded, images, spec)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_downstream_eval_every_controls_recorded_steps() -> None:
    spec = tiny_cnn_spec()
    cfg = fast_config(
        device="cpu",
        downstream_steps=5,
        downstream_eval_every=2,
        downstream_batch_size=4,
        finite_penalty=1e6,
    )
    generator = torch.Generator().manual_seed(42)
    task_tensors = {
        "tiny_cnn": TaskTensorSet(
            task_name="tiny_cnn",
            train_images=torch.randn(8, 1, 28, 28, generator=generator),
            train_labels=torch.randint(0, 10, (8,), generator=generator),
            test_images=torch.randn(4, 1, 28, 28, generator=generator),
            test_labels=torch.randint(0, 10, (4,), generator=generator),
        )
    }
    weights = torch.randn(3, spec.dim, generator=generator) * 0.01
    normalizer = WeightNormalizer.fit(weights)
    vae = WeightVAE(weight_dim=spec.dim, latent_dim=2, hidden_dim=8)
    ctx = DownstreamContext(cfg=cfg, spec=spec, vae=vae, normalizer=normalizer, task_tensors=task_tensors)

    rows, metrics = run_downstream_curve(
        ctx=ctx,
        w0=weights[0],
        method="raw",
        lr=1e-4,
        steps=int(cfg.downstream_steps),
        start_index=0,
        start_metadata={"task_name": "tiny_cnn", "tau": 1.0, "source_weight_index": 0},
        split="eval",
    )

    assert [int(row["step"]) for row in rows] == [0, 2, 4, 5]
    assert int(metrics["steps_to_threshold"]) == -1 or int(metrics["steps_to_threshold"]) in {0, 2, 4, 5}


def test_bigvae_style_vae_loss_is_finite_and_differentiable() -> None:
    spec = tiny_cnn_spec()
    cfg = fast_config(device="cpu", latent_dim=4, vae_hidden_dim=16, vae_loss_kind="big_vae", bigvae_operator_probe_rows=2)
    weights = torch.randn(3, spec.dim)
    normalizer = WeightNormalizer.fit(weights)
    vae = WeightVAE(weight_dim=spec.dim, latent_dim=4, hidden_dim=16)
    x_norm = normalizer.normalize(weights)

    recon, mu, logvar = vae(x_norm)
    loss, row = vae_loss(cfg, x_norm, weights, recon, mu, logvar, normalizer=normalizer, spec=spec)
    loss.backward()

    assert torch.isfinite(loss)
    assert row["bigvae_weight_loss"] >= 0.0
    assert "behavioral_operator" in row
    assert "structural" in row
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in vae.parameters())


def test_tiny_big_weight_vae_api_and_loss_are_differentiable() -> None:
    spec = celo_meta_mlp_spec(fast_config())
    cfg = fast_config(
        device="cpu",
        latent_dim=3,
        vae_hidden_dim=8,
        vae_arch="tiny_big_vae",
        tiny_bigvae_patch_size=16,
        tiny_bigvae_token_dim=8,
        tiny_bigvae_pos_dim=4,
        tiny_bigvae_resampler_latents=2,
        tiny_bigvae_attention_heads=1,
        vae_loss_kind="big_vae",
        bigvae_operator_probe_rows=2,
    )
    weights = torch.randn(4, spec.dim)
    normalizer = WeightNormalizer.fit(weights)
    vae = build_weight_vae(cfg, weight_dim=spec.dim)
    assert isinstance(vae, TinyBigWeightVAE)
    assert vae.output_mode == "direction_scale"
    assert int(vae.resampler_latents) == 2
    assert int(vae.attention_heads) == 1
    assert int(vae.decoder_qkv.out_features) == 3 * int(vae.token_dim)
    assert int(vae.decoder_scale_head[-1].out_features) == 1
    assert tuple(vae.decoder_scale_head(torch.randn(2, vae.num_patches, vae.token_dim)).shape) == (
        2,
        vae.num_patches,
        1,
    )

    x_norm = normalizer.normalize(weights)
    recon, mu, logvar = vae(x_norm)
    decoded = decode_weights(vae, normalizer, mu)
    loss, row = vae_loss(cfg, x_norm, weights, recon, mu, logvar, normalizer=normalizer, spec=spec)
    loss.backward()

    assert tuple(recon.shape) == tuple(x_norm.shape)
    assert tuple(mu.shape) == (4, 3)
    assert tuple(decoded.shape) == tuple(weights.shape)
    assert torch.isfinite(loss)
    assert row["bigvae_weight_loss"] >= 0.0
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in vae.parameters())
    assert vae.decoder_qkv.weight.grad is not None
    assert torch.isfinite(vae.decoder_qkv.weight.grad).all()
    assert vae.decoder_scale_head[-1].weight.grad is not None
    assert torch.isfinite(vae.decoder_scale_head[-1].weight.grad).all()


def test_tiny_big_weight_vae_direct_output_mode_is_differentiable() -> None:
    spec = celo_meta_mlp_spec(fast_config())
    cfg = fast_config(
        device="cpu",
        latent_dim=5,
        vae_hidden_dim=16,
        vae_arch="tiny_big_vae",
        tiny_bigvae_patch_size=16,
        tiny_bigvae_token_dim=8,
        tiny_bigvae_pos_dim=4,
        tiny_bigvae_resampler_latents=3,
        tiny_bigvae_attention_heads=1,
        tiny_bigvae_output_mode="direct",
        vae_loss_kind="mse",
    )
    weights = torch.randn(4, spec.dim)
    normalizer = WeightNormalizer.fit(weights)
    vae = build_weight_vae(cfg, weight_dim=spec.dim)
    assert isinstance(vae, TinyBigWeightVAE)
    assert vae.output_mode == "direct"
    assert int(vae.decoder_context_count) == 3

    x_norm = normalizer.normalize(weights)
    recon, mu, logvar = vae(x_norm)
    loss, row = vae_loss(cfg, x_norm, weights, recon, mu, logvar, normalizer=normalizer, spec=spec)
    loss.backward()

    assert tuple(recon.shape) == tuple(x_norm.shape)
    assert tuple(mu.shape) == (4, 5)
    assert torch.isfinite(loss)
    assert row["recon_mse"] >= 0.0
    assert vae.decoder_cross_attn.in_proj_weight.grad is not None
    assert torch.isfinite(vae.decoder_cross_attn.in_proj_weight.grad).all()
    assert vae.patch_decoder[-1].weight.grad is not None
    assert torch.isfinite(vae.patch_decoder[-1].weight.grad).all()
    assert vae.decoder_scale_head[-1].weight.grad is None


def test_li_a_preconditioning_regularizer_is_differentiable() -> None:
    cfg = fast_config(
        device="cpu",
        celo_image_size=8,
        celo_hidden_dim=4,
        latent_dim=2,
        vae_hidden_dim=8,
        vae_precond_loss_kind="li_a_hvp",
        vae_precond_coeff=1.0,
        vae_precond_every=1,
        vae_precond_samples=1,
        vae_precond_pairs=1,
        vae_precond_batch_size=4,
        vae_precond_hvp_mode="stopped_composite",
    )
    spec = celo_meta_mlp_spec(cfg)
    weights = torch.randn(4, spec.dim) * 0.01
    normalizer = WeightNormalizer.fit(weights)
    vae = WeightVAE(weight_dim=spec.dim, latent_dim=2, hidden_dim=8)
    z_samples = torch.randn(1, 2)
    records = [{"task_name": "mnist", "tau": 1.0, "source_weight_index": 0}]

    loss, row = preconditioning_regularizer(
        cfg,
        state=PreconditioningState(),
        vae=vae,
        normalizer=normalizer,
        z_samples=z_samples,
        records=records,
        task_tensors=_fake_celo_task_tensors(task_names=("mnist",)),
        spec=spec,
        step=1,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert row["precond_a_loss"] == row["precond_a_loss"]
    assert row["precond_trace_m_per_dim"] >= 0.0
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in vae.parameters())


def test_li_a_preconditioning_estimator_scope_is_explicit() -> None:
    base_kwargs = dict(
        device="cpu",
        celo_image_size=8,
        celo_hidden_dim=4,
        latent_dim=2,
        vae_hidden_dim=8,
        vae_precond_loss_kind="li_a_hvp",
        vae_precond_coeff=1.0,
        vae_precond_every=1,
        vae_precond_samples=2,
        vae_precond_pairs=1,
        vae_precond_batch_size=4,
        vae_precond_hvp_mode="stopped_composite",
    )
    task_tensors = _fake_celo_task_tensors(task_names=("mnist",))
    records = [
        {"task_name": "mnist", "tau": 1.0, "source_weight_index": 0},
        {"task_name": "mnist", "tau": 1.0, "source_weight_index": 1},
    ]

    rows = {}
    for scope in ("local", "global"):
        cfg = fast_config(**base_kwargs, vae_precond_estimator_scope=scope)
        spec = celo_meta_mlp_spec(cfg)
        weights = torch.randn(4, spec.dim) * 0.01
        normalizer = WeightNormalizer.fit(weights)
        vae = WeightVAE(weight_dim=spec.dim, latent_dim=2, hidden_dim=8)
        z_samples = torch.randn(2, 2)
        loss, row = preconditioning_regularizer(
            cfg,
            state=PreconditioningState(),
            vae=vae,
            normalizer=normalizer,
            z_samples=z_samples,
            records=records,
            task_tensors=task_tensors,
            spec=spec,
            step=1,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        loss.backward()
        assert torch.isfinite(loss)
        rows[scope] = row

    assert rows["local"]["precond_estimator_global"] == 0.0
    assert rows["local"]["precond_estimator_scope_code"] == 0.0
    assert rows["global"]["precond_estimator_global"] == 1.0
    assert rows["global"]["precond_estimator_scope_code"] == 1.0
    assert "precond_a_global_audit_loss" in rows["local"]
    assert "precond_a_train_loss" in rows["global"]


def test_li_e_preconditioning_regularizer_is_differentiable() -> None:
    cfg = fast_config(
        device="cpu",
        celo_image_size=8,
        celo_hidden_dim=4,
        latent_dim=2,
        vae_hidden_dim=8,
        vae_precond_loss_kind="li_e_whitening",
        vae_precond_coeff=1.0,
        vae_precond_every=1,
        vae_precond_samples=1,
        vae_precond_pairs=1,
        vae_precond_batch_size=4,
        vae_precond_grad_damping=0.01,
    )
    spec = celo_meta_mlp_spec(cfg)
    weights = torch.randn(4, spec.dim) * 0.01
    normalizer = WeightNormalizer.fit(weights)
    vae = WeightVAE(weight_dim=spec.dim, latent_dim=2, hidden_dim=8)
    z_samples = torch.randn(1, 2)
    records = [{"task_name": "mnist", "tau": 1.0, "source_weight_index": 0}]

    loss, row = preconditioning_regularizer(
        cfg,
        state=PreconditioningState(),
        vae=vae,
        normalizer=normalizer,
        z_samples=z_samples,
        records=records,
        task_tensors=_fake_celo_task_tensors(task_names=("mnist",)),
        spec=spec,
        step=1,
        generator=torch.Generator(device="cpu").manual_seed(1),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert row["precond_a_loss"] == row["precond_a_loss"]
    assert row["precond_trace_m_per_dim"] >= 0.0
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in vae.parameters())


def test_decoder_geometry_jacobian_shape_and_finite_metrics() -> None:
    spec = tiny_cnn_spec()
    normalizer = WeightNormalizer.fit(torch.randn(4, spec.dim))
    vae = WeightVAE(weight_dim=spec.dim, latent_dim=4, hidden_dim=16)
    samples = torch.randn(2, 4)

    jac = decoder_jacobians(vae, normalizer, None, samples, create_graph=False, chunk_size=2)
    metrics = geometry_metrics_from_jacobians(jac)

    assert tuple(jac.shape) == (2, spec.dim, 4)
    assert metrics["isometry_objective"] >= 0.0
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_sage_cnn_vae_smoothing_smoke_writes_outputs(tmp_path, monkeypatch) -> None:
    import post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline as pipeline

    def fake_loader(_cfg):
        generator = torch.Generator().manual_seed(0)
        train_images = torch.randn(16, 1, 28, 28, generator=generator)
        train_labels = torch.randint(0, 10, (16,), generator=generator)
        test_images = torch.randn(8, 1, 28, 28, generator=generator)
        test_labels = torch.randint(0, 10, (8,), generator=generator)
        return train_images, train_labels, test_images, test_labels

    monkeypatch.setattr(pipeline, "load_vision_tensors", fake_loader)
    cfg = fast_config(
        run_label="smoke",
        artifact_root=str(tmp_path),
        device="cpu",
        cache_first=False,
        force_rerun=True,
        weight_runs=2,
        weight_train_steps=1,
        weight_snapshot_every=1,
        cnn_batch_size=8,
        latent_dim=4,
        vae_hidden_dim=16,
        vae_steps=2,
        vae_batch_size=2,
        flow_steps=2,
        flow_batch_size=1,
        flow_num_layers=2,
        flow_hidden_dim=8,
        flow_network_depth=1,
        flow_log_every=1,
        geometry_eval_samples=1,
        tune_starts=1,
        eval_starts=1,
        downstream_steps=1,
        raw_lrs=(1e-3,),
        latent_lrs=(1e-2,),
        nf_lrs=(1e-2,),
        save_figures=False,
    )

    tables = pipeline.run_or_load(cfg)

    assert not tables.vae_metrics.empty
    assert not tables.geometry.empty
    assert not tables.selected_lrs.empty
    assert not tables.downstream_results.empty
    assert not tables.downstream_curves.empty
    assert set(tables.downstream_results["method"].unique().tolist()) == {"raw", "decoder_latent"}
    assert set(tables.geometry["coordinate"].unique().tolist()) == {"decoder_latent"}
    for name in (
        "config.json",
        "weight_pool.pt",
        "vae_checkpoint.pt",
        "vae_metrics.csv",
        "geometry.csv",
        "preconditioning_diagnostics.csv",
        "downstream_start_bank.csv",
        "selected_lrs.csv",
        "downstream_results.csv",
        "downstream_curves.csv",
    ):
        assert (tables.output_dir / name).is_file()
    assert not (tables.output_dir / "flow_state.pt").exists()
    assert not (tables.output_dir / "random_flow_state.pt").exists()
    assert not (tables.output_dir / "flow_history.csv").exists()


def test_sage_cnn_vae_smoothing_celo_meta_smoke_writes_task_metadata(tmp_path, monkeypatch) -> None:
    import post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline as pipeline

    monkeypatch.setattr(pipeline, "load_celo_meta_task_tensors", lambda _cfg: _fake_celo_task_tensors())
    cfg = fast_config(
        run_label="celo_smoke",
        artifact_root=str(tmp_path),
        device="cpu",
        cache_first=False,
        force_rerun=True,
        show_progress=False,
        weight_distribution="celo_meta_mlp",
        celo_tasks=("mnist", "fashion_mnist"),
        celo_adam_lrs=(1e-3, 3e-3),
        celo_tau_min=0.5,
        celo_tau_max=2.0,
        weight_runs=2,
        weight_train_steps=1,
        weight_snapshot_every=1,
        cnn_batch_size=4,
        latent_dim=2,
        vae_hidden_dim=8,
        vae_arch="tiny_big_vae",
        tiny_bigvae_patch_size=16,
        tiny_bigvae_token_dim=8,
        tiny_bigvae_pos_dim=4,
        tiny_bigvae_resampler_latents=2,
        tiny_bigvae_attention_heads=1,
        tiny_bigvae_output_mode="direct",
        vae_steps=1,
        vae_batch_size=1,
        bigvae_operator_probe_rows=1,
        flow_steps=1,
        flow_batch_size=1,
        flow_num_layers=2,
        flow_hidden_dim=8,
        flow_network_depth=1,
        flow_log_every=1,
        geometry_eval_samples=1,
        tune_starts=1,
        eval_starts=1,
        downstream_steps=1,
        raw_lrs=(1e-3,),
        latent_lrs=(1e-2,),
        nf_lrs=(1e-2,),
        save_figures=False,
    )

    tables = pipeline.run_or_load(cfg)
    records = torch.load(tables.output_dir / "weight_pool.pt", map_location="cpu", weights_only=False)["records"]

    assert not tables.downstream_results.empty
    assert set(tables.downstream_results["method"].unique().tolist()) == {"raw", "decoder_latent"}
    assert "task_name" in tables.vae_metrics.columns
    assert "tau" in tables.downstream_curves.columns
    assert {row["optimizer"] for row in records} == {"adam"}
    assert {row["task_name"] for row in records}.issubset({"mnist", "fashion_mnist"})


def test_sage_cnn_vae_smoothing_reg_coeff_runs_baseline_and_regularized(tmp_path, monkeypatch) -> None:
    import post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.pipeline as pipeline

    def fake_loader(_cfg):
        generator = torch.Generator().manual_seed(1)
        train_images = torch.randn(12, 1, 28, 28, generator=generator)
        train_labels = torch.randint(0, 10, (12,), generator=generator)
        test_images = torch.randn(6, 1, 28, 28, generator=generator)
        test_labels = torch.randint(0, 10, (6,), generator=generator)
        return train_images, train_labels, test_images, test_labels

    monkeypatch.setattr(pipeline, "load_vision_tensors", fake_loader)
    cfg = fast_config(
        run_label="reg_smoke",
        artifact_root=str(tmp_path),
        device="cpu",
        cache_first=False,
        force_rerun=True,
        show_progress=False,
        weight_runs=1,
        weight_train_steps=1,
        weight_snapshot_every=1,
        cnn_batch_size=6,
        latent_dim=2,
        vae_hidden_dim=8,
        vae_steps=1,
        vae_batch_size=1,
        bigvae_operator_probe_rows=1,
        vae_geometry_reg_coeff=1e-4,
        vae_geometry_reg_samples=1,
        flow_steps=1,
        flow_batch_size=1,
        flow_num_layers=2,
        flow_hidden_dim=8,
        flow_network_depth=1,
        flow_log_every=1,
        geometry_eval_samples=1,
        tune_starts=1,
        eval_starts=1,
        downstream_steps=1,
        raw_lrs=(1e-3,),
        latent_lrs=(1e-2,),
        nf_lrs=(1e-2,),
        save_figures=False,
    )

    tables = pipeline.run_or_load(cfg)

    assert set(tables.vae_metrics["vae_variant"].dropna().unique().tolist()) == {"baseline", "regularized"}
    assert set(tables.geometry["vae_variant"].dropna().unique().tolist()) == {"baseline", "regularized"}
    assert set(tables.downstream_results["vae_variant"].dropna().unique().tolist()) == {"baseline", "regularized"}
    assert (tables.output_dir / "vae_checkpoint_baseline.pt").is_file()
    assert (tables.output_dir / "vae_checkpoint_regularized.pt").is_file()
    assert not (tables.output_dir / "flow_state_baseline.pt").exists()
    assert not (tables.output_dir / "flow_state_regularized.pt").exists()


def test_corrupt_torch_cache_is_renamed_and_ignored(tmp_path) -> None:
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a torch checkpoint")

    payload = load_torch_cache(path)

    assert payload is None
    assert not path.exists()
    assert list(tmp_path.glob("broken.pt.broken-*"))


def test_weight_pool_cache_key_tracks_weight_generation_config() -> None:
    base = fast_config(weight_runs=2, weight_train_steps=3, weight_snapshot_every=1)

    assert weight_pool_cache_key(base) != weight_pool_cache_key(fast_config(weight_runs=3, weight_train_steps=3, weight_snapshot_every=1))
    assert weight_pool_cache_key(base) != weight_pool_cache_key(fast_config(weight_runs=2, weight_train_steps=4, weight_snapshot_every=1))
    assert weight_pool_cache_key(base) != weight_pool_cache_key(fast_config(weight_runs=2, weight_train_steps=3, weight_snapshot_every=2))


def test_sage_cnn_vae_smoothing_notebook_code_cells_compile() -> None:
    path = Path("post_train_research/loss_landscape_analysis/sage_cnn_vae_posthoc_smoothing.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{path}:cell{idx}", "exec")
