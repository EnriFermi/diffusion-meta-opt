from __future__ import annotations

import json
from pathlib import Path

import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import fast_config
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    CeloMetaMLP,
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
    payload = load_torch_cache(tmp_path / "weight_pool.pt")
    assert payload["spec"]["model_kind"] == "celo_meta_mlp"


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
    assert "bias_mse" in row
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
    assert int(vae.resampler_latents) == 2
    assert int(vae.attention_heads) == 1

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


def test_decoder_geometry_jacobian_shape_and_finite_metrics() -> None:
    spec = tiny_cnn_spec()
    normalizer = WeightNormalizer.fit(torch.randn(4, spec.dim))
    vae = WeightVAE(weight_dim=spec.dim, latent_dim=4, hidden_dim=16)
    samples = torch.randn(2, 4)

    jac = decoder_jacobians(vae, normalizer, None, samples, create_graph=False)
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
    for name in (
        "config.json",
        "weight_pool.pt",
        "vae_checkpoint.pt",
        "flow_state.pt",
        "random_flow_state.pt",
        "vae_metrics.csv",
        "geometry.csv",
        "selected_lrs.csv",
        "downstream_results.csv",
        "downstream_curves.csv",
    ):
        assert (tables.output_dir / name).is_file()


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
    assert (tables.output_dir / "flow_state_baseline.pt").is_file()
    assert (tables.output_dir / "flow_state_regularized.pt").is_file()


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
