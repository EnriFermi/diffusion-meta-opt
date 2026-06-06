from __future__ import annotations

from .runtime import *

def train_setup(
    setup: str,
    cfg: ExperimentConfig,
    vit_cfg: ViTTinyConfig,
    initial_tensors: dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    output_dir: Path,
    big_vae_decoder: BigWeightVAE | None = None,
    big_vae_diffusion_prior: Any | None = None,
) -> dict[str, Any]:
    setup = normalize_setup_name(setup)
    if setup not in {"direct", "lowrank_latent", "bigvae_latent"}:
        raise ValueError(f"unsupported setup: {setup}")
    if setup == "bigvae_latent" and big_vae_decoder is None:
        raise ValueError("--big-vae-checkpoint is required for setup=bigvae_latent")

    seed_everything(int(cfg.seed))
    model = FunctionalViTTiny(
        vit_cfg,
        initial_tensors,
        parameter_mode=setup,
        latent_rank=int(cfg.latent_rank),
        latent_delta_scale=float(cfg.latent_delta_scale),
        latent_factor_init_std=float(cfg.latent_factor_init_std),
        big_vae=big_vae_decoder,
        big_vae_latent_init=str(cfg.big_vae_latent_init),
        big_vae_diffusion_prior=big_vae_diffusion_prior,
        big_vae_diffusion_prior_steps=int(cfg.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(cfg.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(cfg.big_vae_diffusion_prior_eta),
        big_vae_random_init_std=float(cfg.big_vae_random_init_std),
        big_vae_latent_noise_std=float(cfg.big_vae_latent_noise_std),
        big_vae_latent_parameterization=str(cfg.big_vae_latent_parameterization),
        big_vae_encoder_context_rows=int(cfg.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(cfg.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(cfg.big_vae_encoder_batch_size),
        big_vae_decode=str(cfg.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.big_vae_tile_d_out),
    ).to(device)
    if bool(cfg.compile):
        model = torch.compile(model)

    if setup == "bigvae_latent" and int(cfg.big_vae_init_fit_steps) > 0:
        fit_big_vae_latents_to_initial_weights(
            model,
            initial_tensors,
            steps=int(cfg.big_vae_init_fit_steps),
            lr=float(cfg.big_vae_init_fit_lr),
            log_every=max(1, int(cfg.big_vae_init_fit_log_every)),
            setup=setup,
        )

    lr = float(cfg.direct_lr if setup == "direct" else cfg.latent_lr)
    weight_decay = float(cfg.direct_weight_decay if setup == "direct" else cfg.latent_weight_decay)
    optimizer_name = str(cfg.direct_optimizer if setup == "direct" else cfg.latent_optimizer).strip().lower()
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = build_optimizer(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        cfg=cfg,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and bool(cfg.amp)))
    metrics_path = output_dir / "metrics.csv"
    checkpoint_dir = output_dir / "checkpoints"

    trainable_params = count_trainable_parameters(model)
    decoded_params = int(getattr(model, "store", getattr(model, "_orig_mod", model).store).decoded_numel())
    store = getattr(model, "store", getattr(model, "_orig_mod", model).store)
    decode_groups = store.decode_group_count() if isinstance(store, BigVAELatentTensorStore) else 0
    big_vae_decoded_params = store.big_vae_decoded_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    big_vae_tile_count = store.decoded_tile_count() if isinstance(store, BigVAELatentTensorStore) else 0
    big_vae_latent_params = store.latent_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    print(
        f"[{setup}] trainable_params={trainable_params} decoded_params={decoded_params} "
        f"bigvae_decoded_params={big_vae_decoded_params} decode_groups={decode_groups} "
        f"optimizer={optimizer_name} lr={lr:g} weight_decay={weight_decay:g}",
        flush=True,
    )
    if isinstance(store, BigVAELatentTensorStore):
        diversity = store.latent_init_diversity()
        tile_d_in, tile_d_out, tile_T = store.tile_decode_shape()
        compression = (
            float(big_vae_decoded_params) / float(big_vae_latent_params)
            if int(big_vae_latent_params) > 0
            else 0.0
        )
        print(
            f"[{setup}] latent_init={store.latent_init_mode} latent_space={store.latent_space} "
            f"latent_tiles={int(diversity['count'])} "
            f"tile_decode_shape=({tile_d_in},{tile_d_out},T={tile_T}) "
            f"latent_params={big_vae_latent_params} decoded_per_latent={compression:.3g} "
            f"across_layer_std_mean={diversity['across_layer_std_mean']:.6g} "
            f"max_pair_delta={diversity['max_pair_delta']:.6g}",
            flush=True,
        )

    global_step = 0
    train_loss_window = 0.0
    train_count_window = 0
    start_time = time.time()
    best_accuracy = 0.0
    best_step = 0
    final_eval = EvalMetrics(loss=float("nan"), accuracy=0.0, examples=0)

    for epoch_idx in range(int(cfg.epochs)):
        model.train()
        for images, labels in train_loader:
            global_step += 1
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, bool(cfg.amp)):
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=float(cfg.label_smoothing),
                )
            scaler.scale(loss).backward()
            if float(cfg.grad_clip_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            batch_examples = int(labels.numel())
            train_loss_window += float(loss.detach().cpu().item()) * batch_examples
            train_count_window += batch_examples

            should_log = global_step == 1 or global_step % max(1, int(cfg.log_every_steps)) == 0
            should_eval = global_step == 1 or global_step % max(1, int(cfg.eval_every_steps)) == 0
            max_steps = int(cfg.max_steps)
            if max_steps > 0 and global_step >= max_steps:
                should_eval = True

            if should_eval:
                final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.amp))
                if final_eval.accuracy > best_accuracy:
                    best_accuracy = float(final_eval.accuracy)
                    best_step = int(global_step)
            if should_log or should_eval:
                avg_train_loss = train_loss_window / max(1, train_count_window)
                elapsed_s = time.time() - start_time
                row = {
                    "setup": setup,
                    "step": int(global_step),
                    "epoch": int(epoch_idx + 1),
                    "train_loss": float(avg_train_loss),
                    "test_loss": float(final_eval.loss),
                    "test_accuracy": float(final_eval.accuracy),
                    "best_accuracy": float(best_accuracy),
                    "lr": float(lr),
                    "optimizer": optimizer_name,
                    "trainable_params": int(trainable_params),
                    "decoded_params": int(decoded_params),
                    "elapsed_s": float(elapsed_s),
                }
                append_csv_row(metrics_path, row)
                print(
                    f"[{setup}] step={global_step} epoch={epoch_idx + 1} "
                    f"train_loss={avg_train_loss:.4f} test_loss={final_eval.loss:.4f} "
                    f"test_acc={final_eval.accuracy:.4f} best={best_accuracy:.4f}",
                    flush=True,
                )
                train_loss_window = 0.0
                train_count_window = 0

            if max_steps > 0 and global_step >= max_steps:
                break
        if int(cfg.max_steps) > 0 and global_step >= int(cfg.max_steps):
            break

    final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.amp))
    if final_eval.accuracy > best_accuracy:
        best_accuracy = float(final_eval.accuracy)
        best_step = int(global_step)

    if bool(cfg.save_checkpoints):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = getattr(model, "_orig_mod", model)
        if setup == "bigvae_latent":
            model_state = {
                "latent_slots": target.store.materialized_latent_slots_state_dict(),
                "big_vae_checkpoint": str(cfg.big_vae_checkpoint),
                "big_vae_latent_parameterization": str(target.store.latent_parameterization),
            }
        else:
            model_state = target.state_dict()
        torch.save(
            {
                "setup": setup,
                "step": int(global_step),
                "model_state": model_state,
                "vit_config": asdict(vit_cfg),
                "experiment_config": asdict(cfg),
            },
            checkpoint_dir / f"{setup}_final.pt",
        )

    summary = {
        "setup": setup,
        "steps": int(global_step),
        "final_test_loss": float(final_eval.loss),
        "final_test_accuracy": float(final_eval.accuracy),
        "best_test_accuracy": float(best_accuracy),
        "best_step": int(best_step),
        "trainable_params": int(trainable_params),
        "decoded_params": int(decoded_params),
        "lr": float(lr),
        "optimizer": optimizer_name,
        "weight_decay": float(weight_decay),
        "big_vae_checkpoint": str(cfg.big_vae_checkpoint) if setup == "bigvae_latent" else "",
        "big_vae_init_fit_steps": int(cfg.big_vae_init_fit_steps) if setup == "bigvae_latent" else 0,
        "big_vae_latent_init": str(cfg.big_vae_latent_init) if setup == "bigvae_latent" else "",
        "big_vae_diffusion_prior_checkpoint": (
            str(cfg.big_vae_diffusion_prior_checkpoint) if setup == "bigvae_latent" else ""
        ),
        "big_vae_diffusion_prior_steps": int(cfg.big_vae_diffusion_prior_steps) if setup == "bigvae_latent" else 0,
        "big_vae_diffusion_prior_sampler": (
            str(cfg.big_vae_diffusion_prior_sampler) if setup == "bigvae_latent" else ""
        ),
        "big_vae_diffusion_prior_eta": float(cfg.big_vae_diffusion_prior_eta) if setup == "bigvae_latent" else 0.0,
        "big_vae_encoder_context_rows": int(cfg.big_vae_encoder_context_rows) if setup == "bigvae_latent" else 0,
        "big_vae_encoder_context_std": float(cfg.big_vae_encoder_context_std) if setup == "bigvae_latent" else 0.0,
        "decode_groups": int(decode_groups),
        "big_vae_decoded_params": int(big_vae_decoded_params),
        "big_vae_tile_count": int(big_vae_tile_count),
        "big_vae_latent_params": int(big_vae_latent_params),
    }
    write_json(output_dir / f"{setup}_summary.json", summary)
    return summary


def maybe_write_plot(output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows: list[dict[str, str]] = []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        return

    by_setup: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_setup.setdefault(row["setup"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for setup, setup_rows in sorted(by_setup.items()):
        steps = [int(row["step"]) for row in setup_rows]
        losses = [float(row["test_loss"]) for row in setup_rows]
        accs = [float(row["test_accuracy"]) for row in setup_rows]
        axes[0].plot(steps, losses, label=setup)
        axes[1].plot(steps, accs, label=setup)
    axes[0].set_title("CIFAR-10 test loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[1].set_title("CIFAR-10 test accuracy")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=160)
    plt.close(fig)

