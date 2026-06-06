from __future__ import annotations

from .training import *

def parse_args() -> tuple[ExperimentConfig, ViTTinyConfig]:
    default_exp = ExperimentConfig()
    default_vit = ViTTinyConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Compare CIFAR-10 ViT-Tiny optimization with direct weights vs "
            "latent factors decoded into weights."
        )
    )
    parser.add_argument("--output-dir", default=default_exp.output_dir)
    parser.add_argument("--data-dir", default=default_exp.data_dir)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=default_exp.download)
    parser.add_argument(
        "--setup",
        choices=("direct", "bigvae_latent", "lowrank_latent", "both"),
        default=default_exp.setup,
    )
    parser.add_argument("--device", default=default_exp.device)
    parser.add_argument("--seed", type=int, default=default_exp.seed)
    parser.add_argument("--epochs", type=int, default=default_exp.epochs)
    parser.add_argument("--max-steps", type=int, default=default_exp.max_steps)
    parser.add_argument("--batch-size", type=int, default=default_exp.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=default_exp.eval_batch_size)
    parser.add_argument("--num-workers", type=int, default=default_exp.num_workers)
    parser.add_argument("--train-subset", type=int, default=default_exp.train_subset)
    parser.add_argument("--test-subset", type=int, default=default_exp.test_subset)
    parser.add_argument("--direct-lr", type=float, default=default_exp.direct_lr)
    parser.add_argument("--latent-lr", type=float, default=default_exp.latent_lr)
    parser.add_argument("--direct-optimizer", choices=("adamw", "sgd"), default=default_exp.direct_optimizer)
    parser.add_argument("--latent-optimizer", choices=("adamw", "sgd"), default=default_exp.latent_optimizer)
    parser.add_argument("--direct-weight-decay", type=float, default=default_exp.direct_weight_decay)
    parser.add_argument("--latent-weight-decay", type=float, default=default_exp.latent_weight_decay)
    parser.add_argument("--sgd-momentum", type=float, default=default_exp.sgd_momentum)
    parser.add_argument("--sgd-nesterov", action=argparse.BooleanOptionalAction, default=default_exp.sgd_nesterov)
    parser.add_argument("--adam-beta1", type=float, default=default_exp.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=default_exp.adam_beta2)
    parser.add_argument("--adam-eps", type=float, default=default_exp.adam_eps)
    parser.add_argument("--label-smoothing", type=float, default=default_exp.label_smoothing)
    parser.add_argument("--grad-clip-norm", type=float, default=default_exp.grad_clip_norm)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=default_exp.amp)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=default_exp.tf32)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=default_exp.compile)
    parser.add_argument("--log-every-steps", type=int, default=default_exp.log_every_steps)
    parser.add_argument("--eval-every-steps", type=int, default=default_exp.eval_every_steps)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=default_exp.save_checkpoints)
    parser.add_argument("--latent-rank", type=int, default=default_exp.latent_rank)
    parser.add_argument("--latent-delta-scale", type=float, default=default_exp.latent_delta_scale)
    parser.add_argument("--latent-factor-init-std", type=float, default=default_exp.latent_factor_init_std)
    parser.add_argument("--big-vae-checkpoint", default=default_exp.big_vae_checkpoint)
    parser.add_argument(
        "--big-vae-latent-init",
        choices=("base", "random", "encoded", "diffusion_prior"),
        default=default_exp.big_vae_latent_init,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-checkpoint",
        default=default_exp.big_vae_diffusion_prior_checkpoint,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-steps",
        type=int,
        default=default_exp.big_vae_diffusion_prior_steps,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-sampler",
        choices=("ddim", "ddpm"),
        default=default_exp.big_vae_diffusion_prior_sampler,
    )
    parser.add_argument(
        "--big-vae-diffusion-prior-eta",
        type=float,
        default=default_exp.big_vae_diffusion_prior_eta,
    )
    parser.add_argument("--big-vae-random-init-std", type=float, default=default_exp.big_vae_random_init_std)
    parser.add_argument("--big-vae-latent-noise-std", type=float, default=default_exp.big_vae_latent_noise_std)
    parser.add_argument(
        "--big-vae-encoder-context-rows",
        type=int,
        default=default_exp.big_vae_encoder_context_rows,
    )
    parser.add_argument(
        "--big-vae-encoder-context-std",
        type=float,
        default=default_exp.big_vae_encoder_context_std,
    )
    parser.add_argument(
        "--big-vae-encoder-batch-size",
        type=int,
        default=default_exp.big_vae_encoder_batch_size,
    )
    parser.add_argument("--big-vae-decode", choices=("weights", "all"), default=default_exp.big_vae_decode)
    parser.add_argument(
        "--big-vae-tile-t-patches",
        "--big-vae-tile-T-patches",
        dest="big_vae_tile_T_patches",
        type=int,
        default=default_exp.big_vae_tile_T_patches,
    )
    parser.add_argument("--big-vae-tile-d-out", type=int, default=default_exp.big_vae_tile_d_out)
    parser.add_argument("--big-vae-init-fit-steps", type=int, default=default_exp.big_vae_init_fit_steps)
    parser.add_argument("--big-vae-init-fit-lr", type=float, default=default_exp.big_vae_init_fit_lr)
    parser.add_argument("--big-vae-init-fit-log-every", type=int, default=default_exp.big_vae_init_fit_log_every)

    parser.add_argument("--patch-size", type=int, default=default_vit.patch_size)
    parser.add_argument("--hidden-dim", type=int, default=default_vit.hidden_dim)
    parser.add_argument("--depth", type=int, default=default_vit.depth)
    parser.add_argument("--num-heads", type=int, default=default_vit.num_heads)
    parser.add_argument("--mlp-ratio", type=float, default=default_vit.mlp_ratio)
    parser.add_argument("--dropout", type=float, default=default_vit.dropout)
    parser.add_argument("--attention-dropout", type=float, default=default_vit.attention_dropout)
    args = parser.parse_args()

    exp_cfg = ExperimentConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        download=bool(args.download),
        setup=str(args.setup),
        device=str(args.device),
        seed=int(args.seed),
        epochs=int(args.epochs),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        train_subset=int(args.train_subset),
        test_subset=int(args.test_subset),
        direct_lr=float(args.direct_lr),
        latent_lr=float(args.latent_lr),
        direct_optimizer=str(args.direct_optimizer),
        latent_optimizer=str(args.latent_optimizer),
        direct_weight_decay=float(args.direct_weight_decay),
        latent_weight_decay=float(args.latent_weight_decay),
        sgd_momentum=float(args.sgd_momentum),
        sgd_nesterov=bool(args.sgd_nesterov),
        adam_beta1=float(args.adam_beta1),
        adam_beta2=float(args.adam_beta2),
        adam_eps=float(args.adam_eps),
        label_smoothing=float(args.label_smoothing),
        grad_clip_norm=float(args.grad_clip_norm),
        amp=bool(args.amp),
        tf32=bool(args.tf32),
        compile=bool(args.compile),
        log_every_steps=int(args.log_every_steps),
        eval_every_steps=int(args.eval_every_steps),
        save_checkpoints=bool(args.save_checkpoints),
        latent_rank=int(args.latent_rank),
        latent_delta_scale=float(args.latent_delta_scale),
        latent_factor_init_std=float(args.latent_factor_init_std),
        big_vae_checkpoint=str(args.big_vae_checkpoint),
        big_vae_latent_init=str(args.big_vae_latent_init),
        big_vae_diffusion_prior_checkpoint=str(args.big_vae_diffusion_prior_checkpoint),
        big_vae_diffusion_prior_steps=int(args.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(args.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(args.big_vae_diffusion_prior_eta),
        big_vae_random_init_std=float(args.big_vae_random_init_std),
        big_vae_latent_noise_std=float(args.big_vae_latent_noise_std),
        big_vae_encoder_context_rows=int(args.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(args.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(args.big_vae_encoder_batch_size),
        big_vae_decode=str(args.big_vae_decode),
        big_vae_tile_T_patches=int(args.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(args.big_vae_tile_d_out),
        big_vae_init_fit_steps=int(args.big_vae_init_fit_steps),
        big_vae_init_fit_lr=float(args.big_vae_init_fit_lr),
        big_vae_init_fit_log_every=int(args.big_vae_init_fit_log_every),
    )
    vit_cfg = ViTTinyConfig(
        patch_size=int(args.patch_size),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
    )
    return exp_cfg, vit_cfg


def main() -> None:
    cfg, vit_cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config_resolved.json"
    summary_path = output_dir / "summary.json"
    config_payload = {"experiment": asdict(cfg), "vit": asdict(vit_cfg)}
    write_json(config_path, config_payload)
    write_json(output_dir / "config.json", config_payload)
    write_artifact_layout(
        output_dir,
        kind="post_train.vit_tiny_latent_optimization",
        run_id=output_dir.name,
        files={
            "config_json": config_path,
            "metrics": output_dir / "metrics.csv",
            "summary": summary_path,
            "comparison_plot": output_dir / "comparison.png",
        },
        dirs={"checkpoints": output_dir / "checkpoints"},
        metadata={"setup": cfg.setup, "big_vae_checkpoint": cfg.big_vae_checkpoint},
    )

    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.benchmark = True

    seed_everything(int(cfg.seed))
    initial_tensors = make_initial_tensors(vit_cfg, seed=int(cfg.seed))
    train_loader, test_loader = build_cifar10_loaders(cfg, device)

    requested_setup = normalize_setup_name(cfg.setup)
    setups = ["direct", "bigvae_latent"] if requested_setup == "both" else [requested_setup]
    big_vae_decoder = None
    big_vae_diffusion_prior = None
    if "bigvae_latent" in setups:
        if not str(cfg.big_vae_checkpoint).strip():
            raise ValueError("--big-vae-checkpoint is required for setup=bigvae_latent/both")
        print(f"[bigvae_latent] loading frozen BigVAE decoder: {cfg.big_vae_checkpoint}", flush=True)
        big_vae_decoder = load_frozen_big_vae_decoder(cfg.big_vae_checkpoint, device=device)
        if str(cfg.big_vae_latent_init).strip().lower() == "diffusion_prior":
            if not str(cfg.big_vae_diffusion_prior_checkpoint).strip():
                raise ValueError(
                    "--big-vae-diffusion-prior-checkpoint is required when --big-vae-latent-init=diffusion_prior"
                )
            print(
                "[bigvae_latent] loading frozen latent diffusion prior: "
                f"{cfg.big_vae_diffusion_prior_checkpoint}",
                flush=True,
            )
            big_vae_diffusion_prior = load_frozen_layer_latent_diffusion_prior(
                cfg.big_vae_diffusion_prior_checkpoint,
                device=device,
            )
            if big_vae_decoder is not None:
                loaded_dist_encoder = load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint(
                    cfg.big_vae_diffusion_prior_checkpoint,
                    big_vae=big_vae_decoder,
                )
                if loaded_dist_encoder:
                    print(
                        "[bigvae_latent] loaded finetuned distribution encoder state from latent diffusion prior checkpoint",
                        flush=True,
                    )

    summaries = []
    for setup in setups:
        summaries.append(
            train_setup(
                setup,
                cfg,
                vit_cfg,
                initial_tensors,
                train_loader,
                test_loader,
                device=device,
                output_dir=output_dir,
                big_vae_decoder=big_vae_decoder,
                big_vae_diffusion_prior=big_vae_diffusion_prior,
            )
        )
    write_json(summary_path, {"summaries": summaries})
    maybe_write_plot(output_dir)


if __name__ == "__main__":
    main()
