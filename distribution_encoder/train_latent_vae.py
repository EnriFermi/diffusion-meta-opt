from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from distribution_encoder.dataset import (
    sample_synthetic_distribution,
    static_quantilize,
)
from distribution_encoder.latent_vae import LatentSetVAE
from distribution_encoder.modules import DistrEncoder
from distribution_encoder.train import CometTracker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch generation: create sets of distribution latents on-the-fly
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_latent_batch(
    distr_encoder: DistrEncoder,
    device: torch.device,
    K: int,
    n_distributions: int,
    batch_size: int,
    rng: np.random.Generator,
    b: int = 256,
    strategy_weights: dict[str, float] | None = None,
    **dataset_kwargs,
) -> torch.Tensor:
    """Generate a batch of distribution latent sets.

    Returns:
        [batch_size, n_distributions, distr_dim]
    """
    all_quantiles = []
    all_loc_scales = []

    for _ in range(batch_size):
        q_list = []
        ls_list = []
        for _ in range(n_distributions):
            raw, qf_override = sample_synthetic_distribution(
                rng,
                n_samples=b,
                K=K,
                strategy_weights=strategy_weights,
                **dataset_kwargs,
            )

            if qf_override is not None:
                q_values = qf_override
                q_min = q_values[0]
                q_max = q_values[-1]
                span = max(q_max - q_min, 1e-8)
                q_norm = 2.0 * (q_values - q_min) / span - 1.0
                mu = float(q_values.mean())
                sigma = float(q_values.std())
                eps = 1e-6
                q_t = torch.from_numpy(q_norm.astype(np.float32))
                ls_t = torch.tensor([
                    np.log1p(abs(mu)),
                    np.sign(mu) if mu != 0 else 0.0,
                    np.log(sigma + eps),
                ], dtype=torch.float32)
            else:
                raw_t = torch.from_numpy(raw)
                q_t, ls_t = static_quantilize(raw_t, K)

            q_list.append(q_t)
            ls_list.append(ls_t)

        all_quantiles.append(torch.stack(q_list))      # [N, K]
        all_loc_scales.append(torch.stack(ls_list))     # [N, 3]

    quantiles = torch.stack(all_quantiles).to(device)   # [B, N, K]
    loc_scales = torch.stack(all_loc_scales).to(device)  # [B, N, 3]

    B, N, K_ = quantiles.shape
    q_flat = quantiles.reshape(B * N, K_)
    ls_flat = loc_scales.reshape(B * N, 3)
    latents = distr_encoder(q_flat, ls_flat).reshape(B, N, -1)

    return latents


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_latent_vae(cfg: DictConfig) -> None:
    vae_cfg = cfg.latent_vae
    device = torch.device(str(vae_cfg.device))
    seed = int(vae_cfg.get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    log.info("Device: %s", device)

    # Load frozen DistrEncoder
    enc_ckpt_path = str(vae_cfg.distr_encoder_checkpoint)
    enc_data = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    K = int(enc_data["K"])
    dim = int(enc_data["dim"])
    distr_encoder = DistrEncoder(K, dim).to(device)
    distr_encoder.load_state_dict(enc_data["model"])
    distr_encoder.eval()
    distr_encoder.requires_grad_(False)
    log.info("Loaded frozen DistrEncoder from %s (K=%d, dim=%d)", enc_ckpt_path, K, dim)

    # VAE model
    z_dim = int(vae_cfg.z_dim)
    hidden_dim = int(vae_cfg.hidden_dim)
    n_distributions = int(vae_cfg.n_distributions)
    vae = LatentSetVAE(
        distr_dim=dim,
        hidden_dim=hidden_dim,
        z_dim=z_dim,
        n_distributions=n_distributions,
    ).to(device)
    log.info(
        "LatentSetVAE: distr_dim=%d, hidden=%d, z_dim=%d, n_dist=%d, params=%d",
        dim, hidden_dim, z_dim, n_distributions,
        sum(p.numel() for p in vae.parameters()),
    )

    # Optimizer
    lr = float(vae_cfg.lr)
    betas = tuple(vae_cfg.betas)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, betas=betas)

    # Comet tracking
    tracker = CometTracker(cfg)

    # Training params
    max_steps = int(vae_cfg.max_steps)
    batch_size = int(vae_cfg.batch_size)
    beta_kl = float(vae_cfg.beta_kl)
    log_every = int(vae_cfg.get("log_every", 50))
    eval_every = int(vae_cfg.get("eval_every", 500))
    checkpoint_every = int(vae_cfg.get("checkpoint_every", 5000))
    checkpoint_dir = Path(str(vae_cfg.get("checkpoint_dir", "./artifacts/latent_vae")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dataset kwargs (reuse from dataset config)
    b = int(cfg.dataset.b)
    strategy_weights_raw = cfg.dataset.get("strategy_weights", None)
    if strategy_weights_raw is not None:
        from omegaconf import OmegaConf
        strategy_weights = dict(OmegaConf.to_container(strategy_weights_raw, resolve=True))
    else:
        strategy_weights = None

    dataset_kwargs = dict(
        max_modes=int(cfg.dataset.max_modes),
        loc_std_range=tuple(cfg.dataset.loc_std_range),
        scale_range=tuple(cfg.dataset.scale_range),
        df_range=tuple(cfg.dataset.df_range),
        skew_range=tuple(cfg.dataset.skew_range),
    )

    # Resume
    start_step = 0
    resume_path = str(vae_cfg.get("resume_checkpoint", "")).strip()
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        log.info("Resumed from %s at step %d", resume_path, start_step)

    # Training
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    t0 = time.time()

    for step in range(start_step, max_steps):
        vae.train()

        latents = generate_latent_batch(
            distr_encoder, device, K,
            n_distributions=n_distributions,
            batch_size=batch_size,
            rng=rng,
            b=b,
            strategy_weights=strategy_weights,
            **dataset_kwargs,
        )

        x_recon, mu, logvar = vae(latents)
        recon = LatentSetVAE.recon_loss(x_recon, latents)
        kl = LatentSetVAE.kl_loss(mu, logvar)
        loss = recon + beta_kl * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_recon += recon.item()
        running_kl += kl.item()

        # ----- Logging -----
        if (step + 1) % log_every == 0:
            n = log_every
            elapsed = time.time() - t0
            metrics = {
                "loss/total": running_loss / n,
                "loss/recon": running_recon / n,
                "loss/kl": running_kl / n,
                "perf/steps_per_sec": n / elapsed,
            }
            log.info(
                "step=%d  loss=%.5f  recon=%.5f  kl=%.5f  steps/s=%.1f",
                step + 1, metrics["loss/total"], metrics["loss/recon"],
                metrics["loss/kl"], metrics["perf/steps_per_sec"],
            )
            tracker.log_metrics(metrics, step=step + 1)
            running_loss = 0.0
            running_recon = 0.0
            running_kl = 0.0
            t0 = time.time()

        # ----- Eval -----
        if (step + 1) % eval_every == 0:
            vae.eval()
            with torch.no_grad():
                eval_latents = generate_latent_batch(
                    distr_encoder, device, K,
                    n_distributions=n_distributions,
                    batch_size=batch_size,
                    rng=np.random.default_rng(9999),
                    b=b,
                    strategy_weights=strategy_weights,
                    **dataset_kwargs,
                )
                x_r, mu_e, lv_e = vae(eval_latents)
                eval_recon = LatentSetVAE.recon_loss(x_r, eval_latents).item()
                eval_kl = LatentSetVAE.kl_loss(mu_e, lv_e).item()

            eval_metrics = {
                "eval/recon_loss": eval_recon,
                "eval/kl_loss": eval_kl,
                "eval/total_loss": eval_recon + beta_kl * eval_kl,
            }
            log.info("step=%d  eval: %s", step + 1, eval_metrics)
            tracker.log_metrics(eval_metrics, step=step + 1)

        # ----- Checkpoint -----
        if (step + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"latent_vae_step_{step + 1}.pt"
            torch.save({
                "step": step + 1,
                "model": vae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vae.config_dict(),
            }, ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

    # Final checkpoint
    ckpt_path = checkpoint_dir / "latent_vae_final.pt"
    torch.save({
        "step": max_steps,
        "model": vae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vae.config_dict(),
    }, ckpt_path)
    log.info("Training complete. Final checkpoint: %s", ckpt_path)

    tracker.end()


@hydra.main(config_path="../conf/distribution_encoder", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train_latent_vae(cfg)


if __name__ == "__main__":
    main()
