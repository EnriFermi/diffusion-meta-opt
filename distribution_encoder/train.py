from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from scipy.stats import ks_2samp, wasserstein_distance as scipy_wasserstein
from torch.utils.data import DataLoader

from distribution_encoder.dataset import SyntheticDistributionDataset, static_quantilize
from distribution_encoder.wgan import WGAN_GP, compute_gradient_penalty

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comet tracker (follows experiments/train_mini_vae.py pattern)
# ---------------------------------------------------------------------------
class CometTracker:
    def __init__(self, cfg: DictConfig) -> None:
        self.experiment: Any | None = None
        self.enabled = False

        comet_cfg = cfg.get("telemetry", {}).get("comet", {})
        if not bool(comet_cfg.get("enabled", False)):
            return

        try:
            from comet_ml import Experiment, OfflineExperiment
        except Exception as exc:
            log.warning("Comet enabled but comet_ml unavailable: %s", exc)
            return

        api_key = str(comet_cfg.get("api_key", "")).strip()
        workspace = str(comet_cfg.get("workspace", "")).strip()
        project_name = str(comet_cfg.get("project_name", "distribution_encoder_wgan")).strip()
        experiment_name = str(comet_cfg.get("experiment_name", "")).strip()
        log_code = bool(comet_cfg.get("log_code", False))

        try:
            if api_key:
                exp = Experiment(
                    api_key=api_key,
                    project_name=project_name,
                    workspace=workspace or None,
                    auto_output_logging="simple",
                    log_code=log_code,
                )
            else:
                exp = OfflineExperiment(
                    project_name=project_name,
                    workspace=workspace or None,
                    auto_output_logging="simple",
                    log_code=log_code,
                )

            if experiment_name:
                exp.set_name(experiment_name)

            tags = comet_cfg.get("tags", [])
            if isinstance(tags, (list, tuple)):
                for tag in tags:
                    exp.add_tag(str(tag))

            exp.log_parameters({
                "model.distribution_latent_dim": int(cfg.model.distribution_latent_dim),
                "model.K": int(cfg.model.K),
                "train.max_steps": int(cfg.train.max_steps),
                "train.batch_size": int(cfg.train.batch_size),
                "train.lr": float(cfg.train.lr),
                "train.n_critic": int(cfg.train.n_critic),
                "train.lambda_gp": float(cfg.train.lambda_gp),
                "dataset.b": int(cfg.dataset.b),
                "dataset.n_real": int(cfg.dataset.n_real),
                "dataset.max_modes": int(cfg.dataset.max_modes),
            })

            self.experiment = exp
            self.enabled = True
            log.info("Comet tracking enabled: project=%s workspace=%s", project_name, workspace or "<default>")
        except Exception as exc:
            log.warning("Failed to initialize Comet tracker: %s", exc)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.log_metrics(metrics, step=int(step))
        except Exception as exc:
            log.warning("Comet metrics log failed at step=%s: %s", step, exc)

    def end(self) -> None:
        if self.experiment is None:
            return
        try:
            self.experiment.end()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    wgan: WGAN_GP,
    device: torch.device,
    K: int,
    n_eval_distributions: int = 32,
    n_samples_per_dist: int = 1024,
    n_gen_per_dist: int = 512,
    b_quantile: int = 1024,
    max_modes: int = 5,
) -> dict[str, float]:
    """Generate samples from held-out distributions and compute metrics."""
    from distribution_encoder.dataset import _sample_mixture_numpy
    import numpy as np

    rng = np.random.default_rng(seed=9999)
    wgan.eval()

    w1_scores: list[float] = []
    ks_scores: list[float] = []

    for _ in range(n_eval_distributions):
        raw = _sample_mixture_numpy(rng, n_samples=b_quantile + n_gen_per_dist, max_modes=max_modes)
        raw_t = torch.from_numpy(raw)
        q_samples = raw_t[:b_quantile]
        ref_samples = raw_t[b_quantile:].numpy()

        quantiles, loc_scale = static_quantilize(q_samples, K)
        quantiles = quantiles.unsqueeze(0).to(device)
        loc_scale = loc_scale.unsqueeze(0).to(device)

        # Generate samples
        gen_parts: list[torch.Tensor] = []
        remaining = n_gen_per_dist
        while remaining > 0:
            chunk = min(remaining, 256)
            q_exp = quantiles.expand(chunk, -1)
            ls_exp = loc_scale.expand(chunk, -1)
            noise = torch.randn(chunk, 1, device=device)
            fake = wgan.generator(q_exp, ls_exp, noise)
            gen_parts.append(fake.squeeze(-1).cpu())
            remaining -= chunk
        gen_np = torch.cat(gen_parts).numpy()

        w1 = scipy_wasserstein(ref_samples, gen_np)
        ks_stat, _ = ks_2samp(ref_samples, gen_np)
        w1_scores.append(float(w1))
        ks_scores.append(float(ks_stat))

    wgan.train()
    return {
        "eval/wasserstein1_mean": float(np.mean(w1_scores)),
        "eval/wasserstein1_median": float(np.median(w1_scores)),
        "eval/ks_stat_mean": float(np.mean(ks_scores)),
        "eval/ks_stat_median": float(np.median(ks_scores)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: DictConfig) -> None:
    device = torch.device(str(cfg.train.device))
    seed = int(cfg.train.get("seed", 42))
    torch.manual_seed(seed)

    log.info("Device: %s", device)

    # Model
    dim = int(cfg.model.distribution_latent_dim)
    K = int(cfg.model.K)
    dropout = float(cfg.model.get("dropout", 0.0))
    wgan = WGAN_GP(dim=dim, K=K, dropout=dropout).to(device)
    log.info(
        "WGAN_GP created: dim=%d, K=%d, params=%d",
        dim, K, sum(p.numel() for p in wgan.parameters()),
    )

    # Optimizers
    lr = float(cfg.train.lr)
    betas = tuple(cfg.train.betas)
    critic_opt = torch.optim.Adam(list(wgan.critic_parameters()), lr=lr, betas=betas)
    gen_opt = torch.optim.Adam(list(wgan.generator_parameters()), lr=lr, betas=betas)

    # Dataset
    strategy_weights_raw = cfg.dataset.get("strategy_weights", None)
    if strategy_weights_raw is not None:
        from omegaconf import OmegaConf
        strategy_weights = dict(OmegaConf.to_container(strategy_weights_raw, resolve=True))
    else:
        strategy_weights = None

    ds = SyntheticDistributionDataset(
        K=K,
        b=int(cfg.dataset.b),
        n_real=int(cfg.dataset.n_real),
        max_modes=int(cfg.dataset.max_modes),
        loc_std_range=tuple(cfg.dataset.loc_std_range),
        scale_range=tuple(cfg.dataset.scale_range),
        df_range=tuple(cfg.dataset.df_range),
        skew_range=tuple(cfg.dataset.skew_range),
        strategy_weights=strategy_weights,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), num_workers=2, pin_memory=True)
    data_iter = iter(loader)

    # Training params
    max_steps = int(cfg.train.max_steps)
    n_critic = int(cfg.train.n_critic)
    lambda_gp = float(cfg.train.lambda_gp)
    log_every = int(cfg.train.get("log_every", 50))
    checkpoint_every = int(cfg.train.get("checkpoint_every", 5000))
    eval_every = int(cfg.train.get("eval_every", 1000))
    eval_n_distributions = int(cfg.train.get("eval_n_distributions", 32))
    checkpoint_dir = Path(str(cfg.train.get("checkpoint_dir", "./artifacts/distribution_encoder")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    resume_path = str(cfg.train.get("resume_checkpoint", "")).strip()
    start_step = 0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        wgan.load_state_dict(ckpt["model"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        gen_opt.load_state_dict(ckpt["gen_opt"])
        start_step = int(ckpt.get("step", 0))
        log.info("Resumed from %s at step %d", resume_path, start_step)

    # Comet
    tracker = CometTracker(cfg)

    wgan.train()
    t0 = time.time()
    running_critic_loss = 0.0
    running_gen_loss = 0.0
    running_gp = 0.0

    for step in range(start_step, max_steps):
        # ----- Critic updates -----
        for _ in range(n_critic):
            batch = next(data_iter)
            quantiles = batch["quantiles"].to(device)
            loc_scale = batch["loc_scale"].to(device)
            real_samples = batch["real_samples"].to(device)

            # Pick a random single sample per batch element for critic
            idx = torch.randint(0, real_samples.size(1), (real_samples.size(0),), device=device)
            real_single = real_samples.gather(1, idx.unsqueeze(1))  # [B, 1]

            # Fake
            noise = torch.randn(quantiles.size(0), 1, device=device)
            fake_single = wgan.generator(quantiles, loc_scale, noise).detach()

            real_score = wgan.critic(quantiles, loc_scale, real_single)
            fake_score = wgan.critic(quantiles, loc_scale, fake_single)

            critic_loss = fake_score.mean() - real_score.mean()
            gp = compute_gradient_penalty(wgan.critic, quantiles, loc_scale, real_single, fake_single)
            total_critic_loss = critic_loss + lambda_gp * gp

            critic_opt.zero_grad()
            total_critic_loss.backward()
            critic_opt.step()

            running_critic_loss += critic_loss.item()
            running_gp += gp.item()

        # ----- Generator update -----
        batch = next(data_iter)
        quantiles = batch["quantiles"].to(device)
        loc_scale = batch["loc_scale"].to(device)

        noise = torch.randn(quantiles.size(0), 1, device=device)
        fake_single = wgan.generator(quantiles, loc_scale, noise)
        gen_score = wgan.critic(quantiles, loc_scale, fake_single)
        gen_loss = -gen_score.mean()

        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        running_gen_loss += gen_loss.item()

        # ----- Logging -----
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg_c = running_critic_loss / (log_every * n_critic)
            avg_g = running_gen_loss / log_every
            avg_gp = running_gp / (log_every * n_critic)
            w_dist = -avg_c  # Wasserstein distance estimate
            log.info(
                "step=%d/%d  critic_loss=%.4f  gen_loss=%.4f  gp=%.4f  w_dist=%.4f  time=%.1fs",
                step + 1, max_steps, avg_c, avg_g, avg_gp, w_dist, elapsed,
            )
            tracker.log_metrics({
                "train/critic_loss": avg_c,
                "train/gen_loss": avg_g,
                "train/gradient_penalty": avg_gp,
                "train/wasserstein_distance": w_dist,
            }, step=step + 1)
            running_critic_loss = 0.0
            running_gen_loss = 0.0
            running_gp = 0.0

        # ----- Evaluation -----
        if (step + 1) % eval_every == 0:
            eval_metrics = evaluate(
                wgan, device, K,
                n_eval_distributions=eval_n_distributions,
                b_quantile=int(cfg.dataset.b),
                max_modes=int(cfg.dataset.max_modes),
            )
            log.info("step=%d  eval: %s", step + 1, eval_metrics)
            tracker.log_metrics(eval_metrics, step=step + 1)

        # ----- Checkpoint -----
        if (step + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"wgan_step_{step + 1}.pt"
            torch.save({
                "step": step + 1,
                "model": wgan.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "gen_opt": gen_opt.state_dict(),
            }, ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

    # Final checkpoint
    ckpt_path = checkpoint_dir / "wgan_final.pt"
    torch.save({
        "step": max_steps,
        "model": wgan.state_dict(),
        "critic_opt": critic_opt.state_dict(),
        "gen_opt": gen_opt.state_dict(),
    }, ckpt_path)
    log.info("Training complete. Final checkpoint: %s", ckpt_path)

    tracker.end()


@hydra.main(config_path="../conf/distribution_encoder", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train(cfg)


if __name__ == "__main__":
    main()
