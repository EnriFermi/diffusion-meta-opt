from __future__ import annotations

import torch
import torch.nn as nn

from distribution_encoder.modules import DistrEncoder


class LatentSetEncoder(nn.Module):
    """DeepSets encoder: per-element phi -> mean pool -> rho -> (mu, logvar)."""

    def __init__(self, distr_dim: int, hidden_dim: int, z_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(distr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, distr_dim] set of distribution latent vectors.
        Returns:
            mu: [B, z_dim], logvar: [B, z_dim]
        """
        B, N, D = x.shape
        h = self.phi(x.reshape(B * N, D)).reshape(B, N, -1)
        h = h.mean(dim=1)  # [B, hidden_dim]
        out = self.rho(h)  # [B, 2*z_dim]
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


class LatentSetDecoder(nn.Module):
    """MLP decoder: z -> [B, N, distr_dim]."""

    def __init__(self, z_dim: int, hidden_dim: int, n_distributions: int, distr_dim: int) -> None:
        super().__init__()
        self.n_distributions = n_distributions
        self.distr_dim = distr_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_distributions * distr_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
        Returns:
            [B, n_distributions, distr_dim]
        """
        return self.net(z).reshape(z.shape[0], self.n_distributions, self.distr_dim)


class LatentSetVAE(nn.Module):
    """VAE that compresses a set of distribution latent vectors into a single latent."""

    def __init__(
        self,
        distr_dim: int = 64,
        hidden_dim: int = 256,
        z_dim: int = 32,
        n_distributions: int = 16,
    ) -> None:
        super().__init__()
        self.distr_dim = distr_dim
        self.z_dim = z_dim
        self.n_distributions = n_distributions
        self.hidden_dim = hidden_dim

        self.encoder = LatentSetEncoder(distr_dim, hidden_dim, z_dim)
        self.decoder = LatentSetDecoder(z_dim, hidden_dim, n_distributions, distr_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def recon_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(x_recon, x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, distr_dim]
        Returns:
            x_recon: [B, N, distr_dim], mu: [B, z_dim], logvar: [B, z_dim]
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def config_dict(self) -> dict:
        return {
            "z_dim": self.z_dim,
            "hidden_dim": self.hidden_dim,
            "n_distributions": self.n_distributions,
            "distr_dim": self.distr_dim,
        }


class DistributionSetEncoder(nn.Module):
    """Composition model: DistrEncoder (frozen) -> VAE encoder -> single latent.

    This is the user-facing model for downstream pipelines.
    """

    def __init__(self, distr_encoder: DistrEncoder, vae: LatentSetVAE) -> None:
        super().__init__()
        self.distr_encoder = distr_encoder
        self.distr_encoder.requires_grad_(False)
        self.vae = vae

    @classmethod
    def from_checkpoints(
        cls,
        distr_encoder_ckpt: str,
        vae_ckpt: str,
        map_location: str | torch.device = "cpu",
    ) -> "DistributionSetEncoder":
        """Load from two checkpoint files.

        Args:
            distr_encoder_ckpt: path to distr_encoder_*.pt (contains K, dim, model).
            vae_ckpt: path to latent_vae_*.pt (contains config, model).
        """
        enc_data = torch.load(distr_encoder_ckpt, map_location=map_location, weights_only=False)
        K = int(enc_data["K"])
        dim = int(enc_data["dim"])
        distr_enc = DistrEncoder(K, dim)
        distr_enc.load_state_dict(enc_data["model"])

        vae_data = torch.load(vae_ckpt, map_location=map_location, weights_only=False)
        vae_cfg = vae_data["config"]
        vae = LatentSetVAE(
            distr_dim=int(vae_cfg["distr_dim"]),
            hidden_dim=int(vae_cfg["hidden_dim"]),
            z_dim=int(vae_cfg["z_dim"]),
            n_distributions=int(vae_cfg["n_distributions"]),
        )
        vae.load_state_dict(vae_data["model"])

        return cls(distr_enc, vae)

    def encode(
        self, quantiles: torch.Tensor, loc_scales: torch.Tensor
    ) -> torch.Tensor:
        """Encode a set of distributions into a single latent vector (deterministic).

        Args:
            quantiles: [B, N, K]
            loc_scales: [B, N, 3]

        Returns:
            z: [B, z_dim] (uses mu, no sampling)
        """
        B, N, K = quantiles.shape
        q_flat = quantiles.reshape(B * N, K)
        ls_flat = loc_scales.reshape(B * N, 3)

        with torch.no_grad():
            latents = self.distr_encoder(q_flat, ls_flat)
        latents = latents.reshape(B, N, -1)

        mu, _ = self.vae.encoder(latents)
        return mu

    def encode_stochastic(
        self, quantiles: torch.Tensor, loc_scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode with reparameterization (for training).

        Returns:
            z: [B, z_dim], mu: [B, z_dim], logvar: [B, z_dim]
        """
        B, N, K = quantiles.shape
        q_flat = quantiles.reshape(B * N, K)
        ls_flat = loc_scales.reshape(B * N, 3)

        with torch.no_grad():
            latents = self.distr_encoder(q_flat, ls_flat)
        latents = latents.reshape(B, N, -1)

        mu, logvar = self.vae.encoder(latents)
        z = self.vae.reparameterize(mu, logvar)
        return z, mu, logvar
