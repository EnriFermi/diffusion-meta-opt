from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from distribution_encoder.modules import Critic, DistrEncoder, Generator


def compute_gradient_penalty(
    critic: Critic,
    quantiles: torch.Tensor,
    loc_scale: torch.Tensor,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP gradient penalty on interpolated samples.

    Computes the penalty on the critic's gradient w.r.t. interpolated
    points between real and fake samples (not w.r.t. the distribution
    description, which is held fixed).
    """
    B = real_samples.size(0)
    alpha = torch.rand(B, 1, device=real_samples.device, dtype=real_samples.dtype)
    interpolated = (alpha * real_samples + (1.0 - alpha) * fake_samples).requires_grad_(True)

    interp_score = critic(quantiles, loc_scale, interpolated)

    grad = torch.autograd.grad(
        outputs=interp_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interp_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grad.norm(2, dim=1)
    penalty = ((grad_norm - 1.0) ** 2).mean()
    return penalty


class WGAN_GP(nn.Module):
    """Conditional Wasserstein GAN with gradient penalty.

    Generator and Critic share the same DistrEncoder instance.
    """

    def __init__(self, dim: int, K: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.generator = Generator(dim, K, dropout)
        self.critic = Critic(dim, self.generator.distr_encoder, dropout)

    def generator_parameters(self) -> Iterator[nn.Parameter]:
        """All generator params (including shared distr_encoder)."""
        return self.generator.parameters()

    def critic_only_parameters(self) -> Iterator[nn.Parameter]:
        """Critic params excluding the shared distr_encoder."""
        shared_ids = {id(p) for p in self.generator.distr_encoder.parameters()}
        return (p for p in self.critic.parameters() if id(p) not in shared_ids)

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        """All critic params (including shared distr_encoder)."""
        return self.critic.parameters()
