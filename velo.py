from __future__ import annotations

from dataclasses import dataclass

import torch

from MetaOpt.models import TransformerActor
from MetaOpt.perf import unwrap_compiled


@dataclass
class VeloUpdateMetrics:
    mean_diff: float
    std_diff: float


class VeloESAgent:
    """
    VeLO-style evolutionary strategies (ES) meta-training.

    Trains only the actor parameters using an antithetic ES gradient estimate:
      g ~ (1/(2*sigma*N)) sum_i (R(θ+σϵ_i) - R(θ-σϵ_i)) * ϵ_i

    Optimizer performs gradient ascent on return (implemented via -g as "grad").
    """

    def __init__(self, state_dim: int, action_dim: int, cfg, device: torch.device):
        self.device = torch.device(device)
        self.action_dim = int(action_dim)

        self.actor = TransformerActor(
            n_params=self.action_dim,
            max_action=cfg.rl.max_action,
            log_std_min=cfg.rl.log_std_min,
            log_std_max=cfg.rl.log_std_max,
        ).to(self.device)

        self.opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(cfg.velo.lr),
            betas=(float(cfg.velo.beta1), float(cfg.velo.beta2)),
            weight_decay=float(cfg.velo.weight_decay),
        )

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Deterministic policy: tanh(mean) * max_action (ignore log_std).
        mean, _ = self.actor(state)
        a = torch.tanh(mean) * float(unwrap_compiled(self.actor).max_action)
        return a

    @torch.no_grad()
    def _add_noise_(self, params, noises, scale: float):
        for p, e in zip(params, noises):
            p.add_(scale * e)

    @torch.no_grad()
    def _eval_return(self, eval_fn, episodes_per_eval: int) -> float:
        total = 0.0
        for _ in range(int(episodes_per_eval)):
            r = float(eval_fn())
            total += r
        return total / float(episodes_per_eval)

    def update(self, cfg, eval_fn) -> VeloUpdateMetrics:
        """
        One ES update step.

        eval_fn: callable that returns episode return (float) for current actor parameters.
        """
        pop_size = int(cfg.velo.population_size)
        if pop_size <= 0:
            raise ValueError("velo.population_size must be > 0")
        sigma = float(cfg.velo.sigma)
        if sigma <= 0:
            raise ValueError("velo.sigma must be > 0")

        base_actor = unwrap_compiled(self.actor)
        params = [p for p in base_actor.parameters() if p.requires_grad]

        grad_accum = [torch.zeros_like(p) for p in params]
        eps_accum = [torch.zeros_like(p) for p in params]
        sum_diff = 0.0
        sum_diff2 = 0.0

        episodes_per_eval = int(cfg.velo.episodes_per_eval)
        if episodes_per_eval <= 0:
            raise ValueError("velo.episodes_per_eval must be > 0")

        for _ in range(pop_size):
            noises = [torch.randn_like(p) for p in params]

            self._add_noise_(params, noises, sigma)
            r_plus = self._eval_return(eval_fn, episodes_per_eval=episodes_per_eval)

            self._add_noise_(params, noises, -2.0 * sigma)
            r_minus = self._eval_return(eval_fn, episodes_per_eval=episodes_per_eval)

            self._add_noise_(params, noises, sigma)  # restore

            diff = float(r_plus - r_minus)
            sum_diff += diff
            sum_diff2 += diff * diff

            for g, e, ea in zip(grad_accum, noises, eps_accum):
                g.add_(diff * e)
                ea.add_(e)

        mean_diff = sum_diff / float(pop_size)
        var = max(0.0, sum_diff2 / float(pop_size) - mean_diff * mean_diff)
        std_diff = (var + 1e-8) ** 0.5

        if bool(cfg.velo.normalize_diffs):
            inv_std = 1.0 / std_diff
            for g, ea in zip(grad_accum, eps_accum):
                g.sub_(mean_diff * ea).mul_(inv_std)

        scale = 1.0 / (2.0 * float(pop_size) * sigma)
        for g in grad_accum:
            g.mul_(scale)

        clip_norm = float(getattr(cfg.velo, "grad_clip_norm", 0.0))
        if clip_norm and clip_norm > 0:
            total_norm_sq = 0.0
            for g in grad_accum:
                total_norm_sq += float(g.float().pow(2).sum().item())
            total_norm = (total_norm_sq + 1e-12) ** 0.5
            if total_norm > clip_norm:
                c = clip_norm / total_norm
                for g in grad_accum:
                    g.mul_(c)

        self.opt.zero_grad(set_to_none=True)
        for p, g in zip(params, grad_accum):
            # Adam performs gradient descent; negate to do ascent on return.
            p.grad = -g
        self.opt.step()

        return VeloUpdateMetrics(mean_diff=mean_diff, std_diff=std_diff)

    def state_dict(self):
        return {
            "actor": unwrap_compiled(self.actor).state_dict(),
            "opt": self.opt.state_dict(),
        }

    def load_state_dict(self, state):
        unwrap_compiled(self.actor).load_state_dict(state["actor"])
        opt_state = state.get("opt")
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
