from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .toy_mlp import collect_trajectory_samples, make_generator, sample_initializations, seed_offset


def _uniform(
    count: int,
    dim: int,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = make_generator(seed)
    samples = torch.rand(int(count), int(dim), generator=generator, dtype=dtype)
    return (low.cpu() + (high.cpu() - low.cpu()) * samples).to(device=device, dtype=dtype)


@dataclass(slots=True)
class StartItem:
    start: torch.Tensor
    target: torch.Tensor | None = None
    target_aux: torch.Tensor | None = None


@dataclass(slots=True)
class Decoder2DProblem:
    cfg: ExperimentConfig
    seed: int
    device: torch.device
    dtype: torch.dtype
    dim: int = field(init=False, default=2)
    low: torch.Tensor = field(init=False)
    high: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.low = torch.tensor(
            [float(self.cfg.e1_domain_z1[0]), float(self.cfg.e1_domain_z2[0])],
            device=self.device,
            dtype=self.dtype,
        )
        self.high = torch.tensor(
            [float(self.cfg.e1_domain_z1[1]), float(self.cfg.e1_domain_z2[1])],
            device=self.device,
            dtype=self.dtype,
        )

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[0], z[1]
        radius = 2.0 + 0.4 * z1
        return torch.stack(
            [
                radius * torch.cos(z2),
                radius * torch.sin(z2),
                0.25 * z2 + 0.3 * torch.sin(2.0 * z1),
            ],
            dim=0,
        )

    def probe(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def loss_for_target(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.decoder(z) - target).pow(2).sum()

    def sample_domain(self, count: int, *, seed: int) -> torch.Tensor:
        return _uniform(
            int(count),
            self.dim,
            self.low,
            self.high,
            seed=int(seed),
            device=self.device,
            dtype=self.dtype,
        )

    def sample_pairs(self, count: int, *, seed: int) -> list[StartItem]:
        z_star = self.sample_domain(int(count), seed=seed_offset(seed, 1))
        starts = self.sample_domain(int(count), seed=seed_offset(seed, 2))
        return [StartItem(start=starts[idx], target=self.decoder(z_star[idx]).detach(), target_aux=z_star[idx]) for idx in range(int(count))]

    def flow_pool(self, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
        random_samples = self.sample_domain(int(cfg.e1_random_samples), seed=seed_offset(seed, 11))
        pairs = self.sample_pairs(int(cfg.e1_trajectory_count), seed=seed_offset(seed, 12))
        rows = []
        for item in pairs:
            rows.append(
                collect_trajectory_samples(
                    lambda z, target=item.target: self.loss_for_target(z, target),
                    item.start.reshape(1, -1),
                    steps=int(cfg.e1_trajectory_steps),
                    lr=float(cfg.flow_trajectory_lr),
                )
            )
        trajectory = torch.cat(rows, dim=0) if rows else random_samples.new_empty((0, self.dim))
        return torch.cat([random_samples, trajectory], dim=0).detach()

    def heldout_pool(self, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
        return self.sample_domain(int(cfg.e1_heldout_geometry_samples), seed=seed_offset(seed, 13)).detach()


@dataclass(slots=True)
class ScalarLandscapeProblem:
    cfg: ExperimentConfig
    seed: int
    dim: int
    objective_name: str
    device: torch.device
    dtype: torch.dtype
    a: torch.Tensor = field(init=False)
    b: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        generator = make_generator(seed_offset(self.seed, 2100 + self.dim + sum(ord(c) for c in self.objective_name)))
        self.a = torch.randn(16, int(self.dim), generator=generator, dtype=self.dtype).to(self.device)
        self.b = torch.randn(16, generator=generator, dtype=self.dtype).to(self.device)

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        name = str(self.objective_name)
        ridges = 0.15 * torch.abs(self.a @ z + self.b).sum()
        if name == "rastrigin_abs":
            base = 10.0 * int(self.dim) + (z.square() - 10.0 * torch.cos(2.0 * torch.pi * z)).sum()
            return base + ridges
        if name == "rosenbrock_abs":
            if int(self.dim) == 1:
                base = (1.0 - z[0]).square()
            else:
                base = (100.0 * (z[1:] - z[:-1].square()).square() + (1.0 - z[:-1]).square()).sum()
            return base + ridges
        raise ValueError(f"unknown scalar objective {self.objective_name!r}")

    def sample_starts(self, count: int, *, seed: int) -> torch.Tensor:
        return sample_initializations(
            int(count),
            int(self.dim),
            init_std=float(self.cfg.e2_init_std),
            seed=int(seed),
            device=self.device,
            dtype=self.dtype,
        )

    def flow_pool(self, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
        random_samples = self.sample_starts(int(cfg.e2_random_samples), seed=seed_offset(seed, 21))
        trajectory_starts = self.sample_starts(int(cfg.e2_trajectory_count), seed=seed_offset(seed, 22))
        trajectory = collect_trajectory_samples(
            self.loss,
            trajectory_starts,
            steps=int(cfg.e2_trajectory_steps),
            lr=float(cfg.flow_trajectory_lr),
        )
        return torch.cat([random_samples, trajectory], dim=0).detach()

    def heldout_pool(self, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
        return self.sample_starts(int(cfg.e2_heldout_geometry_samples), seed=seed_offset(seed, 23)).detach()


@dataclass(frozen=True, slots=True)
class GraphProbeScales:
    z_rms: float
    loss_rms: float


def fit_graph_probe_scales(loss_fn: Callable[[torch.Tensor], torch.Tensor], samples: torch.Tensor, *, eps: float = 1e-12) -> GraphProbeScales:
    with torch.no_grad():
        losses = torch.stack([loss_fn(sample).detach().reshape(()) for sample in samples], dim=0)
        z_rms = samples.pow(2).mean().sqrt().clamp_min(float(eps))
        loss_rms = losses.pow(2).mean().sqrt().clamp_min(float(eps))
    return GraphProbeScales(float(z_rms.cpu().item()), float(loss_rms.cpu().item()))


@dataclass(frozen=True, slots=True)
class GraphProbe:
    loss_fn: Callable[[torch.Tensor], torch.Tensor]
    scales: GraphProbeScales
    gamma: float
    rho: float = 1.0
    eps: float = 1e-12

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        z_part = float(self.rho) * z / max(float(self.scales.z_rms), float(self.eps))
        loss_part = float(self.gamma) * self.loss_fn(z).reshape(1) / max(float(self.scales.loss_rms), float(self.eps))
        return torch.cat([z_part.reshape(-1), loss_part], dim=0)


@dataclass(slots=True)
class NonsmoothResidualProblem:
    cfg: ExperimentConfig
    seed: int
    condition_number: float
    device: torch.device
    dtype: torch.dtype
    dim: int = field(init=False)
    q: torch.Tensor = field(init=False)
    scale: torch.Tensor = field(init=False)
    b1: torch.Tensor = field(init=False)
    b1_matrix: torch.Tensor = field(init=False)
    b2_matrix: torch.Tensor = field(init=False)
    z_star: torch.Tensor = field(init=False)
    target: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.dim = int(self.cfg.e3_dim)
        generator = make_generator(seed_offset(self.seed, 3100 + int(round(float(self.condition_number)))))
        q, _ = torch.linalg.qr(torch.randn(self.dim, self.dim, generator=generator, dtype=self.dtype))
        self.q = q.to(self.device)
        if self.dim == 1:
            scales = torch.ones(1, dtype=self.dtype)
        else:
            scales = torch.logspace(0.0, float(torch.log10(torch.tensor(float(self.condition_number))).item()), steps=self.dim, dtype=self.dtype)
        self.scale = scales.to(self.device)
        hidden = int(self.cfg.e3_hidden_dim)
        out_dim = int(self.cfg.e3_output_dim)
        self.b1_matrix = (torch.randn(hidden, self.dim, generator=generator, dtype=self.dtype) / float(self.dim) ** 0.5).to(self.device)
        self.b2_matrix = (torch.randn(out_dim, hidden, generator=generator, dtype=self.dtype) / float(hidden) ** 0.5).to(self.device)
        self.b1 = (torch.randn(hidden, generator=generator, dtype=self.dtype) * 0.1).to(self.device)
        self.z_star = (torch.randn(self.dim, generator=generator, dtype=self.dtype) * float(self.cfg.init_std)).to(self.device)
        self.target = self.probe(self.z_star).detach()

    def probe(self, z: torch.Tensor) -> torch.Tensor:
        transformed = self.scale * (self.q @ z)
        hidden = F.relu(self.b1_matrix @ transformed + self.b1)
        residual = self.b2_matrix @ hidden
        skip = torch.zeros_like(residual)
        skip[: self.dim] = z
        return residual + float(self.cfg.e3_skip) * skip

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.probe(z) - self.target).pow(2).sum()

    def sample_starts(self, count: int, *, seed: int) -> torch.Tensor:
        return sample_initializations(
            int(count),
            self.dim,
            init_std=float(self.cfg.init_std),
            seed=int(seed),
            device=self.device,
            dtype=self.dtype,
        )

    def flow_pool(self, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
        random_samples = self.sample_starts(int(cfg.e3_random_samples), seed=seed_offset(seed, 31))
        trajectory_starts = self.sample_starts(int(cfg.e3_trajectory_count), seed=seed_offset(seed, 32))
        trajectory = collect_trajectory_samples(
            self.loss,
            trajectory_starts,
            steps=int(cfg.e3_trajectory_steps),
            lr=float(cfg.flow_trajectory_lr),
        )
        return torch.cat([random_samples, trajectory], dim=0).detach()

    def heldout_pool(self, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
        return self.sample_starts(int(cfg.e3_heldout_geometry_samples), seed=seed_offset(seed, 33)).detach()
