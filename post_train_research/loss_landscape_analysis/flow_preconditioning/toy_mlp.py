from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import ExperimentConfig


MLP_DIM = 25
MLP_WIDTH = 8


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def seed_offset(seed: int, offset: int) -> int:
    return int(seed) * 100_000 + int(offset)


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(3.0 * x) + 0.3 * torch.sign(torch.sin(9.0 * x))


def shifted_grid(n: int, shift: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Wrapping keeps shifted deterministic grids inside [-2, 2] while making splits non-identical.
    fraction = torch.remainder(torch.arange(int(n), device=device, dtype=dtype) + 0.5 + float(shift), int(n))
    fraction = torch.sort(fraction / float(n)).values
    return -2.0 + 4.0 * fraction


def unpack_theta(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if theta.ndim != 1 or int(theta.numel()) != MLP_DIM:
        raise ValueError(f"theta must be flat [{MLP_DIM}], got {tuple(theta.shape)}")
    w1 = theta[0:8].reshape(MLP_WIDTH, 1)
    b1 = theta[8:16]
    w2 = theta[16:24].reshape(1, MLP_WIDTH)
    b2 = theta[24:25]
    return w1, b1, w2, b2


def pack_theta(w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    flat = torch.cat([w1.reshape(-1), b1.reshape(-1), w2.reshape(-1), b2.reshape(-1)], dim=0)
    if int(flat.numel()) != MLP_DIM:
        raise ValueError(f"packed theta must have {MLP_DIM} entries, got {int(flat.numel())}")
    return flat


def mlp_forward(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = unpack_theta(theta)
    hidden = F.relu(x.reshape(-1, 1) @ w1.t() + b1.reshape(1, -1))
    return (hidden @ w2.t() + b2.reshape(1, 1)).reshape(-1)


def sample_initializations(
    count: int,
    dim: int,
    *,
    init_std: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    samples = torch.randn(int(count), int(dim), generator=make_generator(seed), dtype=dtype) * float(init_std)
    return samples.to(device=device)


@dataclass(slots=True)
class MLPRegressionProblem:
    cfg: ExperimentConfig
    seed: int
    device: torch.device
    dtype: torch.dtype
    dim: int = field(init=False)
    x_train: torch.Tensor = field(init=False)
    x_probe: torch.Tensor = field(init=False)
    x_test: torch.Tensor = field(init=False)
    y_train: torch.Tensor = field(init=False)
    y_probe: torch.Tensor = field(init=False)
    y_test: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if int(self.cfg.mlp_width) != MLP_WIDTH:
            raise ValueError(f"fixed protocol requires mlp_width={MLP_WIDTH}, got {self.cfg.mlp_width}")
        if int(self.cfg.mlp_dim) != MLP_DIM:
            raise ValueError(f"fixed protocol requires mlp_dim={MLP_DIM}, got {self.cfg.mlp_dim}")
        self.dim = MLP_DIM
        self.x_train = shifted_grid(self.cfg.train_points, self.cfg.train_shift, device=self.device, dtype=self.dtype)
        self.x_probe = shifted_grid(self.cfg.probe_points, self.cfg.probe_shift, device=self.device, dtype=self.dtype)
        self.x_test = shifted_grid(self.cfg.test_points, self.cfg.test_shift, device=self.device, dtype=self.dtype)
        self.y_train = target_function(self.x_train)
        self.y_probe = target_function(self.x_probe)
        self.y_test = target_function(self.x_test)

    def train_residual(self, theta: torch.Tensor) -> torch.Tensor:
        return mlp_forward(theta, self.x_train) - self.y_train

    def probe_residual(self, theta: torch.Tensor) -> torch.Tensor:
        return mlp_forward(theta, self.x_probe) - self.y_probe

    def test_residual(self, theta: torch.Tensor) -> torch.Tensor:
        return mlp_forward(theta, self.x_test) - self.y_test

    def train_loss(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.train_residual(theta).pow(2).mean()

    def test_loss(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.test_residual(theta).pow(2).mean()

    def sample_starts(self, count: int, *, seed: int) -> torch.Tensor:
        return sample_initializations(
            int(count),
            self.dim,
            init_std=float(self.cfg.init_std),
            seed=int(seed),
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(slots=True)
class QuadraticProblem:
    dim: int
    condition_number: float
    seed: int
    device: torch.device
    dtype: torch.dtype
    a: torch.Tensor = field(init=False)
    z_star: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        generator = make_generator(seed_offset(self.seed, 701))
        q_left, _ = torch.linalg.qr(torch.randn(self.dim, self.dim, generator=generator, dtype=self.dtype))
        q_right, _ = torch.linalg.qr(torch.randn(self.dim, self.dim, generator=generator, dtype=self.dtype))
        singular = torch.logspace(
            0.0,
            float(torch.log10(torch.tensor(float(self.condition_number))).item()),
            steps=int(self.dim),
            dtype=self.dtype,
        )
        self.a = (q_left @ torch.diag(singular) @ q_right.t()).to(device=self.device)
        self.z_star = (
            torch.randn(int(self.dim), generator=generator, dtype=self.dtype) * 0.25
        ).to(device=self.device)

    def residual(self, theta: torch.Tensor) -> torch.Tensor:
        return self.a @ (theta - self.z_star)

    def train_loss(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.residual(theta).pow(2).sum()

    def test_loss(self, theta: torch.Tensor) -> torch.Tensor:
        return self.train_loss(theta)

    def sample_starts(self, count: int, *, seed: int, init_std: float) -> torch.Tensor:
        return sample_initializations(
            int(count),
            int(self.dim),
            init_std=float(init_std),
            seed=int(seed),
            device=self.device,
            dtype=self.dtype,
        )


def collect_trajectory_samples(
    train_loss_fn,
    starts: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in starts:
        theta = torch.nn.Parameter(start.detach().clone())
        optimizer = torch.optim.Adam([theta], lr=float(lr))
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss_fn(theta)
            loss.backward()
            optimizer.step()
            rows.append(theta.detach().clone())
    if not rows:
        return starts.new_empty((0, int(starts.shape[1])))
    return torch.stack(rows, dim=0)


def collect_mlp_flow_pool(problem: MLPRegressionProblem, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
    random_samples = problem.sample_starts(cfg.flow_random_samples, seed=seed_offset(seed, 101))
    trajectory_starts = problem.sample_starts(cfg.flow_trajectory_count, seed=seed_offset(seed, 102))
    trajectory_samples = collect_trajectory_samples(
        problem.train_loss,
        trajectory_starts,
        steps=int(cfg.flow_trajectory_steps),
        lr=float(cfg.flow_trajectory_lr),
    )
    return torch.cat([random_samples, trajectory_samples], dim=0).detach()


def collect_mlp_heldout_geometry_pool(problem: MLPRegressionProblem, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
    total = int(cfg.heldout_geometry_samples)
    random_count = max(1, int(round(total * float(cfg.flow_random_samples) / float(cfg.flow_random_samples + cfg.flow_trajectory_count * cfg.flow_trajectory_steps))))
    trajectory_count = max(1, int(torch.ceil(torch.tensor((total - random_count) / max(1, cfg.flow_trajectory_steps))).item()))
    random_samples = problem.sample_starts(random_count, seed=seed_offset(seed, 201))
    trajectory_starts = problem.sample_starts(trajectory_count, seed=seed_offset(seed, 202))
    trajectory_samples = collect_trajectory_samples(
        problem.train_loss,
        trajectory_starts,
        steps=int(cfg.flow_trajectory_steps),
        lr=float(cfg.flow_trajectory_lr),
    )
    return torch.cat([random_samples, trajectory_samples], dim=0)[:total].detach()


def collect_quadratic_flow_pool(problem: QuadraticProblem, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
    random_samples = problem.sample_starts(
        cfg.sanity_random_samples,
        seed=seed_offset(seed, 301),
        init_std=float(cfg.init_std),
    )
    trajectory_starts = problem.sample_starts(
        cfg.sanity_trajectory_count,
        seed=seed_offset(seed, 302),
        init_std=float(cfg.init_std),
    )
    trajectory_samples = collect_trajectory_samples(
        problem.train_loss,
        trajectory_starts,
        steps=int(cfg.sanity_trajectory_steps),
        lr=float(cfg.flow_trajectory_lr),
    )
    return torch.cat([random_samples, trajectory_samples], dim=0).detach()


def collect_quadratic_heldout_geometry_pool(problem: QuadraticProblem, cfg: ExperimentConfig, *, seed: int) -> torch.Tensor:
    total = int(cfg.sanity_heldout_geometry_samples)
    random_count = max(1, total // 2)
    trajectory_count = max(1, int(torch.ceil(torch.tensor((total - random_count) / max(1, cfg.sanity_trajectory_steps))).item()))
    random_samples = problem.sample_starts(random_count, seed=seed_offset(seed, 401), init_std=float(cfg.init_std))
    trajectory_starts = problem.sample_starts(trajectory_count, seed=seed_offset(seed, 402), init_std=float(cfg.init_std))
    trajectory_samples = collect_trajectory_samples(
        problem.train_loss,
        trajectory_starts,
        steps=int(cfg.sanity_trajectory_steps),
        lr=float(cfg.flow_trajectory_lr),
    )
    return torch.cat([random_samples, trajectory_samples], dim=0)[:total].detach()
