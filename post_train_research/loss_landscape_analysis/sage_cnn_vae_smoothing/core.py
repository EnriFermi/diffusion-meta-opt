from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset, TensorDataset

from post_train_research.big_vae_latent_flattening.flow import RQSplineFlow, RQSplineFlowConfig
from post_train_research.loss_landscape_analysis.flow_preconditioning.probe_geometry import (
    exact_jacobians,
    isometry_objective_from_jacobians,
    metric_tensors_from_jacobians,
)

from .config import ExperimentConfig, torch_dtype
from .progress import make_progress


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        h = F.max_pool2d(F.relu(self.conv2(h)), kernel_size=2)
        h = h.reshape(h.shape[0], -1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


@dataclass(frozen=True, slots=True)
class FlatSpec:
    keys: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]

    @property
    def dim(self) -> int:
        return int(sum(self.sizes))


def tiny_cnn_spec() -> FlatSpec:
    model = TinyCNN()
    keys: list[str] = []
    shapes: list[tuple[int, ...]] = []
    sizes: list[int] = []
    for key, value in model.state_dict().items():
        keys.append(str(key))
        shapes.append(tuple(int(v) for v in value.shape))
        sizes.append(int(value.numel()))
    return FlatSpec(tuple(keys), tuple(shapes), tuple(sizes))


def state_dict_to_flat(state_dict: dict[str, torch.Tensor], spec: FlatSpec) -> torch.Tensor:
    return torch.cat([state_dict[key].detach().reshape(-1) for key in spec.keys], dim=0)


def flat_to_state_dict(flat: torch.Tensor, spec: FlatSpec) -> dict[str, torch.Tensor]:
    if int(flat.numel()) != int(spec.dim):
        raise ValueError(f"flat vector must have {spec.dim} values, got {int(flat.numel())}")
    state: dict[str, torch.Tensor] = {}
    offset = 0
    for key, shape, size in zip(spec.keys, spec.shapes, spec.sizes, strict=True):
        state[key] = flat[offset : offset + int(size)].reshape(shape)
        offset += int(size)
    return state


def tiny_cnn_logits_from_flat(flat: torch.Tensor, images: torch.Tensor, spec: FlatSpec) -> torch.Tensor:
    model = TinyCNN().to(device=images.device, dtype=images.dtype)
    state = flat_to_state_dict(flat.to(device=images.device, dtype=images.dtype), spec)
    return functional_call(model, state, (images,))


def _dataset_class(name: str):
    from torchvision import datasets

    value = str(name).strip().lower()
    if value in {"fashion_mnist", "fashionmnist"}:
        return datasets.FashionMNIST
    if value == "mnist":
        return datasets.MNIST
    raise ValueError(f"dataset_name must be fashion_mnist or mnist, got {name!r}")


def load_vision_tensors(cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision import transforms

    dataset_cls = _dataset_class(cfg.dataset_name)
    transform = transforms.Compose([transforms.ToTensor()])
    root = Path(cfg.data_root).expanduser().resolve()
    train_set = dataset_cls(root=str(root), train=True, download=bool(cfg.download), transform=transform)
    test_set = dataset_cls(root=str(root), train=False, download=bool(cfg.download), transform=transform)
    train_count = min(int(cfg.train_subset), len(train_set))
    test_count = min(int(cfg.test_subset), len(test_set))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed))
    train_indices = torch.randperm(len(train_set), generator=generator)[:train_count].tolist()
    test_indices = torch.randperm(len(test_set), generator=generator)[:test_count].tolist()
    train_loader = DataLoader(Subset(train_set, train_indices), batch_size=train_count, shuffle=False)
    test_loader = DataLoader(Subset(test_set, test_indices), batch_size=test_count, shuffle=False)
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    return train_images, train_labels.long(), test_images, test_labels.long()


def make_tensor_loaders(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=int(batch_size), shuffle=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=int(batch_size), shuffle=False)
    return train_loader, test_loader


def evaluate_flat_model(
    flat: torch.Tensor,
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    spec: FlatSpec,
) -> tuple[float, float]:
    with torch.no_grad():
        logits = tiny_cnn_logits_from_flat(flat, images, spec)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
    return float(loss.detach().cpu().item()), float(acc.detach().cpu().item())


def generate_weight_pool(
    cfg: ExperimentConfig,
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    output_path: Path,
) -> tuple[torch.Tensor, pd.DataFrame, FlatSpec]:
    if output_path.is_file() and bool(cfg.cache_first) and not bool(cfg.force_rerun):
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        spec = FlatSpec(tuple(payload["spec"]["keys"]), tuple(tuple(s) for s in payload["spec"]["shapes"]), tuple(payload["spec"]["sizes"]))
        return payload["weights"], pd.DataFrame(payload["records"]), spec

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    train_images = train_images.to(device=device, dtype=dtype)
    train_labels = train_labels.to(device=device)
    test_images = test_images.to(device=device, dtype=dtype)
    test_labels = test_labels.to(device=device)
    spec = tiny_cnn_spec()
    flats: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    snapshot_every = max(1, int(cfg.weight_snapshot_every))
    snapshot_steps = set(range(0, int(cfg.weight_train_steps) + 1, snapshot_every))
    snapshot_steps.add(int(cfg.weight_train_steps))
    train_loader, _test_loader = make_tensor_loaders(
        train_images,
        train_labels,
        test_images,
        test_labels,
        batch_size=int(cfg.cnn_batch_size),
    )
    progress = make_progress(cfg, total=int(cfg.weight_runs) * (int(cfg.weight_train_steps) + 1), desc="weight pool")
    try:
        for run_idx in range(int(cfg.weight_runs)):
            torch.manual_seed(int(cfg.seed) + 1000 + run_idx)
            model = TinyCNN().to(device=device, dtype=dtype)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.weight_lr))
            loader_iter = iter(train_loader)
            for step in range(int(cfg.weight_train_steps) + 1):
                if step in snapshot_steps:
                    flat = state_dict_to_flat(model.state_dict(), spec).detach().cpu()
                    train_loss, train_acc = evaluate_flat_model(flat.to(device=device, dtype=dtype), images=train_images, labels=train_labels, spec=spec)
                    test_loss, test_acc = evaluate_flat_model(flat.to(device=device, dtype=dtype), images=test_images, labels=test_labels, spec=spec)
                    flats.append(flat)
                    records.append(
                        {
                            "run": int(run_idx),
                            "step": int(step),
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                        }
                    )
                    progress.set_postfix({"run": run_idx, "step": step, "snapshots": len(flats), "test_acc": f"{test_acc:.3f}"})
                progress.update(1)
                if step == int(cfg.weight_train_steps):
                    break
                try:
                    batch_images, batch_labels = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch_images, batch_labels = next(loader_iter)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_images)
                loss = F.cross_entropy(logits, batch_labels)
                loss.backward()
                optimizer.step()
    finally:
        progress.close()

    weights = torch.stack(flats, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": weights,
            "records": records,
            "spec": {"keys": spec.keys, "shapes": spec.shapes, "sizes": spec.sizes},
        },
        output_path,
    )
    return weights, pd.DataFrame(records), spec


@dataclass(slots=True)
class WeightNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    @classmethod
    def fit(cls, values: torch.Tensor, *, eps: float = 1e-6) -> "WeightNormalizer":
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(float(eps))
        return cls(mean=mean.detach(), std=std.detach(), eps=float(eps))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(device=values.device, dtype=values.dtype)) / self.std.to(device=values.device, dtype=values.dtype)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(device=values.device, dtype=values.dtype) + self.mean.to(device=values.device, dtype=values.dtype)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu(), "eps": torch.tensor(float(self.eps))}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "WeightNormalizer":
        return cls(mean=state["mean"], std=state["std"], eps=float(state.get("eps", torch.tensor(1e-6)).item()))


class WeightVAE(nn.Module):
    def __init__(self, *, weight_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.weight_dim = int(weight_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(int(weight_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(int(hidden_dim), int(latent_dim))
        self.to_logvar = nn.Linear(int(hidden_dim), int(latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(weight_dim)),
        )

    def encode(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x_norm)
        return self.to_mu(h), self.to_logvar(h).clamp(min=-12.0, max=8.0)

    def decode_norm(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_norm)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        return self.decode_norm(z), mu, logvar


def vae_loss(x_norm: torch.Tensor, recon_norm: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, *, beta_kl: float) -> tuple[torch.Tensor, dict[str, float]]:
    recon = F.mse_loss(recon_norm, x_norm)
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
    loss = recon + float(beta_kl) * kl
    return loss, {"loss": float(loss.detach().cpu().item()), "recon_mse": float(recon.detach().cpu().item()), "kl": float(kl.detach().cpu().item())}


@dataclass(slots=True)
class TrainedVAE:
    vae: WeightVAE
    normalizer: WeightNormalizer
    train_indices: torch.Tensor
    val_indices: torch.Tensor
    metrics: pd.DataFrame


def train_weight_vae(cfg: ExperimentConfig, weights: torch.Tensor, *, output_path: Path) -> TrainedVAE:
    if output_path.is_file() and bool(cfg.cache_first) and not bool(cfg.force_rerun):
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        normalizer = WeightNormalizer.from_state_dict(payload["normalizer"])
        vae = WeightVAE(weight_dim=int(weights.shape[1]), latent_dim=int(cfg.latent_dim), hidden_dim=int(cfg.vae_hidden_dim))
        vae.load_state_dict(payload["model_state"])
        return TrainedVAE(
            vae=vae,
            normalizer=normalizer,
            train_indices=payload["train_indices"],
            val_indices=payload["val_indices"],
            metrics=pd.DataFrame(payload["metrics"]),
        )

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed) + 2000)
    perm = torch.randperm(int(weights.shape[0]), generator=generator)
    train_count = max(1, int(round(float(cfg.vae_train_fraction) * int(weights.shape[0]))))
    train_indices = perm[:train_count]
    val_indices = perm[train_count:]
    if int(val_indices.numel()) == 0:
        val_indices = train_indices[:1]
    normalizer = WeightNormalizer.fit(weights.index_select(0, train_indices))
    weights_device = weights.to(device=device, dtype=dtype)
    x_norm = normalizer.normalize(weights_device)
    vae = WeightVAE(weight_dim=int(weights.shape[1]), latent_dim=int(cfg.latent_dim), hidden_dim=int(cfg.vae_hidden_dim)).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(vae.parameters(), lr=float(cfg.vae_lr))
    metrics: list[dict[str, float]] = []
    train_indices_device = train_indices.to(device=device)
    batch_size = min(max(1, int(cfg.vae_batch_size)), int(train_indices.numel()))
    progress = make_progress(cfg, total=int(cfg.vae_steps), desc="weight VAE")
    try:
        for step in range(1, int(cfg.vae_steps) + 1):
            batch_pos = torch.randint(0, int(train_indices.numel()), (batch_size,), generator=generator, device="cpu").to(device=device)
            batch = x_norm.index_select(0, train_indices_device.index_select(0, batch_pos))
            vae.train()
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = vae(batch)
            loss, row = vae_loss(batch, recon, mu, logvar, beta_kl=float(cfg.beta_kl))
            loss.backward()
            optimizer.step()
            if step == 1 or step % max(1, int(cfg.vae_steps) // 20) == 0 or step == int(cfg.vae_steps):
                vae.eval()
                with torch.no_grad():
                    val = x_norm.index_select(0, val_indices.to(device=device))
                    recon_val, mu_val, logvar_val = vae(val)
                    _loss_val, val_row = vae_loss(val, recon_val, mu_val, logvar_val, beta_kl=float(cfg.beta_kl))
                metrics.append({"step": float(step), **{f"train_{k}": v for k, v in row.items()}, **{f"val_{k}": v for k, v in val_row.items()}})
                progress.set_postfix({"loss": f"{row['loss']:.4g}", "recon": f"{row['recon_mse']:.4g}", "val": f"{val_row['loss']:.4g}"})
            progress.update(1)
    finally:
        progress.close()
    vae.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": vae.detach().cpu().state_dict() if hasattr(vae, "detach") else {k: v.detach().cpu() for k, v in vae.state_dict().items()},
            "normalizer": normalizer.state_dict(),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "metrics": metrics,
        },
        output_path,
    )
    vae.to(device=device, dtype=dtype)
    return TrainedVAE(vae=vae, normalizer=normalizer, train_indices=train_indices, val_indices=val_indices, metrics=pd.DataFrame(metrics))


def encode_weights(vae: WeightVAE, normalizer: WeightNormalizer, weights: torch.Tensor) -> torch.Tensor:
    vae.eval()
    with torch.no_grad():
        mu, _logvar = vae.encode(normalizer.normalize(weights))
    return mu.detach()


def decode_weights(vae: WeightVAE, normalizer: WeightNormalizer, z: torch.Tensor) -> torch.Tensor:
    return normalizer.denormalize(vae.decode_norm(z))


def decoder_jacobians(
    vae: WeightVAE,
    normalizer: WeightNormalizer,
    flow: torch.nn.Module | None,
    samples: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    def decoder_from_coord(coord: torch.Tensor) -> torch.Tensor:
        if flow is None:
            z = coord
        else:
            z = flow.inverse(coord.unsqueeze(0))[0].squeeze(0)
        return decode_weights(vae, normalizer, z.unsqueeze(0)).squeeze(0)

    return exact_jacobians(decoder_from_coord, samples, create_graph=bool(create_graph))


def geometry_metrics_from_jacobians(jacobians: torch.Tensor, *, eps: float = 1e-12) -> dict[str, float]:
    metric, trace_g, trace_g2 = metric_tensors_from_jacobians(jacobians.detach())
    eig = torch.linalg.eigvalsh(metric.float()).detach()
    eig_min = eig.min(dim=-1).values
    eig_max = eig.max(dim=-1).values
    cond = eig_max / eig_min.clamp_min(float(eps))
    log_spread = torch.log(eig_max.clamp_min(float(eps))) - torch.log(eig_min.clamp_min(float(eps)))
    dim = int(jacobians.shape[-1])
    distortion = isometry_objective_from_jacobians(jacobians.detach(), dim=dim, eps=eps)
    return {
        "isometry_objective": float(distortion.detach().cpu().item()),
        "trace_g_median": float(trace_g.float().median().cpu().item()),
        "trace_g2_median": float(trace_g2.float().median().cpu().item()),
        "eig_min_median": float(eig_min.median().cpu().item()),
        "eig_max_median": float(eig_max.median().cpu().item()),
        "condition_median": float(cond.median().cpu().item()),
        "condition_p90": float(torch.quantile(cond, 0.90).cpu().item()),
        "log_eig_spread_median": float(log_spread.median().cpu().item()),
    }


def make_rq_flow(cfg: ExperimentConfig, dim: int, *, device: torch.device, dtype: torch.dtype) -> RQSplineFlow:
    flow = RQSplineFlow(
        RQSplineFlowConfig(
            dim=int(dim),
            num_layers=int(cfg.flow_num_layers),
            hidden_dim=int(cfg.flow_hidden_dim),
            network_depth=int(cfg.flow_network_depth),
            num_bins=int(cfg.flow_spline_bins),
            bound=float(cfg.flow_spline_bound),
        )
    )
    return flow.to(device=device, dtype=dtype)


def train_posthoc_flow(
    cfg: ExperimentConfig,
    *,
    vae: WeightVAE,
    normalizer: WeightNormalizer,
    z_train: torch.Tensor,
    output_path: Path,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    if output_path.is_file() and bool(cfg.cache_first) and not bool(cfg.force_rerun):
        payload = torch.load(output_path, map_location="cpu", weights_only=False)
        flow = make_rq_flow(cfg, int(z_train.shape[1]), device=z_train.device, dtype=z_train.dtype)
        flow.load_state_dict(payload["flow_state"])
        return flow, pd.DataFrame(payload["history"])

    flow = make_rq_flow(cfg, int(z_train.shape[1]), device=z_train.device, dtype=z_train.dtype)
    optimizer = torch.optim.Adam(flow.parameters(), lr=float(cfg.flow_lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed) + 3000)
    sample_count = int(z_train.shape[0])
    batch_size = min(max(1, int(cfg.flow_batch_size)), sample_count)
    history: list[dict[str, float]] = []
    progress = make_progress(cfg, total=int(cfg.flow_steps), desc="post-hoc NF")
    try:
        for step in range(1, int(cfg.flow_steps) + 1):
            idx_i = torch.randint(0, sample_count, (batch_size,), generator=generator, device="cpu").to(device=z_train.device)
            idx_j = torch.randint(0, sample_count, (batch_size,), generator=generator, device="cpu").to(device=z_train.device)
            z_i = z_train.index_select(0, idx_i)
            z_j = z_train.index_select(0, idx_j)
            u_i = flow(z_i)[0]
            u_j = flow(z_j)[0]
            alpha = (
                torch.rand(batch_size, 1, generator=generator, device="cpu", dtype=z_train.dtype).to(device=z_train.device)
                * (float(cfg.mixup_alpha_max) - float(cfg.mixup_alpha_min))
                + float(cfg.mixup_alpha_min)
            )
            u_mix = alpha * u_i + (1.0 - alpha) * u_j
            optimizer.zero_grad(set_to_none=True)
            jac = decoder_jacobians(vae, normalizer, flow, u_mix, create_graph=True)
            iso = isometry_objective_from_jacobians(jac, dim=int(z_train.shape[1]))
            z_mix = flow.inverse(u_mix)[0]
            z_norm = z_mix.square().mean()
            loss = iso + float(cfg.flow_eta) * z_norm
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite posthoc flow loss at step {step}: {float(loss.detach().cpu().item())}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(flow.parameters(), float(cfg.flow_grad_clip_norm))
            optimizer.step()
            if step == 1 or step % max(1, int(cfg.flow_log_every)) == 0 or step == int(cfg.flow_steps):
                history.append(
                    {
                        "step": float(step),
                        "loss": float(loss.detach().cpu().item()),
                        "isometry": float(iso.detach().cpu().item()),
                        "z_norm": float(z_norm.detach().cpu().item()),
                        "grad_norm": float(torch.as_tensor(grad).detach().cpu().item()),
                    }
                )
                progress.set_postfix(
                    {
                        "loss": f"{float(loss.detach().cpu().item()):.4g}",
                        "iso": f"{float(iso.detach().cpu().item()):.4g}",
                        "grad": f"{float(torch.as_tensor(grad).detach().cpu().item()):.3g}",
                    }
                )
            progress.update(1)
    finally:
        progress.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"flow_state": flow.state_dict(), "history": history}, output_path)
    return flow, pd.DataFrame(history)


def random_near_identity_flow(cfg: ExperimentConfig, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    return make_rq_flow(cfg, int(dim), device=device, dtype=dtype)


def finite_value(value: float, *, penalty: float) -> float:
    if math.isfinite(float(value)):
        return float(value)
    return float(penalty)
