from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset, TensorDataset

from big_vae.models.big_weight_vae_parts.loss_mixin import BigWeightVAELossMixin
from post_train_research.big_vae_latent_flattening.flow import RQSplineFlow, RQSplineFlowConfig
from post_train_research.loss_landscape_analysis.flow_preconditioning.probe_geometry import (
    isometry_objective_from_jacobians,
    metric_tensors_from_jacobians,
)

from .config import ExperimentConfig, torch_dtype
from .progress import make_progress


def load_torch_cache(path: Path) -> Any | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        broken_path = path.with_name(f"{path.name}.broken-{uuid.uuid4().hex[:8]}")
        try:
            path.rename(broken_path)
            print(f"[cache] ignoring corrupt torch cache: {path} -> {broken_path} ({exc})", flush=True)
        except OSError:
            print(f"[cache] ignoring corrupt torch cache: {path} ({exc})", flush=True)
        return None


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _stage_cache_key(stage: str, cfg: ExperimentConfig, keys: tuple[str, ...], *, extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"stage": stage, "extra": extra or {}}
    payload["config"] = {key: getattr(cfg, key) for key in keys}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _cache_payload_matches(payload: dict[str, Any], expected_key: str, path: Path) -> bool:
    actual_key = payload.get("cache_key")
    if actual_key == expected_key:
        print(f"[cache] hit: {path}", flush=True)
        return True
    print(f"[cache] miss: {path} cache_key={actual_key!r} expected={expected_key!r}", flush=True)
    return False


def weight_pool_cache_key(cfg: ExperimentConfig) -> str:
    return _stage_cache_key(
        "weight_pool",
        cfg,
        (
            "weight_distribution",
            "dataset_name",
            "data_root",
            "download",
            "train_subset",
            "test_subset",
            "cnn_batch_size",
            "celo_tasks",
            "celo_image_size",
            "celo_hidden_dim",
            "celo_tau_min",
            "celo_tau_max",
            "celo_adam_lrs",
            "weight_runs",
            "weight_train_steps",
            "weight_snapshot_every",
            "weight_lr",
            "seed",
            "dtype",
        ),
    )


def vae_training_cache_key(cfg: ExperimentConfig, weights: torch.Tensor, *, upstream_cache_key: str) -> str:
    return _stage_cache_key(
        "vae",
        cfg,
        (
            "vae_train_fraction",
            "latent_dim",
            "vae_hidden_dim",
            "vae_arch",
            "tiny_bigvae_patch_size",
            "tiny_bigvae_token_dim",
            "tiny_bigvae_pos_dim",
            "tiny_bigvae_resampler_latents",
            "tiny_bigvae_attention_heads",
            "vae_steps",
            "vae_batch_size",
            "vae_lr",
            "vae_loss_kind",
            "beta_kl",
            "bigvae_operator_probe_rows",
            "bigvae_patch_size",
            "bigvae_behavioral_coef",
            "bigvae_structural_coef",
            "bigvae_bias_coef",
            "bigvae_behavioral_lambda_operator",
            "bigvae_behavioral_lambda_dir",
            "bigvae_behavioral_lambda_scale",
            "bigvae_behavioral_gamma",
            "bigvae_behavioral_huber_delta",
            "bigvae_struct_gamma",
            "bigvae_struct_lambda_dir",
            "bigvae_struct_lambda_scale",
            "bigvae_struct_lambda_rec",
            "bigvae_struct_lambda_rel",
            "bigvae_struct_huber_delta",
            "vae_geometry_reg_coeff",
            "vae_geometry_reg_samples",
            "vae_geometry_reg_detach_latents",
            "seed",
            "dtype",
        ),
        extra={"weights_shape": list(weights.shape), "upstream_cache_key": upstream_cache_key},
    )


def flow_training_cache_key(cfg: ExperimentConfig, z_train: torch.Tensor, *, upstream_cache_key: str) -> str:
    return _stage_cache_key(
        "posthoc_flow",
        cfg,
        (
            "flow_steps",
            "flow_batch_size",
            "flow_lr",
            "flow_grad_clip_norm",
            "flow_num_layers",
            "flow_hidden_dim",
            "flow_network_depth",
            "flow_spline_bins",
            "flow_spline_bound",
            "flow_eta",
            "mixup_alpha_min",
            "mixup_alpha_max",
            "seed",
            "dtype",
        ),
        extra={"z_train_shape": list(z_train.shape), "upstream_cache_key": upstream_cache_key},
    )


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


class CeloMetaMLP(nn.Module):
    def __init__(self, *, image_shape: tuple[int, int, int] = (1, 8, 8), hidden_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.image_shape = tuple(int(v) for v in image_shape)
        input_dim = int(math.prod(self.image_shape))
        self.fc1 = nn.Linear(input_dim, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.reshape(x.shape[0], -1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


@dataclass(frozen=True, slots=True)
class FlatSpec:
    keys: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]
    model_kind: str = "tiny_cnn"
    image_shape: tuple[int, int, int] = (1, 28, 28)
    hidden_dim: int = 0
    num_classes: int = 10

    @property
    def dim(self) -> int:
        return int(sum(self.sizes))


@dataclass(frozen=True, slots=True)
class TaskTensorSet:
    task_name: str
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def _module_flat_spec(
    model: nn.Module,
    *,
    model_kind: str,
    image_shape: tuple[int, int, int],
    hidden_dim: int,
    num_classes: int,
) -> FlatSpec:
    keys: list[str] = []
    shapes: list[tuple[int, ...]] = []
    sizes: list[int] = []
    for key, value in model.state_dict().items():
        keys.append(str(key))
        shapes.append(tuple(int(v) for v in value.shape))
        sizes.append(int(value.numel()))
    return FlatSpec(
        tuple(keys),
        tuple(shapes),
        tuple(sizes),
        model_kind=str(model_kind),
        image_shape=tuple(int(v) for v in image_shape),
        hidden_dim=int(hidden_dim),
        num_classes=int(num_classes),
    )


def tiny_cnn_spec() -> FlatSpec:
    return _module_flat_spec(
        TinyCNN(),
        model_kind="tiny_cnn",
        image_shape=(1, 28, 28),
        hidden_dim=32,
        num_classes=10,
    )


def celo_meta_mlp_spec(cfg: ExperimentConfig) -> FlatSpec:
    image_size = int(cfg.celo_image_size)
    hidden_dim = int(cfg.celo_hidden_dim)
    return _module_flat_spec(
        CeloMetaMLP(image_shape=(1, image_size, image_size), hidden_dim=hidden_dim, num_classes=10),
        model_kind="celo_meta_mlp",
        image_shape=(1, image_size, image_size),
        hidden_dim=hidden_dim,
        num_classes=10,
    )


def spec_to_payload(spec: FlatSpec) -> dict[str, Any]:
    return {
        "keys": spec.keys,
        "shapes": spec.shapes,
        "sizes": spec.sizes,
        "model_kind": spec.model_kind,
        "image_shape": spec.image_shape,
        "hidden_dim": int(spec.hidden_dim),
        "num_classes": int(spec.num_classes),
    }


def spec_from_payload(payload: Mapping[str, Any]) -> FlatSpec:
    return FlatSpec(
        tuple(str(v) for v in payload["keys"]),
        tuple(tuple(int(x) for x in shape) for shape in payload["shapes"]),
        tuple(int(v) for v in payload["sizes"]),
        model_kind=str(payload.get("model_kind", "tiny_cnn")),
        image_shape=tuple(int(v) for v in payload.get("image_shape", (1, 28, 28))),
        hidden_dim=int(payload.get("hidden_dim", 32)),
        num_classes=int(payload.get("num_classes", 10)),
    )


def _model_from_spec(spec: FlatSpec) -> nn.Module:
    if spec.model_kind == "tiny_cnn":
        return TinyCNN()
    if spec.model_kind == "celo_meta_mlp":
        return CeloMetaMLP(
            image_shape=tuple(int(v) for v in spec.image_shape),
            hidden_dim=int(spec.hidden_dim),
            num_classes=int(spec.num_classes),
        )
    raise ValueError(f"unknown FlatSpec model_kind {spec.model_kind!r}")


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


def logits_from_flat(flat: torch.Tensor, images: torch.Tensor, spec: FlatSpec, *, tau: float = 1.0) -> torch.Tensor:
    model = _model_from_spec(spec).to(device=images.device, dtype=images.dtype)
    effective_flat = flat.to(device=images.device, dtype=images.dtype) * float(tau)
    state = flat_to_state_dict(effective_flat, spec)
    return functional_call(model, state, (images,))


def tiny_cnn_logits_from_flat(flat: torch.Tensor, images: torch.Tensor, spec: FlatSpec) -> torch.Tensor:
    return logits_from_flat(flat, images, spec, tau=1.0)


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


def _canonical_celo_task_name(name: str) -> str:
    value = str(name).strip().lower().replace("-", "_")
    aliases = {
        "fashionmnist": "fashion_mnist",
        "fashion_mnist": "fashion_mnist",
        "mnist": "mnist",
        "svhn": "svhn",
        "cifar10": "cifar10",
        "cifar_10": "cifar10",
    }
    if value not in aliases:
        raise ValueError(f"celo task must be one of mnist, fashion_mnist, svhn, cifar10; got {name!r}")
    return aliases[value]


def _load_celo_dataset(name: str, *, root: Path, train: bool, download: bool, transform):
    from torchvision import datasets

    task_name = _canonical_celo_task_name(name)
    if task_name == "mnist":
        return datasets.MNIST(root=str(root), train=bool(train), download=bool(download), transform=transform)
    if task_name == "fashion_mnist":
        return datasets.FashionMNIST(root=str(root), train=bool(train), download=bool(download), transform=transform)
    if task_name == "svhn":
        return datasets.SVHN(root=str(root), split="train" if bool(train) else "test", download=bool(download), transform=transform)
    if task_name == "cifar10":
        return datasets.CIFAR10(root=str(root), train=bool(train), download=bool(download), transform=transform)
    raise AssertionError(f"unhandled celo task {task_name!r}")


def _tensor_subset(dataset, *, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count = min(int(count), len(dataset))
    if sample_count <= 0:
        raise ValueError("train_subset and test_subset must select at least one sample")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=sample_count, shuffle=False)
    images, labels = next(iter(loader))
    return images, labels.long()


def load_celo_meta_task_tensors(cfg: ExperimentConfig) -> dict[str, TaskTensorSet]:
    from torchvision import transforms

    image_size = int(cfg.celo_image_size)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    root = Path(cfg.data_root).expanduser().resolve()
    task_sets: dict[str, TaskTensorSet] = {}
    for task_idx, raw_name in enumerate(tuple(cfg.celo_tasks)):
        task_name = _canonical_celo_task_name(raw_name)
        train_set = _load_celo_dataset(task_name, root=root, train=True, download=bool(cfg.download), transform=transform)
        test_set = _load_celo_dataset(task_name, root=root, train=False, download=bool(cfg.download), transform=transform)
        train_images, train_labels = _tensor_subset(train_set, count=int(cfg.train_subset), seed=int(cfg.seed) + 10_000 + task_idx)
        test_images, test_labels = _tensor_subset(test_set, count=int(cfg.test_subset), seed=int(cfg.seed) + 20_000 + task_idx)
        task_sets[task_name] = TaskTensorSet(
            task_name=task_name,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
    return task_sets


def task_tensor_set_from_vision_tensors(
    *,
    task_name: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
) -> TaskTensorSet:
    return TaskTensorSet(
        task_name=str(task_name),
        train_images=train_images,
        train_labels=train_labels.long(),
        test_images=test_images,
        test_labels=test_labels.long(),
    )


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
    tau: float = 1.0,
) -> tuple[float, float]:
    with torch.no_grad():
        logits = logits_from_flat(flat, images, spec, tau=float(tau))
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
    return float(loss.detach().cpu().item()), float(acc.detach().cpu().item())


def _weight_distribution_name(cfg: ExperimentConfig) -> str:
    value = str(cfg.weight_distribution).strip().lower()
    if value in {"tiny", "tiny_cnn", "cnn", "fashion_mnist_tiny_cnn"}:
        return "tiny_cnn"
    if value in {"celo", "celo_meta", "celo_meta_mlp", "paper_meta_mlp"}:
        return "celo_meta_mlp"
    raise ValueError("weight_distribution must be 'tiny_cnn' or 'celo_meta_mlp', got " f"{cfg.weight_distribution!r}")


def _move_task_tensor_set(task_set: TaskTensorSet, *, device: torch.device, dtype: torch.dtype) -> TaskTensorSet:
    return TaskTensorSet(
        task_name=str(task_set.task_name),
        train_images=task_set.train_images.to(device=device, dtype=dtype),
        train_labels=task_set.train_labels.to(device=device),
        test_images=task_set.test_images.to(device=device, dtype=dtype),
        test_labels=task_set.test_labels.to(device=device),
    )


def move_task_tensors(
    task_tensors: Mapping[str, TaskTensorSet],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, TaskTensorSet]:
    return {str(name): _move_task_tensor_set(task_set, device=device, dtype=dtype) for name, task_set in task_tensors.items()}


def _sample_batch(task_set: TaskTensorSet, *, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    sample_count = int(task_set.train_images.shape[0])
    if sample_count <= 0:
        raise ValueError(f"task {task_set.task_name!r} has no train samples")
    count = min(max(1, int(batch_size)), sample_count)
    indices = torch.randint(0, sample_count, (count,), generator=generator, device="cpu").to(device=task_set.train_images.device)
    return task_set.train_images.index_select(0, indices), task_set.train_labels.index_select(0, indices)


def _task_tensors_from_legacy_args(
    *,
    train_images: torch.Tensor | None,
    train_labels: torch.Tensor | None,
    test_images: torch.Tensor | None,
    test_labels: torch.Tensor | None,
) -> dict[str, TaskTensorSet]:
    if train_images is None or train_labels is None or test_images is None or test_labels is None:
        raise ValueError("tiny_cnn weight generation requires train_images/train_labels/test_images/test_labels or task_tensors")
    return {
        "tiny_cnn": task_tensor_set_from_vision_tensors(
            task_name="tiny_cnn",
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
    }


def generate_weight_pool(
    cfg: ExperimentConfig,
    *,
    train_images: torch.Tensor | None = None,
    train_labels: torch.Tensor | None = None,
    test_images: torch.Tensor | None = None,
    test_labels: torch.Tensor | None = None,
    task_tensors: Mapping[str, TaskTensorSet] | None = None,
    output_path: Path,
) -> tuple[torch.Tensor, pd.DataFrame, FlatSpec]:
    cache_key = weight_pool_cache_key(cfg)
    if output_path.is_file() and bool(cfg.cache_first) and not bool(cfg.force_rerun):
        payload = load_torch_cache(output_path)
        if payload is not None and _cache_payload_matches(payload, cache_key, output_path):
            spec = spec_from_payload(payload["spec"])
            return payload["weights"], pd.DataFrame(payload["records"]), spec

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg)
    distribution = _weight_distribution_name(cfg)
    if task_tensors is None:
        if distribution == "tiny_cnn":
            task_tensors = _task_tensors_from_legacy_args(
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
            )
        else:
            task_tensors = load_celo_meta_task_tensors(cfg)
    task_tensors_device = move_task_tensors(task_tensors, device=device, dtype=dtype)

    flats: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    snapshot_every = max(1, int(cfg.weight_snapshot_every))
    snapshot_steps = set(range(0, int(cfg.weight_train_steps) + 1, snapshot_every))
    snapshot_steps.add(int(cfg.weight_train_steps))
    expected_snapshots = int(cfg.weight_runs) * len(snapshot_steps)
    print(
        "[weight_pool] building "
        f"distribution={distribution} "
        f"runs={int(cfg.weight_runs)} train_steps={int(cfg.weight_train_steps)} "
        f"snapshot_every={snapshot_every} snapshots_per_run={len(snapshot_steps)} "
        f"expected_snapshots={expected_snapshots} output={output_path}",
        flush=True,
    )
    progress = make_progress(cfg, total=int(cfg.weight_runs) * (int(cfg.weight_train_steps) + 1), desc="weight pool")
    try:
        if distribution == "tiny_cnn":
            spec = tiny_cnn_spec()
            task_set = task_tensors_device.get("tiny_cnn") or next(iter(task_tensors_device.values()))
            train_loader, _test_loader = make_tensor_loaders(
                task_set.train_images,
                task_set.train_labels,
                task_set.test_images,
                task_set.test_labels,
                batch_size=int(cfg.cnn_batch_size),
            )
            for run_idx in range(int(cfg.weight_runs)):
                torch.manual_seed(int(cfg.seed) + 1000 + run_idx)
                model = TinyCNN().to(device=device, dtype=dtype)
                optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.weight_lr))
                loader_iter = iter(train_loader)
                for step in range(int(cfg.weight_train_steps) + 1):
                    if step in snapshot_steps:
                        flat = state_dict_to_flat(model.state_dict(), spec).detach().cpu()
                        train_loss, train_acc = evaluate_flat_model(
                            flat.to(device=device, dtype=dtype),
                            images=task_set.train_images,
                            labels=task_set.train_labels,
                            spec=spec,
                            tau=1.0,
                        )
                        test_loss, test_acc = evaluate_flat_model(
                            flat.to(device=device, dtype=dtype),
                            images=task_set.test_images,
                            labels=task_set.test_labels,
                            spec=spec,
                            tau=1.0,
                        )
                        flats.append(flat)
                        records.append(
                            {
                                "run": int(run_idx),
                                "step": int(step),
                                "task_name": str(task_set.task_name),
                                "tau": 1.0,
                                "optimizer": "adam",
                                "source_lr": float(cfg.weight_lr),
                                "weight_distribution": distribution,
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
        elif distribution == "celo_meta_mlp":
            spec = celo_meta_mlp_spec(cfg)
            task_names = tuple(_canonical_celo_task_name(name) for name in tuple(cfg.celo_tasks))
            missing = [name for name in task_names if name not in task_tensors_device]
            if missing:
                raise ValueError(f"missing task tensors for Celo tasks: {missing}")
            lr_grid = tuple(float(v) for v in tuple(cfg.celo_adam_lrs))
            if not lr_grid:
                raise ValueError("celo_adam_lrs must contain at least one LR")
            tau_min = float(cfg.celo_tau_min)
            tau_max = float(cfg.celo_tau_max)
            if tau_min <= 0.0 or tau_max <= 0.0 or tau_max < tau_min:
                raise ValueError(f"invalid tau range [{tau_min}, {tau_max}]")
            rng = random.Random(int(cfg.seed) + 40_000)
            batch_generator = torch.Generator(device="cpu")
            batch_generator.manual_seed(int(cfg.seed) + 41_000)
            log_tau_min = math.log(tau_min)
            log_tau_max = math.log(tau_max)
            print(
                "[weight_pool] celo_meta_mlp setup "
                f"tasks={task_names} image_shape={(1, int(cfg.celo_image_size), int(cfg.celo_image_size))} "
                f"hidden_dim={int(cfg.celo_hidden_dim)} batch_size={int(cfg.cnn_batch_size)} "
                f"adam_lrs={lr_grid} tau_log_uniform=[{tau_min:g}, {tau_max:g}]",
                flush=True,
            )
            for run_idx in range(int(cfg.weight_runs)):
                task_name = rng.choice(task_names)
                source_lr = float(rng.choice(lr_grid))
                tau = float(math.exp(rng.uniform(log_tau_min, log_tau_max)))
                task_set = task_tensors_device[task_name]
                torch.manual_seed(int(cfg.seed) + 50_000 + run_idx)
                model = CeloMetaMLP(
                    image_shape=tuple(int(v) for v in spec.image_shape),
                    hidden_dim=int(spec.hidden_dim),
                    num_classes=int(spec.num_classes),
                ).to(device=device, dtype=dtype)
                initial_flat = state_dict_to_flat(model.state_dict(), spec).to(device=device, dtype=dtype)
                theta = (initial_flat / float(tau)).detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([theta], lr=source_lr)
                for step in range(int(cfg.weight_train_steps) + 1):
                    if step in snapshot_steps:
                        flat = theta.detach().cpu()
                        train_loss, train_acc = evaluate_flat_model(
                            flat.to(device=device, dtype=dtype),
                            images=task_set.train_images,
                            labels=task_set.train_labels,
                            spec=spec,
                            tau=tau,
                        )
                        test_loss, test_acc = evaluate_flat_model(
                            flat.to(device=device, dtype=dtype),
                            images=task_set.test_images,
                            labels=task_set.test_labels,
                            spec=spec,
                            tau=tau,
                        )
                        flats.append(flat)
                        records.append(
                            {
                                "run": int(run_idx),
                                "step": int(step),
                                "task_name": task_name,
                                "tau": tau,
                                "optimizer": "adam",
                                "source_lr": source_lr,
                                "weight_distribution": distribution,
                                "train_loss": train_loss,
                                "train_acc": train_acc,
                                "test_loss": test_loss,
                                "test_acc": test_acc,
                            }
                        )
                        progress.set_postfix(
                            {
                                "run": run_idx,
                                "task": task_name,
                                "step": step,
                                "lr": f"{source_lr:.1e}",
                                "tau": f"{tau:.1e}",
                                "test_acc": f"{test_acc:.3f}",
                            }
                        )
                    progress.update(1)
                    if step == int(cfg.weight_train_steps):
                        break
                    batch_images, batch_labels = _sample_batch(
                        task_set,
                        batch_size=int(cfg.cnn_batch_size),
                        generator=batch_generator,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    logits = logits_from_flat(theta, batch_images, spec, tau=tau)
                    loss = F.cross_entropy(logits, batch_labels)
                    if not bool(torch.isfinite(loss).detach().cpu().item()):
                        progress.set_postfix(
                            {
                                "run": run_idx,
                                "task": task_name,
                                "step": step,
                                "lr": f"{source_lr:.1e}",
                                "tau": f"{tau:.1e}",
                                "status": "nonfinite_loss",
                            }
                        )
                        break
                    loss.backward()
                    if theta.grad is None or not bool(torch.isfinite(theta.grad).all().detach().cpu().item()):
                        progress.set_postfix(
                            {
                                "run": run_idx,
                                "task": task_name,
                                "step": step,
                                "lr": f"{source_lr:.1e}",
                                "tau": f"{tau:.1e}",
                                "status": "nonfinite_grad",
                            }
                        )
                        break
                    optimizer.step()
                    if not bool(torch.isfinite(theta).all().detach().cpu().item()):
                        progress.set_postfix(
                            {
                                "run": run_idx,
                                "task": task_name,
                                "step": step,
                                "lr": f"{source_lr:.1e}",
                                "tau": f"{tau:.1e}",
                                "status": "nonfinite_theta",
                            }
                        )
                        break
        else:
            raise AssertionError(f"unhandled weight distribution {distribution!r}")
    finally:
        progress.close()

    weights = torch.stack(flats, dim=0)
    print(
        f"[weight_pool] built snapshots={len(flats)} weights_shape={tuple(weights.shape)} records={len(records)} output={output_path}",
        flush=True,
    )
    atomic_torch_save(
        {
            "cache_key": cache_key,
            "weights": weights,
            "records": records,
            "spec": spec_to_payload(spec),
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


class TinyBigWeightVAE(nn.Module):
    """Very small BigVAE-like weight VAE.

    It keeps the pipeline-level VAE API while replacing one dense
    weight-vector encoder with shared patch token processing.
    """

    def __init__(
        self,
        *,
        weight_dim: int,
        latent_dim: int,
        hidden_dim: int,
        patch_size: int,
        token_dim: int,
        pos_dim: int,
        resampler_latents: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        self.weight_dim = int(weight_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = max(1, int(patch_size))
        self.token_dim = max(4, int(token_dim))
        self.pos_dim = max(2, int(pos_dim))
        self.resampler_latents = max(1, int(resampler_latents))
        self.attention_heads = max(1, int(attention_heads))
        if int(self.token_dim) % int(self.attention_heads) != 0:
            raise ValueError(
                f"tiny_bigvae_token_dim ({self.token_dim}) must be divisible by "
                f"tiny_bigvae_attention_heads ({self.attention_heads})"
            )
        self.num_patches = int(math.ceil(float(self.weight_dim) / float(self.patch_size)))
        self.padded_dim = int(self.num_patches * self.patch_size)

        encoder_hidden = max(int(self.hidden_dim), int(self.token_dim))
        self.patch_encoder = nn.Sequential(
            nn.Linear(int(self.patch_size + self.pos_dim), encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, int(self.token_dim)),
            nn.GELU(),
        )
        self.resampler_slots = nn.Parameter(torch.randn(int(self.resampler_latents), int(self.token_dim)) * 0.02)
        self.resampler_q_norm = nn.LayerNorm(int(self.token_dim))
        self.resampler_kv_norm = nn.LayerNorm(int(self.token_dim))
        self.resampler_attn = nn.MultiheadAttention(
            embed_dim=int(self.token_dim),
            num_heads=int(self.attention_heads),
            batch_first=True,
        )
        self.resampler_ffn = nn.Sequential(
            nn.LayerNorm(int(self.token_dim)),
            nn.Linear(int(self.token_dim), encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, int(self.token_dim)),
        )
        self.head_norm = nn.LayerNorm(int(self.token_dim))
        self.head_attn = nn.MultiheadAttention(
            embed_dim=int(self.token_dim),
            num_heads=int(self.attention_heads),
            batch_first=True,
        )
        self.head_ffn = nn.Sequential(
            nn.LayerNorm(int(self.token_dim)),
            nn.Linear(int(self.token_dim), encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, int(self.token_dim)),
        )
        self.token_norm = nn.LayerNorm(int(self.token_dim))
        self.to_mu = nn.Linear(int(self.token_dim), int(self.latent_dim))
        self.to_logvar = nn.Linear(int(self.token_dim), int(self.latent_dim))

        self.latent_to_context = nn.Sequential(
            nn.Linear(int(self.latent_dim), encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, int(self.token_dim)),
        )
        self.decoder_pos_proj = nn.Linear(int(self.pos_dim), int(self.token_dim))
        self.decoder_attn_norm = nn.LayerNorm(int(self.token_dim))
        self.decoder_qkv = nn.Linear(int(self.token_dim), int(3 * self.token_dim))
        self.decoder_out_proj = nn.Linear(int(self.token_dim), int(self.token_dim))
        self.decoder_ffn = nn.Sequential(
            nn.LayerNorm(int(self.token_dim)),
            nn.Linear(int(self.token_dim), encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, int(self.token_dim)),
        )
        self.decoder_scale_head = nn.Sequential(
            nn.LayerNorm(int(self.token_dim)),
            nn.Linear(int(self.token_dim), encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, 1),
        )
        self.output_eps = 1e-6
        self.output_s_min = -3.0
        self.output_s_max = 6.0
        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(int(self.token_dim)),
            nn.Linear(int(self.token_dim), encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, int(self.patch_size)),
        )

    def _patch_positions(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tau = (torch.arange(int(self.num_patches), device=device, dtype=torch.float32) + 0.5) / float(self.num_patches)
        features = [tau]
        half = max(1, int((self.pos_dim - 1) // 2))
        for idx in range(half):
            freq = float(2**idx) * math.pi
            features.append(torch.sin(freq * tau))
            if len(features) >= int(self.pos_dim):
                break
            features.append(torch.cos(freq * tau))
            if len(features) >= int(self.pos_dim):
                break
        while len(features) < int(self.pos_dim):
            features.append(tau.new_zeros(tau.shape))
        return torch.stack(features[: int(self.pos_dim)], dim=-1).to(dtype=dtype)

    def _patchify(self, x_norm: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim != 2:
            raise ValueError(f"x_norm must be [B,D], got {tuple(x_norm.shape)}")
        if int(x_norm.shape[1]) != int(self.weight_dim):
            raise ValueError(f"x_norm dim must be {self.weight_dim}, got {int(x_norm.shape[1])}")
        pad = int(self.padded_dim - self.weight_dim)
        if pad > 0:
            x_norm = F.pad(x_norm, (0, pad))
        return x_norm.reshape(int(x_norm.shape[0]), int(self.num_patches), int(self.patch_size))

    def encode(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self._patchify(x_norm)
        pos = self._patch_positions(device=x_norm.device, dtype=x_norm.dtype).unsqueeze(0).expand(int(x_norm.shape[0]), -1, -1)
        tokens = self.patch_encoder(torch.cat([patches, pos], dim=-1))
        slots = self.resampler_slots.to(device=x_norm.device, dtype=x_norm.dtype).unsqueeze(0).expand(int(x_norm.shape[0]), -1, -1)
        tokens_norm = self.resampler_kv_norm(tokens)
        slot_update, _ = self.resampler_attn(
            self.resampler_q_norm(slots),
            tokens_norm,
            tokens_norm,
            need_weights=False,
        )
        slots = slots + slot_update
        slots = slots + self.resampler_ffn(slots)
        head_tokens = self.head_norm(slots)
        head_update, _ = self.head_attn(head_tokens, head_tokens, head_tokens, need_weights=False)
        slots = slots + head_update
        slots = slots + self.head_ffn(slots)
        pooled = self.token_norm(slots.mean(dim=1))
        return self.to_mu(pooled), self.to_logvar(pooled).clamp(min=-12.0, max=8.0)

    def decode_norm(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be [B,Z], got {tuple(z.shape)}")
        if int(z.shape[1]) != int(self.latent_dim):
            raise ValueError(f"z dim must be {self.latent_dim}, got {int(z.shape[1])}")
        context = self.latent_to_context(z).unsqueeze(1)
        pos = self._patch_positions(device=z.device, dtype=z.dtype)
        pos_tokens = self.decoder_pos_proj(pos).unsqueeze(0)
        tokens = context + pos_tokens
        tokens_norm = self.decoder_attn_norm(tokens)
        qkv = self.decoder_qkv(tokens_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        batch_size, token_count, _ = q.shape
        head_dim = int(self.token_dim // self.attention_heads)
        q = q.reshape(batch_size, token_count, int(self.attention_heads), head_dim).transpose(1, 2)
        k = k.reshape(batch_size, token_count, int(self.attention_heads), head_dim).transpose(1, 2)
        v = v.reshape(batch_size, token_count, int(self.attention_heads), head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(head_dim)), dim=-1)
        update = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, token_count, int(self.token_dim))
        tokens = tokens + self.decoder_out_proj(update)
        tokens = tokens + self.decoder_ffn(tokens)
        direction_patches = self.patch_decoder(tokens)
        valid_mask = direction_patches.new_ones(int(self.padded_dim))
        pad = int(self.padded_dim - self.weight_dim)
        if pad > 0:
            valid_mask[-pad:] = 0.0
        valid_mask = valid_mask.reshape(1, int(self.num_patches), int(self.patch_size))
        raw_direction = direction_patches * valid_mask
        direction_norm = torch.linalg.vector_norm(raw_direction, ord=2, dim=-1, keepdim=True).clamp_min(float(self.output_eps))
        direction = raw_direction / direction_norm
        scale_logits = self.decoder_scale_head(tokens)
        log_scale = float(self.output_s_min) + F.softplus(scale_logits - float(self.output_s_min))
        log_scale = float(self.output_s_max) - F.softplus(float(self.output_s_max) - log_scale)
        scale = torch.exp(log_scale)
        patches = direction * scale * valid_mask
        flat = patches.reshape(int(z.shape[0]), int(self.padded_dim))
        return flat[:, : int(self.weight_dim)]

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_norm)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        return self.decode_norm(z), mu, logvar


def _normalize_vae_arch(value: str) -> str:
    arch = str(value).strip().lower()
    if arch in {"weight_mlp", "mlp", "flat_mlp", "weight_vae"}:
        return "weight_mlp"
    if arch in {"tiny_big_vae", "tiny_bigvae", "mini_big_vae", "mini_bigvae"}:
        return "tiny_big_vae"
    raise ValueError(f"vae_arch must be 'weight_mlp' or 'tiny_big_vae', got {value!r}")


def build_weight_vae(cfg: ExperimentConfig, *, weight_dim: int) -> nn.Module:
    arch = _normalize_vae_arch(cfg.vae_arch)
    if arch == "weight_mlp":
        return WeightVAE(weight_dim=int(weight_dim), latent_dim=int(cfg.latent_dim), hidden_dim=int(cfg.vae_hidden_dim))
    if arch == "tiny_big_vae":
        return TinyBigWeightVAE(
            weight_dim=int(weight_dim),
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.vae_hidden_dim),
            patch_size=int(cfg.tiny_bigvae_patch_size),
            token_dim=int(cfg.tiny_bigvae_token_dim),
            pos_dim=int(cfg.tiny_bigvae_pos_dim),
            resampler_latents=int(cfg.tiny_bigvae_resampler_latents),
            attention_heads=int(cfg.tiny_bigvae_attention_heads),
        )
    raise AssertionError(f"unhandled vae_arch {arch!r}")


def _matrix_and_bias_blocks(flat_batch: torch.Tensor, spec: FlatSpec) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    matrix_blocks: list[tuple[str, torch.Tensor]] = []
    bias_blocks: list[tuple[str, torch.Tensor]] = []
    offset = 0
    batch_size = int(flat_batch.shape[0])
    for key, shape, size in zip(spec.keys, spec.shapes, spec.sizes, strict=True):
        value = flat_batch[:, offset : offset + int(size)].reshape((batch_size, *shape))
        offset += int(size)
        if str(key).endswith(".weight") and len(shape) == 2:
            matrix_blocks.append((key, value.transpose(1, 2).contiguous()))
        elif str(key).endswith(".weight") and len(shape) == 4:
            matrix_blocks.append((key, value.reshape(batch_size, int(shape[0]), -1).transpose(1, 2).contiguous()))
        elif str(key).endswith(".bias") or len(shape) == 1:
            bias_blocks.append((key, value.reshape(batch_size, -1)))
        else:
            bias_blocks.append((key, value.reshape(batch_size, -1)))
    return matrix_blocks, bias_blocks


def _normalized_bias_mse(
    target_bias_blocks: list[tuple[str, torch.Tensor]],
    recon_bias_blocks: list[tuple[str, torch.Tensor]],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for (target_key, target), (recon_key, recon) in zip(target_bias_blocks, recon_bias_blocks, strict=True):
        if target_key != recon_key:
            raise ValueError(f"bias block key mismatch: {target_key!r} vs {recon_key!r}")
        denom = target.detach().pow(2).mean(dim=1).clamp_min(float(eps))
        terms.append((recon - target).pow(2).mean(dim=1) / denom)
    if not terms:
        return target_bias_blocks[0][1].new_zeros(()) if target_bias_blocks else torch.tensor(0.0)
    return torch.stack([term.mean() for term in terms]).mean()


def bigvae_style_weight_loss(
    cfg: ExperimentConfig,
    *,
    target_weights: torch.Tensor,
    recon_weights: torch.Tensor,
    spec: FlatSpec,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_matrices, target_biases = _matrix_and_bias_blocks(target_weights, spec)
    recon_matrices, recon_biases = _matrix_and_bias_blocks(recon_weights, spec)
    behavioral_terms: list[torch.Tensor] = []
    operator_terms: list[torch.Tensor] = []
    behavioral_dir_terms: list[torch.Tensor] = []
    behavioral_scale_terms: list[torch.Tensor] = []
    structural_terms: list[torch.Tensor] = []
    struct_dir_terms: list[torch.Tensor] = []
    struct_scale_terms: list[torch.Tensor] = []
    struct_rec_terms: list[torch.Tensor] = []
    struct_rel_terms: list[torch.Tensor] = []

    for (target_key, W), (recon_key, W_hat) in zip(target_matrices, recon_matrices, strict=True):
        if target_key != recon_key:
            raise ValueError(f"matrix block key mismatch: {target_key!r} vs {recon_key!r}")
        rows = max(1, int(cfg.bigvae_operator_probe_rows))
        X = torch.randn((int(W.shape[0]), rows, int(W.shape[1])), device=W.device, dtype=W.dtype)
        operator = BigWeightVAELossMixin.operator_recon_loss(X, W, W_hat)
        if float(cfg.bigvae_behavioral_lambda_dir) != 0.0 or float(cfg.bigvae_behavioral_lambda_scale) != 0.0:
            behavioral_dir, behavioral_scale = BigWeightVAELossMixin.operator_direction_scale_loss(
                X,
                W,
                W_hat,
                gamma=float(cfg.bigvae_behavioral_gamma),
                huber_delta=float(cfg.bigvae_behavioral_huber_delta),
            )
        else:
            behavioral_dir = operator.new_zeros(())
            behavioral_scale = operator.new_zeros(())
        behavioral = (
            float(cfg.bigvae_behavioral_lambda_operator) * operator
            + float(cfg.bigvae_behavioral_lambda_dir) * behavioral_dir
            + float(cfg.bigvae_behavioral_lambda_scale) * behavioral_scale
        )
        structural, struct_details = BigWeightVAELossMixin.patch_structure_loss(
            W,
            W_hat,
            patch_size=int(cfg.bigvae_patch_size),
            gamma=float(cfg.bigvae_struct_gamma),
            lambda_dir=float(cfg.bigvae_struct_lambda_dir),
            lambda_scale=float(cfg.bigvae_struct_lambda_scale),
            lambda_rec=float(cfg.bigvae_struct_lambda_rec),
            lambda_rel=float(cfg.bigvae_struct_lambda_rel),
            huber_delta=float(cfg.bigvae_struct_huber_delta),
        )
        behavioral_terms.append(behavioral)
        operator_terms.append(operator)
        behavioral_dir_terms.append(behavioral_dir)
        behavioral_scale_terms.append(behavioral_scale)
        structural_terms.append(structural)
        struct_dir_terms.append(struct_details["L_dir"])
        struct_scale_terms.append(struct_details["L_scale"])
        struct_rec_terms.append(struct_details["L_rec"])
        struct_rel_terms.append(struct_details["L_rel"])

    zero = recon_weights.new_zeros(())
    behavioral_loss = torch.stack(behavioral_terms).mean() if behavioral_terms else zero
    operator_loss = torch.stack(operator_terms).mean() if operator_terms else zero
    behavioral_dir_loss = torch.stack(behavioral_dir_terms).mean() if behavioral_dir_terms else zero
    behavioral_scale_loss = torch.stack(behavioral_scale_terms).mean() if behavioral_scale_terms else zero
    structural_loss = torch.stack(structural_terms).mean() if structural_terms else zero
    struct_dir_loss = torch.stack(struct_dir_terms).mean() if struct_dir_terms else zero
    struct_scale_loss = torch.stack(struct_scale_terms).mean() if struct_scale_terms else zero
    struct_rec_loss = torch.stack(struct_rec_terms).mean() if struct_rec_terms else zero
    struct_rel_loss = torch.stack(struct_rel_terms).mean() if struct_rel_terms else zero
    bias_loss = _normalized_bias_mse(target_biases, recon_biases) if target_biases else zero
    loss = (
        float(cfg.bigvae_behavioral_coef) * behavioral_loss
        + float(cfg.bigvae_structural_coef) * structural_loss
        + float(cfg.bigvae_bias_coef) * bias_loss
    )
    return loss, {
        "bigvae_weight_loss": loss.detach(),
        "behavioral": behavioral_loss.detach(),
        "behavioral_operator": operator_loss.detach(),
        "behavioral_dir": behavioral_dir_loss.detach(),
        "behavioral_scale": behavioral_scale_loss.detach(),
        "structural": structural_loss.detach(),
        "struct_L_dir": struct_dir_loss.detach(),
        "struct_L_scale": struct_scale_loss.detach(),
        "struct_L_rec": struct_rec_loss.detach(),
        "struct_L_rel": struct_rel_loss.detach(),
        "bias_mse": bias_loss.detach(),
    }


def vae_loss(
    cfg: ExperimentConfig,
    x_norm: torch.Tensor,
    target_weights: torch.Tensor,
    recon_norm: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    normalizer: WeightNormalizer,
    spec: FlatSpec,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon = F.mse_loss(recon_norm, x_norm)
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
    loss_kind = str(cfg.vae_loss_kind).strip().lower()
    if loss_kind in {"mse", "normalized_mse"}:
        primary = recon
        details: dict[str, torch.Tensor] = {
            "bigvae_weight_loss": recon.detach(),
            "behavioral": recon.new_zeros(()),
            "behavioral_operator": recon.new_zeros(()),
            "behavioral_dir": recon.new_zeros(()),
            "behavioral_scale": recon.new_zeros(()),
            "structural": recon.new_zeros(()),
            "struct_L_dir": recon.new_zeros(()),
            "struct_L_scale": recon.new_zeros(()),
            "struct_L_rec": recon.new_zeros(()),
            "struct_L_rel": recon.new_zeros(()),
            "bias_mse": recon.new_zeros(()),
        }
    elif loss_kind in {"big_vae", "bigvae"}:
        recon_weights = normalizer.denormalize(recon_norm)
        primary, details = bigvae_style_weight_loss(cfg, target_weights=target_weights, recon_weights=recon_weights, spec=spec)
    else:
        raise ValueError(f"vae_loss_kind must be 'big_vae' or 'mse', got {cfg.vae_loss_kind!r}")
    loss = primary + float(cfg.beta_kl) * kl
    row = {
        "loss": float(loss.detach().cpu().item()),
        "recon_mse": float(recon.detach().cpu().item()),
        "kl": float(kl.detach().cpu().item()),
        "geometry_reg": 0.0,
        "geometry_reg_coeff": float(cfg.vae_geometry_reg_coeff),
    }
    row.update({key: float(value.detach().cpu().item()) for key, value in details.items()})
    return loss, row


def vae_decoder_geometry_regularizer(
    vae: nn.Module,
    normalizer: WeightNormalizer,
    z_samples: torch.Tensor,
) -> torch.Tensor:
    jac = decoder_jacobians(vae, normalizer, None, z_samples, create_graph=True)
    return isometry_objective_from_jacobians(jac, dim=int(z_samples.shape[1]))


@dataclass(slots=True)
class TrainedVAE:
    vae: nn.Module
    normalizer: WeightNormalizer
    train_indices: torch.Tensor
    val_indices: torch.Tensor
    metrics: pd.DataFrame


def train_weight_vae(
    cfg: ExperimentConfig,
    weights: torch.Tensor,
    *,
    output_path: Path,
    upstream_cache_key: str = "",
    spec: FlatSpec | None = None,
) -> TrainedVAE:
    cache_key = vae_training_cache_key(cfg, weights, upstream_cache_key=upstream_cache_key)
    if output_path.is_file() and bool(cfg.cache_first) and not bool(cfg.force_rerun):
        payload = load_torch_cache(output_path)
        if payload is not None and _cache_payload_matches(payload, cache_key, output_path):
            normalizer = WeightNormalizer.from_state_dict(payload["normalizer"])
            vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1]))
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
    if spec is None:
        spec = tiny_cnn_spec()
        if int(spec.dim) != int(weights.shape[1]):
            raise ValueError("train_weight_vae requires spec when weights are not TinyCNN-shaped")
    torch.manual_seed(int(cfg.seed) + 2100)
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(vae.parameters(), lr=float(cfg.vae_lr))
    metrics: list[dict[str, float]] = []
    train_indices_device = train_indices.to(device=device)
    batch_size = min(max(1, int(cfg.vae_batch_size)), int(train_indices.numel()))
    progress = make_progress(cfg, total=int(cfg.vae_steps), desc="weight VAE")
    try:
        for step in range(1, int(cfg.vae_steps) + 1):
            batch_pos = torch.randint(0, int(train_indices.numel()), (batch_size,), generator=generator, device="cpu").to(device=device)
            batch_indices = train_indices_device.index_select(0, batch_pos)
            batch = x_norm.index_select(0, batch_indices)
            batch_raw = weights_device.index_select(0, batch_indices)
            vae.train()
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = vae(batch)
            loss, row = vae_loss(
                cfg,
                batch,
                batch_raw,
                recon,
                mu,
                logvar,
                normalizer=normalizer,
                spec=spec,
            )
            if float(cfg.vae_geometry_reg_coeff) > 0.0:
                reg_count = min(max(1, int(cfg.vae_geometry_reg_samples)), int(mu.shape[0]))
                z_reg = mu[:reg_count]
                if bool(cfg.vae_geometry_reg_detach_latents):
                    z_reg = z_reg.detach()
                geometry_reg = vae_decoder_geometry_regularizer(vae, normalizer, z_reg)
                loss = loss + float(cfg.vae_geometry_reg_coeff) * geometry_reg
                row["loss"] = float(loss.detach().cpu().item())
                row["geometry_reg"] = float(geometry_reg.detach().cpu().item())
                row["geometry_reg_coeff"] = float(cfg.vae_geometry_reg_coeff)
            loss.backward()
            optimizer.step()
            if step == 1 or step % max(1, int(cfg.vae_steps) // 20) == 0 or step == int(cfg.vae_steps):
                vae.eval()
                with torch.no_grad():
                    val_indices_device = val_indices.to(device=device)
                    val = x_norm.index_select(0, val_indices_device)
                    val_raw = weights_device.index_select(0, val_indices_device)
                    recon_val, mu_val, logvar_val = vae(val)
                    _loss_val, val_row = vae_loss(
                        cfg,
                        val,
                        val_raw,
                        recon_val,
                        mu_val,
                        logvar_val,
                        normalizer=normalizer,
                        spec=spec,
                    )
                metrics.append({"step": float(step), **{f"train_{k}": v for k, v in row.items()}, **{f"val_{k}": v for k, v in val_row.items()}})
                progress.set_postfix(
                    {
                        "loss": f"{row['loss']:.4g}",
                        "bigvae": f"{row['bigvae_weight_loss']:.4g}",
                        "geom": f"{row['geometry_reg']:.4g}",
                        "mse": f"{row['recon_mse']:.4g}",
                        "val": f"{val_row['loss']:.4g}",
                    }
                )
            progress.update(1)
    finally:
        progress.close()
    vae.eval()
    atomic_torch_save(
        {
            "cache_key": cache_key,
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


def encode_weights(vae: nn.Module, normalizer: WeightNormalizer, weights: torch.Tensor) -> torch.Tensor:
    vae.eval()
    with torch.no_grad():
        mu, _logvar = vae.encode(normalizer.normalize(weights))
    return mu.detach()


def decode_weights(vae: nn.Module, normalizer: WeightNormalizer, z: torch.Tensor) -> torch.Tensor:
    return normalizer.denormalize(vae.decode_norm(z))


def _forward_mode_jacobians(func, inputs: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be [B,D], got {tuple(inputs.shape)}")
    try:
        from torch.func import jacfwd, vmap

        return vmap(jacfwd(func))(inputs)
    except (ImportError, AttributeError) as exc:
        rows: list[torch.Tensor] = []
        for x in inputs:
            x_req = x.detach().clone().requires_grad_(bool(create_graph))
            try:
                jac = torch.autograd.functional.jacobian(
                    func,
                    x_req,
                    create_graph=bool(create_graph),
                    strict=False,
                    vectorize=True,
                    strategy="forward-mode",
                )
            except TypeError as type_exc:
                raise RuntimeError("decoder geometry requires torch.func.jacfwd or forward-mode autograd") from type_exc
            rows.append(jac)
        return torch.stack(rows, dim=0)
    except RuntimeError as exc:
        raise RuntimeError(
            "forward-mode decoder Jacobian failed. This stage intentionally avoids reverse-mode jacrev "
            "because the decoder output dimension is the full weight vector and reverse-mode can OOM."
        ) from exc


def decoder_jacobians(
    vae: nn.Module,
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

    return _forward_mode_jacobians(decoder_from_coord, samples, create_graph=bool(create_graph))


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
    vae: nn.Module,
    normalizer: WeightNormalizer,
    z_train: torch.Tensor,
    output_path: Path,
    upstream_cache_key: str = "",
) -> tuple[torch.nn.Module, pd.DataFrame]:
    cache_key = flow_training_cache_key(cfg, z_train, upstream_cache_key=upstream_cache_key)
    if output_path.is_file() and bool(cfg.cache_first) and not bool(cfg.force_rerun):
        payload = load_torch_cache(output_path)
        if payload is not None and _cache_payload_matches(payload, cache_key, output_path):
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
    atomic_torch_save({"cache_key": cache_key, "flow_state": flow.state_dict(), "history": history}, output_path)
    return flow, pd.DataFrame(history)


def random_near_identity_flow(cfg: ExperimentConfig, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    return make_rq_flow(cfg, int(dim), device=device, dtype=dtype)


def finite_value(value: float, *, penalty: float) -> float:
    if math.isfinite(float(value)):
        return float(value)
    return float(penalty)
