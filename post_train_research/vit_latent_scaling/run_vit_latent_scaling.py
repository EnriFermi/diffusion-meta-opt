from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compare_vit_tiny_latent_optimization import (
    BigVAELatentTensorStore,
    EvalMetrics,
    FunctionalViTTiny,
    ViTTinyConfig,
    autocast_context,
    count_trainable_parameters,
    load_frozen_big_vae_decoder,
    make_initial_tensors,
    resolve_device,
    seed_everything,
)
from models.weight_quantile_vae import BigWeightVAE
from training.big_vae_latent_diffusion import (
    load_distribution_encoder_state_from_latent_diffusion_prior_checkpoint,
    load_frozen_layer_latent_diffusion_prior,
)


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class ScalingRunConfig:
    output_dir: str
    dataset: str
    model_size: str
    setup: str
    data_dir: str
    download: bool
    device: str
    seed: int
    epochs: int
    max_steps: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    train_subset: int
    test_subset: int
    lr: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    label_smoothing: float
    grad_clip_norm: float
    amp: bool
    tf32: bool
    compile: bool
    log_every_steps: int
    eval_every_steps: int
    save_checkpoints: bool
    raw_checkpoint: str
    big_vae_checkpoint: str
    big_vae_latent_init: str
    big_vae_diffusion_prior_checkpoint: str
    big_vae_diffusion_prior_steps: int
    big_vae_diffusion_prior_sampler: str
    big_vae_diffusion_prior_eta: float
    big_vae_decode: str
    big_vae_tile_T_patches: int
    big_vae_tile_d_out: int
    big_vae_encoder_context_rows: int
    big_vae_encoder_context_std: float
    big_vae_encoder_batch_size: int
    latent_weight_decay: float
    raw_init_steps: int


def validate_scaling_run_config(cfg: ScalingRunConfig) -> None:
    setup = str(cfg.setup).strip().lower()
    if setup == "latent":
        if not str(cfg.big_vae_checkpoint).strip():
            raise ValueError("--big-vae-checkpoint is required for setup=latent")
        latent_init = str(cfg.big_vae_latent_init).strip().lower()
        if latent_init == "encoded" and not str(cfg.raw_checkpoint).strip():
            raise ValueError("--raw-checkpoint is required when --big-vae-latent-init=encoded")
        if latent_init == "diffusion_prior" and not str(
            cfg.big_vae_diffusion_prior_checkpoint
        ).strip():
            raise ValueError(
                "--big-vae-diffusion-prior-checkpoint is required when --big-vae-latent-init=diffusion_prior"
            )
    if str(cfg.dataset).strip().lower() == "imagenet" and bool(cfg.download):
        cfg.download = False


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> EvalMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss.detach().cpu().item())
        correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu().item())
        examples += int(labels.numel())
    return EvalMetrics(
        loss=loss_sum / max(1, examples),
        accuracy=correct / max(1, examples),
        examples=examples,
    )


def build_optimizer(
    parameters: list[torch.nn.Parameter],
    *,
    cfg: ScalingRunConfig,
    setup: str,
) -> torch.optim.Optimizer:
    weight_decay = float(cfg.latent_weight_decay if setup == "latent" else cfg.weight_decay)
    return torch.optim.AdamW(
        parameters,
        lr=float(cfg.lr),
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=float(cfg.adam_eps),
        weight_decay=weight_decay,
    )


def maybe_subset(dataset: Any, limit: int) -> Any:
    if int(limit) <= 0:
        return dataset
    return Subset(dataset, list(range(min(int(limit), len(dataset)))))


def build_mnist_loaders(cfg: ScalingRunConfig, device: torch.device, vit_cfg: ViTTinyConfig) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for MNIST") from exc

    train_steps = []
    eval_steps = []
    if int(vit_cfg.image_size) != 28:
        train_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
        eval_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    eval_steps.extend([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])

    train_set = datasets.MNIST(
        root=str(cfg.data_dir),
        train=True,
        download=bool(cfg.download),
        transform=transforms.Compose(train_steps),
    )
    test_set = datasets.MNIST(
        root=str(cfg.data_dir),
        train=False,
        download=bool(cfg.download),
        transform=transforms.Compose(eval_steps),
    )
    return make_loaders(cfg, device, maybe_subset(train_set, cfg.train_subset), maybe_subset(test_set, cfg.test_subset))


def build_cifar10_loaders(
    cfg: ScalingRunConfig,
    device: torch.device,
    vit_cfg: ViTTinyConfig,
) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for CIFAR-10") from exc

    train_steps = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    eval_steps = []
    if int(vit_cfg.image_size) != 32:
        train_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
        eval_steps.append(transforms.Resize((int(vit_cfg.image_size), int(vit_cfg.image_size))))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    eval_steps.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    train_set = datasets.CIFAR10(
        root=str(cfg.data_dir),
        train=True,
        download=bool(cfg.download),
        transform=transforms.Compose(train_steps),
    )
    test_set = datasets.CIFAR10(
        root=str(cfg.data_dir),
        train=False,
        download=bool(cfg.download),
        transform=transforms.Compose(eval_steps),
    )
    return make_loaders(cfg, device, maybe_subset(train_set, cfg.train_subset), maybe_subset(test_set, cfg.test_subset))


def build_imagenet_loaders(
    cfg: ScalingRunConfig,
    device: torch.device,
    vit_cfg: ViTTinyConfig,
) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchvision is required for ImageNet") from exc

    data_root = Path(cfg.data_dir).expanduser()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(int(vit_cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_resize = max(int(vit_cfg.image_size), int(round(float(vit_cfg.image_size) * 256.0 / 224.0)))
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(int(vit_cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        train_set = datasets.ImageFolder(str(train_dir), transform=train_transform)
        test_set = datasets.ImageFolder(str(val_dir), transform=eval_transform)
    else:
        train_set = datasets.ImageNet(root=str(data_root), split="train", transform=train_transform)
        test_set = datasets.ImageNet(root=str(data_root), split="val", transform=eval_transform)

    return make_loaders(cfg, device, maybe_subset(train_set, cfg.train_subset), maybe_subset(test_set, cfg.test_subset))


def make_loaders(cfg: ScalingRunConfig, device: torch.device, train_set: Any, test_set: Any) -> tuple[DataLoader, DataLoader]:
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(cfg.seed))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(cfg.num_workers) > 0,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(cfg.num_workers) > 0,
    )
    return train_loader, test_loader


def build_loaders(cfg: ScalingRunConfig, device: torch.device, vit_cfg: ViTTinyConfig) -> tuple[DataLoader, DataLoader]:
    dataset = str(cfg.dataset).strip().lower()
    if dataset == "mnist":
        return build_mnist_loaders(cfg, device, vit_cfg)
    if dataset == "cifar10":
        return build_cifar10_loaders(cfg, device, vit_cfg)
    if dataset == "imagenet":
        return build_imagenet_loaders(cfg, device, vit_cfg)
    raise ValueError(f"unsupported dataset: {cfg.dataset!r}")


def export_named_tensors(model: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    target = getattr(model, "_orig_mod", model)
    return {name: target.w(name).detach().cpu().clone() for name in names}


def load_raw_checkpoint(path: str, vit_cfg: ViTTinyConfig) -> tuple[dict[str, torch.Tensor], int]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"raw checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "named_tensors" not in payload:
        raise KeyError(f"raw checkpoint must contain named_tensors: {checkpoint_path}")
    named_tensors = payload["named_tensors"]
    if not isinstance(named_tensors, dict):
        raise TypeError(f"named_tensors must be a dict in {checkpoint_path}")

    expected = make_initial_tensors(vit_cfg, seed=0)
    loaded: dict[str, torch.Tensor] = {}
    for name, expected_tensor in expected.items():
        if name not in named_tensors:
            raise KeyError(f"raw checkpoint missing tensor {name!r}: {checkpoint_path}")
        tensor = named_tensors[name].detach().cpu().to(dtype=torch.float32)
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"raw checkpoint tensor shape mismatch for {name}: "
                f"got {tuple(tensor.shape)}, expected {tuple(expected_tensor.shape)}"
            )
        loaded[name] = tensor.contiguous()
    return loaded, int(payload.get("step", 0))


def build_model(
    cfg: ScalingRunConfig,
    vit_cfg: ViTTinyConfig,
    initial_tensors: dict[str, torch.Tensor],
    *,
    big_vae_decoder: BigWeightVAE | None,
    big_vae_diffusion_prior: Any | None,
) -> nn.Module:
    if cfg.setup == "raw":
        return FunctionalViTTiny(vit_cfg, initial_tensors, parameter_mode="direct")
    if cfg.setup != "latent":
        raise ValueError(f"setup must be 'raw' or 'latent', got {cfg.setup!r}")
    if big_vae_decoder is None:
        raise ValueError("--big-vae-checkpoint is required for setup=latent")
    return FunctionalViTTiny(
        vit_cfg,
        initial_tensors,
        parameter_mode="bigvae_latent",
        big_vae=big_vae_decoder,
        big_vae_latent_init=str(cfg.big_vae_latent_init),
        big_vae_diffusion_prior=big_vae_diffusion_prior,
        big_vae_diffusion_prior_steps=int(cfg.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(cfg.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(cfg.big_vae_diffusion_prior_eta),
        big_vae_decode=str(cfg.big_vae_decode),
        big_vae_tile_T_patches=int(cfg.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(cfg.big_vae_tile_d_out),
        big_vae_encoder_context_rows=int(cfg.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(cfg.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(cfg.big_vae_encoder_batch_size),
    )


def train_once(cfg: ScalingRunConfig, vit_cfg: ViTTinyConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"run": asdict(cfg), "vit": asdict(vit_cfg)})

    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.benchmark = True

    seed_everything(int(cfg.seed))
    train_loader, test_loader = build_loaders(cfg, device, vit_cfg)

    init_checkpoint_steps = 0
    if cfg.setup == "latent":
        if str(cfg.raw_checkpoint).strip():
            initial_tensors, init_checkpoint_steps = load_raw_checkpoint(cfg.raw_checkpoint, vit_cfg)
            print(f"[latent] loaded raw checkpoint init: {cfg.raw_checkpoint}", flush=True)
        else:
            initial_tensors = make_initial_tensors(vit_cfg, seed=int(cfg.seed))
            print("[latent] no raw checkpoint provided; using fresh ViT initialization from seed", flush=True)
    else:
        initial_tensors = make_initial_tensors(vit_cfg, seed=int(cfg.seed))

    big_vae_decoder = None
    big_vae_diffusion_prior = None
    if cfg.setup == "latent":
        print(f"[latent] loading frozen BigVAE decoder: {cfg.big_vae_checkpoint}", flush=True)
        big_vae_decoder = load_frozen_big_vae_decoder(cfg.big_vae_checkpoint, device=device)
        if str(cfg.big_vae_latent_init).strip().lower() == "diffusion_prior":
            print(
                f"[latent] loading frozen latent diffusion prior: {cfg.big_vae_diffusion_prior_checkpoint}",
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
                        "[latent] loaded finetuned distribution encoder state from latent diffusion prior checkpoint",
                        flush=True,
                    )

    model = build_model(
        cfg,
        vit_cfg,
        initial_tensors,
        big_vae_decoder=big_vae_decoder,
        big_vae_diffusion_prior=big_vae_diffusion_prior,
    ).to(device)
    if bool(cfg.compile):
        model = torch.compile(model)

    target = getattr(model, "_orig_mod", model)
    store = getattr(target, "store")
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = build_optimizer(trainable_parameters, cfg=cfg, setup=str(cfg.setup))
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and bool(cfg.amp)))

    trainable_params = count_trainable_parameters(model)
    decoded_params = int(store.decoded_numel())
    latent_params = store.latent_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    decoded_big_vae_params = store.big_vae_decoded_numel() if isinstance(store, BigVAELatentTensorStore) else 0
    tile_count = store.decoded_tile_count() if isinstance(store, BigVAELatentTensorStore) else 0
    print(
        f"[{cfg.setup}] dataset={cfg.dataset} model={cfg.model_size} lr={float(cfg.lr):g} "
        f"trainable_params={trainable_params} decoded_params={decoded_params} "
        f"latent_params={latent_params} bigvae_decoded_params={decoded_big_vae_params} tiles={tile_count}",
        flush=True,
    )

    metrics_path = output_dir / "metrics.csv"
    checkpoint_dir = output_dir / "checkpoints"
    global_step = 0
    train_loss_window = 0.0
    train_count_window = 0
    start_time = time.time()
    best_accuracy = 0.0
    best_step = 0
    final_eval = EvalMetrics(loss=float("nan"), accuracy=0.0, examples=0)

    for epoch_idx in range(int(cfg.epochs)):
        model.train()
        for images, labels in train_loader:
            global_step += 1
            images = images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, bool(cfg.amp)):
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=float(cfg.label_smoothing),
                )
            scaler.scale(loss).backward()
            if float(cfg.grad_clip_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            batch_examples = int(labels.numel())
            train_loss_window += float(loss.detach().cpu().item()) * batch_examples
            train_count_window += batch_examples

            should_log = global_step == 1 or global_step % max(1, int(cfg.log_every_steps)) == 0
            should_eval = global_step == 1 or global_step % max(1, int(cfg.eval_every_steps)) == 0
            max_steps = int(cfg.max_steps)
            if max_steps > 0 and global_step >= max_steps:
                should_eval = True

            if should_eval:
                final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.amp))
                if final_eval.accuracy > best_accuracy:
                    best_accuracy = float(final_eval.accuracy)
                    best_step = int(global_step)
            if should_log or should_eval:
                avg_train_loss = train_loss_window / max(1, train_count_window)
                elapsed_s = time.time() - start_time
                total_steps_with_raw_init = int(global_step + (init_checkpoint_steps if cfg.setup == "latent" else 0))
                row = {
                    "dataset": str(cfg.dataset),
                    "model_size": str(cfg.model_size),
                    "setup": str(cfg.setup),
                    "lr": float(cfg.lr),
                    "step": int(global_step),
                    "total_steps_with_raw_init": int(total_steps_with_raw_init),
                    "epoch": int(epoch_idx + 1),
                    "train_loss": float(avg_train_loss),
                    "test_loss": float(final_eval.loss),
                    "test_accuracy": float(final_eval.accuracy),
                    "best_accuracy": float(best_accuracy),
                    "trainable_params": int(trainable_params),
                    "decoded_params": int(decoded_params),
                    "latent_params": int(latent_params),
                    "elapsed_s": float(elapsed_s),
                }
                append_csv_row(metrics_path, row)
                print(
                    f"[{cfg.setup}] step={global_step} total_step={total_steps_with_raw_init} "
                    f"epoch={epoch_idx + 1} train_loss={avg_train_loss:.4f} "
                    f"test_loss={final_eval.loss:.4f} test_acc={final_eval.accuracy:.4f} "
                    f"best={best_accuracy:.4f}",
                    flush=True,
                )
                train_loss_window = 0.0
                train_count_window = 0

            if max_steps > 0 and global_step >= max_steps:
                break
        if int(cfg.max_steps) > 0 and global_step >= int(cfg.max_steps):
            break

    final_eval = evaluate(model, test_loader, device=device, amp_enabled=bool(cfg.amp))
    if final_eval.accuracy > best_accuracy:
        best_accuracy = float(final_eval.accuracy)
        best_step = int(global_step)

    if bool(cfg.save_checkpoints):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if cfg.setup == "raw":
            torch.save(
                {
                    "setup": "raw",
                    "step": int(global_step),
                    "named_tensors": export_named_tensors(model, list(initial_tensors.keys())),
                    "vit_config": asdict(vit_cfg),
                    "run_config": asdict(cfg),
                },
                checkpoint_dir / "raw_final.pt",
            )
        else:
            torch.save(
                {
                    "setup": "latent",
                    "step": int(global_step),
                    "init_raw_checkpoint": str(cfg.raw_checkpoint),
                    "init_raw_steps": int(init_checkpoint_steps),
                    "latent_slots": target.store.latent_slots.state_dict(),
                    "vit_config": asdict(vit_cfg),
                    "run_config": asdict(cfg),
                },
                checkpoint_dir / "latent_final.pt",
            )

    summary = {
        "dataset": str(cfg.dataset),
        "model_size": str(cfg.model_size),
        "setup": str(cfg.setup),
        "steps": int(global_step),
        "init_raw_steps": int(init_checkpoint_steps if cfg.setup == "latent" else 0),
        "total_steps_with_raw_init": int(global_step + (init_checkpoint_steps if cfg.setup == "latent" else 0)),
        "final_test_loss": float(final_eval.loss),
        "final_test_accuracy": float(final_eval.accuracy),
        "best_test_accuracy": float(best_accuracy),
        "best_step": int(best_step),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.latent_weight_decay if cfg.setup == "latent" else cfg.weight_decay),
        "trainable_params": int(trainable_params),
        "decoded_params": int(decoded_params),
        "latent_params": int(latent_params),
        "bigvae_decoded_params": int(decoded_big_vae_params),
        "bigvae_tile_count": int(tile_count),
        "raw_checkpoint": str(cfg.raw_checkpoint) if cfg.setup == "latent" else "",
        "big_vae_checkpoint": str(cfg.big_vae_checkpoint) if cfg.setup == "latent" else "",
        "big_vae_latent_init": str(cfg.big_vae_latent_init) if cfg.setup == "latent" else "",
        "big_vae_diffusion_prior_checkpoint": (
            str(cfg.big_vae_diffusion_prior_checkpoint) if cfg.setup == "latent" else ""
        ),
        "big_vae_diffusion_prior_steps": int(cfg.big_vae_diffusion_prior_steps) if cfg.setup == "latent" else 0,
        "big_vae_diffusion_prior_sampler": (
            str(cfg.big_vae_diffusion_prior_sampler) if cfg.setup == "latent" else ""
        ),
        "big_vae_diffusion_prior_eta": float(cfg.big_vae_diffusion_prior_eta) if cfg.setup == "latent" else 0.0,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> tuple[ScalingRunConfig, ViTTinyConfig]:
    parser = argparse.ArgumentParser(
        description="Scale raw-weight AdamW vs BigVAE latent AdamW optimization for ViT classifiers.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "imagenet"), required=True)
    parser.add_argument("--model-size", required=True)
    parser.add_argument("--setup", choices=("raw", "latent"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--latent-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-checkpoint", default="")
    parser.add_argument("--big-vae-checkpoint", default="")
    parser.add_argument(
        "--big-vae-latent-init",
        choices=("base", "random", "encoded", "diffusion_prior"),
        default="encoded",
    )
    parser.add_argument("--big-vae-diffusion-prior-checkpoint", default="")
    parser.add_argument("--big-vae-diffusion-prior-steps", type=int, default=50)
    parser.add_argument("--big-vae-diffusion-prior-sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--big-vae-diffusion-prior-eta", type=float, default=0.0)
    parser.add_argument("--big-vae-decode", choices=("weights", "all"), default="all")
    parser.add_argument("--big-vae-tile-t-patches", "--big-vae-tile-T-patches", dest="big_vae_tile_T_patches", type=int, default=16)
    parser.add_argument("--big-vae-tile-d-out", type=int, default=8)
    parser.add_argument("--big-vae-encoder-context-rows", type=int, default=64)
    parser.add_argument("--big-vae-encoder-context-std", type=float, default=1.0)
    parser.add_argument("--big-vae-encoder-batch-size", type=int, default=16)
    parser.add_argument("--raw-init-steps", type=int, default=0)

    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)

    args = parser.parse_args()
    setup = str(args.setup).strip().lower()

    run_cfg = ScalingRunConfig(
        output_dir=str(args.output_dir),
        dataset=str(args.dataset),
        model_size=str(args.model_size),
        setup=setup,
        data_dir=str(args.data_dir),
        download=bool(args.download),
        device=str(args.device),
        seed=int(args.seed),
        epochs=int(args.epochs),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        train_subset=int(args.train_subset),
        test_subset=int(args.test_subset),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
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
        raw_checkpoint=str(args.raw_checkpoint),
        big_vae_checkpoint=str(args.big_vae_checkpoint),
        big_vae_latent_init=str(args.big_vae_latent_init),
        big_vae_diffusion_prior_checkpoint=str(args.big_vae_diffusion_prior_checkpoint),
        big_vae_diffusion_prior_steps=int(args.big_vae_diffusion_prior_steps),
        big_vae_diffusion_prior_sampler=str(args.big_vae_diffusion_prior_sampler),
        big_vae_diffusion_prior_eta=float(args.big_vae_diffusion_prior_eta),
        big_vae_decode=str(args.big_vae_decode),
        big_vae_tile_T_patches=int(args.big_vae_tile_T_patches),
        big_vae_tile_d_out=int(args.big_vae_tile_d_out),
        big_vae_encoder_context_rows=int(args.big_vae_encoder_context_rows),
        big_vae_encoder_context_std=float(args.big_vae_encoder_context_std),
        big_vae_encoder_batch_size=int(args.big_vae_encoder_batch_size),
        latent_weight_decay=float(args.latent_weight_decay),
        raw_init_steps=int(args.raw_init_steps),
    )
    validate_scaling_run_config(run_cfg)
    vit_cfg = ViTTinyConfig(
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        in_channels=int(args.in_channels),
        num_classes=int(args.num_classes),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attention_dropout=float(args.attention_dropout),
    )
    return run_cfg, vit_cfg


def main() -> None:
    cfg, vit_cfg = parse_args()
    random.seed(int(cfg.seed))
    train_once(cfg, vit_cfg)


if __name__ == "__main__":
    main()
