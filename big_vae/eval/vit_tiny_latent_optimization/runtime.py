from __future__ import annotations

from big_vae.runtime.artifacts import append_csv_row as append_standard_csv_row
from big_vae.runtime.artifacts import write_artifact_layout, write_json_file

from .modeling import *

def _normal(shape: Iterable[int], *, generator: torch.Generator, std: float) -> torch.Tensor:
    return torch.empty(tuple(int(dim) for dim in shape), dtype=torch.float32).normal_(
        mean=0.0,
        std=float(std),
        generator=generator,
    )


def make_initial_tensors(cfg: ViTTinyConfig, *, seed: int) -> dict[str, torch.Tensor]:
    if cfg.image_size % cfg.patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    mlp_dim = int(round(cfg.hidden_dim * cfg.mlp_ratio))
    tensors: dict[str, torch.Tensor] = {}

    tensors["cls_token"] = _normal((1, 1, cfg.hidden_dim), generator=generator, std=0.02)
    tensors["pos_embed"] = _normal((1, num_patches + 1, cfg.hidden_dim), generator=generator, std=0.02)
    tensors["patch_embed.weight"] = _normal(
        (cfg.hidden_dim, cfg.in_channels, cfg.patch_size, cfg.patch_size),
        generator=generator,
        std=0.02,
    )
    tensors["patch_embed.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)

    for layer_idx in range(cfg.depth):
        prefix = f"blocks.{layer_idx}"
        tensors[f"{prefix}.norm1.weight"] = torch.ones(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.norm1.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.attn.qkv.weight"] = _normal(
            (3 * cfg.hidden_dim, cfg.hidden_dim),
            generator=generator,
            std=0.02,
        )
        tensors[f"{prefix}.attn.qkv.bias"] = torch.zeros(3 * cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.attn.proj.weight"] = _normal(
            (cfg.hidden_dim, cfg.hidden_dim),
            generator=generator,
            std=0.02,
        )
        tensors[f"{prefix}.attn.proj.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.norm2.weight"] = torch.ones(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.norm2.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
        tensors[f"{prefix}.mlp.fc1.weight"] = _normal((mlp_dim, cfg.hidden_dim), generator=generator, std=0.02)
        tensors[f"{prefix}.mlp.fc1.bias"] = torch.zeros(mlp_dim, dtype=torch.float32)
        tensors[f"{prefix}.mlp.fc2.weight"] = _normal((cfg.hidden_dim, mlp_dim), generator=generator, std=0.02)
        tensors[f"{prefix}.mlp.fc2.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)

    tensors["norm.weight"] = torch.ones(cfg.hidden_dim, dtype=torch.float32)
    tensors["norm.bias"] = torch.zeros(cfg.hidden_dim, dtype=torch.float32)
    tensors["head.weight"] = _normal((cfg.num_classes, cfg.hidden_dim), generator=generator, std=0.02)
    tensors["head.bias"] = torch.zeros(cfg.num_classes, dtype=torch.float32)
    return tensors


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def normalize_setup_name(setup: str) -> str:
    value = str(setup).strip().lower()
    return value


def load_frozen_big_vae_decoder(checkpoint_path: str, *, device: torch.device) -> BigWeightVAE:
    return load_frozen_big_vae_from_checkpoint(checkpoint_path, device=device)


def resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_cifar10_loaders(cfg: ExperimentConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - exercised only in missing optional dependency envs.
        raise RuntimeError("torchvision is required for CIFAR-10 pipeline") from exc

    data_root = Path(cfg.data_dir)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.CIFAR10(root=str(data_root), train=True, download=bool(cfg.download), transform=train_transform)
    test_set = datasets.CIFAR10(root=str(data_root), train=False, download=bool(cfg.download), transform=eval_transform)

    if int(cfg.train_subset) > 0:
        train_set = Subset(train_set, list(range(min(int(cfg.train_subset), len(train_set)))))
    if int(cfg.test_subset) > 0:
        test_set = Subset(test_set, list(range(min(int(cfg.test_subset), len(test_set)))))

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


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


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


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    append_standard_csv_row(path, row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_file(path, payload)


def build_optimizer(
    parameters: list[torch.nn.Parameter],
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    cfg: ExperimentConfig,
) -> torch.optim.Optimizer:
    name = str(optimizer_name).strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(lr),
            betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
            eps=float(cfg.adam_eps),
            weight_decay=float(weight_decay),
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(lr),
            momentum=float(cfg.sgd_momentum),
            weight_decay=float(weight_decay),
            nesterov=bool(cfg.sgd_nesterov),
        )
    raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {optimizer_name!r}")


def fit_big_vae_latents_to_initial_weights(
    model: nn.Module,
    initial_tensors: dict[str, torch.Tensor],
    *,
    steps: int,
    lr: float,
    log_every: int,
    setup: str,
) -> None:
    target = getattr(model, "_orig_mod", model)
    store = getattr(target, "store", None)
    if not isinstance(store, BigVAELatentTensorStore):
        raise TypeError("BigVAE latent fitting requires FunctionalViTTiny.store to be BigVAELatentTensorStore")

    optimizer = torch.optim.AdamW(store.latent_slots.parameters(), lr=float(lr), weight_decay=0.0)
    target_matrices = {
        name: store.target_matrix(name, initial_tensors[name]).to(next(store.latent_slots.parameters()).device)
        for name in store.decoded_tensor_names()
    }
    total_numel = sum(matrix.numel() for matrix in target_matrices.values())
    for step_idx in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = None
        decoded_matrices = store.decode_all_matrices()
        for name, target_matrix in target_matrices.items():
            decoded = decoded_matrices[name]
            term = F.mse_loss(decoded, target_matrix, reduction="sum")
            loss = term if loss is None else loss + term
        assert loss is not None
        loss = loss / max(1, int(total_numel))
        loss.backward()
        optimizer.step()
        if step_idx == 1 or step_idx % max(1, int(log_every)) == 0 or step_idx == int(steps):
            print(f"[{setup}] latent_init_fit step={step_idx}/{steps} mse={float(loss.detach().cpu().item()):.6e}", flush=True)

