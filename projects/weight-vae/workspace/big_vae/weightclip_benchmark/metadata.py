"""Frozen WeightCLIP benchmark metadata and protocol validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetTask:
    key: str
    paper_name: str
    aliases: tuple[str, ...]
    raw_tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OODDatasetTask:
    """Exact released WeightCLIP/TANS OOD dataset identity."""

    key: str
    raw_task: str
    classes: int


# The paper list is authoritative.  In particular, ASL is not a substitute for
# the distinct Lego Bricks task.
SOURCE_TASKS: tuple[DatasetTask, ...] = (
    DatasetTask("artworks", "Artworks", ("artworks_resnet",), ("best-artworks-of-all-time_ikarus777_0_17", "best-artworks-of-all-time_ikarus777_17_34", "best-artworks-of-all-time_ikarus777_34_51")),
    DatasetTask("blood_cells", "Blood Cells", ("blood_cells_resnet", "blood-cells"), ("blood-cells_paultimothymooney_0_4",)),
    DatasetTask("breast_cancer_tissues", "Breast Cancer Tissues", ("breakhis_resnet", "breakhis"), ("breakhis_ambarish_0_8",)),
    DatasetTask("aerial_cactus", "Aerial Cactus", ("cactus_aerial_resnet", "cactus-aerial"), ("cactus-aerial-photos_irvingvasquez_0_2",)),
    DatasetTask("cassava_leaf", "Cassava Leaf", ("cassava_leaf_resnet",), ("cassava-leaf-disease-classification_0_5",)),
    DatasetTask("ct_images", "CT Images", ("ct_images_resnet",), ("computed-tomography-ct-images_vbookshelf_0_2",)),
    DatasetTask("land_use", "Land Use", ("land_cover_resnet", "land-cover"), ("land-cover-class_0_10",)),
    DatasetTask("lego_bricks", "Lego Bricks", ("lego_bricks_resnet", "lego-bricks"), ("lego-brick-images_joosthazelzet_0_16",)),
    DatasetTask("real_fake_legos", "Real/Fake Legos", ("lego_vs_generic_resnet",), ("lego-vs-generic-brick-image-recognition_pacogarciam3_0_4",)),
    DatasetTask("casting", "Casting", ("casting_resnet",), ("real-life-industrial-dataset-of-casting-product_ravirajsinh45_0_2",)),
)

OOD_ARCHIVE_TASKS: tuple[OODDatasetTask, ...] = (
    OODDatasetTask("colorectal-histology", "colorectal-histology-mnist_kmader", 8),
    OODDatasetTask("covid19", "covid19-radiography-database_tawsifurrahman", 3),
    OODDatasetTask("speed-limit-signs", "drr-sign", 4),
    OODDatasetTask("honeybee-pollen", "honey-bee-pollen_ivanfel", 2),
    OODDatasetTask("real-or-drawing", "ml2020spring-hw12", 10),
)
OOD_TASKS: tuple[str, ...] = tuple(task.key for task in OOD_ARCHIVE_TASKS) + ("cifar10",)
OOD_CLASS_COUNTS: dict[str, int] = {task.key: task.classes for task in OOD_ARCHIVE_TASKS} | {"cifar10": 10}


@dataclass(frozen=True, slots=True)
class ZooProtocol:
    lineages_per_dataset: int = 50
    epochs: int = 45
    scheduler_epochs: int = 50
    primary_checkpoint_indices: tuple[int, int] = (43, 44)
    archive_initialization: bool = True
    archive_every_epoch: bool = True
    lineage_split: tuple[int, int, int] = (35, 7, 8)
    lr_candidates: tuple[float, ...] = (0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7)
    lr_sweep_seeds: int = 2
    batch_size: int = 256
    momentum: float = 0.9
    weight_decay: float = 5e-4
    dropout: float = 0.15
    width_mult: float = 0.5
    init_type: str = "normal"
    grad_clip: float = 0.0
    models_per_gpu: int = 4
    seed_namespace: str = "weightclip-resnet18slim-zoo-v1"

    def validate(self) -> None:
        if sum(self.lineage_split) != self.lineages_per_dataset:
            raise ValueError("lineage_split must sum to lineages_per_dataset")
        expected = (self.epochs - 2, self.epochs - 1)
        if self.primary_checkpoint_indices != expected:
            raise ValueError(f"primary checkpoint indices must be {expected}")
        if self.lr_sweep_seeds != 2 or len(self.lr_candidates) != 7:
            raise ValueError("the frozen protocol requires the exact 2x7 LR sweep")
        if self.archive_every_epoch is not True or self.archive_initialization is not True:
            raise ValueError("initialization and every one-based epoch must be archived")
        if self.init_type != "normal" or self.dropout != 0.15 or self.grad_clip != 0.0:
            raise ValueError("resolved config conflicts with the released shell contract")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperatorDatasetProtocol:
    tile_rows: int = 128
    tile_cols: int = 128
    activation_rows: int = 512
    quantiles: int = 16
    context_shard_records: int = 256
    weight_shard_records: int = 512
    checkpoint_splits: tuple[str, ...] = ("train",)
    primary_checkpoint_indices: tuple[int, int] = (43, 44)
    permutation_views: int = 5
    raw_dtype: str = "float32"
    bank_dtype: str = "float32"

    def validate(self) -> None:
        if (self.tile_rows, self.tile_cols) != (128, 128):
            raise ValueError("primary approved tile is 128x128")
        if self.activation_rows != 512:
            raise ValueError("native activation cap must be 512 rows")
        if self.quantiles != 16:
            raise ValueError("production activation summary must use exactly 16 quantiles")
        # Shard sizes are frozen artifact-layout knobs, not free operational
        # tuning parameters: changing them changes immutable object boundaries,
        # manifest hashes, cache identity, and exact resume provenance.
        if self.context_shard_records != 256 or self.weight_shard_records != 512:
            raise ValueError("production shard layout must be context=256 and weight=512 records")
        if tuple(self.checkpoint_splits) != ("train",):
            raise ValueError("production operator bank checkpoint_splits must be exactly ('train',)")
        if tuple(self.primary_checkpoint_indices) != (43, 44):
            raise ValueError("production primary checkpoint indices must be exactly (43, 44)")
        if self.permutation_views != 5:
            raise ValueError("production operator bank must use exactly 5 permutation views")
        if self.raw_dtype != "float32" or self.bank_dtype != "float32":
            raise ValueError("production raw and bank dtypes must both be float32")


# Match credential field names, not scientific words that merely contain
# ``token`` (for example ``patch_tokenizer`` or ``max_latent_tokens``).
# Values still receive Bearer/Hugging Face token-pattern redaction below.
_SECRET_KEY = re.compile(
    r"(?:^token$|(?:^|[_-])(?:"
    r"auth[_-]?token|access[_-]?token|refresh[_-]?token|bearer[_-]?token|hf[_-]?token|"
    r"api[_-]?key|access[_-]?key|secret[_-]?key|secret|client[_-]?secret|"
    r"password|passwd|credential|credentials"
    r")$)",
    re.I,
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+")
_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def redact_secrets(value: Any) -> Any:
    """Recursively redact secret-looking keys and token-shaped string values."""

    if isinstance(value, dict):
        return {str(k): "<redacted>" if _SECRET_KEY.search(str(k)) else redact_secrets(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _HF_TOKEN.sub("<redacted>", _BEARER.sub("Bearer <redacted>", value))
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(redact_secrets(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def content_id(payload: Any, prefix: str = "sha256") -> str:
    return f"{prefix}:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def frozen_contract() -> dict[str, Any]:
    protocol = ZooProtocol()
    protocol.validate()
    return {
        "protocol_version": 1,
        "source_tasks": [asdict(task) for task in SOURCE_TASKS],
        "ood_tasks": [asdict(task) for task in OOD_ARCHIVE_TASKS] + [
            {"key": "cifar10", "raw_task": "torchvision.datasets.CIFAR10", "classes": 10}
        ],
        "zoo": protocol.to_dict(),
        "operator_dataset": asdict(OperatorDatasetProtocol()),
    }
