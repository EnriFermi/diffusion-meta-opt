"""Immutable scientific contract for the first WeightCLIP comparison.

This module contains no model code.  Its job is to make protocol drift an
explicit error and to give every artifact a stable contract fingerprint.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


BENCHMARK_CONTRACT_VERSION = "weightclip-resnet18slim-v1-20260818"

OFFICIAL_GIT_URL = "https://github.com/HSG-AIML/weightCLIP.git"
OFFICIAL_GIT_COMMIT = "be080677a6eceacdbe3b2823caffe3c0cc73fa7e"
OFFICIAL_HF_REPO = "aasefaw/WeightCLIP"
OFFICIAL_HF_REVISION = "cb672e6fe1b56e929705d1b6b2eb6635fcc1e604"


class CandidateProtocol(str, enum.Enum):
    """Candidate-selection estimators that must never share a result table."""

    CONTROLLED_SINGLE = "controlled_single"
    CONTROLLED_VALIDATION_BEST_K = "controlled_validation_best_k"
    NATIVE_TEST_TOP5_ORACLE = "native_test_top5_oracle"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    slug: str


SOURCE_DATASETS = (
    DatasetSpec("Artworks", "artworks"),
    DatasetSpec("Blood Cells", "blood_cells"),
    DatasetSpec("Breast Cancer Tissues", "breast_cancer_tissues"),
    DatasetSpec("Aerial Cactus", "aerial_cactus"),
    DatasetSpec("Cassava Leaf", "cassava_leaf"),
    DatasetSpec("CT Images", "ct_images"),
    DatasetSpec("Land Use", "land_use"),
    DatasetSpec("Lego Bricks", "lego_bricks"),
    DatasetSpec("Real/Fake Legos", "real_fake_legos"),
    DatasetSpec("Casting", "casting"),
)

OOD_DATASETS = (
    "colorectal-histology",
    "covid19",
    "speed-limit-signs",
    "honeybee-pollen",
    "real-or-drawing",
    "cifar10",
)


@dataclass(frozen=True)
class OfficialArtifactSpec:
    path: str
    sha256: str
    size_bytes: int


OFFICIAL_RESNET_CHECKPOINT = OfficialArtifactSpec(
    path="checkpoints/resnet18/checkpoint.pt",
    sha256="73b2a5de1a9a167145ecdfe3684725c9094ed766156589d4a00647e21128d95a",
    size_bytes=9_143_918_586,
)
OFFICIAL_RESNET_DATASET_ENCODER = OfficialArtifactSpec(
    path="checkpoints/resnet18/dataset_encoder.pt",
    sha256="9b507ba96c0d5afd400c32ca760a134127c6ed2a0301a97db6f41e151d601b26",
    size_bytes=3_763_936,
)


@dataclass(frozen=True)
class PopulationContract:
    architecture: str = "ResNet18Slim"
    width_mult: float = 0.5
    dropout: float = 0.15
    lineages_per_dataset: int = 50
    epochs_per_lineage: int = 45
    primary_epoch_indices_zero_based: tuple[int, int] = (43, 44)
    primary_epochs_one_based: tuple[int, int] = (44, 45)
    train_lineages_per_dataset: int = 35
    validation_lineages_per_dataset: int = 7
    internal_test_lineages_per_dataset: int = 8


@dataclass(frozen=True)
class AccessContract:
    dataset_prompt_images: int = 10
    dataset_prompt_split: str = "train"
    activation_context_max_images: int = 512
    classifier_head_policy: str = "fresh_default_random_paired_seed"
    ours_batchnorm_policy: str = "default_affine_and_running_then_common_calibration"
    weightclip_batchnorm_policy: str = "released_affine_then_common_running_calibration"
    bn_calibration_max_batches: int = 200
    target_validation_source: str = "fixed_train_derived_validation"


@dataclass(frozen=True)
class EvaluationContract:
    controlled_candidate_count: int = 100
    controlled_top_k: int = 5
    primary_protocol: CandidateProtocol = CandidateProtocol.CONTROLLED_SINGLE
    secondary_protocol: CandidateProtocol = CandidateProtocol.CONTROLLED_VALIDATION_BEST_K
    native_protocol: CandidateProtocol = CandidateProtocol.NATIVE_TEST_TOP5_ORACLE
    native_candidate_count: int = 100
    native_top_k: int = 5
    finetune_epochs: int = 10
    report_epochs: tuple[int, int, int] = (0, 1, 10)
    finetune_optimizer: str = "SGD"
    finetune_lr: float = 1.5e-4
    finetune_momentum: float = 0.9
    finetune_weight_decay: float = 0.0
    batch_size: int = 10


@dataclass(frozen=True)
class WeightCLIPBenchmarkContract:
    """Frozen top-level contract with deterministic serialization."""

    version: str = BENCHMARK_CONTRACT_VERSION
    official_git_url: str = OFFICIAL_GIT_URL
    official_git_commit: str = OFFICIAL_GIT_COMMIT
    official_hf_repo: str = OFFICIAL_HF_REPO
    official_hf_revision: str = OFFICIAL_HF_REVISION
    source_datasets: tuple[DatasetSpec, ...] = SOURCE_DATASETS
    ood_datasets: tuple[str, ...] = OOD_DATASETS
    population: PopulationContract = PopulationContract()
    access: AccessContract = AccessContract()
    evaluation: EvaluationContract = EvaluationContract()

    def validate(self) -> None:
        pop = self.population
        if pop.primary_epoch_indices_zero_based != tuple(x - 1 for x in pop.primary_epochs_one_based):
            raise ValueError("Primary checkpoint epoch mapping has an off-by-one mismatch")
        split_total = (
            pop.train_lineages_per_dataset
            + pop.validation_lineages_per_dataset
            + pop.internal_test_lineages_per_dataset
        )
        if split_total != pop.lineages_per_dataset:
            raise ValueError(f"Lineage split totals {split_total}, expected {pop.lineages_per_dataset}")
        if len({item.slug for item in self.source_datasets}) != len(self.source_datasets):
            raise ValueError("Source dataset slugs are not unique")
        if set(item.slug for item in self.source_datasets) & set(self.ood_datasets):
            raise ValueError("Source and OOD dataset lists overlap")
        ev = self.evaluation
        if ev.native_candidate_count != 100 or ev.native_top_k != 5:
            raise ValueError("The mandatory native oracle protocol must remain 100 -> top 5")
        if ev.controlled_candidate_count != 100 or ev.controlled_top_k != 5:
            raise ValueError("Controlled secondary protocol must remain validation-selected 100 -> top 5")
        if self.access.dataset_prompt_split != "train":
            raise ValueError("Dataset prompts may only use target training images")

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        return _enum_values(payload)

    def canonical_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def assert_matches(self, payload: Mapping[str, Any]) -> None:
        observed = payload.get("contract_fingerprint")
        if observed != self.fingerprint():
            raise ValueError(
                "Benchmark contract mismatch: "
                f"artifact={observed!r}, expected={self.fingerprint()!r}"
            )


def _enum_values(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_values(item) for item in value]
    return value


DEFAULT_CONTRACT = WeightCLIPBenchmarkContract()
DEFAULT_CONTRACT.validate()
