from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .features import validate_semantic_token_features


@dataclass(frozen=True, slots=True)
class TileIdentity:
    """Identity fields that make an anchor/target pairing scientifically valid."""

    dataset_id: str
    lineage_id: str
    checkpoint_id: str
    layer_key: str
    tile_row: int
    tile_col: int

    @property
    def pairing_key(self) -> tuple[str, str, str, str, int, int]:
        return (
            self.dataset_id,
            self.lineage_id,
            self.checkpoint_id,
            self.layer_key,
            int(self.tile_row),
            int(self.tile_col),
        )


_RUNTIME_CONFIG_KEYS = {
    "device",
    "num_classes",
    "n_classes",
    "classes",
    "o_dim",
    "dataset",
    "datasets",
    "split",
    "source",
    "anchor",
    "template",
}


def _learned_config_projection(value: Any) -> Any:
    """Remove runtime/data-instance fields from learned architecture configs."""

    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _RUNTIME_CONFIG_KEYS or any(
                token in normalized for token in ("path", "root", "cache", "manifest")
            ):
                continue
            result[str(key)] = _learned_config_projection(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_learned_config_projection(item) for item in value]
    return value


def canonical_codec_fingerprint(codec: str, decoder_provenance: Mapping[str, Any]) -> str:
    """Hash the immutable decoder substrate, excluding runtime path/device noise."""

    if decoder_provenance.get("codec") != codec:
        raise ValueError(f"decoder provenance codec {decoder_provenance.get('codec')!r} != {codec!r}")
    if codec == "ours":
        payload = {
            "schema_version": 1,
            "codec": codec,
            "checkpoint_sha256": decoder_provenance.get("checkpoint_sha256"),
            "model_config_sha256": decoder_provenance.get("model_config_sha256"),
        }
        if not payload["checkpoint_sha256"] or not payload["model_config_sha256"]:
            raise ValueError("ours decoder provenance lacks checkpoint/model-config SHA-256")
    elif codec == "weightclip":
        official = decoder_provenance.get("official")
        if not isinstance(official, Mapping):
            raise ValueError("WeightCLIP decoder provenance lacks official resolved provenance")
        resolved = official.get("config")
        if not isinstance(resolved, Mapping):
            raise ValueError("WeightCLIP decoder provenance lacks resolved learned config")
        data_config = resolved.get("data", {})
        learned_codec_config = {
            "model": _learned_config_projection(resolved.get("model", {})),
            "tokenizer": _learned_config_projection(
                data_config.get("tokenizer", {}) if isinstance(data_config, Mapping) else {}
            ),
            "dataset_encoder": _learned_config_projection(resolved.get("dataset_encoder", {})),
        }
        payload = {
            "schema_version": 1,
            "codec": codec,
            "checkpoint_sha256": decoder_provenance.get("checkpoint_sha256"),
            "official_checkpoint_sha256": official.get("checkpoint", {}).get("sha256"),
            "dataset_encoder_sha256": official.get("dataset_encoder", {}).get("sha256"),
            "git_commit": official.get("git_commit"),
            "contract_fingerprint": official.get("contract_fingerprint"),
            "learned_codec_config": learned_codec_config,
        }
        missing = [key for key, value in payload.items() if key not in {"schema_version", "codec"} and not value]
        if missing:
            raise ValueError(f"WeightCLIP decoder provenance lacks fingerprint fields: {missing}")
    else:
        raise ValueError(f"unsupported codec {codec!r}")
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    except TypeError as exc:
        raise TypeError("decoder fingerprint provenance must be JSON-serializable") from exc
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


@dataclass(slots=True)
class GroupedLatentRecord:
    """All tile codes belonging to one independently trained checkpoint.

    Keeping checkpoint codes grouped prevents tile-level train/validation leakage and
    makes the statistical unit explicit. ``z_enc`` and ``z_task`` are [tiles, tokens,
    latent_dim] (a [tiles, latent_dim] tensor is accepted and promoted to one token).
    """

    group_id: str
    dataset_id: str
    lineage_id: str
    checkpoint_id: str
    split: str
    codec: str
    codec_fingerprint: str
    identities: tuple[TileIdentity, ...]
    z_task: torch.Tensor
    dataset_embedding: torch.Tensor
    architecture_features: torch.Tensor
    dataset_embedding_bank: torch.Tensor | None = None
    z_enc: torch.Tensor | None = None
    anchor_identities: tuple[TileIdentity, ...] | None = None
    tile_mask: torch.Tensor | None = None
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.codec_fingerprint.startswith("sha256:") or len(self.codec_fingerprint) != 71:
            raise ValueError("grouped latent record requires a canonical sha256 codec_fingerprint")
        self.z_task = _promote_codes(self.z_task, "z_task")
        if self.z_enc is not None:
            self.z_enc = _promote_codes(self.z_enc, "z_enc")
            if self.z_enc.shape != self.z_task.shape:
                raise ValueError(f"z_enc/z_task shape mismatch: {self.z_enc.shape} vs {self.z_task.shape}")
            if self.anchor_identities is None:
                self.anchor_identities = self.identities
        n_tiles = int(self.z_task.shape[0])
        if len(self.identities) != n_tiles:
            raise ValueError(f"identities has {len(self.identities)} entries for {n_tiles} tiles")
        if self.anchor_identities is not None and len(self.anchor_identities) != n_tiles:
            raise ValueError(f"anchor_identities has {len(self.anchor_identities)} entries for {n_tiles} tiles")
        if self.dataset_embedding.ndim != 1:
            raise ValueError("dataset_embedding must be one-dimensional")
        if self.dataset_embedding_bank is None:
            self.dataset_embedding_bank = self.dataset_embedding.unsqueeze(0)
        if self.dataset_embedding_bank.ndim != 2 or self.dataset_embedding_bank.shape[1:] != self.dataset_embedding.shape:
            raise ValueError("dataset_embedding_bank must be [candidates,embedding_dim]")
        if not torch.equal(self.dataset_embedding_bank[0], self.dataset_embedding):
            raise ValueError("dataset_embedding_bank[0] must be the canonical dataset embedding")
        if not torch.isfinite(self.dataset_embedding_bank).all():
            raise ValueError("dataset_embedding_bank contains NaN/inf")
        if self.tile_mask is None:
            self.tile_mask = torch.ones(self.z_task.shape[:2], dtype=torch.bool)
        elif tuple(self.tile_mask.shape) != tuple(self.z_task.shape[:2]):
            raise ValueError("tile_mask must be [tiles, tokens]")
        self.tile_mask = self.tile_mask.to(dtype=torch.bool)
        validate_semantic_token_features(self.architecture_features, self.z_task)
        if not torch.equal(self.architecture_features[..., 0].to(dtype=torch.bool), self.tile_mask):
            raise ValueError("shared architecture valid_token feature disagrees with tile_mask")
        if torch.any(self.architecture_features[..., 1].to(dtype=torch.bool) & ~self.tile_mask):
            raise ValueError("shared architecture controlled_body marks padded tokens")
        for identity in self.identities:
            if identity.dataset_id != self.dataset_id or identity.lineage_id != self.lineage_id:
                raise ValueError("tile identity disagrees with parent dataset/lineage")
            if identity.checkpoint_id != self.checkpoint_id:
                raise ValueError("tile identity disagrees with parent checkpoint")

    def assert_anchor_pairing(self) -> None:
        if self.z_enc is None:
            raise ValueError(f"record {self.group_id!r} has no z_enc anchor")
        if self.anchor_identities is None:
            raise ValueError(f"record {self.group_id!r} has no anchor identities")
        if len({identity.pairing_key for identity in self.identities}) != len(self.identities):
            raise ValueError(f"record {self.group_id!r} has duplicate tile pairing keys")
        for anchor, target in zip(self.anchor_identities, self.identities, strict=True):
            if anchor.pairing_key != target.pairing_key:
                raise ValueError(
                    f"fatal z_enc/z_task pairing mismatch in {self.group_id!r}: "
                    f"anchor={anchor.pairing_key}, target={target.pairing_key}"
                )

    def tile(self, index: int, *, prompt_index: int = 0) -> dict[str, Any]:
        identity = self.identities[index]
        if prompt_index < 0 or prompt_index >= len(self.dataset_embedding_bank):
            raise IndexError("dataset prompt augmentation index is out of bounds")
        result: dict[str, Any] = {
            "group_id": self.group_id,
            "dataset_id": self.dataset_id,
            "lineage_id": self.lineage_id,
            "checkpoint_id": self.checkpoint_id,
            "split": self.split,
            "codec": self.codec,
            "codec_fingerprint": self.codec_fingerprint,
            "identity": identity,
            "anchor_identity": None if self.anchor_identities is None else self.anchor_identities[index],
            "z_task": self.z_task[index],
            "dataset_embedding": self.dataset_embedding_bank[prompt_index],
            "dataset_prompt_index": int(prompt_index),
            "architecture_features": self.architecture_features[index],
            "token_mask": self.tile_mask[index],
        }
        if self.z_enc is not None:
            result["z_enc"] = self.z_enc[index]
        return result

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 3,
            "group_id": self.group_id,
            "dataset_id": self.dataset_id,
            "lineage_id": self.lineage_id,
            "checkpoint_id": self.checkpoint_id,
            "split": self.split,
            "codec": self.codec,
            "codec_fingerprint": self.codec_fingerprint,
            "identities": [asdict(item) for item in self.identities],
            "anchor_identities": None if self.anchor_identities is None else [asdict(item) for item in self.anchor_identities],
            "z_enc": self.z_enc,
            "z_task": self.z_task,
            "dataset_embedding": self.dataset_embedding,
            "dataset_embedding_bank": self.dataset_embedding_bank,
            "architecture_features": self.architecture_features,
            "tile_mask": self.tile_mask,
            "provenance": dict(self.provenance or {}),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        proposed_sha = _sha256_path(tmp)
        proposed_bytes = tmp.stat().st_size
        if path.exists():
            if path.stat().st_size != proposed_bytes or _sha256_path(path) != proposed_sha:
                tmp.unlink(missing_ok=True)
                raise FileExistsError(f"immutable grouped latent conflict: {path}")
            tmp.unlink()
        else:
            tmp.replace(path)
        manifest = {
            "schema_version": 1,
            "kind": "grouped_task_latent",
            "group_id": self.group_id,
            "codec": self.codec,
            "codec_fingerprint": self.codec_fingerprint,
            "fit_status": dict(self.provenance or {}).get("fit_status"),
            "artifact": {"path": str(path.resolve()), "sha256": proposed_sha, "bytes": proposed_bytes},
        }
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
            raise FileExistsError(f"immutable grouped latent manifest conflict: {manifest_path}")
        if not manifest_path.exists():
            temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            temporary_manifest.write_bytes(manifest_bytes)
            temporary_manifest.replace(manifest_path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "GroupedLatentRecord":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if int(payload.get("schema_version", -1)) != 3:
            raise ValueError(f"unsupported grouped latent schema in {path}")
        return cls(
            group_id=str(payload["group_id"]),
            dataset_id=str(payload["dataset_id"]),
            lineage_id=str(payload["lineage_id"]),
            checkpoint_id=str(payload["checkpoint_id"]),
            split=str(payload["split"]),
            codec=str(payload["codec"]),
            codec_fingerprint=str(payload["codec_fingerprint"]),
            identities=tuple(TileIdentity(**item) for item in payload["identities"]),
            anchor_identities=None
            if payload.get("anchor_identities") is None
            else tuple(TileIdentity(**item) for item in payload["anchor_identities"]),
            z_enc=payload.get("z_enc"),
            z_task=payload["z_task"],
            dataset_embedding=payload["dataset_embedding"],
            dataset_embedding_bank=payload["dataset_embedding_bank"],
            architecture_features=payload["architecture_features"],
            tile_mask=payload.get("tile_mask"),
            provenance=payload.get("provenance", {}),
        )


def _promote_codes(value: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{label} must be a tensor")
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3:
        raise ValueError(f"{label} must be [tiles,tokens,dim], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains NaN/inf")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class LatentNormalizer:
    """Per-dimension robust transform fit strictly on train-lineage codes."""

    center: torch.Tensor
    scale: torch.Tensor
    eps: float = 1e-6
    fit_split: str = "train"
    codec_fingerprint: str = ""

    @classmethod
    def fit(
        cls,
        codes: torch.Tensor,
        *,
        masks: torch.Tensor | None = None,
        split: str = "train",
        codec_fingerprint: str = "",
    ) -> "LatentNormalizer":
        if split != "train":
            raise ValueError("latent normalization statistics must be fit on split='train'")
        if codes.ndim < 2:
            raise ValueError("codes must have at least sample and feature dimensions")
        flat = codes.reshape(-1, codes.shape[-1]).float()
        if masks is not None:
            valid = masks.reshape(-1).to(dtype=torch.bool)
            if valid.numel() != flat.shape[0]:
                raise ValueError("mask does not align with code tokens")
            flat = flat[valid]
        if flat.shape[0] == 0:
            raise ValueError("cannot fit normalizer from zero valid codes")
        center = flat.median(dim=0).values
        q1 = torch.quantile(flat, 0.25, dim=0)
        q3 = torch.quantile(flat, 0.75, dim=0)
        robust = (q3 - q1) / 1.349
        std = flat.std(dim=0, unbiased=False)
        scale = torch.where(robust > 1e-6, robust, std).clamp_min(1e-6)
        return cls(center=center, scale=scale, fit_split=split, codec_fingerprint=codec_fingerprint)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.center.to(value)) / self.scale.to(value)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale.to(value) + self.center.to(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "center": self.center,
            "scale": self.scale,
            "eps": self.eps,
            "fit_split": self.fit_split,
            "codec_fingerprint": self.codec_fingerprint,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LatentNormalizer":
        return cls(
            center=state["center"],
            scale=state["scale"],
            eps=float(state["eps"]),
            fit_split=str(state["fit_split"]),
            codec_fingerprint=str(state.get("codec_fingerprint", "")),
        )


class GroupedLatentDataset(Dataset[dict[str, Any]]):
    """Tile-level sampler backed by grouped records and lineage-safe manifests."""

    def __init__(self, records: Sequence[GroupedLatentRecord], *, split: str, require_anchor: bool) -> None:
        selected = [record for record in records if record.split == split]
        if not selected:
            raise ValueError(f"no records for split {split!r}")
        validate_lineage_splits(records)
        self.records = selected
        bank_sizes = {int(len(record.dataset_embedding_bank)) for record in selected}
        if len(bank_sizes) != 1:
            raise ValueError("all grouped records in a split must use the same prompt-bank cardinality")
        self.prompt_bank_size = next(iter(bank_sizes))
        self.require_anchor = bool(require_anchor)
        self.index: list[tuple[int, int]] = []
        for record_idx, record in enumerate(selected):
            if require_anchor:
                record.assert_anchor_pairing()
            self.index.extend((record_idx, tile_idx) for tile_idx in range(len(record.identities)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            index, prompt_index = index
        else:
            prompt_index = 0
        record_idx, tile_idx = self.index[index]
        return self.records[record_idx].tile(tile_idx, prompt_index=prompt_index)


class DeterministicStepBatchSampler(Sampler[list[int]]):
    """Step-addressable uniform sampler for exact checkpoint resume.

    Each optimizer step has its own RNG seed, so worker prefetch cannot change the
    resumed batch stream. Sampling with replacement avoids an O(dataset_size)
    permutation per step and is uniform over the grouped dataset index.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        total_steps: int,
        seed: int,
        start_step: int = 0,
        prompt_bank_size: int = 1,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0 or total_steps <= 0:
            raise ValueError("dataset_size, batch_size, and total_steps must be positive")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.start_step = int(start_step)
        if prompt_bank_size <= 0:
            raise ValueError("prompt_bank_size must be positive")
        self.prompt_bank_size = int(prompt_bank_size)

    def set_start_step(self, step: int) -> None:
        if step < 0 or step > self.total_steps:
            raise ValueError(f"invalid sampler start step {step}")
        self.start_step = int(step)

    def __iter__(self):
        for step in range(self.start_step, self.total_steps):
            digest = hashlib.sha256(f"flow-batch:{self.seed}:{step}".encode()).digest()
            step_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(step_seed)
            indices = torch.randint(self.dataset_size, (self.batch_size,), generator=generator).tolist()
            prompts = torch.randint(self.prompt_bank_size, (self.batch_size,), generator=generator).tolist()
            yield list(zip(indices, prompts, strict=True))

    def __len__(self) -> int:
        return self.total_steps - self.start_step


def collate_latent_tiles(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    result: dict[str, Any] = {
        "group_id": [str(item["group_id"]) for item in items],
        "lineage_id": [str(item["lineage_id"]) for item in items],
        "identity": [item["identity"] for item in items],
        "anchor_identity": [item.get("anchor_identity") for item in items],
        "dataset_prompt_index": [int(item["dataset_prompt_index"]) for item in items],
        "codec": [str(item["codec"]) for item in items],
    }
    for key in ("z_task", "dataset_embedding", "architecture_features", "token_mask"):
        result[key] = torch.stack([item[key] for item in items])
    if "z_enc" in items[0]:
        if not all("z_enc" in item for item in items):
            raise ValueError("mixed anchored/unanchored examples in one batch")
        result["z_enc"] = torch.stack([item["z_enc"] for item in items])
    return result


def validate_lineage_splits(records: Iterable[GroupedLatentRecord]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        key = (record.dataset_id, record.lineage_id)
        prior = seen.setdefault(key, record.split)
        if prior != record.split:
            raise ValueError(f"lineage leakage for {key}: {prior!r} and {record.split!r}")


def validate_record_codec_fingerprints(
    records: Sequence[GroupedLatentRecord],
    *,
    expected_codec: str,
) -> str:
    if not records:
        raise ValueError("cannot validate codec fingerprint for zero records")
    codecs = {record.codec for record in records}
    if codecs != {expected_codec}:
        raise ValueError(f"record codecs {codecs} do not equal configured codec {expected_codec!r}")
    fingerprints = {record.codec_fingerprint for record in records}
    if len(fingerprints) != 1:
        raise ValueError(f"grouped latent records mix decoder codec fingerprints: {sorted(fingerprints)}")
    fingerprint = next(iter(fingerprints))
    if not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
        raise ValueError("grouped latent record codec fingerprint is not canonical SHA-256")
    return fingerprint


def validate_successful_task_fit_records(
    records: Sequence[GroupedLatentRecord],
    *,
    expected_group_ids: Sequence[str],
) -> None:
    """Fail closed unless the exact expected successful z_task inventory is present."""

    expected = tuple(str(value) for value in expected_group_ids)
    if not expected:
        raise ValueError("flow training requires a non-empty explicit expected z_task inventory")
    if len(set(expected)) != len(expected):
        raise ValueError("expected z_task inventory contains duplicate group ids")
    observed = [record.group_id for record in records]
    if len(set(observed)) != len(observed):
        raise ValueError("flow record glob contains duplicate group ids")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise ValueError(f"incomplete z_task inventory: missing={missing} unexpected={unexpected}")
    for record in records:
        provenance = dict(record.provenance or {})
        initial = provenance.get("initial_validation_loss")
        best = provenance.get("best_validation_loss")
        if provenance.get("fit_status") != "success":
            raise ValueError(f"record {record.group_id!r} is not a successful task fit")
        if not isinstance(initial, (int, float)) or not isinstance(best, (int, float)):
            raise ValueError(f"record {record.group_id!r} lacks finite task-fit losses")
        if not math.isfinite(float(initial)) or not math.isfinite(float(best)) or not float(best) < float(initial):
            raise ValueError(
                f"record {record.group_id!r} failed strict improvement: initial={initial!r}, best={best!r}"
            )


def validate_cross_codec_prompt_banks(
    ours_records: Sequence[GroupedLatentRecord],
    weightclip_records: Sequence[GroupedLatentRecord],
) -> None:
    """Require bitwise-identical prompt augmentation contracts for matched units."""

    def keyed(records: Sequence[GroupedLatentRecord]) -> dict[tuple[str, str, str], GroupedLatentRecord]:
        result = {(record.dataset_id, record.lineage_id, record.checkpoint_id): record for record in records}
        if len(result) != len(records):
            raise ValueError("duplicate statistical unit in prompt-bank inventory")
        return result

    ours = keyed(ours_records)
    weightclip = keyed(weightclip_records)
    if set(ours) != set(weightclip):
        raise ValueError("ours/WeightCLIP prompt-bank inventories contain different statistical units")
    compared_fields = (
        "candidate_count",
        "images_per_candidate",
        "seed",
        "candidate_indices",
        "indices_sha256",
        "embedding_tensor_sha256",
        "dataset_encoder",
    )
    for key in ours:
        ours_provenance = dict(ours[key].provenance or {}).get("dataset_embedding_bank_provenance", {})
        wc_provenance = dict(weightclip[key].provenance or {}).get("dataset_embedding_bank_provenance", {})
        if any(ours_provenance.get(field) != wc_provenance.get(field) for field in compared_fields):
            raise ValueError(f"ours/WeightCLIP prompt bank mismatch for {key}")
        if not torch.equal(ours[key].dataset_embedding_bank, weightclip[key].dataset_embedding_bank):
            raise ValueError(f"ours/WeightCLIP prompt embedding tensors differ for {key}")


def records_manifest(records: Sequence[GroupedLatentRecord]) -> dict[str, Any]:
    rows = [
        {
            "group_id": r.group_id,
            "dataset_id": r.dataset_id,
            "lineage_id": r.lineage_id,
            "checkpoint_id": r.checkpoint_id,
            "split": r.split,
            "codec": r.codec,
            "codec_fingerprint": r.codec_fingerprint,
            "tiles": len(r.identities),
        }
        for r in records
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": 2, "records": rows, "sha256": hashlib.sha256(canonical).hexdigest()}
