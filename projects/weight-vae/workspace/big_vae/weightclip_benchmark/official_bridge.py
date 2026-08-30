"""Verified, lazy bridge to the unmodified official WeightCLIP checkout."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .contract import (
    DEFAULT_CONTRACT,
    OFFICIAL_RESNET_CHECKPOINT,
    OFFICIAL_RESNET_DATASET_ENCODER,
    OfficialArtifactSpec,
    WeightCLIPBenchmarkContract,
)


_SECRET_KEY = re.compile(r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)", re.I)
_SECRET_VALUE = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,})")


def redact_secrets(value: Any) -> Any:
    """Recursively redact common credential fields before logging artifacts."""

    if isinstance(value, Mapping):
        return {
            str(key): ("<redacted>" if _SECRET_KEY.search(str(key)) else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("<redacted>", value)
    return value


def sha256_file(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedArtifact:
    path: Path
    sha256: str
    size_bytes: int
    source: str


@dataclass
class LoadedOfficialBundle:
    weight_model: Any
    trainset: Any
    valset: Any
    tokenizer: Any
    config: dict[str, Any]
    dataset_encoder: Any
    official_utils: Any
    official_entrypoint: Any
    provenance: dict[str, Any]


class OfficialWeightCLIPBridge:
    """Load exact official code and artifacts only after provenance checks.

    Artifact download is opt-in because the released ResNet checkpoint is over
    9 GB.  The small dataset encoder can use the same pinned resolver.
    """

    def __init__(
        self,
        official_repo: str | Path,
        cache_dir: str | Path,
        *,
        device: str = "cpu",
        contract: WeightCLIPBenchmarkContract = DEFAULT_CONTRACT,
        verbose: bool = True,
    ) -> None:
        self.repo = Path(official_repo).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.device = str(device)
        self.contract = contract
        self.verbose = bool(verbose)

    def verify_checkout(self) -> dict[str, Any]:
        self.contract.validate()
        if not (self.repo / ".git").exists():
            raise FileNotFoundError(f"Official WeightCLIP checkout is not a git repository: {self.repo}")
        head = _git(self.repo, "rev-parse", "HEAD")
        if head != self.contract.official_git_commit:
            raise RuntimeError(
                "Official WeightCLIP commit mismatch: "
                f"checkout={head}, required={self.contract.official_git_commit}"
            )
        remote = _git(self.repo, "remote", "get-url", "origin")
        normalized = remote.removesuffix(".git").rstrip("/").lower()
        required = self.contract.official_git_url.removesuffix(".git").rstrip("/").lower()
        if normalized != required:
            raise RuntimeError(f"Official WeightCLIP origin mismatch: checkout={remote}, required={self.contract.official_git_url}")
        return {"repo": str(self.repo), "git_commit": head, "git_origin": remote}

    def resolve_artifact(
        self,
        spec: OfficialArtifactSpec,
        *,
        explicit_path: str | Path | None = None,
        allow_download: bool = False,
    ) -> ResolvedArtifact:
        if explicit_path is not None:
            path = Path(explicit_path).expanduser().resolve()
            source = "explicit"
        else:
            path = self.cache_dir / self.contract.official_hf_repo.replace("/", "--") / self.contract.official_hf_revision / spec.path
            source = "content_addressed_cache"
        if not path.exists() and allow_download:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("huggingface_hub is required for artifact download") from exc
            downloaded = hf_hub_download(
                repo_id=self.contract.official_hf_repo,
                filename=spec.path,
                revision=self.contract.official_hf_revision,
                local_dir=str(path.parents[len(Path(spec.path).parts) - 1]),
            )
            path = Path(downloaded).resolve()
            source = "huggingface_pinned_download"
        if not path.exists():
            raise FileNotFoundError(
                f"Official artifact is absent: {path}. Pass an explicit path or allow_download=True."
            )
        size = path.stat().st_size
        if size != spec.size_bytes:
            raise RuntimeError(f"Artifact size mismatch for {path}: {size} != {spec.size_bytes}")
        digest = sha256_file(path)
        if digest != spec.sha256:
            raise RuntimeError(f"Artifact SHA-256 mismatch for {path}: {digest} != {spec.sha256}")
        if self.verbose:
            print(f"[weightclip:artifact] source={source} path={path} bytes={size} sha256={digest}")
        return ResolvedArtifact(path=path, sha256=digest, size_bytes=size, source=source)

    def load(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        dataset_encoder_path: str | Path | None = None,
        allow_download: bool = False,
        checkpoint_data_config: Mapping[str, Any] | None = None,
    ) -> LoadedOfficialBundle:
        checkout = self.verify_checkout()
        weight = self.resolve_artifact(
            OFFICIAL_RESNET_CHECKPOINT,
            explicit_path=checkpoint_path,
            allow_download=allow_download,
        )
        dataset_encoder_artifact = self.resolve_artifact(
            OFFICIAL_RESNET_DATASET_ENCODER,
            explicit_path=dataset_encoder_path,
            allow_download=allow_download,
        )
        utils, entrypoint = self._import_official_modules()
        if self.verbose:
            print(f"[weightclip:load] device={self.device} stage=weight_model path={weight.path}")
        if checkpoint_data_config is None:
            sane_model, trainset, valset, tokenizer, config = utils.load_sane_bundle(weight.path, self.device)
        else:
            sane_model, trainset, valset, tokenizer, config = _load_bundle_with_data_override(
                utils,
                weight.path,
                self.device,
                checkpoint_data_config,
            )
        if self.verbose:
            print(f"[weightclip:load] device={self.device} stage=dataset_encoder path={dataset_encoder_artifact.path}")
        dataset_cfg = dict(config.get("dataset_encoder", {}))
        dataset_encoder = utils.load_dataset_encoder(dataset_encoder_artifact.path, self.device, dataset_cfg)
        _freeze_eval(sane_model)
        _freeze_eval(dataset_encoder)
        provenance = {
            **checkout,
            "hf_repo": self.contract.official_hf_repo,
            "hf_revision": self.contract.official_hf_revision,
            "contract_version": self.contract.version,
            "contract_fingerprint": self.contract.fingerprint(),
            "checkpoint": dataclass_dict(weight),
            "dataset_encoder": dataclass_dict(dataset_encoder_artifact),
            "parameters": {
                "weight_model": _parameter_ledger(sane_model),
                "dataset_encoder": _parameter_ledger(dataset_encoder),
            },
            "device": self.device,
            "config": redact_secrets(config),
        }
        return LoadedOfficialBundle(
            weight_model=sane_model,
            trainset=trainset,
            valset=valset,
            tokenizer=tokenizer,
            config=config,
            dataset_encoder=dataset_encoder,
            official_utils=utils,
            official_entrypoint=entrypoint,
            provenance=provenance,
        )

    def load_codec_tokenizer_only(
        self,
        *,
        reference_state: Mapping[str, Any],
        reference_state_sha256: str,
        checkpoint_path: str | Path | None = None,
        dataset_encoder_path: str | Path | None = None,
        allow_download: bool = False,
    ) -> LoadedOfficialBundle:
        """Load the released codec without touching unreleased zoo paths.

        The sparse tokenizer is stateless apart from its exact checkpoint
        reference. Its class/config come from the hash-bound official 9 GB
        checkpoint; positions/masks are then computed from the supplied
        hash-bound zoo checkpoint by the unchanged official tokenizer code.
        """

        import torch

        if len(reference_state_sha256) != 64 or not reference_state:
            raise ValueError("tokenizer-only load requires a SHA-bound nonempty reference state")
        checkout = self.verify_checkout()
        weight = self.resolve_artifact(
            OFFICIAL_RESNET_CHECKPOINT, explicit_path=checkpoint_path, allow_download=allow_download
        )
        encoder_artifact = self.resolve_artifact(
            OFFICIAL_RESNET_DATASET_ENCODER, explicit_path=dataset_encoder_path, allow_download=allow_download
        )
        utils, entrypoint = self._import_official_modules()
        try:
            state = torch.load(weight.path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:  # pragma: no cover - older supported torch
            state = torch.load(weight.path, map_location="cpu", weights_only=False)
        config = copy.deepcopy(state.get("config") or utils._load_trial_config(weight.path.parent.parent))
        model_cfg = copy.deepcopy(config["model"])
        model_cfg["device"] = self.device
        if model_cfg.get("n_tokens") == "auto" or model_cfg.get("max_positions") == "auto":
            raise ValueError("tokenizer-only load refuses unbound auto model geometry")
        model = utils.SANEAutoEncoder.from_config(model_cfg)
        model_state = state.get("model") or state.get("state_dict") or state
        if any(key.startswith("module.") for key in model_state):
            model_state = {key.removeprefix("module."): value for key, value in model_state.items()}
        model.load_state_dict(model_state, strict=True)
        model.to(self.device).eval()
        tokenizer_cfg = utils.from_dict(utils.TokenizerConfig, copy.deepcopy(config["data"]["tokenizer"]))
        tokenizers = importlib.import_module("sane.data.tokenizers")
        tokenizer_cls = {"sparse": tokenizers.SparseTokenizer, "dense": tokenizers.DenseTokenizer}[tokenizer_cfg.tokenizer_class]
        tokenizer = tokenizer_cls(
            mode=tokenizer_cfg.mode,
            tokensize=tokenizer_cfg.tokensize,
            device=self.device,
            reference_statedict=dict(reference_state),
            padding=tokenizer_cfg.padding,
        )
        dataset_cfg = dict(config.get("dataset_encoder", {}))
        dataset_encoder = utils.load_dataset_encoder(encoder_artifact.path, self.device, dataset_cfg)
        _freeze_eval(model)
        _freeze_eval(dataset_encoder)
        provenance = {
            **checkout,
            "hf_repo": self.contract.official_hf_repo,
            "hf_revision": self.contract.official_hf_revision,
            "contract_version": self.contract.version,
            "contract_fingerprint": self.contract.fingerprint(),
            "checkpoint": dataclass_dict(weight),
            "dataset_encoder": dataclass_dict(encoder_artifact),
            "parameters": {"weight_model": _parameter_ledger(model), "dataset_encoder": _parameter_ledger(dataset_encoder)},
            "device": self.device,
            "config": redact_secrets(config),
            "tokenizer_reconstruction": "embedded_hash_bound_config_plus_hash_bound_reference_state",
            "reference_state_sha256": reference_state_sha256,
            "unreleased_zoo_accessed": False,
        }
        return LoadedOfficialBundle(
            weight_model=model,
            trainset=None,
            valset=None,
            tokenizer=tokenizer,
            config=config,
            dataset_encoder=dataset_encoder,
            official_utils=utils,
            official_entrypoint=entrypoint,
            provenance=provenance,
        )

    def retarget_tokenizer(
        self,
        bundle: LoadedOfficialBundle,
        *,
        reference_state: Mapping[str, Any],
        reference_state_sha256: str,
    ) -> LoadedOfficialBundle:
        """Reuse the 9 GB frozen model and retarget only its stateless tokenizer."""

        if bundle.provenance.get("tokenizer_reconstruction") != "embedded_hash_bound_config_plus_hash_bound_reference_state":
            raise ValueError("retargeting requires a tokenizer-only official bundle")
        if len(reference_state_sha256) != 64 or not reference_state:
            raise ValueError("retargeting requires a SHA-bound reference state")
        utils, _ = self._import_official_modules()
        tokenizer_cfg = utils.from_dict(utils.TokenizerConfig, copy.deepcopy(bundle.config["data"]["tokenizer"]))
        tokenizers = importlib.import_module("sane.data.tokenizers")
        tokenizer_cls = {"sparse": tokenizers.SparseTokenizer, "dense": tokenizers.DenseTokenizer}[tokenizer_cfg.tokenizer_class]
        tokenizer = tokenizer_cls(
            mode=tokenizer_cfg.mode,
            tokensize=tokenizer_cfg.tokensize,
            device=self.device,
            reference_statedict=dict(reference_state),
            padding=tokenizer_cfg.padding,
        )
        provenance = dict(bundle.provenance) | {"reference_state_sha256": reference_state_sha256}
        return LoadedOfficialBundle(
            weight_model=bundle.weight_model,
            trainset=None,
            valset=None,
            tokenizer=tokenizer,
            config=bundle.config,
            dataset_encoder=bundle.dataset_encoder,
            official_utils=bundle.official_utils,
            official_entrypoint=bundle.official_entrypoint,
            provenance=provenance,
        )

    def load_dataset_encoder_only(
        self,
        *,
        dataset_encoder_path: str | Path | None = None,
        allow_download: bool = False,
        encoder_config: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Load the 3.76 MB frozen encoder without resolving the 9 GB codec."""

        checkout = self.verify_checkout()
        artifact = self.resolve_artifact(
            OFFICIAL_RESNET_DATASET_ENCODER,
            explicit_path=dataset_encoder_path,
            allow_download=allow_download,
        )
        utils, _ = self._import_official_modules()
        config = {
            "preset": "deepsets_conv",
            "embedding_dim": 192,
            "input_channels": 3,
            "normalize_embeddings": True,
            "embedding_scale": 4,
        }
        config.update(dict(encoder_config or {}))
        model = utils.load_dataset_encoder(artifact.path, self.device, config)
        _freeze_eval(model)
        provenance = {
            **checkout,
            "hf_repo": self.contract.official_hf_repo,
            "hf_revision": self.contract.official_hf_revision,
            "contract_fingerprint": self.contract.fingerprint(),
            "dataset_encoder": dataclass_dict(artifact),
            "dataset_encoder_config": config,
            "parameters": _parameter_ledger(model),
            "device": self.device,
        }
        return model, provenance

    def encode_dataset_prompt(self, bundle: LoadedOfficialBundle, image_sets: Any) -> Any:
        """Encode `[B,10,3,32,32]` official-normalized training-image sets."""

        shape = tuple(int(x) for x in image_sets.shape)
        required = self.contract.access.dataset_prompt_images
        if len(shape) != 5 or shape[1:] != (required, 3, 32, 32):
            raise ValueError(f"Expected dataset prompt shape [B,{required},3,32,32], got {shape}")
        if not bool(image_sets.is_floating_point()):
            raise TypeError("Dataset prompt tensor must be floating point")
        minimum = float(image_sets.min().detach().cpu())
        maximum = float(image_sets.max().detach().cpu())
        if minimum < -1.0001 or maximum > 1.0001:
            raise ValueError(f"Dataset prompt must use official [-1,1] normalization, got [{minimum},{maximum}]")
        image_sets = image_sets.to(self.device)
        with _torch_inference_mode():
            if hasattr(bundle.dataset_encoder, "encode"):
                return bundle.dataset_encoder.encode(image_sets)
            return bundle.dataset_encoder.get_embeddings(image_sets)

    def predict_weight_embeddings(
        self,
        bundle: LoadedOfficialBundle,
        mode: str,
        dataset_embeddings: Any,
        *,
        training_dataset_embeddings: Any,
        training_weight_embeddings: Any,
        direct_decoder_state: Mapping[str, Any] | None = None,
        memory_bank_mapper: Any | None = None,
        neighbour_metric: str = "cosine",
        memory_temperature: float = 0.5,
        memory_top_k: int = 32,
        sample_batch_size: int = 50,
    ) -> tuple[Any, dict[str, Any]]:
        """Expose official direct, memory-bank, and neighbour mapper modes."""

        mode = {"nearest": "neighbour", "neighbor": "neighbour"}.get(mode, mode)
        entry = bundle.official_entrypoint
        if mode == "direct_decode":
            state = direct_decoder_state or entry.fit_direct_decoder(
                training_dataset_embeddings,
                training_weight_embeddings,
                device=self.device,
            )
            predicted = entry.apply_direct_decoder(dataset_embeddings, state, device=self.device)
            return predicted, {"mode": mode, "mapper_source": "ridge_fit", "state": state}
        if mode == "neighbour":
            predicted, stats = entry.retrieve_neighbour_embeddings(
                dataset_embeddings,
                training_dataset_embeddings,
                training_weight_embeddings,
                metric=neighbour_metric,
            )
            return predicted, {"mode": mode, "metric": neighbour_metric, **stats}
        if mode == "memory_bank":
            if memory_bank_mapper is None:
                raise ValueError("memory_bank mode requires a trained/cached official memory_bank_mapper")
            predicted = entry._sample_memory_bank_embeddings(
                memory_bank_mapper,
                dataset_embeddings,
                sample_batch_size,
                self.device,
                memory_temperature,
                memory_top_k,
            )
            return predicted, {
                "mode": mode,
                "temperature": float(memory_temperature),
                "top_k_cap": int(memory_top_k),
            }
        raise ValueError(f"Unsupported official mapper mode: {mode!r}")

    def train_memory_bank_mapper(
        self,
        bundle: LoadedOfficialBundle,
        training_dataset_embeddings: Any,
        training_weight_embeddings: Any,
        **official_kwargs: Any,
    ) -> tuple[Any, float, list[Any]]:
        """Delegate mapper fitting to the pinned official implementation."""

        return bundle.official_utils.train_memory_bank_translator(
            training_dataset_embeddings,
            training_weight_embeddings,
            device=self.device,
            **official_kwargs,
        )

    def decode_weight_embedding(self, bundle: LoadedOfficialBundle, embedding: Any, **kwargs: Any) -> tuple[Any, Any]:
        return bundle.official_utils.decode_embedding(bundle.weight_model, embedding, tokenizer=bundle.tokenizer, device=self.device, **kwargs)

    def write_provenance(self, bundle: LoadedOfficialBundle, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(redact_secrets(bundle.provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.verbose:
            print(f"[weightclip:output] provenance={output}")
        return output

    def _import_official_modules(self) -> tuple[Any, Any]:
        self.verify_checkout()
        for module_name in ("ood_utils", "dataset_to_model", "sane"):
            existing = sys.modules.get(module_name)
            existing_file = getattr(existing, "__file__", None)
            if existing_file is not None and self.repo not in Path(existing_file).resolve().parents:
                raise RuntimeError(
                    f"Refusing mixed official-code provenance: {module_name} is already loaded from {existing_file}"
                )
        with _official_import_path(self.repo):
            utils = importlib.import_module("ood_utils")
            entrypoint = importlib.import_module("dataset_to_model")
        module_file = Path(utils.__file__).resolve()
        if self.repo not in module_file.parents:
            raise RuntimeError(f"Imported ood_utils from an unverified path: {module_file}")
        return utils, entrypoint


def dataclass_dict(value: ResolvedArtifact) -> dict[str, Any]:
    return {
        "path": str(value.path),
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "source": value.source,
    }


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@contextlib.contextmanager
def _official_import_path(repo: Path) -> Iterator[None]:
    additions = [str(repo / "scripts"), str(repo / "src"), str(repo)]
    old = list(sys.path)
    sys.path[:0] = [item for item in additions if item not in sys.path]
    try:
        yield
    finally:
        sys.path[:] = old


@contextlib.contextmanager
def _torch_inference_mode() -> Iterator[None]:
    import torch

    with torch.inference_mode():
        yield


def _freeze_eval(module: Any) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _parameter_ledger(module: Any) -> dict[str, int]:
    total = sum(int(parameter.numel()) for parameter in module.parameters())
    trainable = sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)
    return {"total": total, "trainable_after_load": trainable, "frozen": total - trainable}


def _load_bundle_with_data_override(
    utils: Any,
    checkpoint_path: Path,
    device: str,
    data_config: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Official loader with only the unreleased-zoo roots explicitly replaced.

    The released checkpoint embeds machine-local zoo roots.  The official
    loader needs a token dataset merely to reconstruct the tokenizer.  This
    compatibility path executes the same official factory/model constructors
    while replacing the complete data block with a manifest-derived one.
    """

    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(state.get("config") or utils._load_trial_config(checkpoint_path.parent.parent))
    config["data"] = copy.deepcopy(dict(data_config))
    data_cfg = copy.deepcopy(config["data"])
    tokens_cfg = copy.deepcopy(data_cfg["tokens_dataset"])
    tokens_cfg["data_config_for_hash"] = copy.deepcopy(data_cfg)
    tokens_cfg["read_only"] = True
    factory_cfg = utils.DatasetFactoryConfig(
        checkpoints_dataset=utils._resolve_chkpt_cfg(data_cfg["checkpoints_dataset"]),
        tokenizer=utils.from_dict(utils.TokenizerConfig, data_cfg["tokenizer"]),
        tokens_dataset=utils.from_dict(utils.CachedWindowedDatasetConfig, tokens_cfg),
        split=utils.from_dict(utils.SplitConfig, data_cfg["split"]),
        dataloader=utils.from_dict(utils.DataLoaderConfig, data_cfg),
        splitter=data_cfg.get("splitter", "RandomSplitter"),
    )
    trainloader, valloader, testloader = utils.DatasetFactory().build(factory_cfg)
    trainset = trainloader.dataset
    valset = valloader.dataset if valloader is not None else None
    testset = testloader.dataset if testloader is not None else None
    model_cfg = copy.deepcopy(config["model"])
    model_cfg["device"] = device
    if model_cfg.get("n_tokens") == "auto":
        model_cfg["n_tokens"] = utils._infer_n_tokens(trainloader)
    if model_cfg.get("max_positions") == "auto":
        model_cfg["max_positions"] = utils._infer_max_positions(trainloader)
    model = utils.SANEAutoEncoder.from_config(model_cfg)
    model_state = state.get("model") or state.get("state_dict") or state
    if any(key.startswith("module.") for key in model_state):
        model_state = {key.removeprefix("module."): value for key, value in model_state.items()}
    model.load_state_dict(model_state)
    model.to(device).eval()
    tokenizer = (valset or testset or trainset).tokenizer
    return model, trainset, valset, tokenizer, config
