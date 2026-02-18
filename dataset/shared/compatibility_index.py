from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dataset.data_raw.core.config import to_plain_dict

LOGGER = logging.getLogger(__name__)


class CompatibilityIndex:
    """Build and query compatibility graph: model -> datasets."""

    def __init__(self, cfg: DictConfig | dict[str, Any]) -> None:
        self.cfg = cfg
        self.cfg_dict = to_plain_dict(cfg)

        self.project_root = Path(__file__).resolve().parents[2]

        self.data_cfg = _resolve_data_section(self.cfg_dict)
        self.collector_cfg = self.cfg_dict.get("collector", {})
        self.train_cfg = self.cfg_dict.get("train", {})

        self.collector_mode = resolve_collector_mode(self.cfg_dict)
        self.collector_device = normalize_device(self.collector_cfg.get("device"))

        self.dataset_cfgs = self._load_enabled_dataset_cfgs()
        self.model_cfgs = self._load_model_cfgs()

        self.model_to_datasets: dict[str, list[str]] = {}
        self.model_to_dataset_weights: dict[str, list[float]] = {}
        self.model_weights: dict[str, float] = {}

        self._build_index()

    def get_models(self) -> list[str]:
        return sorted(self.model_to_datasets.keys())

    def get_datasets_for_model(self, model_name: str) -> list[str]:
        return list(self.model_to_datasets.get(model_name, []))

    def get_dataset_weights_for_model(self, model_name: str) -> list[float]:
        return list(self.model_to_dataset_weights.get(model_name, []))

    def get_model_weights(self) -> dict[str, float]:
        return dict(self.model_weights)

    def get_dataset_cfg(self, dataset_name: str) -> dict[str, Any]:
        return dict(self.dataset_cfgs[dataset_name])

    def get_model_cfg(self, model_name: str) -> dict[str, Any]:
        return dict(self.model_cfgs[model_name])

    def get_model_cfgs(self) -> dict[str, dict[str, Any]]:
        return {name: dict(cfg) for name, cfg in self.model_cfgs.items()}

    def _build_index(self) -> None:
        for dataset_name, ds_cfg in self.dataset_cfgs.items():
            if not bool(ds_cfg.get("enabled", True)):
                continue
            if not self._dataset_is_eligible_for_mode(ds_cfg):
                continue

            models = [str(name) for name in ds_cfg.get("models", [])]
            if not models:
                continue

            ds_weight = float(ds_cfg.get("sampling_weight", 1.0))

            for model_name in models:
                if model_name not in self.model_cfgs:
                    raise FileNotFoundError(
                        f"Model '{model_name}' referenced by dataset '{dataset_name}' has no config"
                    )

                self.model_to_datasets.setdefault(model_name, []).append(dataset_name)
                self.model_to_dataset_weights.setdefault(model_name, []).append(ds_weight)

        for model_name in self.model_to_datasets:
            model_cfg = self.model_cfgs.get(model_name, {})
            self.model_weights[model_name] = float(model_cfg.get("sampling_weight", 1.0))

    def _dataset_is_eligible_for_mode(self, ds_cfg: dict[str, Any]) -> bool:
        if self.collector_mode != "async":
            return True

        requested_device = normalize_device(ds_cfg.get("collector_device"))
        effective_device = requested_device if requested_device is not None else self.collector_device

        if effective_device is None:
            return True

        if self.collector_device is None:
            return True

        if effective_device != self.collector_device:
            LOGGER.info(
                "Skipping dataset '%s' for async collector: effective collector_device=%s != %s",
                ds_cfg.get("name"),
                effective_device,
                self.collector_device,
            )
            return False

        return True

    def _load_enabled_dataset_cfgs(self) -> dict[str, dict[str, Any]]:
        enabled_names = [str(name) for name in self.data_cfg.get("enabled_datasets", [])]
        if not enabled_names:
            raise ValueError("data.enabled_datasets is empty")

        config_dirs = self._dataset_config_dirs()
        dataset_overrides = self.data_cfg.get("dataset_overrides", {})
        if dataset_overrides is None:
            dataset_overrides = {}
        if not isinstance(dataset_overrides, dict):
            raise TypeError("data.dataset_overrides must be a mapping")

        resolved: dict[str, dict[str, Any]] = {}
        for dataset_name in enabled_names:
            cfg = self._load_cfg_file(dataset_name=dataset_name, config_dirs=config_dirs)
            if cfg is None:
                raise FileNotFoundError(f"Dataset config not found for '{dataset_name}' in {config_dirs}")

            override_cfg = dataset_overrides.get(dataset_name)
            if isinstance(override_cfg, dict):
                cleaned = _drop_empty_values(override_cfg)
                if cleaned:
                    cfg = _deep_merge(cfg, cleaned)

            cfg.setdefault("name", dataset_name)
            resolved[dataset_name] = cfg

        return resolved

    def _load_model_cfgs(self) -> dict[str, dict[str, Any]]:
        cfg_dirs = self._model_config_dirs()

        model_cfgs: dict[str, dict[str, Any]] = {}
        for cfg_dir in cfg_dirs:
            if not cfg_dir.exists():
                continue
            for path in sorted(cfg_dir.glob("*.yaml")):
                payload = to_plain_dict(OmegaConf.load(path))
                name = str(payload.get("name") or path.stem)
                payload["name"] = name
                model_cfgs[name] = payload

        if not model_cfgs:
            raise FileNotFoundError(f"No model configs found in {cfg_dirs}")

        return model_cfgs

    def _dataset_config_dirs(self) -> list[Path]:
        configured = self.data_cfg.get("dataset_config_dirs")
        if configured is None:
            configured = ["conf/data/datasets"]

        if isinstance(configured, str):
            configured = [configured]

        dirs: list[Path] = []
        for item in configured:
            path = Path(str(item))
            if not path.is_absolute():
                path = self.project_root / path
            dirs.append(path)

        return dirs

    def _model_config_dirs(self) -> list[Path]:
        models_cfg = self.cfg_dict.get("models", {})
        configured = models_cfg.get("model_config_dirs")
        if configured is None:
            configured = ["conf/data/models"]

        if isinstance(configured, str):
            configured = [configured]

        dirs: list[Path] = []
        for item in configured:
            path = Path(str(item))
            if not path.is_absolute():
                path = self.project_root / path
            dirs.append(path)

        return dirs

    @staticmethod
    def _load_cfg_file(dataset_name: str, config_dirs: list[Path]) -> dict[str, Any] | None:
        for config_dir in config_dirs:
            candidate = config_dir / f"{dataset_name}.yaml"
            if candidate.exists():
                return to_plain_dict(OmegaConf.load(candidate))
        return None


def resolve_collector_mode(cfg: dict[str, Any]) -> str:
    collector_cfg = cfg.get("collector", {})
    mode = str(collector_cfg.get("mode", "auto")).lower()

    if mode in {"async", "interleaved"}:
        return mode

    collector_device = normalize_device(collector_cfg.get("device"))
    train_device = resolve_train_device(cfg)

    if collector_device is None:
        return "interleaved"
    if train_device is None:
        return "interleaved"
    if collector_device == train_device:
        return "interleaved"
    return "async"


def resolve_train_device(cfg: dict[str, Any]) -> str | None:
    mini_train_cfg = cfg.get("mini_train")
    if isinstance(mini_train_cfg, dict):
        mini_train_device = normalize_device(mini_train_cfg.get("device"))
        if mini_train_device is not None:
            return mini_train_device

    train_cfg = cfg.get("train")
    if isinstance(train_cfg, dict):
        return normalize_device(train_cfg.get("device"))

    return None


def normalize_device(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    return text


def _resolve_data_section(cfg: dict[str, Any]) -> dict[str, Any]:
    if "data" in cfg and isinstance(cfg["data"], dict):
        return cfg["data"]
    raise KeyError("Top-level config must define 'data'")


def _drop_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, nested in value.items():
            cleaned = _drop_empty_values(nested)
            if cleaned is None:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            if isinstance(cleaned, list) and not cleaned:
                continue
            output[key] = cleaned
        return output

    if isinstance(value, list):
        output_list = [_drop_empty_values(item) for item in value]
        return [item for item in output_list if item is not None]

    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if key in output and isinstance(output[key], dict) and isinstance(value, dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = value
    return output
