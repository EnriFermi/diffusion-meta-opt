from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


CELO_PAPER_17_TASKS: tuple[str, ...] = (
    "ImageMLP_FashionMnist_Relu128x128",
    "ImageMLP_Cifar10_128x128x128_LayerNorm_Relu",
    "ImageMLP_Cifar10_128x128x128_Tanh_bs128",
    "Conv_Cifar10_32x64x64",
    "Conv_Cifar10_32x64x64_batchnorm",
    "Conv_Cifar10_32x64x64_layernorm",
    "Conv_Cifar100_32x64x64",
    "VIT_Cifar100_wideshallow",
    "VIT_Cifar100_skinnydeep",
    "TransformerLM_LM1B_MultiRuntime_0",
    "TransformerLM_LM1B_MultiRuntime_2",
    "TransformerLM_LM1B_MultiRuntime_5",
    "ImageMLPAE_Cifar10_128x32x128_bs256",
    "ImageMLPAE_Mnist_128x32x128_bs128",
    "RNNLM_lm1b32k_Patch32_LSTM256_Embed128",
    "RNNLM_wikipediaen32k_Patch32_LSTM256_Embed128",
    "LOpt_AdafacMLPLOpt_FashionMnist_50",
)

ADAM_LR_GRID: tuple[float, ...] = (
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    run_label: str = "celo_paper_17"
    artifact_root: str = "./artifacts/celo_bench"
    run_dir: str = ""
    cache_first: bool = True
    force_rerun: bool = False
    continue_on_task_error: bool = False
    show_progress: bool = True
    log_level: str = "INFO"
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 0
    xla_preallocate: str = "false"
    tensorflow_hide_gpus: bool = True
    tfds_try_gcs_for_wikipedia: bool = True


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    steps: int = 2000
    seeds: tuple[int, ...] = (0, 1, 2)
    eval_every: int = 10
    eval_batches: int = 5
    last_eval_batches: int = 10
    metrics_every: int = 10
    score_split: str = "eval/train/loss"
    ema_alpha: float = 0.9


@dataclass(frozen=True, slots=True)
class AdamReferenceConfig:
    enabled: bool = True
    lrs: tuple[float, ...] = ADAM_LR_GRID


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    name: str
    kind: str = "celo_factory"
    optimizer_name: str = ""
    checkpoint_path: str = ""
    import_path: str = ""
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CeloBenchConfig:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    adam_reference: AdamReferenceConfig = field(default_factory=AdamReferenceConfig)
    task_names: tuple[str, ...] = CELO_PAPER_17_TASKS
    methods: tuple[OptimizerSpec, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CeloBenchConfig":
        data = _plain_mapping(raw)
        benchmark = BenchmarkConfig(**_known_kwargs(BenchmarkConfig, data.get("benchmark", {})))
        runtime = RuntimeConfig(**_known_kwargs(RuntimeConfig, data.get("runtime", {})))
        evaluation_kwargs = _known_kwargs(EvaluationConfig, data.get("evaluation", {}))
        if "seeds" in evaluation_kwargs:
            evaluation_kwargs["seeds"] = tuple(int(seed) for seed in _as_list(evaluation_kwargs["seeds"]))
        evaluation = EvaluationConfig(**evaluation_kwargs)

        adam_kwargs = _known_kwargs(AdamReferenceConfig, data.get("adam_reference", {}))
        if "lrs" in adam_kwargs:
            adam_kwargs["lrs"] = tuple(float(lr) for lr in _as_list(adam_kwargs["lrs"]))
        adam_reference = AdamReferenceConfig(**adam_kwargs)

        tasks_cfg = data.get("tasks", {})
        if isinstance(tasks_cfg, Mapping):
            preset = str(tasks_cfg.get("preset", "celo_paper_17"))
            enabled = tasks_cfg.get("enabled", None)
            if enabled is None and preset == "celo_paper_17":
                task_names = CELO_PAPER_17_TASKS
            elif enabled is None:
                raise ValueError(f"Unknown task preset without explicit tasks.enabled: {preset!r}")
            else:
                task_names = tuple(str(name) for name in _as_list(enabled))
        else:
            task_names = tuple(str(name) for name in _as_list(tasks_cfg))

        methods_cfg = data.get("methods", {})
        raw_methods: Sequence[Any]
        if isinstance(methods_cfg, Mapping):
            raw_methods = _as_list(methods_cfg.get("items", []))
        else:
            raw_methods = _as_list(methods_cfg)
        methods = tuple(_optimizer_spec_from_mapping(item) for item in raw_methods)

        return cls(
            benchmark=benchmark,
            runtime=runtime,
            evaluation=evaluation,
            adam_reference=adam_reference,
            task_names=task_names,
            methods=methods,
        )


def config_to_dict(cfg: CeloBenchConfig) -> dict[str, Any]:
    return _jsonable(asdict(cfg))


def benchmark_relevant_config(cfg: CeloBenchConfig) -> dict[str, Any]:
    payload = config_to_dict(cfg)
    benchmark = payload.get("benchmark", {})
    for key in ("run_label", "artifact_root", "run_dir", "cache_first", "force_rerun", "show_progress", "log_level"):
        benchmark.pop(key, None)
    payload["benchmark"] = benchmark
    return payload


def config_hash(cfg: CeloBenchConfig) -> str:
    payload = json.dumps(benchmark_relevant_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_config_json(path: str | Path) -> CeloBenchConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return CeloBenchConfig.from_mapping(payload)


def _optimizer_spec_from_mapping(value: Any) -> OptimizerSpec:
    if isinstance(value, OptimizerSpec):
        return value
    if isinstance(value, str):
        return OptimizerSpec(name=value, optimizer_name=value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Optimizer spec must be a mapping or string, got {type(value)!r}")
    kwargs = _known_kwargs(OptimizerSpec, value)
    if not str(kwargs.get("name", "")).strip():
        inferred = str(kwargs.get("optimizer_name") or kwargs.get("import_path") or kwargs.get("kind") or "").strip()
        if not inferred:
            raise ValueError("Optimizer spec requires a non-empty name")
        kwargs["name"] = inferred
    if "metadata" in kwargs and kwargs["metadata"] is None:
        kwargs["metadata"] = {}
    return OptimizerSpec(**kwargs)


def _known_kwargs(cls: type[Any], raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{cls.__name__} config must be a mapping, got {type(raw)!r}")
    field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {str(key): _plain_value(value) for key, value in raw.items() if str(key) in field_names}


def _plain_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in raw.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return copy.deepcopy(value)
