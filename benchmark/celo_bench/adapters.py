from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import OptimizerSpec
from .registry import import_object


@dataclass(frozen=True, slots=True)
class LearnedOptimizationTaskAdapter:
    name: str

    def build(self) -> Any:
        from .registry import build_task

        return build_task(self.name)


@dataclass(frozen=True, slots=True)
class ImportPathTaskAdapter:
    name: str
    import_path: str

    def build(self) -> Any:
        factory = import_object(self.import_path)
        return factory()


@dataclass(frozen=True, slots=True)
class AdamOptimizerAdapter:
    lr: float

    @property
    def name(self) -> str:
        return f"adam_lr_{self.lr:g}"

    def build(self, *, num_steps: int) -> Any:
        del num_steps
        from learned_optimization.optimizers import optax_opts

        return optax_opts.Adam(self.lr)


@dataclass(frozen=True, slots=True)
class AdamWOptimizerAdapter:
    lr: float
    weight_decay: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0

    @property
    def name(self) -> str:
        return f"adamw_lr_{self.lr:g}_wd_{self.weight_decay:g}"

    def build(self, *, num_steps: int) -> Any:
        del num_steps
        from learned_optimization.optimizers import optax_opts

        return optax_opts.AdamW(
            learning_rate=self.lr,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            eps_root=self.eps_root,
            weight_decay=self.weight_decay,
        )


@dataclass(frozen=True, slots=True)
class CeloFactoryOptimizerAdapter:
    name: str
    optimizer_name: str
    checkpoint_path: str = ""

    def build(self, *, num_steps: int) -> Any:
        del num_steps
        from celo.factory import get_optimizer

        optimizer_or_lopt = get_optimizer(self.optimizer_name)
        if hasattr(optimizer_or_lopt, "opt_fn"):
            if not self.checkpoint_path:
                raise ValueError(
                    f"Optimizer {self.optimizer_name!r} is a learned optimizer and requires checkpoint_path"
                )
            from celo.utils import init_lopt_from_ckpt

            return init_lopt_from_ckpt(optimizer_or_lopt, self.checkpoint_path)
        return optimizer_or_lopt


@dataclass(frozen=True, slots=True)
class ImportPathOptimizerAdapter:
    name: str
    import_path: str
    checkpoint_path: str = ""

    def build(self, *, num_steps: int) -> Any:
        factory = import_object(self.import_path)
        try:
            return factory(num_steps=num_steps, checkpoint_path=self.checkpoint_path)
        except TypeError:
            try:
                return factory(num_steps=num_steps)
            except TypeError:
                return factory()


def build_optimizer_adapter(spec: OptimizerSpec) -> Any:
    kind = spec.kind.strip().lower()
    if kind == "adam":
        lr = spec.metadata.get("lr") if spec.metadata else None
        if lr is None:
            raise ValueError("Adam optimizer specs must provide metadata.lr")
        return AdamOptimizerAdapter(float(lr))
    if kind == "adamw":
        metadata = dict(spec.metadata or {})
        lr = metadata.get("lr")
        if lr is None:
            raise ValueError("AdamW optimizer specs must provide metadata.lr")
        return AdamWOptimizerAdapter(
            lr=float(lr),
            weight_decay=float(metadata.get("weight_decay", 1e-4)),
            b1=float(metadata.get("b1", 0.9)),
            b2=float(metadata.get("b2", 0.999)),
            eps=float(metadata.get("eps", 1e-8)),
            eps_root=float(metadata.get("eps_root", 0.0)),
        )
    if kind in {"celo_factory", "celo"}:
        optimizer_name = spec.optimizer_name or spec.name
        return CeloFactoryOptimizerAdapter(
            name=spec.name,
            optimizer_name=optimizer_name,
            checkpoint_path=spec.checkpoint_path,
        )
    if kind in {"import_path", "callable"}:
        if not spec.import_path:
            raise ValueError(f"Optimizer spec {spec.name!r} requires import_path")
        return ImportPathOptimizerAdapter(
            name=spec.name,
            import_path=spec.import_path,
            checkpoint_path=spec.checkpoint_path,
        )
    raise ValueError(f"Unsupported optimizer adapter kind: {spec.kind!r}")
