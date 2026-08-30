"""Common downstream evaluator for WeightCLIP and generated initializations."""

from __future__ import annotations

import random
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contract import CandidateProtocol, DEFAULT_CONTRACT, WeightCLIPBenchmarkContract


class HeadPolicy(str, Enum):
    FRESH_DEFAULT = "fresh_default_random_paired_seed"
    NATIVE_RELEASED = "native_released_decoded_head"
    PRESERVE_SOURCE = "preserve_complete_source_head"


class BatchNormPolicy(str, Enum):
    OURS_DEFAULT = "default_affine_and_running_then_common_calibration"
    WEIGHTCLIP_NATIVE = "released_affine_then_common_running_calibration"
    DEFAULT = "architecture_default_then_common_calibration"
    PRESERVE_SOURCE = "preserve_complete_source_affine_and_running_no_calibration"


_METHOD_POLICIES: dict[str, tuple[HeadPolicy, BatchNormPolicy]] = {
    "scratch": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.DEFAULT),
    "anchor": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "anchor_untouched": (HeadPolicy.PRESERVE_SOURCE, BatchNormPolicy.PRESERVE_SOURCE),
    "ours_flow": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.OURS_DEFAULT),
    "ours_flow_oracle_anchor": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.OURS_DEFAULT),
    "ours_reconstruction": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.OURS_DEFAULT),
    "weightclip_flow": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_flow_oracle_anchor": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_controlled": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_commonzoo_fullwindow_ridge": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_commonzoo_fullwindow_memory": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_commonzoo_fullwindow_nearest_code": (HeadPolicy.FRESH_DEFAULT, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_commonzoo_fullwindow_ridge_native_oracle": (HeadPolicy.NATIVE_RELEASED, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_commonzoo_fullwindow_memory_native_oracle": (HeadPolicy.NATIVE_RELEASED, BatchNormPolicy.WEIGHTCLIP_NATIVE),
    "weightclip_commonzoo_fullwindow_nearest_code_native_oracle": (HeadPolicy.NATIVE_RELEASED, BatchNormPolicy.WEIGHTCLIP_NATIVE),
}


def method_policies(method_kind: str) -> tuple[HeadPolicy, BatchNormPolicy]:
    try:
        return _METHOD_POLICIES[str(method_kind)]
    except KeyError as exc:
        raise ValueError(f"unsupported evaluation method_kind {method_kind!r}") from exc


def validate_candidate_group_contract(
    method: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[HeadPolicy, BatchNormPolicy, str]:
    """Reject policy smuggling through mixed rows or arbitrary payload labels."""

    if not records:
        raise ValueError("candidate group cannot be empty")
    kinds = {str(row.get("payload", {}).get("method_kind", "")) for row in records}
    if len(kinds) != 1 or "" in kinds:
        raise ValueError(f"candidate group has missing/mixed payload method_kind: {sorted(kinds)}")
    kind = next(iter(kinds))
    if kind != str(method):
        raise ValueError(f"manifest method {method!r} disagrees with payload method_kind {kind!r}")
    if kind not in _METHOD_POLICIES:
        raise ValueError(f"unsupported evaluation method_kind {kind!r}")
    head_values = {str(row.get("head_policy", "")) for row in records}
    bn_values = {str(row.get("batchnorm_policy", "")) for row in records}
    head_paths = {str(row.get("head_module_path", "")) for row in records}
    if len(head_values) != 1 or len(bn_values) != 1 or len(head_paths) != 1:
        raise ValueError("candidate group mixes head/BatchNorm policy fields")
    expected_head, expected_bn = _METHOD_POLICIES[kind]
    if head_values != {expected_head.value} or bn_values != {expected_bn.value} or head_paths != {"fc"}:
        raise ValueError(
            f"method {kind!r} requires head={expected_head.value}, BN={expected_bn.value}, head_module_path=fc; "
            f"got head={head_values}, BN={bn_values}, paths={head_paths}"
        )
    return expected_head, expected_bn, "fc"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    payload: Any
    generation_seconds: float = 0.0
    generation_nfe: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRun:
    experiment_id: str
    method: str
    dataset: str
    evaluation_seed: int
    protocol: CandidateProtocol
    head_policy: HeadPolicy = HeadPolicy.FRESH_DEFAULT
    batchnorm_policy: BatchNormPolicy = BatchNormPolicy.DEFAULT
    head_module_path: str = "fc"
    verbose: bool = True


@dataclass(frozen=True)
class FineTuneConfig:
    epochs: int = 10
    learning_rate: float = 1.5e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    bn_calibration_batches: int = 200


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    validation_accuracy: float | None
    test_accuracy: float | None
    selected: bool
    selection_rank: int | None
    generation_seconds: float
    generation_nfe: int


@dataclass
class EvaluationOutput:
    rows: list[dict[str, Any]]
    candidate_scores: list[CandidateScore]
    selected_candidate_ids: list[str]
    summary: dict[str, Any]


ModelFactory = Callable[[Candidate], Any]
SelectionCommitter = Callable[[Mapping[str, Any]], None]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def reset_loader_streams(seed: int, *loaders: Iterable[Any]) -> None:
    """Replay candidate-paired sampler epochs from a committed seed."""

    seen: set[int] = set()
    for loader in loaders:
        if id(loader) in seen:
            continue
        seen.add(id(loader))
        sampler = getattr(loader, "sampler", None)
        reset = getattr(sampler, "reset", None)
        if callable(reset):
            reset(int(seed))


def reset_classifier_head(model: Any, module_path: str, seed: int) -> None:
    """Reset exactly one classifier module under an isolated paired RNG seed."""

    import torch

    module = model.get_submodule(module_path) if module_path else model
    if not hasattr(module, "reset_parameters"):
        raise TypeError(f"Classifier module {module_path!r} has no reset_parameters()")
    cuda_devices: list[int] = []
    for parameter in module.parameters(recurse=True):
        if parameter.is_cuda and parameter.device.index is not None:
            cuda_devices.append(int(parameter.device.index))
    with torch.random.fork_rng(devices=sorted(set(cuda_devices))):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        module.reset_parameters()


def prepare_batchnorm(model: Any, policy: BatchNormPolicy) -> None:
    """Apply the declared pre-calibration BN state without touching other layers."""

    import torch.nn as nn

    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if policy in {BatchNormPolicy.OURS_DEFAULT, BatchNormPolicy.DEFAULT}:
            if module.affine:
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
            module.reset_running_stats()
        elif policy == BatchNormPolicy.WEIGHTCLIP_NATIVE:
            # Controlled WeightCLIP transfers decoded BN affine parameters but
            # never trusts decoded/source running estimates on a paired dataset.
            module.reset_running_stats()


def calibrate_batchnorm(model: Any, loader: Iterable[Any], device: str, max_batches: int = 200) -> int:
    """Update BN running state while keeping dropout and other modules in eval mode."""

    import torch
    import torch.nn as nn

    model.eval()
    bn_modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    for module in bn_modules:
        module.train()
    seen = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= int(max_batches):
                break
            images, _ = _batch_xy(batch)
            model(images.to(device, non_blocking=True))
            seen += 1
    model.eval()
    return seen


def accuracy(model: Any, loader: Iterable[Any], device: str) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            images, targets = _batch_xy(batch)
            logits = model(images.to(device, non_blocking=True))
            targets = targets.to(device, non_blocking=True)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Non-finite logits during downstream evaluation")
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
    if total == 0:
        raise ValueError("Cannot evaluate an empty loader")
    value = correct / total
    if not 0.0 <= value <= 1.0:
        raise AssertionError(f"Impossible accuracy: {value}")
    return float(value)


def fine_tune_curve(
    model: Any,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any],
    test_loader: Iterable[Any],
    *,
    device: str,
    config: FineTuneConfig,
    verbose_prefix: str = "",
) -> list[dict[str, float]]:
    import torch
    import torch.nn.functional as F

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config.learning_rate),
        momentum=float(config.momentum),
        weight_decay=float(config.weight_decay),
    )
    curve: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_items = 0
        for batch in train_loader:
            images, targets = _batch_xy(batch)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite fine-tuning loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            count = int(targets.numel())
            train_loss += float(loss.detach().item()) * count
            train_items += count
        val_accuracy = accuracy(model, validation_loader, device)
        test_accuracy = accuracy(model, test_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(train_items, 1),
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "elapsed_seconds": time.perf_counter() - start,
        }
        curve.append(row)
        if verbose_prefix:
            print(
                f"{verbose_prefix} epoch={epoch}/{config.epochs} "
                f"loss={row['train_loss']:.6f} val_acc={val_accuracy:.4f} "
                f"test_acc={test_accuracy:.4f} elapsed={row['elapsed_seconds']:.1f}s"
            )
    return curve


def normalized_aulc(curve: Sequence[Mapping[str, float]], metric: str = "test_accuracy") -> float:
    if not curve:
        raise ValueError("AULC requires a nonempty curve")
    points = sorted((float(row["epoch"]), float(row[metric])) for row in curve)
    if len(points) == 1:
        return points[0][1]
    area = sum((x1 - x0) * (y0 + y1) * 0.5 for (x0, y0), (x1, y1) in zip(points, points[1:]))
    horizon = points[-1][0] - points[0][0]
    return area / horizon if horizon > 0 else points[-1][1]


def evaluate_initializations(
    run: EvaluationRun,
    candidates: Sequence[Candidate],
    model_factory: ModelFactory,
    *,
    train_loader: Iterable[Any],
    selection_loader: Iterable[Any],
    test_loader: Iterable[Any],
    bn_loader: Iterable[Any] | None = None,
    device: str = "cuda",
    finetune: FineTuneConfig | None = None,
    contract: WeightCLIPBenchmarkContract = DEFAULT_CONTRACT,
    selection_committer: SelectionCommitter | None = None,
) -> EvaluationOutput:
    """Rank candidates under one declared estimator, then fine-tune selections."""

    contract.validate()
    config = finetune or FineTuneConfig(
        epochs=contract.evaluation.finetune_epochs,
        learning_rate=contract.evaluation.finetune_lr,
        momentum=contract.evaluation.finetune_momentum,
        weight_decay=contract.evaluation.finetune_weight_decay,
        bn_calibration_batches=contract.access.bn_calibration_max_batches,
    )
    required_count, selected_count, selection_split = _protocol_budget(run.protocol, contract)
    if len(candidates) != required_count:
        raise ValueError(f"{run.protocol.value} requires exactly {required_count} candidates, got {len(candidates)}")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique within an evaluation run")
    if run.verbose:
        print(
            f"[evaluation:start] experiment={run.experiment_id} method={run.method} dataset={run.dataset} "
            f"seed={run.evaluation_seed} protocol={run.protocol.value} device={device} "
            f"candidates={len(candidates)} select_split={selection_split} output=caller-managed"
        )

    bn_loader = bn_loader or train_loader
    preliminary: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        started = time.perf_counter()
        seed_everything(run.evaluation_seed)
        reset_loader_streams(run.evaluation_seed, train_loader, selection_loader, test_loader, bn_loader)
        model = model_factory(candidate).to(device)
        if run.head_policy == HeadPolicy.FRESH_DEFAULT:
            reset_classifier_head(model, run.head_module_path, run.evaluation_seed)
        prepare_batchnorm(model, run.batchnorm_policy)
        calibrated = (
            0
            if run.batchnorm_policy == BatchNormPolicy.PRESERVE_SOURCE
            else calibrate_batchnorm(model, bn_loader, device, config.bn_calibration_batches)
        )
        validation_accuracy = accuracy(model, selection_loader, device) if selection_split == "validation" else None
        test_accuracy = accuracy(model, test_loader, device) if selection_split != "validation" else None
        preliminary.append(
            {
                "candidate": candidate,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                "bn_calibration_batches": calibrated,
                "evaluation_seconds": time.perf_counter() - started,
            }
        )
        if run.verbose:
            print(
                f"[evaluation:candidate] experiment={run.experiment_id} method={run.method} "
                f"candidate={candidate.candidate_id} index={index}/{len(candidates)} "
                f"val_acc={validation_accuracy} test_acc={test_accuracy} "
                f"elapsed={preliminary[-1]['evaluation_seconds']:.1f}s"
            )

    if run.protocol == CandidateProtocol.CONTROLLED_SINGLE:
        ranked = preliminary
    elif selection_split == "validation":
        ranked = sorted(preliminary, key=lambda item: float(item["validation_accuracy"]), reverse=True)
    else:
        ranked = sorted(preliminary, key=lambda item: float(item["test_accuracy"]), reverse=True)
    selected = ranked[:selected_count]
    selected_ids = [str(item["candidate"].candidate_id) for item in selected]
    selection_commit_sha256 = hashlib.sha256(
        json.dumps(selected_ids, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    if selection_split == "validation":
        if selection_committer is None:
            raise ValueError("validation-selected evaluation requires an immutable pre-test selection committer")
        selection_committer(
            {
                "schema_version": 1,
                "experiment_id": run.experiment_id,
                "method": run.method,
                "dataset": run.dataset,
                "evaluation_seed": run.evaluation_seed,
                "protocol": run.protocol.value,
                "selected_candidate_ids": selected_ids,
                "selection_commit_sha256": selection_commit_sha256,
                "test_access_at_commit": False,
            }
        )
        if run.verbose:
            print(f"[evaluation:selection-committed] ids={selected_ids} sha256={selection_commit_sha256}")
        for item in selected:
            candidate = item["candidate"]
            seed_everything(run.evaluation_seed)
            reset_loader_streams(run.evaluation_seed, train_loader, selection_loader, test_loader, bn_loader)
            model = model_factory(candidate).to(device)
            if run.head_policy == HeadPolicy.FRESH_DEFAULT:
                reset_classifier_head(model, run.head_module_path, run.evaluation_seed)
            prepare_batchnorm(model, run.batchnorm_policy)
            if run.batchnorm_policy != BatchNormPolicy.PRESERVE_SOURCE:
                calibrate_batchnorm(model, bn_loader, device, config.bn_calibration_batches)
            item["test_accuracy"] = accuracy(model, test_loader, device)
    rank_by_id = {str(item["candidate"].candidate_id): rank for rank, item in enumerate(ranked, start=1)}
    rows: list[dict[str, Any]] = []
    candidate_scores: list[CandidateScore] = []

    for item in preliminary:
        candidate = item["candidate"]
        is_selected = candidate.candidate_id in selected_ids
        candidate_scores.append(
            CandidateScore(
                candidate_id=candidate.candidate_id,
                validation_accuracy=item["validation_accuracy"],
                test_accuracy=None if item["test_accuracy"] is None else float(item["test_accuracy"]),
                selected=is_selected,
                selection_rank=rank_by_id[candidate.candidate_id] if is_selected else None,
                generation_seconds=float(candidate.generation_seconds),
                generation_nfe=int(candidate.generation_nfe),
            )
        )
        rows.append(
            _row_base(run, contract, candidate, required_count, selected_count, selection_split)
            | {
                "epoch": 0,
                "validation_accuracy": item["validation_accuracy"],
                "test_accuracy": None if item["test_accuracy"] is None else float(item["test_accuracy"]),
                "selected": is_selected,
                "selection_rank": rank_by_id[candidate.candidate_id] if is_selected else None,
                "train_loss": None,
                "elapsed_seconds": float(item["evaluation_seconds"]),
                "bn_calibration_batches": int(item["bn_calibration_batches"]),
                "aulc": None,
            }
        )

    selected_aulc: dict[str, float] = {}
    for rank, item in enumerate(selected, start=1):
        candidate = item["candidate"]
        seed_everything(run.evaluation_seed)
        reset_loader_streams(run.evaluation_seed, train_loader, selection_loader, test_loader, bn_loader)
        model = model_factory(candidate).to(device)
        if run.head_policy == HeadPolicy.FRESH_DEFAULT:
            reset_classifier_head(model, run.head_module_path, run.evaluation_seed)
        prepare_batchnorm(model, run.batchnorm_policy)
        calibrated = (
            0
            if run.batchnorm_policy == BatchNormPolicy.PRESERVE_SOURCE
            else calibrate_batchnorm(model, bn_loader, device, config.bn_calibration_batches)
        )
        curve = fine_tune_curve(
            model,
            train_loader,
            selection_loader,
            test_loader,
            device=device,
            config=config,
            verbose_prefix=(
                f"[evaluation:finetune] experiment={run.experiment_id} method={run.method} "
                f"candidate={candidate.candidate_id} rank={rank}/{selected_count}"
                if run.verbose
                else ""
            ),
        )
        with_epoch_zero = [
            {"epoch": 0.0, "test_accuracy": float(item["test_accuracy"]), "validation_accuracy": float(item["validation_accuracy"] or 0.0)}
        ] + curve
        aulc = normalized_aulc(with_epoch_zero)
        selected_aulc[candidate.candidate_id] = aulc
        for curve_row in curve:
            rows.append(
                _row_base(run, contract, candidate, required_count, selected_count, selection_split)
                | {
                    "epoch": int(curve_row["epoch"]),
                    "validation_accuracy": float(curve_row["validation_accuracy"]),
                    "test_accuracy": float(curve_row["test_accuracy"]),
                    "selected": True,
                    "selection_rank": rank,
                    "train_loss": float(curve_row["train_loss"]),
                    "elapsed_seconds": float(curve_row["elapsed_seconds"]),
                    "bn_calibration_batches": calibrated,
                    "aulc": aulc,
                }
            )

    summary = {
        "experiment_id": run.experiment_id,
        "method": run.method,
        "dataset": run.dataset,
        "protocol": run.protocol.value,
        "table_id": run.protocol.value,
        "candidate_budget": required_count,
        "selected_count": selected_count,
        "selection_split": selection_split,
        "selected_candidate_ids": selected_ids,
        "mean_candidate_epoch0_test_accuracy": (
            sum(float(item["test_accuracy"]) for item in preliminary) / len(preliminary)
            if all(item["test_accuracy"] is not None for item in preliminary)
            else None
        ),
        "test_scope": "selected_only_after_validation_commit" if selection_split == "validation" else "all_candidates_for_declared_estimator",
        "selection_commit_sha256": selection_commit_sha256,
        "selected_epoch0_test_accuracy": [float(item["test_accuracy"]) for item in selected],
        "selected_aulc": selected_aulc,
        "contract_fingerprint": contract.fingerprint(),
    }
    if run.verbose:
        print(
            f"[evaluation:done] experiment={run.experiment_id} method={run.method} "
            f"selected={selected_ids} mean_epoch0={summary['mean_candidate_epoch0_test_accuracy']}"
        )
    return EvaluationOutput(rows=rows, candidate_scores=candidate_scores, selected_candidate_ids=selected_ids, summary=summary)


def _protocol_budget(
    protocol: CandidateProtocol,
    contract: WeightCLIPBenchmarkContract,
) -> tuple[int, int, str]:
    if protocol == CandidateProtocol.CONTROLLED_SINGLE:
        return 1, 1, "none_random_precommitted"
    if protocol == CandidateProtocol.CONTROLLED_VALIDATION_BEST_K:
        return contract.evaluation.controlled_candidate_count, contract.evaluation.controlled_top_k, "validation"
    if protocol == CandidateProtocol.NATIVE_TEST_TOP5_ORACLE:
        return contract.evaluation.native_candidate_count, contract.evaluation.native_top_k, "test"
    raise ValueError(f"Unknown candidate protocol: {protocol}")


def _row_base(
    run: EvaluationRun,
    contract: WeightCLIPBenchmarkContract,
    candidate: Candidate,
    candidate_budget: int,
    selected_count: int,
    selection_split: str,
) -> dict[str, Any]:
    return {
        "experiment_id": run.experiment_id,
        "method": run.method,
        "dataset": run.dataset,
        "evaluation_seed": int(run.evaluation_seed),
        "protocol": run.protocol.value,
        "table_id": run.protocol.value,
        "candidate_id": candidate.candidate_id,
        "candidate_budget": int(candidate_budget),
        "selected_count_budget": int(selected_count),
        "selection_split": selection_split,
        "head_policy": run.head_policy.value,
        "head_seed": int(run.evaluation_seed),
        "batchnorm_policy": run.batchnorm_policy.value,
        "generation_seconds": float(candidate.generation_seconds),
        "generation_nfe": int(candidate.generation_nfe),
        "candidate_metadata": dict(candidate.metadata),
        "contract_version": contract.version,
        "contract_fingerprint": contract.fingerprint(),
    }


def _batch_xy(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        images = batch.get("images", batch.get("input", batch.get("pixel_values")))
        targets = batch.get("targets", batch.get("target", batch.get("labels")))
        if images is None or targets is None:
            raise TypeError("Mapping batch must contain images/input and targets/labels")
        return images, targets
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Expected batch as (images, targets) or a supported mapping")
