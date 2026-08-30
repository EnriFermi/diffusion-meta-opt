#!/usr/bin/env python3
"""Strict, CPU-only conformance audit for kron-torch 0.3.3.

This is a practical package audit.  It is deliberately not presented as an
exact audit of Li's full c3 criterion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch


EXPECTED_VERSION = "0.3.3"
EPS32 = float(torch.finfo(torch.float32).eps)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "kron_c3_conformance"

_COMPILE_CALLS = 0
_KRON_MODULE: Any | None = None
_KRON_CLASS: Any | None = None


def _identity_compile(fn: Callable[..., Any] | None = None, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    global _COMPILE_CALLS
    _COMPILE_CALLS += 1
    if fn is None:
        return lambda inner: inner
    return fn


def _disable_compile() -> None:
    """Disable compilation before importing kron-torch's decorated helpers."""
    torch.compile = _identity_compile  # type: ignore[assignment]


def _load_package() -> tuple[Any, Any]:
    global _KRON_MODULE, _KRON_CLASS
    if _KRON_MODULE is None:
        _disable_compile()
        package = importlib.import_module("kron_torch")
        _KRON_MODULE = importlib.import_module("kron_torch.kron")
        _KRON_CLASS = package.Kron
    return _KRON_MODULE, _KRON_CLASS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_manifest() -> dict[str, Any]:
    module, _ = _load_package()
    package_dir = Path(module.__file__).resolve().parent
    files = []
    for path in sorted(package_dir.rglob("*.py")):
        files.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "distribution": "kron-torch",
        "version": importlib.metadata.version("kron-torch"),
        "module": str(Path(module.__file__).resolve()),
        "source_files": files,
    }


def _log(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"[kron_c3_conformance] {message}", flush=True)


def _rho(g: torch.Tensor) -> float:
    return math.sqrt(EPS32) * float(g.abs().mean())


def _reference_pair(
    q: list[torch.Tensor], g: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent A/B reference, including package rho perturbation."""
    perturbed = g + _rho(g) * v
    if g.ndim == 1:
        return q[0] * perturbed, v / q[0]
    if g.ndim == 2 and len(q) == 2:
        a = q[0] @ perturbed @ q[1].T
        b = torch.linalg.solve(q[0].T, v)
        b = torch.linalg.solve(q[1].T, b.T).T
        return a, b
    raise ValueError(f"reference only supports 1D and 2D factors, got {tuple(g.shape)}")


def _exprs_for(g: torch.Tensor) -> tuple[Any, Any]:
    module, _ = _load_package()
    _, exprs = module._init_Q_exprs(g, 1.0, 8192, 2, None, dtype=g.dtype)
    return exprs


@contextmanager
def _fixed_randn(v: torch.Tensor):
    module, _ = _load_package()
    original = module.torch.randn_like

    def fixed(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del input_tensor, args, kwargs
        return v.clone()

    module.torch.randn_like = fixed
    try:
        yield
    finally:
        module.torch.randn_like = original


def _package_pair(
    q: list[torch.Tensor], g: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    module, _ = _load_package()
    work = g.clone()
    with _fixed_randn(v):
        return module._calc_A_and_conjB(_exprs_for(g)[0], work, q)


def _terms(a: torch.Tensor, b: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if a.ndim == 1:
        return [a.square() - b.square()], [a.square() + b.square()]
    return [a @ a.T - b @ b.T, a.T @ a - b.T @ b], [
        a @ a.T + b @ b.T,
        a.T @ a + b.T @ b,
    ]


def _assert_pair_terms(
    package: tuple[torch.Tensor, torch.Tensor], reference: tuple[torch.Tensor, torch.Tensor], tol: float = 2e-11
) -> None:
    for actual, expected in zip(package, reference, strict=True):
        if not torch.allclose(actual, expected, atol=tol, rtol=tol):
            raise AssertionError(f"package pair mismatch max_abs={float((actual - expected).abs().max())}")


def _assert_term_conformance(
    package: tuple[torch.Tensor, torch.Tensor], reference: tuple[torch.Tensor, torch.Tensor], tol: float = 2e-10
) -> None:
    actual_a, actual_b = package
    expected_a, expected_b = reference
    # 0.3.3 casts triangular solves to float32 and casts back.
    _assert_pair_terms(package, reference, tol=max(tol, 2e-6))
    actual_s, actual_t = _terms(actual_a, actual_b)
    expected_s, expected_t = _terms(expected_a, expected_b)
    for actual, expected in zip(actual_s + actual_t, expected_s + expected_t, strict=True):
        if not torch.allclose(actual, expected, atol=max(tol, 2e-6), rtol=max(tol, 2e-6)):
            raise AssertionError(f"A/B/S term mismatch max_abs={float((actual - expected).abs().max())}")


def _objective(q: list[torch.Tensor], g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    a, b = _reference_pair(q, g, v)
    return 0.5 * (a.square().sum() + b.square().sum())


def _lb_norm(t: torch.Tensor) -> torch.Tensor:
    module, _ = _load_package()
    norm = t.norm(float("inf"))
    if float(norm) == 0.0:
        return norm
    return module._lb(t.clone(), norm)


def _expected_delta(q: list[torch.Tensor], a: torch.Tensor, b: torch.Tensor, step: float) -> list[torch.Tensor]:
    s, t = _terms(a, b)
    out = []
    for factor, raw_s, positive_t in zip(q, s, t, strict=True):
        denom = _lb_norm(positive_t)
        if factor.ndim == 1:
            out.append(-step * raw_s * factor / denom.clamp_min(torch.finfo(factor.dtype).tiny))
        else:
            out.append(-step * torch.triu(raw_s) / denom.clamp_min(torch.finfo(factor.dtype).tiny) @ factor)
    return out


def _package_update(
    q: list[torch.Tensor], g: torch.Tensor, v: torch.Tensor, step: float
) -> list[torch.Tensor]:
    module, _ = _load_package()
    exprs = _exprs_for(g)
    before = [x.clone() for x in q]
    with _fixed_randn(v):
        module._update_precond(q, exprs, g.clone(), torch.tensor(step, dtype=g.dtype), torch.tensor(torch.finfo(g.dtype).tiny, dtype=g.dtype))
    return [after - prior for after, prior in zip(q, before, strict=True)]


def _autodiff_lie_reference(q: list[torch.Tensor], g: torch.Tensor, v: torch.Tensor) -> list[torch.Tensor]:
    variables = [x.detach().clone().requires_grad_(True) for x in q]
    value = _objective(variables, g, v)
    gradients = torch.autograd.grad(value, variables)
    if g.ndim == 1:
        return [gradients[0] * variables[0]]
    return [grad @ factor.T for grad, factor in zip(gradients, variables, strict=True)]


def _matrix_cases() -> list[tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]]:
    dtype = torch.float64
    return [
        [
            torch.tensor([[1.2, 0.25], [0.0, 0.8]], dtype=dtype),
            torch.tensor([[0.9, -0.1], [0.0, 1.3]], dtype=dtype),
        ],
        [
            torch.tensor([[0.7, -0.35], [0.0, 1.6]], dtype=dtype),
            torch.tensor([[1.4, 0.2], [0.0, 0.6]], dtype=dtype),
        ],
        [
            torch.tensor([[2.0, 0.4], [0.0, 0.55]], dtype=dtype),
            torch.tensor([[0.65, -0.25], [0.0, 1.8]], dtype=dtype),
        ],
    ]


def _pair_and_lie_check() -> dict[str, Any]:
    dtype = torch.float64
    g = torch.tensor([[0.7, -0.2], [0.3, 0.5]], dtype=dtype)
    v = torch.tensor([[0.4, -0.8], [1.1, 0.2]], dtype=dtype)
    q = _matrix_cases()[0]
    package = _package_pair(q, g, v)
    reference = _reference_pair(q, g, v)
    _assert_term_conformance(package, reference)
    a, b = reference
    s, positive = _terms(a, b)
    if any(float(x.min()) < -1e-12 for x in positive):
        raise AssertionError("positive normalization statistic is negative")
    delta = _package_update([x.clone() for x in q], g, v, 1e-3)
    expected = _expected_delta(q, a, b, 1e-3)
    delta_errors = [float((x - y).abs().max()) for x, y in zip(delta, expected, strict=True)]
    if max(delta_errors) > 2e-10:
        raise AssertionError(f"normalized package delta mismatch errors={delta_errors}")
    lie = _autodiff_lie_reference(q, g, v)
    lie_errors = []
    for actual, expected_s in zip(lie, _terms(a, b)[0], strict=True):
        lie_errors.append(float((torch.triu(actual) - torch.triu(expected_s)).detach().abs().max()))
    if max(lie_errors) > 2e-10:
        raise AssertionError(f"Lie/autodiff reference mismatch errors={lie_errors}")
    return {
        "rho": _rho(g),
        "positive_min": min(float(x.min()) for x in positive),
        "max_delta_error": max(delta_errors),
        "max_lie_error": max(lie_errors),
    }


def _central_fd_check() -> dict[str, Any]:
    dtype = torch.float64
    values = []
    for index, q in enumerate(_matrix_cases()):
        g = torch.tensor([[0.3 + index, -0.7], [0.8, 0.2 - index]], dtype=dtype)
        v = torch.tensor([[0.6, -0.4 - index], [0.9, 0.5]], dtype=dtype)
        a, b = _reference_pair(q, g, v)
        direction = _expected_delta(q, a, b, 1.0)
        t = 1e-5
        plus = [x + t * dx for x, dx in zip(q, direction, strict=True)]
        minus = [x - t * dx for x, dx in zip(q, direction, strict=True)]
        derivative = float((_objective(plus, g, v) - _objective(minus, g, v)) / (2 * t))
        values.append(derivative)
        if not derivative < -1e-8:
            raise AssertionError(f"central FD derivative is not negative case={index} value={derivative}")
    return {"cases": len(values), "max_directional_derivative": max(values), "derivatives": values}


def _small_step_descent_check() -> dict[str, Any]:
    dtype = torch.float64
    improvements = []
    for index, q in enumerate(_matrix_cases()):
        g = torch.tensor([[0.2 + index, -0.9], [0.5, 0.4 - index]], dtype=dtype)
        v = torch.tensor([[0.4, 0.8], [-0.6, 0.3 + index]], dtype=dtype)
        before = float(_objective(q, g, v))
        step = 1e-4
        updated = [x.clone() for x in q]
        _package_update(updated, g, v, step)
        after = float(_objective(updated, g, v))
        improvements.append(before - after)
        if not after < before:
            raise AssertionError(f"small-step criterion did not descend case={index}: {before} -> {after}")
    return {"cases": len(improvements), "min_improvement": min(improvements), "improvements": improvements}


def _monte_carlo_check(seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    dtype = torch.float64
    q = _matrix_cases()[1]
    g = torch.tensor([[0.8, -0.4], [0.2, 1.1]], dtype=dtype)
    module, _ = _load_package()
    original = module.torch.randn_like
    seen: list[torch.Tensor] = []
    package_sums = [torch.zeros_like(q[0]), torch.zeros_like(q[1])]
    reference_sums = [torch.zeros_like(q[0]), torch.zeros_like(q[1])]

    def capture(input_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        value = original(input_tensor, *args, **kwargs)
        seen.append(value.clone())
        return value

    module.torch.randn_like = capture
    try:
        for _ in range(256):
            package = module._calc_A_and_conjB(_exprs_for(g)[0], g.clone(), q)
            reference = _reference_pair(q, g, seen[-1])
            package_s, _ = _terms(*package)
            reference_s, _ = _terms(*reference)
            for i in range(2):
                package_sums[i] += package_s[i]
                reference_sums[i] += reference_s[i]
    finally:
        module.torch.randn_like = original
    errors = [float(((x - y) / len(seen)).abs().max()) for x, y in zip(package_sums, reference_sums, strict=True)]
    if max(errors) > 1e-6:
        raise AssertionError(f"Monte Carlo A/B/S expectation mismatch errors={errors}")
    rho_effect = _rho(g) * float(g.abs().mean())
    if not rho_effect > 0:
        raise AssertionError("rho perturbation was not active")
    return {"samples": len(seen), "max_mean_S_error": max(errors), "rho": _rho(g), "rho_effect_scale": rho_effect}


def _celo_shape_check() -> dict[str, Any]:
    dtype = torch.float64
    shapes = [(32, 64), (10, 32), (32,), (10,)]
    counts = []
    for index, shape in enumerate(shapes):
        g = torch.arange(1, math.prod(shape) + 1, dtype=dtype).reshape(shape) / 100.0
        q, exprs = _load_package()[0]._init_Q_exprs(g, 1.0, 8192, 2, None, dtype=dtype)
        counts.append(sum(x.numel() for x in q))
        if any((x.ndim == 2 and not torch.allclose(x, torch.eye(x.shape[0], x.shape[1], dtype=dtype))) or (x.ndim == 1 and not torch.allclose(x, torch.ones_like(x))) for x in q):
            raise AssertionError(f"CELO identity initialization failed shape={shape}")
        v = torch.sin(torch.arange(1, math.prod(shape) + 1, dtype=dtype)).reshape(shape)
        package = _package_pair(q, g, v)
        reference = _reference_pair(q, g, v)
        _assert_pair_terms(package, reference, tol=2e-6)
        if index < 2:
            a, b = reference
            lie = _autodiff_lie_reference(q, g, v)
            s, _ = _terms(a, b)
            if max(float((torch.triu(x) - torch.triu(y)).detach().abs().max()) for x, y in zip(lie, s, strict=True)) > 2e-9:
                raise AssertionError(f"CELO factorwise Lie descent mismatch shape={shape}")
    if sum(math.prod(shape) for shape in shapes) != 2410:
        raise AssertionError("CELO synthetic parameter dimension changed")
    return {"parameter_dim": sum(math.prod(shape) for shape in shapes), "preconditioner_numel": sum(counts), "shapes": shapes}


def _one_dimensional_check() -> dict[str, Any]:
    dtype = torch.float64
    g = torch.tensor([0.7, -1.1, 0.3], dtype=dtype)
    v = torch.tensor([0.2, 0.9, -0.4], dtype=dtype)
    q, _ = _load_package()[0]._init_Q_exprs(g, 1.0, 8192, 2, None, dtype=dtype)
    package = _package_pair(q, g, v)
    reference = _reference_pair(q, g, v)
    _assert_term_conformance(package, reference)
    gradient = _load_package()[0]._precond_grad(q, _exprs_for(g), g)
    if not torch.allclose(gradient, g):
        raise AssertionError("identity diagonal preconditioner did not preserve 1D gradient")
    delta = _package_update([x.clone() for x in q], g, v, 1e-4)
    if not all(torch.isfinite(x).all() for x in delta):
        raise AssertionError("1D diagonal update is not finite")
    return {"factor_count": len(q), "factor_shape": list(q[0].shape), "max_delta": float(delta[0].abs().max())}


def _preconditioned_gradient_check() -> dict[str, Any]:
    dtype = torch.float64
    module, _ = _load_package()
    g = torch.tensor([[0.4, -0.8, 0.2], [0.7, 0.1, -0.3]], dtype=dtype)
    q = [torch.tensor([[1.1, 0.2], [0.0, 0.8]], dtype=dtype), torch.tensor([[0.9, -0.1, 0.3], [0.0, 1.2, -0.2], [0.0, 0.0, 0.7]], dtype=dtype)]
    got = module._precond_grad(q, _exprs_for(g), g)
    expected = q[0].T @ q[0] @ g @ q[1].T @ q[1]
    error = float((got - expected).abs().max())
    if error > 2e-11:
        raise AssertionError(f"P=Q^TQ action mismatch error={error}")
    return {"max_action_error": error, "gradient_norm": float(got.norm())}


def _optimizer_state_check() -> dict[str, Any]:
    dtype = torch.float64
    module, Kron = _load_package()
    parameter = torch.nn.Parameter(torch.zeros((2, 3), dtype=dtype))
    gradient = torch.tensor([[10.0, -4.0, 2.0], [3.0, 8.0, -6.0]], dtype=dtype)
    optimizer = Kron([parameter], lr=0.02, b1=0.0, preconditioner_update_probability=1e-9, precond_lr=0.0, precond_dtype=dtype)
    parameter.grad = gradient.clone()
    before = parameter.detach().clone()
    optimizer.step()
    state = optimizer.state[parameter]
    if "Q" not in state or "momentum_buffer" not in state:
        raise AssertionError("optimizer did not distinguish initialization state")
    raw = module._precond_grad(state["Q"], state["exprs"], gradient)
    clipped = raw.clone()
    module._clip_update_rms(clipped)
    expected_parameter = before - 0.02 * clipped
    clip_error = float((parameter.detach() - expected_parameter).abs().max())
    q_before = [x.clone() for x in state["Q"]]
    if clip_error > 2e-11:
        raise AssertionError(f"clipping reconstruction mismatch error={clip_error}")
    if max(float((x - y).abs().max()) for x, y in zip(state["Q"], q_before, strict=True)) != 0.0:
        raise AssertionError("Q changed when preconditioner update was disabled")

    update_parameter = torch.nn.Parameter(torch.zeros((2, 2), dtype=dtype))
    update_optimizer = Kron([update_parameter], lr=0.001, b1=0.0, preconditioner_update_probability=1.0, precond_lr=0.1, precond_dtype=dtype)
    update_parameter.grad = torch.tensor([[0.5, -0.2], [0.8, 0.4]], dtype=dtype)
    update_optimizer.step()
    update_state = update_optimizer.state[update_parameter]
    q_mutation = max(float(x.abs().max()) for x in update_state["Q"] if torch.any(x != torch.eye(x.shape[0], x.shape[1], dtype=x.dtype)))
    if not q_mutation > 0:
        raise AssertionError("Q update was not observable separately from parameter update")
    return {"clip_max_error": clip_error, "q_update_observed": True, "preconditioner_step": int(update_optimizer._prob_step)}


class _ZeroRandom:
    def random(self) -> float:
        return 0.0


def _balance_check() -> dict[str, Any]:
    dtype = torch.float64
    module, Kron = _load_package()
    parameter = torch.nn.Parameter(torch.zeros((2, 3), dtype=dtype))
    optimizer = Kron([parameter], lr=0.0, b1=0.0, preconditioner_update_probability=1.0, precond_lr=0.0, precond_dtype=dtype)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    state["Q"][0].mul_(2.0)
    state["Q"][1].mul_(0.5)
    optimizer.rng = _ZeroRandom()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    norms = [float(x.norm(float("inf"))) for x in state["Q"]]
    if max(norms) - min(norms) > 2e-11:
        raise AssertionError(f"forced balance did not equalize factor norms={norms}")

    no_balance_parameter = torch.nn.Parameter(torch.zeros((2, 3), dtype=dtype))
    no_balance = Kron([no_balance_parameter], lr=0.0, b1=0.0, preconditioner_update_probability=1e-9, precond_lr=0.0, precond_dtype=dtype)
    no_balance_parameter.grad = torch.ones_like(no_balance_parameter)
    no_balance.step()
    no_state = no_balance.state[no_balance_parameter]
    no_state["Q"][0].mul_(2.0)
    no_state["Q"][1].mul_(0.5)
    no_balance.rng = _ZeroRandom()
    no_balance_parameter.grad = torch.ones_like(no_balance_parameter)
    no_balance.step()
    no_norms = [float(x.norm(float("inf"))) for x in no_state["Q"]]
    if max(no_norms) - min(no_norms) < 0.5:
        raise AssertionError("balance was not gated by do_update")
    del module
    return {"balanced_norms": norms, "ungated_norms": no_norms}


def _merge_restore_check() -> dict[str, Any]:
    dtype = torch.float64
    module, Kron = _load_package()
    parameter = torch.nn.Parameter(torch.zeros((2, 3, 4), dtype=dtype))
    optimizer = Kron([parameter], lr=0.0, b1=0.0, preconditioner_update_probability=1e-9, precond_lr=0.0, merge_dims=True, precond_dtype=dtype)
    gradient = torch.arange(1, 25, dtype=dtype).reshape(2, 3, 4) / 10.0
    parameter.grad = gradient.clone()
    optimizer.step()
    state = optimizer.state[parameter]
    if tuple(state["merged_shape"]) != (6, 4):
        raise AssertionError(f"unexpected merge shape={state.get('merged_shape')}")
    merged = gradient.view(6, 4)
    restored = module._precond_grad(state["Q"], state["exprs"], merged).view_as(gradient)
    if not torch.allclose(restored, gradient):
        raise AssertionError("merge/restore changed identity preconditioned gradient")
    return {
        "original_shape": [int(x) for x in gradient.shape],
        "merged_shape": [int(x) for x in state["merged_shape"]],
        "restored_shape": [int(x) for x in restored.shape],
    }


def _check(name: str, fn: Callable[[], dict[str, Any]], details: list[dict[str, Any]], quiet: bool) -> dict[str, Any]:
    started = time.perf_counter()
    _log(f"stage={name} start", quiet)
    try:
        metrics = fn()
        elapsed = time.perf_counter() - started
        row = {"check": name, "passed": True, "elapsed_s": elapsed, **{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in metrics.items()}}
        details.append(row)
        _log(f"stage={name} PASS elapsed_s={elapsed:.3f} metrics={json.dumps(metrics, sort_keys=True)}", quiet)
        return {"passed": True, "elapsed_s": elapsed, "metrics": metrics}
    except Exception as exc:  # keep independent checks running and report all failures
        elapsed = time.perf_counter() - started
        details.append({"check": name, "passed": False, "elapsed_s": elapsed, "error": f"{type(exc).__name__}: {exc}"})
        _log(f"stage={name} FAIL elapsed_s={elapsed:.3f} error={type(exc).__name__}: {exc}", quiet)
        return {"passed": False, "elapsed_s": elapsed, "error": f"{type(exc).__name__}: {exc}"}


def run_audit(output_dir: Path, *, seed: int = 1234, quiet: bool = False) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    _disable_compile()
    _log(
        f"startup device=cpu dtype=torch.float64 seed={seed} cache_mode=disabled output_dir={output_dir} "
        f"cuda_available={torch.cuda.is_available()} gpu_used=False",
        quiet,
    )
    package = _package_manifest()
    if package["version"] != EXPECTED_VERSION:
        raise RuntimeError(f"kron-torch version must be {EXPECTED_VERSION}, got {package['version']}")
    details: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for name, fn in (
        ("package_version_source_hashes", lambda: {"version": package["version"], "source_file_count": len(package["source_files"])}),
        ("torch_compile_disabled", lambda: {"compile_calls_during_import": _COMPILE_CALLS, "torch_compile_is_identity": torch.compile is _identity_compile}),
        ("two_by_two_A_B_S_positive_normalization_lie", _pair_and_lie_check),
        ("central_fd_directional_derivative", _central_fd_check),
        ("small_step_sample_criterion_descent", _small_step_descent_check),
        ("monte_carlo_V_expectation_with_rho", lambda: _monte_carlo_check(seed)),
        ("celo_shaped_2d_factor_identities_and_lie_descent", _celo_shape_check),
        ("one_dimensional_diagonal_factors", _one_dimensional_check),
        ("exact_preconditioned_gradient_P_QTQ", _preconditioned_gradient_check),
        ("clipping_reconstruction_and_separate_Q_update", _optimizer_state_check),
        ("init_update_balance_distinction", _balance_check),
        ("synthetic_3d_merge_restore", _merge_restore_check),
    ):
        checks[name] = _check(name, fn, details, quiet)

    csv_path = output_dir / "kron_c3_conformance_details.csv"
    fieldnames = sorted({key for row in details for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)

    script_path = Path(__file__).resolve()
    validation = {
        "accepted": all(bool(value.get("passed")) for value in checks.values()),
        "scope": "practical package criterion-conformance; not exact Li c3",
        "package": package,
        "config": {"device": "cpu", "dtype": "torch.float64", "seed": seed, "cache_mode": "disabled", "torch_compile_disabled": True},
        "checks": checks,
        "artifacts": {
            "json": str(output_dir / "kron_c3_conformance_validation.json"),
            "csv": str(csv_path),
            "csv_sha256": _sha256_file(csv_path),
            "script": str(script_path),
            "script_sha256": _sha256_file(script_path),
        },
        "allowed_conclusion": "The installed kron-torch 0.3.3 package conforms to the audited practical package criterion and update semantics on CPU for these probes; this does not establish exact Li c3 equivalence.",
    }
    json_path = output_dir / "kron_c3_conformance_validation.json"
    json_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_hash = _sha256_file(json_path)
    _log(f"artifacts json={json_path} sha256={json_hash}", quiet)
    _log(f"artifacts csv={csv_path} sha256={validation['artifacts']['csv_sha256']}", quiet)
    _log(f"summary accepted={validation['accepted']} checks={len(checks)}", quiet)
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.output_dir, seed=args.seed, quiet=args.quiet)
    except Exception as exc:
        print(f"[kron_c3_conformance] FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
