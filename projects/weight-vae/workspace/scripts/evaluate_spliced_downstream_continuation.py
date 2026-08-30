#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    decode_weights,
    encode_weights,
    finite_value,
    logits_from_flat,
)
from scripts.analyze_block_recon_head_splice import (
    ARTIFACT_ROOT,
    BLOCK_GROUPS,
    KEY_COLS,
    _load_run,
    _splice,
    _spec_slices,
    _task_tensor_set,
)


def _log(message: str) -> None:
    print(f"[spliced_downstream] {message}", flush=True)


def _selected_lr(output_dir: Path, method: str) -> float:
    rows = pd.read_csv(output_dir / "selected_lrs.csv")
    selected = pd.to_numeric(rows["selected"], errors="coerce").fillna(0).astype(int)
    sub = rows[(rows["method"].astype(str) == str(method)) & (selected == 1)]
    if sub.empty:
        raise RuntimeError(f"no selected LR for method={method!r} in {output_dir / 'selected_lrs.csv'}")
    return float(sub.iloc[0]["candidate_lr"])


def _train_batch_indices(task_set: Any, *, batch_size: int, step: int, start_index: int) -> torch.Tensor | None:
    train_count = int(task_set.train_labels.shape[0])
    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size >= train_count:
        return None
    offset = (int(start_index) * 1009 + int(step) * batch_size) % train_count
    return (torch.arange(batch_size, device=task_set.train_labels.device) + offset).remainder(train_count).long()


def _loss_acc(
    flat: torch.Tensor,
    *,
    task_set: Any,
    spec: Any,
    split: str,
    tau: float,
    batch_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split == "train":
        images = task_set.train_images
        labels = task_set.train_labels
        if batch_indices is not None:
            images = images.index_select(0, batch_indices)
            labels = labels.index_select(0, batch_indices)
    elif split == "test":
        images = task_set.test_images
        labels = task_set.test_labels
    else:
        raise ValueError(f"split must be train or test, got {split!r}")
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc


def _run_raw_theta_continuation(
    *,
    theta0: torch.Tensor,
    task_set: Any,
    spec: Any,
    tau: float,
    lr: float,
    steps: int,
    eval_every: int,
    batch_size: int,
    finite_penalty: float,
    success_threshold: float,
    start_index: int,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    theta = theta0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=float(lr))
    rows: list[dict[str, Any]] = []
    diverged = False
    threshold_step = -1
    eval_every = max(1, int(eval_every))
    steps = int(steps)

    for step in range(steps + 1):
        with torch.enable_grad():
            batch_indices = _train_batch_indices(
                task_set,
                batch_size=int(batch_size),
                step=int(step),
                start_index=int(start_index),
            )
            train_step_loss, _ = _loss_acc(
                theta,
                task_set=task_set,
                spec=spec,
                split="train",
                tau=float(tau),
                batch_indices=batch_indices,
            )
        train_step_finite = bool(torch.isfinite(train_step_loss).detach().cpu().item())
        if not train_step_finite:
            diverged = True

        should_record = int(step) == 0 or int(step) % eval_every == 0 or int(step) == steps or bool(diverged)
        if should_record:
            with torch.no_grad():
                theta_eval = theta.detach()
                train_eval_loss, train_eval_acc = _loss_acc(
                    theta_eval,
                    task_set=task_set,
                    spec=spec,
                    split="train",
                    tau=float(tau),
                )
                test_loss, test_acc = _loss_acc(
                    theta_eval,
                    task_set=task_set,
                    spec=spec,
                    split="test",
                    tau=float(tau),
                )
            finite = (
                train_step_finite
                and bool(torch.isfinite(train_eval_loss).detach().cpu().item())
                and bool(torch.isfinite(test_loss).detach().cpu().item())
            )
            if not finite:
                diverged = True
            train_value = finite_value(
                float(train_eval_loss.detach().cpu().item()) if finite else float("inf"),
                penalty=float(finite_penalty),
            )
            test_value = finite_value(
                float(test_loss.detach().cpu().item()) if finite else float("inf"),
                penalty=float(finite_penalty),
            )
            if threshold_step < 0 and train_value <= float(success_threshold):
                threshold_step = int(step)
            rows.append(
                {
                    **metadata,
                    "lr": float(lr),
                    "step": int(step),
                    "train_loss": train_value,
                    "train_acc": float(train_eval_acc.detach().cpu().item()) if finite else 0.0,
                    "test_loss": test_value,
                    "test_acc": float(test_acc.detach().cpu().item()) if finite else 0.0,
                    "theta_norm": float(theta_eval.float().norm().detach().cpu().item()),
                    "diverged": bool(diverged),
                }
            )
        if int(step) == steps or bool(diverged):
            break
        with torch.enable_grad():
            optimizer.zero_grad(set_to_none=True)
            train_step_loss.backward()
            optimizer.step()

    train_losses = np.array([float(row["train_loss"]) for row in rows], dtype=np.float64)
    final = rows[-1]
    metrics = {
        **metadata,
        "lr": float(lr),
        "aulc": float(np.mean(np.minimum(train_losses, float(finite_penalty)))),
        "final_train_loss": float(final["train_loss"]),
        "best_train_loss": float(np.min(train_losses)),
        "final_test_loss": float(final["test_loss"]),
        "final_test_acc": float(final["test_acc"]),
        "steps_to_threshold": int(threshold_step),
        "diverged": bool(diverged),
    }
    return rows, metrics


def _eval_decoder_starts(run: dict[str, Any]) -> pd.DataFrame:
    rows = run["results"][
        (run["results"]["split"].astype(str) == "eval")
        & (run["results"]["method"].astype(str) == "decoder_latent")
    ].copy()
    return rows.sort_values("start_index").reset_index(drop=True)


def _selected_starts(control: dict[str, Any], a_run: dict[str, Any], selected_bank: pd.DataFrame) -> pd.DataFrame:
    control_starts = _eval_decoder_starts(control)
    a_starts = _eval_decoder_starts(a_run)
    required = ["source_weight_index", "start_bank_position", "task_name", "tau"]
    missing = [col for col in required if col not in selected_bank.columns]
    if missing:
        raise ValueError(f"selected bank missing required columns: {missing}")
    selected = selected_bank.copy()
    selected["source_weight_index"] = selected["source_weight_index"].astype(int)
    selected["start_bank_position"] = selected["start_bank_position"].astype(int)
    selected["task_name"] = selected["task_name"].astype(str)
    selected["tau"] = pd.to_numeric(selected["tau"], errors="raise").astype(float)
    keys = ["source_weight_index", "start_bank_position", "task_name", "tau"]
    keep_cols = ["start_index", *keys]
    control_match = selected.merge(control_starts[keep_cols], on=keys, how="left", validate="one_to_one")
    a_match = selected.merge(a_starts[keep_cols], on=keys, how="left", validate="one_to_one", suffixes=("", "_a"))
    if control_match["start_index"].isna().any() or a_match["start_index"].isna().any():
        raise ValueError("selected bank contains starts absent from control or A downstream eval rows")
    if not control_match["start_index"].astype(int).equals(a_match["start_index"].astype(int)):
        raise ValueError("control/A eval start_index mismatch for selected bank")
    result = control_match.copy()
    result["start_index"] = result["start_index"].astype(int)
    return result.reset_index(drop=True)


def _logged_step0(run: dict[str, Any]) -> pd.DataFrame:
    rows = run["curves"][
        (run["curves"]["split"].astype(str) == "eval")
        & (run["curves"]["method"].astype(str) == "decoder_latent")
        & (pd.to_numeric(run["curves"]["step"], errors="coerce") == 0)
    ].copy()
    return rows[KEY_COLS + ["train_loss", "test_loss"]].copy()


def _candidate_thetas(
    *,
    control: dict[str, Any],
    a_run: dict[str, Any],
    source_weight_index: int,
    include_raw: bool,
    include_injection: bool,
) -> dict[str, tuple[str, str, torch.Tensor]]:
    slices = _spec_slices(control["spec"])
    w0 = control["weights"][int(source_weight_index)].detach()
    with torch.no_grad():
        z_control = encode_weights(control["vae"], control["normalizer"], w0.reshape(1, -1)).squeeze(0).detach()
        z_a = encode_weights(a_run["vae"], a_run["normalizer"], w0.reshape(1, -1)).squeeze(0).detach()
        dec_control = decode_weights(control["vae"], control["normalizer"], z_control.reshape(1, -1)).squeeze(0).detach()
        dec_a = decode_weights(a_run["vae"], a_run["normalizer"], z_a.reshape(1, -1)).squeeze(0).detach()

    candidates: dict[str, tuple[str, str, torch.Tensor]] = {
        "control_decoded": ("baseline", "control_decoded", dec_control),
        "A_decoded": ("baseline", "A_decoded", dec_a),
    }
    if include_raw:
        candidates["raw_w0"] = ("baseline", "raw_w0", w0)
    for block_group, keys in BLOCK_GROUPS.items():
        candidates[f"A_with_control_{block_group}"] = (
            str(block_group),
            "control_block_into_A_decoded",
            _splice(dec_a, dec_control, slices, keys),
        )
        if include_injection:
            candidates[f"control_with_A_{block_group}"] = (
                str(block_group),
                "A_block_into_control_decoded",
                _splice(dec_control, dec_a, slices, keys),
            )
    return candidates


def _validation_rows(curves: pd.DataFrame, control: dict[str, Any], a_run: dict[str, Any]) -> pd.DataFrame:
    step0 = curves[curves["step"].astype(int) == 0].copy()
    logged_control = _logged_step0(control).rename(
        columns={"train_loss": "logged_train_loss", "test_loss": "logged_test_loss"}
    )
    logged_a = _logged_step0(a_run).rename(columns={"train_loss": "logged_train_loss", "test_loss": "logged_test_loss"})
    rows: list[dict[str, Any]] = []
    for candidate, logged in [("control_decoded", logged_control), ("A_decoded", logged_a)]:
        sub = step0[step0["candidate"].astype(str) == candidate].copy()
        merged = sub.merge(logged, on=KEY_COLS, how="left", validate="one_to_one")
        rows.append(
            {
                "candidate": candidate,
                "rows": int(len(merged)),
                "missing_logged": int(merged["logged_train_loss"].isna().sum()),
                "max_train_loss_absdiff": float((merged["train_loss"] - merged["logged_train_loss"]).abs().max()),
                "max_test_loss_absdiff": float((merged["test_loss"] - merged["logged_test_loss"]).abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _summarize(results: pd.DataFrame, curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_ref = results[results["candidate"].astype(str) == "control_decoded"][
        ["source_weight_index", "start_bank_position", "task_name", "tau", "aulc", "final_test_loss"]
    ].rename(columns={"aulc": "control_decoded_aulc", "final_test_loss": "control_decoded_final_test_loss"})
    paired = results.merge(
        control_ref,
        on=["source_weight_index", "start_bank_position", "task_name", "tau"],
        how="left",
        validate="many_to_one",
    )
    paired["aulc_gap_vs_control_decoded"] = paired["aulc"] - paired["control_decoded_aulc"]
    paired["final_test_loss_gap_vs_control_decoded"] = (
        paired["final_test_loss"] - paired["control_decoded_final_test_loss"]
    )

    step0 = curves[curves["step"].astype(int) == 0].copy()
    c0 = step0[step0["candidate"].astype(str) == "control_decoded"][
        ["source_weight_index", "start_bank_position", "task_name", "tau", "test_loss"]
    ].rename(columns={"test_loss": "control_decoded_step0_test_loss"})
    a0 = step0[step0["candidate"].astype(str) == "A_decoded"][
        ["source_weight_index", "start_bank_position", "task_name", "tau", "test_loss"]
    ].rename(columns={"test_loss": "A_decoded_step0_test_loss"})
    step0_gap = c0.merge(a0, on=["source_weight_index", "start_bank_position", "task_name", "tau"], validate="one_to_one")
    step0_gap["A_minus_control_step0_test_loss"] = (
        step0_gap["A_decoded_step0_test_loss"] - step0_gap["control_decoded_step0_test_loss"]
    )
    paired = paired.merge(
        step0_gap[
            [
                "source_weight_index",
                "start_bank_position",
                "task_name",
                "tau",
                "A_minus_control_step0_test_loss",
            ]
        ],
        on=["source_weight_index", "start_bank_position", "task_name", "tau"],
        how="left",
        validate="many_to_one",
    )
    paired["positive_step0_damage"] = paired["A_minus_control_step0_test_loss"] > 1e-4

    rows: list[dict[str, Any]] = []
    for subset_name, subset in [
        ("all", paired),
        ("positive_step0_damage", paired[paired["positive_step0_damage"]].copy()),
    ]:
        for (candidate, block_group, action), group in subset.groupby(["candidate", "block_group", "action"], sort=True):
            values = group["aulc_gap_vs_control_decoded"].to_numpy(dtype=np.float64)
            test_values = group["final_test_loss_gap_vs_control_decoded"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "subset": subset_name,
                    "candidate": str(candidate),
                    "block_group": str(block_group),
                    "action": str(action),
                    "starts": int(group["source_weight_index"].nunique()),
                    "aulc_gap_mean": float(np.nanmean(values)) if values.size else float("nan"),
                    "aulc_gap_median": float(np.nanmedian(values)) if values.size else float("nan"),
                    "final_test_loss_gap_mean": float(np.nanmean(test_values)) if test_values.size else float("nan"),
                    "final_test_loss_gap_median": float(np.nanmedian(test_values)) if test_values.size else float("nan"),
                    "worse_count_aulc_gap_gt0": int(np.sum(values > 0.0)),
                }
            )
    return paired, pd.DataFrame(rows)


def _write_notes(
    *,
    output_root: Path,
    prefix: str,
    pair_name: str,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    lr: float,
    steps: int,
) -> None:
    def _summary_row(candidate: str, subset: str) -> pd.Series | None:
        rows = summary[(summary["candidate"].astype(str) == candidate) & (summary["subset"].astype(str) == subset)]
        if rows.empty:
            return None
        return rows.iloc[0]

    a_all = _summary_row("A_decoded", "all")
    head_pos = _summary_row("A_with_control_classifier_head", "positive_step0_damage")
    weight_pos = _summary_row("A_with_control_fc2.weight", "positive_step0_damage")
    max_validation = float(
        validation[["max_train_loss_absdiff", "max_test_loss_absdiff"]].to_numpy(dtype=np.float64).max()
    )
    lines = [
        f"# {pair_name} Spliced Downstream Continuation",
        "",
        "Diagnostic: raw-theta Adam continuations from decoded/spliced theta starts. This is a downstream sufficiency test for immediate decoded head/fc2 damage, not a latent-z trajectory replacement.",
        "",
        "## Validity",
        "",
        f"- Starts: {results['source_weight_index'].nunique()}.",
        f"- Raw-theta Adam LR: {lr:.6g}; steps: {steps}.",
        f"- Max recompute step0 diff vs logged decoder_latent starts: {max_validation:.6g}.",
        "",
        "## Key Gaps",
        "",
    ]
    if a_all is not None:
        lines.append(
            f"- `A_decoded` all-start AULC gap vs `control_decoded`: mean {float(a_all['aulc_gap_mean']):+.6g}, median {float(a_all['aulc_gap_median']):+.6g}."
        )
    if weight_pos is not None:
        lines.append(
            f"- Positive step0 subset, `A_with_control_fc2.weight`: AULC gap mean {float(weight_pos['aulc_gap_mean']):+.6g}, median {float(weight_pos['aulc_gap_median']):+.6g}, starts {int(weight_pos['starts'])}."
        )
    if head_pos is not None:
        lines.append(
            f"- Positive step0 subset, `A_with_control_classifier_head`: AULC gap mean {float(head_pos['aulc_gap_mean']):+.6g}, median {float(head_pos['aulc_gap_median']):+.6g}, starts {int(head_pos['starts'])}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Support for CH3/CH7 downstream sufficiency requires the positive-step0 AULC gap from `A_decoded` to shrink materially when the control `fc2.weight` or `classifier_head` is spliced into A decoded weights. If only the step0 loss is rescued but continuation AULC is not, the splice remains an immediate localization diagnostic.",
        ]
    )
    (output_root / f"{prefix}_spliced_downstream_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw-theta downstream continuations from decoded/spliced starts.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--a-run", required=True)
    parser.add_argument("--start-bank-csv", type=Path, required=True)
    parser.add_argument("--pair-name", default="spliced_downstream")
    parser.add_argument("--prefix", default="spliced_downstream")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=float("nan"))
    parser.add_argument("--lr-source", choices=["control", "A"], default="control")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--include-injection", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    selected_bank = pd.read_csv(args.start_bank_csv)
    control_dir = ARTIFACT_ROOT / str(args.control_run)
    a_dir = ARTIFACT_ROOT / str(args.a_run)
    _log(
        "startup "
        f"pair={args.pair_name} device={args.device} dtype=float32 starts_csv={args.start_bank_csv} "
        f"steps={args.steps} eval_every={args.eval_every} batch_size={args.batch_size} output_root={output_root}"
    )
    control = _load_run(control_dir, device=str(args.device))
    a_run = _load_run(a_dir, device=str(args.device))
    if tuple(control["weights"].shape) != tuple(a_run["weights"].shape):
        raise ValueError("weight shape mismatch")
    max_weight_diff = float((control["weights"] - a_run["weights"]).abs().max().detach().cpu().item())
    if max_weight_diff > 1e-6:
        raise ValueError(f"weight pool mismatch: max abs diff={max_weight_diff}")
    starts = _selected_starts(control, a_run, selected_bank)
    lr_dir = control_dir if str(args.lr_source) == "control" else a_dir
    lr = float(args.lr) if math.isfinite(float(args.lr)) and float(args.lr) > 0.0 else _selected_lr(lr_dir, "raw")
    cfg = replace(
        control["cfg"],
        device=str(args.device),
        downstream_steps=int(args.steps),
        downstream_eval_every=int(args.eval_every),
        downstream_batch_size=int(args.batch_size),
    )
    _log(
        "resolved "
        f"control={control_dir.name} A={a_dir.name} lr={lr:.6g} lr_source={args.lr_source} "
        f"latent_dim={cfg.latent_dim} hidden_dim={cfg.vae_hidden_dim} starts={starts['source_weight_index'].astype(int).tolist()}"
    )

    curve_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    total = int(len(starts)) * (2 + int(args.include_raw) + len(BLOCK_GROUPS) * (1 + int(args.include_injection)))
    completed = 0
    for _, start_row in starts.iterrows():
        source_weight_index = int(start_row["source_weight_index"])
        task_name = str(start_row["task_name"])
        tau = float(start_row["tau"])
        start_index = int(start_row["start_index"])
        task_set = _task_tensor_set(control["task_tensors"], task_name)
        candidates = _candidate_thetas(
            control=control,
            a_run=a_run,
            source_weight_index=source_weight_index,
            include_raw=bool(args.include_raw),
            include_injection=bool(args.include_injection),
        )
        for candidate, (block_group, action, theta0) in candidates.items():
            completed += 1
            metadata = {
                "pair": str(args.pair_name),
                "candidate": str(candidate),
                "block_group": str(block_group),
                "action": str(action),
                "source_weight_index": int(source_weight_index),
                "start_index": int(start_index),
                "start_bank_position": int(start_row["start_bank_position"]),
                "task_name": task_name,
                "tau": float(tau),
                "optimizer": "raw_theta_adam",
            }
            rows, metrics = _run_raw_theta_continuation(
                theta0=theta0,
                task_set=task_set,
                spec=control["spec"],
                tau=float(tau),
                lr=float(lr),
                steps=int(args.steps),
                eval_every=int(args.eval_every),
                batch_size=int(args.batch_size),
                finite_penalty=float(cfg.finite_penalty),
                success_threshold=float(cfg.success_threshold),
                start_index=int(start_index),
                metadata=metadata,
            )
            curve_rows.extend(rows)
            result_rows.append(metrics)
            _log(
                "progress "
                f"{completed}/{total} source={source_weight_index} candidate={candidate} "
                f"aulc={float(metrics['aulc']):.6g} final_test={float(metrics['final_test_loss']):.6g}"
            )

    curves = pd.DataFrame(curve_rows)
    results = pd.DataFrame(result_rows)
    validation = _validation_rows(curves, control, a_run)
    paired, summary = _summarize(results, curves)

    curves_path = output_root / f"{args.prefix}_spliced_downstream_curves.csv"
    results_path = output_root / f"{args.prefix}_spliced_downstream_results.csv"
    paired_path = output_root / f"{args.prefix}_spliced_downstream_paired.csv"
    summary_path = output_root / f"{args.prefix}_spliced_downstream_summary.csv"
    validation_path = output_root / f"{args.prefix}_spliced_downstream_validation.csv"
    manifest_path = output_root / f"{args.prefix}_spliced_downstream_manifest.json"
    curves.to_csv(curves_path, index=False)
    results.to_csv(results_path, index=False)
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    validation.to_csv(validation_path, index=False)
    _write_notes(
        output_root=output_root,
        prefix=str(args.prefix),
        pair_name=str(args.pair_name),
        results=results,
        summary=summary,
        validation=validation,
        lr=float(lr),
        steps=int(args.steps),
    )
    manifest = {
        "pair_name": str(args.pair_name),
        "control_run": str(control_dir.resolve()),
        "a_run": str(a_dir.resolve()),
        "start_bank_csv": str(Path(args.start_bank_csv).expanduser().resolve()),
        "source_indices": [int(v) for v in starts["source_weight_index"].astype(int).tolist()],
        "lr": float(lr),
        "lr_source": str(args.lr_source),
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "batch_size": int(args.batch_size),
        "include_raw": bool(args.include_raw),
        "include_injection": bool(args.include_injection),
        "elapsed_sec": float(time.perf_counter() - started),
        "outputs": {
            "curves": str(curves_path),
            "results": str(results_path),
            "paired": str(paired_path),
            "summary": str(summary_path),
            "validation": str(validation_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _log(
        "done "
        f"results={results_path} curves={curves_path} summary={summary_path} "
        f"validation={validation_path} elapsed_sec={time.perf_counter() - started:.2f}"
    )


if __name__ == "__main__":
    main()
