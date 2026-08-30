from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import ExperimentConfig, config_hash
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import (
    WeightNormalizer,
    build_weight_vae,
    decode_weights,
    encode_weights,
    load_celo_meta_task_tensors,
    load_torch_cache,
    logits_from_flat,
    move_task_tensors,
    spec_from_payload,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
OUT_DIR = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_analysis/head_error_sensitivity"

RUN_PAIRS = {
    "fc2_lam0p03": (
        0.03,
        "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0",
        "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p03_clean_harness_v1_seed0",
    ),
    "fc2_lam0p03_cap0p25": (
        0.03,
        "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0",
        "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap0p25_fc2recon_lam0p03_clean_harness_v1_seed0",
    ),
    "fc2_lam0p06": (
        0.06,
        "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p06_clean_harness_v1_seed0",
        "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p06_clean_harness_v1_seed0",
    ),
}

KEY_COLS = ["source_weight_index", "start_index", "task_name", "tau"]


def _load_cfg(output_dir: Path, *, device: str) -> ExperimentConfig:
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{output_dir / 'config.json'} does not contain a config mapping")
    values = dict(raw_cfg)
    values["device"] = str(device)
    values["dtype"] = "float32"
    return ExperimentConfig(**values)


def _load_run(run_name: str, *, device: str) -> dict[str, Any]:
    output_dir = ARTIFACT_ROOT / run_name
    cfg = _load_cfg(output_dir, device=device)
    dev = torch.device(cfg.device)
    print(
        "[head_sensitivity] load_run "
        f"run={run_name} hash={config_hash(cfg)} device={cfg.device} dtype={cfg.dtype}",
        flush=True,
    )
    for name in ["weight_pool.pt", "vae_checkpoint.pt", "downstream_results.csv", "downstream_curves.csv"]:
        print(f"[head_sensitivity] cache_hit run={run_name} file={output_dir / name}", flush=True)
    weight_payload = load_torch_cache(output_dir / "weight_pool.pt")
    vae_payload = load_torch_cache(output_dir / "vae_checkpoint.pt")
    if weight_payload is None or vae_payload is None:
        raise FileNotFoundError(f"{output_dir} missing weight_pool.pt or vae_checkpoint.pt")
    weights = weight_payload["weights"].to(device=dev, dtype=torch.float32)
    spec = spec_from_payload(weight_payload["spec"])
    normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
    vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=dev, dtype=torch.float32).eval()
    vae.load_state_dict(vae_payload["model_state"])
    task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=dev, dtype=torch.float32)
    return {
        "cfg": cfg,
        "device": dev,
        "run_name": run_name,
        "output_dir": output_dir,
        "weights": weights,
        "records": pd.DataFrame(weight_payload["records"]),
        "spec": spec,
        "normalizer": normalizer,
        "vae": vae,
        "task_tensors": task_tensors,
        "results": pd.read_csv(output_dir / "downstream_results.csv"),
        "curves": pd.read_csv(output_dir / "downstream_curves.csv"),
    }


def _task_tensor_set(task_tensors: dict[str, Any], task_name: str):
    if task_name in task_tensors:
        return task_tensors[task_name]
    if len(task_tensors) == 1:
        return next(iter(task_tensors.values()))
    raise KeyError(f"task {task_name!r} not found")


def _spec_slices(spec) -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for key, size in zip(spec.keys, spec.sizes, strict=True):
        result[str(key)] = slice(offset, offset + int(size))
        offset += int(size)
    return result


def _splice(base: torch.Tensor, donor: torch.Tensor, slices: dict[str, slice], keys: tuple[str, ...]) -> torch.Tensor:
    result = base.detach().clone()
    for key in keys:
        result[slices[key]] = donor.detach()[slices[key]]
    return result


def _loss_acc_logits(flat: torch.Tensor, *, task_set, spec, split: str, tau: float) -> tuple[float, float, torch.Tensor]:
    if split == "train":
        images, labels = task_set.train_images, task_set.train_labels
    elif split == "test":
        images, labels = task_set.test_images, task_set.test_labels
    else:
        raise ValueError(split)
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return float(loss.detach().cpu().item()), float(acc.detach().cpu().item()), logits.detach()


def _l2(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


def _rel_l2(delta: torch.Tensor, base: torch.Tensor) -> float:
    return _l2(delta) / max(_l2(base), 1e-30)


def _eval_starts(run: dict[str, Any], *, samples: int) -> pd.DataFrame:
    rows = run["results"][
        (run["results"]["split"].astype(str) == "eval")
        & (run["results"]["method"].astype(str) == "decoder_latent")
    ].copy()
    rows = rows.sort_values("start_index").reset_index(drop=True)
    if int(samples) > 0:
        rows = rows.iloc[: int(samples)].copy()
    return rows


def _assert_common_start_bank(control: pd.DataFrame, variant: pd.DataFrame, *, tag: str) -> None:
    left = control[KEY_COLS].reset_index(drop=True)
    right = variant[KEY_COLS].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"start bank mismatch tag={tag}")


def _row_for_start(
    *,
    tag: str,
    lam: float,
    variant_label: str,
    run: dict[str, Any],
    start_row: pd.Series,
) -> dict[str, Any]:
    slices = _spec_slices(run["spec"])
    source_weight_index = int(start_row["source_weight_index"])
    record = run["records"].iloc[source_weight_index].to_dict()
    task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
    tau = float(record.get("tau", start_row.get("tau", 1.0)))
    task_set = _task_tensor_set(run["task_tensors"], task_name)
    raw = run["weights"][source_weight_index].detach()
    with torch.no_grad():
        z = encode_weights(run["vae"], run["normalizer"], raw.reshape(1, -1)).squeeze(0)
        decoded = decode_weights(run["vae"], run["normalizer"], z.reshape(1, -1)).squeeze(0).detach()
        raw_norm = run["normalizer"].normalize(raw.reshape(1, -1)).squeeze(0)
        decoded_norm = run["normalizer"].normalize(decoded.reshape(1, -1)).squeeze(0)

        raw_train_loss, raw_train_acc, raw_train_logits = _loss_acc_logits(
            raw, task_set=task_set, spec=run["spec"], split="train", tau=tau
        )
        raw_test_loss, raw_test_acc, raw_test_logits = _loss_acc_logits(
            raw, task_set=task_set, spec=run["spec"], split="test", tau=tau
        )
        dec_train_loss, dec_train_acc, dec_train_logits = _loss_acc_logits(
            decoded, task_set=task_set, spec=run["spec"], split="train", tau=tau
        )
        dec_test_loss, dec_test_acc, dec_test_logits = _loss_acc_logits(
            decoded, task_set=task_set, spec=run["spec"], split="test", tau=tau
        )

        raw_body_decoded_fc2 = _splice(raw, decoded, slices, ("fc2.weight",))
        decoded_body_raw_fc2 = _splice(decoded, raw, slices, ("fc2.weight",))
        swap_train_loss, _swap_train_acc, swap_train_logits = _loss_acc_logits(
            raw_body_decoded_fc2, task_set=task_set, spec=run["spec"], split="train", tau=tau
        )
        swap_test_loss, _swap_test_acc, swap_test_logits = _loss_acc_logits(
            raw_body_decoded_fc2, task_set=task_set, spec=run["spec"], split="test", tau=tau
        )
        rescue_train_loss, _rescue_train_acc, rescue_train_logits = _loss_acc_logits(
            decoded_body_raw_fc2, task_set=task_set, spec=run["spec"], split="train", tau=tau
        )
        rescue_test_loss, _rescue_test_acc, rescue_test_logits = _loss_acc_logits(
            decoded_body_raw_fc2, task_set=task_set, spec=run["spec"], split="test", tau=tau
        )

    fc2_slice = slices["fc2.weight"]
    bias_slice = slices["fc2.bias"]
    fc2_delta = decoded[fc2_slice] - raw[fc2_slice]
    fc2_norm_delta = decoded_norm[fc2_slice] - raw_norm[fc2_slice]
    bias_delta = decoded[bias_slice] - raw[bias_slice]
    bias_norm_delta = decoded_norm[bias_slice] - raw_norm[bias_slice]
    decoded_gap_test = dec_test_loss - raw_test_loss
    decoded_gap_train = dec_train_loss - raw_train_loss
    return {
        "tag": tag,
        "lam": float(lam),
        "variant_label": variant_label,
        "source_weight_index": source_weight_index,
        "start_index": int(start_row["start_index"]),
        "task_name": task_name,
        "tau": tau,
        "raw_train_loss": raw_train_loss,
        "raw_test_loss": raw_test_loss,
        "decoded_train_loss": dec_train_loss,
        "decoded_test_loss": dec_test_loss,
        "decoded_minus_raw_train_loss": decoded_gap_train,
        "decoded_minus_raw_test_loss": decoded_gap_test,
        "decoded_train_acc": dec_train_acc,
        "decoded_test_acc": dec_test_acc,
        "raw_train_acc": raw_train_acc,
        "raw_test_acc": raw_test_acc,
        "full_raw_rel_l2": _rel_l2(decoded - raw, raw),
        "fc2_weight_raw_rel_l2": _rel_l2(fc2_delta, raw[fc2_slice]),
        "fc2_weight_raw_mse": float((fc2_delta.square().mean()).detach().cpu().item()),
        "fc2_weight_norm_mse": float((fc2_norm_delta.square().mean()).detach().cpu().item()),
        "fc2_bias_raw_rel_l2": _rel_l2(bias_delta, raw[bias_slice]),
        "fc2_bias_norm_mse": float((bias_norm_delta.square().mean()).detach().cpu().item()),
        "fc2_swap_into_raw_train_loss_delta": swap_train_loss - raw_train_loss,
        "fc2_swap_into_raw_test_loss_delta": swap_test_loss - raw_test_loss,
        "fc2_rescue_train_loss_delta": rescue_train_loss - raw_train_loss,
        "fc2_rescue_test_loss_delta": rescue_test_loss - raw_test_loss,
        "fc2_rescue_removed_test_loss": dec_test_loss - rescue_test_loss,
        "fc2_rescue_fraction_test": (dec_test_loss - rescue_test_loss) / decoded_gap_test
        if abs(decoded_gap_test) > 1e-12
        else float("nan"),
        "fc2_raw_body_train_logit_mse": float((swap_train_logits - raw_train_logits).square().mean().cpu().item()),
        "fc2_raw_body_test_logit_mse": float((swap_test_logits - raw_test_logits).square().mean().cpu().item()),
        "fc2_decoded_body_train_logit_mse": float((dec_train_logits - rescue_train_logits).square().mean().cpu().item()),
        "fc2_decoded_body_test_logit_mse": float((dec_test_logits - rescue_test_logits).square().mean().cpu().item()),
    }


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _pair_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_frames = []
    for tag in rows["tag"].drop_duplicates():
        sub = rows[rows["tag"] == tag]
        control = sub[sub["variant_label"] == "control"].copy()
        variant = sub[sub["variant_label"] == "A"].copy()
        keep = KEY_COLS + [
            "full_raw_rel_l2",
            "fc2_weight_raw_rel_l2",
            "fc2_weight_raw_mse",
            "fc2_weight_norm_mse",
            "fc2_swap_into_raw_test_loss_delta",
            "fc2_rescue_test_loss_delta",
            "fc2_rescue_removed_test_loss",
            "fc2_raw_body_test_logit_mse",
            "fc2_decoded_body_test_logit_mse",
            "decoded_minus_raw_test_loss",
        ]
        merged = control[keep].merge(variant[keep], on=KEY_COLS, suffixes=("_control", "_A"))
        merged["tag"] = tag
        merged["lam"] = float(sub["lam"].iloc[0])
        for col in keep:
            if col in KEY_COLS:
                continue
            merged[f"delta_{col}"] = merged[f"{col}_A"] - merged[f"{col}_control"]
        pair_frames.append(merged)
    paired = pd.concat(pair_frames, ignore_index=True)
    summary_rows = []
    for tag, sub in paired.groupby("tag"):
        for task_name, task_rows in [("all", sub), *list(sub.groupby("task_name"))]:
            target = task_rows["delta_decoded_minus_raw_test_loss"]
            summary_rows.append(
                {
                    "tag": tag,
                    "task_name": task_name,
                    "starts": int(len(task_rows)),
                    "mean_A_minus_control_decoded_test_loss": float(target.mean()),
                    "median_A_minus_control_decoded_test_loss": float(target.median()),
                    "mean_delta_fc2_weight_norm_mse": float(task_rows["delta_fc2_weight_norm_mse"].mean()),
                    "median_delta_fc2_weight_norm_mse": float(task_rows["delta_fc2_weight_norm_mse"].median()),
                    "mean_delta_fc2_weight_raw_rel_l2": float(task_rows["delta_fc2_weight_raw_rel_l2"].mean()),
                    "mean_delta_fc2_raw_body_test_logit_mse": float(
                        task_rows["delta_fc2_raw_body_test_logit_mse"].mean()
                    ),
                    "mean_delta_fc2_decoded_body_test_logit_mse": float(
                        task_rows["delta_fc2_decoded_body_test_logit_mse"].mean()
                    ),
                    "corr_delta_norm_mse_vs_loss": _safe_corr(task_rows["delta_fc2_weight_norm_mse"], target),
                    "corr_delta_raw_body_logit_mse_vs_loss": _safe_corr(
                        task_rows["delta_fc2_raw_body_test_logit_mse"], target
                    ),
                    "corr_delta_decoded_body_logit_mse_vs_loss": _safe_corr(
                        task_rows["delta_fc2_decoded_body_test_logit_mse"], target
                    ),
                }
            )
    return paired, pd.DataFrame(summary_rows)


def _plot_norm_vs_functional(paired: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    markers = {"fc2_lam0p03": "^", "fc2_lam0p03_cap0p25": "D", "fc2_lam0p06": "s"}
    colors = {"mnist": "#4c78a8", "fashion_mnist": "#f58518"}
    for ax, xcol, title in [
        (axes[0], "delta_fc2_weight_norm_mse", "weight-space MSE delta"),
        (axes[1], "delta_fc2_raw_body_test_logit_mse", "raw-body logit MSE delta"),
        (axes[2], "delta_fc2_decoded_body_test_logit_mse", "decoded-body logit MSE delta"),
    ]:
        for tag in paired["tag"].drop_duplicates():
            sub = paired[paired["tag"] == tag]
            for task_name, task_rows in sub.groupby("task_name"):
                ax.scatter(
                    task_rows[xcol],
                    task_rows["delta_decoded_minus_raw_test_loss"],
                    marker=markers.get(tag, "o"),
                    color=colors.get(str(task_name), "#666666"),
                    s=52,
                    alpha=0.82,
                    label=f"{tag} {task_name}",
                )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel(xcol)
        ax.set_ylabel("A - control decoded step0 test loss")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles, strict=False))
    axes[-1].legend(dedup.values(), dedup.keys(), fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "head_error_metric_vs_step0_gap.png", dpi=190)
    plt.close(fig)


def _plot_worst_bars(paired: pd.DataFrame) -> None:
    for tag, output_name, color in [
        ("fc2_lam0p06", "lam0p06_worst_starts_mse_vs_functional.png", "#b35c35"),
        ("fc2_lam0p03_cap0p25", "lam0p03_cap0p25_worst_starts_mse_vs_functional.png", "#7b6bbd"),
    ]:
        worst = paired[paired["tag"] == tag].copy()
        if worst.empty:
            continue
        worst = worst.sort_values("delta_decoded_minus_raw_test_loss", ascending=False).head(8)
        labels = (
            worst["source_weight_index"].astype(str)
            + "\n"
            + worst["task_name"].astype(str).str.replace("fashion_mnist", "fashion", regex=False)
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(len(worst))
        axes[0].bar(x, worst["delta_decoded_minus_raw_test_loss"], color=color)
        axes[0].axhline(0.0, color="black", linewidth=1)
        axes[0].set_title(f"{tag} worst starts: CE gap")
        axes[0].set_ylabel("A - control decoded step0 test loss")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[0].grid(axis="y", alpha=0.25)

        width = 0.36
        axes[1].bar(
            x - width / 2,
            worst["delta_fc2_weight_norm_mse"],
            width=width,
            label="normalized fc2 MSE delta",
            color="#4c78a8",
        )
        axes[1].bar(
            x + width / 2,
            worst["delta_fc2_decoded_body_test_logit_mse"],
            width=width,
            label="decoded-body logit MSE delta",
            color="#f58518",
        )
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set_title("same starts: MSE vs functional delta")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / output_name, dpi=190)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "[head_sensitivity] startup "
        f"device={args.device} samples={args.samples} output_dir={OUT_DIR}",
        flush=True,
    )
    rows = []
    normalizer_checks = []
    for tag, (lam, control_run, a_run) in RUN_PAIRS.items():
        print(f"[head_sensitivity] pair={tag} lam={lam}", flush=True)
        control = _load_run(control_run, device=str(args.device))
        variant = _load_run(a_run, device=str(args.device))
        if list(control["spec"].keys) != list(variant["spec"].keys):
            raise ValueError(f"spec mismatch tag={tag}")
        max_weight_diff = float((control["weights"] - variant["weights"]).abs().max().detach().cpu().item())
        mean_diff = float((control["normalizer"].mean - variant["normalizer"].mean).abs().max().cpu().item())
        std_diff = float((control["normalizer"].std - variant["normalizer"].std).abs().max().cpu().item())
        normalizer_checks.append(
            {
                "tag": tag,
                "max_weight_pool_abs_diff": max_weight_diff,
                "max_normalizer_mean_abs_diff": mean_diff,
                "max_normalizer_std_abs_diff": std_diff,
            }
        )
        if max_weight_diff > 1e-6 or mean_diff > 1e-6 or std_diff > 1e-6:
            raise ValueError(f"paired run mismatch tag={tag}")
        control_starts = _eval_starts(control, samples=int(args.samples))
        variant_starts = _eval_starts(variant, samples=int(args.samples))
        _assert_common_start_bank(control_starts, variant_starts, tag=tag)
        for idx, start_row in control_starts.iterrows():
            print(
                "[head_sensitivity] start "
                f"pair={tag} {idx + 1}/{len(control_starts)} source={int(start_row['source_weight_index'])} "
                f"task={start_row['task_name']}",
                flush=True,
            )
            rows.append(_row_for_start(tag=tag, lam=lam, variant_label="control", run=control, start_row=start_row))
            rows.append(_row_for_start(tag=tag, lam=lam, variant_label="A", run=variant, start_row=start_row))

    rows_df = pd.DataFrame(rows)
    paired, summary = _pair_rows(rows_df)
    pd.DataFrame(normalizer_checks).to_csv(OUT_DIR / "pair_validity_checks.csv", index=False)
    rows_df.to_csv(OUT_DIR / "head_error_sensitivity_rows.csv", index=False)
    paired.to_csv(OUT_DIR / "head_error_sensitivity_paired.csv", index=False)
    summary.to_csv(OUT_DIR / "head_error_sensitivity_summary.csv", index=False)
    _plot_norm_vs_functional(paired)
    _plot_worst_bars(paired)
    print("[head_sensitivity] summary")
    print(summary.to_string(index=False))
    print(f"[head_sensitivity] wrote {OUT_DIR}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether fc2 MSE controls functional head error.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
