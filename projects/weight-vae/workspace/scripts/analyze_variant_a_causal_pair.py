from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
DEFAULT_OUT = ROOT / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/causal_pair_analysis"

REQUIRED_FILES = [
    "config.json",
    "artifact_manifest.json",
    "weight_pool_records.csv",
    "vae_checkpoint.pt",
    "vae_metrics.csv",
    "geometry.csv",
    "preconditioning_diagnostics.csv",
    "downstream_start_bank.csv",
    "selected_lrs.csv",
    "downstream_results.csv",
    "downstream_curves.csv",
]

START_BANK_KEYS = [
    "start_bank_position",
    "source_weight_index",
    "run",
    "step",
    "source_lr",
    "task_name",
    "tau",
    "start_role",
]
DOWNSTREAM_KEYS = ["source_weight_index", "start_index", "task_name", "tau"]
EXPECTED_STEPS = list(range(0, 301, 25))


def _log(message: str) -> None:
    print(f"[variant_a_causal_pair] {message}", flush=True)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    return df


def _as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    arr = _finite(values)
    if arr.size == 0:
        return {name: float("nan") for name in ["mean", "median", "p90", "p95", "p99", "max", "min"]}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def _exact_sign_p(values: Iterable[float]) -> float:
    arr = _finite(values)
    if arr.size == 0 or arr.size > 20:
        return float("nan")
    observed = abs(float(np.mean(arr)))
    count = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=int(arr.size)):
        total += 1
        signed = arr * np.asarray(signs, dtype=float)
        if abs(float(np.mean(signed))) >= observed - 1e-15:
            count += 1
    return float(count / total)


def _bootstrap_ci(values: Iterable[float], *, seed: int = 0, samples: int = 50_000) -> tuple[float, float]:
    arr = _finite(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(samples, int(arr.size)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _curve_auc(group: pd.DataFrame, column: str, *, trapz: bool) -> float:
    g = group.sort_values("step")
    y = _as_num(g[column]).to_numpy(dtype=float)
    x = _as_num(g["step"]).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if y.size == 0:
        return float("nan")
    if not trapz or y.size == 1 or float(x[-1] - x[0]) == 0.0:
        return float(np.mean(y))
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _load_run(run_name: str) -> dict[str, object]:
    output_dir = ARTIFACT_ROOT / run_name
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    _log(f"load run={run_name} output_dir={output_dir}")
    data: dict[str, object] = {
        "run_name": run_name,
        "output_dir": output_dir,
        "config": _read_json(output_dir / "config.json"),
        "manifest": _read_json(output_dir / "artifact_manifest.json"),
    }
    for name in REQUIRED_FILES:
        path = output_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".csv":
            data[name.removesuffix(".csv")] = _read_csv(path)
    return data


def _completion_audit(label: str, run: dict[str, object]) -> list[dict[str, object]]:
    output_dir = Path(run["output_dir"])
    rows: list[dict[str, object]] = []
    for filename in REQUIRED_FILES:
        path = output_dir / filename
        row: dict[str, object] = {
            "run_label": label,
            "file": filename,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "rows": float("nan"),
            "columns": float("nan"),
        }
        if path.suffix == ".csv" and path.exists():
            df = pd.read_csv(path)
            row["rows"] = len(df)
            row["columns"] = len(df.columns)
        rows.append(row)
    return rows


def _selected_lrs(label: str, run: dict[str, object]) -> pd.DataFrame:
    selected = run["selected_lrs"].copy()
    selected.insert(0, "run_label", label)
    return selected


def _start_bank_audit(control: dict[str, object], a_run: dict[str, object]) -> pd.DataFrame:
    control_bank = control["downstream_start_bank"].copy()
    a_bank = a_run["downstream_start_bank"].copy()
    common = [key for key in START_BANK_KEYS if key in control_bank.columns and key in a_bank.columns]
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "check": "row_count",
            "control": len(control_bank),
            "A": len(a_bank),
            "pass": len(control_bank) == len(a_bank),
            "details": "",
        }
    )
    rows.append(
        {
            "check": "common_key_columns",
            "control": len(common),
            "A": len(common),
            "pass": set(common) == set(START_BANK_KEYS),
            "details": ",".join(common),
        }
    )
    if common:
        equal = control_bank[common].reset_index(drop=True).equals(a_bank[common].reset_index(drop=True))
        rows.append(
            {
                "check": "start_bank_keys_equal",
                "control": "",
                "A": "",
                "pass": bool(equal),
                "details": "" if equal else "see start_bank_mismatch.csv",
            }
        )
    return pd.DataFrame(rows)


def _curve_group(curves: pd.DataFrame, row: pd.Series, method: str) -> pd.DataFrame:
    mask = (
        (curves["split"].astype(str) == "eval")
        & (curves["method"].astype(str) == method)
        & (curves["source_weight_index"] == row["source_weight_index"])
        & (curves["start_index"] == row["start_index"])
        & (curves["task_name"].astype(str) == str(row["task_name"]))
        & np.isclose(_as_num(curves["tau"]).to_numpy(dtype=float), float(row["tau"]))
    )
    return curves[mask].copy()


def _paired_downstream(control: dict[str, object], a_run: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    res_c = control["downstream_results"].copy()
    res_a = a_run["downstream_results"].copy()
    curves_c = control["downstream_curves"].copy()
    curves_a = a_run["downstream_curves"].copy()
    paired_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for method in ["raw", "decoder_latent"]:
        rc = res_c[(res_c["split"].astype(str) == "eval") & (res_c["method"].astype(str) == method)]
        ra = res_a[(res_a["split"].astype(str) == "eval") & (res_a["method"].astype(str) == method)]
        merged = rc[DOWNSTREAM_KEYS + ["aulc", "final_test_loss", "final_test_acc", "lr", "diverged"]].merge(
            ra[DOWNSTREAM_KEYS + ["aulc", "final_test_loss", "final_test_acc", "lr", "diverged"]],
            on=DOWNSTREAM_KEYS,
            suffixes=("_control", "_A"),
            how="inner",
        )
        for _, row in merged.iterrows():
            gc = _curve_group(curves_c, row, method)
            ga = _curve_group(curves_a, row, method)
            step0_c = gc[gc["step"] == 0]
            step0_a = ga[ga["step"] == 0]
            post_c = gc[gc["step"] > 0]
            post_a = ga[ga["step"] > 0]
            final_step = min(_as_num(gc["step"]).max(), _as_num(ga["step"]).max())
            final_c = gc[gc["step"] == final_step]
            final_a = ga[ga["step"] == final_step]
            paired_rows.append(
                {
                    "method": method,
                    **{key: row[key] for key in DOWNSTREAM_KEYS},
                    "lr_control": float(row["lr_control"]),
                    "lr_A": float(row["lr_A"]),
                    "lr_match": bool(math.isclose(float(row["lr_control"]), float(row["lr_A"]))),
                    "train_mean_aulc_control": float(row["aulc_control"]),
                    "train_mean_aulc_A": float(row["aulc_A"]),
                    "train_mean_aulc_delta": float(row["aulc_A"] - row["aulc_control"]),
                    "final_test_loss_delta": float(row["final_test_loss_A"] - row["final_test_loss_control"]),
                    "final_test_acc_delta": float(row["final_test_acc_A"] - row["final_test_acc_control"]),
                    "step0_test_loss_delta": float(step0_a["test_loss"].iloc[0] - step0_c["test_loss"].iloc[0])
                    if len(step0_a) and len(step0_c)
                    else float("nan"),
                    "step0_train_loss_delta": float(step0_a["train_loss"].iloc[0] - step0_c["train_loss"].iloc[0])
                    if len(step0_a) and len(step0_c)
                    else float("nan"),
                    "step0_test_acc_delta": float(step0_a["test_acc"].iloc[0] - step0_c["test_acc"].iloc[0])
                    if len(step0_a) and len(step0_c)
                    else float("nan"),
                    "post0_test_loss_mean_delta": float(_as_num(post_a["test_loss"]).mean() - _as_num(post_c["test_loss"]).mean())
                    if len(post_a) and len(post_c)
                    else float("nan"),
                    "post0_train_loss_mean_delta": float(_as_num(post_a["train_loss"]).mean() - _as_num(post_c["train_loss"]).mean())
                    if len(post_a) and len(post_c)
                    else float("nan"),
                    "final_curve_test_loss_delta": float(final_a["test_loss"].iloc[0] - final_c["test_loss"].iloc[0])
                    if len(final_a) and len(final_c)
                    else float("nan"),
                    "test_mean_aulc_delta": _curve_auc(ga, "test_loss", trapz=False)
                    - _curve_auc(gc, "test_loss", trapz=False),
                    "test_trapz_aulc_delta": _curve_auc(ga, "test_loss", trapz=True)
                    - _curve_auc(gc, "test_loss", trapz=True),
                    "train_curve_mean_delta": _curve_auc(ga, "train_loss", trapz=False)
                    - _curve_auc(gc, "train_loss", trapz=False),
                    "diverged_control": bool(str(row["diverged_control"]).lower() in {"true", "1"}),
                    "diverged_A": bool(str(row["diverged_A"]).lower() in {"true", "1"}),
                }
            )
        method_rows = pd.DataFrame([r for r in paired_rows if r["method"] == method])
        for metric in [
            "train_mean_aulc_delta",
            "test_mean_aulc_delta",
            "test_trapz_aulc_delta",
            "step0_test_loss_delta",
            "post0_test_loss_mean_delta",
        ]:
            values = method_rows[metric].to_numpy(dtype=float) if len(method_rows) else np.asarray([])
            lo, hi = _bootstrap_ci(values)
            loo = [
                float(np.delete(values, idx).mean())
                for idx in range(len(values))
                if len(values) > 1 and np.isfinite(np.delete(values, idx)).any()
            ]
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)) if len(values) else float("nan"),
                    "median": float(np.nanmedian(values)) if len(values) else float("nan"),
                    "worse_count_A_gt_control": int(np.nansum(values > 0)),
                    "better_count_A_lt_control": int(np.nansum(values < 0)),
                    "exact_sign_p": _exact_sign_p(values),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "leave_one_out_min": min(loo) if loo else float("nan"),
                    "leave_one_out_max": max(loo) if loo else float("nan"),
                    "trim1_mean": float(np.nanmean(np.sort(values[np.isfinite(values)])[1:-1]))
                    if np.isfinite(values).sum() > 2
                    else float("nan"),
                }
            )
    return pd.DataFrame(paired_rows), pd.DataFrame(summary_rows)


def _curve_completeness(label: str, run: dict[str, object]) -> pd.DataFrame:
    curves = run["downstream_curves"].copy()
    rows: list[dict[str, object]] = []
    for method, group in curves[curves["split"].astype(str) == "eval"].groupby("method"):
        for key, sub in group.groupby(DOWNSTREAM_KEYS, sort=False):
            steps = sorted(int(x) for x in _as_num(sub["step"]).dropna().tolist())
            rows.append(
                {
                    "run_label": label,
                    "method": method,
                    "source_weight_index": key[0],
                    "start_index": key[1],
                    "task_name": key[2],
                    "tau": key[3],
                    "complete": steps == EXPECTED_STEPS,
                    "steps": json.dumps(steps),
                }
            )
    return pd.DataFrame(rows)


def _spike_summary(label: str, run: dict[str, object]) -> pd.DataFrame:
    metrics = run["vae_metrics"].copy()
    history = metrics[metrics["record_type"].astype(str).eq("vae_train_history")].copy()
    rows: list[dict[str, object]] = []
    if history.empty:
        return pd.DataFrame(rows)
    derived = {
        "train_precond_grad_ratio": _as_num(history.get("train_precond_grad_norm", pd.Series(dtype=float)))
        / _as_num(history.get("train_precond_base_grad_norm", pd.Series(dtype=float))).replace(0.0, np.nan),
        "train_function_anchor_grad_ratio": _as_num(history.get("train_function_anchor_grad_ratio", pd.Series(dtype=float))),
        "train_block_recon_grad_ratio": _as_num(history.get("train_block_recon_grad_ratio", pd.Series(dtype=float))),
    }
    for name, values in derived.items():
        history[name] = values
    columns = [
        "train_loss",
        "train_recon_mse",
        "val_loss",
        "train_precond_a_loss",
        "train_precond_effective_loss",
        "train_precond_grad_ratio",
        "train_precond_grad_scale",
        "train_function_anchor_loss",
        "train_function_anchor_effective_loss",
        "train_function_anchor_margin_drop",
        "train_function_anchor_margin_drop_active_fraction",
        "train_function_anchor_grad_ratio",
        "train_block_recon_loss",
        "train_block_recon_effective_loss",
        "train_block_recon_grad_ratio",
    ]
    for column in columns:
        if column not in history.columns:
            continue
        q = _quantiles(_as_num(history[column]).tolist())
        row = {"run_label": label, "metric": column, **q}
        median = row["median"]
        row["max_over_median"] = row["max"] / median if np.isfinite(median) and abs(median) > 1e-12 else float("nan")
        if column == "train_precond_grad_scale":
            values = _as_num(history[column]).to_numpy(dtype=float)
            row["frac_lt_1"] = float(np.nanmean(values < 1.0))
            row["frac_le_0p1"] = float(np.nanmean(values <= 0.1))
            row["frac_le_0p01"] = float(np.nanmean(values <= 0.01))
            row["frac_le_0p001"] = float(np.nanmean(values <= 0.001))
        rows.append(row)
    return pd.DataFrame(rows)


def _pareto_summary(label: str, run: dict[str, object]) -> dict[str, object]:
    downstream = run["downstream_results"].copy()
    decoder = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "decoder_latent")]
    raw = downstream[(downstream["split"].astype(str) == "eval") & (downstream["method"].astype(str) == "raw")]
    diagnostics = run["preconditioning_diagnostics"].copy()
    metrics = run["vae_metrics"].copy()
    quality = metrics[~metrics["record_type"].astype(str).eq("vae_train_history")].copy()
    row: dict[str, object] = {
        "run_label": label,
        "decoder_train_aulc_mean": float(_as_num(decoder["aulc"]).mean()),
        "decoder_train_aulc_median": float(_as_num(decoder["aulc"]).median()),
        "raw_train_aulc_mean": float(_as_num(raw["aulc"]).mean()),
        "raw_train_aulc_median": float(_as_num(raw["aulc"]).median()),
        "decoder_final_test_loss_mean": float(_as_num(decoder["final_test_loss"]).mean()),
        "decoder_final_test_acc_mean": float(_as_num(decoder["final_test_acc"]).mean()),
        "diag_li_A_full_p95": float(_as_num(diagnostics["li_A_full_per_dim"]).quantile(0.95)),
        "diag_li_A_full_median": float(_as_num(diagnostics["li_A_full_per_dim"]).median()),
        "diag_hvp_a_loss_p95": float(_as_num(diagnostics["hvp_probe_a_loss_per_dim"]).quantile(0.95)),
        "diag_trace_m_median": float(_as_num(diagnostics["hvp_probe_trace_m_per_dim"]).median()),
    }
    for column in [
        "reconstruction_rel_l2",
        "decoded_test_acc",
        "decoded_test_loss",
        "raw_test_acc",
        "raw_test_loss",
    ]:
        if column in quality.columns:
            row[f"quality_{column}_mean"] = float(_as_num(quality[column]).mean())
            row[f"quality_{column}_median"] = float(_as_num(quality[column]).median())
    return row


def _write_plot_downstream(deltas: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    dec = deltas[deltas["method"] == "decoder_latent"].copy().sort_values("train_mean_aulc_delta")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = dec["task_name"].map({"mnist": "#3572A5", "fashion_mnist": "#C44E52"}).fillna("#666666")
    axes[0, 0].bar(np.arange(len(dec)), dec["train_mean_aulc_delta"], color=colors)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Decoder Paired Train-AULC Delta")
    axes[0, 0].set_ylabel("A - control")
    axes[0, 0].set_xlabel("matched eval start, sorted")
    axes[0, 1].bar(np.arange(len(dec)), dec["test_mean_aulc_delta"], color=colors)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Decoder Paired Test-Mean-AULC Delta")
    axes[0, 1].set_xlabel("matched eval start, same order")
    axes[1, 0].scatter(dec["step0_test_loss_delta"], dec["post0_test_loss_mean_delta"], c=colors, s=55)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Step0 vs Post-Step Test-Loss Delta")
    axes[1, 0].set_xlabel("step0 A - control")
    axes[1, 0].set_ylabel("post0 mean A - control")
    text = summary[summary["method"] == "decoder_latent"].copy()
    axes[1, 1].axis("off")
    lines = []
    for _, row in text.iterrows():
        lines.append(
            f"{row['metric']}: mean={row['mean']:.6g}, med={row['median']:.6g}, "
            f"worse={int(row['worse_count_A_gt_control'])}/{int(row['n'])}"
        )
    axes[1, 1].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_plot_spikes(control: dict[str, object], a_run: dict[str, object], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False, constrained_layout=True)
    for label, run, color in [("control", control, "#4C72B0"), ("A", a_run, "#DD8452")]:
        history = run["vae_metrics"]
        history = history[history["record_type"].astype(str).eq("vae_train_history")].copy()
        if history.empty:
            continue
        step = _as_num(history["step"])
        axes[0].plot(step, _as_num(history["train_loss"]), label=f"{label} train_loss", color=color, alpha=0.75)
        axes[0].plot(step, _as_num(history["val_loss"]), label=f"{label} val_loss", color=color, linestyle="--", alpha=0.75)
        if "train_precond_a_loss" in history.columns:
            axes[1].plot(step, _as_num(history["train_precond_a_loss"]), label=f"{label} A loss", color=color, alpha=0.75)
        if "train_precond_grad_scale" in history.columns:
            axes[2].plot(step, _as_num(history["train_precond_grad_scale"]), label=f"{label} grad_scale", color=color, alpha=0.75)
        if "train_function_anchor_margin_drop" in history.columns:
            axes[2].plot(
                step,
                _as_num(history["train_function_anchor_margin_drop"]),
                label=f"{label} margin_drop",
                color=color,
                linestyle=":",
                alpha=0.65,
            )
    axes[0].set_title("VAE Train/Validation Loss")
    axes[1].set_title("A/HVP Surrogate Tail During Training")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[2].set_title("Trust-Region Scale And Function-Anchor Margin Drop")
    axes[2].set_yscale("symlog", linthresh=1e-4)
    for ax in axes:
        ax.set_xlabel("VAE step")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_plot_pareto(pareto: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for _, row in pareto.iterrows():
        label = str(row["run_label"])
        color = "#4C72B0" if label == "control" else "#DD8452"
        axes[0].scatter(row["diag_li_A_full_p95"], row["decoder_train_aulc_mean"], s=100, color=color)
        axes[0].annotate(label, (row["diag_li_A_full_p95"], row["decoder_train_aulc_mean"]))
        axes[1].scatter(row["quality_decoded_test_acc_mean"], row["decoder_train_aulc_mean"], s=100, color=color)
        axes[1].annotate(label, (row["quality_decoded_test_acc_mean"], row["decoder_train_aulc_mean"]))
    axes[0].set_xlabel("diagnostic li_A p95 / dim")
    axes[0].set_ylabel("decoder downstream train-AULC mean")
    axes[0].set_title("Proxy vs Downstream")
    axes[1].set_xlabel("decoded test accuracy mean")
    axes[1].set_ylabel("decoder downstream train-AULC mean")
    axes[1].set_title("Function Quality vs Downstream")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--a-run", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(
        "start "
        f"control_run={args.control_run} a_run={args.a_run} prefix={args.prefix} out_dir={out_dir}"
    )

    control = _load_run(args.control_run)
    a_run = _load_run(args.a_run)

    completion = pd.DataFrame(_completion_audit("control", control) + _completion_audit("A", a_run))
    completion_path = out_dir / f"{args.prefix}_completion_audit.csv"
    completion.to_csv(completion_path, index=False)

    selected = pd.concat([_selected_lrs("control", control), _selected_lrs("A", a_run)], ignore_index=True)
    selected_path = out_dir / f"{args.prefix}_selected_lrs.csv"
    selected.to_csv(selected_path, index=False)

    start_audit = _start_bank_audit(control, a_run)
    start_audit_path = out_dir / f"{args.prefix}_start_bank_audit.csv"
    start_audit.to_csv(start_audit_path, index=False)

    curve_complete = pd.concat(
        [_curve_completeness("control", control), _curve_completeness("A", a_run)],
        ignore_index=True,
    )
    curve_complete_path = out_dir / f"{args.prefix}_curve_completeness.csv"
    curve_complete.to_csv(curve_complete_path, index=False)

    deltas, downstream_summary = _paired_downstream(control, a_run)
    deltas_path = out_dir / f"{args.prefix}_paired_downstream_deltas.csv"
    summary_path = out_dir / f"{args.prefix}_downstream_summary.csv"
    deltas.to_csv(deltas_path, index=False)
    downstream_summary.to_csv(summary_path, index=False)

    spikes = pd.concat(
        [_spike_summary("control", control), _spike_summary("A", a_run)],
        ignore_index=True,
    )
    spikes_path = out_dir / f"{args.prefix}_spike_summary.csv"
    spikes.to_csv(spikes_path, index=False)

    pareto = pd.DataFrame([_pareto_summary("control", control), _pareto_summary("A", a_run)])
    pareto_path = out_dir / f"{args.prefix}_pareto_summary.csv"
    pareto.to_csv(pareto_path, index=False)

    downstream_plot = out_dir / f"{args.prefix}_paired_downstream_deltas.png"
    spike_plot = out_dir / f"{args.prefix}_vae_spike_traces.png"
    pareto_plot = out_dir / f"{args.prefix}_pareto_proxy_function_downstream.png"
    _write_plot_downstream(deltas, downstream_summary, downstream_plot)
    _write_plot_spikes(control, a_run, spike_plot)
    _write_plot_pareto(pareto, pareto_plot)

    _log("wrote artifacts:")
    for path in [
        completion_path,
        selected_path,
        start_audit_path,
        curve_complete_path,
        deltas_path,
        summary_path,
        spikes_path,
        pareto_path,
        downstream_plot,
        spike_plot,
        pareto_plot,
    ]:
        _log(f"  {path}")
    decoder = downstream_summary[downstream_summary["method"] == "decoder_latent"]
    _log("decoder summary:")
    _log(decoder.to_string(index=False))


if __name__ == "__main__":
    main()
