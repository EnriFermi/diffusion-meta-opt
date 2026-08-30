from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUN = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_tinybigvae_direct_mse_baseline_final_v2_seed0"
)
FINAL_RUN = (
    ROOT
    / "artifacts/loss_landscape_analysis/sage_cnn_vae_smoothing"
    / "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_verified_nocap_v1_trainseed1"
)
AUDIT_ROOT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "verified_a_finetune_h2048"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pair(initial_path: Path, final_path: Path, *, bank: str) -> pd.DataFrame:
    columns = ["weight_index", "task_name", "tau", "li_A_full_per_dim", "li_trace_h2_per_dim", "task_loss"]
    initial = pd.read_csv(initial_path)[columns].copy()
    final = pd.read_csv(final_path)[columns].copy()
    initial = initial.rename(columns={column: f"initial_{column}" for column in columns if column != "weight_index"})
    final = final.rename(columns={column: f"final_{column}" for column in columns if column != "weight_index"})
    paired = initial.merge(final, on="weight_index", how="outer", validate="one_to_one", indicator=True)
    if len(paired) != 24 or not paired["_merge"].eq("both").all():
        raise ValueError(f"{bank}: expected 24 matched rows, got merge={paired['_merge'].value_counts().to_dict()}")
    if not paired["initial_task_name"].eq(paired["final_task_name"]).all():
        raise ValueError(f"{bank}: task-name mismatch")
    if not np.allclose(paired["initial_tau"], paired["final_tau"], rtol=0.0, atol=0.0):
        raise ValueError(f"{bank}: tau mismatch")
    paired = paired.drop(columns="_merge")
    paired.insert(0, "bank", bank)
    paired["a_delta"] = paired["final_li_A_full_per_dim"] - paired["initial_li_A_full_per_dim"]
    paired["a_ratio"] = paired["final_li_A_full_per_dim"] / paired["initial_li_A_full_per_dim"]
    return paired


def _bootstrap_upper(delta: np.ndarray, *, seed: int, draws: int = 200_000) -> float:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(delta), size=(draws, len(delta)))
    means = delta[indices].mean(axis=1)
    return float(np.quantile(means, 0.95))


def _bank_summary(frame: pd.DataFrame, *, seed: int) -> dict[str, float | int | bool]:
    initial = frame["initial_li_A_full_per_dim"].to_numpy(dtype=np.float64)
    final = frame["final_li_A_full_per_dim"].to_numpy(dtype=np.float64)
    delta = final - initial
    loo_ratios = []
    for dropped in range(len(frame)):
        keep = np.arange(len(frame)) != dropped
        loo_ratios.append(float(final[keep].mean() / initial[keep].mean()))
    result: dict[str, float | int | bool] = {
        "rows": int(len(frame)),
        "initial_mean": float(initial.mean()),
        "final_mean": float(final.mean()),
        "mean_ratio": float(final.mean() / initial.mean()),
        "initial_median": float(np.median(initial)),
        "final_median": float(np.median(final)),
        "initial_p90": float(np.quantile(initial, 0.90)),
        "final_p90": float(np.quantile(final, 0.90)),
        "improved_fraction": float(np.mean(delta < 0.0)),
        "paired_median_delta": float(np.median(delta)),
        "bootstrap_mean_delta_upper_95": _bootstrap_upper(delta, seed=seed),
        "worst_leave_one_out_mean_ratio": float(max(loo_ratios)),
        "all_finite": bool(np.isfinite(initial).all() and np.isfinite(final).all()),
    }
    result["gate_mean_ratio"] = bool(result["mean_ratio"] <= 0.50)
    result["gate_p90"] = bool(result["final_p90"] <= result["initial_p90"])
    result["gate_improved_fraction"] = bool(result["improved_fraction"] >= 0.60)
    result["gate_median_delta"] = bool(result["paired_median_delta"] < 0.0)
    result["gate_leave_one_out"] = bool(result["worst_leave_one_out_mean_ratio"] < 0.90)
    return result


def _final_val_recon(run_dir: Path) -> float:
    metrics_path = run_dir / "vae_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, low_memory=False)
        history = metrics.loc[metrics["record_type"].eq("vae_train_history")]
        values = pd.to_numeric(history["val_recon_mse"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[-1])

    # The checkpoint is written before optional downstream evaluation and already
    # contains the complete training history.
    import torch

    checkpoint = torch.load(run_dir / "vae_checkpoint.pt", map_location="cpu", weights_only=False)
    values = [
        float(row["val_recon_mse"])
        for row in checkpoint.get("metrics", [])
        if np.isfinite(row.get("val_recon_mse", np.nan))
    ]
    if not values:
        raise ValueError(f"{run_dir}: no validation reconstruction history")
    return values[-1]


def _split_audit(baseline_run: Path, final_run: Path, audit_indices: set[int]) -> dict[str, int | bool]:
    import torch

    baseline = torch.load(baseline_run / "vae_checkpoint.pt", map_location="cpu", weights_only=False)
    final = torch.load(final_run / "vae_checkpoint.pt", map_location="cpu", weights_only=False)
    baseline_train = baseline["train_indices"].detach().cpu().to(torch.int64)
    baseline_val = baseline["val_indices"].detach().cpu().to(torch.int64)
    final_train = final["train_indices"].detach().cpu().to(torch.int64)
    final_val = final["val_indices"].detach().cpu().to(torch.int64)
    train = set(int(value) for value in baseline_train.tolist())
    val = set(int(value) for value in baseline_val.tolist())
    return {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_val_overlap": len(train & val),
        "audit_rows": len(audit_indices),
        "audit_in_train": len(audit_indices & train),
        "audit_in_val": len(audit_indices & val),
        "same_train_indices": bool(torch.equal(baseline_train, final_train)),
        "same_val_indices": bool(torch.equal(baseline_val, final_val)),
        "passed": bool(
            torch.equal(baseline_train, final_train)
            and torch.equal(baseline_val, final_val)
            and not (train & val)
            and not (audit_indices & train)
            and audit_indices <= val
        ),
    }


def _plot(paired: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for axis, (bank, group) in zip(axes, paired.groupby("bank", sort=True), strict=True):
        initial = group["initial_li_A_full_per_dim"]
        final = group["final_li_A_full_per_dim"]
        low = float(min(initial.min(), final.min())) * 0.8
        high = float(max(initial.max(), final.max())) * 1.25
        axis.scatter(initial, final, color="#2563eb", s=36, alpha=0.82)
        axis.plot([low, high], [low, high], color="#111827", linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_title(f"{bank}: fixed exact A")
        axis.set_xlabel("initial li_A_full_per_dim")
        axis.set_ylabel("final li_A_full_per_dim")
        axis.grid(alpha=0.22, which="both")
    fig.suptitle("Variant A fine-tune: identical source and CE-batch audits")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the pre-registered exact-A acceptance gate.")
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--final-run", type=Path, default=FINAL_RUN)
    parser.add_argument("--bank-a-initial", type=Path, default=AUDIT_ROOT / "initial_fixed24_exact/a_outlier_diagnostics.csv")
    parser.add_argument("--bank-a-final", type=Path, default=FINAL_RUN / "preconditioning_diagnostics.csv")
    parser.add_argument("--bank-a-label", default="bank_a")
    parser.add_argument(
        "--bank-b-initial",
        type=Path,
        default=AUDIT_ROOT / "preflight_old_nocap_bank_b_initial/a_outlier_diagnostics.csv",
    )
    parser.add_argument("--bank-b-final", type=Path, default=AUDIT_ROOT / "final_bank_b_exact/a_outlier_diagnostics.csv")
    parser.add_argument("--bank-b-label", default="bank_b")
    parser.add_argument("--output-dir", type=Path, default=AUDIT_ROOT / "acceptance")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.bank_a_label == args.bank_b_label:
        raise ValueError("bank labels must differ")
    paired_a = _load_pair(args.bank_a_initial, args.bank_a_final, bank=args.bank_a_label)
    paired_b = _load_pair(args.bank_b_initial, args.bank_b_final, bank=args.bank_b_label)
    if set(paired_a["weight_index"]) & set(paired_b["weight_index"]):
        raise ValueError("audit banks overlap")
    paired = pd.concat([paired_a, paired_b], ignore_index=True)
    paired_path = args.output_dir / "paired_exact_a.csv"
    paired.to_csv(paired_path, index=False)

    bank_summaries = {
        args.bank_a_label: _bank_summary(paired_a, seed=2026071201),
        args.bank_b_label: _bank_summary(paired_b, seed=2026071202),
    }
    combined_bootstrap_upper = _bootstrap_upper(paired["a_delta"].to_numpy(dtype=np.float64), seed=2026071248)
    initial_recon = _final_val_recon(args.baseline_run)
    final_recon = _final_val_recon(args.final_run)
    reconstruction_ratio = final_recon / initial_recon
    split_audit = _split_audit(
        args.baseline_run,
        args.final_run,
        set(int(value) for value in paired["weight_index"]),
    )
    gates = {
        args.bank_a_label: all(
            bool(bank_summaries[args.bank_a_label][key])
            for key in (
                "gate_mean_ratio",
                "gate_p90",
                "gate_improved_fraction",
                "gate_median_delta",
                "gate_leave_one_out",
                "all_finite",
            )
        ),
        args.bank_b_label: all(
            bool(bank_summaries[args.bank_b_label][key])
            for key in (
                "gate_mean_ratio",
                "gate_p90",
                "gate_improved_fraction",
                "gate_median_delta",
                "gate_leave_one_out",
                "all_finite",
            )
        ),
        "combined_bootstrap": bool(combined_bootstrap_upper < 0.0),
        "reconstruction": bool(np.isfinite(reconstruction_ratio) and reconstruction_ratio <= 1.10),
        "held_out_split": bool(split_audit["passed"]),
    }
    passed = bool(all(gates.values()))
    plot_path = args.output_dir / "paired_exact_a.png"
    _plot(paired, plot_path)
    source_paths = {
        f"{args.bank_a_label}_initial": str(args.bank_a_initial.resolve()),
        f"{args.bank_a_label}_final": str(args.bank_a_final.resolve()),
        f"{args.bank_b_label}_initial": str(args.bank_b_initial.resolve()),
        f"{args.bank_b_label}_final": str(args.bank_b_final.resolve()),
    }
    summary = {
        "passed": passed,
        "gates": gates,
        "bank_summaries": bank_summaries,
        "combined_bootstrap_mean_delta_upper_95": combined_bootstrap_upper,
        "initial_val_recon_mse": initial_recon,
        "final_val_recon_mse": final_recon,
        "reconstruction_ratio": reconstruction_ratio,
        "split_audit": split_audit,
        "baseline_checkpoint_sha256": _sha256(args.baseline_run / "vae_checkpoint.pt"),
        "final_checkpoint_sha256": _sha256(args.final_run / "vae_checkpoint.pt"),
        "sources": {
            label: {"path": path, "sha256": _sha256(Path(path))} for label, path in source_paths.items()
        },
        "outputs": {"paired_csv": str(paired_path), "plot": str(plot_path)},
    }
    summary_path = args.output_dir / "acceptance.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[verified_a_acceptance] "
        f"passed={passed} {args.bank_a_label}_ratio={bank_summaries[args.bank_a_label]['mean_ratio']:.6g} "
        f"{args.bank_b_label}_ratio={bank_summaries[args.bank_b_label]['mean_ratio']:.6g} "
        f"combined_bootstrap_upper={combined_bootstrap_upper:.6g} recon_ratio={reconstruction_ratio:.6g} "
        f"summary={summary_path} plot={plot_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
