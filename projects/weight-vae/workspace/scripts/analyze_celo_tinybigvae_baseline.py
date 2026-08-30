from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _median(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").median())


def _p90(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(0.9))


def _pass(name: str, value: Any, passed: bool, detail: str, *, severity: str = "acceptance") -> dict[str, Any]:
    return {
        "check": name,
        "value": value,
        "passed": bool(passed),
        "detail": detail,
        "severity": severity,
    }


def _diagnostic(name: str, value: Any, detail: str) -> dict[str, Any]:
    return _pass(name, value, True, detail, severity="diagnostic")


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _load_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    return pd.read_csv(path)


def analyze(output_dir: Path) -> tuple[Path, bool]:
    output_dir = output_dir.expanduser().resolve()
    figures_dir = output_dir / "figures"
    report_path = output_dir / "baseline_report.md"

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    weight_records = _load_csv(output_dir, "weight_pool_records.csv")
    vae_metrics = _load_csv(output_dir, "vae_metrics.csv")
    geometry = _load_csv(output_dir, "geometry.csv")
    selected_lrs = _load_csv(output_dir, "selected_lrs.csv")
    downstream_results = _load_csv(output_dir, "downstream_results.csv")
    downstream_curves = _load_csv(output_dir, "downstream_curves.csv")

    checks: list[dict[str, Any]] = []
    cfg = config.get("config", config)
    expected_snapshots = int(cfg["weight_runs"]) * (int(cfg["weight_train_steps"]) // int(cfg["weight_snapshot_every"]) + 1)
    checks.append(_pass("weight_pool_record_count", len(weight_records), len(weight_records) == expected_snapshots, f"expected {expected_snapshots}"))
    checks.append(_pass("weight_pool_finite", "finite", weight_records[["train_loss", "test_loss"]].notna().all().all(), "train/test losses are present"))
    checks.append(_pass("tau_range", f"{weight_records['tau'].min():.4g}..{weight_records['tau'].max():.4g}", weight_records["tau"].between(float(cfg["celo_tau_min"]), float(cfg["celo_tau_max"])).all(), "all source taus inside config range"))
    checks.append(_pass("tiny_bigvae_direct_decoder", cfg.get("tiny_bigvae_output_mode"), cfg.get("tiny_bigvae_output_mode") == "direct", "CELO baseline uses scalable direct patch decoder, not direction/scale patch bottleneck"))

    final_step = int(weight_records["step"].max())
    final_records = weight_records[weight_records["step"] == final_step].copy()
    coverage = final_records.groupby(["task_name", "source_lr"]).size()
    min_coverage = int(coverage.min()) if not coverage.empty else 0
    checks.append(_pass("source_task_lr_coverage", min_coverage, min_coverage >= 40, "at least 40 final runs per task/LR bucket"))

    task_acc = final_records.groupby("task_name")["test_acc"].median().to_dict()
    mnist_ok = float(task_acc.get("mnist", 0.0)) >= 0.85
    fashion_ok = float(task_acc.get("fashion_mnist", 0.0)) >= 0.75
    checks.append(_pass("source_mnist_acc", task_acc.get("mnist", float("nan")), mnist_ok, "median final MNIST source test acc >= 0.85"))
    checks.append(_pass("source_fashion_acc", task_acc.get("fashion_mnist", float("nan")), fashion_ok, "median final FashionMNIST source test acc >= 0.75"))
    checks.append(_pass("source_pooled_acc", _median(final_records["test_acc"]), _median(final_records["test_acc"]) >= 0.80, "pooled final source test acc >= 0.80"))

    train_history = vae_metrics[vae_metrics["record_type"] == "vae_train_history"].copy()
    quality = vae_metrics[vae_metrics["record_type"] == "vae_quality"].copy()
    final_val_recon = float(pd.to_numeric(train_history["val_recon_mse"], errors="coerce").dropna().iloc[-1]) if not train_history.empty else float("inf")
    final_train_recon = float(pd.to_numeric(train_history["train_recon_mse"], errors="coerce").dropna().iloc[-1]) if not train_history.empty else float("inf")
    gap = final_val_recon / max(final_train_recon, 1e-12)
    checks.append(_pass("vae_val_recon_mse", final_val_recon, final_val_recon <= 0.10, "final val normalized reconstruction MSE <= 0.10"))
    checks.append(_pass("vae_train_val_gap", gap, gap <= 2.0, "final val/train recon MSE gap <= 2x"))

    median_rel = _median(quality["reconstruction_rel_l2"]) if not quality.empty else float("inf")
    p90_rel = _p90(quality["reconstruction_rel_l2"]) if not quality.empty else float("inf")
    checks.append(_pass("heldout_reconstruction_rel_l2_median", median_rel, median_rel <= 0.50, "median heldout relative L2 <= 0.50"))
    checks.append(_pass("heldout_reconstruction_rel_l2_p90", p90_rel, p90_rel <= 0.80, "p90 heldout relative L2 <= 0.80"))

    if not quality.empty and {"decoded_test_acc", "raw_test_acc"}.issubset(quality.columns):
        acc_gap = pd.to_numeric(quality["raw_test_acc"], errors="coerce") - pd.to_numeric(quality["decoded_test_acc"], errors="coerce")
        loss_gap = pd.to_numeric(quality["decoded_test_loss"], errors="coerce") - pd.to_numeric(quality["raw_test_loss"], errors="coerce")
        checks.append(_pass("decoded_test_acc_gap_median", _median(acc_gap), _median(acc_gap) <= 0.10, "median raw-decoded test acc gap <= 0.10"))
        checks.append(_pass("decoded_test_loss_gap_median", _median(loss_gap), _median(loss_gap) <= 0.20, "median decoded-raw test loss gap <= 0.20"))
    if not quality.empty and {"normalized_reconstruction_bias_mse", "normalized_reconstruction_weight_mse"}.issubset(quality.columns):
        bias_ratio = _median(quality["normalized_reconstruction_bias_mse"]) / max(_median(quality["normalized_reconstruction_weight_mse"]), 1e-12)
        checks.append(_pass("bias_reconstruction_ratio", bias_ratio, bias_ratio <= 2.0, "normalized bias MSE <= 2x normalized weight MSE"))

    if not geometry.empty:
        cond_median = _median(geometry["condition_median"])
        cond_p90 = _median(geometry["condition_p90"])
        eig_min = _median(geometry["eig_min_median"])
        log_spread = _median(geometry["log_eig_spread_median"])
        checks.append(_diagnostic("geometry_condition_median", cond_median, "decoder metric condition median; diagnostic for unregularized baseline anisotropy"))
        checks.append(_diagnostic("geometry_condition_p90", cond_p90, "decoder metric condition p90; non-blocking because this baseline intentionally has no geometry/smoothing regularizer"))
        checks.append(_diagnostic("geometry_log_eig_spread_median", log_spread, "median log eig spread; diagnostic for future smoothing-loss comparisons"))
        checks.append(_pass("geometry_eig_min", eig_min, eig_min > 1e-8, "decoder geometry eig_min_median > 1e-8"))

    lr_boundary_ok = True
    selected = selected_lrs[selected_lrs.get("selected", 0) == 1].copy()
    for method, group in selected.groupby("method"):
        lr = float(group["candidate_lr"].iloc[0])
        candidates = selected_lrs[selected_lrs["method"] == method]["candidate_lr"].astype(float)
        if lr <= float(candidates.min()) or lr >= float(candidates.max()):
            lr_boundary_ok = False
    checks.append(_pass("selected_lrs_not_boundary", selected[["method", "candidate_lr"]].to_dict("records"), lr_boundary_ok, "selected raw/latent LRs are not at grid min/max"))

    diverged_rate = downstream_results.groupby("method")["diverged"].mean().to_dict() if not downstream_results.empty else {}
    checks.append(_pass("downstream_no_divergence", diverged_rate, all(float(v) == 0.0 for v in diverged_rate.values()), "all downstream eval curves finite"))
    median_aulc = downstream_results.groupby("method")["aulc"].median().to_dict() if not downstream_results.empty else {}
    raw_aulc = float(median_aulc.get("raw", float("inf")))
    latent_aulc = float(median_aulc.get("decoder_latent", float("inf")))
    checks.append(_pass("latent_aulc_close_to_raw", latent_aulc - raw_aulc, latent_aulc <= raw_aulc + 0.10, "decoder latent median AULC within +0.10 CE of raw"))

    if not train_history.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(train_history["step"], train_history["train_recon_mse"], label="train")
        plt.plot(train_history["step"], train_history["val_recon_mse"], label="val")
        plt.yscale("log")
        plt.xlabel("VAE step")
        plt.ylabel("normalized reconstruction MSE")
        plt.legend()
        _savefig(figures_dir / "vae_recon_mse.png")

    if not final_records.empty:
        plt.figure(figsize=(8, 4))
        labels = []
        data = []
        for key, group in final_records.groupby(["task_name", "source_lr"]):
            labels.append(f"{key[0]}\\n{key[1]:.0e}")
            data.append(group["test_acc"].astype(float).to_numpy())
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("final source test acc")
        plt.xticks(rotation=45, ha="right")
        _savefig(figures_dir / "source_final_acc.png")

    if not quality.empty:
        plt.figure(figsize=(7, 4))
        for task_name, group in quality.groupby("task_name"):
            plt.hist(group["reconstruction_rel_l2"].astype(float), bins=30, alpha=0.45, label=str(task_name))
        plt.xlabel("heldout reconstruction relative L2")
        plt.ylabel("count")
        plt.legend()
        _savefig(figures_dir / "reconstruction_rel_l2.png")

        plt.figure(figsize=(5, 5))
        plt.scatter(quality["raw_test_acc"], quality["decoded_test_acc"], alpha=0.7)
        lim = [0.0, max(float(quality["raw_test_acc"].max()), float(quality["decoded_test_acc"].max()), 1e-6)]
        plt.plot(lim, lim, color="black", linewidth=1)
        plt.xlabel("raw test acc")
        plt.ylabel("decoded test acc")
        _savefig(figures_dir / "decoded_vs_raw_acc.png")

    if not downstream_results.empty:
        plt.figure(figsize=(6, 4))
        labels = []
        data = []
        for method, group in downstream_results.groupby("method"):
            labels.append(str(method))
            data.append(group["aulc"].astype(float).to_numpy())
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("eval AULC")
        _savefig(figures_dir / "downstream_aulc.png")

    if not downstream_curves.empty:
        plt.figure(figsize=(7, 4))
        for method, group in downstream_curves.groupby("method"):
            curve = group.groupby("step")["train_loss"].median()
            plt.plot(curve.index, curve.values, label=str(method))
        plt.xlabel("downstream step")
        plt.ylabel("median train loss")
        plt.yscale("log")
        plt.legend()
        _savefig(figures_dir / "downstream_train_loss.png")

    passed = all(item["passed"] for item in checks if item.get("severity", "acceptance") == "acceptance")
    lines = [
        "# CELO TinyBigVAE Baseline Report",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Overall status: `{'PASS' if passed else 'FAIL'}`",
        f"- Run label: `{cfg.get('run_label')}`",
        f"- Config hash: `{manifest.get('config_hash', config.get('config_hash', 'unknown'))}`",
        f"- Device: `{manifest.get('cuda_device_name', 'unknown')}`",
        "",
        "## Config Summary",
        "",
        f"- Tasks: `{tuple(cfg.get('celo_tasks', ()))}`",
        f"- Source model: image `{cfg.get('celo_image_size')}`, hidden `{cfg.get('celo_hidden_dim')}`, tau `{cfg.get('celo_tau_min')}..{cfg.get('celo_tau_max')}`",
        f"- Weight pool: `{cfg.get('weight_runs')}` runs, `{cfg.get('weight_train_steps')}` steps, snapshot every `{cfg.get('weight_snapshot_every')}`",
        f"- VAE: `{cfg.get('vae_arch')}`, latent `{cfg.get('latent_dim')}`, hidden `{cfg.get('vae_hidden_dim')}`, patch `{cfg.get('tiny_bigvae_patch_size')}`, token `{cfg.get('tiny_bigvae_token_dim')}`, output `{cfg.get('tiny_bigvae_output_mode')}`",
        f"- Loss: `{cfg.get('vae_loss_kind')}` + beta_kl `{cfg.get('beta_kl')}`",
        f"- Geometry: `{cfg.get('geometry_eval_samples')}` samples, Jacobian chunk `{cfg.get('geometry_jacobian_chunk_size')}`",
        "",
        "## Acceptance Checks",
        "",
        "| Check | Value | Status | Detail |",
        "|---|---:|---:|---|",
    ]
    for item in checks:
        if item.get("severity", "acceptance") == "diagnostic":
            status = "DIAG"
        else:
            status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"| `{item['check']}` | `{item['value']}` | `{status}` | {item['detail']} |")

    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![VAE reconstruction MSE](figures/vae_recon_mse.png)",
            "",
            "![Source final accuracy](figures/source_final_acc.png)",
            "",
            "![Heldout reconstruction relative L2](figures/reconstruction_rel_l2.png)",
            "",
            "![Decoded vs raw accuracy](figures/decoded_vs_raw_acc.png)",
            "",
            "![Downstream AULC](figures/downstream_aulc.png)",
            "",
            "![Downstream train loss](figures/downstream_train_loss.png)",
            "",
            "## Geometry Note",
            "",
            "The plain CELO baseline does not include a geometry or smoothing regularizer. Decoder anisotropy is therefore reported as a diagnostic rather than a blocking criterion; it is one of the quantities future smoothing-loss variants should improve. The blocking representation checks are heldout decode quality and downstream latent optimization versus raw weights.",
            "",
            "## Key Tables",
            "",
            "### Median Source Final Test Accuracy",
            "",
            "```text",
            final_records.groupby("task_name")["test_acc"].median().to_string() if not final_records.empty else "No source records.",
            "```",
            "",
            "### Median Downstream AULC",
            "",
            "```text",
            pd.Series(median_aulc, name="median_aulc").to_string() if median_aulc else "No downstream results.",
            "```",
            "",
        ]
    )
    report_path.write_text("\\n".join(lines), encoding="utf-8")
    (output_dir / "baseline_acceptance.json").write_text(json.dumps({"passed": passed, "checks": checks}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report_path, passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report_path, passed = analyze(args.output_dir)
    print(f"[baseline-analysis] report={report_path} status={'PASS' if passed else 'FAIL'}", flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
