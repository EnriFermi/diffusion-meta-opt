from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
DEFAULT_OUTPUT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/clean_harness_analysis/cross_variant_head_splice"
)

RUN_PAIRS: dict[str, tuple[str, str]] = {
    "c0": (
        "sage_cnn_vae_smoothing_celo_meta_control_clean_harness_v1_seed0",
        "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_clean_harness_v1_seed0",
    ),
    "c1e-4": (
        "sage_cnn_vae_smoothing_celo_meta_control_logit_anchor_c0p0001_clean_harness_v1_seed0",
        "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_logit_anchor_c0p0001_clean_harness_v1_seed0",
    ),
    "c3e-4": (
        "sage_cnn_vae_smoothing_celo_meta_control_logit_anchor_c0p0003_clean_harness_v1_seed0",
        "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_logit_anchor_c0p0003_clean_harness_v1_seed0",
    ),
}


def _load_cfg(output_dir: Path, *, device: str) -> ExperimentConfig:
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{output_dir / 'config.json'} does not contain a config mapping")
    values = dict(raw_cfg)
    values["device"] = str(device)
    values["dtype"] = "float32"
    return ExperimentConfig(**values)


def _load_run(output_dir: Path, *, device: str) -> dict[str, Any]:
    cfg = _load_cfg(output_dir, device=device)
    dev = torch.device(cfg.device)
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
        "weights": weights,
        "records": pd.DataFrame(weight_payload["records"]),
        "spec": spec,
        "normalizer": normalizer,
        "vae": vae,
        "task_tensors": task_tensors,
        "results": pd.read_csv(output_dir / "downstream_results.csv"),
    }


def _task_tensor_set(task_tensors: dict[str, Any], task_name: str):
    if task_name in task_tensors:
        return task_tensors[task_name]
    if len(task_tensors) == 1:
        return next(iter(task_tensors.values()))
    raise KeyError(f"task {task_name!r} not found")


def _loss_acc(flat: torch.Tensor, *, task_set, spec, split: str, tau: float) -> tuple[float, float]:
    if split == "train":
        images, labels = task_set.train_images, task_set.train_labels
    elif split == "test":
        images, labels = task_set.test_images, task_set.test_labels
    else:
        raise ValueError(split)
    logits = logits_from_flat(flat, images, spec, tau=float(tau))
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return float(loss.detach().cpu().item()), float(acc.detach().cpu().item())


def _spec_slices(spec) -> dict[str, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    for key, size in zip(spec.keys, spec.sizes, strict=True):
        slices[str(key)] = slice(offset, offset + int(size))
        offset += int(size)
    return slices


def _block_groups(spec) -> OrderedDict[str, tuple[str, ...]]:
    keys = tuple(str(k) for k in spec.keys)
    groups: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    fc2 = tuple(k for k in keys if k.startswith("fc2."))
    weights = tuple(k for k in keys if k.endswith(".weight"))
    biases = tuple(k for k in keys if k.endswith(".bias"))
    if "fc2.weight" in keys:
        groups["fc2.weight"] = ("fc2.weight",)
    if fc2:
        groups["classifier_head"] = fc2
    if weights:
        groups["all_weight_tensors"] = weights
    if biases:
        groups["all_bias_tensors"] = biases
    groups["all_tensors"] = keys
    return groups


def _splice(base: torch.Tensor, donor: torch.Tensor, slices: dict[str, slice], keys: tuple[str, ...]) -> torch.Tensor:
    result = base.detach().clone()
    for key in keys:
        result[slices[key]] = donor.detach()[slices[key]]
    return result


def _safe_fraction(num: float, den: float) -> float:
    if abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


def _eval_starts(run: dict[str, Any], *, samples: int, source_indices: set[int]) -> pd.DataFrame:
    starts = run["results"][
        (run["results"]["split"].astype(str) == "eval")
        & (run["results"]["method"].astype(str) == "decoder_latent")
    ].copy()
    starts = starts.sort_values("start_index").reset_index(drop=True)
    if source_indices:
        starts = starts[starts["source_weight_index"].astype(int).isin(source_indices)].reset_index(drop=True)
    elif int(samples) > 0:
        starts = starts.iloc[: int(samples)].copy()
    return starts


def _assert_common_start_bank(control_starts: pd.DataFrame, a_starts: pd.DataFrame, *, pair: str) -> None:
    cols = ["start_index", "source_weight_index", "task_name", "tau"]
    left = control_starts[cols].reset_index(drop=True)
    right = a_starts[cols].reset_index(drop=True)
    if not left.equals(right):
        merged = pd.concat({"control": left, "A": right}, axis=1)
        raise ValueError(f"start bank mismatch for pair={pair}\n{merged}")


def _start_group(task_name: str) -> str:
    return "fashion" if str(task_name) == "fashion_mnist" else "mnist"


def _pair_rows(
    *,
    pair: str,
    control_dir: Path,
    a_dir: Path,
    device: str,
    samples: int,
    source_indices: set[int],
) -> list[dict[str, Any]]:
    print(
        "[cross_head_splice] load "
        f"pair={pair} control={control_dir.name} A={a_dir.name} device={device}",
        flush=True,
    )
    control = _load_run(control_dir, device=device)
    a_run = _load_run(a_dir, device=device)
    print(
        "[cross_head_splice] configs "
        f"pair={pair} control_hash={config_hash(control['cfg'])} A_hash={config_hash(a_run['cfg'])}",
        flush=True,
    )
    if list(control["spec"].keys) != list(a_run["spec"].keys):
        raise ValueError(f"spec key mismatch for pair={pair}")
    if control["weights"].shape != a_run["weights"].shape:
        raise ValueError(f"weight shape mismatch for pair={pair}")
    max_weight_diff = float((control["weights"] - a_run["weights"]).abs().max().detach().cpu().item())
    if max_weight_diff > 1e-6:
        raise ValueError(f"weight pool mismatch for pair={pair}: max abs diff={max_weight_diff}")

    control_starts = _eval_starts(control, samples=samples, source_indices=source_indices)
    a_starts = _eval_starts(a_run, samples=samples, source_indices=source_indices)
    _assert_common_start_bank(control_starts, a_starts, pair=pair)
    print(
        "[cross_head_splice] starts "
        f"pair={pair} count={len(control_starts)} indices={control_starts['source_weight_index'].astype(int).tolist()}",
        flush=True,
    )
    slices = _spec_slices(control["spec"])
    rows: list[dict[str, Any]] = []
    for pos, start_row in control_starts.iterrows():
        source_weight_index = int(start_row["source_weight_index"])
        record = control["records"].iloc[source_weight_index].to_dict()
        task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
        tau = float(record.get("tau", start_row.get("tau", 1.0)))
        task_set = _task_tensor_set(control["task_tensors"], task_name)
        w0 = control["weights"][source_weight_index].detach()
        print(
            "[cross_head_splice] probe "
            f"pair={pair} {pos + 1}/{len(control_starts)} source={source_weight_index} task={task_name}",
            flush=True,
        )
        with torch.no_grad():
            z_control = encode_weights(control["vae"], control["normalizer"], w0.reshape(1, -1)).squeeze(0).detach()
            z_a = encode_weights(a_run["vae"], a_run["normalizer"], w0.reshape(1, -1)).squeeze(0).detach()
            dec_control = decode_weights(control["vae"], control["normalizer"], z_control.reshape(1, -1)).squeeze(0)
            dec_a = decode_weights(a_run["vae"], a_run["normalizer"], z_a.reshape(1, -1)).squeeze(0)
            raw_train_loss, raw_train_acc = _loss_acc(w0, task_set=task_set, spec=control["spec"], split="train", tau=tau)
            raw_test_loss, raw_test_acc = _loss_acc(w0, task_set=task_set, spec=control["spec"], split="test", tau=tau)
            c_train_loss, c_train_acc = _loss_acc(
                dec_control, task_set=task_set, spec=control["spec"], split="train", tau=tau
            )
            c_test_loss, c_test_acc = _loss_acc(
                dec_control, task_set=task_set, spec=control["spec"], split="test", tau=tau
            )
            a_train_loss, a_train_acc = _loss_acc(dec_a, task_set=task_set, spec=control["spec"], split="train", tau=tau)
            a_test_loss, a_test_acc = _loss_acc(dec_a, task_set=task_set, spec=control["spec"], split="test", tau=tau)
        gap_train = a_train_loss - c_train_loss
        gap_test = a_test_loss - c_test_loss
        for group_name, keys in _block_groups(control["spec"]).items():
            candidates = {
                "control_block_into_A_decoded": _splice(dec_a, dec_control, slices, keys),
                "A_block_into_control_decoded": _splice(dec_control, dec_a, slices, keys),
            }
            for action, candidate in candidates.items():
                with torch.no_grad():
                    train_loss, train_acc = _loss_acc(
                        candidate, task_set=task_set, spec=control["spec"], split="train", tau=tau
                    )
                    test_loss, test_acc = _loss_acc(
                        candidate, task_set=task_set, spec=control["spec"], split="test", tau=tau
                    )
                rows.append(
                    {
                        "pair": pair,
                        "source_weight_index": source_weight_index,
                        "start_index": int(start_row["start_index"]),
                        "task_name": task_name,
                        "start_group": _start_group(task_name),
                        "tau": tau,
                        "block_group": group_name,
                        "block_keys": ";".join(keys),
                        "action": action,
                        "raw_train_loss": raw_train_loss,
                        "raw_test_loss": raw_test_loss,
                        "control_decoded_train_loss": c_train_loss,
                        "control_decoded_test_loss": c_test_loss,
                        "A_decoded_train_loss": a_train_loss,
                        "A_decoded_test_loss": a_test_loss,
                        "A_minus_control_train_loss": gap_train,
                        "A_minus_control_test_loss": gap_test,
                        "candidate_train_loss": train_loss,
                        "candidate_test_loss": test_loss,
                        "candidate_train_acc": train_acc,
                        "candidate_test_acc": test_acc,
                        "raw_train_acc": raw_train_acc,
                        "raw_test_acc": raw_test_acc,
                        "control_decoded_train_acc": c_train_acc,
                        "control_decoded_test_acc": c_test_acc,
                        "A_decoded_train_acc": a_train_acc,
                        "A_decoded_test_acc": a_test_acc,
                        "control_block_rescue_fraction_train": _safe_fraction(a_train_loss - train_loss, gap_train),
                        "control_block_rescue_fraction_test": _safe_fraction(a_test_loss - test_loss, gap_test),
                        "A_block_injection_fraction_train": _safe_fraction(train_loss - c_train_loss, gap_train),
                        "A_block_injection_fraction_test": _safe_fraction(test_loss - c_test_loss, gap_test),
                    }
                )
    return rows


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["pair", "start_group", "block_group", "action"], as_index=False)
    return grouped.agg(
        starts=("source_weight_index", "nunique"),
        A_minus_control_test_loss_mean=("A_minus_control_test_loss", "mean"),
        A_minus_control_test_loss_median=("A_minus_control_test_loss", "median"),
        candidate_test_loss_mean=("candidate_test_loss", "mean"),
        control_block_rescue_fraction_test_median=("control_block_rescue_fraction_test", "median"),
        control_block_rescue_fraction_test_mean=("control_block_rescue_fraction_test", "mean"),
        A_block_injection_fraction_test_median=("A_block_injection_fraction_test", "median"),
        A_block_injection_fraction_test_mean=("A_block_injection_fraction_test", "mean"),
        A_minus_control_test_acc_mean=("A_decoded_test_acc", "mean"),
        control_decoded_test_acc_mean=("control_decoded_test_acc", "mean"),
    )


def _write_plots(rows: pd.DataFrame, summary: pd.DataFrame, output_root: Path) -> None:
    order = ["c0", "c1e-4", "c3e-4"]
    pair_labels = [p for p in order if p in set(rows["pair"].astype(str))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    base = rows[
        (rows["action"] == "control_block_into_A_decoded")
        & (rows["block_group"] == "classifier_head")
    ].drop_duplicates(["pair", "source_weight_index"])
    means = base.groupby("pair")["A_minus_control_test_loss"].mean().reindex(pair_labels)
    axes[0].bar(pair_labels, means.to_numpy(), color="#e15759")
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("A decoded - control decoded test loss")
    axes[0].set_ylabel("mean step0 test-loss gap")
    for i, value in enumerate(means.to_numpy()):
        axes[0].text(i, value, f"{value:.3g}", ha="center", va="bottom" if value >= 0 else "top")

    rescue = summary[
        (summary["action"] == "control_block_into_A_decoded")
        & (summary["start_group"] == "fashion")
        & (summary["block_group"].isin(["fc2.weight", "classifier_head", "all_weight_tensors"]))
    ].copy()
    pivot = rescue.pivot(index="pair", columns="block_group", values="control_block_rescue_fraction_test_mean")
    pivot = pivot.reindex(pair_labels)
    pivot.plot(kind="bar", ax=axes[1])
    axes[1].axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_title("control block into A decoded: fashion rescue")
    axes[1].set_ylabel("fraction of A-control gap removed")
    axes[1].legend(fontsize=8)
    fig.savefig(output_root / "cross_variant_rescue_summary.png", dpi=180)
    plt.close(fig)

    pair = "c3e-4" if "c3e-4" in pair_labels else pair_labels[-1]
    c3 = rows[
        (rows["pair"] == pair)
        & (rows["action"] == "control_block_into_A_decoded")
        & (rows["block_group"].isin(["fc2.weight", "classifier_head", "all_weight_tensors"]))
    ].copy()
    c3 = c3.sort_values(["source_weight_index", "block_group"])
    major = c3[c3["A_minus_control_test_loss"] > 0.05].copy()
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for block, group in c3.groupby("block_group"):
        ax.plot(
            group["source_weight_index"].astype(str),
            group["control_block_rescue_fraction_test"].clip(lower=-1.5, upper=2.5),
            marker="o",
            label=block,
        )
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title(f"{pair}: per-start rescue fraction, clipped to [-1.5, 2.5]")
    ax.set_ylabel("fraction removed")
    ax.set_xlabel("source_weight_index")
    ax.legend(fontsize=8)
    fig.savefig(output_root / "cross_variant_c3e4_per_start_rescue.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    head = rows[
        (rows["pair"] == pair)
        & (rows["action"] == "control_block_into_A_decoded")
        & (rows["block_group"] == "classifier_head")
    ].copy()
    head["gap_after_control_head"] = head["candidate_test_loss"] - head["control_decoded_test_loss"]
    head = head.sort_values("A_minus_control_test_loss", ascending=False)
    labels = head["source_weight_index"].astype(str).tolist()
    x = range(len(head))
    width = 0.38
    ax.bar([i - width / 2 for i in x], head["A_minus_control_test_loss"], width=width, label="A decoded - control decoded")
    ax.bar(
        [i + width / 2 for i in x],
        head["gap_after_control_head"],
        width=width,
        label="A body + control head - control decoded",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"{pair}: absolute step0 gap before/after control-head rescue")
    ax.set_ylabel("test-loss gap vs control decoded")
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_root / "cross_variant_c3e4_absolute_head_rescue.png", dpi=180)
    plt.close(fig)

    if not major.empty:
        major_summary = (
            major.groupby("block_group", as_index=False)
            .agg(
                starts=("source_weight_index", "nunique"),
                gap_mean=("A_minus_control_test_loss", "mean"),
                rescue_fraction_mean=("control_block_rescue_fraction_test", "mean"),
                rescue_fraction_median=("control_block_rescue_fraction_test", "median"),
            )
            .sort_values("block_group")
        )
        major_summary.to_csv(output_root / "cross_variant_c3e4_major_positive_gap_summary.csv", index=False)


def _write_notes(rows: pd.DataFrame, summary: pd.DataFrame, output_root: Path) -> None:
    lines = [
        "# Cross-Variant Head Splice",
        "",
        "This diagnostic swaps decoded blocks between matched control and Variant A VAE reconstructions.",
        "It tests whether the A-vs-control decoded step0 gap is caused by A-specific decoded head blocks.",
        "",
        "## Key Tables",
        "",
        "- `cross_variant_head_splice_rows.csv`",
        "- `cross_variant_head_splice_summary.csv`",
        "",
        "## Main Numbers",
        "",
    ]
    view = summary[
        (summary["action"] == "control_block_into_A_decoded")
        & (summary["start_group"] == "fashion")
        & (summary["block_group"].isin(["fc2.weight", "classifier_head", "all_weight_tensors"]))
    ].copy()
    for _, row in view.sort_values(["pair", "block_group"]).iterrows():
        lines.append(
            "- "
            f"pair={row['pair']} block={row['block_group']} starts={int(row['starts'])} "
            f"A-control test gap mean={row['A_minus_control_test_loss_mean']:.6g} "
            f"rescue fraction mean={row['control_block_rescue_fraction_test_mean']:.6g} "
            f"median={row['control_block_rescue_fraction_test_median']:.6g}"
        )
    (output_root / "cross_variant_head_splice_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-variant block splice between matched control and Variant A decoders.")
    parser.add_argument("--pair", action="append", default=[], choices=sorted(RUN_PAIRS), help="Run only selected pair(s).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--source-index", action="append", type=int, default=[])
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_pairs = args.pair or list(RUN_PAIRS)
    source_indices = {int(v) for v in args.source_index}
    print(
        "[cross_head_splice] start "
        f"pairs={selected_pairs} samples={args.samples} source_indices={sorted(source_indices)} "
        f"device={args.device} output_root={output_root}",
        flush=True,
    )
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    for pair in selected_pairs:
        control_name, a_name = RUN_PAIRS[pair]
        all_rows.extend(
            _pair_rows(
                pair=pair,
                control_dir=ARTIFACT_ROOT / control_name,
                a_dir=ARTIFACT_ROOT / a_name,
                device=str(args.device),
                samples=int(args.samples),
                source_indices=source_indices,
            )
        )
    rows = pd.DataFrame(all_rows)
    summary = _summarize(rows)
    rows_path = output_root / "cross_variant_head_splice_rows.csv"
    summary_path = output_root / "cross_variant_head_splice_summary.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_plots(rows, summary, output_root)
    _write_notes(rows, summary, output_root)
    elapsed = time.perf_counter() - t0
    print(
        "[cross_head_splice] wrote "
        f"rows={rows_path} rows_count={len(rows)} summary={summary_path} "
        f"summary_rows={len(summary)} elapsed_sec={elapsed:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
