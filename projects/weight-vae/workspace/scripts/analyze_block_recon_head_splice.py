from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    config_hash,
)
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
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug/block_recon_analysis/lam0p03/head_splice"
)

CONTROL_RUN = "sage_cnn_vae_smoothing_celo_meta_control_fc2recon_lam0p03_clean_harness_v1_seed0"
A_RUN = "sage_cnn_vae_smoothing_celo_meta_li_a_hvp_local_cap1_fc2recon_lam0p03_clean_harness_v1_seed0"
PAIR_NAME = "block_fc2recon_lam0p03"
KEY_COLS = ["start_index", "source_weight_index", "task_name", "tau"]
BLOCK_GROUPS: dict[str, tuple[str, ...]] = {
    "fc2.weight": ("fc2.weight",),
    "fc2.bias": ("fc2.bias",),
    "classifier_head": ("fc2.weight", "fc2.bias"),
}


def _log(message: str) -> None:
    print(f"[block_head_splice] {message}", flush=True)


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
    _log(
        "resolved_config "
        f"run={output_dir.name} hash={config_hash(cfg)} device={cfg.device} dtype={cfg.dtype} "
        f"seed={cfg.seed} cache_first={cfg.cache_first}"
    )
    for filename in ["weight_pool.pt", "vae_checkpoint.pt", "downstream_results.csv", "downstream_curves.csv"]:
        _log(f"cache_hit run={output_dir.name} file={output_dir / filename}")
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
        "curves": pd.read_csv(output_dir / "downstream_curves.csv"),
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


def _assert_common_start_bank(control_starts: pd.DataFrame, a_starts: pd.DataFrame) -> None:
    left = control_starts[KEY_COLS].reset_index(drop=True)
    right = a_starts[KEY_COLS].reset_index(drop=True)
    if not left.equals(right):
        merged = pd.concat({"control": left, "A": right}, axis=1)
        raise ValueError(f"start bank mismatch\n{merged}")


def _logged_step0(run: dict[str, Any]) -> pd.DataFrame:
    rows = run["curves"][
        (run["curves"]["split"].astype(str) == "eval")
        & (run["curves"]["method"].astype(str) == "decoder_latent")
        & (pd.to_numeric(run["curves"]["step"], errors="coerce") == 0)
    ].copy()
    return rows[KEY_COLS + ["train_loss", "train_acc", "test_loss", "test_acc"]].copy()


def _start_group(task_name: str) -> str:
    return "fashion" if str(task_name) == "fashion_mnist" else "mnist"


def _evaluate_splices(
    *,
    control: dict[str, Any],
    a_run: dict[str, Any],
    starts: pd.DataFrame,
    pair_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slices = _spec_slices(control["spec"])
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    control_logged = _logged_step0(control)
    a_logged = _logged_step0(a_run)
    logged = control_logged.merge(a_logged, on=KEY_COLS, suffixes=("_control_logged", "_A_logged"))

    for pos, start_row in starts.iterrows():
        elapsed_label = f"{pos + 1}/{len(starts)}"
        source_weight_index = int(start_row["source_weight_index"])
        record = control["records"].iloc[source_weight_index].to_dict()
        task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
        tau = float(record.get("tau", start_row.get("tau", 1.0)))
        task_set = _task_tensor_set(control["task_tensors"], task_name)
        w0 = control["weights"][source_weight_index].detach()
        _log(f"stage=decode_and_eval start={elapsed_label} source={source_weight_index} task={task_name} tau={tau:.6g}")
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
        match = logged[
            (logged["start_index"].astype(int) == int(start_row["start_index"]))
            & (logged["source_weight_index"].astype(int) == source_weight_index)
        ]
        if len(match) == 1:
            m = match.iloc[0]
            validation_rows.append(
                {
                    "pair": pair_name,
                    "source_weight_index": source_weight_index,
                    "start_index": int(start_row["start_index"]),
                    "task_name": task_name,
                    "control_train_loss_absdiff": abs(c_train_loss - float(m["train_loss_control_logged"])),
                    "control_test_loss_absdiff": abs(c_test_loss - float(m["test_loss_control_logged"])),
                    "A_train_loss_absdiff": abs(a_train_loss - float(m["train_loss_A_logged"])),
                    "A_test_loss_absdiff": abs(a_test_loss - float(m["test_loss_A_logged"])),
                }
            )

        for block_group, keys in BLOCK_GROUPS.items():
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
                        "pair": pair_name,
                        "source_weight_index": source_weight_index,
                        "start_index": int(start_row["start_index"]),
                        "task_name": task_name,
                        "start_group": _start_group(task_name),
                        "tau": tau,
                        "block_group": block_group,
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
                        "gap_after_candidate_vs_control_test_loss": test_loss - c_test_loss,
                    }
                )
        _log(
            "progress "
            f"start={elapsed_label} source={source_weight_index} A_minus_control_test_loss={gap_test:.6g}"
        )
    return rows, validation_rows


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["pair", "start_group", "block_group", "action"]
    by_group = (
        rows.groupby(group_cols, as_index=False)
        .agg(
            starts=("source_weight_index", "nunique"),
            A_minus_control_test_loss_mean=("A_minus_control_test_loss", "mean"),
            A_minus_control_test_loss_median=("A_minus_control_test_loss", "median"),
            candidate_test_loss_mean=("candidate_test_loss", "mean"),
            gap_after_candidate_vs_control_test_loss_mean=("gap_after_candidate_vs_control_test_loss", "mean"),
            control_block_rescue_fraction_test_mean=("control_block_rescue_fraction_test", "mean"),
            control_block_rescue_fraction_test_median=("control_block_rescue_fraction_test", "median"),
            A_block_injection_fraction_test_mean=("A_block_injection_fraction_test", "mean"),
            A_block_injection_fraction_test_median=("A_block_injection_fraction_test", "median"),
        )
        .sort_values(group_cols)
    )
    overall = (
        rows.assign(start_group="all")
        .groupby(group_cols, as_index=False)
        .agg(
            starts=("source_weight_index", "nunique"),
            A_minus_control_test_loss_mean=("A_minus_control_test_loss", "mean"),
            A_minus_control_test_loss_median=("A_minus_control_test_loss", "median"),
            candidate_test_loss_mean=("candidate_test_loss", "mean"),
            gap_after_candidate_vs_control_test_loss_mean=("gap_after_candidate_vs_control_test_loss", "mean"),
            control_block_rescue_fraction_test_mean=("control_block_rescue_fraction_test", "mean"),
            control_block_rescue_fraction_test_median=("control_block_rescue_fraction_test", "median"),
            A_block_injection_fraction_test_mean=("A_block_injection_fraction_test", "mean"),
            A_block_injection_fraction_test_median=("A_block_injection_fraction_test", "median"),
        )
        .sort_values(group_cols)
    )
    return pd.concat([overall, by_group], ignore_index=True)


def _write_plots(rows: pd.DataFrame, summary: pd.DataFrame, output_root: Path, *, pair_name: str, prefix: str) -> None:
    control_into_a = rows[rows["action"] == "control_block_into_A_decoded"].copy()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for block_group, group in control_into_a.groupby("block_group"):
        group = group.sort_values("source_weight_index")
        ax.plot(
            group["source_weight_index"].astype(str),
            group["control_block_rescue_fraction_test"].clip(lower=-2.0, upper=2.5),
            marker="o",
            label=block_group,
        )
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title(f"{pair_name}: control block into A decoded")
    ax.set_ylabel("fraction of A-control step0 test-loss gap removed")
    ax.set_xlabel("source_weight_index")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_root / f"{prefix}_control_into_A_rescue.png", dpi=180)
    plt.close(fig)

    head = control_into_a[control_into_a["block_group"] == "classifier_head"].copy()
    head = head.sort_values("A_minus_control_test_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    labels = head["source_weight_index"].astype(str).tolist()
    x = range(len(head))
    width = 0.38
    ax.bar([i - width / 2 for i in x], head["A_minus_control_test_loss"], width=width, label="A - control")
    ax.bar(
        [i + width / 2 for i in x],
        head["gap_after_candidate_vs_control_test_loss"],
        width=width,
        label="A body + control head - control",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"{pair_name}: absolute classifier-head rescue")
    ax.set_ylabel("step0 test-loss gap vs control decoded")
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_root / f"{prefix}_classifier_head_absolute_gap.png", dpi=180)
    plt.close(fig)

    view = summary[
        (summary["start_group"] == "all")
        & (summary["action"] == "control_block_into_A_decoded")
        & (summary["block_group"].isin(BLOCK_GROUPS))
    ].copy()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(view["block_group"], view["control_block_rescue_fraction_test_mean"], color=["#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_title("mean step0 test-loss gap rescue")
    ax.set_ylabel("fraction removed")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_root / f"{prefix}_rescue_summary.png", dpi=180)
    plt.close(fig)


def _write_notes(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    output_root: Path,
    *,
    pair_name: str,
    prefix: str,
) -> None:
    control_into_a = rows[rows["action"] == "control_block_into_A_decoded"].copy()
    starts = (
        rows.drop_duplicates("source_weight_index")[
            ["source_weight_index", "task_name", "A_minus_control_test_loss"]
        ]
        .sort_values("A_minus_control_test_loss", ascending=False)
        .reset_index(drop=True)
    )
    positive_sources = set(
        starts.loc[starts["A_minus_control_test_loss"] > 1e-4, "source_weight_index"].astype(int).tolist()
    )
    near_zero_sources = set(
        starts.loc[starts["A_minus_control_test_loss"].abs() <= 1e-4, "source_weight_index"].astype(int).tolist()
    )
    negative_sources = set(
        starts.loc[starts["A_minus_control_test_loss"] < -1e-4, "source_weight_index"].astype(int).tolist()
    )

    def _positive_block(block_group: str) -> dict[str, float]:
        view = control_into_a[
            (control_into_a["block_group"] == block_group)
            & (control_into_a["source_weight_index"].astype(int).isin(positive_sources))
        ]
        if view.empty:
            return {
                "starts": 0.0,
                "gap_mean": float("nan"),
                "gap_median": float("nan"),
                "after_mean": float("nan"),
                "after_median": float("nan"),
                "rescue_mean": float("nan"),
                "rescue_median": float("nan"),
            }
        return {
            "starts": float(view["source_weight_index"].nunique()),
            "gap_mean": float(view["A_minus_control_test_loss"].mean()),
            "gap_median": float(view["A_minus_control_test_loss"].median()),
            "after_mean": float(view["gap_after_candidate_vs_control_test_loss"].mean()),
            "after_median": float(view["gap_after_candidate_vs_control_test_loss"].median()),
            "rescue_mean": float(view["control_block_rescue_fraction_test"].mean()),
            "rescue_median": float(view["control_block_rescue_fraction_test"].median()),
        }

    control_head = summary[
        (summary["start_group"] == "all")
        & (summary["block_group"] == "classifier_head")
        & (summary["action"] == "control_block_into_A_decoded")
    ].iloc[0]
    bias = summary[
        (summary["start_group"] == "all")
        & (summary["block_group"] == "fc2.bias")
        & (summary["action"] == "control_block_into_A_decoded")
    ].iloc[0]
    weight = summary[
        (summary["start_group"] == "all")
        & (summary["block_group"] == "fc2.weight")
        & (summary["action"] == "control_block_into_A_decoded")
    ].iloc[0]
    max_validation_diff = float(validation.drop(columns=["pair", "source_weight_index", "start_index", "task_name"]).max().max())
    pos_head = _positive_block("classifier_head")
    pos_bias = _positive_block("fc2.bias")
    pos_weight = _positive_block("fc2.weight")
    conclusion = (
        "The all-start average is not a clean causal statistic because selected starts have mixed gap signs. "
        "Use the positive-gap subset for decoded-damage localization."
    )
    if pos_head["starts"] > 0 and abs(pos_head["after_median"]) <= 1e-3:
        conclusion = (
            "For starts where Variant A is worse at step0, the decoded-start damage is substantially localized to "
            "classifier-head / fc2.weight: splicing the control block into A decoded weights removes the positive gap."
        )
    lines = [
        f"# {pair_name} Head Splice",
        "",
        "Diagnostic: splice `fc2.weight`, `fc2.bias`, and `classifier_head` between matched control and Variant A decoded starts.",
        f"Pair: `{pair_name}`.",
        "",
        "## Validity",
        "",
        f"- Common start bank: {rows['source_weight_index'].nunique()} eval starts.",
        f"- Max absolute recompute-vs-logged step0 loss diff: {max_validation_diff:.6g}.",
        "",
        "## Overall Control-Into-A Rescue",
        "",
        "These all-start fractions are descriptive only when A-control gaps have mixed signs.",
        "",
        (
            f"- `fc2.weight`: gap mean={weight['A_minus_control_test_loss_mean']:.6g}, "
            f"rescue mean={weight['control_block_rescue_fraction_test_mean']:.6g}, "
            f"median={weight['control_block_rescue_fraction_test_median']:.6g}."
        ),
        (
            f"- `fc2.bias`: gap mean={bias['A_minus_control_test_loss_mean']:.6g}, "
            f"rescue mean={bias['control_block_rescue_fraction_test_mean']:.6g}, "
            f"median={bias['control_block_rescue_fraction_test_median']:.6g}."
        ),
        (
            f"- `classifier_head`: gap mean={control_head['A_minus_control_test_loss_mean']:.6g}, "
            f"rescue mean={control_head['control_block_rescue_fraction_test_mean']:.6g}, "
            f"median={control_head['control_block_rescue_fraction_test_median']:.6g}."
        ),
        "",
        "## Positive Step0-Damage Subset",
        "",
        (
            f"- Gap signs: positive>{1e-4:g}: {len(positive_sources)}, "
            f"near-zero: {len(near_zero_sources)}, negative<-{1e-4:g}: {len(negative_sources)}."
        ),
        (
            f"- `fc2.weight`: positive-gap starts={pos_weight['starts']:.0f}, "
            f"gap mean={pos_weight['gap_mean']:.6g}, after-splice gap mean={pos_weight['after_mean']:.6g}, "
            f"after median={pos_weight['after_median']:.6g}, rescue median={pos_weight['rescue_median']:.6g}."
        ),
        (
            f"- `fc2.bias`: positive-gap starts={pos_bias['starts']:.0f}, "
            f"gap mean={pos_bias['gap_mean']:.6g}, after-splice gap mean={pos_bias['after_mean']:.6g}, "
            f"after median={pos_bias['after_median']:.6g}, rescue median={pos_bias['rescue_median']:.6g}."
        ),
        (
            f"- `classifier_head`: positive-gap starts={pos_head['starts']:.0f}, "
            f"gap mean={pos_head['gap_mean']:.6g}, after-splice gap mean={pos_head['after_mean']:.6g}, "
            f"after median={pos_head['after_median']:.6g}, rescue median={pos_head['rescue_median']:.6g}."
        ),
        "",
        "## Narrow Conclusion",
        "",
        conclusion,
    ]
    (output_root / f"{prefix}_head_splice_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-recon cross-variant head/start splice diagnostic.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--source-index", action="append", type=int, default=[])
    parser.add_argument("--control-run", default=CONTROL_RUN)
    parser.add_argument("--a-run", default=A_RUN)
    parser.add_argument("--pair-name", default=PAIR_NAME)
    parser.add_argument("--prefix", default="block_lam0p03")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_indices = {int(v) for v in args.source_index}
    _log(
        "startup "
        f"pair={args.pair_name} device={args.device} dtype=float32 samples={args.samples} "
        f"source_indices={sorted(source_indices)} output_root={output_root}"
    )
    _log(f"stage=load_runs control={args.control_run} A={args.a_run}")
    t0 = time.perf_counter()
    control = _load_run(ARTIFACT_ROOT / args.control_run, device=str(args.device))
    a_run = _load_run(ARTIFACT_ROOT / args.a_run, device=str(args.device))
    if list(control["spec"].keys) != list(a_run["spec"].keys):
        raise ValueError("spec key mismatch")
    if control["weights"].shape != a_run["weights"].shape:
        raise ValueError("weight shape mismatch")
    missing = [key for keys in BLOCK_GROUPS.values() for key in keys if key not in set(control["spec"].keys)]
    if missing:
        raise KeyError(f"missing block keys: {sorted(set(missing))}")
    max_weight_diff = float((control["weights"] - a_run["weights"]).abs().max().detach().cpu().item())
    _log(f"validity weight_pool_max_abs_diff={max_weight_diff:.6g}")
    if max_weight_diff > 1e-6:
        raise ValueError(f"weight pool mismatch: max abs diff={max_weight_diff}")
    control_starts = _eval_starts(control, samples=int(args.samples), source_indices=source_indices)
    a_starts = _eval_starts(a_run, samples=int(args.samples), source_indices=source_indices)
    _assert_common_start_bank(control_starts, a_starts)
    if source_indices:
        observed = set(control_starts["source_weight_index"].astype(int).tolist())
        missing_sources = sorted(source_indices - observed)
        extra_sources = sorted(observed - source_indices)
        if missing_sources or extra_sources:
            raise ValueError(
                "selected source-index coverage mismatch: "
                f"missing={missing_sources} extra={extra_sources} observed={sorted(observed)}"
            )
    _log(
        "stage=eval_starts "
        f"count={len(control_starts)} source_indices={control_starts['source_weight_index'].astype(int).tolist()}"
    )
    rows_list, validation_list = _evaluate_splices(
        control=control,
        a_run=a_run,
        starts=control_starts,
        pair_name=str(args.pair_name),
    )
    rows = pd.DataFrame(rows_list)
    validation = pd.DataFrame(validation_list)
    summary = _summarize(rows)

    rows_path = output_root / f"{args.prefix}_head_splice_rows.csv"
    summary_path = output_root / f"{args.prefix}_head_splice_summary.csv"
    validation_path = output_root / f"{args.prefix}_head_splice_validation.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    validation.to_csv(validation_path, index=False)
    _write_plots(rows, summary, output_root, pair_name=str(args.pair_name), prefix=str(args.prefix))
    _write_notes(
        rows,
        summary,
        validation,
        output_root,
        pair_name=str(args.pair_name),
        prefix=str(args.prefix),
    )
    elapsed = time.perf_counter() - t0
    _log(
        "done "
        f"rows={rows_path} rows_count={len(rows)} summary={summary_path} "
        f"validation={validation_path} elapsed_sec={elapsed:.2f}"
    )
    _log("summary_control_into_A_all")
    view = summary[(summary["start_group"] == "all") & (summary["action"] == "control_block_into_A_decoded")]
    print(view.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
