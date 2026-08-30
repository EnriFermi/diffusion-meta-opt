from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

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


WORST_FASHION_SOURCES = {942, 10946, 13160, 6026, 14431}


def _load_cfg(output_dir: Path, *, device: str) -> ExperimentConfig:
    payload = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    raw_cfg = payload.get("config", payload)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"{output_dir / 'config.json'} does not contain a config mapping")
    values = dict(raw_cfg)
    values["device"] = str(device)
    values["dtype"] = "float32"
    return ExperimentConfig(**values)


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


def _norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu().item())


def _safe_fraction(num: float, den: float) -> float:
    if abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


def _spec_slices(spec) -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for key, size in zip(spec.keys, spec.sizes, strict=True):
        result[str(key)] = slice(offset, offset + int(size))
        offset += int(size)
    return result


def _block_groups(spec) -> OrderedDict[str, tuple[str, ...]]:
    keys = tuple(str(k) for k in spec.keys)
    groups: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for key in keys:
        groups[key] = (key,)
    fc1 = tuple(k for k in keys if k.startswith("fc1."))
    fc2 = tuple(k for k in keys if k.startswith("fc2."))
    weights = tuple(k for k in keys if k.endswith(".weight"))
    biases = tuple(k for k in keys if k.endswith(".bias"))
    if fc1:
        groups["input_hidden_block"] = fc1
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


def _start_group(source_weight_index: int, task_name: str) -> str:
    if int(source_weight_index) in WORST_FASHION_SOURCES:
        return "worst_fashion"
    if str(task_name) == "fashion_mnist":
        return "fashion_eval"
    return "mnist_eval"


def _rows_for_start(
    *,
    cfg: ExperimentConfig,
    vae: torch.nn.Module,
    normalizer: WeightNormalizer,
    weights_device: torch.Tensor,
    weight_records: pd.DataFrame,
    task_tensors: dict[str, Any],
    spec,
    start_row: pd.Series,
    label: str,
    run_name: str,
) -> list[dict[str, Any]]:
    source_weight_index = int(start_row["source_weight_index"])
    start_index = int(start_row["start_index"])
    record = weight_records.iloc[source_weight_index].to_dict()
    task_name = str(record.get("task_name", start_row.get("task_name", "tiny_cnn")))
    tau = float(record.get("tau", start_row.get("tau", 1.0)))
    task_set = _task_tensor_set(task_tensors, task_name)
    w0 = weights_device[source_weight_index].detach()
    with torch.no_grad():
        z0 = encode_weights(vae, normalizer, w0.reshape(1, -1)).squeeze(0).detach()
        decoded = decode_weights(vae, normalizer, z0.reshape(1, -1)).squeeze(0).detach()
        raw_train_loss, raw_train_acc = _loss_acc(w0, task_set=task_set, spec=spec, split="train", tau=tau)
        raw_test_loss, raw_test_acc = _loss_acc(w0, task_set=task_set, spec=spec, split="test", tau=tau)
        decoded_train_loss, decoded_train_acc = _loss_acc(decoded, task_set=task_set, spec=spec, split="train", tau=tau)
        decoded_test_loss, decoded_test_acc = _loss_acc(decoded, task_set=task_set, spec=spec, split="test", tau=tau)
    train_gap = decoded_train_loss - raw_train_loss
    test_gap = decoded_test_loss - raw_test_loss
    rows: list[dict[str, Any]] = []
    slices = _spec_slices(spec)
    for group_name, keys in _block_groups(spec).items():
        for action, candidate in (
            ("swap_decoded_block_into_raw", _splice(w0, decoded, slices, keys)),
            ("rescue_raw_block_into_decoded", _splice(decoded, w0, slices, keys)),
        ):
            with torch.no_grad():
                train_loss, train_acc = _loss_acc(candidate, task_set=task_set, spec=spec, split="train", tau=tau)
                test_loss, test_acc = _loss_acc(candidate, task_set=task_set, spec=spec, split="test", tau=tau)
            candidate_gap_train = train_loss - raw_train_loss
            candidate_gap_test = test_loss - raw_test_loss
            removed_train = decoded_train_loss - train_loss
            removed_test = decoded_test_loss - test_loss
            rows.append(
                {
                    "run_name": run_name,
                    "variant_label": label,
                    "source_weight_index": source_weight_index,
                    "start_index": start_index,
                    "start_group": _start_group(source_weight_index, task_name),
                    "task_name": task_name,
                    "tau": tau,
                    "w0_norm": _norm(w0),
                    "reconstruction_rel_l2": _norm(decoded - w0) / max(_norm(w0), 1e-30),
                    "block_group": group_name,
                    "block_keys": ";".join(keys),
                    "action": action,
                    "raw_train_loss": raw_train_loss,
                    "decoded_train_loss": decoded_train_loss,
                    "decoded_minus_raw_train_loss": train_gap,
                    "candidate_train_loss": train_loss,
                    "candidate_minus_raw_train_loss": candidate_gap_train,
                    "candidate_train_acc": train_acc,
                    "swap_fraction_of_decoded_train_gap": _safe_fraction(candidate_gap_train, train_gap),
                    "rescue_removed_fraction_of_decoded_train_gap": _safe_fraction(removed_train, train_gap),
                    "raw_test_loss": raw_test_loss,
                    "decoded_test_loss": decoded_test_loss,
                    "decoded_minus_raw_test_loss": test_gap,
                    "candidate_test_loss": test_loss,
                    "candidate_minus_raw_test_loss": candidate_gap_test,
                    "candidate_test_acc": test_acc,
                    "swap_fraction_of_decoded_test_gap": _safe_fraction(candidate_gap_test, test_gap),
                    "rescue_removed_fraction_of_decoded_test_gap": _safe_fraction(removed_test, test_gap),
                    "raw_train_acc": raw_train_acc,
                    "decoded_train_acc": decoded_train_acc,
                    "raw_test_acc": raw_test_acc,
                    "decoded_test_acc": decoded_test_acc,
                }
            )
    return rows


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(["variant_label", "start_group", "block_group", "action"], as_index=False)
    return grouped.agg(
        starts=("source_weight_index", "nunique"),
        decoded_minus_raw_test_loss_median=("decoded_minus_raw_test_loss", "median"),
        candidate_minus_raw_test_loss_median=("candidate_minus_raw_test_loss", "median"),
        swap_fraction_of_decoded_test_gap_median=("swap_fraction_of_decoded_test_gap", "median"),
        rescue_removed_fraction_of_decoded_test_gap_median=("rescue_removed_fraction_of_decoded_test_gap", "median"),
        candidate_test_acc_median=("candidate_test_acc", "median"),
        decoded_test_acc_median=("decoded_test_acc", "median"),
        raw_test_acc_median=("raw_test_acc", "median"),
    )


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    print(
        "[block_splice] start "
        f"runs={len(args.run_dir)} device={args.device} samples={args.samples} output_root={output_root}",
        flush=True,
    )
    for run_dir_value in args.run_dir:
        run_dir_path = Path(run_dir_value).expanduser().resolve()
        run_name = run_dir_path.name
        label = str(args.label.get(run_name, run_name)) if isinstance(args.label, dict) else run_name
        cfg = _load_cfg(run_dir_path, device=str(args.device))
        device = torch.device(cfg.device)
        print(
            "[block_splice] run "
            f"name={run_name} label={label} dir={run_dir_path} config_hash={config_hash(cfg)} device={device}",
            flush=True,
        )
        weight_payload = load_torch_cache(run_dir_path / "weight_pool.pt")
        vae_payload = load_torch_cache(run_dir_path / "vae_checkpoint.pt")
        if weight_payload is None or vae_payload is None:
            raise FileNotFoundError(f"{run_dir_path} missing weight_pool.pt or vae_checkpoint.pt")
        weights = weight_payload["weights"]
        weight_records = pd.DataFrame(weight_payload["records"])
        spec = spec_from_payload(weight_payload["spec"])
        normalizer = WeightNormalizer.from_state_dict(vae_payload["normalizer"])
        vae = build_weight_vae(cfg, weight_dim=int(weights.shape[1])).to(device=device, dtype=torch.float32).eval()
        vae.load_state_dict(vae_payload["model_state"])
        weights_device = weights.to(device=device, dtype=torch.float32)
        task_tensors = move_task_tensors(load_celo_meta_task_tensors(cfg), device=device, dtype=torch.float32)
        results = pd.read_csv(run_dir_path / "downstream_results.csv")
        starts = results[(results["split"].astype(str) == "eval") & (results["method"].astype(str) == "decoder_latent")]
        starts = starts.sort_values("start_index").reset_index(drop=True)
        if args.source_index:
            allowed = {int(v) for v in args.source_index}
            starts = starts[starts["source_weight_index"].astype(int).isin(allowed)].reset_index(drop=True)
        elif int(args.samples) > 0:
            starts = starts.iloc[: int(args.samples)].copy()
        print(
            "[block_splice] starts "
            f"count={len(starts)} indices={starts['source_weight_index'].astype(int).tolist()} "
            f"groups={list(_block_groups(spec).keys())}",
            flush=True,
        )
        for pos, start_row in starts.iterrows():
            print(
                "[block_splice] probe "
                f"run={label} start={pos + 1}/{len(starts)} source={int(start_row['source_weight_index'])} "
                f"task={start_row['task_name']} tau={float(start_row['tau']):.6g}",
                flush=True,
            )
            all_rows.extend(
                _rows_for_start(
                    cfg=cfg,
                    vae=vae,
                    normalizer=normalizer,
                    weights_device=weights_device,
                    weight_records=weight_records,
                    task_tensors=task_tensors,
                    spec=spec,
                    start_row=start_row,
                    label=label,
                    run_name=run_name,
                )
            )
    frame = pd.DataFrame(all_rows)
    summary = _summarize(frame)
    rows_path = output_root / "decoder_block_splice_rows.csv"
    summary_path = output_root / "decoder_block_splice_summary.csv"
    frame.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    elapsed = time.perf_counter() - start_time
    print(
        "[block_splice] wrote "
        f"rows={rows_path} rows_count={len(frame)} summary={summary_path} summary_rows={len(summary)} "
        f"elapsed_sec={elapsed:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal block-splice interventions for decoded CELO weight artifacts.")
    parser.add_argument("--run-dir", action="append", required=True, help="Experiment output directory. Repeat for multiple runs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=16, help="Use first N eval starts when --source-index is not set. <=0 means all.")
    parser.add_argument("--source-index", action="append", type=int, default=[], help="Restrict to specific source_weight_index values.")
    parser.add_argument("--label-json", default="")
    args = parser.parse_args()
    args.label = json.loads(args.label_json) if args.label_json else {}
    run(args)


if __name__ == "__main__":
    main()
