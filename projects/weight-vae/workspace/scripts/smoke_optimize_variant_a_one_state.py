from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import torch_dtype
from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.core import encode_weights
from scripts.audit_variant_a_estimator_stability import (
    DEFAULT_RUN_DIR,
    _atomic_pair_loss,
    _load_run,
    _probe_cfg,
    sha256_file,
    stable_uint63,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "one_state_a_optimization_smoke_h2048"
)
STATE_BANK = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_fixed_state_probe_variance_h2048/state_bank.csv"
)
ACTIVE_PARAMETERS = (
    ROOT
    / "docs/reparam_preconditioning_experiments/variant_A_causal_debug"
    / "a_estimator_stability_unfinetuned_h2048/active_parameters.csv"
)
PROTOCOL_ID = "one_state_a_optimization_smoke_h2048_v1"
EXPECTED_CHECKPOINT = "7bbf3bce6c18da02fdc3a72e9dcbe9d14cfda900a8cf9e338ec40f4ab1706397"


def generators(*parts: object) -> tuple[torch.Generator, torch.Generator]:
    seeds = [stable_uint63(PROTOCOL_ID, *parts, branch) for branch in (0, 1)]
    return tuple(torch.Generator(device="cpu").manual_seed(seed) for seed in seeds)  # type: ignore[return-value]


def evaluate(
    *,
    run: object,
    cfg: object,
    z: torch.Tensor,
    record: dict[str, object],
    draws: int,
) -> tuple[float, float, float, list[float]]:
    values: list[float] = []
    run.vae.zero_grad(set_to_none=True)
    for draw in range(draws):
        for pair in range(int(cfg.vae_precond_pairs)):
            g1, g2 = generators("eval", draw, pair)
            loss, _stats = _atomic_pair_loss(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                step=10,
                pair_index=pair,
                probe_generator_1=g1,
                probe_generator_2=g2,
            )
            values.append(float(loss.detach().cpu()))
            del loss
    run.vae.zero_grad(set_to_none=True)
    tensor = torch.tensor(values, dtype=torch.float64)
    return (
        float(tensor.mean()),
        float(tensor.std(unbiased=True)),
        float(tensor.std(unbiased=True) / math.sqrt(len(values))),
        values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-position", type=int, default=2)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--eval-draws", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(DEFAULT_RUN_DIR / "vae_checkpoint.pt")
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint_hash}")

    print(
        f"[one-state-A] load device={device} dtype=float32 state_position={args.state_position} "
        f"steps={args.steps} lr={args.lr:g} pairs={args.pairs} eval_draws={args.eval_draws} "
        f"output={args.output_dir}",
        flush=True,
    )
    run = _load_run(DEFAULT_RUN_DIR, device=device)
    cfg = _probe_cfg(run.cfg, sample_count=1, pair_count=args.pairs, batch_size=16384)
    bank = pd.read_csv(STATE_BANK)
    state_row = bank.loc[bank["state_position"].eq(args.state_position)]
    if len(state_row) != 1:
        raise ValueError(f"state_position {args.state_position} is not unique")
    state_meta = state_row.iloc[0].to_dict()
    source_index = int(state_meta["source_weight_index"])
    weight = run.weights[[source_index]].to(device=device, dtype=torch_dtype(run.cfg))
    with torch.no_grad():
        z = encode_weights(run.vae, run.normalizer, weight).detach()[0]
    record = run.records.iloc[source_index].to_dict()
    record["source_weight_index"] = source_index

    active_names = set(pd.read_csv(ACTIVE_PARAMETERS)["parameter"].astype(str))
    named = dict(run.vae.named_parameters())
    missing = sorted(active_names - set(named))
    if missing:
        raise RuntimeError(f"missing active parameters: {missing}")
    for name, parameter in named.items():
        parameter.requires_grad_(name in active_names)
    active = [named[name] for name in sorted(active_names)]
    optimizer = torch.optim.Adam(active, lr=args.lr)
    eval_steps = sorted(step for step in {0, 1, 2, 5, 10, 20, 40, 60, 100, 200, int(args.steps)} if step <= args.steps)
    train_rows: list[dict[str, float | int]] = []
    eval_rows: list[dict[str, float | int]] = []
    eval_atomic_rows: list[dict[str, float | int]] = []
    torch.cuda.reset_peak_memory_stats(device)

    def run_eval(step: int) -> None:
        t0 = time.perf_counter()
        mean, std, se, values = evaluate(run=run, cfg=cfg, z=z, record=record, draws=args.eval_draws)
        for index, value in enumerate(values):
            eval_atomic_rows.append(
                {
                    "step": step,
                    "draw": index // args.pairs,
                    "pair": index % args.pairs,
                    "eval_a": value,
                }
            )
        row = {
            "step": step,
            "eval_a_mean": mean,
            "eval_a_std_atomic": std,
            "eval_a_se": se,
            "eval_atomic_count": args.eval_draws * args.pairs,
            "elapsed_sec": time.perf_counter() - started,
        }
        eval_rows.append(row)
        print(
            f"[one-state-A] eval step={step}/{args.steps} A={mean:.7g} se={se:.3g} "
            f"eval_sec={time.perf_counter() - t0:.1f} elapsed={row['elapsed_sec']:.1f}s",
            flush=True,
        )

    run_eval(0)
    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for pair in range(args.pairs):
            g1, g2 = generators("train", step, pair)
            loss, _stats = _atomic_pair_loss(
                cfg=cfg,
                run=run,
                z=z,
                record=record,
                step=10,
                pair_index=pair,
                probe_generator_1=g1,
                probe_generator_2=g2,
            )
            losses.append(float(loss.detach().cpu()))
            (loss / float(args.pairs)).backward()
            del loss
        grad_norm = float(torch.nn.utils.clip_grad_norm_(active, max_norm=1.0).detach().cpu())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step}: {grad_norm}")
        optimizer.step()
        row = {
            "step": step,
            "train_a": sum(losses) / len(losses),
            "grad_norm_preclip": grad_norm,
            "step_sec": time.perf_counter() - step_started,
            "elapsed_sec": time.perf_counter() - started,
        }
        train_rows.append(row)
        if step <= 2 or step % 5 == 0:
            print(
                f"[one-state-A] train step={step}/{args.steps} A={row['train_a']:.7g} "
                f"grad_norm={grad_norm:.4g} step_sec={row['step_sec']:.2f} elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )
        if step in eval_steps:
            run_eval(step)

    train = pd.DataFrame(train_rows)
    evaluation = pd.DataFrame(eval_rows)
    eval_atomic = pd.DataFrame(eval_atomic_rows)
    train.to_csv(args.output_dir / "train_curve.csv", index=False)
    evaluation.to_csv(args.output_dir / "fixed_probe_eval_curve.csv", index=False)
    eval_atomic.to_csv(args.output_dir / "fixed_probe_eval_atomic.csv", index=False)
    initial = float(evaluation.iloc[0]["eval_a_mean"])
    final = float(evaluation.iloc[-1]["eval_a_mean"])
    initial_atomic = eval_atomic.loc[eval_atomic["step"].eq(0), "eval_a"].reset_index(drop=True)
    final_atomic = eval_atomic.loc[eval_atomic["step"].eq(args.steps), "eval_a"].reset_index(drop=True)
    paired_delta = final_atomic - initial_atomic
    summary = {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "dtype": str(torch_dtype(run.cfg)),
        "state_position": int(args.state_position),
        "source_weight_index": source_index,
        "task_name": str(state_meta["task_name"]),
        "prior_measured_p4_mean_cosine": 0.369678 if args.state_position == 2 else None,
        "steps": args.steps,
        "lr": args.lr,
        "optimizer": "Adam",
        "adam_betas": [0.9, 0.999],
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "pairs_per_update": args.pairs,
        "ce_batch_size": 16384,
        "eval_draws": args.eval_draws,
        "eval_atomic_count": args.eval_draws * args.pairs,
        "initial_eval_a": initial,
        "final_eval_a": final,
        "absolute_change": final - initial,
        "relative_change": final / initial - 1.0,
        "paired_atomic_delta_mean": float(paired_delta.mean()),
        "paired_atomic_delta_se": float(paired_delta.std(ddof=1) / math.sqrt(len(paired_delta))),
        "paired_atomic_fraction_improved": float((paired_delta < 0.0).mean()),
        "elapsed_sec": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(
        {
            "active_model_state": {name: named[name].detach().cpu() for name in sorted(active_names)},
            "summary": summary,
        },
        args.output_dir / "final_active_checkpoint.pt",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(train["step"], train["train_a"], alpha=0.55, linewidth=1.0)
    axes[0].scatter(evaluation["step"], evaluation["eval_a_mean"], color="black", s=24, label="fixed probes")
    axes[0].set(xlabel="optimizer step", ylabel="A surrogate", title="Raw stochastic train and fixed-probe eval")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    ratio = evaluation["eval_a_mean"] / initial
    axes[1].plot(evaluation["step"], ratio, marker="o")
    axes[1].axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set(xlabel="optimizer step", ylabel="eval A / initial A", title="Independent fixed-probe convergence")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "convergence.png", dpi=180)
    plt.close(fig)
    print(f"[one-state-A] done {json.dumps(summary, sort_keys=True)}", flush=True)
    print(
        f"[one-state-A] artifacts={args.output_dir / 'train_curve.csv'},"
        f"{args.output_dir / 'fixed_probe_eval_curve.csv'},"
        f"{args.output_dir / 'fixed_probe_eval_atomic.csv'},"
        f"{args.output_dir / 'summary.json'},"
        f"{args.output_dir / 'convergence.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
