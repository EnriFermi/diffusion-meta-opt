#!/usr/bin/env python3
"""Benchmark the shared-core compute proposed for the cross-modal control.

This is deliberately a compute microbenchmark, not an end-to-end Flickr8k
training run.  It includes the common 256->d input projection, a student
Transformer, an EMA teacher, top-k teacher targets, masked Smooth-L1 loss,
backward, AdamW, and the EMA update.  It excludes real data decoding,
preprocessing, checkpoint I/O, and downstream/VAE evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelSpec:
    label: str
    layers: int
    width: int
    heads: int


MODEL_SPECS = {
    "micro": ModelSpec("micro", layers=6, width=384, heads=6),
    "base": ModelSpec("base", layers=12, width=768, heads=12),
}


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = width // heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_in = nn.Linear(width, 4 * width)
        self.ffn_out = nn.Linear(4 * width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(inputs)
        batch, length, width = normalized.shape
        qkv = self.qkv(normalized)
        qkv = qkv.view(batch, length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(*qkv.unbind(0))
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        hidden = inputs + self.attention_output(attended)
        ffn = self.ffn_in(self.ffn_norm(hidden))
        ffn = F.gelu(ffn, approximate="tanh")
        return hidden + self.ffn_out(ffn)


class TransformerCore(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(spec.width, spec.heads) for _ in range(spec.layers)]
        )
        self.output_norm = nn.LayerNorm(spec.width)

    def forward(self, inputs: torch.Tensor, average_top_k: int = 0) -> torch.Tensor:
        hidden = inputs
        normalized_layers: list[torch.Tensor] = []
        for block in self.blocks:
            hidden = block(hidden)
            if average_top_k:
                normalized_layers.append(self.output_norm(hidden))
        if average_top_k:
            return torch.stack(normalized_layers[-average_top_k:]).mean(dim=0)
        return self.output_norm(hidden)


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters(), strict=True
    ):
        teacher_parameter.lerp_(student_parameter, 1.0 - decay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["micro", "base", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--average-top-k", type=int, default=6)
    parser.add_argument("--smooth-l1-beta", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=30)
    parser.add_argument("--projected-training-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("crossmodal_h100_benchmark")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def benchmark_model(
    spec: ModelSpec, args: argparse.Namespace, logger: logging.Logger
) -> dict[str, object]:
    logger.info("stage=model_build model=%s spec=%s", spec.label, asdict(spec))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    student_frontend = nn.Linear(args.input_width, spec.width).to(device)
    student = TransformerCore(spec).to(device)
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [*student_frontend.parameters(), *student.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=True,
    )

    raw_inputs = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.input_width,
        device=device,
        dtype=torch.bfloat16,
    )
    mask = torch.rand(
        args.batch_size, args.sequence_length, device=device
    ) < args.mask_ratio
    mask_embedding = torch.zeros(spec.width, device=device, dtype=torch.bfloat16)
    core_parameters = sum(parameter.numel() for parameter in student.parameters())
    frontend_parameters = sum(
        parameter.numel() for parameter in student_frontend.parameters()
    )
    logger.info(
        "stage=synthetic_cache model=%s input_shape=%s mask_fraction=%.6f "
        "core_params=%d frontend_params=%d",
        spec.label,
        tuple(raw_inputs.shape),
        mask.float().mean().item(),
        core_parameters,
        frontend_parameters,
    )

    total_steps = args.warmup_steps + args.measure_steps
    start_time: float | None = None
    final_loss = float("nan")
    logger.info(
        "stage=benchmark model=%s warmup_steps=%d measure_steps=%d",
        spec.label,
        args.warmup_steps,
        args.measure_steps,
    )
    for step in range(total_steps):
        if step == args.warmup_steps:
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            logger.info("stage=measurement_start model=%s", spec.label)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embedded = student_frontend(raw_inputs)
            with torch.no_grad():
                targets = teacher(
                    embedded.detach(),
                    average_top_k=min(args.average_top_k, spec.layers),
                )
            masked_inputs = torch.where(mask[..., None], mask_embedding, embedded)
            predictions = student(masked_inputs)
            loss = F.smooth_l1_loss(
                predictions[mask],
                targets[mask],
                beta=args.smooth_l1_beta,
            )
        loss.backward()
        optimizer.step()
        update_ema(teacher, student, args.ema_decay)
        final_loss = loss.detach().float().item()

        measured_step = step - args.warmup_steps + 1
        if measured_step > 0 and (
            measured_step == 1
            or measured_step % args.log_every == 0
            or measured_step == args.measure_steps
        ):
            logger.info(
                "stage=benchmark_progress model=%s measured_step=%d/%d loss=%.7f",
                spec.label,
                measured_step,
                args.measure_steps,
                final_loss,
            )

    if start_time is None:
        raise RuntimeError("measurement did not start")
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    seconds_per_step = elapsed_seconds / args.measure_steps
    tokens_per_step = args.batch_size * args.sequence_length
    tokens_per_second = tokens_per_step / seconds_per_step
    projected_minutes = (
        seconds_per_step * args.projected_training_steps / 60.0
    )
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)

    result: dict[str, object] = {
        **asdict(spec),
        "core_parameters": core_parameters,
        "frontend_parameters": frontend_parameters,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "input_width": args.input_width,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_step": seconds_per_step,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gib": peak_gib,
        "projected_training_steps": args.projected_training_steps,
        "projected_training_minutes": projected_minutes,
        "final_loss": final_loss,
    }
    logger.info("stage=benchmark_complete model=%s result=%s", spec.label, result)

    del optimizer, teacher, student, student_frontend
    del raw_inputs, mask, mask_embedding, embedded, targets, predictions, loss
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.device != "cuda":
        raise ValueError("this benchmark currently requires --device cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)
    specs = list(MODEL_SPECS.values()) if args.model == "all" else [MODEL_SPECS[args.model]]
    resolved_config = {
        **vars(args),
        "output_dir": str(args.output_dir.resolve()),
        "models": [asdict(spec) for spec in specs],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory
        / (1024**3),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "command": [sys.executable, *sys.argv],
        "benchmark_scope": (
            "common frontend + student/EMA teacher + top-k targets + loss + "
            "backward + AdamW + EMA; excludes real data/preprocessing/checkpoint/VAE"
        ),
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        " ".join(str(part) for part in resolved_config["command"]) + "\n",
        encoding="utf-8",
    )
    logger.info("stage=startup resolved_config=%s", resolved_config)

    results = [benchmark_model(spec, args, logger) for spec in specs]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "stage=output_complete artifacts=%s summary=%s",
        sorted(str(path.resolve()) for path in args.output_dir.iterdir()),
        [
            {
                "model": row["label"],
                "seconds_per_step": row["seconds_per_step"],
                "peak_memory_gib": row["peak_memory_gib"],
                "projected_training_minutes": row["projected_training_minutes"],
            }
            for row in results
        ],
    )


if __name__ == "__main__":
    main()
