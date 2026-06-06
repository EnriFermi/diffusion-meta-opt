from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

from big_vae.models import build_weight_quantile_vae
from training.big_vae_latent_diffusion import build_big_vae_model_cfg


def _strip_state_prefix(name: str) -> str:
    out = str(name)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
    return out


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping payload, got {type(payload)!r}")
    return payload


def _extract_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("model_state", payload.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint has no mapping model_state/state_dict")
    return state


def _group_key(name: str, depth: int) -> str:
    parts = str(name).split(".")
    return ".".join(parts[: max(1, int(depth))])


def _fmt(value: int) -> str:
    return f"{int(value):,}"


def _print_table(rows: list[tuple[str, int, int]], *, total: int) -> None:
    if not rows:
        return
    name_width = max(6, min(72, max(len(name) for name, _, _ in rows)))
    print(f"{'module':{name_width}s} {'params':>16s} {'trainable':>16s} {'share':>9s}")
    print(f"{'-' * name_width} {'-' * 16} {'-' * 16} {'-' * 9}")
    for name, params, trainable in rows:
        share = 100.0 * float(params) / float(total) if total else 0.0
        print(f"{name:{name_width}s} {_fmt(params):>16s} {_fmt(trainable):>16s} {share:8.2f}%")


def inspect_checkpoint(path: Path, *, depth: int, json_output: bool) -> None:
    payload = _load_payload(path)
    state = {_strip_state_prefix(str(key)): value for key, value in _extract_state(payload).items()}
    raw_cfg = payload.get("config")
    if not isinstance(raw_cfg, Mapping):
        raise KeyError("BigVAE checkpoint has no mapping config; cannot separate parameters from buffers")

    model = build_weight_quantile_vae(build_big_vae_model_cfg(raw_cfg))
    model_params = dict(model.named_parameters())
    state_params = {
        name: value
        for name, value in state.items()
        if name in model_params and isinstance(value, torch.Tensor)
    }

    missing = sorted(set(model_params) - set(state_params))
    unexpected_param_like = sorted(
        name for name, value in state.items() if isinstance(value, torch.Tensor) and name not in model_params
    )

    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"params": 0, "trainable": 0})
    total = 0
    trainable = 0
    for name, param in model_params.items():
        count = int(param.numel())
        group = _group_key(name, depth)
        total += count
        by_group[group]["params"] += count
        if param.requires_grad:
            trainable += count
            by_group[group]["trainable"] += count

    rows = sorted(
        ((name, values["params"], values["trainable"]) for name, values in by_group.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    payload_out = {
        "checkpoint": str(path),
        "depth": int(depth),
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "state_param_tensors": int(len(state_params)),
        "model_param_tensors": int(len(model_params)),
        "missing_param_tensors": missing,
        "unexpected_state_tensors": unexpected_param_like,
        "groups": [
            {
                "module": name,
                "params": int(params),
                "trainable": int(group_trainable),
                "share": float(params) / float(total) if total else 0.0,
            }
            for name, params, group_trainable in rows
        ],
    }
    if json_output:
        print(json.dumps(payload_out, indent=2, sort_keys=True))
        return

    print(f"checkpoint: {path}")
    print(f"total params:     {_fmt(total)}")
    print(f"trainable params: {_fmt(trainable)}")
    print(f"frozen params:    {_fmt(total - trainable)}")
    print(f"state tensors matched to model params: {len(state_params):,}/{len(model_params):,}")
    if missing:
        print(f"missing param tensors in checkpoint: {len(missing):,}")
    if unexpected_param_like:
        print(f"extra tensor entries in state_dict, likely buffers or old keys: {len(unexpected_param_like):,}")
    print()
    print(f"by module, depth={int(depth)}:")
    _print_table(rows, total=total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count BigVAE checkpoint parameters by module.")
    parser.add_argument("checkpoint", type=Path, help="Path to BigVAE checkpoint .pt/.pth")
    parser.add_argument("--depth", type=int, default=1, help="How many name components to group by")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint.expanduser(), depth=args.depth, json_output=bool(args.json))


if __name__ == "__main__":
    main()
