from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from training.weightclip_benchmark import (
    run_gptq_token_bottleneck_comparison as baseline,
)
from training.weightclip_benchmark.run_conditioned_tokenizer_b_small import (
    ConditionedTokenizerB,
    _atomic_json,
    _load_baseline_contract,
    _make_config,
    _prepare_normalized,
)


RUN_ROOT = Path(
    "/mnt/shared/weightclip_benchmark/"
    "conditioned_tokenizer_b_small_3k_v1_20260827T1950Z"
)


@torch.no_grad()
def main() -> None:
    run_config = json.loads((RUN_ROOT / "config.json").read_text(encoding="utf-8"))
    baseline_contract = _load_baseline_contract(Path(run_config["baseline_root"]))

    class Args:
        smoke = False
        steps = 3000
        eval_every = 250

    cfg = _make_config(Args(), baseline_contract)
    device = torch.device("cuda:0")
    baseline._seed_everything(cfg.seed)
    data = baseline._load_exact64_tiles(
        Path(run_config["selection"]),
        Path(run_config["resolved_config"]),
    )
    prepared = _prepare_normalized(data, cfg)
    baseline._seed_everything(cfg.seed)
    model = ConditionedTokenizerB(cfg).to(device)
    state = torch.load(RUN_ROOT / "final_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    original_gate = model.tokenizer_b_activation_gate.weight.detach().clone()
    totals = defaultdict(float)
    gate_values: list[torch.Tensor] = []

    for start in range(0, len(data["train_operator_indices"]), cfg.eval_batch_size):
        operators = data["train_operator_indices"][start : start + cfg.eval_batch_size]
        batch = len(operators)
        indices = torch.tensor(
            [operator * 9 + tile for operator in operators for tile in range(9)],
            dtype=torch.long,
        )
        content = prepared["normalized_tokens"].index_select(0, indices).to(device)
        log_scale = prepared["standardized_log_scale"].index_select(0, indices).to(device)
        tile_row = prepared["flat_tile_rows"].index_select(0, indices).to(device)
        context = prepared["flat_contexts"].index_select(0, indices).to(device)
        target = prepared["flat_weights"].index_select(0, indices).to(device)
        dist_var, dist_patch = model._encode_distribution_context_full(context)
        context_shuffled = context.view(batch, 9, 512, 128).roll(1, dims=0).reshape_as(context)
        dist_var_shuffled, _dist_patch_shuffled = model._encode_distribution_context_full(
            context_shuffled
        )

        # Disable autocast's weight-cast cache because this diagnostic mutates the
        # FP32 gate weight between forwards.  Otherwise the disabled forward can
        # silently reuse the matched BF16 cast.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            cache_enabled=False,
        ):
            _token, record, gate = model.tokenize(
                content, log_scale, tile_row, dist_var
            )
            z, _ = model.encode_b(
                content, log_scale, tile_row, dist_var, dist_patch
            )
            matched = model.decode(z, tile_row, dist_patch_by_patch=dist_patch)
            z_token_xshuffle, _ = model.encode_b(
                content,
                log_scale,
                tile_row,
                dist_var_shuffled,
                dist_patch,
            )
            token_xshuffle = model.decode(
                z_token_xshuffle,
                tile_row,
                dist_patch_by_patch=dist_patch,
            )
            model.tokenizer_b_activation_gate.weight.zero_()
            z_disabled, _ = model.encode_b(
                content, log_scale, tile_row, dist_var, dist_patch
            )
            disabled = model.decode(
                z_disabled,
                tile_row,
                dist_patch_by_patch=dist_patch,
            )
            model.tokenizer_b_activation_gate.weight.copy_(original_gate)

        matched_loss, _ = baseline._big_vae_structural_loss(
            matched.float(), target.float(), cfg
        )
        shuffled_loss, _ = baseline._big_vae_structural_loss(
            token_xshuffle.float(), target.float(), cfg
        )
        disabled_loss, _ = baseline._big_vae_structural_loss(
            disabled.float(), target.float(), cfg
        )
        base_contribution = F.linear(
            record[..., :16].float(),
            model.tokenizer_b_projection.weight[:, :16].float(),
        )
        interaction_contribution = F.linear(
            record[..., 16:32].float(),
            model.tokenizer_b_projection.weight[:, 16:32].float(),
        )
        scale_contribution = F.linear(
            record[..., 32:34].float(),
            model.tokenizer_b_projection.weight[:, 32:34].float(),
        )
        count = int(target.shape[0])
        totals["count"] += count
        totals["matched_loss"] += float(matched_loss.item()) * count
        totals["token_xshuffle_loss"] += float(shuffled_loss.item()) * count
        totals["interaction_disabled_loss"] += float(disabled_loss.item()) * count
        totals["matched_output_sq"] += float(matched.float().square().sum().item())
        totals["token_xshuffle_output_delta_sq"] += float(
            (token_xshuffle.float() - matched.float()).square().sum().item()
        )
        totals["interaction_disabled_output_delta_sq"] += float(
            (disabled.float() - matched.float()).square().sum().item()
        )
        totals["latent_sq"] += float(z.float().square().sum().item())
        totals["token_xshuffle_latent_delta_sq"] += float(
            (z_token_xshuffle.float() - z.float()).square().sum().item()
        )
        totals["interaction_disabled_latent_delta_sq"] += float(
            (z_disabled.float() - z.float()).square().sum().item()
        )
        totals["base_contribution_sq"] += float(base_contribution.square().sum().item())
        totals["interaction_contribution_sq"] += float(
            interaction_contribution.square().sum().item()
        )
        totals["scale_contribution_sq"] += float(scale_contribution.square().sum().item())
        gate_values.append(gate.float().cpu().reshape(-1))

    if not torch.equal(model.tokenizer_b_activation_gate.weight, original_gate):
        raise RuntimeError("interaction intervention did not restore gate weights")
    gates = torch.cat(gate_values)
    gate_abs = gates.abs()
    output_den = max(totals["matched_output_sq"], 1.0e-20)
    latent_den = max(totals["latent_sq"], 1.0e-20)
    base_den = max(totals["base_contribution_sq"], 1.0e-20)
    result = {
        "method": (
            "zero-update exact train split; tokenizer-only X shuffle keeps encoder-key "
            "and decoder-query dist_patch matched; interaction-disabled zeros only the "
            "shared 256->1 gate"
        ),
        "matched_train_structural_loss": totals["matched_loss"] / totals["count"],
        "tokenizer_only_x_shuffle_train_structural_loss": totals["token_xshuffle_loss"]
        / totals["count"],
        "interaction_disabled_train_structural_loss": totals["interaction_disabled_loss"]
        / totals["count"],
        "tokenizer_only_x_shuffle_output_relative_rms_delta": math.sqrt(
            totals["token_xshuffle_output_delta_sq"] / output_den
        ),
        "interaction_disabled_output_relative_rms_delta": math.sqrt(
            totals["interaction_disabled_output_delta_sq"] / output_den
        ),
        "tokenizer_only_x_shuffle_latent_relative_rms_delta": math.sqrt(
            totals["token_xshuffle_latent_delta_sq"] / latent_den
        ),
        "interaction_disabled_latent_relative_rms_delta": math.sqrt(
            totals["interaction_disabled_latent_delta_sq"] / latent_den
        ),
        "projected_interaction_over_base_rms": math.sqrt(
            totals["interaction_contribution_sq"] / base_den
        ),
        "projected_scale_over_base_rms": math.sqrt(
            totals["scale_contribution_sq"] / base_den
        ),
        "gate_mean": float(gates.mean().item()),
        "gate_rms": float(gates.square().mean().sqrt().item()),
        "gate_abs_p50": float(torch.quantile(gate_abs, 0.50).item()),
        "gate_abs_p95": float(torch.quantile(gate_abs, 0.95).item()),
        "gate_abs_p99": float(torch.quantile(gate_abs, 0.99).item()),
        "gate_abs_max": float(gate_abs.max().item()),
        "gate_near_saturated_fraction": float((gate_abs > 0.99).float().mean().item()),
    }
    output = RUN_ROOT / "analysis" / "tokenizer_b_usage.json"
    output.parent.mkdir(exist_ok=True)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
