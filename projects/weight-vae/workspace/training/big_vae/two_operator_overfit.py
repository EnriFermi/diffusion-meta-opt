from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from big_vae.datasets.operator_bank import (
    CommittedBundleSampler,
    OperatorBankBundleDataset,
    OperatorBankTrainingDataset,
)
from big_vae.models import WeightQuantileVAE


class TwoOperatorTrainingDataset(OperatorBankTrainingDataset):
    """Consumer-only fixed canonical view over two immutable bank operators."""

    def __init__(
        self,
        pair_manifest: str | Path,
        *,
        selected_operators: Sequence[Mapping[str, str]],
        seed: int,
        hot_shards: int,
        expected_pair_manifest_sha256: str,
    ) -> None:
        super().__init__(
            pair_manifest,
            seed=seed,
            repeat=True,
            permutation_views=False,
            canonical_probability=1.0,
            hot_shards=hot_shards,
            expected_pair_manifest_sha256=expected_pair_manifest_sha256,
            rank=0,
            world_size=1,
        )
        keys = tuple(
            (str(row.get("checkpoint_sha256", "")), str(row.get("layer_key", "")))
            for row in selected_operators
        )
        if len(keys) != 2 or len(set(keys)) != 2:
            raise ValueError("two-operator dataset requires two unique identities")
        missing = [key for key in keys if key not in self.operator_groups]
        if missing:
            raise ValueError(f"selected operator identities are absent from the immutable bank: {missing}")
        self.operator_groups = {key: self.operator_groups[key] for key in keys}
        self._keys = list(keys)
        self._records_per_cycle = sum(len(self.operator_groups[key]) for key in self._keys)
        self._cycle_cache.clear()
        self._group_cache.clear()
        self._locality_schedule_cache.clear()
        first, second = self._keys
        first_meta = self.operator_groups[first][0].metadata["operator"]
        second_meta = self.operator_groups[second][0].metadata["operator"]
        if (
            first[1] != second[1]
            or tuple(first_meta["matrix_shape"]) != tuple(second_meta["matrix_shape"])
            or len(self.operator_groups[first]) != len(self.operator_groups[second])
        ):
            raise ValueError("two-operator smoke requires the same layer, matrix shape, and tile count")

    @property
    def exact_operator_cycle(self) -> bool:
        return True

    def _cycle_plan(
        self,
        cycle: int,
    ) -> tuple[list[tuple[tuple[str, str], int]], dict[str, dict[str, Any] | None]]:
        cached = self._cycle_cache.get(int(cycle))
        if cached is not None:
            self._cycle_cache.move_to_end(int(cycle))
            return cached
        plan = [
            (key, local_index)
            for local_index in range(len(self.operator_groups[self._keys[0]]))
            for key in self._keys
        ]
        if len(plan) != self._records_per_cycle or len(set(plan)) != self._records_per_cycle:
            raise RuntimeError("fixed two-operator cycle must contain each selected tile exactly once")
        result = (plan, {checkpoint: None for checkpoint, _layer in self._keys})
        self._cycle_cache[int(cycle)] = result
        while len(self._cycle_cache) > 2:
            self._cycle_cache.popitem(last=False)
        return result

    def locality_plan(self, cycle: int) -> tuple[tuple[tuple[str, str], int], ...]:
        # Preserve A_i,B_i adjacency so each B18 diagnostic microbatch contains
        # complete matched swap pairs. Only two small bundles are active.
        plan, _views = self._cycle_plan(int(cycle))
        return tuple(plan)


@contextmanager
def two_operator_data_pipeline(
    pair_manifest: str | Path,
    *,
    selected_operators: Sequence[Mapping[str, str]],
    seed: int,
    hot_shards: int,
    expected_pair_manifest_sha256: str,
    max_active_strata: int,
    max_active_bundle_bytes: int,
) -> Iterator[tuple[OperatorBankBundleDataset, CommittedBundleSampler]]:
    source = TwoOperatorTrainingDataset(
        pair_manifest,
        selected_operators=selected_operators,
        seed=seed,
        hot_shards=hot_shards,
        expected_pair_manifest_sha256=expected_pair_manifest_sha256,
    )
    dataset = OperatorBankBundleDataset(
        source,
        max_active_strata=max_active_strata,
        max_active_bundle_bytes=max_active_bundle_bytes,
    )
    yield dataset, CommittedBundleSampler(dataset)


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    value = model.module if hasattr(model, "module") else model
    return value._orig_mod if hasattr(value, "_orig_mod") else value


def _stitch(
    tiles: torch.Tensor,
    assignments: Sequence[tuple[tuple[str, str], int]],
    source: OperatorBankTrainingDataset,
    key: tuple[str, str],
) -> torch.Tensor:
    locations = source.operator_groups[key]
    spec = locations[0].metadata["operator"]
    rows, cols = map(int, spec["matrix_shape"])
    matrix = tiles.new_zeros((rows, cols))
    seen: set[int] = set()
    for batch_index, (assigned_key, local_index) in enumerate(assignments):
        if assigned_key != key:
            continue
        if local_index in seen:
            raise RuntimeError(f"duplicate tile {local_index} for {key}")
        seen.add(local_index)
        tile_meta = locations[local_index].metadata["tile"]
        row_start = int(tile_meta["row_start"])
        col_start = int(tile_meta["col_start"])
        valid_rows = int(tile_meta["valid_rows"])
        valid_cols = int(tile_meta["valid_cols"])
        matrix[row_start : row_start + valid_rows, col_start : col_start + valid_cols] = tiles[
            batch_index, :valid_rows, :valid_cols
        ]
    if len(seen) != len(locations):
        raise RuntimeError(f"batch does not cover full operator {key}: {len(seen)}/{len(locations)} tiles")
    return matrix


def _loss_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    patch_size: int,
    gamma: float,
    lambda_dir: float,
    lambda_scale: float,
    huber_delta: float,
) -> dict[str, float]:
    total, details = WeightQuantileVAE.patch_structure_loss(
        target,
        prediction,
        patch_size=patch_size,
        gamma=gamma,
        lambda_dir=lambda_dir,
        lambda_scale=lambda_scale,
        lambda_rec=0.0,
        lambda_rel=0.0,
        huber_delta=huber_delta,
    )
    return {
        "total": float(total.detach().float().item()),
        "dir": float(details["L_dir"].detach().float().item()),
        "scale": float(details["L_scale"].detach().float().item()),
    }


def _nrmse(target: torch.Tensor, prediction: torch.Tensor) -> float:
    error_rms = (prediction.float() - target.float()).square().mean().sqrt()
    target_rms = target.float().square().mean().sqrt().clamp_min(1.0e-12)
    return float((error_rms / target_rms).item())


def _paired_relative_delta(first: torch.Tensor, second: torch.Tensor) -> float:
    first_f = first.float()
    second_f = second.float()
    delta_rms = (first_f - second_f).square().mean().sqrt()
    pair_rms = (0.5 * (first_f.square().mean() + second_f.square().mean())).sqrt().clamp_min(1.0e-12)
    return float((delta_rms / pair_rms).item())


def _centered_energy_fraction(value: torch.Tensor, *, dim: int) -> float:
    value_f = value.float()
    centered = value_f - value_f.mean(dim=int(dim), keepdim=True)
    centered_energy = centered.square().mean()
    total_energy = value_f.square().mean().clamp_min(1.0e-24)
    return float((centered_energy / total_energy).item())


@contextmanager
def strict_fp32_replay_backend() -> Iterator[None]:
    """Temporarily force IEEE FP32 matmuls without changing training policy."""

    previous_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32


def overfit_success_passed(metrics: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    if "swap_dir_delta_min" in thresholds:
        swap_passed = float(metrics["swap_min_dir_delta"]) >= float(
            thresholds["swap_dir_delta_min"]
        )
    else:
        swap_passed = float(metrics["swap_min_total_delta"]) >= float(
            thresholds["swap_total_delta_min"]
        )
    base_passed = (
        float(metrics["matched_max_dir"]) <= float(thresholds["struct_dir_max"])
        and float(metrics["matched_max_scale"]) <= float(thresholds["struct_scale_max"])
        and float(metrics["matched_max_nrmse"]) <= float(thresholds["nrmse_max"])
        and swap_passed
    )
    if "fixed_x_w_swap_dir_delta_min" in thresholds:
        base_passed = base_passed and float(metrics["fixed_x_w_swap_min_dir_delta"]) >= float(
            thresholds["fixed_x_w_swap_dir_delta_min"]
        )
    if "matched_mean_dir_max" not in thresholds:
        return base_passed
    return base_passed and float(metrics["matched_mean_dir"]) < float(thresholds["matched_mean_dir_max"])


class TwoOperatorOverfitEvaluator:
    def __init__(
        self,
        *,
        source: OperatorBankTrainingDataset,
        output_path: str | Path,
        every_steps: int,
        patch_size: int,
        gamma: float,
        lambda_dir: float,
        lambda_scale: float,
        huber_delta: float,
        free_weight_steps: int = 50,
        free_weight_lr: float = 1.0e-2,
        include_identity_diagnostics: bool = False,
    ) -> None:
        if len(source._keys) != 2 or not source.exact_operator_cycle:
            raise ValueError("two-operator evaluator requires exactly two fixed-cycle operator keys")
        first, second = source._keys
        first_locations = source.operator_groups[first]
        second_locations = source.operator_groups[second]
        first_shape = tuple(first_locations[0].metadata["operator"]["matrix_shape"])
        second_shape = tuple(second_locations[0].metadata["operator"]["matrix_shape"])
        if first[1] != second[1] or first_shape != second_shape or len(first_locations) != len(second_locations):
            raise ValueError("two-operator evaluator requires same layer, shape, and tile count")
        if first[0] == second[0]:
            raise ValueError("two-operator evaluator requires distinct checkpoints")
        self.source = source
        self.output_path = Path(output_path)
        self.every_steps = int(every_steps)
        self.patch_size = int(patch_size)
        self.gamma = float(gamma)
        self.lambda_dir = float(lambda_dir)
        self.lambda_scale = float(lambda_scale)
        self.huber_delta = float(huber_delta)
        self.free_weight_steps = int(free_weight_steps)
        self.free_weight_lr = float(free_weight_lr)
        self.include_identity_diagnostics = bool(include_identity_diagnostics)
        self._positive_controls_done = False

    def should_run(self, step: int) -> bool:
        return int(step) == 1 or int(step) % self.every_steps == 0

    def _assignments(self, logical_indices: Sequence[int]) -> list[tuple[tuple[str, str], int]]:
        records = len(self.source)
        if len(logical_indices) != records:
            raise RuntimeError(f"one overfit step must consume exactly {records} tiles, got {len(logical_indices)}")
        plan, _views = self.source._cycle_plan(0)
        assignments = [plan[int(index) % records] for index in logical_indices]
        if len(set(assignments)) != records:
            raise RuntimeError("overfit optimizer step is not one exact full-set cycle")
        return assignments

    def _positive_controls(self, target_tiles: torch.Tensor) -> dict[str, float]:
        identity = _loss_metrics(
            target_tiles,
            target_tiles,
            patch_size=self.patch_size,
            gamma=self.gamma,
            lambda_dir=self.lambda_dir,
            lambda_scale=self.lambda_scale,
            huber_delta=self.huber_delta,
        )
        target = target_tiles.detach().float()
        with torch.enable_grad():
            generator = torch.Generator(device=target.device).manual_seed(17_029)
            scale = target.std().clamp_min(1.0e-6)
            free = torch.nn.Parameter(
                torch.randn(
                    target.shape,
                    dtype=torch.float32,
                    device=target.device,
                    generator=generator,
                )
                * scale
            )
            optimizer = torch.optim.AdamW([free], lr=self.free_weight_lr, weight_decay=0.0)
            initial = _loss_metrics(
                target,
                free,
                patch_size=self.patch_size,
                gamma=self.gamma,
                lambda_dir=self.lambda_dir,
                lambda_scale=self.lambda_scale,
                huber_delta=self.huber_delta,
            )
            for _ in range(self.free_weight_steps):
                optimizer.zero_grad(set_to_none=True)
                loss, _details = WeightQuantileVAE.patch_structure_loss(
                    target,
                    free,
                    patch_size=self.patch_size,
                    gamma=self.gamma,
                    lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale,
                    lambda_rec=0.0,
                    lambda_rel=0.0,
                    huber_delta=self.huber_delta,
                )
                loss.backward()
                optimizer.step()
            final = _loss_metrics(
                target,
                free,
                patch_size=self.patch_size,
                gamma=self.gamma,
                lambda_dir=self.lambda_dir,
                lambda_scale=self.lambda_scale,
                huber_delta=self.huber_delta,
            )
        if identity["total"] > 1.0e-6 or not final["total"] < initial["total"]:
            raise RuntimeError(
                f"structural-loss positive controls failed: identity={identity} initial={initial} final={final}"
            )
        return {
            "identity_total": identity["total"],
            "identity_dir": identity["dir"],
            "identity_scale": identity["scale"],
            "free_weight_initial_total": initial["total"],
            "free_weight_final_total": final["total"],
            "free_weight_steps": float(self.free_weight_steps),
        }

    @torch.no_grad()
    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        step: int,
        microbatches: Sequence[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Sequence[int],
            ]
        ],
    ) -> dict[str, float]:
        raw_model = _raw_model(model)
        was_training = raw_model.training
        raw_model.eval()
        try:
            logical_indices = tuple(
                int(index)
                for _W, _x, _x_mask, _d_in_mask, _d_out_mask, indices in microbatches
                for index in indices
            )
            assignments = self._assignments(logical_indices)
            lookup = {assignment: idx for idx, assignment in enumerate(assignments)}
            first, second = self.source._keys
            global_swapped_indices = [
                lookup[(second if key == first else first, local_index)]
                for key, local_index in assignments
            ]
            target_chunks: list[torch.Tensor] = []
            matched_chunks: list[torch.Tensor] = []
            swapped_chunks: list[torch.Tensor] = []
            latent_chunks: list[torch.Tensor] = []
            bridge_chunks: list[torch.Tensor] = []
            position_film_chunks: list[torch.Tensor] = []
            qnorm_chunks: list[torch.Tensor] = []
            query_mask_chunks: list[torch.Tensor] = []
            refresh_layer_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            refresh_raw_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            refresh_w_swap_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            refresh_w_swap_raw_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            refresh_x_swap_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            refresh_x_swap_raw_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            refresh_zero_w_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            refresh_zero_w_raw_chunks: dict[str, list[torch.Tensor]] = {"layer0": [], "layer9": []}
            v8_chunks: dict[str, list[torch.Tensor]] = {
                name: []
                for name in (
                    "z",
                    "raw_v",
                    "w_swap_z",
                    "w_swap_raw_v",
                    "x_swap_z",
                    "x_swap_raw_v",
                    "zero_z",
                    "zero_raw_v",
                    "scale_z",
                    "sum_z",
                    "zero_w_hat",
                    "w_swap_w_hat",
                )
            }
            v8_invariant_rows: list[dict[str, float]] = []
            v8_fp32_invariant_rows: list[dict[str, float]] = []
            v8_routing_rows: list[dict[str, float]] = []
            collect_refresh_diagnostics = bool(
                getattr(raw_model, "use_mandatory_cross_refresh_v7", False)
            )
            collect_v8_diagnostics = bool(getattr(raw_model, "use_clean_content_readout_v8", False))
            if collect_refresh_diagnostics and len(raw_model.encoder_layers) != 10:
                raise RuntimeError("V7 two-operator refresh diagnostics require the production 10-layer encoder")
            offset = 0
            for W, x, x_mask, d_in_mask, d_out_mask, indices in microbatches:
                size = len(indices)
                hook_outputs: dict[str, list[torch.Tensor]] = {
                    "bridge": [],
                    "position_film": [],
                    "qnorm": [],
                    "layer0": [],
                    "layer9": [],
                    "layer0_raw": [],
                    "layer9_raw": [],
                    "v8_z": [],
                }
                v8_readout_inputs: list[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                ] = []
                handles: list[torch.utils.hooks.RemovableHandle] = []
                if self.include_identity_diagnostics:
                    bridge = getattr(raw_model, "mandatory_latent_bridge", None)
                    position_film = getattr(raw_model, "position_only_film_v6", None)
                    qnorm = getattr(raw_model, "q_tokens_norm", None)
                    if not isinstance(bridge, torch.nn.Module) or not isinstance(qnorm, torch.nn.Module):
                        raise RuntimeError("epsilon-fork identity diagnostics require V5 bridge and q_tokens_norm")
                    handles.extend(
                        [
                            bridge.register_forward_hook(
                                lambda _module, _inputs, output: hook_outputs["bridge"].append(output.detach())
                            ),
                            qnorm.register_forward_hook(
                                lambda _module, _inputs, output: hook_outputs["qnorm"].append(output.detach())
                            ),
                        ]
                    )
                    if isinstance(position_film, torch.nn.Module):
                        handles.append(
                            position_film.register_forward_hook(
                                lambda _module, _inputs, output: hook_outputs["position_film"].append(
                                    output.detach()
                                )
                            )
                        )
                if collect_refresh_diagnostics:
                    handles.extend(
                        [
                            raw_model.encoder_layers[0].perceiver_block.register_forward_hook(
                                lambda _module, _inputs, output: hook_outputs["layer0"].append(output.detach())
                            ),
                            raw_model.encoder_layers[9].perceiver_block.register_forward_hook(
                                lambda _module, _inputs, output: hook_outputs["layer9"].append(output.detach())
                            ),
                            raw_model.encoder_layers[0].perceiver_block.dropout.register_forward_hook(
                                lambda _module, _inputs, output: hook_outputs["layer0_raw"].append(output.detach())
                            ),
                            raw_model.encoder_layers[9].perceiver_block.dropout.register_forward_hook(
                                lambda _module, _inputs, output: hook_outputs["layer9_raw"].append(output.detach())
                            ),
                        ]
                    )
                if collect_v8_diagnostics:
                    readout = getattr(raw_model, "clean_content_readout_v8", None)
                    if not isinstance(readout, torch.nn.Module):
                        raise RuntimeError("V8 diagnostics require clean_content_readout_v8")
                    readout.capture_routing_diagnostics = True

                    def capture_v8_readout_input(
                        _module: torch.nn.Module,
                        args: tuple[Any, ...],
                        kwargs: dict[str, Any],
                    ) -> None:
                        if len(args) != 2 or set(kwargs) != {"patch_mask", "output_mask"}:
                            raise RuntimeError("V8 readout input capture contract changed")
                        v8_readout_inputs.append(
                            (
                                args[0].detach(),
                                args[1].detach(),
                                kwargs["patch_mask"].detach(),
                                kwargs["output_mask"].detach(),
                            )
                        )

                    handles.append(
                        readout.register_forward_hook(
                            lambda _module, _inputs, output: hook_outputs["v8_z"].append(output.detach())
                        )
                    )
                    handles.append(
                        readout.register_forward_pre_hook(
                            capture_v8_readout_input,
                            with_kwargs=True,
                        )
                    )
                try:
                    matched_W_hat, _mu, _logvar, _pred_dirs, debug = raw_model.forward_debug(
                        W,
                        x,
                        x_mask=x_mask,
                        d_in_mask=d_in_mask,
                        d_out_mask=d_out_mask,
                    )
                finally:
                    for handle in handles:
                        handle.remove()
                    if collect_v8_diagnostics:
                        raw_model.clean_content_readout_v8.capture_routing_diagnostics = False
                if self.include_identity_diagnostics:
                    if len(hook_outputs["bridge"]) != 1 or len(hook_outputs["qnorm"]) != 1:
                        raise RuntimeError(
                            "epsilon-fork matched forward did not execute exactly one bridge and qnorm path"
                        )
                    latent_chunks.append(debug["latent_decoder_z"].detach())
                    bridge_chunks.append(hook_outputs["bridge"][0])
                    if hook_outputs["position_film"]:
                        if len(hook_outputs["position_film"]) != 1:
                            raise RuntimeError("V6 matched forward did not execute exactly one position FiLM")
                        position_film_chunks.append(hook_outputs["position_film"][0])
                    qnorm_chunks.append(hook_outputs["qnorm"][0])
                    query_mask_chunks.append(debug["decoder_query_mask"].detach())
                local_swap_global = global_swapped_indices[offset : offset + size]
                if any(index < offset or index >= offset + size for index in local_swap_global):
                    raise RuntimeError("paired latent swap crosses diagnostic microbatch boundaries")
                swap = torch.tensor(
                    [index - offset for index in local_swap_global],
                    dtype=torch.long,
                    device=W.device,
                )
                if collect_v8_diagnostics:
                    if len(hook_outputs["v8_z"]) != 1 or len(v8_readout_inputs) != 1:
                        raise RuntimeError("V8 matched forward did not execute/capture one clean readout")
                    v8_chunks["z"].append(hook_outputs["v8_z"][0])
                    v8_chunks["raw_v"].append(W.detach())
                    routing_diagnostics = raw_model.clean_content_readout_v8.last_routing_diagnostics
                    if not isinstance(routing_diagnostics, dict):
                        raise RuntimeError("V8 matched forward did not capture routing diagnostics")
                    v8_routing_rows.append(
                        {name: float(value.item()) for name, value in routing_diagnostics.items()}
                    )

                    def capture_v8_variant(
                        variant_W: torch.Tensor,
                        variant_x: torch.Tensor,
                        variant_x_mask: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        zs: list[torch.Tensor] = []
                        readout = raw_model.clean_content_readout_v8
                        variant_handle = (
                            readout.register_forward_hook(
                                lambda _module, _inputs, output: zs.append(output.detach())
                            )
                        )
                        try:
                            variant_w_hat, _mu, _logvar, _directions, _debug = raw_model.forward_debug(
                                variant_W,
                                variant_x,
                                x_mask=variant_x_mask,
                                d_in_mask=d_in_mask,
                                d_out_mask=d_out_mask,
                            )
                        finally:
                            variant_handle.remove()
                        if len(zs) != 1:
                            raise RuntimeError("V8 variant did not execute one clean readout")
                        return zs[0], variant_w_hat.detach()

                    variant_specs = {
                        "w_swap": (W.index_select(0, swap), x, x_mask),
                        "x_swap": (W, x.index_select(0, swap), x_mask.index_select(0, swap)),
                        "zero": (torch.zeros_like(W), x, x_mask),
                        "scale": (2.0 * W, x, x_mask),
                        "sum": (W + W.index_select(0, swap), x, x_mask),
                    }
                    variant_outputs = {
                        name: capture_v8_variant(*variant)
                        for name, variant in variant_specs.items()
                    }
                    readout_w, readout_x, readout_patch_mask, readout_output_mask = (
                        v8_readout_inputs[0]
                    )
                    with strict_fp32_replay_backend(), torch.autocast(
                        device_type=readout_w.device.type,
                        enabled=False,
                    ):
                        if torch.backends.cuda.matmul.allow_tf32:
                            raise RuntimeError("strict FP32 replay did not disable CUDA TF32")
                        if torch.get_float32_matmul_precision() != "highest":
                            raise RuntimeError("strict FP32 replay did not select highest precision")
                        fp32_kwargs = {
                            "patch_mask": readout_patch_mask,
                            "output_mask": readout_output_mask,
                        }
                        fp32_w = readout_w.float()
                        fp32_x = readout_x.float()
                        fp32_matched = raw_model.clean_content_readout_v8(
                            fp32_w,
                            fp32_x,
                            **fp32_kwargs,
                        )
                        fp32_swapped = raw_model.clean_content_readout_v8(
                            fp32_w.index_select(0, swap),
                            fp32_x,
                            **fp32_kwargs,
                        )
                        fp32_scaled = raw_model.clean_content_readout_v8(
                            2.0 * fp32_w,
                            fp32_x,
                            **fp32_kwargs,
                        )
                        fp32_sum = raw_model.clean_content_readout_v8(
                            fp32_w + fp32_w.index_select(0, swap),
                            fp32_x,
                            **fp32_kwargs,
                        )
                    v8_fp32_invariant_rows.append(
                        {
                            "readout_fp32_tf32_disabled_scaling_error": _paired_relative_delta(
                                fp32_scaled,
                                2.0 * fp32_matched,
                            ),
                            "readout_fp32_tf32_disabled_superposition_error": _paired_relative_delta(
                                fp32_sum,
                                fp32_matched + fp32_swapped,
                            ),
                            "readout_fp32_tf32_disabled_backend_allow_tf32": 0.0,
                            "readout_fp32_tf32_disabled_backend_precision_highest": 1.0,
                        }
                    )
                    for name in ("w_swap", "x_swap", "zero"):
                        v8_chunks[f"{name}_z"].append(variant_outputs[name][0])
                    v8_chunks["w_swap_raw_v"].append(W.index_select(0, swap).detach())
                    v8_chunks["x_swap_raw_v"].append(W.detach())
                    v8_chunks["zero_raw_v"].append(torch.zeros_like(W))
                    v8_chunks["scale_z"].append(variant_outputs["scale"][0])
                    v8_chunks["sum_z"].append(variant_outputs["sum"][0])
                    v8_chunks["zero_w_hat"].append(variant_outputs["zero"][1])
                    v8_chunks["w_swap_w_hat"].append(variant_outputs["w_swap"][1])
                    matched_z = hook_outputs["v8_z"][0]
                    matched_bridge = hook_outputs["bridge"][0]
                    q_tokens = debug["decoder_queries_init"]
                    q_mask = debug["decoder_query_mask"]
                    bridge = raw_model.mandatory_latent_bridge
                    bridge_scale = bridge(q_tokens, variant_outputs["scale"][0], query_mask=q_mask)
                    bridge_sum = bridge(q_tokens, variant_outputs["sum"][0], query_mask=q_mask)
                    bridge_w_swap = bridge(q_tokens, variant_outputs["w_swap"][0], query_mask=q_mask)
                    bridge_zero = bridge(q_tokens, variant_outputs["zero"][0], query_mask=q_mask)
                    v8_invariant_rows.append(
                        {
                            "readout_native_scaling_error": _paired_relative_delta(
                                variant_outputs["scale"][0], 2.0 * matched_z
                            ),
                            "readout_native_superposition_error": _paired_relative_delta(
                                variant_outputs["sum"][0], matched_z + variant_outputs["w_swap"][0]
                            ),
                            "readout_native_dtype_epsilon": torch.finfo(matched_z.dtype).eps,
                            "bridge_scaling_error": _paired_relative_delta(bridge_scale, 2.0 * matched_bridge),
                            "bridge_superposition_error": _paired_relative_delta(
                                bridge_sum, matched_bridge + bridge_w_swap
                            ),
                            "bridge_zero_rms": float(bridge_zero.float().square().mean().sqrt().item()),
                        }
                    )
                if collect_refresh_diagnostics:
                    if any(
                        len(hook_outputs[name]) != 1
                        for name in ("layer0", "layer9", "layer0_raw", "layer9_raw")
                    ):
                        raise RuntimeError("V7 matched forward did not execute exact layer0/layer9 raw+refresh paths")
                    refresh_layer_chunks["layer0"].append(hook_outputs["layer0"][0])
                    refresh_layer_chunks["layer9"].append(hook_outputs["layer9"][0])
                    refresh_raw_chunks["layer0"].append(hook_outputs["layer0_raw"][0])
                    refresh_raw_chunks["layer9"].append(hook_outputs["layer9_raw"][0])

                    def capture_refresh_variant(
                        variant_W: torch.Tensor,
                        variant_x: torch.Tensor,
                        variant_x_mask: torch.Tensor,
                    ) -> dict[str, torch.Tensor]:
                        outputs: dict[str, list[torch.Tensor]] = {
                            "layer0": [],
                            "layer9": [],
                            "layer0_raw": [],
                            "layer9_raw": [],
                        }
                        variant_handles = [
                            raw_model.encoder_layers[0].perceiver_block.register_forward_hook(
                                lambda _module, _inputs, output: outputs["layer0"].append(output.detach())
                            ),
                            raw_model.encoder_layers[9].perceiver_block.register_forward_hook(
                                lambda _module, _inputs, output: outputs["layer9"].append(output.detach())
                            ),
                            raw_model.encoder_layers[0].perceiver_block.dropout.register_forward_hook(
                                lambda _module, _inputs, output: outputs["layer0_raw"].append(output.detach())
                            ),
                            raw_model.encoder_layers[9].perceiver_block.dropout.register_forward_hook(
                                lambda _module, _inputs, output: outputs["layer9_raw"].append(output.detach())
                            ),
                        ]
                        try:
                            raw_model.forward_debug(
                                variant_W,
                                variant_x,
                                x_mask=variant_x_mask,
                                d_in_mask=d_in_mask,
                                d_out_mask=d_out_mask,
                            )
                        finally:
                            for variant_handle in variant_handles:
                                variant_handle.remove()
                        if any(len(values) != 1 for values in outputs.values()):
                            raise RuntimeError("V7 variant forward did not execute exact layer0/layer9 raw+refresh paths")
                        return {name: values[0] for name, values in outputs.items()}

                    w_swap_outputs = capture_refresh_variant(W.index_select(0, swap), x, x_mask)
                    x_swap_outputs = capture_refresh_variant(
                        W,
                        x.index_select(0, swap),
                        x_mask.index_select(0, swap),
                    )
                    zero_w_outputs = capture_refresh_variant(torch.zeros_like(W), x, x_mask)
                    for layer_name in ("layer0", "layer9"):
                        refresh_w_swap_chunks[layer_name].append(w_swap_outputs[layer_name])
                        refresh_w_swap_raw_chunks[layer_name].append(w_swap_outputs[f"{layer_name}_raw"])
                        refresh_x_swap_chunks[layer_name].append(x_swap_outputs[layer_name])
                        refresh_x_swap_raw_chunks[layer_name].append(x_swap_outputs[f"{layer_name}_raw"])
                        refresh_zero_w_chunks[layer_name].append(zero_w_outputs[layer_name])
                        refresh_zero_w_raw_chunks[layer_name].append(zero_w_outputs[f"{layer_name}_raw"])
                decoder_z = debug["latent_decoder_z"].index_select(0, swap)
                swapped_W_hat, _z, _zero_logvar, _swapped_dirs = raw_model._decode_from_decoder_latent(
                    decoder_z,
                    dist_patch_by_patch=debug["dist_patch_by_patch"],
                    patch_mask=debug["patch_mask"],
                    d_in_mask=d_in_mask,
                    d_out_mask=d_out_mask,
                    d_in=int(W.shape[1]),
                    d_out=int(W.shape[2]),
                    d_in_pad=int(debug["d_in_pad"]),
                    T=int(debug["T"]),
                )
                target_chunks.append(W)
                matched_chunks.append(matched_W_hat)
                swapped_chunks.append(swapped_W_hat)
                offset += size
            W = torch.cat(target_chunks, dim=0)
            matched_W_hat = torch.cat(matched_chunks, dim=0)
            swapped_W_hat = torch.cat(swapped_chunks, dim=0)
            metrics: dict[str, float] = {"step": float(step)}
            for operator_index, key in enumerate(self.source._keys):
                target = _stitch(W, assignments, self.source, key)
                matched = _stitch(matched_W_hat, assignments, self.source, key)
                swapped = _stitch(swapped_W_hat, assignments, self.source, key)
                zero_w_output = (
                    _stitch(torch.cat(v8_chunks["zero_w_hat"], dim=0), assignments, self.source, key)
                    if collect_v8_diagnostics
                    else None
                )
                fixed_x_w_swap_output = (
                    _stitch(torch.cat(v8_chunks["w_swap_w_hat"], dim=0), assignments, self.source, key)
                    if collect_v8_diagnostics
                    else None
                )
                matched_metrics = _loss_metrics(
                    target,
                    matched,
                    patch_size=self.patch_size,
                    gamma=self.gamma,
                    lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale,
                    huber_delta=self.huber_delta,
                )
                swapped_metrics = _loss_metrics(
                    target,
                    swapped,
                    patch_size=self.patch_size,
                    gamma=self.gamma,
                    lambda_dir=self.lambda_dir,
                    lambda_scale=self.lambda_scale,
                    huber_delta=self.huber_delta,
                )
                prefix = f"operator_{operator_index}"
                for name, value in matched_metrics.items():
                    metrics[f"{prefix}_matched_{name}"] = value
                for name, value in swapped_metrics.items():
                    metrics[f"{prefix}_swapped_{name}"] = value
                metrics[f"{prefix}_matched_nrmse"] = _nrmse(target, matched)
                metrics[f"{prefix}_swapped_nrmse"] = _nrmse(target, swapped)
                metrics[f"{prefix}_swap_dir_delta"] = swapped_metrics["dir"] - matched_metrics["dir"]
                metrics[f"{prefix}_swap_total_delta"] = swapped_metrics["total"] - matched_metrics["total"]
                if zero_w_output is not None:
                    zero_w_metrics = _loss_metrics(
                        target,
                        zero_w_output,
                        patch_size=self.patch_size,
                        gamma=self.gamma,
                        lambda_dir=self.lambda_dir,
                        lambda_scale=self.lambda_scale,
                        huber_delta=self.huber_delta,
                    )
                    metrics[f"{prefix}_zero_w_dir"] = zero_w_metrics["dir"]
                    metrics[f"{prefix}_zero_w_total"] = zero_w_metrics["total"]
                    metrics[f"{prefix}_zero_w_nrmse"] = _nrmse(target, zero_w_output)
                if fixed_x_w_swap_output is not None:
                    fixed_x_w_swap_metrics = _loss_metrics(
                        target,
                        fixed_x_w_swap_output,
                        patch_size=self.patch_size,
                        gamma=self.gamma,
                        lambda_dir=self.lambda_dir,
                        lambda_scale=self.lambda_scale,
                        huber_delta=self.huber_delta,
                    )
                    metrics[f"{prefix}_fixed_x_w_swap_dir"] = fixed_x_w_swap_metrics["dir"]
                    metrics[f"{prefix}_fixed_x_w_swap_total"] = fixed_x_w_swap_metrics["total"]
                    metrics[f"{prefix}_fixed_x_w_swap_nrmse"] = _nrmse(target, fixed_x_w_swap_output)
                    metrics[f"{prefix}_fixed_x_w_swap_dir_delta"] = (
                        fixed_x_w_swap_metrics["dir"] - matched_metrics["dir"]
                    )
                    metrics[f"{prefix}_fixed_x_w_swap_total_delta"] = (
                        fixed_x_w_swap_metrics["total"] - matched_metrics["total"]
                    )
            metrics["matched_max_dir"] = max(metrics["operator_0_matched_dir"], metrics["operator_1_matched_dir"])
            metrics["matched_mean_dir"] = 0.5 * (
                metrics["operator_0_matched_dir"] + metrics["operator_1_matched_dir"]
            )
            metrics["matched_max_scale"] = max(
                metrics["operator_0_matched_scale"], metrics["operator_1_matched_scale"]
            )
            metrics["matched_max_nrmse"] = max(
                metrics["operator_0_matched_nrmse"], metrics["operator_1_matched_nrmse"]
            )
            metrics["swap_min_total_delta"] = min(
                metrics["operator_0_swap_total_delta"], metrics["operator_1_swap_total_delta"]
            )
            metrics["swap_min_dir_delta"] = min(
                metrics["operator_0_swap_dir_delta"], metrics["operator_1_swap_dir_delta"]
            )
            if collect_v8_diagnostics:
                metrics["fixed_x_w_swap_min_dir_delta"] = min(
                    metrics["operator_0_fixed_x_w_swap_dir_delta"],
                    metrics["operator_1_fixed_x_w_swap_dir_delta"],
                )
                metrics["fixed_x_w_swap_min_total_delta"] = min(
                    metrics["operator_0_fixed_x_w_swap_total_delta"],
                    metrics["operator_1_fixed_x_w_swap_total_delta"],
                )
                swap_index = torch.tensor(global_swapped_indices, dtype=torch.long, device=W.device)
                metrics["identity_v8_output_paired_relative_delta"] = _paired_relative_delta(
                    matched_W_hat,
                    matched_W_hat.index_select(0, swap_index),
                )
            if collect_refresh_diagnostics:
                swap_index = torch.tensor(global_swapped_indices, dtype=torch.long, device=W.device)
                for layer_name in ("layer0", "layer9"):
                    matched_refresh = torch.cat(refresh_layer_chunks[layer_name], dim=0)
                    paired_refresh = matched_refresh.index_select(0, swap_index)
                    w_swap_refresh = torch.cat(refresh_w_swap_chunks[layer_name], dim=0)
                    raw_refresh = torch.cat(refresh_raw_chunks[layer_name], dim=0)
                    paired_raw_refresh = raw_refresh.index_select(0, swap_index)
                    w_swap_raw_refresh = torch.cat(refresh_w_swap_raw_chunks[layer_name], dim=0)
                    x_swap_refresh = torch.cat(refresh_x_swap_chunks[layer_name], dim=0)
                    x_swap_raw_refresh = torch.cat(refresh_x_swap_raw_chunks[layer_name], dim=0)
                    zero_w_refresh = torch.cat(refresh_zero_w_chunks[layer_name], dim=0)
                    zero_w_raw_refresh = torch.cat(refresh_zero_w_raw_chunks[layer_name], dim=0)
                    metrics[f"identity_encoder_{layer_name}_paired_relative_delta"] = _paired_relative_delta(
                        matched_refresh,
                        paired_refresh,
                    )
                    metrics[f"identity_encoder_{layer_name}_w_swap_relative_delta"] = _paired_relative_delta(
                        matched_refresh,
                        w_swap_refresh,
                    )
                    metrics[f"identity_encoder_{layer_name}_x_swap_relative_delta"] = _paired_relative_delta(
                        matched_refresh,
                        x_swap_refresh,
                    )
                    metrics[f"identity_encoder_{layer_name}_zero_w_relative_delta"] = _paired_relative_delta(
                        matched_refresh,
                        zero_w_refresh,
                    )
                    metrics[f"identity_encoder_{layer_name}_rms"] = float(
                        matched_refresh.float().square().mean().sqrt().item()
                    )
                    metrics[f"identity_encoder_{layer_name}_zero_w_rms"] = float(
                        zero_w_refresh.float().square().mean().sqrt().item()
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_rms"] = float(
                        raw_refresh.float().square().mean().sqrt().item()
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_query_centered_fraction"] = (
                        _centered_energy_fraction(raw_refresh, dim=1)
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_batch_centered_fraction"] = (
                        _centered_energy_fraction(raw_refresh, dim=0)
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_paired_relative_delta"] = (
                        _paired_relative_delta(raw_refresh, paired_raw_refresh)
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_w_swap_relative_delta"] = (
                        _paired_relative_delta(raw_refresh, w_swap_raw_refresh)
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_x_swap_relative_delta"] = (
                        _paired_relative_delta(raw_refresh, x_swap_raw_refresh)
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_zero_w_relative_delta"] = (
                        _paired_relative_delta(raw_refresh, zero_w_raw_refresh)
                    )
                    metrics[f"identity_encoder_{layer_name}_raw_cross_zero_w_rms"] = float(
                        zero_w_raw_refresh.float().square().mean().sqrt().item()
                    )
            if collect_v8_diagnostics:
                matched_z = torch.cat(v8_chunks["z"], dim=0)
                matched_v = torch.cat(v8_chunks["raw_v"], dim=0)
                metrics["identity_v8_z_rms"] = float(matched_z.float().square().mean().sqrt().item())
                metrics["identity_v8_raw_v_rms"] = float(matched_v.float().square().mean().sqrt().item())
                for variant in ("w_swap", "x_swap", "zero"):
                    variant_z = torch.cat(v8_chunks[f"{variant}_z"], dim=0)
                    variant_v = torch.cat(v8_chunks[f"{variant}_raw_v"], dim=0)
                    metrics[f"identity_v8_z_{variant}_relative_delta"] = _paired_relative_delta(
                        matched_z, variant_z
                    )
                    metrics[f"identity_v8_raw_v_{variant}_relative_delta"] = _paired_relative_delta(
                        matched_v, variant_v
                    )
                metrics["identity_v8_zero_z_rms"] = float(
                    torch.cat(v8_chunks["zero_z"], dim=0).float().square().mean().sqrt().item()
                )
                metrics["identity_v8_zero_w_output_rms"] = float(
                    torch.cat(v8_chunks["zero_w_hat"], dim=0).float().square().mean().sqrt().item()
                )
                for name in v8_invariant_rows[0]:
                    metrics[f"identity_v8_{name}"] = max(float(row[name]) for row in v8_invariant_rows)
                for name in v8_fp32_invariant_rows[0]:
                    metrics[f"identity_v8_{name}"] = max(
                        float(row[name]) for row in v8_fp32_invariant_rows
                    )
                for name in v8_routing_rows[0]:
                    values = [float(row[name]) for row in v8_routing_rows]
                    metrics[f"identity_v8_routing_{name}"] = min(values) if name == "unique_argmax_min" else sum(values) / len(values)
            if self.include_identity_diagnostics:
                latent = torch.cat(latent_chunks, dim=0).float()
                swapped_latent = latent.index_select(
                    0,
                    torch.tensor(global_swapped_indices, dtype=torch.long, device=latent.device),
                )
                latent_delta_rms = (latent - swapped_latent).square().mean().sqrt()
                latent_pair_rms = (
                    0.5 * (latent.square().mean() + swapped_latent.square().mean())
                ).sqrt().clamp_min(1.0e-12)
                metrics["identity_raw_z_paired_relative_delta"] = float(
                    (latent_delta_rms / latent_pair_rms).item()
                )

                query_mask = torch.cat(query_mask_chunks, dim=0).bool()
                identity_tensors = [("bridge", bridge_chunks), ("qnorm", qnorm_chunks)]
                if position_film_chunks:
                    identity_tensors.append(("posfilm", position_film_chunks))
                for name, values in identity_tensors:
                    tensor = torch.cat(values, dim=0).float()
                    paired_tensor = tensor.index_select(
                        0,
                        torch.tensor(global_swapped_indices, dtype=torch.long, device=tensor.device),
                    )
                    metrics[f"identity_{name}_paired_relative_delta"] = _paired_relative_delta(
                        tensor, paired_tensor
                    )
                    mask = query_mask.unsqueeze(-1).to(tensor.dtype)
                    counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                    mean = (tensor * mask).sum(dim=1, keepdim=True) / counts
                    centered = (tensor - mean) * mask
                    denominator = (mask.sum() * tensor.shape[-1]).clamp_min(1.0)
                    centered_energy = centered.square().sum() / denominator
                    total_energy = (tensor * mask).square().sum() / denominator
                    metrics[f"identity_{name}_query_centered_energy"] = float(centered_energy.item())
                    metrics[f"identity_{name}_query_centered_fraction"] = float(
                        (centered_energy / total_energy.clamp_min(1.0e-24)).item()
                    )
            if not self._positive_controls_done:
                metrics.update(self._positive_controls(W.detach()))
                self._positive_controls_done = True
        finally:
            raw_model.train(was_training)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        return metrics


__all__ = [
    "TwoOperatorOverfitEvaluator",
    "TwoOperatorTrainingDataset",
    "overfit_success_passed",
    "two_operator_data_pipeline",
]
