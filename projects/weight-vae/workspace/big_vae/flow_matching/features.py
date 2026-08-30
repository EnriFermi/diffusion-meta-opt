from __future__ import annotations

from typing import Final

import torch


# Frozen E1 schema. Every field has substrate-invariant meaning; codec-native
# window/group/slot positions are deliberately absent.
FLOW_ARCHITECTURE_FEATURE_SCHEMA: Final[tuple[str, ...]] = (
    "valid_token",
    "controlled_body",
    "op_conv",
    "op_batchnorm",
    "op_head",
    "op_other",
    "kernel_h",
    "kernel_w",
    "stride_h",
    "stride_w",
    "normalized_depth",
    "role_stem",
    "role_residual_conv1",
    "role_residual_conv2",
    "role_projection",
    "role_batchnorm",
    "role_head",
    "role_other",
    "tile_row_normalized",
    "tile_col_normalized",
    "tile_grid_rows_log",
    "tile_grid_cols_log",
    "operator_input_rows_log",
    "operator_output_cols_log",
)
FLOW_ARCHITECTURE_FEATURE_DIM: Final[int] = len(FLOW_ARCHITECTURE_FEATURE_SCHEMA)


def layer_role(key: str) -> str:
    normalized = key.removesuffix(".weight").removesuffix(".bias")
    if normalized == "conv1":
        return "stem"
    if normalized == "fc":
        return "head"
    if normalized == "bn1" or ".bn" in normalized or ".shortcut.1" in normalized:
        return "batchnorm"
    if ".shortcut.0" in normalized:
        return "projection"
    if normalized.endswith(".conv1"):
        return "residual_conv1"
    if normalized.endswith(".conv2"):
        return "residual_conv2"
    return "other"


def semantic_token_features(
    *,
    valid: torch.Tensor,
    body: torch.Tensor,
    layer_indices: torch.Tensor,
    layer_keys: tuple[str, ...],
    operator_matrix_shapes: tuple[tuple[int, int], ...],
    kernels: tuple[tuple[int, int], ...],
    strides: tuple[tuple[int, int], ...],
    normalized_depths: tuple[float, ...],
    tile_rows: torch.Tensor,
    tile_cols: torch.Tensor,
) -> torch.Tensor:
    """Build exact E1 metadata for each codec token/slot.

    ``operator_matrix_shapes`` always means ``[in*k*k, out]`` for convolutions.
    ``tile_rows/cols`` use the common conceptual 128x128 operator tiling. Ours
    repeats the tile metadata over its AE slots. WeightCLIP maps each sparse token
    to the conceptual tile containing its first scalar; a deviation ledger must
    disclose that one sparse token can span several conceptual input tiles.
    """

    geometry = (valid, body, layer_indices, tile_rows, tile_cols)
    if valid.ndim != 2 or any(item.shape != valid.shape for item in geometry[1:]):
        raise ValueError("E1 token geometry tensors must share [groups,tokens]")
    if not (
        layer_keys
        and len(layer_keys) == len(operator_matrix_shapes) == len(kernels) == len(strides) == len(normalized_depths)
    ):
        raise ValueError("E1 layer metadata ledger is empty or misaligned")
    if layer_indices[valid].numel() and (
        int(layer_indices[valid].min()) < 0 or int(layer_indices[valid].max()) >= len(layer_keys)
    ):
        raise ValueError("valid token layer id lies outside the E1 layer ledger")
    groups, tokens = valid.shape
    output = torch.zeros((groups, tokens, FLOW_ARCHITECTURE_FEATURE_DIM), dtype=torch.float32)
    output[..., 0] = valid.float()
    output[..., 1] = body.float()
    roles = ("stem", "residual_conv1", "residual_conv2", "projection", "batchnorm", "head", "other")
    for layer_index, (key, shape, kernel, stride, depth) in enumerate(
        zip(layer_keys, operator_matrix_shapes, kernels, strides, normalized_depths, strict=True)
    ):
        selected = valid & (layer_indices == layer_index)
        if not selected.any():
            continue
        input_rows, output_cols = map(int, shape)
        grid_rows = (input_rows + 127) // 128
        grid_cols = (output_cols + 127) // 128
        role = layer_role(key)
        op_index = 2 if role in {"stem", "residual_conv1", "residual_conv2", "projection"} else 3 if role == "batchnorm" else 4 if role == "head" else 5
        output[..., op_index][selected] = 1.0
        output[..., 6][selected] = float(kernel[0])
        output[..., 7][selected] = float(kernel[1])
        output[..., 8][selected] = float(stride[0])
        output[..., 9][selected] = float(stride[1])
        output[..., 10][selected] = float(depth)
        output[..., 11 + roles.index(role)][selected] = 1.0
        output[..., 18][selected] = tile_rows[selected].float() / max(grid_rows - 1, 1)
        output[..., 19][selected] = tile_cols[selected].float() / max(grid_cols - 1, 1)
        output[..., 20][selected] = float(torch.log1p(torch.tensor(grid_rows)))
        output[..., 21][selected] = float(torch.log1p(torch.tensor(grid_cols)))
        output[..., 22][selected] = float(torch.log1p(torch.tensor(input_rows)))
        output[..., 23][selected] = float(torch.log1p(torch.tensor(output_cols)))
    output *= valid.unsqueeze(-1)
    return output


def validate_semantic_token_features(features: torch.Tensor, code: torch.Tensor) -> None:
    expected = (*code.shape[:2], FLOW_ARCHITECTURE_FEATURE_DIM)
    if tuple(features.shape) != expected:
        raise ValueError(f"architecture features must use frozen E1 schema {expected}, got {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("architecture features contain NaN/inf")
    valid = features[..., 0].bool()
    if torch.any(features[..., 1].bool() & ~valid):
        raise ValueError("E1 controlled_body marks a padded token")
    op_sum = features[..., 2:6].sum(dim=-1)
    role_sum = features[..., 11:18].sum(dim=-1)
    if not torch.all(op_sum[valid] == 1) or not torch.all(role_sum[valid] == 1):
        raise ValueError("every valid E1 token requires exactly one op type and residual role")


def retokenize_semantic_group_features(features: torch.Tensor, token_count: int) -> torch.Tensor:
    """Repeat codec-neutral OOD E1 tile metadata after ours fixes AE slot count."""

    if tuple(features.shape[1:]) != (1, FLOW_ARCHITECTURE_FEATURE_DIM):
        raise ValueError("retokenization requires [groups,1,E1_feature_dim] templates")
    return features.expand(int(features.shape[0]), int(token_count), -1).clone()
