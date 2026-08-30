from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import yaml

from big_vae.models.big_weight_vae import BigWeightVAE
from training.big_vae.model_config import build_big_vae_model_config

from .official_bridge import OfficialWeightCLIPBridge


def _required_path(value: str | Path, field: str) -> Path:
    if str(value).startswith("REQUIRED_"):
        raise ValueError(f"{field} is a required runtime path, got unresolved marker {value!r}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state", payload.get("model", payload.get("state_dict")))
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no model_state/model/state_dict mapping")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


class BigVAETileDecoderAdapter(nn.Module):
    """Expose the deterministic BigWeightVAE decoder through ``decode_tiles``."""

    def __init__(self, model: BigWeightVAE, *, provenance: Mapping[str, Any]) -> None:
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.provenance = dict(provenance)
        cfg = model.cfg.big_vae
        if bool(cfg.use_latent_sampling) or bool(cfg.use_encoder_mu_head):
            raise ValueError("primary BigVAE tile adapter requires a deterministic AE checkpoint")

    def decode_tiles(
        self,
        codes: torch.Tensor,
        *,
        activation_context: torch.Tensor | None,
        architecture_features: torch.Tensor,
        tile_mask: torch.Tensor | None,
        weight_mask: torch.Tensor,
        tile_indices: torch.Tensor,
        layer_key: str,
    ) -> torch.Tensor:
        del architecture_features, tile_mask, tile_indices, layer_key
        if activation_context is None:
            raise ValueError("BigVAE decoder requires native activation context")
        if activation_context.ndim != 3 or activation_context.shape[0] != codes.shape[0]:
            raise ValueError("activation_context must be [tiles,rows,tile_d_in]")
        if weight_mask.ndim != 3 or weight_mask.shape[0] != codes.shape[0]:
            raise ValueError("weight_mask must be [tiles,tile_d_in,tile_d_out]")
        d_in, d_out = int(weight_mask.shape[1]), int(weight_mask.shape[2])
        d_in_mask = weight_mask.any(dim=2)
        d_out_mask = weight_mask.any(dim=1)
        x_mask = torch.ones(activation_context.shape[:2], dtype=torch.bool, device=activation_context.device)
        (
            tokens,
            d_in_pad,
            patch_mask,
            _structural_patch_mask,
            _dist_var,
            dist_patch,
            _dist_pooled,
        ) = self.model._encode_distribution_context(
            activation_context,
            x_mask=x_mask,
            d_in_mask=d_in_mask,
        )
        output = self.model._decode_from_latent_slots(
            codes,
            dist_patch_by_patch=dist_patch,
            patch_mask=patch_mask,
            d_in_mask=d_in_mask,
            d_out_mask=d_out_mask,
            d_in=d_in,
            d_out=d_out,
            d_in_pad=d_in_pad,
            T=tokens,
            disable_z_shortcut=bool(self.model.cfg.big_vae.disable_z_shortcut),
        )[0]
        return output * weight_mask.to(dtype=output.dtype)


def build_bigvae_tile_decoder_adapter(
    *,
    checkpoint_path: str,
    model_config_path: str,
    device: str = "cpu",
) -> BigVAETileDecoderAdapter:
    checkpoint = _required_path(checkpoint_path, "checkpoint_path")
    config_path = _required_path(model_config_path, "model_config_path")
    raw_config = yaml.safe_load(config_path.read_text())
    model = BigWeightVAE(build_big_vae_model_config(raw_config))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(_model_state(payload), strict=True)
    model.to(device)
    return BigVAETileDecoderAdapter(
        model,
        provenance={
            "codec": "ours",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "model_config_path": str(config_path),
            "model_config_sha256": _sha256(config_path),
        },
    )


class _StateDictModule(nn.Module):
    """Minimal official ``Checkpoint`` carrier without reinterpreting state keys."""

    def __init__(self, state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self._state = {key: value.detach().cpu().clone() for key, value in state.items()}

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore[override]
        del args, kwargs
        return {key: value.clone() for key, value in self._state.items()}


def _official_window_batch(
    tokenizer: Any,
    state: Mapping[str, torch.Tensor],
    *,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the pinned official ``WindowedDataset`` full-coverage algorithm.

    The released ResNet substrate is sparse/full-model, 512 tokens per window,
    ``num_windows_per_model='auto'``, padded, consecutive and non-overlapping.
    Calling the official class (instead of duplicating slicing locally) makes any
    future change in its mask/position padding semantics visible to parity tests.
    """

    datasets = importlib.import_module("sane.data.datasets")
    checkpoint = datasets.Checkpoint(model=_StateDictModule(state), metadata={})
    checkpoints = datasets.CheckpointsDataset([checkpoint])
    windowed = datasets.WindowedDataset(
        checkpoints,
        tokenizer,
        window_size=int(window_size),
        num_windows_per_model="auto",
        pad_windows=True,
        pad_to_max_length=False,
        allow_overlapping_windows=False,
        window_distribution_strategy="consecutive",
    )
    windows = windowed.__getmodel__(0)
    if not windows:
        raise ValueError("official WindowedDataset emitted no windows")
    tokens, masks, positions = (torch.stack([window[index] for window in windows]) for index in range(3))
    if tokens.shape[1] != window_size or masks.shape != tokens.shape:
        raise ValueError("official WeightCLIP window/mask shape mismatch")
    if positions.shape[:2] != tokens.shape[:2]:
        raise ValueError("official WeightCLIP window/position shape mismatch")
    return tokens, masks.to(dtype=torch.bool), positions


class OfficialWeightCLIPMultiWindowDecoderAdapter(nn.Module):
    """Complete-checkpoint multi-window extension over the official substrate.

    Each official window is encoded/decoded as an independent batch item. Decoded
    windows are concatenated in the exact official consecutive order and passed once
    to ``tokenizer.detokenize``. No giant OOD transformer sequence is ever created.
    The released mapper itself handles only one 512-token window, so this adapter and
    any mapper/flow applied independently to all windows must never be labeled the
    released end-to-end baseline.
    """

    def __init__(
        self,
        weight_model: nn.Module,
        tokenizer: Any,
        *,
        anchor_state: Mapping[str, torch.Tensor],
        window_size: int,
        provenance: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.weight_model = weight_model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.anchor_state = {key: value.detach().clone() for key, value in anchor_state.items()}
        tokens, mask, position = _official_window_batch(tokenizer, anchor_state, window_size=window_size)
        self.register_buffer("anchor_tokens", tokens, persistent=True)
        self.register_buffer("anchor_mask", mask, persistent=True)
        self.register_buffer("anchor_pos", position.to(dtype=torch.long), persistent=True)
        weight_keys = tuple(key for key in anchor_state if "weight" in key)
        if not weight_keys:
            raise ValueError("official sparse tokenizer found no ordered weight keys")
        layer_ids = position[..., 1].to(dtype=torch.long)
        valid_layer_ids = layer_ids[mask.any(dim=-1)]
        if valid_layer_ids.numel() and int(valid_layer_ids.max()) >= len(weight_keys):
            raise ValueError("official tokenizer layer ids exceed ordered state-dict weight keys")
        body_layers = torch.tensor(
            [_is_controlled_body_key(key) for key in weight_keys],
            dtype=torch.bool,
            device=layer_ids.device,
        )
        body_token_mask = mask.any(dim=-1) & body_layers[layer_ids.clamp(min=0, max=len(weight_keys) - 1)]
        body_scalar_mask = mask & body_token_mask.unsqueeze(-1)
        self.register_buffer("body_token_mask", body_token_mask, persistent=True)
        self.register_buffer("body_scalar_mask", body_scalar_mask, persistent=True)
        self.ordered_weight_keys = weight_keys
        self.window_size = int(window_size)
        self.provenance = dict(provenance)
        self.provenance.update(
            {
                "window_size": self.window_size,
                "window_count": self.window_count,
                "windowing": "official_WindowedDataset_auto_padded_consecutive_nonoverlap",
                "method_status": "multiwindow_extension_not_released_mapper",
                "per_window_mapping": True,
                "single_detokenize_after_ordered_concatenation": True,
            }
        )

    @property
    def window_count(self) -> int:
        return int(self.anchor_tokens.shape[0])

    @property
    def window_token_mask(self) -> torch.Tensor:
        return self.anchor_mask.any(dim=-1)

    def rate_ledger(self, latent_dim: int | None = None) -> dict[str, int | float]:
        """Report native full-window and controlled body-effective code rates."""

        if latent_dim is None:
            latent_dim = int(getattr(self.weight_model.encoder, "output_dim"))
        full_window_tokens = int(self.anchor_tokens.shape[0] * self.anchor_tokens.shape[1])
        native_tokens = int(self.window_token_mask.sum().item())
        body_tokens = int(self.body_token_mask.sum().item())
        native_values = native_tokens * int(latent_dim)
        body_values = body_tokens * int(latent_dim)
        native_scalars = int(self.anchor_mask.sum().item())
        body_scalars = int(self.body_scalar_mask.sum().item())
        return {
            "full_window_tokens": full_window_tokens,
            "native_valid_tokens": native_tokens,
            "body_effective_tokens": body_tokens,
            "nonbody_valid_tokens": native_tokens - body_tokens,
            "latent_dim": int(latent_dim),
            "native_latent_values": native_values,
            "body_effective_latent_values": body_values,
            "native_decoded_scalars": native_scalars,
            "body_decoded_scalars": body_scalars,
            "native_scalars_per_latent_value": native_scalars / max(native_values, 1),
            "body_scalars_per_effective_latent_value": body_scalars / max(body_values, 1),
        }

    def encode_anchor(self) -> torch.Tensor:
        """Encode every official window independently, preserving window grouping."""

        device = next(self.weight_model.parameters()).device
        return self.weight_model.encoder(self.anchor_tokens.to(device), self.anchor_pos.to(device))

    def decode_state(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        expected = (self.window_count, self.window_size)
        if latent.ndim != 3 or tuple(latent.shape[:2]) != expected:
            raise ValueError(
                "official WeightCLIP decode requires grouped [windows,window_size,dim] "
                f"with prefix {expected}, got {tuple(latent.shape)}"
            )
        # Batch dimension is the official window unit. Transformer attention never
        # crosses a window boundary and therefore stays inside its trained substrate.
        decoded_windows = self.weight_model.decoder(latent, self.anchor_pos.to(latent.device))
        if tuple(decoded_windows.shape[:2]) != expected:
            raise ValueError("official decoder changed the window grouping")
        decoded = decoded_windows.reshape(-1, decoded_windows.shape[-1])
        mask = self.anchor_mask.reshape(-1, self.anchor_mask.shape[-1]).to(latent.device)
        position = self.anchor_pos.reshape(-1, self.anchor_pos.shape[-1]).to(latent.device)
        reference = {key: value.to(latent.device) for key, value in self.anchor_state.items()}
        return self.tokenizer.detokenize(
            decoded,
            mask=mask,
            position=position,
            reference_statedict=reference,
        )


def _is_controlled_body_key(key: str) -> bool:
    """Controlled benchmark transfers convolution and BN affine parameters only."""

    is_conv = key == "conv1.weight" or ".conv" in key or ".shortcut.0.weight" in key
    is_bn_affine = key.startswith("bn1.") or ".bn" in key or ".shortcut.1." in key
    return bool(is_conv or is_bn_affine)


def build_official_weightclip_multiwindow_decoder_adapter(
    *,
    official_repo: str,
    cache_dir: str,
    checkpoint_path: str,
    dataset_encoder_path: str,
    anchor_state: Mapping[str, torch.Tensor],
    window_size: int = 512,
    device: str = "cpu",
    checkpoint_data_config: Mapping[str, Any] | None = None,
) -> OfficialWeightCLIPMultiWindowDecoderAdapter:
    checkpoint = _required_path(checkpoint_path, "checkpoint_path")
    dataset_encoder = _required_path(dataset_encoder_path, "dataset_encoder_path")
    bridge = OfficialWeightCLIPBridge(official_repo, cache_dir, device=device)
    bundle = bridge.load(
        checkpoint_path=checkpoint,
        dataset_encoder_path=dataset_encoder,
        checkpoint_data_config=checkpoint_data_config,
    )
    return OfficialWeightCLIPMultiWindowDecoderAdapter(
        bundle.weight_model,
        bundle.tokenizer,
        anchor_state=anchor_state,
        window_size=window_size,
        provenance={
            "codec": "weightclip",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "official": bundle.provenance,
        },
    )


def build_official_weightclip_tokenizer_only_decoder_adapter(
    *,
    official_repo: str,
    cache_dir: str,
    checkpoint_path: str,
    dataset_encoder_path: str,
    anchor_state: Mapping[str, torch.Tensor],
    anchor_state_sha256: str,
    window_size: int = 512,
    device: str = "cpu",
) -> OfficialWeightCLIPMultiWindowDecoderAdapter:
    """Production factory that never resolves the checkpoint's stale zoo roots."""

    checkpoint = _required_path(checkpoint_path, "checkpoint_path")
    dataset_encoder = _required_path(dataset_encoder_path, "dataset_encoder_path")
    bridge = OfficialWeightCLIPBridge(official_repo, cache_dir, device=device)
    bundle = bridge.load_codec_tokenizer_only(
        reference_state=anchor_state,
        reference_state_sha256=anchor_state_sha256,
        checkpoint_path=checkpoint,
        dataset_encoder_path=dataset_encoder,
        allow_download=False,
    )
    return OfficialWeightCLIPMultiWindowDecoderAdapter(
        bundle.weight_model,
        bundle.tokenizer,
        anchor_state=anchor_state,
        window_size=window_size,
        provenance={
            "codec": "weightclip",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "reference_state_sha256": anchor_state_sha256,
            "official": bundle.provenance,
        },
    )
