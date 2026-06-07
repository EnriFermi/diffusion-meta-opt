from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from post_train_research.big_vae_latent_flattening.flow import RealNVPConfig, RealNVPFlow
from ..common import env_bool, env_path


_ADAPTER_CHECKPOINT_ENV = "EVAL_DECODER_ADAPTER_CHECKPOINT"
_LEGACY_FLOW_CHECKPOINT_ENV = "EVAL_LATENT_FLATTENING_CHECKPOINT"


@dataclass(slots=True)
class DecoderAdapterOutput:
    W_hat: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    pred_dirs: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class IdentityDecoderAdapter:
    kind = "identity"

    def metadata(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def decode(
        self,
        *,
        model: nn.Module,
        W_s: torch.Tensor,
        x_s: torch.Tensor,
        x_mask_s: torch.Tensor,
        d_in_mask_s: torch.Tensor,
        d_out_mask_s: torch.Tensor,
    ) -> DecoderAdapterOutput:
        W_hat, mu, logvar, pred_dirs = model(
            W_s,
            x_s,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        return DecoderAdapterOutput(W_hat=W_hat, mu=mu, logvar=logvar, pred_dirs=pred_dirs)


def _flow_config_from_payload(payload: dict[str, Any], *, latent_dim: int) -> RealNVPConfig:
    raw_config = payload.get("config", {})
    if not isinstance(raw_config, dict):
        raw_config = {}
    raw_flow = raw_config.get("flow", {})
    if not isinstance(raw_flow, dict):
        raw_flow = {}
    return RealNVPConfig(
        dim=int(latent_dim),
        num_layers=int(raw_flow.get("num_layers", 8)),
        hidden_dim=int(raw_flow.get("hidden_dim", 512)),
        network_depth=int(raw_flow.get("network_depth", 2)),
        log_scale_clamp=float(raw_flow.get("log_scale_clamp", 2.0)),
        dropout=float(raw_flow.get("dropout", 0.0)),
    )


def _payload_big_vae_checkpoint(payload: dict[str, Any]) -> str:
    direct = str(payload.get("big_vae_checkpoint", "")).strip()
    if direct:
        return direct
    raw_config = payload.get("config", {})
    if not isinstance(raw_config, dict):
        return ""
    raw_big_vae = raw_config.get("big_vae", {})
    if not isinstance(raw_big_vae, dict):
        return ""
    return str(raw_big_vae.get("checkpoint", "")).strip()


def _paths_match(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except Exception:
        return str(left).strip() == str(right).strip()


def latent_flattening_payload_big_vae_checkpoint(payload: dict[str, Any]) -> str:
    return _payload_big_vae_checkpoint(payload)


def paths_match(left: str | Path, right: str | Path) -> bool:
    return _paths_match(left, right)


def _masked_weight_mse(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    d_in_mask: torch.Tensor,
    d_out_mask: torch.Tensor,
) -> torch.Tensor:
    valid = d_in_mask.to(dtype=lhs.dtype).unsqueeze(-1) * d_out_mask.to(dtype=lhs.dtype).unsqueeze(1)
    denom = valid.sum().clamp_min(1.0)
    return ((lhs - rhs).pow(2) * valid).sum() / denom


def _decode_big_vae_decoder_latent(
    *,
    model: nn.Module,
    decoder_z: torch.Tensor,
    debug_info: dict[str, Any],
    W_s: torch.Tensor,
    d_in_mask_s: torch.Tensor,
    d_out_mask_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    decode_from_decoder_latent = getattr(model, "_decode_from_decoder_latent", None)
    if not callable(decode_from_decoder_latent):
        raise TypeError("Decoder adapter requires BigVAE _decode_from_decoder_latent API")
    return decode_from_decoder_latent(
        decoder_z,
        dist_patch_by_patch=debug_info.get("dist_patch_by_patch"),
        patch_mask=debug_info["patch_mask"],
        d_in_mask=d_in_mask_s,
        d_out_mask=d_out_mask_s,
        d_in=int(W_s.shape[1]),
        d_out=int(W_s.shape[2]),
        d_in_pad=int(debug_info["d_in_pad"]),
        T=int(debug_info["T"]),
    )


def load_latent_flattening_flow_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    model: nn.Module,
    device: torch.device,
) -> tuple[RealNVPFlow, RealNVPConfig, dict[str, Any]]:
    resolved_checkpoint_path = Path(checkpoint_path).expanduser()
    if not resolved_checkpoint_path.is_absolute():
        resolved_checkpoint_path = env_path(_ADAPTER_CHECKPOINT_ENV, resolved_checkpoint_path)
    payload = torch.load(resolved_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Decoder adapter checkpoint must be a dict, got {type(payload)!r}: {resolved_checkpoint_path}")
    flow_state = payload.get("flow_state")
    if not isinstance(flow_state, dict):
        raise KeyError(f"Decoder adapter checkpoint does not contain dict 'flow_state': {resolved_checkpoint_path}")

    latent_dim = int(getattr(model, "flat_lat_dim"))
    flow_config = _flow_config_from_payload(payload, latent_dim=latent_dim)
    flow = RealNVPFlow(flow_config).to(device)
    flow.load_state_dict(flow_state, strict=True)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad_(False)
    return flow, flow_config, payload


class LatentFlatteningFlowDecoderAdapter:
    kind = "latent_flattening_flow"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        model: nn.Module,
        big_vae_checkpoint_path: str | Path,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = env_path(_ADAPTER_CHECKPOINT_ENV, self.checkpoint_path)
        flow, flow_config, payload = load_latent_flattening_flow_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=model,
            device=device,
        )

        self.flow = flow
        self.flow_config = flow_config
        self.adapter_big_vae_checkpoint = _payload_big_vae_checkpoint(payload)
        self.big_vae_checkpoint_path = str(big_vae_checkpoint_path)
        self.checkpoint_matches_big_vae = (
            bool(self.adapter_big_vae_checkpoint)
            and _paths_match(self.adapter_big_vae_checkpoint, self.big_vae_checkpoint_path)
        )
        if self.adapter_big_vae_checkpoint and not self.checkpoint_matches_big_vae:
            message = (
                "Latent flattening adapter was trained for a different BigVAE checkpoint: "
                f"adapter_big_vae_checkpoint={self.adapter_big_vae_checkpoint} "
                f"eval_big_vae_checkpoint={self.big_vae_checkpoint_path}"
            )
            if env_bool("EVAL_DECODER_ADAPTER_REQUIRE_CHECKPOINT_MATCH", False):
                raise ValueError(message)
            logger.warning(message)
        logger.info(
            "Decoder adapter enabled: kind=%s checkpoint=%s flow_layers=%s hidden=%s depth=%s",
            self.kind,
            self.checkpoint_path,
            int(flow_config.num_layers),
            int(flow_config.hidden_dim),
            int(flow_config.network_depth),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "checkpoint": str(self.checkpoint_path),
            "adapter_big_vae_checkpoint": str(self.adapter_big_vae_checkpoint),
            "eval_big_vae_checkpoint": str(self.big_vae_checkpoint_path),
            "checkpoint_matches_big_vae": bool(self.checkpoint_matches_big_vae),
            "flow": {
                "dim": int(self.flow_config.dim),
                "num_layers": int(self.flow_config.num_layers),
                "hidden_dim": int(self.flow_config.hidden_dim),
                "network_depth": int(self.flow_config.network_depth),
                "log_scale_clamp": float(self.flow_config.log_scale_clamp),
                "dropout": float(self.flow_config.dropout),
            },
        }

    def decode(
        self,
        *,
        model: nn.Module,
        W_s: torch.Tensor,
        x_s: torch.Tensor,
        x_mask_s: torch.Tensor,
        d_in_mask_s: torch.Tensor,
        d_out_mask_s: torch.Tensor,
    ) -> DecoderAdapterOutput:
        forward_debug = getattr(model, "forward_debug", None)
        if not callable(forward_debug):
            raise TypeError("Latent flattening decoder adapter requires BigVAE forward_debug API")

        outputs = forward_debug(
            W_s,
            x_s,
            x_mask=x_mask_s,
            d_in_mask=d_in_mask_s,
            d_out_mask=d_out_mask_s,
        )
        if len(outputs) != 5:
            raise RuntimeError(f"Expected forward_debug to return 5 tensors/debug entries, got {len(outputs)}")
        base_W_hat, mu, logvar, _base_pred_dirs, debug_info = outputs
        if not isinstance(debug_info, dict):
            raise TypeError(f"forward_debug debug_info must be a dict, got {type(debug_info)!r}")

        decoder_z = debug_info.get("latent_decoder_z")
        if not torch.is_tensor(decoder_z):
            raise KeyError("forward_debug debug_info is missing tensor 'latent_decoder_z'")
        decoder_z_flat = decoder_z.reshape(int(decoder_z.shape[0]), -1) if decoder_z.ndim == 3 else decoder_z
        adapter_z, forward_log_det = self.flow(decoder_z_flat.to(dtype=torch.float32))
        W_hat, adapter_decoder_z, _adapter_logvar, pred_dirs, inverse_log_det = self.decode_from_adapter_latent(
            model=model,
            adapter_z=adapter_z,
            debug_info=debug_info,
            W_s=W_s,
            d_in_mask_s=d_in_mask_s,
            d_out_mask_s=d_out_mask_s,
        )
        metrics = {
            "decoder_adapter_base_recon_mse": float(
                _masked_weight_mse(base_W_hat, W_s, d_in_mask=d_in_mask_s, d_out_mask=d_out_mask_s).detach().cpu().item()
            ),
            "decoder_adapter_recon_mse": float(
                _masked_weight_mse(W_hat, W_s, d_in_mask=d_in_mask_s, d_out_mask=d_out_mask_s).detach().cpu().item()
            ),
            "decoder_adapter_decode_delta_mse": float(
                _masked_weight_mse(W_hat, base_W_hat, d_in_mask=d_in_mask_s, d_out_mask=d_out_mask_s)
                .detach()
                .cpu()
                .item()
            ),
            "decoder_adapter_latent_delta_mse": float(
                (adapter_z - decoder_z_flat.to(dtype=adapter_z.dtype)).pow(2).mean().detach().cpu().item()
            ),
            "decoder_adapter_cycle_mse": float(
                (adapter_decoder_z - decoder_z_flat.to(dtype=adapter_decoder_z.dtype)).pow(2).mean().detach().cpu().item()
            ),
            "decoder_adapter_forward_logdet_mean": float(forward_log_det.detach().mean().cpu().item()),
            "decoder_adapter_inverse_logdet_mean": float(inverse_log_det.detach().mean().cpu().item()),
        }
        return DecoderAdapterOutput(W_hat=W_hat, mu=mu, logvar=logvar, pred_dirs=pred_dirs, metrics=metrics)

    def decode_from_adapter_latent(
        self,
        *,
        model: nn.Module,
        adapter_z: torch.Tensor,
        debug_info: dict[str, Any],
        W_s: torch.Tensor,
        d_in_mask_s: torch.Tensor,
        d_out_mask_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder_z, inverse_log_det = self.flow.inverse(adapter_z.to(dtype=torch.float32))
        W_hat, _decoder_z_out, logvar, pred_dirs = _decode_big_vae_decoder_latent(
            model=model,
            decoder_z=decoder_z.to(device=adapter_z.device, dtype=adapter_z.dtype),
            debug_info=debug_info,
            W_s=W_s,
            d_in_mask_s=d_in_mask_s,
            d_out_mask_s=d_out_mask_s,
        )
        return W_hat, decoder_z, logvar, pred_dirs, inverse_log_det


def build_decoder_adapter_from_env(
    *,
    model: nn.Module,
    big_vae_checkpoint_path: str | Path,
    device: torch.device,
    logger: logging.Logger,
) -> IdentityDecoderAdapter | LatentFlatteningFlowDecoderAdapter:
    raw_kind = str(os.environ.get("EVAL_DECODER_ADAPTER", "identity")).strip().lower()
    raw_checkpoint = str(
        os.environ.get(_ADAPTER_CHECKPOINT_ENV, os.environ.get(_LEGACY_FLOW_CHECKPOINT_ENV, ""))
    ).strip()
    if raw_kind in {"", "none", "off", "false", "identity"} and raw_checkpoint:
        raw_kind = "latent_flattening_flow"
    if raw_kind in {"", "none", "off", "false", "identity"}:
        return IdentityDecoderAdapter()
    if raw_kind not in {"latent_flattening_flow", "flow", "ir_smoothing", "latent_smoothing"}:
        raise ValueError(
            "Unsupported EVAL_DECODER_ADAPTER="
            f"{raw_kind!r}; expected identity or latent_flattening_flow"
        )
    if not raw_checkpoint:
        raise ValueError(
            "Set EVAL_DECODER_ADAPTER_CHECKPOINT=/path/to/latent_flattening/checkpoints/latest.pt "
            "when EVAL_DECODER_ADAPTER=latent_flattening_flow"
        )
    return LatentFlatteningFlowDecoderAdapter(
        checkpoint_path=raw_checkpoint,
        model=model,
        big_vae_checkpoint_path=big_vae_checkpoint_path,
        device=device,
        logger=logger,
    )
