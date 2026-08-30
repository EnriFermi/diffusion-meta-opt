from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from post_train_research.big_vae_heldout_eval.evaluate_parts.decoder_adapter import (
    LatentFlatteningFlowDecoderAdapter,
    build_decoder_adapter_from_env,
)
from post_train_research.big_vae_latent_flattening.flow import RealNVPConfig, RealNVPFlow


class _DummyDecoderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flat_lat_dim = 4

    def forward_debug(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        *,
        x_mask: torch.Tensor,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        del X, x_mask, d_in_mask, d_out_mask
        batch_size = int(W.shape[0])
        device = W.device
        latent = torch.arange(batch_size * 4, device=device, dtype=torch.float32).view(batch_size, 4)
        debug_info = {
            "latent_decoder_z": latent,
            "dist_patch_by_patch": None,
            "patch_mask": torch.ones((batch_size, 2), device=device, dtype=torch.bool),
            "d_in_pad": 2,
            "T": 2,
        }
        base_W_hat = latent.view(batch_size, 2, 2)
        return (
            base_W_hat,
            torch.zeros((batch_size, 4), device=device),
            torch.zeros((batch_size, 4), device=device),
            torch.zeros((batch_size, 2, 2, 1), device=device),
            debug_info,
        )

    def _decode_from_decoder_latent(
        self,
        decoder_z: torch.Tensor,
        *,
        dist_patch_by_patch: torch.Tensor | None,
        patch_mask: torch.Tensor,
        d_in_mask: torch.Tensor,
        d_out_mask: torch.Tensor,
        d_in: int,
        d_out: int,
        d_in_pad: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del dist_patch_by_patch, patch_mask, d_in_mask, d_out_mask, d_in_pad, T
        batch_size = int(decoder_z.shape[0])
        W_hat = decoder_z.view(batch_size, int(d_in), int(d_out))
        pred_dirs = torch.zeros((batch_size, int(d_out), 2, 1), device=decoder_z.device)
        return W_hat, decoder_z, torch.zeros_like(decoder_z), pred_dirs


def _write_flow_checkpoint(path: Path, *, big_vae_checkpoint: Path) -> None:
    flow = RealNVPFlow(RealNVPConfig(dim=4, num_layers=1, hidden_dim=8, network_depth=1, log_scale_clamp=1.0))
    with torch.no_grad():
        flow.layers[0].net[-1].bias[1] = 2.0
    torch.save(
        {
            "config": {
                "flow": {
                    "num_layers": 1,
                    "hidden_dim": 8,
                    "network_depth": 1,
                    "log_scale_clamp": 1.0,
                    "dropout": 0.0,
                },
                "big_vae": {"checkpoint": str(big_vae_checkpoint)},
            },
            "flow_state": flow.state_dict(),
            "big_vae_checkpoint": str(big_vae_checkpoint),
        },
        path,
    )


def test_latent_flattening_decoder_adapter_loads_flow_and_reports_metrics(tmp_path: Path) -> None:
    big_vae_checkpoint = tmp_path / "big_vae.pt"
    flow_checkpoint = tmp_path / "flow.pt"
    big_vae_checkpoint.write_bytes(b"placeholder")
    _write_flow_checkpoint(flow_checkpoint, big_vae_checkpoint=big_vae_checkpoint)
    model = _DummyDecoderModel()

    adapter = LatentFlatteningFlowDecoderAdapter(
        checkpoint_path=flow_checkpoint,
        model=model,
        big_vae_checkpoint_path=big_vae_checkpoint,
        device=torch.device("cpu"),
        logger=__import__("logging").getLogger("test"),
    )
    output = adapter.decode(
        model=model,
        W_s=torch.zeros((1, 2, 2)),
        x_s=torch.zeros((1, 3, 2)),
        x_mask_s=torch.ones((1, 3), dtype=torch.bool),
        d_in_mask_s=torch.ones((1, 2), dtype=torch.bool),
        d_out_mask_s=torch.ones((1, 2), dtype=torch.bool),
    )

    assert adapter.metadata()["checkpoint_matches_big_vae"] is True
    assert output.W_hat.tolist() == [[[0.0, 1.0], [2.0, 3.0]]]
    assert output.metrics["decoder_adapter_cycle_mse"] == pytest.approx(0.0)
    assert output.metrics["decoder_adapter_latent_delta_mse"] > 0.0
    assert output.metrics["decoder_adapter_decode_delta_mse"] == pytest.approx(0.0)
    assert output.metrics["decoder_adapter_base_recon_mse"] == pytest.approx(3.5)
    assert output.metrics["decoder_adapter_recon_mse"] == pytest.approx(3.5)


def test_decoder_adapter_auto_enables_flow_when_checkpoint_env_is_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_vae_checkpoint = tmp_path / "big_vae.pt"
    flow_checkpoint = tmp_path / "flow.pt"
    big_vae_checkpoint.write_bytes(b"placeholder")
    _write_flow_checkpoint(flow_checkpoint, big_vae_checkpoint=big_vae_checkpoint)
    monkeypatch.setenv("EVAL_DECODER_ADAPTER_CHECKPOINT", str(flow_checkpoint))

    adapter = build_decoder_adapter_from_env(
        model=_DummyDecoderModel(),
        big_vae_checkpoint_path=big_vae_checkpoint,
        device=torch.device("cpu"),
        logger=__import__("logging").getLogger("test"),
    )

    assert isinstance(adapter, LatentFlatteningFlowDecoderAdapter)
