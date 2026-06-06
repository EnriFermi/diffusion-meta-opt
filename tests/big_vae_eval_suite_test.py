from __future__ import annotations

from pathlib import Path

import pytest

from post_train_research.big_vae_eval_suite import main as eval_suite_main
from post_train_research.big_vae_eval_suite.main import EvalSuite


def _suite_cfg(tmp_path: Path, *, dry_run: bool) -> dict:
    return {
        "suite": {
            "root_dir": str(tmp_path),
            "run_label": "smoke",
            "fail_fast": True,
            "dry_run": dry_run,
            "python": "",
            "conda_env": "",
            "conda_executable": "conda",
        },
        "checkpoint": {
            "big_vae": str(tmp_path / "future_big_vae.pt"),
            "latent_diffusion_prior": str(tmp_path / "future_prior.pt"),
            "label": "future_ckpt",
        },
        "stages": {
            "heldout_eval": {
                "enabled": False,
            },
            "scaling_check": {
                "enabled": True,
                "python": "",
                "conda_env": "",
                "run_latent": True,
                "run_raw_baseline": False,
                "common": {
                    "notes": "notes with spaces",
                    "init": {
                        "kind": "diffusion_prior",
                    },
                },
                "jobs": [
                    {
                        "label": "smoke",
                        "profile": "cifar10_50k",
                        "max_steps": 1,
                    }
                ],
            },
            "landscape_ablation": {
                "enabled": False,
            },
        },
    }


def test_eval_suite_dry_run_allows_future_checkpoint_paths_and_quotes_commands(tmp_path: Path) -> None:
    cfg = _suite_cfg(tmp_path, dry_run=True)

    suite = EvalSuite(cfg)
    suite.run()

    scaling_result = next(result for result in suite.results if result.name == "scaling_smoke_latent")
    log_text = Path(scaling_result.log_path).read_text(encoding="utf-8")
    assert "DRY RUN" in log_text
    assert "'experiment.notes=notes with spaces'" in log_text
    assert str(tmp_path / "future_big_vae.pt") in log_text
    assert str(tmp_path / "future_prior.pt") in log_text


def test_eval_suite_dry_run_logs_heldout_env(tmp_path: Path) -> None:
    cfg = _suite_cfg(tmp_path, dry_run=True)
    cfg["stages"]["heldout_eval"] = {
        "enabled": True,
        "heldout_root": str(tmp_path / "future_heldout"),
    }
    cfg["stages"]["scaling_check"]["enabled"] = False

    suite = EvalSuite(cfg)
    suite.run()

    heldout_result = next(result for result in suite.results if result.name == "heldout_eval")
    log_text = Path(heldout_result.log_path).read_text(encoding="utf-8")
    assert "BIG_VAE_CHECKPOINT=" in log_text
    assert "HELDOUT_ROOT=" in log_text
    assert "EVAL_OUTPUT_DIR=" in log_text
    assert heldout_result.env["BIG_VAE_CHECKPOINT"] == str(tmp_path / "future_big_vae.pt")


def test_eval_suite_validates_missing_prior_before_real_run(tmp_path: Path) -> None:
    cfg = _suite_cfg(tmp_path, dry_run=False)
    big_vae_path = tmp_path / "future_big_vae.pt"
    big_vae_path.write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError, match="checkpoint.latent_diffusion_prior"):
        EvalSuite(cfg)


def test_eval_suite_stage_conda_env_builds_conda_command_in_dry_run(tmp_path: Path) -> None:
    cfg = _suite_cfg(tmp_path, dry_run=True)
    cfg["suite"]["python"] = "/usr/bin/python3"
    cfg["stages"]["scaling_check"]["conda_env"] = "stage-env"

    suite = EvalSuite(cfg)
    suite.run()

    scaling_result = next(result for result in suite.results if result.name == "scaling_smoke_latent")
    assert scaling_result.command[:6] == ["conda", "run", "--no-capture-output", "-n", "stage-env", "python"]


def test_eval_suite_run_id_collision_gets_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _suite_cfg(tmp_path, dry_run=True)
    monkeypatch.setattr(eval_suite_main.time, "strftime", lambda _format: "20260101_000000")

    first = EvalSuite(cfg)
    second = EvalSuite(cfg)

    assert first.run_id == "20260101_000000__smoke"
    assert second.run_id == "20260101_000000__smoke_01"
