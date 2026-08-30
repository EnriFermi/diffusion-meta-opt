from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from scripts import evaluate_celo_psgd_kron_positive_control as harness


def test_candidate_grid_is_exact_cartesian_product() -> None:
    grid = harness._candidate_grid(
        methods=["raw_adam", "raw_sgd_momentum", "kron_whiten_momentum"],
        raw_adam_lrs=[1e-4, 3e-4],
        raw_sgd_momentum_lrs=[1e-3],
        kron_lrs=[1e-4, 3e-4],
        kron_update_probabilities=[0.03, 0.1],
        include_kron_package_default=False,
    )

    assert len(grid) == 7
    assert grid["candidate_id"].nunique() == 7
    kron = grid[grid["method"] == "kron_whiten_momentum"]
    assert set(zip(kron["lr"], kron["kron_precond_update_probability"], strict=True)) == {
        (1e-4, 0.03),
        (1e-4, 0.1),
        (3e-4, 0.03),
        (3e-4, 0.1),
    }


def test_candidate_grid_distinguishes_package_default_and_front_loaded_constant() -> None:
    grid = harness._candidate_grid(
        methods=["kron_whiten_momentum"],
        raw_adam_lrs=[],
        raw_sgd_momentum_lrs=[],
        kron_lrs=[3e-4],
        kron_update_probabilities=[0.03, 0.1, 1.0],
    )

    assert len(grid) == 4
    assert grid["candidate_id"].nunique() == 4
    default = grid[grid["kron_update_schedule"] == harness.KRON_PACKAGE_DEFAULT_SCHEDULE].iloc[0]
    front_loaded = grid[grid["kron_update_schedule"] == "constant_1"].iloc[0]
    assert pd.isna(default["kron_precond_update_probability"])
    assert float(front_loaded["kron_precond_update_probability"]) == 1.0
    assert default["candidate_id"] != front_loaded["candidate_id"]


def test_protocol_start_partition_is_disjoint_and_resets_split_indices() -> None:
    bank = pd.DataFrame(
        {
            "start_bank_position": range(5),
            "source_weight_index": [10, 11, 12, 13, 14],
            "start_role": ["input"] * 5,
        }
    )

    tune, evaluation = harness._split_protocol_start_bank(bank, tune_starts=2, eval_starts=3)

    assert tune["source_weight_index"].tolist() == [10, 11]
    assert evaluation["source_weight_index"].tolist() == [12, 13, 14]
    assert tune["split_start_index"].tolist() == [0, 1]
    assert evaluation["split_start_index"].tolist() == [0, 1, 2]
    assert tune["global_stream_index"].tolist() == [0, 1]
    assert evaluation["global_stream_index"].tolist() == [2, 3, 4]
    assert set(tune["start_role"]) == {"positive_control_tune"}
    assert set(evaluation["start_role"]) == {"positive_control_eval"}


def _tuning_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = harness._candidate_grid(
        methods=["raw_adam", "raw_sgd_momentum"],
        raw_adam_lrs=[1e-4, 3e-4, 1e-3],
        raw_sgd_momentum_lrs=[1e-3, 3e-3, 1e-2],
        kron_lrs=[],
        kron_update_probabilities=[],
    )
    tune_bank = pd.DataFrame({"source_weight_index": [10, 11]})
    metric_by_method_lr = {
        ("raw_adam", 1e-4): 0.4,
        ("raw_adam", 3e-4): 0.2,
        ("raw_adam", 1e-3): 0.5,
        ("raw_sgd_momentum", 1e-3): 0.5,
        ("raw_sgd_momentum", 3e-3): 0.3,
        ("raw_sgd_momentum", 1e-2): 0.6,
    }
    rows = []
    for candidate in candidates.to_dict(orient="records"):
        for source in tune_bank["source_weight_index"]:
            rows.append(
                {
                    "protocol_split": "tune",
                    "candidate_id": candidate["candidate_id"],
                    **{column: candidate[column] for column in harness.CANDIDATE_CONFIG_COLUMNS},
                    "source_weight_index": int(source),
                    harness.PRIMARY_ENDPOINT: metric_by_method_lr[
                        (str(candidate["method"]), float(candidate["lr"]))
                    ]
                    + 0.001 * int(source),
                    "diverged": False,
                }
            )
    return candidates, tune_bank, pd.DataFrame(rows)


def test_selection_uses_only_exact_tune_grid_and_freezes_one_per_method() -> None:
    candidates, tune_bank, tuning = _tuning_fixture()

    selected = harness._select_frozen_candidates(
        tuning,
        candidates=candidates,
        tune_start_bank=tune_bank,
        finite_penalty=1e12,
    )
    frozen = selected[selected["selected"] == 1].set_index("method")

    assert len(frozen) == 2
    assert float(frozen.loc["raw_adam", "lr"]) == pytest.approx(3e-4)
    assert float(frozen.loc["raw_sgd_momentum", "lr"]) == pytest.approx(3e-3)
    assert set(frozen["primary_endpoint"]) == {harness.PRIMARY_ENDPOINT}
    assert frozen["lr_boundary_adequate"].astype(bool).all()


def test_fixed_cli_lr_is_labeled_without_selection_claim() -> None:
    candidates = harness._candidate_grid(
        methods=["raw_adam"],
        raw_adam_lrs=[3e-4],
        raw_sgd_momentum_lrs=[],
        kron_lrs=[],
        kron_update_probabilities=[],
        raw_adam_lr_fixed=True,
    )
    tune_bank = pd.DataFrame({"source_weight_index": [10, 11]})
    candidate = candidates.iloc[0].to_dict()
    tuning = pd.DataFrame(
        [
            {
                "protocol_split": "tune",
                "candidate_id": candidate["candidate_id"],
                **{column: candidate[column] for column in harness.CANDIDATE_CONFIG_COLUMNS},
                "source_weight_index": source,
                harness.PRIMARY_ENDPOINT: 0.2,
                "diverged": False,
            }
            for source in tune_bank["source_weight_index"]
        ]
    )

    selected = harness._select_frozen_candidates(
        tuning,
        candidates=candidates,
        tune_start_bank=tune_bank,
        finite_penalty=1e12,
    ).iloc[0]

    assert selected["lr_selection_mode"] == "fixed_cli"
    assert bool(selected["lr_boundary_adequate"])
    assert selected["lr_boundary_status"] == "fixed_cli_no_lr_selection_claim"


def test_selection_rejects_split_leakage_and_missing_grid_rows() -> None:
    candidates, tune_bank, tuning = _tuning_fixture()
    leaked = tuning.copy()
    leaked.loc[0, "protocol_split"] = "eval"
    with pytest.raises(ValueError, match="split leakage"):
        harness._select_frozen_candidates(
            leaked,
            candidates=candidates,
            tune_start_bank=tune_bank,
            finite_penalty=1e12,
        )

    with pytest.raises(ValueError, match="not exact"):
        harness._select_frozen_candidates(
            tuning.iloc[:-1].copy(),
            candidates=candidates,
            tune_start_bank=tune_bank,
            finite_penalty=1e12,
        )

    wrong_config = tuning.copy()
    wrong_config.loc[0, "lr"] = 9e-4
    with pytest.raises(ValueError, match="do not match frozen candidate configs"):
        harness._select_frozen_candidates(
            wrong_config,
            candidates=candidates,
            tune_start_bank=tune_bank,
            finite_penalty=1e12,
        )


def test_curve_cartesian_keys_context_and_primary_recomputation() -> None:
    candidates = harness._candidate_grid(
        methods=["raw_adam"],
        raw_adam_lrs=[3e-4],
        raw_sgd_momentum_lrs=[],
        kron_lrs=[],
        kron_update_probabilities=[],
        raw_adam_lr_fixed=True,
    )
    bank = pd.DataFrame(
        {
            "source_weight_index": [10, 11],
            "global_stream_index": [0, 1],
            "start_bank_position": [0, 1],
        }
    )
    rows = []
    results = []
    candidate = candidates.iloc[0].to_dict()
    for source, stream in [(10, 0), (11, 1)]:
        losses = [1.0 + stream, 0.5 + stream, 0.25 + stream]
        for step, loss in zip([0, 2, 4], losses, strict=True):
            rows.append(
                {
                    "protocol_split": "tune",
                    "candidate_id": candidate["candidate_id"],
                    **{column: candidate[column] for column in harness.CANDIDATE_CONFIG_COLUMNS},
                    "source_weight_index": source,
                    "global_stream_index": stream,
                    "start_bank_position": stream,
                    "step": step,
                    "train_loss": loss,
                }
            )
        results.append(
            {
                "protocol_split": "tune",
                "candidate_id": candidate["candidate_id"],
                "source_weight_index": source,
                harness.PRIMARY_ENDPOINT: sum(losses[1:]) / 2,
            }
        )
    curves = pd.DataFrame(rows)

    validation = harness._validate_artifact_grid(
        curves,
        candidates=candidates,
        start_bank=bank,
        protocol_split="tune",
        artifact_name="curves",
        step_values=[0, 2, 4],
    )
    endpoint = harness._recompute_post0_train_aulc(
        curves=curves,
        results=pd.DataFrame(results),
        finite_penalty=1e12,
    )

    assert validation["accepted"] is True
    assert endpoint["accepted"] is True
    with pytest.raises(ValueError, match="Cartesian keys are not exact"):
        harness._validate_artifact_grid(
            pd.concat([curves, curves.iloc[[0]]], ignore_index=True),
            candidates=candidates,
            start_bank=bank,
            protocol_split="tune",
            artifact_name="curves",
            step_values=[0, 2, 4],
        )
    bad_results = pd.DataFrame(results)
    bad_results.loc[0, harness.PRIMARY_ENDPOINT] += 0.1
    assert not harness._recompute_post0_train_aulc(
        curves=curves,
        results=bad_results,
        finite_penalty=1e12,
    )["accepted"]


class _GoodFakeKron(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=3e-4,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        memory_save_mode=None,
        precond_lr=0.1,
    ):
        del b1, weight_decay, memory_save_mode, precond_lr
        if preconditioner_update_probability is None:

            def schedule(step):
                return torch.tensor(
                    harness._package_default_probability(int(step.detach().cpu().item())),
                    dtype=torch.float32,
                )

        else:
            schedule = float(preconditioner_update_probability)
        super().__init__(params, {"lr": float(lr), "preconditioner_update_probability": schedule})
        self._prob_step = torch.tensor(0, dtype=torch.int32)

    @torch.no_grad()
    def step(self, closure=None):
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    self.state[parameter]["Q"] = [torch.ones_like(parameter)]
                    parameter.add_(parameter.grad, alpha=-float(group["lr"]))
        self._prob_step += 1


class _BadFakeKron(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4):
        super().__init__(params, {"lr": float(lr)})


class _NoPreconditionerStateFakeKron(_GoodFakeKron):
    @torch.no_grad()
    def step(self, closure=None):
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-float(group["lr"]))


def test_kron_dependency_gate_checks_exact_version_api_and_cpu_step() -> None:
    fake_module = SimpleNamespace(Kron=_GoodFakeKron, __name__="kron_torch", __file__="/fake/kron_torch.py")
    real_import_module = harness.importlib.import_module

    def import_only_fake_kron(name: str, *args, **kwargs):
        if name == "kron_torch":
            return fake_module
        return real_import_module(name, *args, **kwargs)

    with (
        patch.object(harness.importlib.metadata, "version", return_value=harness.KRON_TORCH_REQUIRED_VERSION),
        patch.object(harness.importlib, "import_module", side_effect=import_only_fake_kron),
    ):
        result = harness._validate_kron_dependency(run_celo_integration_smoke=False)

    assert result["installed_version"] == harness.KRON_TORCH_REQUIRED_VERSION
    assert result["cpu_smoke_step_completed"] is True
    assert result["cpu_smoke_preconditioner_numel"] == 4
    assert result["cpu_smoke_prob_step"] == 1
    assert result["cpu_smoke_q_hash_changed"] is True
    assert result["cpu_smoke_q_sha256_before"] == ""
    assert len(result["cpu_smoke_q_sha256_after"]) == 64
    assert "preconditioner_update_probability" in result["kron_signature"]


def test_kron_dependency_gate_rejects_wrong_version_and_api() -> None:
    fake_module = SimpleNamespace(Kron=_BadFakeKron, __name__="kron_torch", __file__="/fake/kron_torch.py")
    real_import_module = harness.importlib.import_module

    def import_only_fake_kron(name: str, *args, **kwargs):
        if name == "kron_torch":
            return fake_module
        return real_import_module(name, *args, **kwargs)

    with patch.object(harness.importlib.metadata, "version", return_value="0.3.2"):
        with pytest.raises(RuntimeError, match="required exact version"):
            harness._validate_kron_dependency()

    with (
        patch.object(harness.importlib.metadata, "version", return_value=harness.KRON_TORCH_REQUIRED_VERSION),
        patch.object(harness.importlib, "import_module", side_effect=import_only_fake_kron),
    ):
        with pytest.raises(RuntimeError, match="API mismatch"):
            harness._validate_kron_dependency(
                run_cpu_smoke_step=False,
                run_celo_integration_smoke=False,
            )

    no_state_module = SimpleNamespace(
        Kron=_NoPreconditionerStateFakeKron,
        __name__="kron_torch",
        __file__="/fake/kron_torch.py",
    )

    def import_no_state_kron(name: str, *args, **kwargs):
        if name == "kron_torch":
            return no_state_module
        return real_import_module(name, *args, **kwargs)

    with (
        patch.object(harness.importlib.metadata, "version", return_value=harness.KRON_TORCH_REQUIRED_VERSION),
        patch.object(harness.importlib, "import_module", side_effect=import_no_state_kron),
    ):
        with pytest.raises(RuntimeError, match="did not expose an initialized Q state"):
            harness._validate_kron_dependency(run_celo_integration_smoke=False)


def test_expected_kron_update_trace_numeric_and_default_schedule() -> None:
    p1 = harness._expected_kron_update_trace(schedule="constant_1", probability=1.0, steps=5)
    p01 = harness._expected_kron_update_trace(schedule="constant_0.1", probability=0.1, steps=20)
    p003 = harness._expected_kron_update_trace(schedule="constant_0.03", probability=0.03, steps=68)
    default = harness._expected_kron_update_trace(
        schedule=harness.KRON_PACKAGE_DEFAULT_SCHEDULE,
        probability=None,
        steps=3,
    )

    assert sum(row["kron_update_event"] for row in p1) == 5
    assert sum(row["kron_update_event"] for row in p01) == 2
    assert sum(row["kron_update_event"] for row in p003) == 2
    assert sum(row["kron_update_event"] for row in default) == 3
    assert harness._package_default_probability(0) == 1.0
    assert harness._package_default_probability(4000) == pytest.approx(0.030197376385331154)
    assert harness._package_default_probability(4008) == pytest.approx(0.03)


def test_kron_trajectory_gate_checks_every_trajectory() -> None:
    candidates = harness._candidate_grid(
        methods=["kron_whiten_momentum"],
        raw_adam_lrs=[],
        raw_sgd_momentum_lrs=[],
        kron_lrs=[3e-4],
        kron_update_probabilities=[0.1],
        include_kron_package_default=False,
        kron_lr_fixed=True,
        kron_schedule_fixed=True,
    )
    candidate = candidates.iloc[0]
    trace = harness._expected_kron_update_trace(schedule="constant_0.1", probability=0.1, steps=10)
    rows = []
    for source in [10, 11]:
        q_after = "a" * 64
        for expected in trace:
            q_before = "" if expected["step"] == 0 else q_after
            if expected["kron_q_update_event"] and expected["step"] > 0:
                q_after = "b" * 64
            q_changed = q_before != q_after
            rows.append(
                {
                    "protocol_split": "eval",
                    "candidate_id": candidate["candidate_id"],
                    "source_weight_index": source,
                    "method": "kron_whiten_momentum",
                    **expected,
                    "kron_expected_update_event": expected["kron_update_event"],
                    "kron_expected_cumulative_update_count": expected["kron_cumulative_update_count"],
                    "kron_counter_transition_matches": True,
                    "kron_probability_matches": True,
                    "kron_q_finite": True,
                    "kron_preconditioner_tensors": 6,
                    "kron_q_factor_shapes": "[[32,32],[64,64],[32],[10,10],[32,32],[10]]",
                    "kron_q_sha256": q_after,
                    "kron_q_sha256_before": q_before,
                    "kron_q_sha256_after": q_after,
                    "kron_q_hash_changed": q_changed,
                    "kron_expected_q_update_event": expected["kron_q_update_event"],
                    "kron_q_hash_change_matches_expected": q_changed == expected["kron_q_update_event"],
                }
            )
    diagnostics = pd.DataFrame(rows)

    assert harness._validate_kron_trajectories(
        diagnostics,
        candidates=candidates,
        downstream_steps=10,
    )["accepted"]
    diagnostics.loc[
        (diagnostics["source_weight_index"] == 11) & (diagnostics["step"] == 5),
        "kron_q_finite",
    ] = False
    rejected = harness._validate_kron_trajectories(
        diagnostics,
        candidates=candidates,
        downstream_steps=10,
    )
    assert not rejected["accepted"]
    assert rejected["trajectories"] == 2

    constant_q = diagnostics.copy()
    event_row = (constant_q["source_weight_index"] == 11) & (constant_q["step"] == 9)
    constant_q.loc[event_row, "kron_q_sha256_after"] = constant_q.loc[event_row, "kron_q_sha256_before"]
    constant_q.loc[event_row, "kron_q_sha256"] = constant_q.loc[event_row, "kron_q_sha256_before"]
    constant_q.loc[event_row, "kron_q_hash_changed"] = False
    constant_q.loc[event_row, "kron_q_hash_change_matches_expected"] = False
    constant_q_result = harness._validate_kron_trajectories(
        constant_q,
        candidates=candidates,
        downstream_steps=10,
    )
    assert not constant_q_result["accepted"]
    assert any(
        "q_hash_mutation_does_not_match_expected_update_events" in failure["reasons"]
        for failure in constant_q_result["failures"]
    )


def test_real_kron_package_celo_shaped_cpu_integration() -> None:
    result = harness._validate_kron_dependency()
    integration = result["celo_shaped_cpu_integration"]

    assert result["installed_version"] == "0.3.3"
    assert integration["status"] == "accepted"
    assert integration["model_kind"] == "celo_meta_mlp"
    assert integration["parameter_dim"] == 2410
    assert integration["arms"][harness.KRON_PACKAGE_DEFAULT_SCHEDULE]["update_count"] == 3
    constant = integration["arms"]["constant_0.1"]
    assert constant["update_count"] == 1
    constant_1 = integration["arms"]["constant_1"]
    assert constant_1["update_count"] == 3
    assert constant_1["q_hash_change_count"] == 3
    assert constant_1["q_hash_changed_on_expected_events"] is True
    assert constant["q_finite_every_step"] is True
    assert constant["q_factor_count"] == 6
    assert len(constant["q_sha256"]) == 64
