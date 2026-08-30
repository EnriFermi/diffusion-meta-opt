from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from post_train_research.loss_landscape_analysis.sage_cnn_vae_smoothing.config import (
    ExperimentConfig,
    config_hash,
)
from scripts.audit_variant_a_fixed_state_probe_variance import (
    BASE_SEED,
    DEFAULT_RUN_DIR,
    DRAW_COUNT,
    EXCLUDED_PRIMARY_PATH,
    EXPECTED_CONFIG_HASH,
    EXPECTED_RUNTIME_CONFIG_HASH,
    PROTOCOL_ID,
    STATE_BANK_PATH,
    _cleanup_owned_scratch,
    _create_scratch,
    _read_bank,
    _resampled_components_from_task_gram,
    balanced_crossed_anova,
    balanced_crossed_anova_from_task_grams,
    common_probe_seed,
    chunked_many_memmap_gram,
    delta_from_crossed_gram,
    corrected_panel_contrast_energy_from_task_grams,
    fixed_panel_probe_bootstrap,
    primary_mechanism_decision,
    reconstructed_update_gram,
    signal_fraction_from_gram,
    stable_uint63,
    summarize_state_gram,
    unit_gram,
    validate_state_bank,
)


def _gram(vectors: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float64) @ np.asarray(vectors, dtype=np.float64).T


def test_common_probe_seed_excludes_state_and_has_exact_frozen_key() -> None:
    expected = stable_uint63(PROTOCOL_ID, BASE_SEED, "probe", 7, 3, 1)
    assert common_probe_seed(7, 3, 1) == expected
    seeds = {
        common_probe_seed(draw, pair, branch)
        for draw in range(DRAW_COUNT)
        for pair in range(4)
        for branch in (0, 1)
    }
    assert len(seeds) == DRAW_COUNT * 4 * 2


def test_saved_and_current_default_config_hashes_are_separately_pinned() -> None:
    payload = json.loads((DEFAULT_RUN_DIR / "config.json").read_text(encoding="utf-8"))
    assert payload["config_hash"] == EXPECTED_CONFIG_HASH
    assert config_hash(ExperimentConfig(**payload["config"])) == EXPECTED_RUNTIME_CONFIG_HASH
    assert EXPECTED_CONFIG_HASH != EXPECTED_RUNTIME_CONFIG_HASH


def test_crossed_anova_closes_and_keeps_corrected_state_variance_untruncated() -> None:
    # Two fixed tasks, two states per task, and three common probes. The state-by-probe
    # interaction is deliberately large enough to make the corrected state component negative.
    task_labels = ["fashion_mnist", "fashion_mnist", "mnist", "mnist"]
    state_effect = np.asarray([-0.2, 0.2, -0.2, 0.2])
    task_effect = np.asarray([-1.0, -1.0, 1.0, 1.0])
    probe_effect = np.asarray([-2.0, 0.5, 1.5])
    interaction = np.asarray(
        [
            [-3.0, 0.0, 3.0],
            [3.0, 0.0, -3.0],
            [-4.0, 1.0, 3.0],
            [4.0, -1.0, -3.0],
        ]
    )
    vectors = []
    for state in range(4):
        for probe in range(3):
            vectors.append([task_effect[state] + state_effect[state] + probe_effect[probe] + interaction[state, probe]])
    gram = _gram(np.asarray(vectors))

    anova = balanced_crossed_anova(gram, task_labels, q=3).set_index("component")
    centered_total = float(np.trace(gram) - gram.sum() / len(gram))
    effect_sum = float(
        anova.loc[
            ["task", "state(task)", "probe", "task*probe", "state(task)*probe"],
            "sum_squared_vector_norms",
        ].sum()
    )
    assert np.isclose(effect_sum, centered_total, rtol=0.0, atol=1e-10)
    assert anova.loc["state(task)", "degrees_freedom"] == 2
    assert anova.loc["state(task)*probe", "degrees_freedom"] == 4

    estimate = delta_from_crossed_gram(gram, task_labels, q=3)
    assert estimate.v_state < 0.0
    assert estimate.v_probe > 0.0
    assert np.isclose(estimate.delta, estimate.v_probe - estimate.v_state)


def test_task_gram_anova_sufficient_statistics_match_dense_full_gram() -> None:
    generator = np.random.default_rng(123)
    vectors = generator.normal(size=(4, 3, 5))
    flat = vectors.reshape(12, 5)
    full = _gram(flat)
    dense = balanced_crossed_anova(
        full,
        ["fashion_mnist", "fashion_mnist", "mnist", "mnist"],
        q=3,
    ).set_index("component")
    fashion = _gram(vectors[:2].reshape(6, 5))
    mnist = _gram(vectors[2:].reshape(6, 5))
    fashion_probe_sum = vectors[:2].sum(axis=0)
    mnist_probe_sum = vectors[2:].sum(axis=0)
    compact = balanced_crossed_anova_from_task_grams(
        {"fashion_mnist": fashion, "mnist": mnist},
        fashion_probe_sum @ mnist_probe_sum.T,
        q=3,
    ).set_index("component")
    np.testing.assert_allclose(
        compact["sum_squared_vector_norms"],
        dense["sum_squared_vector_norms"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compact["random_effect_trace_estimate"],
        dense["random_effect_trace_estimate"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_gram_metrics_signal_splits_and_reconstruction_are_exact() -> None:
    base = np.tile(
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        (16, 1),
    )
    gram = _gram(base)
    summary = summarize_state_gram(gram, state_position=0, prefix=4)
    assert summary["split_gate_pass"] is True
    assert np.isclose(summary["mean_gradient_norm"], np.sqrt(0.5))
    assert np.isclose(signal_fraction_from_gram(gram), 31.0 / 63.0)
    normalized = unit_gram(gram)
    np.testing.assert_allclose(np.diag(normalized), 1.0)

    right = 2.0 * base
    right_gram = _gram(right)
    cross = base @ right.T
    common = reconstructed_update_gram(gram, right_gram, cross, independent=False)
    expected = _gram(1.5 * base)
    np.testing.assert_allclose(common, expected, rtol=0.0, atol=1e-12)


def test_conditional_mean_gate_uses_unit_directions_not_raw_norm_weighting() -> None:
    vectors = np.zeros((DRAW_COUNT, 2), dtype=np.float64)
    vectors[: DRAW_COUNT // 2, 1] = 1.0
    vectors[DRAW_COUNT // 2 :, 1] = -1.0
    # Every frozen split side contains a huge +x draw, so raw split means align.
    # Unit means preserve the opposing majority directions in the first split.
    vectors[[0, 1, 16, 32, 33, 48]] = np.asarray([1e6, 0.0])
    summary = summarize_state_gram(_gram(vectors), state_position=0, prefix=4)
    raw_split_cosines = np.asarray(json.loads(summary["raw_split_cosines"]), dtype=float)
    unit_split_cosines = np.asarray(json.loads(summary["split_cosines"]), dtype=float)
    assert (raw_split_cosines >= 0.80).all()
    assert unit_split_cosines[0] < 0.0
    assert summary["split_vector_kind"] == "unit"
    assert summary["split_gate_pass"] is False


def test_frozen_decision_logic_requires_both_probe_gates() -> None:
    assert (
        primary_mechanism_decision(
            delta_dir_ci_low=0.01,
            delta_dir_ci_high=0.2,
            within_state_cosine_median_ci_high=0.79,
            state_nonalignment_gate_passed=False,
        )
        == "probe-direction-major"
    )
    assert (
        primary_mechanism_decision(
            delta_dir_ci_low=-0.3,
            delta_dir_ci_high=-0.01,
            within_state_cosine_median_ci_high=0.9,
            state_nonalignment_gate_passed=True,
        )
        == "state-direction-major"
    )
    assert (
        primary_mechanism_decision(
            delta_dir_ci_low=0.0,
            delta_dir_ci_high=0.2,
            within_state_cosine_median_ci_high=0.79,
            state_nonalignment_gate_passed=True,
        )
        == "comparable/inconclusive"
    )
    assert (
        primary_mechanism_decision(
            delta_dir_ci_low=-0.2,
            delta_dir_ci_high=-0.01,
            within_state_cosine_median_ci_high=0.1,
            state_nonalignment_gate_passed=False,
        )
        == "comparable/inconclusive"
    )


def test_corrected_panel_energy_matches_off_diagonal_task_difference_signal() -> None:
    generator = np.random.default_rng(44)
    vectors = generator.normal(size=(2, 16, DRAW_COUNT, 4))
    task_grams = {
        "fashion_mnist": _gram(vectors[0].reshape(16 * DRAW_COUNT, 4)),
        "mnist": _gram(vectors[1].reshape(16 * DRAW_COUNT, 4)),
    }
    task_sums = vectors.sum(axis=1)
    probes = generator.integers(0, DRAW_COUNT, size=DRAW_COUNT)
    observed = corrected_panel_contrast_energy_from_task_grams(
        task_grams,
        task_sums[0] @ task_sums[1].T,
        probes,
    )
    differences = vectors[0].mean(axis=0) - vectors[1].mean(axis=0)
    selected = differences[probes]
    selected_gram = selected @ selected.T
    signal = (selected_gram.sum() - np.trace(selected_gram)) / (
        DRAW_COUNT * (DRAW_COUNT - 1)
    )
    expected = 16.0 * float(signal) / (2.0 * 31.0)
    assert np.isclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_primary_bootstrap_reports_total_state_and_interaction_components() -> None:
    generator = np.random.default_rng(45)
    vectors = generator.normal(size=(2, 16, DRAW_COUNT, 3))
    flattened = vectors.reshape(2, 16 * DRAW_COUNT, 3)
    task_grams = {
        "fashion_mnist": _gram(flattened[0]),
        "mnist": _gram(flattened[1]),
    }
    raw_sums = vectors.sum(axis=1)
    unit_vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    unit_sums = unit_vectors.sum(axis=1)
    cross = {
        "raw": raw_sums[0] @ raw_sums[1].T,
        "unit": unit_sums[0] @ unit_sums[1].T,
    }
    observed = fixed_panel_probe_bootstrap(task_grams, cross, draws=2, seed=99)
    assert len(observed) == 4
    assert set(observed["vector_kind"]) == {"raw", "unit"}
    required = [
        "E_task_panel",
        "E_state_within",
        "V_state_total",
        "Delta_total",
        "fashion_minus_mnist_state_x_probe_ms",
    ]
    assert np.isfinite(observed[required].to_numpy(dtype=float)).all()
    np.testing.assert_allclose(
        observed["Delta_total"],
        observed["V_probe"] - observed["V_state_total"],
        rtol=0.0,
        atol=1e-12,
    )


def test_state_bank_exact_cardinality_balance_and_string_u63_hashes() -> None:
    bank = _read_bank(Path(STATE_BANK_PATH))
    excluded = pd.read_csv(EXCLUDED_PRIMARY_PATH)
    result = validate_state_bank(bank, excluded)
    assert result["state_count"] == 44
    assert result["primary_count"] == 32
    assert result["sentinel_count"] == 12
    primary = bank.loc[bank["panel"] == "primary"]
    assert primary["run_selection_hash"].str.match(r"^u63_[0-9]+$").all()
    assert primary["snapshot_selection_hash"].str.match(r"^u63_[0-9]+$").all()
    assert primary.groupby(["task_name", "step_stratum"]).size().eq(4).all()


def test_many_memmap_gram_uses_one_combined_state_matrix(tmp_path: Path) -> None:
    paths = []
    matrices = []
    for state in range(3):
        matrix = np.arange(14, dtype=np.float32).reshape(2, 7) + state
        path = tmp_path / f"state_{state}.f32"
        mapped = np.memmap(path, mode="w+", dtype=np.float32, shape=matrix.shape)
        mapped[:] = matrix
        mapped.flush()
        del mapped
        paths.append(path)
        matrices.append(matrix)
    observed = chunked_many_memmap_gram(
        paths,
        rows_per_path=2,
        dimension=7,
        chunk_size=3,
        device=torch.device("cpu"),
    )
    dense = np.concatenate(matrices, axis=0).astype(np.float64)
    np.testing.assert_allclose(observed, dense @ dense.T, rtol=0.0, atol=1e-12)


def test_weighted_resample_components_match_explicit_duplicated_design() -> None:
    generator = np.random.default_rng(91)
    vectors = generator.normal(size=(16 * DRAW_COUNT, 3))
    gram = _gram(vectors)
    states = generator.integers(0, 16, size=16)
    probes = generator.integers(0, DRAW_COUNT, size=DRAW_COUNT)
    observed = _resampled_components_from_task_gram(gram, states, probes)
    cells = np.arange(16 * DRAW_COUNT).reshape(16, DRAW_COUNT)
    selected = cells[states][:, probes].ravel()
    expected = delta_from_crossed_gram(
        gram[np.ix_(selected, selected)],
        ["fashion_mnist"] * 16,
        q=DRAW_COUNT,
    )
    assert np.isclose(observed.v_probe, expected.v_probe, rtol=1e-12, atol=1e-12)
    assert np.isclose(observed.v_state, expected.v_state, rtol=1e-12, atol=1e-12)
    assert np.isclose(observed.delta, expected.delta, rtol=1e-12, atol=1e-12)


def test_scratch_ownership_refuses_existing_path_and_cleans_only_owned(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "foreign.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _create_scratch(scratch)
    assert (scratch / "foreign.txt").exists()

    owned = tmp_path / "owned"
    _create_scratch(owned)
    _cleanup_owned_scratch(owned)
    assert not owned.exists()
