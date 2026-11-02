from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from lemon_connectivity.networks import (
    SEGREGATION_DEFINITIONS,
    calculate_network_statistics,
    preprocess_connectivity_matrix,
)
from lemon_connectivity.sensitivity import (
    calculate_segregation_sensitivities,
    compare_sensitivity_models,
    fcm_manifest_checksum,
    fit_age_group_comparison,
    prepare_sensitivity_analysis,
    select_sensitivity_inputs,
    sha256_file,
    summarize_sensitivity_eligibility,
    verify_primary_inputs,
    verify_primary_replication,
)

LABELS = ["A", "A", "B", "B"]


def example_matrix():
    # Within edges: -0.2, 0.6. Between edges: -0.1, 0.2, 0.3, -0.4.
    return np.array(
        [
            [1.0, -0.2, -0.1, 0.2],
            [-0.2, 1.0, 0.3, -0.4],
            [-0.1, 0.3, 1.0, 0.6],
            [0.2, -0.4, 0.6, 1.0],
        ]
    )


@pytest.mark.parametrize("definition", SEGREGATION_DEFINITIONS)
def test_definitions_match_hand_calculated_means_and_preserve_input(definition):
    matrix = example_matrix()
    original = matrix.copy()
    transformed = preprocess_connectivity_matrix(matrix, definition=definition)
    result = calculate_network_statistics(matrix, LABELS, definition=definition)
    fisher = definition in {"seg_z_positive", "seg_z_signed"}
    positive = definition in {"seg_z_positive", "seg_r_positive"}
    f = np.arctanh if fisher else float
    # Explicit sums retain all two/four edge denominators, including zeros.
    expected_w = (f(0.6) + (0 if positive else f(-0.2))) / 2
    expected_b = (f(0.2) + f(0.3) + (0 if positive else f(-0.1) + f(-0.4))) / 4
    assert result["within_mean"] == pytest.approx(expected_w)
    assert result["between_mean"] == pytest.approx(expected_b)
    assert result["system_segregation"] == pytest.approx(
        (expected_w - expected_b) / expected_w
    )
    assert result["within_edge_count"] == 2
    assert result["between_edge_count"] == 4
    assert result["calculation_status"] == "calculated"
    assert transformed.shape == (4, 4)
    np.testing.assert_array_equal(transformed, transformed.T)
    np.testing.assert_array_equal(np.diag(transformed), 0)
    np.testing.assert_array_equal(matrix, original)
    assert not np.shares_memory(matrix, transformed)


def test_default_primary_matches_legacy_behavior_exactly():
    matrix = example_matrix()
    off_diagonal = ~np.eye(4, dtype=bool)
    legacy = matrix.copy()
    values = np.arctanh(matrix[off_diagonal])
    values[values < 0] = 0
    legacy[off_diagonal] = values
    np.fill_diagonal(legacy, 0)
    np.testing.assert_array_equal(preprocess_connectivity_matrix(matrix), legacy)
    assert calculate_network_statistics(matrix, LABELS) == calculate_network_statistics(
        matrix, LABELS, definition="seg_z_positive"
    )


@pytest.mark.parametrize("definition", SEGREGATION_DEFINITIONS)
@pytest.mark.parametrize("endpoint", [-1.0, 1.0])
def test_endpoint_handling_is_definition_specific(definition, endpoint):
    matrix = example_matrix()
    matrix[0, 1] = matrix[1, 0] = endpoint
    if definition in {"seg_z_positive", "seg_z_signed"}:
        with pytest.raises(ValueError, match="strictly between"):
            preprocess_connectivity_matrix(matrix, definition=definition)
    else:
        result = preprocess_connectivity_matrix(matrix, definition=definition)
        expected = max(endpoint, 0) if definition == "seg_r_positive" else endpoint
        assert result[0, 1] == expected


@pytest.mark.parametrize("definition", SEGREGATION_DEFINITIONS)
@pytest.mark.parametrize("value", [1.01, -1.01, np.nan, np.inf])
def test_invalid_correlations_are_not_clipped(definition, value):
    matrix = example_matrix()
    matrix[0, 1] = matrix[1, 0] = value
    with pytest.raises(ValueError):
        preprocess_connectivity_matrix(matrix, definition=definition)


@pytest.mark.parametrize("definition", SEGREGATION_DEFINITIONS)
@pytest.mark.parametrize("target_w", [-0.2, 0.0, 1e-13, 1e-12])
def test_all_definitions_keep_means_when_denominator_is_unavailable(
    definition, target_w
):
    fisher = definition in {"seg_z_positive", "seg_z_signed"}
    matrix = np.full((4, 4), 0.01)
    r = np.tanh(target_w) if fisher else target_w
    matrix[0, 1] = matrix[1, 0] = r
    matrix[2, 3] = matrix[3, 2] = r
    np.fill_diagonal(matrix, 1)
    result = calculate_network_statistics(matrix, LABELS, definition=definition)
    expected_w = max(target_w, 0) if definition.endswith("positive") else target_w
    status = (
        "negative" if expected_w < 0 else "zero" if expected_w == 0 else "near_zero"
    )
    assert result["calculation_status"] == "undefined_within_mean_" + status
    assert np.isnan(result["system_segregation"])
    assert result["within_mean"] == pytest.approx(expected_w, abs=1e-20)
    assert result["between_mean"] == pytest.approx(np.arctanh(0.01) if fisher else 0.01)
    assert result["calculation_reason"]


@pytest.mark.parametrize("definition", ["seg_r_signed", "seg_z_signed"])
def test_signed_scores_above_one_are_retained(definition):
    matrix = np.full((4, 4), -0.1)
    matrix[0, 1] = matrix[1, 0] = 0.2
    matrix[2, 3] = matrix[3, 2] = 0.2
    np.fill_diagonal(matrix, 1)
    result = calculate_network_statistics(matrix, LABELS, definition=definition)
    assert result["system_segregation"] > 1
    assert result["calculation_status"] == "calculated"


@pytest.mark.parametrize("definition", SEGREGATION_DEFINITIONS)
def test_all_definitions_are_invariant_to_joint_roi_reordering(definition):
    matrix = example_matrix()
    permutation = [2, 0, 3, 1]
    before = calculate_network_statistics(matrix, LABELS, definition=definition)
    after = calculate_network_statistics(
        matrix[np.ix_(permutation, permutation)],
        np.array(LABELS)[permutation],
        definition=definition,
    )
    for column in ["within_mean", "between_mean", "system_segregation"]:
        assert after[column] == pytest.approx(before[column])


@pytest.mark.parametrize(
    "definition, error", [("unknown", ValueError), (None, TypeError)]
)
def test_unknown_definitions_are_rejected(definition, error):
    with pytest.raises(error, match="definition"):
        preprocess_connectivity_matrix(example_matrix(), definition=definition)


def write_edges(path, matrix):
    i, j = np.triu_indices(len(matrix), k=1)
    np.savetxt(
        path,
        np.column_stack([i, j, matrix[i, j]]),
        delimiter="\t",
        fmt=["%d", "%d", "%.18e"],
    )


def test_batch_retains_failures_and_evaluates_raw_endpoints(tmp_path):
    paths = [tmp_path / f"fcm_sub_{i:05d}.txt" for i in range(1, 5)]
    negative = example_matrix()
    negative[2, 3] = negative[3, 2] = -0.3
    endpoint = example_matrix()
    endpoint[0, 1] = endpoint[1, 0] = -1
    for path, matrix in zip(
        paths[:3], [example_matrix(), negative, endpoint], strict=True
    ):
        write_edges(path, matrix)
    inputs = pd.DataFrame(
        {
            "canonical_id": [f"sub-{i:06d}" for i in range(1, 5)],
            "matrix_path": [str(path) for path in paths],
        }
    )
    ledger = calculate_segregation_sensitivities(inputs.iloc[::-1], LABELS)
    assert len(ledger) == 16
    assert ledger["canonical_id"].tolist() == sorted(ledger["canonical_id"].tolist())
    assert ledger.groupby("canonical_id").size().eq(4).all()
    assert not ledger.duplicated(["canonical_id", "definition"]).any()
    rows = ledger.set_index(["canonical_id", "definition"])
    assert (
        rows.loc[("sub-000002", "seg_z_positive"), "calculation_status"]
        == "undefined_within_mean_zero"
    )
    assert (
        rows.loc[("sub-000002", "seg_r_signed"), "calculation_status"]
        == "undefined_within_mean_negative"
    )
    assert (
        rows.loc[("sub-000003", "seg_r_positive"), "calculation_status"] == "calculated"
    )
    assert (
        rows.loc[("sub-000003", "seg_z_positive"), "calculation_status"]
        == "calculation_failed"
    )
    assert rows.loc["sub-000004", "calculation_status"].eq("load_failed").all()
    assert rows.loc["sub-000004", "system_segregation"].isna().all()
    assert rows.loc["sub-000001", "positive_edge_density"].eq(0.5).all()
    # Primary replication includes unavailable records without discarding their means.
    primary = ledger.loc[ledger["definition"].eq("seg_z_positive")].copy()
    assert verify_primary_replication(ledger, primary)["matches"].all()


@pytest.fixture
def synthetic_tables():
    ids = [f"sub-{i:06d}" for i in range(1, 13)]
    cohort = pd.DataFrame(
        {
            "canonical_id": ids,
            "cohort": ["young"] * 6 + ["elderly"] * 6,
            "sex": (["female"] * 3 + ["male"] * 3) * 2,
        }
    )
    alignment = pd.DataFrame(
        {
            "canonical_id": ids,
            "alignment_status": "matched",
            "matrix_filename": [f"fcm_sub_{i:05d}.txt" for i in range(1, 13)],
        }
    )
    records = []
    noise = [
        -0.017,
        0.004,
        0.013,
        -0.022,
        0.007,
        0.015,
        -0.012,
        -0.001,
        0.013,
        -0.025,
        0.009,
        0.016,
    ]
    for row_number, row in enumerate(cohort.itertuples()):
        for definition, effect in zip(
            SEGREGATION_DEFINITIONS, [-0.03, -0.02, 0.04, -0.01], strict=True
        ):
            score = (
                0.6
                + effect * (row.cohort == "elderly")
                + 0.02 * (row.sex == "male")
                + noise[row_number]
            )
            records.append(
                {
                    "canonical_id": row.canonical_id,
                    "definition": definition,
                    "edge_policy_version": "edge-policy-v1",
                    "within_mean": 0.3,
                    "between_mean": 0.3 * (1 - score),
                    "system_segregation": score,
                    "within_edge_count": 2,
                    "between_edge_count": 4,
                    "calculation_status": "calculated",
                    "calculation_reason": "",
                }
            )
    return pd.DataFrame(records), cohort, alignment


@pytest.mark.parametrize(
    "column",
    [
        "within_mean",
        "between_mean",
        "system_segregation",
        "within_edge_count",
        "between_edge_count",
        "calculation_status",
    ],
)
def test_primary_replication_detects_changed_results(synthetic_tables, column):
    ledger, _, _ = synthetic_tables
    baseline = ledger.loc[ledger["definition"].eq("seg_z_positive")].copy()
    if column == "calculation_status":
        baseline.loc[baseline.index[0], column] = "calculation_failed"
    elif column.endswith("edge_count"):
        baseline.loc[baseline.index[0], column] += 1
    else:
        baseline.loc[baseline.index[0], column] += 0.1
    with pytest.raises(ValueError, match="Primary replication failed"):
        verify_primary_replication(ledger, baseline)


def test_missing_definition_and_changed_primary_inventory_are_rejected(
    synthetic_tables,
):
    ledger, cohort, alignment = synthetic_tables
    baseline = ledger.loc[ledger["definition"].eq("seg_z_positive")].copy()
    with pytest.raises(ValueError, match="inventory changed"):
        verify_primary_replication(ledger, baseline.iloc[1:])
    with pytest.raises(ValueError, match="all four"):
        prepare_sensitivity_analysis(ledger.iloc[1:], cohort, alignment)


def test_common_sample_is_intersection_and_primary_reference_is_separate(
    synthetic_tables,
):
    ledger, cohort, alignment = synthetic_tables
    failed = ledger["canonical_id"].eq("sub-000001") & ledger["definition"].eq(
        "seg_r_signed"
    )
    ledger.loc[failed, ["within_mean", "system_segregation"]] = [-0.1, np.nan]
    ledger.loc[failed, "calculation_status"] = "undefined_within_mean_negative"
    ledger.loc[failed, "calculation_reason"] = "Synthetic nonpositive W"
    cohort.loc[1, "sex"] = None
    original = ledger.copy(deep=True)
    analysis = prepare_sensitivity_analysis(ledger, cohort, alignment)
    pd.testing.assert_frame_equal(ledger, original)
    assert len(analysis) == 48
    common = analysis.loc[analysis["in_common_sample"]]
    assert common["canonical_id"].nunique() == 10
    assert set(common["canonical_id"]) == set(cohort["canonical_id"]) - {
        "sub-000001",
        "sub-000002",
    }
    assert common.groupby("canonical_id").size().eq(4).all()
    summary = summarize_sensitivity_eligibility(analysis).set_index("definition")
    assert summary["common_sample"].eq(10).all()
    assert summary.loc["seg_z_positive", "eligible_for_definition"] == 11
    assert summary.loc["seg_r_signed", "eligible_for_definition"] == 10
    models = compare_sensitivity_models(analysis)
    assert models["model_status"].eq("estimated").all()
    assert models.loc[models["sample"].eq("common"), "n_total"].eq(10).all()
    assert (
        models.loc[models["sample"].eq("primary_full_reference"), "n_total"].item()
        == 11
    )


@pytest.mark.parametrize(
    "problem",
    ["missing_metadata", "duplicate_metadata", "unmatched", "duplicate_ledger"],
)
def test_alignment_problems_do_not_silently_drop_participants(
    synthetic_tables, problem
):
    ledger, cohort, alignment = synthetic_tables
    if problem == "missing_metadata":
        cohort = cohort.iloc[1:]
    elif problem == "duplicate_metadata":
        cohort = pd.concat([cohort, cohort.iloc[:1]])
    elif problem == "unmatched":
        alignment.loc[0, "alignment_status"] = "metadata_only"
    else:
        ledger = pd.concat([ledger, ledger.iloc[:1]])
    with pytest.raises(ValueError):
        prepare_sensitivity_analysis(ledger, cohort, alignment)


def test_hc3_model_matches_independent_matrix_calculation(synthetic_tables):
    ledger, cohort, alignment = synthetic_tables
    analysis = prepare_sensitivity_analysis(ledger, cohort, alignment)
    data = analysis.loc[analysis["definition"].eq("seg_z_positive")].copy()
    data["cohort"] = data["cohort"].astype("string")
    data["sex"] = data["sex"].astype("string")
    result = fit_age_group_comparison(data)
    x = np.column_stack(
        [
            np.ones(len(data)),
            data["cohort"].eq("elderly").astype(int),
            data["sex"].eq("male").astype(int),
        ]
    )
    y = data["system_segregation"].to_numpy()
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.diag(x @ inverse @ x.T)
    covariance = inverse @ x.T @ np.diag((residual / (1 - leverage)) ** 2) @ x @ inverse
    se = np.sqrt(covariance[1, 1])
    degrees = len(data) - 3
    critical = t.ppf(0.975, degrees)
    assert result["adjusted_difference"] == pytest.approx(beta[1])
    assert result["hc3_robust_se"] == pytest.approx(se)
    assert result["ci_95_lower"] == pytest.approx(beta[1] - critical * se)
    assert result["ci_95_upper"] == pytest.approx(beta[1] + critical * se)
    assert result["p_value_two_sided"] == pytest.approx(
        2 * t.sf(abs(beta[1] / se), degrees)
    )
    assert result["residual_df"] == degrees


def test_rank_deficient_model_is_not_silently_fitted(synthetic_tables):
    ledger, cohort, alignment = synthetic_tables
    cohort["sex"] = np.where(cohort["cohort"].eq("elderly"), "male", "female")
    analysis = prepare_sensitivity_analysis(ledger, cohort, alignment)
    models = compare_sensitivity_models(analysis)
    assert len(models) == 5
    assert models["model_status"].eq("not_estimable").all()
    assert models["model_reason"].str.contains("not separately estimable").all()
    assert models["adjusted_difference"].isna().all()


def test_no_common_sample_preserves_all_models_and_primary_reference(synthetic_tables):
    ledger, cohort, alignment = synthetic_tables
    signed = ledger["definition"].eq("seg_r_signed")
    ledger.loc[signed, "within_mean"] = -0.1
    ledger.loc[signed, "system_segregation"] = np.nan
    ledger.loc[signed, "calculation_status"] = "undefined_within_mean_negative"
    analysis = prepare_sensitivity_analysis(ledger, cohort, alignment)
    models = compare_sensitivity_models(analysis)
    common = models.loc[models["sample"].eq("common")]
    assert common["definition"].tolist() == list(SEGREGATION_DEFINITIONS)
    assert common["model_status"].eq("not_estimable").all()
    assert common["n_total"].eq(0).all()
    assert models.iloc[-1]["sample"] == "primary_full_reference"
    assert models.iloc[-1]["model_status"] == "estimated"


def test_filename_id_mismatch_and_duplicate_paths_are_rejected(
    tmp_path, synthetic_tables
):
    ledger, _, alignment = synthetic_tables
    primary = ledger.loc[ledger["definition"].eq("seg_z_positive")]
    inputs = select_sensitivity_inputs(primary, alignment, tmp_path)
    assert Path(inputs.iloc[0]["matrix_path"]).parent == tmp_path
    alignment.loc[0, "matrix_filename"] = "fcm_sub_99999.txt"
    with pytest.raises(ValueError, match="do not match"):
        select_sensitivity_inputs(primary, alignment, tmp_path)
    inputs.loc[1, "matrix_path"] = inputs.loc[0, "matrix_path"]
    with pytest.raises(ValueError, match="multiple participants"):
        calculate_segregation_sensitivities(inputs, LABELS)


def test_primary_provenance_requires_exact_source_checksums(tmp_path):
    matrix_path = tmp_path / "fcm_sub_00001.txt"
    write_edges(matrix_path, example_matrix())
    inputs = pd.DataFrame(
        {"canonical_id": ["sub-000001"], "matrix_path": [str(matrix_path)]}
    )
    atlas_path = tmp_path / "labels.tsv"
    atlas_path.write_text("synthetic atlas\n", encoding="utf-8")
    provenance = pd.DataFrame(
        [
            {
                "edge_policy_version": "edge-policy-v1",
                "segregation_definition": "seg_z_positive",
                "within_mean_denominator_rule": "W <= 1e-12 is undefined",
                "n_rois": 200,
                "n_retained_participants": 1,
                "n_processed_fcm_files": 1,
                "atlas_sha256": sha256_file(atlas_path),
                "fcm_manifest_sha256": fcm_manifest_checksum(inputs, tmp_path),
            }
        ]
    )
    hashes = verify_primary_inputs(provenance, inputs, atlas_path, tmp_path)
    assert hashes["atlas_sha256"] == sha256_file(atlas_path)
    atlas_path.write_text("changed atlas\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Inputs changed.*atlas_sha256"):
        verify_primary_inputs(provenance, inputs, atlas_path, tmp_path)
