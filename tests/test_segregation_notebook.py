"""Regression checks for issue #58 using the notebook's actual source cells."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lemon_connectivity.io import edge_list_to_matrix, load_fcm_edge_list
from lemon_connectivity.networks import (
    calculate_network_statistics,
    calculate_system_segregation,
    create_network_masks,
)


def notebook_cell(marker):
    path = Path(__file__).parents[1] / "notebooks" / "07_network_segregation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "".join(cell["source"]).lstrip().startswith(marker)
    ]
    assert len(matches) == 1, f"Notebook source cell not uniquely found: {marker}"
    return matches[0]


@pytest.mark.parametrize("within_mean", [0.0, 1e-13, 1e-12])
def test_notebook_example_continues_after_undefined_score(within_mean):
    namespace = {
        "np": np,
        "calculate_system_segregation": calculate_system_segregation,
        "within_mean": within_mean,
        "between_mean": 0.001,
    }
    source = notebook_cell("# Calculate the whole-brain system segregation score")
    # Execute only the trusted, checked-in notebook source with synthetic inputs.
    exec(compile(source, "<notebook-example>", "exec"), namespace)  # noqa: S102
    assert np.isnan(namespace["system_segregation"])
    assert namespace["example_calculation_status"].startswith("undefined_")


def test_notebook_batch_retains_means_and_round_trips_undefined_rows(tmp_path):
    labels = np.repeat(np.arange(7), [30, 30, 30, 30, 30, 25, 25])
    within_mask, between_mask = create_network_masks(labels)
    roi_i, roi_j = np.triu_indices(200, k=1)
    participants = []
    for number, within_z in enumerate([-0.002, 1e-13, 0.2], start=1):
        z_values = np.where(labels[:, None] == labels[None, :], within_z, 0.001)
        matrix = np.tanh(z_values)
        np.fill_diagonal(matrix, 1.0)
        path = tmp_path / f"fcm_sub_{number:05d}.txt"
        np.savetxt(
            path,
            np.column_stack([roi_i, roi_j, matrix[roi_i, roi_j]]),
            delimiter="\t",
            fmt=["%d", "%d", "%.18e"],
        )
        participants.append(
            {
                "canonical_id": f"synthetic-{number}",
                "matrix_filename": path.name,
                "matrix_path": str(path),
            }
        )
    namespace = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "repo_root": tmp_path,
        "load_fcm_edge_list": load_fcm_edge_list,
        "edge_list_to_matrix": edge_list_to_matrix,
        "calculate_network_statistics": calculate_network_statistics,
        "participant_files": pd.DataFrame(participants),
        "retained_qc": pd.DataFrame(participants),
        "roi_networks": labels,
        "within_mask": within_mask,
        "between_mask": between_mask,
        "expected_within_edges": int(within_mask.sum()),
        "expected_between_edges": int(between_mask.sum()),
    }
    for marker in [
        "# Calculate network segregation for every retained QC participant",
        "# Verify that the segregation ledger contains exactly the retained QC participants",
        "# Save the participant-level network segregation results",
        "# Reload the saved participant-level results for verification",
    ]:
        # Execute only the trusted, checked-in notebook source with synthetic inputs.
        exec(compile(notebook_cell(marker), "<notebook-batch>", "exec"), namespace)  # noqa: S102

    results = namespace["saved_segregation_results"]
    assert results["canonical_id"].tolist() == [f"synthetic-{i}" for i in (1, 2, 3)]
    assert results["calculation_status"].tolist() == [
        "undefined_within_mean_zero",
        "undefined_within_mean_near_zero",
        "calculated",
    ]
    np.testing.assert_allclose(
        results["within_mean"], [0.0, 1e-13, 0.2], rtol=1e-10, atol=1e-16
    )
    np.testing.assert_allclose(results["between_mean"], 0.001)
    assert results.loc[:1, "system_segregation"].isna().all()
    assert results.loc[2, "system_segregation"] == pytest.approx(0.995)
    assert results.loc[:1, "calculation_reason"].notna().all()
