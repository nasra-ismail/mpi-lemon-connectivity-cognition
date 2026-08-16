from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lemon_connectivity.io import (
    edge_list_to_matrix,
    expected_edge_count,
    load_fcm_edge_list,
    parse_subject_id,
)


def _complete_edges(n_rois: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            rows.append((i, j, (i + j) / 10))
    return pd.DataFrame(rows, columns=["roi_i", "roi_j", "correlation"])


def test_expected_edge_count() -> None:
    assert expected_edge_count(200) == 19_900


def test_parse_subject_id() -> None:
    assert parse_subject_id(Path("fcm_sub_32301.txt")) == "32301"


def test_load_and_reconstruct_complete_matrix(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_32301.txt"
    _complete_edges().to_csv(path, sep=" ", header=False, index=False)

    edges = load_fcm_edge_list(path, n_rois=4)
    matrix = edge_list_to_matrix(edges, n_rois=4)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_rejects_incomplete_edge_list(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_32301.txt"
    _complete_edges().iloc[:-1].to_csv(path, sep=" ", header=False, index=False)

    with pytest.raises(ValueError, match="Expected 6 edges"):
        load_fcm_edge_list(path, n_rois=4)

