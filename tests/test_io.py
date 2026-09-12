from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lemon_connectivity.io import (
    edge_list_to_matrix,
    expected_edge_count,
    load_fcm_edge_list,
    normalize_fcm_edge_list,
    parse_subject_id,
    read_fcm_edge_list,
    summarize_fcm_edge_list,
)


def _complete_edges(n_rois: int = 4) -> pd.DataFrame:
    rows = []
    for roi_i in range(n_rois):
        for roi_j in range(roi_i + 1, n_rois):
            rows.append((roi_i, roi_j, (roi_i + roi_j) / 10))
    return pd.DataFrame(rows, columns=["roi_i", "roi_j", "correlation"])


def _write_edges(path: Path, edges: pd.DataFrame) -> None:
    edges.to_csv(path, sep="\t", header=False, index=False)


def test_expected_edge_count() -> None:
    assert expected_edge_count(200) == 19_900


def test_expected_edge_count_rejects_fewer_than_two_rois() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        expected_edge_count(1)


def test_parse_subject_id() -> None:
    assert parse_subject_id(Path("fcm_sub_000001.txt")) == "000001"


def test_parse_subject_id_rejects_unexpected_filename() -> None:
    with pytest.raises(ValueError, match="Unexpected FCM filename"):
        parse_subject_id(Path("matrix_000001.csv"))


def test_read_and_normalize_edge_list(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    _write_edges(path, _complete_edges())

    raw = read_fcm_edge_list(path)
    edges = normalize_fcm_edge_list(raw)

    assert raw.shape == (6, 3)
    assert list(edges.columns) == ["roi_i", "roi_j", "correlation"]
    assert edges["roi_i"].dtype.kind in "iu"
    assert edges["correlation"].dtype.kind == "f"


def test_load_summarize_and_reconstruct_complete_matrix(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    _write_edges(path, _complete_edges())

    edges = load_fcm_edge_list(path, n_rois=4)
    summary = summarize_fcm_edge_list(edges, n_rois=4)
    matrix = edge_list_to_matrix(edges, n_rois=4)

    assert summary["passed"].all()
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_rejects_incomplete_edge_list(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    _write_edges(path, _complete_edges().iloc[:-1])

    with pytest.raises(ValueError, match="Edge rows"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_wrong_number_of_columns(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    _write_edges(path, _complete_edges().iloc[:, :2])

    with pytest.raises(ValueError, match="Expected 3 columns"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_space_delimited_file(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    _complete_edges().to_csv(path, sep=" ", header=False, index=False)

    with pytest.raises(ValueError, match="Expected 3 columns"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_non_numeric_values(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges().astype(object)
    edges.loc[0, "correlation"] = "not-a-number"
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="non-numeric"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_fractional_roi_indices(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges["roi_i"] = edges["roi_i"].astype(float)
    edges.loc[0, "roi_i"] = 0.5
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="ROI indices must be integers"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_out_of_range_roi_indices(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.loc[0, "roi_j"] = 4
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="ROI index set"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_self_edges(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.loc[0, "roi_j"] = edges.loc[0, "roi_i"]
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="Self-edges"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_duplicate_unordered_pairs(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.loc[edges.index[-1], ["roi_i", "roi_j"]] = [1, 0]
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="Duplicate unordered pairs"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_lower_triangle_orientation(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.loc[0, ["roi_i", "roi_j"]] = [1, 0]
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="Upper-triangle ordering"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.loc[0, "correlation"] = np.nan
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="Missing values"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_non_finite_correlations(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.loc[0, "correlation"] = np.inf
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="Non-finite correlations"):
        load_fcm_edge_list(path, n_rois=4)


def test_rejects_correlations_outside_valid_range(tmp_path: Path) -> None:
    path = tmp_path / "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.loc[0, "correlation"] = 1.01
    _write_edges(path, edges)

    with pytest.raises(ValueError, match="Correlation range"):
        load_fcm_edge_list(path, n_rois=4)
