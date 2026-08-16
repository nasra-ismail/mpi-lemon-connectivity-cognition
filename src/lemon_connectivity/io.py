"""Input and validation helpers for Yadav Schaefer-200 edge lists."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_COLUMNS = ["roi_i", "roi_j", "correlation"]
SUBJECT_PATTERN = re.compile(r"^fcm_sub_(\d+)\.txt$")


def expected_edge_count(n_rois: int) -> int:
    """Return the number of unique undirected edges without self-connections."""
    if n_rois < 2:
        raise ValueError("n_rois must be at least 2")
    return n_rois * (n_rois - 1) // 2


def parse_subject_id(path: str | Path) -> str:
    """Extract the numeric subject ID from a Yadav FCM filename."""
    name = Path(path).name
    match = SUBJECT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected FCM filename: {name}")
    return match.group(1)


def load_fcm_edge_list(path: str | Path, n_rois: int = 200) -> pd.DataFrame:
    """Load and validate one complete undirected FCM edge list."""
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] != 3:
        raise ValueError(f"Expected 3 columns, found {raw.shape[1]}")

    edges = raw.copy()
    edges.columns = EDGE_COLUMNS
    for column in EDGE_COLUMNS:
        edges[column] = pd.to_numeric(edges[column], errors="raise")

    edges[["roi_i", "roi_j"]] = edges[["roi_i", "roi_j"]].astype(int)
    expected = expected_edge_count(n_rois)
    if len(edges) != expected:
        raise ValueError(f"Expected {expected} edges, found {len(edges)}")
    if edges.isna().any().any():
        raise ValueError("Edge list contains missing values")
    if not np.isfinite(edges["correlation"]).all():
        raise ValueError("Correlations must be finite")
    if not edges["correlation"].between(-1.0, 1.0).all():
        raise ValueError("Correlations must lie between -1 and 1")

    nodes = edges[["roi_i", "roi_j"]].to_numpy()
    if nodes.min() < 0 or nodes.max() >= n_rois:
        raise ValueError(f"ROI indices must lie between 0 and {n_rois - 1}")
    if (edges["roi_i"] == edges["roi_j"]).any():
        raise ValueError("Self-connections are not allowed")

    unordered_pairs = np.sort(nodes, axis=1)
    if pd.DataFrame(unordered_pairs).duplicated().any():
        raise ValueError("Duplicate undirected ROI pairs found")

    return edges


def edge_list_to_matrix(edges: pd.DataFrame, n_rois: int = 200) -> np.ndarray:
    """Convert a validated complete edge list into a symmetric matrix."""
    missing_columns = set(EDGE_COLUMNS) - set(edges.columns)
    if missing_columns:
        raise ValueError(f"Missing edge columns: {sorted(missing_columns)}")
    if len(edges) != expected_edge_count(n_rois):
        raise ValueError("A complete undirected edge list is required")

    matrix = np.full((n_rois, n_rois), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    i = edges["roi_i"].to_numpy(dtype=int)
    j = edges["roi_j"].to_numpy(dtype=int)
    r = edges["correlation"].to_numpy(dtype=float)
    matrix[i, j] = r
    matrix[j, i] = r

    if not np.isfinite(matrix).all():
        raise ValueError("Edge list did not fill the complete matrix")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Reconstructed matrix is not symmetric")
    return matrix

