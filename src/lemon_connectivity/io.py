"""Input and validation helpers for Yadav Schaefer-200 edge lists."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

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


def _coerce_edge_types(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with validated numeric edge-list data types."""
    edges = raw.copy()
    edges.columns = EDGE_COLUMNS

    try:
        for column in EDGE_COLUMNS:
            edges[column] = pd.to_numeric(edges[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Edge list contains non-numeric values") from error

    if edges.isna().any().any():
        raise ValueError("Edge list contains missing values")

    roi_values = edges[["roi_i", "roi_j"]].to_numpy(dtype=float)
    if not np.isfinite(roi_values).all():
        raise ValueError("ROI indices must be finite")
    if not np.equal(roi_values, np.floor(roi_values)).all():
        raise ValueError("ROI indices must be integers")

    edges[["roi_i", "roi_j"]] = edges[["roi_i", "roi_j"]].astype(int)
    edges["correlation"] = edges["correlation"].astype(float)
    return edges


def summarize_fcm_edge_list(
    edges: pd.DataFrame,
    n_rois: int = 200,
) -> pd.DataFrame:
    """Return an expected-versus-observed QC table for one FCM edge list."""
    expected_edges = expected_edge_count(n_rois)
    columns_match = list(edges.columns) == EDGE_COLUMNS
    if not columns_match:
        raise ValueError(
            f"Expected columns {EDGE_COLUMNS}, found {list(edges.columns)}"
        )

    roi_types_valid = all(
        is_integer_dtype(edges[column]) for column in ("roi_i", "roi_j")
    )
    correlation_type_valid = is_numeric_dtype(edges["correlation"])
    if not roi_types_valid or not correlation_type_valid:
        raise TypeError("Summarize only a numerically normalized FCM edge list")

    roi_values = edges[["roi_i", "roi_j"]].to_numpy(dtype=int)
    node_values = set(roi_values.ravel().tolist())
    expected_nodes = set(range(n_rois))
    observed_node_range = (
        f"{min(node_values)}-{max(node_values)}" if node_values else "empty"
    )

    unordered_pairs = np.sort(roi_values, axis=1)
    pair_tuples = [tuple(pair) for pair in unordered_pairs.tolist()]
    unique_pairs = set(pair_tuples)
    expected_pairs = {
        (roi_i, roi_j) for roi_i in range(n_rois) for roi_j in range(roi_i + 1, n_rois)
    }

    self_edges = int((edges["roi_i"] == edges["roi_j"]).sum())
    duplicate_pairs = len(pair_tuples) - len(unique_pairs)
    ordered_pairs = int((edges["roi_i"] < edges["roi_j"]).sum())
    missing_values = int(edges.isna().sum().sum())

    correlations = edges["correlation"].to_numpy(dtype=float)
    finite_mask = np.isfinite(correlations)
    non_finite_correlations = int((~finite_mask).sum())
    if finite_mask.any():
        finite_correlations = correlations[finite_mask]
        correlation_range = (
            f"[{finite_correlations.min():.6f}, {finite_correlations.max():.6f}]"
        )
    else:
        correlation_range = "no finite values"
    correlations_in_range = bool(
        finite_mask.all()
        and (correlations >= -1.0).all()
        and (correlations <= 1.0).all()
    )

    checks = [
        {
            "check": "Edge columns",
            "expected": str(EDGE_COLUMNS),
            "observed": str(list(edges.columns)),
            "passed": columns_match,
        },
        {
            "check": "Edge rows",
            "expected": expected_edges,
            "observed": len(edges),
            "passed": len(edges) == expected_edges,
        },
        {
            "check": "ROI data types",
            "expected": "integer",
            "observed": f"{edges['roi_i'].dtype}, {edges['roi_j'].dtype}",
            "passed": roi_types_valid,
        },
        {
            "check": "Correlation data type",
            "expected": "numeric",
            "observed": str(edges["correlation"].dtype),
            "passed": correlation_type_valid,
        },
        {
            "check": "Unique nodes",
            "expected": n_rois,
            "observed": len(node_values),
            "passed": len(node_values) == n_rois,
        },
        {
            "check": "ROI index set",
            "expected": f"0-{n_rois - 1}",
            "observed": observed_node_range,
            "passed": node_values == expected_nodes,
        },
        {
            "check": "Self-edges",
            "expected": 0,
            "observed": self_edges,
            "passed": self_edges == 0,
        },
        {
            "check": "Duplicate unordered pairs",
            "expected": 0,
            "observed": duplicate_pairs,
            "passed": duplicate_pairs == 0,
        },
        {
            "check": "Upper-triangle ordering",
            "expected": f"{len(edges)}/{len(edges)} rows with roi_i < roi_j",
            "observed": f"{ordered_pairs}/{len(edges)} rows",
            "passed": ordered_pairs == len(edges),
        },
        {
            "check": "Complete unordered pairs",
            "expected": expected_edges,
            "observed": len(unique_pairs),
            "passed": unique_pairs == expected_pairs,
        },
        {
            "check": "Missing values",
            "expected": 0,
            "observed": missing_values,
            "passed": missing_values == 0,
        },
        {
            "check": "Non-finite correlations",
            "expected": 0,
            "observed": non_finite_correlations,
            "passed": non_finite_correlations == 0,
        },
        {
            "check": "Correlation range",
            "expected": "within [-1, 1]",
            "observed": correlation_range,
            "passed": correlations_in_range,
        },
    ]
    return pd.DataFrame(checks)


def load_fcm_edge_list(path: str | Path, n_rois: int = 200) -> pd.DataFrame:
    """Load and validate one complete undirected FCM edge list."""
    try:
        raw = pd.read_csv(path, sep="\t", header=None)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(
            "Could not parse FCM edge list as tab-separated text"
        ) from error

    if raw.shape[1] != 3:
        raise ValueError(f"Expected 3 columns, found {raw.shape[1]}")

    edges = _coerce_edge_types(raw)
    qc_summary = summarize_fcm_edge_list(edges, n_rois=n_rois)
    failed_checks = qc_summary.loc[~qc_summary["passed"], "check"].tolist()
    if failed_checks:
        raise ValueError("FCM edge-list validation failed: " + ", ".join(failed_checks))

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
