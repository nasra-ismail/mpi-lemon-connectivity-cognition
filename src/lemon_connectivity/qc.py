from pathlib import Path

import numpy as np
import pandas as pd

from lemon_connectivity.io import (
    expected_edge_count,
    normalize_fcm_edge_list,
    read_fcm_edge_list,
    summarize_fcm_edge_list,
)

__all__ = ["batch_quality_control", "summarize_batch_quality_control"]

REQUIRED_ALIGNMENT_COLUMNS = {
    "canonical_id",
    "matrix_filename",
    "filename_valid",
    "alignment_status",
}

QC_OUTPUT_COLUMNS = [
    "canonical_id",
    "matrix_filename",
    "alignment_status",
    "filename_pattern_passed",
    "file_read_successfully",
    "file_loaded_successfully",
    "observed_rows",
    "observed_columns",
    "expected_rows",
    "row_count_passed",
    "missing_values",
    "unique_nodes",
    "roi_min",
    "roi_max",
    "exact_node_set_passed",
    "self_edges",
    "duplicate_unordered_pairs",
    "upper_triangle_passed",
    "complete_pairs_passed",
    "non_finite_correlations",
    "correlation_min",
    "correlation_max",
    "correlation_mean",
    "correlation_std",
    "correlations_in_bounds",
    "content_qc_passed",
    "qc_passed",
    "retained_for_analysis",
    "failure_code",
    "failure_reason",
]


def _validate_alignment_table(alignment_table: pd.DataFrame) -> pd.DataFrame:
    """Validate the batch-QC input and return matrix candidate rows."""
    missing_columns = REQUIRED_ALIGNMENT_COLUMNS - set(alignment_table.columns)
    if missing_columns:
        raise ValueError(
            f"Alignment table is missing columns: {sorted(missing_columns)}"
        )

    candidates = alignment_table.loc[alignment_table["matrix_filename"].notna()].copy()
    if candidates.empty:
        raise ValueError("Alignment table contains no matrix candidates")

    if candidates["canonical_id"].isna().any():
        raise ValueError("Matrix candidates contain missing canonical IDs")
    if candidates["canonical_id"].duplicated().any():
        raise ValueError("Matrix candidates contain duplicated canonical IDs")
    if candidates["matrix_filename"].duplicated().any():
        raise ValueError("Matrix candidates contain duplicated matrix filenames")

    invalid_booleans = ~candidates["filename_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if invalid_booleans.any():
        raise TypeError("filename_valid must contain Boolean values")

    unsafe_names = candidates["matrix_filename"].map(
        lambda value: (
            not isinstance(value, str)
            or not value
            or "/" in value
            or "\\" in value
            or value in {".", ".."}
        )
    )
    if unsafe_names.any():
        raise ValueError("matrix_filename must contain safe filenames, not paths")

    return candidates.sort_values(
        ["canonical_id", "matrix_filename"], kind="stable"
    ).reset_index(drop=True)


def _new_record(row: pd.Series, n_rois: int) -> dict[str, object]:
    """Create one QC record with explicit defaults."""
    return {
        "canonical_id": row["canonical_id"],
        "matrix_filename": row["matrix_filename"],
        "alignment_status": row["alignment_status"],
        "filename_pattern_passed": bool(row["filename_valid"]),
        "file_read_successfully": False,
        "file_loaded_successfully": False,
        "observed_rows": pd.NA,
        "observed_columns": pd.NA,
        "expected_rows": expected_edge_count(n_rois),
        "row_count_passed": False,
        "missing_values": pd.NA,
        "unique_nodes": pd.NA,
        "roi_min": pd.NA,
        "roi_max": pd.NA,
        "exact_node_set_passed": False,
        "self_edges": pd.NA,
        "duplicate_unordered_pairs": pd.NA,
        "upper_triangle_passed": False,
        "complete_pairs_passed": False,
        "non_finite_correlations": pd.NA,
        "correlation_min": np.nan,
        "correlation_max": np.nan,
        "correlation_mean": np.nan,
        "correlation_std": np.nan,
        "correlations_in_bounds": False,
        "content_qc_passed": False,
        "qc_passed": False,
        "retained_for_analysis": False,
        "failure_code": None,
        "failure_reason": None,
    }


def _record_raw_shape(record: dict[str, object], raw: pd.DataFrame) -> None:
    """Store metrics that remain meaningful before type normalization."""
    record["file_read_successfully"] = True
    record["observed_rows"] = len(raw)
    record["observed_columns"] = raw.shape[1]
    record["row_count_passed"] = len(raw) == record["expected_rows"]
    record["missing_values"] = int(raw.isna().sum().sum())


def _record_edge_metrics(
    record: dict[str, object],
    edges: pd.DataFrame,
    n_rois: int,
) -> None:
    """Store participant-level structural and numerical metrics."""
    roi_values = edges[["roi_i", "roi_j"]].to_numpy(dtype=int)
    nodes = set(roi_values.ravel().tolist())
    expected_nodes = set(range(n_rois))

    unordered_pairs = np.sort(roi_values, axis=1)
    pair_tuples = [tuple(pair) for pair in unordered_pairs.tolist()]
    unique_pairs = set(pair_tuples)
    expected_pairs = {
        (roi_i, roi_j) for roi_i in range(n_rois) for roi_j in range(roi_i + 1, n_rois)
    }

    correlations = edges["correlation"].to_numpy(dtype=float)
    finite_correlations = correlations[np.isfinite(correlations)]
    if len(finite_correlations) > 0:
        correlation_min = float(finite_correlations.min())
        correlation_max = float(finite_correlations.max())
        correlation_mean = float(finite_correlations.mean())
        correlation_std = float(finite_correlations.std(ddof=0))
    else:
        correlation_min = np.nan
        correlation_max = np.nan
        correlation_mean = np.nan
        correlation_std = np.nan

    record.update(
        {
            "file_loaded_successfully": True,
            "unique_nodes": len(nodes),
            "roi_min": min(nodes) if nodes else pd.NA,
            "roi_max": max(nodes) if nodes else pd.NA,
            "exact_node_set_passed": nodes == expected_nodes,
            "self_edges": int((edges["roi_i"] == edges["roi_j"]).sum()),
            "duplicate_unordered_pairs": len(pair_tuples) - len(unique_pairs),
            "upper_triangle_passed": bool((edges["roi_i"] < edges["roi_j"]).all()),
            "complete_pairs_passed": unique_pairs == expected_pairs,
            "non_finite_correlations": int((~np.isfinite(correlations)).sum()),
            "correlation_min": correlation_min,
            "correlation_max": correlation_max,
            "correlation_mean": correlation_mean,
            "correlation_std": correlation_std,
            "correlations_in_bounds": bool(
                len(correlations) > 0
                and np.isfinite(correlations).all()
                and (correlations >= -1.0).all()
                and (correlations <= 1.0).all()
            ),
        }
    )


def batch_quality_control(
    alignment_table: pd.DataFrame,
    fcm_directory: str | Path,
    n_rois: int = 200,
    expected_files: int | None = None,
) -> pd.DataFrame:
    """Run deterministic, failure-isolated QC over aligned FCM candidates.

    Every row with a non-missing ``matrix_filename`` is inspected, including
    matrix-only alignment rows. A file is retained only when its filename and
    content pass QC and its alignment status is ``matched``. Correlation
    summaries use finite values; missing and non-finite counts remain explicit.
    """
    candidates = _validate_alignment_table(alignment_table)
    if expected_files is not None and len(candidates) != expected_files:
        raise ValueError(
            f"Expected {expected_files} FCM candidates, found {len(candidates)}"
        )

    fcm_root = Path(fcm_directory)
    if not fcm_root.is_dir():
        raise FileNotFoundError(f"FCM directory not found: {fcm_root}")

    records: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        record = _new_record(row, n_rois)
        matrix_path = fcm_root / str(row["matrix_filename"])

        try:
            raw = read_fcm_edge_list(matrix_path)
            _record_raw_shape(record, raw)
            edges = normalize_fcm_edge_list(raw)
            _record_edge_metrics(record, edges, n_rois)
            qc_summary = summarize_fcm_edge_list(edges, n_rois=n_rois)
        except FileNotFoundError:
            record["failure_code"] = "file_not_found"
            record["failure_reason"] = "FCM file was not found"
        except (OSError, TypeError, ValueError) as error:
            record["failure_code"] = "load_error"
            record["failure_reason"] = str(error)
        else:
            failed_checks = qc_summary.loc[~qc_summary["passed"], "check"].tolist()
            content_passed = not failed_checks
            filename_passed = bool(record["filename_pattern_passed"])
            overall_passed = content_passed and filename_passed

            record["content_qc_passed"] = content_passed
            record["qc_passed"] = overall_passed
            record["retained_for_analysis"] = bool(
                overall_passed and row["alignment_status"] == "matched"
            )

            if not filename_passed:
                record["failure_code"] = "invalid_filename"
                record["failure_reason"] = "Filename pattern"
            elif failed_checks:
                record["failure_code"] = "qc_failed"
                record["failure_reason"] = "; ".join(failed_checks)

        records.append(record)

    return pd.DataFrame.from_records(records).reindex(columns=QC_OUTPUT_COLUMNS)


def summarize_batch_quality_control(qc_table: pd.DataFrame) -> pd.Series:
    """Return aggregate counts without exposing participant identifiers."""
    required = {
        "file_read_successfully",
        "file_loaded_successfully",
        "qc_passed",
        "retained_for_analysis",
    }
    missing_columns = required - set(qc_table.columns)
    if missing_columns:
        raise ValueError(f"QC table is missing columns: {sorted(missing_columns)}")

    return pd.Series(
        {
            "FCM candidates": len(qc_table),
            "Files read successfully": int(qc_table["file_read_successfully"].sum()),
            "Files normalized successfully": int(
                qc_table["file_loaded_successfully"].sum()
            ),
            "QC passed": int(qc_table["qc_passed"].sum()),
            "QC failed": int((~qc_table["qc_passed"]).sum()),
            "Retained for analysis": int(qc_table["retained_for_analysis"].sum()),
            "Excluded from analysis": int((~qc_table["retained_for_analysis"]).sum()),
        },
        dtype="int64",
    )
