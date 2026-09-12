from pathlib import Path

import pandas as pd
import pytest

from lemon_connectivity.qc import (
    batch_quality_control,
    summarize_batch_quality_control,
)


def _complete_edges(n_rois: int = 4) -> pd.DataFrame:
    rows = []
    for roi_i in range(n_rois):
        for roi_j in range(roi_i + 1, n_rois):
            rows.append((roi_i, roi_j, (roi_i + roi_j) / 10))
    return pd.DataFrame(rows)


def _write_edges(path: Path, edges: pd.DataFrame) -> None:
    edges.to_csv(path, sep="\t", header=False, index=False)


def _alignment_row(
    canonical_id: str,
    filename: str,
    *,
    filename_valid: bool = True,
    alignment_status: str = "matched",
) -> dict[str, object]:
    return {
        "canonical_id": canonical_id,
        "matrix_filename": filename,
        "filename_valid": filename_valid,
        "alignment_status": alignment_status,
    }


def test_batch_qc_records_metrics_in_deterministic_order(tmp_path: Path) -> None:
    for participant_id in ("000002", "000001"):
        _write_edges(
            tmp_path / f"fcm_sub_{participant_id}.txt",
            _complete_edges(),
        )
    alignment = pd.DataFrame(
        [
            _alignment_row("sub-000002", "fcm_sub_000002.txt"),
            _alignment_row("sub-000001", "fcm_sub_000001.txt"),
        ]
    )

    result = batch_quality_control(
        alignment,
        tmp_path,
        n_rois=4,
        expected_files=2,
    )

    assert result["canonical_id"].tolist() == ["sub-000001", "sub-000002"]
    assert result["observed_rows"].tolist() == [6, 6]
    assert result["missing_values"].tolist() == [0, 0]
    assert result["duplicate_unordered_pairs"].tolist() == [0, 0]
    assert result["correlation_min"].tolist() == pytest.approx([0.1, 0.1])
    assert result["correlation_max"].tolist() == pytest.approx([0.5, 0.5])
    assert result["correlation_mean"].tolist() == pytest.approx([0.3, 0.3])
    assert result["correlation_std"].tolist() == pytest.approx(
        [0.1290994449, 0.1290994449]
    )
    assert result["qc_passed"].all()
    assert result["retained_for_analysis"].all()


def test_batch_qc_keeps_metrics_for_readable_qc_failure(tmp_path: Path) -> None:
    filename = "fcm_sub_000001.txt"
    edges = _complete_edges()
    edges.iloc[0, 2] = 1.5
    _write_edges(tmp_path / filename, edges)
    alignment = pd.DataFrame([_alignment_row("sub-000001", filename)])

    result = batch_quality_control(alignment, tmp_path, n_rois=4)
    row = result.iloc[0]

    assert bool(row["file_read_successfully"])
    assert bool(row["file_loaded_successfully"])
    assert not bool(row["qc_passed"])
    assert row["correlation_max"] == pytest.approx(1.5)
    assert row["failure_code"] == "qc_failed"
    assert "Correlation range" in row["failure_reason"]


def test_batch_qc_isolates_non_numeric_file_failure(tmp_path: Path) -> None:
    good_name = "fcm_sub_000001.txt"
    bad_name = "fcm_sub_000002.txt"
    _write_edges(tmp_path / good_name, _complete_edges())
    bad_edges = _complete_edges().astype(object)
    bad_edges.iloc[0, 2] = "invalid"
    _write_edges(tmp_path / bad_name, bad_edges)
    alignment = pd.DataFrame(
        [
            _alignment_row("sub-000001", good_name),
            _alignment_row("sub-000002", bad_name),
        ]
    )

    result = batch_quality_control(alignment, tmp_path, n_rois=4)
    bad_row = result.loc[result["canonical_id"] == "sub-000002"].iloc[0]

    assert len(result) == 2
    assert bool(result.loc[0, "qc_passed"])
    assert bool(bad_row["file_read_successfully"])
    assert not bool(bad_row["file_loaded_successfully"])
    assert bad_row["observed_rows"] == 6
    assert bad_row["failure_code"] == "load_error"


def test_invalid_filename_is_measured_but_not_retained(tmp_path: Path) -> None:
    filename = "unexpected.txt"
    _write_edges(tmp_path / filename, _complete_edges())
    alignment = pd.DataFrame(
        [
            _alignment_row(
                "sub-000001",
                filename,
                filename_valid=False,
            )
        ]
    )

    row = batch_quality_control(alignment, tmp_path, n_rois=4).iloc[0]

    assert bool(row["content_qc_passed"])
    assert not bool(row["qc_passed"])
    assert not bool(row["retained_for_analysis"])
    assert row["failure_code"] == "invalid_filename"


def test_unmatched_matrix_is_qced_but_not_retained(tmp_path: Path) -> None:
    filename = "fcm_sub_000001.txt"
    _write_edges(tmp_path / filename, _complete_edges())
    alignment = pd.DataFrame(
        [
            _alignment_row(
                "sub-000001",
                filename,
                alignment_status="matrix_without_metadata",
            )
        ]
    )

    row = batch_quality_control(alignment, tmp_path, n_rois=4).iloc[0]

    assert bool(row["qc_passed"])
    assert not bool(row["retained_for_analysis"])


def test_batch_qc_does_not_mutate_alignment_table(tmp_path: Path) -> None:
    filename = "fcm_sub_000001.txt"
    _write_edges(tmp_path / filename, _complete_edges())
    alignment = pd.DataFrame([_alignment_row("sub-000001", filename)])
    original = alignment.copy(deep=True)

    batch_quality_control(alignment, tmp_path, n_rois=4)

    pd.testing.assert_frame_equal(alignment, original)


def test_batch_qc_rejects_missing_alignment_columns(tmp_path: Path) -> None:
    alignment = pd.DataFrame({"canonical_id": ["sub-000001"]})

    with pytest.raises(ValueError, match="missing columns"):
        batch_quality_control(alignment, tmp_path, n_rois=4)


def test_batch_qc_rejects_duplicate_canonical_ids(tmp_path: Path) -> None:
    alignment = pd.DataFrame(
        [
            _alignment_row("sub-000001", "fcm_sub_000001.txt"),
            _alignment_row("sub-000001", "fcm_sub_000002.txt"),
        ]
    )

    with pytest.raises(ValueError, match="duplicated canonical IDs"):
        batch_quality_control(alignment, tmp_path, n_rois=4)


def test_batch_qc_rejects_string_boolean(tmp_path: Path) -> None:
    alignment = pd.DataFrame(
        [
            _alignment_row(
                "sub-000001",
                "fcm_sub_000001.txt",
                filename_valid="False",  # type: ignore[arg-type]
            )
        ]
    )

    with pytest.raises(TypeError, match="Boolean"):
        batch_quality_control(alignment, tmp_path, n_rois=4)


def test_batch_qc_checks_expected_inventory(tmp_path: Path) -> None:
    filename = "fcm_sub_000001.txt"
    _write_edges(tmp_path / filename, _complete_edges())
    alignment = pd.DataFrame([_alignment_row("sub-000001", filename)])

    with pytest.raises(ValueError, match="Expected 2 FCM candidates"):
        batch_quality_control(
            alignment,
            tmp_path,
            n_rois=4,
            expected_files=2,
        )


def test_aggregate_summary_contains_no_identifiers(tmp_path: Path) -> None:
    filename = "fcm_sub_000001.txt"
    _write_edges(tmp_path / filename, _complete_edges())
    alignment = pd.DataFrame([_alignment_row("sub-000001", filename)])
    result = batch_quality_control(alignment, tmp_path, n_rois=4)

    summary = summarize_batch_quality_control(result)

    assert summary["FCM candidates"] == 1
    assert summary["QC passed"] == 1
    assert summary["Retained for analysis"] == 1
    assert "canonical_id" not in summary.index
