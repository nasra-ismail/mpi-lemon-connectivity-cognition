from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .identifiers import (
    DEFAULT_CANONICAL_WIDTH,
    canonicalize_participant_ids,
    clean_participant_ids,
)

DEFAULT_FCM_PREFIX = "fcm_sub_"
DEFAULT_FCM_FILENAME_REGEX = r"^fcm_sub_([0-9]+)\.txt$"

INVENTORY_COLUMNS = [
    "name",
    "path",
    "entry_type",
    "extension",
    "has_expected_prefix",
    "is_system_entry",
]

ALIGNMENT_COLUMNS = [
    "original_subject_id",
    "canonical_id",
    "metadata_present",
    "metadata_row_count",
    "matrix_present",
    "matrix_file_count",
    "matrix_filename",
    "matrix_path",
    "filename_valid",
    "alignment_status",
]

__all__ = [
    "ALIGNMENT_COLUMNS",
    "DEFAULT_FCM_FILENAME_REGEX",
    "DEFAULT_FCM_PREFIX",
    "FCMAlignmentResult",
    "align_participants_to_fcm",
    "build_alignment_table",
    "compare_participant_id_sets",
    "inventory_fcm_directory",
    "parse_fcm_filenames",
    "summarize_alignment",
    "summarize_fcm_directory",
    "summarize_fcm_filename_validation",
    "validate_alignment_cardinality",
    "validate_fcm_id_uniqueness",
]


@dataclass(frozen=True)
class FCMAlignmentResult:
    """All participant-to-FCM audit products returned by one pipeline run."""

    inventory: pd.DataFrame
    directory_report: pd.Series
    filename_report: pd.Series
    uniqueness_report: pd.Series
    set_report: pd.Series
    alignment_table: pd.DataFrame
    cardinality_report: pd.Series
    alignment_summary: pd.Series


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    frame_name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{frame_name} is missing required columns: {missing}")


def _join_unique_text(values: pd.Series) -> str:
    unique_values = sorted(set(values.dropna().astype("string").tolist()))
    return "; ".join(unique_values)


def inventory_fcm_directory(
    fcm_directory: str | Path,
    *,
    expected_prefix: str = DEFAULT_FCM_PREFIX,
) -> pd.DataFrame:
    """Return a deterministic inventory of entries in an FCM directory.

    The function records paths but does not read matrix contents. System entries
    are identified by a leading dot. No input files are changed.
    """

    directory = Path(fcm_directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"FCM directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"FCM path is not a directory: {directory}")

    directory = directory.resolve()
    records: list[dict[str, object]] = []

    for entry in sorted(directory.iterdir(), key=lambda path: path.name):
        if entry.is_file():
            entry_type = "file"
            extension: object = entry.suffix.lower() or ""
        elif entry.is_dir():
            entry_type = "directory"
            extension = pd.NA
        else:
            entry_type = "other"
            extension = pd.NA

        records.append(
            {
                "name": entry.name,
                "path": str(entry),
                "entry_type": entry_type,
                "extension": extension,
                "has_expected_prefix": entry.name.startswith(expected_prefix),
                "is_system_entry": entry.name.startswith("."),
            }
        )

    return pd.DataFrame.from_records(records, columns=INVENTORY_COLUMNS)


def summarize_fcm_directory(inventory: pd.DataFrame) -> pd.Series:
    """Summarize file types and unexpected entries in an FCM inventory."""

    _require_columns(
        inventory,
        INVENTORY_COLUMNS,
        frame_name="inventory",
    )

    regular_file_mask = inventory["entry_type"].eq("file")
    system_mask = inventory["is_system_entry"].fillna(False)
    non_system_file_mask = regular_file_mask & ~system_mask
    text_file_mask = inventory["extension"].eq(".txt")

    expected_candidate_mask = (
        non_system_file_mask
        & text_file_mask
        & inventory["has_expected_prefix"].fillna(False)
    )
    unexpected_extension_mask = non_system_file_mask & ~text_file_mask
    directory_mask = inventory["entry_type"].eq("directory")

    report = {
        "total_entries": len(inventory),
        "regular_files": int(regular_file_mask.sum()),
        "non_system_txt_files": int(
            (non_system_file_mask & text_file_mask).sum()
        ),
        "files_with_expected_prefix_and_txt_extension": int(
            expected_candidate_mask.sum()
        ),
        "unexpected_file_extensions": inventory.loc[
            unexpected_extension_mask, "name"
        ].astype(str).tolist(),
        "unexpected_subdirectories": inventory.loc[
            directory_mask, "name"
        ].astype(str).tolist(),
        "system_entries": inventory.loc[system_mask, "name"].astype(str).tolist(),
    }
    return pd.Series(report, dtype="object")


def parse_fcm_filenames(
    inventory: pd.DataFrame,
    *,
    filename_regex: str = DEFAULT_FCM_FILENAME_REGEX,
    canonical_width: int = DEFAULT_CANONICAL_WIDTH,
) -> pd.DataFrame:
    """Parse participant IDs from non-system FCM ``.txt`` candidates.

    The returned copy contains candidate, pattern-match, canonical-ID, and
    filename-validity columns. Malformed candidates remain present for the
    discrepancy audit, while directories and unrelated files are not treated as
    matrices.
    """

    _require_columns(
        inventory,
        INVENTORY_COLUMNS,
        frame_name="inventory",
    )

    result = inventory.copy(deep=True)
    candidate_mask = (
        result["entry_type"].eq("file")
        & result["extension"].eq(".txt")
        & ~result["is_system_entry"].fillna(False)
    )

    extracted_ids = result["name"].astype("string").str.extract(
        filename_regex,
        expand=False,
    )
    if isinstance(extracted_ids, pd.DataFrame):
        raise TypeError("filename_regex must contain exactly one capture group")

    extracted_ids = clean_participant_ids(extracted_ids)
    pattern_match_mask = candidate_mask & extracted_ids.notna()
    participant_ids = extracted_ids.where(pattern_match_mask, pd.NA)
    canonical_ids = canonicalize_participant_ids(
        participant_ids,
        width=canonical_width,
    )
    valid_filename_mask = pattern_match_mask & canonical_ids.notna()

    result["is_candidate_fcm_file"] = candidate_mask.astype(bool)
    result["matches_fcm_pattern"] = pattern_match_mask.astype(bool)
    result["participant_id"] = participant_ids.astype("string")
    result["canonical_id"] = canonical_ids.astype("string")
    result["is_valid_fcm_filename"] = valid_filename_mask.astype(bool)
    return result


def summarize_fcm_filename_validation(parsed_inventory: pd.DataFrame) -> pd.Series:
    """Summarize FCM candidate parsing and canonicalization."""

    required = {
        "is_candidate_fcm_file",
        "matches_fcm_pattern",
        "participant_id",
        "canonical_id",
        "is_valid_fcm_filename",
    }
    _require_columns(
        parsed_inventory,
        required,
        frame_name="parsed_inventory",
    )

    candidate_mask = parsed_inventory["is_candidate_fcm_file"].fillna(False)
    pattern_mask = parsed_inventory["matches_fcm_pattern"].fillna(False)
    valid_mask = parsed_inventory["is_valid_fcm_filename"].fillna(False)

    report = {
        "non_system_txt_files_checked": int(candidate_mask.sum()),
        "filenames_matching_expected_pattern": int(pattern_mask.sum()),
        "valid_fcm_filenames": int(valid_mask.sum()),
        "malformed_or_unparseable_filenames": int(
            (candidate_mask & ~valid_mask).sum()
        ),
        "matrix_ids_normalized": int(
            parsed_inventory.loc[valid_mask, "canonical_id"].notna().sum()
        ),
        "pattern_matches_with_invalid_canonical_id": int(
            (pattern_mask & parsed_inventory["canonical_id"].isna()).sum()
        ),
    }
    return pd.Series(report, dtype="object")


def validate_fcm_id_uniqueness(parsed_inventory: pd.DataFrame) -> pd.Series:
    """Check parsed matrix IDs for duplicates and zero-padding collisions."""

    required = {
        "participant_id",
        "canonical_id",
        "is_valid_fcm_filename",
    }
    _require_columns(
        parsed_inventory,
        required,
        frame_name="parsed_inventory",
    )

    valid_matrix_rows = parsed_inventory.loc[
        parsed_inventory["is_valid_fcm_filename"].fillna(False)
    ]
    parsed_ids = clean_participant_ids(valid_matrix_rows["participant_id"]).dropna()
    canonical_ids = clean_participant_ids(valid_matrix_rows["canonical_id"]).dropna()

    numeric_ids = parsed_ids.map(int)
    report = {
        "valid_fcm_files": len(valid_matrix_rows),
        "parsed_matrix_ids": len(parsed_ids),
        "unique_parsed_matrix_ids": int(parsed_ids.nunique()),
        "duplicate_filename_ids": int(parsed_ids.duplicated().sum()),
        "canonical_collision_count": int(
            parsed_ids.nunique() - numeric_ids.nunique()
        ),
        "duplicate_canonical_ids": int(canonical_ids.duplicated().sum()),
    }
    return pd.Series(report, dtype="object")


def compare_participant_id_sets(
    cohort_metadata: pd.DataFrame,
    parsed_inventory: pd.DataFrame,
    *,
    metadata_id_column: str = "sub_id",
    canonical_width: int = DEFAULT_CANONICAL_WIDTH,
) -> pd.Series:
    """Compare unique canonical IDs in cohort metadata and valid FCM files."""

    _require_columns(
        cohort_metadata,
        {metadata_id_column},
        frame_name="cohort_metadata",
    )
    _require_columns(
        parsed_inventory,
        {"canonical_id", "is_valid_fcm_filename"},
        frame_name="parsed_inventory",
    )

    metadata_raw_ids = clean_participant_ids(cohort_metadata[metadata_id_column])
    metadata_canonical_ids = canonicalize_participant_ids(
        metadata_raw_ids,
        width=canonical_width,
    )
    matrix_canonical_ids = clean_participant_ids(
        parsed_inventory.loc[
            parsed_inventory["is_valid_fcm_filename"].fillna(False),
            "canonical_id",
        ]
    )

    metadata_set = set(metadata_canonical_ids.dropna().astype(str))
    matrix_set = set(matrix_canonical_ids.dropna().astype(str))

    report = {
        "metadata_rows": len(cohort_metadata),
        "metadata_ids_not_canonicalized": int(metadata_canonical_ids.isna().sum()),
        "unique_metadata_ids": len(metadata_set),
        "unique_matrix_ids": len(matrix_set),
        "metadata_without_matrices": len(metadata_set - matrix_set),
        "matrices_without_metadata": len(matrix_set - metadata_set),
        "matched_participants": len(metadata_set & matrix_set),
    }
    return pd.Series(report, dtype="object")


def _classify_alignment_row(row: pd.Series) -> str:
    metadata_count = int(row["metadata_row_count"])
    matrix_count = int(row["matrix_file_count"])

    if metadata_count > 1 and matrix_count > 1:
        return "duplicate_metadata_and_matrix"
    if metadata_count > 1:
        return "duplicate_metadata"
    if matrix_count > 1:
        return "duplicate_matrix"
    if metadata_count == 1 and matrix_count == 1:
        return "matched"
    if metadata_count == 1 and matrix_count == 0:
        return "metadata_only"
    if metadata_count == 0 and matrix_count == 1:
        return "matrix_only"
    return "anomaly"


def build_alignment_table(
    cohort_metadata: pd.DataFrame,
    parsed_inventory: pd.DataFrame,
    *,
    metadata_id_column: str = "sub_id",
    canonical_width: int = DEFAULT_CANONICAL_WIDTH,
) -> pd.DataFrame:
    """Build a participant-level metadata-to-FCM alignment table.

    Only non-system ``.txt`` files are considered matrix candidates. Malformed
    candidates and invalid metadata IDs are retained as explicit discrepancy
    rows. The function never writes the participant-level table to disk.
    """

    _require_columns(
        cohort_metadata,
        {metadata_id_column},
        frame_name="cohort_metadata",
    )
    _require_columns(
        parsed_inventory,
        {
            "name",
            "path",
            "is_candidate_fcm_file",
            "matches_fcm_pattern",
            "participant_id",
            "canonical_id",
            "is_valid_fcm_filename",
        },
        frame_name="parsed_inventory",
    )

    metadata_raw_ids = clean_participant_ids(cohort_metadata[metadata_id_column])
    metadata_canonical_ids = canonicalize_participant_ids(
        metadata_raw_ids,
        width=canonical_width,
    )
    metadata_records = pd.DataFrame(
        {
            "metadata_subject_id": metadata_raw_ids,
            "canonical_id": metadata_canonical_ids,
        },
        index=cohort_metadata.index,
    )

    valid_metadata = metadata_records.loc[
        metadata_records["canonical_id"].notna()
    ]
    metadata_groups = (
        valid_metadata.groupby("canonical_id", as_index=False)
        .agg(
            metadata_subject_id=("metadata_subject_id", _join_unique_text),
            metadata_row_count=("canonical_id", "size"),
        )
    )

    valid_file_mask = parsed_inventory["is_valid_fcm_filename"].fillna(False)
    valid_files = parsed_inventory.loc[valid_file_mask]
    file_groups = (
        valid_files.groupby("canonical_id", as_index=False)
        .agg(
            matrix_subject_id=("participant_id", _join_unique_text),
            matrix_file_count=("name", "size"),
            matrix_filename=("name", _join_unique_text),
            matrix_path=("path", _join_unique_text),
            filename_valid=("is_valid_fcm_filename", "all"),
        )
    )

    aligned = metadata_groups.merge(
        file_groups,
        on="canonical_id",
        how="outer",
        validate="one_to_one",
    )
    aligned["metadata_row_count"] = (
        aligned["metadata_row_count"].fillna(0).astype(int)
    )
    aligned["matrix_file_count"] = (
        aligned["matrix_file_count"].fillna(0).astype(int)
    )
    aligned["metadata_present"] = aligned["metadata_row_count"].gt(0)
    aligned["matrix_present"] = aligned["matrix_file_count"].gt(0)
    aligned["filename_valid"] = aligned["filename_valid"].fillna(False).astype(bool)
    aligned["original_subject_id"] = aligned["metadata_subject_id"].combine_first(
        aligned["matrix_subject_id"]
    )
    aligned["alignment_status"] = aligned.apply(
        _classify_alignment_row,
        axis=1,
    )

    candidate_mask = parsed_inventory["is_candidate_fcm_file"].fillna(False)
    invalid_file_rows = parsed_inventory.loc[candidate_mask & ~valid_file_mask]
    invalid_files = pd.DataFrame(
        {
            "original_subject_id": invalid_file_rows["participant_id"],
            "canonical_id": invalid_file_rows["canonical_id"],
            "metadata_present": False,
            "metadata_row_count": 0,
            "matrix_present": True,
            "matrix_file_count": 1,
            "matrix_filename": invalid_file_rows["name"].astype("string"),
            "matrix_path": invalid_file_rows["path"].astype("string"),
            "filename_valid": False,
            "alignment_status": invalid_file_rows["matches_fcm_pattern"].map(
                {True: "invalid_matrix_id", False: "invalid_filename"}
            ),
        },
        index=invalid_file_rows.index,
    )

    invalid_metadata_rows = metadata_records.loc[
        metadata_records["canonical_id"].isna()
    ]
    invalid_metadata = pd.DataFrame(
        {
            "original_subject_id": invalid_metadata_rows["metadata_subject_id"],
            "canonical_id": pd.Series(
                pd.NA,
                index=invalid_metadata_rows.index,
                dtype="string",
            ),
            "metadata_present": True,
            "metadata_row_count": 1,
            "matrix_present": False,
            "matrix_file_count": 0,
            "matrix_filename": pd.Series(
                pd.NA,
                index=invalid_metadata_rows.index,
                dtype="string",
            ),
            "matrix_path": pd.Series(
                pd.NA,
                index=invalid_metadata_rows.index,
                dtype="string",
            ),
            "filename_valid": False,
            "alignment_status": invalid_metadata_rows["metadata_subject_id"].map(
                lambda value: "missing_metadata_id"
                if pd.isna(value)
                else "invalid_metadata_id"
            ),
        },
        index=invalid_metadata_rows.index,
    )

    aligned = aligned.drop(
        columns=["metadata_subject_id", "matrix_subject_id"],
        errors="ignore",
    )
    alignment_table = pd.concat(
        [aligned[ALIGNMENT_COLUMNS], invalid_files, invalid_metadata],
        ignore_index=True,
    )
    alignment_table = alignment_table[ALIGNMENT_COLUMNS].sort_values(
        ["canonical_id", "alignment_status", "matrix_filename"],
        kind="stable",
        na_position="last",
    )
    return alignment_table.reset_index(drop=True)


def validate_alignment_cardinality(alignment_table: pd.DataFrame) -> pd.Series:
    """Report whether metadata rows and matrix files form one-to-one matches."""

    _require_columns(
        alignment_table,
        ALIGNMENT_COLUMNS,
        frame_name="alignment_table",
    )

    metadata_rows = alignment_table.loc[alignment_table["metadata_present"]]
    matrix_rows = alignment_table.loc[alignment_table["matrix_present"]]
    duplicate_statuses = {
        "duplicate_metadata",
        "duplicate_matrix",
        "duplicate_metadata_and_matrix",
    }

    report = {
        "metadata_row_count_equals_one_for_all": bool(
            metadata_rows["metadata_row_count"].eq(1).all()
        ),
        "matrix_file_count_equals_one_for_all": bool(
            matrix_rows["matrix_file_count"].eq(1).all()
        ),
        "perfect_one_to_one_matches": int(
            alignment_table["alignment_status"].eq("matched").sum()
        ),
        "cardinality_violations": int(
            alignment_table["alignment_status"].isin(duplicate_statuses).sum()
        ),
    }
    return pd.Series(report, dtype="object")


def summarize_alignment(alignment_table: pd.DataFrame) -> pd.Series:
    """Return an aggregate, non-sensitive alignment summary."""

    _require_columns(
        alignment_table,
        ALIGNMENT_COLUMNS,
        frame_name="alignment_table",
    )

    status_counts = alignment_table["alignment_status"].value_counts()
    discrepancy_mask = ~alignment_table["alignment_status"].eq("matched")

    report = {
        "matched_participants": int(status_counts.get("matched", 0)),
        "metadata_without_matrices": int(status_counts.get("metadata_only", 0)),
        "matrices_without_metadata": int(status_counts.get("matrix_only", 0)),
        "duplicate_metadata_ids": int(status_counts.get("duplicate_metadata", 0)),
        "duplicate_matrix_ids": int(status_counts.get("duplicate_matrix", 0)),
        "duplicate_metadata_and_matrix_ids": int(
            status_counts.get("duplicate_metadata_and_matrix", 0)
        ),
        "invalid_filenames": int(status_counts.get("invalid_filename", 0)),
        "invalid_matrix_ids": int(status_counts.get("invalid_matrix_id", 0)),
        "invalid_metadata_ids": int(status_counts.get("invalid_metadata_id", 0)),
        "missing_metadata_ids": int(status_counts.get("missing_metadata_id", 0)),
        "anomalies": int(status_counts.get("anomaly", 0)),
        "total_discrepancies": int(discrepancy_mask.sum()),
        "alignment_passed": bool(
            len(alignment_table) > 0 and not discrepancy_mask.any()
        ),
    }
    return pd.Series(report, dtype="object")


def align_participants_to_fcm(
    cohort_metadata: pd.DataFrame,
    fcm_directory: str | Path,
    *,
    metadata_id_column: str = "sub_id",
    expected_prefix: str = DEFAULT_FCM_PREFIX,
    filename_regex: str = DEFAULT_FCM_FILENAME_REGEX,
    canonical_width: int = DEFAULT_CANONICAL_WIDTH,
) -> FCMAlignmentResult:
    """Run the complete reusable participant-to-FCM alignment workflow."""

    inventory = inventory_fcm_directory(
        fcm_directory,
        expected_prefix=expected_prefix,
    )
    parsed_inventory = parse_fcm_filenames(
        inventory,
        filename_regex=filename_regex,
        canonical_width=canonical_width,
    )
    alignment_table = build_alignment_table(
        cohort_metadata,
        parsed_inventory,
        metadata_id_column=metadata_id_column,
        canonical_width=canonical_width,
    )

    return FCMAlignmentResult(
        inventory=parsed_inventory,
        directory_report=summarize_fcm_directory(parsed_inventory),
        filename_report=summarize_fcm_filename_validation(parsed_inventory),
        uniqueness_report=validate_fcm_id_uniqueness(parsed_inventory),
        set_report=compare_participant_id_sets(
            cohort_metadata,
            parsed_inventory,
            metadata_id_column=metadata_id_column,
            canonical_width=canonical_width,
        ),
        alignment_table=alignment_table,
        cardinality_report=validate_alignment_cardinality(alignment_table),
        alignment_summary=summarize_alignment(alignment_table),
    )
