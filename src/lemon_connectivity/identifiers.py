import re

import pandas as pd

DEFAULT_CANONICAL_PREFIX = "sub-"
DEFAULT_CANONICAL_WIDTH = 6

__all__ = [
    "DEFAULT_CANONICAL_PREFIX",
    "DEFAULT_CANONICAL_WIDTH",
    "add_canonical_participant_id",
    "canonicalize_participant_ids",
    "clean_participant_ids",
    "validate_canonical_participant_ids",
    "validate_raw_participant_ids",
]


def clean_participant_ids(participant_ids: pd.Series) -> pd.Series:
    """Return whitespace-trimmed participant IDs using pandas string dtype.

    Existing missing values remain missing, and empty or whitespace-only values
    are converted to ``pd.NA``. The returned Series retains the original index
    and name.

    Parameters
    ----------
    participant_ids
        Raw participant identifiers.
    """

    if not isinstance(participant_ids, pd.Series):
        raise TypeError("participant_ids must be a pandas Series")

    cleaned = participant_ids.astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""), pd.NA)


def canonicalize_participant_ids(
    participant_ids: pd.Series,
    *,
    width: int = DEFAULT_CANONICAL_WIDTH,
    prefix: str = DEFAULT_CANONICAL_PREFIX,
) -> pd.Series:
    """Convert raw numeric IDs to canonical identifiers such as ``sub-032301``.

    Only non-missing ASCII-digit identifiers that fit within ``width`` are
    canonicalized. Missing or invalid values become ``pd.NA`` so malformed IDs
    cannot silently enter a merge.

    Parameters
    ----------
    participant_ids
        Raw participant identifiers.
    width
        Number of digits following the prefix.
    prefix
        Text placed before the zero-padded numeric identifier.
    """

    if width < 1:
        raise ValueError("width must be at least 1")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")

    cleaned = clean_participant_ids(participant_ids)
    numeric_mask = cleaned.str.fullmatch(r"[0-9]+", na=False)
    width_mask = cleaned.str.len().le(width).fillna(False)
    valid_mask = numeric_mask & width_mask

    canonical = pd.Series(
        pd.NA,
        index=cleaned.index,
        dtype="string",
        name=cleaned.name,
    )
    canonical.loc[valid_mask] = prefix + cleaned.loc[valid_mask].str.zfill(width)
    return canonical



def add_canonical_participant_id(
    frame: pd.DataFrame,
    *,
    source_column: str = "sub_id",
    output_column: str = "canonical_id",
    width: int = DEFAULT_CANONICAL_WIDTH,
    prefix: str = DEFAULT_CANONICAL_PREFIX,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a canonical participant-ID column."""

    if source_column not in frame.columns:
        raise KeyError(f"Missing participant-ID column: {source_column!r}")

    result = frame.copy(deep=True)
    result[output_column] = canonicalize_participant_ids(
        result[source_column],
        width=width,
        prefix=prefix,
    )
    return result


def validate_raw_participant_ids(
    participant_ids: pd.Series,
    *,
    expected_width: int | None = None,
) -> pd.Series:
    """Summarize missingness, formatting, and uniqueness of raw IDs.

    ``expected_width`` can be set to five for the Yadav cohort audit. When it
    is omitted, any non-empty sequence of ASCII digits is accepted.
    """

    if expected_width is not None and expected_width < 1:
        raise ValueError("expected_width must be at least 1")

    cleaned = clean_participant_ids(participant_ids)
    non_missing_mask = cleaned.notna()
    numeric_mask = cleaned.str.fullmatch(r"[0-9]+", na=False)
   
    if expected_width is None:
        width_mask = non_missing_mask.copy()
        all_expected_width = pd.NA
    else:
        width_mask = cleaned.str.len().eq(expected_width).fillna(False)
        all_expected_width = bool(width_mask.loc[non_missing_mask].all())

    valid_mask = non_missing_mask & numeric_mask & width_mask
    non_missing_ids = cleaned.loc[non_missing_mask]

    report = {
        "total_ids": int(len(cleaned)),
        "missing_ids": int(cleaned.isna().sum()),
        "non_missing_ids": int(non_missing_mask.sum()),
        "all_non_missing_ids_numeric": bool(
            numeric_mask.loc[non_missing_mask].all()
        ),
        "all_non_missing_ids_expected_width": all_expected_width,
        "any_alphabetic_ids": bool(
            cleaned.str.contains(r"[A-Za-z]", regex=True, na=False).any()
        ),
        "invalid_raw_ids": int((non_missing_mask & ~valid_mask).sum()),
        "duplicate_raw_ids": int(non_missing_ids.duplicated().sum()),
    }
    return pd.Series(report, dtype="object")


def validate_canonical_participant_ids(
    canonical_ids: pd.Series,
    *,
    width: int = DEFAULT_CANONICAL_WIDTH,
    prefix: str = DEFAULT_CANONICAL_PREFIX,
) -> pd.Series:
    """Summarize format, missingness, and uniqueness of canonical IDs."""

    if width < 1:
        raise ValueError("width must be at least 1")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")

    cleaned = clean_participant_ids(canonical_ids)
    pattern = rf"{re.escape(prefix)}[0-9]{{{width}}}"
    valid_mask = cleaned.str.fullmatch(pattern, na=False)
    non_missing_ids = cleaned.dropna()

    report = {
        "total_canonical_ids": int(len(cleaned)),
        "valid_canonical_ids": int(valid_mask.sum()),
        "invalid_canonical_ids": int((cleaned.notna() & ~valid_mask).sum()),
        "missing_canonical_ids": int(cleaned.isna().sum()),
        "duplicate_canonical_ids": int(non_missing_ids.duplicated().sum()),
        "all_match_canonical_format": bool(valid_mask.all()),
        "all_have_expected_prefix": bool(
            cleaned.str.startswith(prefix, na=False).all()
        ),
        "all_have_expected_length": bool(
            cleaned.str.len().eq(len(prefix) + width).fillna(False).all()
        ),
    }
    return pd.Series(report, dtype="object")
