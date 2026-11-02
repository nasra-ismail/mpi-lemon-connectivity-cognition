import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from lemon_connectivity.identifiers import (
    canonicalize_participant_ids,
    validate_canonical_participant_ids,
)
from lemon_connectivity.io import (
    edge_list_to_matrix,
    load_fcm_edge_list,
    parse_subject_id,
)
from lemon_connectivity.networks import (
    SEGREGATION_DEFINITIONS,
    calculate_network_statistics,
    create_network_masks,
)

PRIMARY_DEFINITION = "seg_z_positive"
EDGE_POLICY_VERSION = "edge-policy-v1"
AGE_GROUP_FORMULA = (
    "system_segregation ~ C(cohort, Treatment(reference='young')) + "
    "C(sex, Treatment(reference='female'))"
)
AGE_GROUP_TERM = "C(cohort, Treatment(reference='young'))[T.elderly]"
RESULT_COLUMNS = [
    "within_mean",
    "between_mean",
    "system_segregation",
    "within_edge_count",
    "between_edge_count",
    "calculation_status",
    "calculation_reason",
]
CALCULATION_STATUSES = {
    "calculated",
    "undefined_within_mean_negative",
    "undefined_within_mean_zero",
    "undefined_within_mean_near_zero",
    "load_failed",
    "matrix_reconstruction_failed",
    "calculation_failed",
}


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{name} contains duplicate column names")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _require_unique_ids(frame: pd.DataFrame, name: str) -> None:
    _require_columns(frame, ["canonical_id"], name)
    report = validate_canonical_participant_ids(frame["canonical_id"])
    if not report["all_match_canonical_format"] or report["duplicate_canonical_ids"]:
        raise ValueError(f"{name} requires valid, nonmissing, unique canonical IDs")
    ids = frame["canonical_id"].astype("string")
    if not ids.eq(ids.str.strip()).all():
        raise ValueError(f"{name} contains whitespace around canonical IDs")


def select_sensitivity_inputs(
    primary_results: pd.DataFrame,
    alignment_table: pd.DataFrame,
    fcm_directory: str | Path,
) -> pd.DataFrame:
    """Recover the fixed notebook-07 QC inventory without silently reselecting it.

    Resolve filenames against the local FCM directory, rather than reusing
    machine-specific absolute paths saved in an older alignment CSV. Validate
    each filename-to-ID association with the existing project ID utilities.
    """
    _require_columns(primary_results, RESULT_COLUMNS, "primary_results")
    _require_unique_ids(primary_results, "primary_results")
    _require_columns(
        alignment_table, ["matrix_filename", "alignment_status"], "alignment_table"
    )
    _require_unique_ids(alignment_table, "alignment_table")
    if primary_results.empty:
        raise ValueError("The saved primary QC inventory is empty")
    # Match notebook 08: unresolved alignment anomalies stop the comparison.
    if not alignment_table["alignment_status"].eq("matched").fillna(False).all():
        raise ValueError(
            "Resolve nonmatched alignment records before sensitivity analysis"
        )
    inputs = primary_results[["canonical_id"]].merge(
        alignment_table[["canonical_id", "matrix_filename"]],
        on="canonical_id",
        how="left",
        validate="one_to_one",
    )
    if inputs["matrix_filename"].isna().any():
        raise ValueError("Some saved primary participants lack an aligned FCM filename")
    if inputs["matrix_filename"].duplicated().any():
        raise ValueError("Aligned FCM filenames must be unique")
    names = inputs["matrix_filename"]
    if not names.map(
        lambda name: isinstance(name, str) and "/" not in name and "\\" not in name
    ).all():
        raise ValueError("matrix_filename must contain filenames, not paths")
    parsed_ids = canonicalize_participant_ids(names.map(parse_subject_id))
    if not parsed_ids.eq(inputs["canonical_id"]).fillna(False).all():
        raise ValueError(
            "Aligned filenames do not match their canonical participant IDs"
        )
    root = Path(fcm_directory).resolve()
    inputs["matrix_path"] = names.map(lambda name: str(root / name))
    return inputs.sort_values("canonical_id", kind="stable").reset_index(drop=True)


def sha256_file(path: str | Path) -> str:
    """Hash the exact bytes of a source or generated file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fcm_manifest_checksum(
    participant_files: pd.DataFrame, repo_root: str | Path
) -> str:
    """Reproduce notebook 07's sorted relative-path/tab/SHA-256 manifest hash."""
    root = Path(repo_root).resolve()
    paths = sorted(Path(path).resolve() for path in participant_files["matrix_path"])
    manifest = "\n".join(
        path.relative_to(root).as_posix() + "\t" + sha256_file(path) for path in paths
    )
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def verify_primary_inputs(
    provenance: pd.DataFrame,
    participant_files: pd.DataFrame,
    atlas_path: str | Path,
    repo_root: str | Path,
) -> dict[str, str]:
    """Stop if the saved primary policy, inventory count, atlas or FCM bytes differ."""
    expected = {
        "edge_policy_version": EDGE_POLICY_VERSION,
        "segregation_definition": PRIMARY_DEFINITION,
        "within_mean_denominator_rule": "W <= 1e-12 is undefined",
        "n_rois": 200,
        "n_retained_participants": len(participant_files),
        "n_processed_fcm_files": len(participant_files),
    }
    _require_columns(
        provenance,
        [*expected, "atlas_sha256", "fcm_manifest_sha256"],
        "primary_provenance",
    )
    if len(provenance) != 1:
        raise ValueError("Expected exactly one primary provenance record")
    saved = provenance.iloc[0]
    for key, value in expected.items():
        if pd.isna(saved[key]) or saved[key] != value:
            raise ValueError(f"Saved primary provenance does not match: {key}")
    hashes = {
        "atlas_sha256": sha256_file(atlas_path),
        "fcm_manifest_sha256": fcm_manifest_checksum(participant_files, repo_root),
    }
    for key, value in hashes.items():
        if pd.isna(saved[key]) or saved[key] != value:
            raise ValueError(f"Inputs changed since notebook 07: {key}")
    return hashes


def calculate_segregation_sensitivities(
    participant_files: pd.DataFrame,
    network_labels: Sequence[str],
) -> pd.DataFrame:
    """Evaluate every retained file under all four definitions, loading it once.

    Keep one row for every participant-definition pair, including failures.
    Each call starts from the same original signed R. Existing functions supply
    loading, reconstruction, masks, pooled means, and the denominator policy.
    """
    _require_columns(participant_files, ["matrix_path"], "participant_files")
    _require_unique_ids(participant_files, "participant_files")
    if participant_files.empty or participant_files["matrix_path"].isna().any():
        raise ValueError("A nonempty inventory with complete matrix paths is required")
    paths = participant_files["matrix_path"].map(lambda path: Path(path).resolve())
    if paths.duplicated().any():
        raise ValueError("The same matrix path cannot represent multiple participants")
    within_mask, between_mask = create_network_masks(network_labels)
    if not within_mask.any() or not between_mask.any():
        raise ValueError("Both within- and between-network masks must contain edges")
    n_rois = len(network_labels)
    n_edges = n_rois * (n_rois - 1) // 2
    records = []
    for participant in participant_files.sort_values("canonical_id").itertuples():
        stage = "load_failed"
        try:
            edges = load_fcm_edge_list(participant.matrix_path, n_rois=n_rois)
            stage = "matrix_reconstruction_failed"
            matrix = edge_list_to_matrix(edges, n_rois=n_rois)
        except (OSError, ValueError) as error:
            input_status, input_reason = stage, f"{type(error).__name__}: {error}"
            matrix = None
        else:
            input_status, input_reason = "", ""

        positive_count = (
            int((matrix[np.triu_indices(n_rois, k=1)] > 0).sum())
            if matrix is not None
            else np.nan
        )
        for definition in SEGREGATION_DEFINITIONS:
            record = {
                "canonical_id": participant.canonical_id,
                "definition": definition,
                "edge_policy_version": EDGE_POLICY_VERSION,
                "within_mean": np.nan,
                "between_mean": np.nan,
                "system_segregation": np.nan,
                "within_edge_count": int(within_mask.sum()),
                "between_edge_count": int(between_mask.sum()),
                "calculation_status": input_status,
                "calculation_reason": input_reason,
                "positive_edge_count": positive_count,
                "positive_edge_density": positive_count / n_edges,
            }
            if matrix is not None:
                try:
                    statistics = calculate_network_statistics(
                        matrix, network_labels, definition=definition
                    )
                except ValueError as error:
                    record["calculation_status"] = "calculation_failed"
                    record["calculation_reason"] = f"{type(error).__name__}: {error}"
                else:
                    record.update(statistics)
            records.append(record)
    ledger = pd.DataFrame(records)
    _validate_ledger(ledger)
    return ledger


def _validate_ledger(ledger: pd.DataFrame) -> None:
    _require_columns(
        ledger,
        ["canonical_id", "definition", "edge_policy_version", *RESULT_COLUMNS],
        "sensitivity_ledger",
    )
    _require_unique_ids(ledger[["canonical_id"]].drop_duplicates(), "ledger inventory")
    if ledger.empty or ledger.duplicated(["canonical_id", "definition"]).any():
        raise ValueError("Expected one row per participant-definition pair")
    if set(ledger["definition"]) != set(SEGREGATION_DEFINITIONS):
        raise ValueError("Ledger must contain all four prespecified definitions")
    if not ledger.groupby("canonical_id").size().eq(4).all():
        raise ValueError("Each participant must have all four definition records")
    if not ledger["edge_policy_version"].eq(EDGE_POLICY_VERSION).fillna(False).all():
        raise ValueError("Unexpected or missing edge policy version")
    if not ledger["calculation_status"].isin(CALCULATION_STATUSES).all():
        raise ValueError("Unexpected or missing calculation status")
    calculated = ledger["calculation_status"].eq("calculated")
    finite_score = np.isfinite(ledger["system_segregation"].to_numpy(dtype=float))
    if not np.array_equal(calculated.to_numpy(), finite_score):
        raise ValueError(
            "Only calculated records may contain a finite segregation score"
        )
    if not ledger.loc[~calculated, "system_segregation"].isna().all():
        raise ValueError("Unavailable scores must be NaN, not infinite")


def verify_primary_replication(
    ledger: pd.DataFrame,
    saved_primary: pd.DataFrame,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Match primary IDs, statuses, counts and W/B/S before sensitivity inference.

    NaN is compared as NaN. Statuses and edge counts must match exactly; numeric
    tolerances allow TSV round-trip differences, not a new eligibility rule.
    Error-message prose is intentionally not used as a reproducibility key.
    """
    _validate_ledger(ledger)
    _require_columns(saved_primary, RESULT_COLUMNS, "saved_primary")
    _require_unique_ids(saved_primary, "saved_primary")
    primary = (
        ledger.loc[ledger["definition"].eq(PRIMARY_DEFINITION)]
        .set_index("canonical_id")
        .sort_index()
    )
    saved = saved_primary.set_index("canonical_id").sort_index()
    if set(primary.index) != set(saved.index):
        raise ValueError("Primary participant inventory changed")
    for column in ["calculation_status", "within_edge_count", "between_edge_count"]:
        if not primary[column].eq(saved[column]).fillna(False).all():
            raise ValueError(f"Primary replication failed: {column}")
    checks = []
    for column in ["within_mean", "between_mean", "system_segregation"]:
        current = primary[column].to_numpy(dtype=float)
        previous = saved[column].to_numpy(dtype=float)
        if not np.allclose(current, previous, rtol=rtol, atol=atol, equal_nan=True):
            raise ValueError(f"Primary replication failed: {column}")
        finite = np.isfinite(current) & np.isfinite(previous)
        maximum = (
            float(np.max(np.abs(current[finite] - previous[finite])))
            if finite.any()
            else np.nan
        )
        checks.append(
            {"measure": column, "matches": True, "max_absolute_difference": maximum}
        )
    return pd.DataFrame(checks)


def prepare_sensitivity_analysis(
    ledger: pd.DataFrame,
    cohort_metadata: pd.DataFrame,
    alignment_table: pd.DataFrame,
) -> pd.DataFrame:
    """Attach metadata and flag both per-definition and common eligibility.

    Use the same cohort, recorded-sex, calculated/finite-score and matched-
    alignment criteria as notebook 08. No participant is silently dropped.
    Missing/unknown cohort or sex are flagged, never imputed or recoded.
    """
    _validate_ledger(ledger)
    _require_columns(cohort_metadata, ["cohort", "sex"], "cohort_metadata")
    _require_unique_ids(cohort_metadata, "cohort_metadata")
    _require_columns(alignment_table, ["alignment_status"], "alignment_table")
    _require_unique_ids(alignment_table, "alignment_table")
    inventory = set(ledger["canonical_id"])
    for name, table in [
        ("cohort metadata", cohort_metadata),
        ("alignment", alignment_table),
    ]:
        if not inventory.issubset(set(table["canonical_id"])):
            raise ValueError(f"Sensitivity participants are missing from {name}")
    if not alignment_table["alignment_status"].eq("matched").fillna(False).all():
        raise ValueError(
            "Resolve nonmatched alignment records before sensitivity analysis"
        )
    analysis = ledger.merge(
        cohort_metadata[["canonical_id", "cohort", "sex"]],
        on="canonical_id",
        how="left",
        validate="many_to_one",
    ).merge(
        alignment_table[["canonical_id", "alignment_status"]],
        on="canonical_id",
        how="left",
        validate="many_to_one",
    )
    analysis["cohort_eligible"] = analysis["cohort"].isin(["young", "elderly"])
    analysis["sex_eligible"] = analysis["sex"].isin(["female", "male"])
    analysis["score_eligible"] = analysis["calculation_status"].eq(
        "calculated"
    ) & np.isfinite(analysis["system_segregation"].to_numpy(dtype=float))
    analysis["eligible_for_definition"] = analysis[
        ["cohort_eligible", "sex_eligible", "score_eligible"]
    ].all(axis=1)
    analysis["in_common_sample"] = analysis.groupby("canonical_id")[
        "eligible_for_definition"
    ].transform("all")
    return analysis


def summarize_sensitivity_eligibility(analysis: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate counts; individual failure criteria may overlap."""
    records = []
    for definition in SEGREGATION_DEFINITIONS:
        rows = analysis.loc[analysis["definition"].eq(definition)]
        eligible = rows["eligible_for_definition"]
        common = rows["in_common_sample"]
        records.append(
            {
                "definition": definition,
                "attempted": len(rows),
                "calculated": int(rows["calculation_status"].eq("calculated").sum()),
                "unavailable_score": int((~rows["score_eligible"]).sum()),
                "invalid_or_missing_cohort": int((~rows["cohort_eligible"]).sum()),
                "invalid_or_missing_sex": int((~rows["sex_eligible"]).sum()),
                "eligible_for_definition": int(eligible.sum()),
                "common_sample": int(common.sum()),
                "eligible_but_not_common": int((eligible & ~common).sum()),
                "common_young": int((common & rows["cohort"].eq("young")).sum()),
                "common_elderly": int((common & rows["cohort"].eq("elderly")).sum()),
            }
        )
    return pd.DataFrame(records)


def fit_age_group_comparison(analysis_rows: pd.DataFrame) -> dict[str, float | int]:
    """Fit notebook 08's sex-adjusted elderly-minus-young OLS/HC3/t model.

    No automatic missing-value deletion, covariate selection, clipping or
    outlier removal is allowed. Reject nonestimable or nonfinite inference.
    """
    _require_columns(
        analysis_rows, ["cohort", "sex", "system_segregation"], "model data"
    )
    _require_unique_ids(analysis_rows, "model data")
    data = analysis_rows[["cohort", "sex", "system_segregation"]].copy()
    if data.isna().any().any() or not np.isfinite(data["system_segregation"]).all():
        raise ValueError("Model data must be complete with finite scores")
    counts = data["cohort"].value_counts()
    if set(counts.index) != {"young", "elderly"} or counts.min() < 2:
        raise ValueError(
            "Each age group must contain at least two eligible participants"
        )
    if set(data["sex"]) != {"female", "male"}:
        raise ValueError(
            "Both recorded sex categories are needed for the planned model"
        )
    # Object strings also support older Patsy/statsmodels with pandas StringDtype.
    data["cohort"] = data["cohort"].astype(object)
    data["sex"] = data["sex"].astype(object)
    model = smf.ols(AGE_GROUP_FORMULA, data=data, missing="raise")
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Cohort and sex effects are not separately estimable")
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required")
    q, _ = np.linalg.qr(design, mode="reduced")
    if np.any(np.sum(q * q, axis=1) >= 1 - 1e-12):
        raise ValueError("HC3 is undefined for observations with unit leverage")
    fitted = model.fit(cov_type="HC3", use_t=True)
    if int(fitted.nobs) != len(data):
        raise ValueError("The fitted model changed the eligible sample")
    ci = fitted.conf_int(alpha=0.05).loc[AGE_GROUP_TERM]
    results = {
        "adjusted_difference": float(fitted.params[AGE_GROUP_TERM]),
        "hc3_robust_se": float(fitted.bse[AGE_GROUP_TERM]),
        "ci_95_lower": float(ci.iloc[0]),
        "ci_95_upper": float(ci.iloc[1]),
        "p_value_two_sided": float(fitted.pvalues[AGE_GROUP_TERM]),
        "residual_df": float(fitted.df_resid),
    }
    if not np.isfinite(list(results.values())).all():
        raise ValueError("The model returned nonfinite estimates or uncertainty")
    return results


def compare_sensitivity_models(analysis: pd.DataFrame) -> pd.DataFrame:
    """Report all four common-sample models and a separate full-primary reference.

    Unestimable models retain their table row and reason. The reference uses
    all primary-eligible participants to distinguish definition changes from
    common-sample selection. Its estimate is not used to select a definition.
    """
    _validate_ledger(analysis)
    _require_columns(
        analysis,
        ["cohort", "sex", "eligible_for_definition", "in_common_sample"],
        "analysis",
    )
    common_sets = [
        set(
            analysis.loc[
                analysis["definition"].eq(definition) & analysis["in_common_sample"],
                "canonical_id",
            ]
        )
        for definition in SEGREGATION_DEFINITIONS
    ]
    if any(ids != common_sets[0] for ids in common_sets[1:]):
        raise ValueError("Common-sample participants differ across definitions")
    if (analysis["in_common_sample"] & ~analysis["eligible_for_definition"]).any():
        raise ValueError("The common sample contains ineligible records")
    specifications = [(definition, "common") for definition in SEGREGATION_DEFINITIONS]
    specifications.append((PRIMARY_DEFINITION, "primary_full_reference"))
    records = []
    for definition, sample in specifications:
        eligibility = (
            "in_common_sample" if sample == "common" else "eligible_for_definition"
        )
        rows = analysis.loc[
            analysis["definition"].eq(definition) & analysis[eligibility]
        ]
        record = {
            "definition": definition,
            "sample": sample,
            "role": "primary" if definition == PRIMARY_DEFINITION else "sensitivity",
            "edge_policy_version": EDGE_POLICY_VERSION,
            "comparison": "elderly_minus_young_adjusted_for_recorded_sex",
            "n_total": len(rows),
            "n_young": int(rows["cohort"].eq("young").sum()),
            "n_elderly": int(rows["cohort"].eq("elderly").sum()),
            "adjusted_difference": np.nan,
            "hc3_robust_se": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
            "p_value_two_sided": np.nan,
            "residual_df": np.nan,
            "model_status": "estimated",
            "model_reason": "",
        }
        try:
            record.update(fit_age_group_comparison(rows))
        except ValueError as error:
            record["model_status"] = "not_estimable"
            record["model_reason"] = str(error)
        records.append(record)
    return pd.DataFrame(records)
