"""Utilities for network-based functional-connectivity analysis."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

SEGREGATION_MIN_WITHIN_MEAN = 1e-12
SEGREGATION_DEFINITIONS = (
    "seg_z_positive",
    "seg_r_positive",
    "seg_z_signed",
    "seg_r_signed",
)


class UndefinedSegregationError(ValueError):
    """A finite within-network mean fails the edge-policy-v1 denominator rule."""

    def __init__(self, message: str, calculation_status: str) -> None:
        super().__init__(message)
        self.calculation_status = calculation_status


# Create a deterministic ROI permutation grouped by canonical network
def create_network_permutation(
    atlas_labels: pd.DataFrame,
    canonical_network_order: Sequence[str],
) -> list[int]:
    """Create a validated deterministic ROI permutation by network."""
    required_columns = {"roi_index", "canonical_network"}

    missing_columns = required_columns - set(atlas_labels.columns)
    if missing_columns:
        raise ValueError(f"Missing required atlas columns: {sorted(missing_columns)}")

    roi_indices = atlas_labels["roi_index"]

    if roi_indices.isna().any():
        raise ValueError("ROI indices must not contain missing values")

    if not roi_indices.map(
        lambda value: (
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
        )
    ).all():
        raise ValueError("ROI indices must be integers")

    if roi_indices.duplicated().any():
        raise ValueError("ROI indices must be unique")

    if atlas_labels["canonical_network"].isna().any():
        raise ValueError("Network labels must not contain missing values")

    if len(set(canonical_network_order)) != len(canonical_network_order):
        raise ValueError("canonical_network_order contains duplicate networks")

    observed_networks = set(atlas_labels["canonical_network"])
    requested_networks = set(canonical_network_order)

    missing_networks = requested_networks - observed_networks
    unexpected_networks = observed_networks - requested_networks

    if missing_networks:
        raise ValueError(
            f"Requested networks are missing from atlas labels: "
            f"{sorted(missing_networks)}"
        )

    if unexpected_networks:
        raise ValueError(
            f"Atlas contains networks absent from canonical order: "
            f"{sorted(unexpected_networks)}"
        )

    permutation = []

    for network in canonical_network_order:
        network_rois = (
            atlas_labels.loc[
                atlas_labels["canonical_network"] == network,
                "roi_index",
            ]
            .sort_values()
            .tolist()
        )

        permutation.extend(network_rois)

    if len(permutation) != len(atlas_labels):
        raise ValueError("Permutation length does not match atlas size")

    if len(set(permutation)) != len(permutation):
        raise ValueError("Permutation contains duplicate ROI indices")

    if set(permutation) != set(atlas_labels["roi_index"]):
        raise ValueError("Permutation does not contain every atlas ROI exactly once")

    return permutation


# Reorder the connectivity matrix using the same permutation for rows and columns
def reorder_connectivity_matrix(
    matrix: np.ndarray,
    permutation: Sequence[int],
) -> np.ndarray:
    """Return a copy of the matrix with rows and columns reordered identically."""
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("Connectivity matrix must be two-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Connectivity matrix must be square")

    permutation_array = np.asarray(permutation)

    if permutation_array.ndim != 1:
        raise ValueError("Permutation must be one-dimensional")

    if len(permutation_array) != matrix.shape[0]:
        raise ValueError("Permutation length must match the matrix dimensions")

    if not np.issubdtype(permutation_array.dtype, np.integer):
        raise ValueError("Permutation values must be integers")

    if len(np.unique(permutation_array)) != len(permutation_array):
        raise ValueError("Permutation contains duplicate indices")

    if np.any(permutation_array < 0) or np.any(permutation_array >= matrix.shape[0]):
        raise ValueError("Permutation contains out-of-range indices")

    expected_indices = np.arange(matrix.shape[0])

    if not np.array_equal(
        np.sort(permutation_array),
        expected_indices,
    ):
        raise ValueError("Permutation must contain every matrix index exactly once")

    return matrix[np.ix_(permutation_array, permutation_array)].copy()


# Create upper-triangle masks for within-network and between-network edges
def create_network_masks(
    network_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Create mutually exclusive within- and between-network masks."""
    labels = np.asarray(network_labels)

    if labels.ndim != 1:
        raise ValueError("network_labels must be a one-dimensional sequence")

    if pd.isna(labels).any():
        raise ValueError("network_labels must not contain missing values")

    if labels.size < 2:
        raise ValueError("network_labels must contain at least two ROIs")

    n_rois = len(labels)

    upper_triangle = np.triu(
        np.ones((n_rois, n_rois), dtype=bool),
        k=1,
    )

    same_network = labels[:, None] == labels[None, :]

    within_mask = upper_triangle & same_network
    between_mask = upper_triangle & ~same_network

    return within_mask, between_mask


# Apply one prespecified definition to an original signed correlation matrix
def preprocess_connectivity_matrix(
    matrix: np.ndarray,
    *,
    definition: str = "seg_z_positive",
) -> np.ndarray:
    """Return a new working matrix under edge-policy-v1.

    ``seg_z_*`` applies Fisher z once to off-diagonal correlations;
    ``seg_r_*`` retains raw correlations. ``*_positive`` replaces negatives
    with zero; ``*_signed`` retains their signs. Every definition sets the
    working diagonal to zero. Pass the original signed R, never a previously
    transformed matrix. The default preserves the primary analysis behavior.

    Fisher definitions reject off-diagonal endpoints +/-1 before truncation.
    Raw definitions allow these endpoints. Neither clips out-of-range input.
    """
    if not isinstance(definition, str):
        raise TypeError("definition must be a string")
    if definition not in SEGREGATION_DEFINITIONS:
        raise ValueError(f"Unknown segregation definition: {definition}")

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Connectivity matrix must be two-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Connectivity matrix must be square")

    if not np.isfinite(matrix).all():
        raise ValueError("Connectivity matrix must contain only finite values")

    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    off_diagonal_values = matrix[off_diagonal]

    use_fisher = definition in {"seg_z_positive", "seg_z_signed"}
    if use_fisher and np.any(np.abs(off_diagonal_values) >= 1):
        raise ValueError(
            "Off-diagonal connectivity values must be strictly between -1 and 1"
        )
    if not use_fisher and np.any(np.abs(off_diagonal_values) > 1):
        raise ValueError("Off-diagonal raw correlations must be between -1 and 1")

    processed_matrix = matrix.copy()

    processed_off_diagonal = (
        np.arctanh(off_diagonal_values) if use_fisher else off_diagonal_values.copy()
    )
    if definition in {"seg_z_positive", "seg_r_positive"}:
        processed_off_diagonal[processed_off_diagonal < 0] = 0

    processed_matrix[off_diagonal] = processed_off_diagonal

    # Set the working-matrix diagonal to zero
    np.fill_diagonal(processed_matrix, 0.0)

    return processed_matrix


# Calculate pooled within-network and between-network connectivity means
def calculate_network_means(
    matrix: np.ndarray,
    within_mask: np.ndarray,
    between_mask: np.ndarray,
) -> tuple[float, float]:
    """Calculate pooled mean connectivity within and between networks."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Connectivity matrix must be square")

    if within_mask.shape != matrix.shape:
        raise ValueError("Within-network mask must match matrix shape")

    if between_mask.shape != matrix.shape:
        raise ValueError("Between-network mask must match matrix shape")

    if within_mask.dtype != np.bool_:
        raise ValueError("Within-network mask must contain Boolean values")

    if between_mask.dtype != np.bool_:
        raise ValueError("Between-network mask must contain Boolean values")

    if not np.isfinite(matrix).all():
        raise ValueError("Connectivity matrix must contain only finite values")

    if np.any(within_mask & between_mask):
        raise ValueError("Within- and between-network masks must not overlap")

    expected_upper_triangle = np.triu(
        np.ones(matrix.shape, dtype=bool),
        k=1,
    )

    if not np.array_equal(
        within_mask | between_mask,
        expected_upper_triangle,
    ):
        raise ValueError(
            "Network masks must partition every upper-triangle edge exactly once"
        )

    within_values = matrix[within_mask]
    between_values = matrix[between_mask]

    if within_values.size == 0:
        raise ValueError("Within-network mask contains no edges")

    if between_values.size == 0:
        raise ValueError("Between-network mask contains no edges")

    within_mean = float(np.mean(within_values))
    between_mean = float(np.mean(between_values))

    return within_mean, between_mean


# Calculate whole-brain system segregation from within- and between-network means
def calculate_system_segregation(
    within_mean: float,
    between_mean: float,
) -> float:
    """Calculate (within - between) / within when within_mean exceeds 1e-12.

    Raise UndefinedSegregationError for finite baselines at or below the
    edge-policy-v1 guard. Non-finite means remain invalid inputs.
    """
    if not np.isfinite(within_mean) or not np.isfinite(between_mean):
        raise ValueError("Network means must be finite")

    if within_mean < 0:
        raise UndefinedSegregationError(
            "System segregation is undefined because within_mean is negative",
            "undefined_within_mean_negative",
        )

    if within_mean == 0:
        raise UndefinedSegregationError(
            "System segregation is undefined because within_mean is zero",
            "undefined_within_mean_zero",
        )

    if within_mean <= SEGREGATION_MIN_WITHIN_MEAN:
        raise UndefinedSegregationError(
            "System segregation is undefined because within_mean is at or below "
            f"{SEGREGATION_MIN_WITHIN_MEAN:g}",
            "undefined_within_mean_near_zero",
        )

    return float((within_mean - between_mean) / within_mean)


# Calculate all network-segregation statistics for one participant
def calculate_network_statistics(
    matrix: np.ndarray,
    network_labels: Sequence[str],
    *,
    definition: str = "seg_z_positive",
) -> dict[str, float | int | str]:
    """Return means, edge counts, segregation, and calculation status.

    If the denominator fails the edge-policy-v1 guard, preserve the computed
    means and counts and return a NaN score with an explicit status and reason.
    Invalid matrices or network labels still raise an error. All definitions
    use the same label-based masks and pooled means, including zero weights.
    Pass the original signed correlation matrix for each definition.
    """
    within_mask, between_mask = create_network_masks(network_labels)
    processed_matrix = preprocess_connectivity_matrix(matrix, definition=definition)

    within_mean, between_mean = calculate_network_means(
        processed_matrix,
        within_mask,
        between_mask,
    )

    try:
        system_segregation = calculate_system_segregation(
            within_mean,
            between_mean,
        )
    except UndefinedSegregationError as error:
        system_segregation = np.nan
        calculation_status = error.calculation_status
        calculation_reason = str(error)
    else:
        calculation_status = "calculated"
        calculation_reason = ""

    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "system_segregation": system_segregation,
        "within_edge_count": int(within_mask.sum()),
        "between_edge_count": int(between_mask.sum()),
        "calculation_status": calculation_status,
        "calculation_reason": calculation_reason,
    }
