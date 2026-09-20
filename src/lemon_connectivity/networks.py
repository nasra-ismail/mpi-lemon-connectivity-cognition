"""Utilities for deterministic network-based connectivity ordering."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


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
