import numpy as np
import pandas as pd
import pytest

from lemon_connectivity.networks import (
    create_network_permutation,
    reorder_connectivity_matrix,
)


# Test that shuffled atlas rows still produce sorted ROI indices within networks
def test_create_network_permutation_sorts_roi_indices():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [2, 0, 5, 3, 1, 4],
            "canonical_network": [
                "Visual",
                "Visual",
                "Default",
                "Control",
                "Visual",
                "Default",
            ],
        }
    )

    canonical_network_order = [
        "Visual",
        "Control",
        "Default",
    ]

    permutation = create_network_permutation(
        atlas_labels,
        canonical_network_order,
    )

    assert permutation == [0, 1, 2, 3, 4, 5]


# Test that the same permutation is applied to matrix rows and columns
def test_reorder_connectivity_matrix():
    matrix = np.array(
        [
            [0, 1, 2],
            [1, 3, 4],
            [2, 4, 5],
        ]
    )

    permutation = [2, 0, 1]

    reordered = reorder_connectivity_matrix(
        matrix,
        permutation,
    )

    expected = np.array(
        [
            [5, 2, 4],
            [2, 0, 1],
            [4, 1, 3],
        ]
    )

    assert np.array_equal(reordered, expected)


# Test that required atlas columns are validated
def test_missing_required_atlas_columns():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required atlas columns",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual"],
        )


# Test that duplicate ROI indices are rejected
def test_duplicate_roi_indices():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, 0],
            "canonical_network": ["Visual", "Visual"],
        }
    )

    with pytest.raises(
        ValueError,
        match="ROI indices must be unique",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual"],
        )


# Test that fractional ROI indices are rejected
def test_fractional_roi_indices():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, 1.5],
            "canonical_network": ["Visual", "Visual"],
        }
    )

    with pytest.raises(
        ValueError,
        match="ROI indices must be integers",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual"],
        )


# Test that string ROI indices are rejected
def test_string_roi_indices():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, "1"],
            "canonical_network": ["Visual", "Visual"],
        }
    )

    with pytest.raises(
        ValueError,
        match="ROI indices must be integers",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual"],
        )


# Test that duplicate canonical network names are rejected
def test_duplicate_canonical_network_names():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, 1],
            "canonical_network": ["Visual", "Default"],
        }
    )

    with pytest.raises(
        ValueError,
        match="canonical_network_order contains duplicate networks",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual", "Visual", "Default"],
        )


# Test that missing requested networks are rejected
def test_missing_requested_networks():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, 1],
            "canonical_network": ["Visual", "Default"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Requested networks are missing",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual", "Control", "Default"],
        )


# Test that unexpected atlas networks are rejected
def test_unexpected_atlas_networks():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, 1],
            "canonical_network": ["Visual", "Limbic"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Atlas contains networks absent",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual"],
        )


# Test that missing network labels are rejected
def test_missing_network_labels():
    atlas_labels = pd.DataFrame(
        {
            "roi_index": [0, 1],
            "canonical_network": ["Visual", None],
        }
    )

    with pytest.raises(
        ValueError,
        match="Network labels must not contain missing values",
    ):
        create_network_permutation(
            atlas_labels,
            ["Visual"],
        )


# Test that non-square matrices are rejected
def test_non_square_matrix():
    matrix = np.ones((2, 3))

    with pytest.raises(
        ValueError,
        match="matrix must be square",
    ):
        reorder_connectivity_matrix(
            matrix,
            [0, 1],
        )


# Test that fractional permutation values are rejected
def test_fractional_permutation_values():
    matrix = np.eye(3)

    with pytest.raises(
        ValueError,
        match="Permutation values must be integers",
    ):
        reorder_connectivity_matrix(
            matrix,
            [0.9, 1.1, 2.0],
        )


# Test that duplicate permutation values are rejected
def test_duplicate_permutation_values():
    matrix = np.eye(3)

    with pytest.raises(
        ValueError,
        match="duplicate indices",
    ):
        reorder_connectivity_matrix(
            matrix,
            [0, 1, 1],
        )


# Test that negative permutation values are rejected
def test_negative_permutation_values():
    matrix = np.eye(3)

    with pytest.raises(
        ValueError,
        match="out-of-range indices",
    ):
        reorder_connectivity_matrix(
            matrix,
            [-1, 0, 1],
        )


# Test that out-of-range permutation values are rejected
def test_out_of_range_permutation_values():
    matrix = np.eye(3)

    with pytest.raises(
        ValueError,
        match="out-of-range indices",
    ):
        reorder_connectivity_matrix(
            matrix,
            [0, 1, 3],
        )


# Test that incorrect permutation lengths are rejected
def test_wrong_permutation_length():
    matrix = np.eye(3)

    with pytest.raises(
        ValueError,
        match="length must match",
    ):
        reorder_connectivity_matrix(
            matrix,
            [0, 1],
        )


# Test that reordering does not modify the original matrix
def test_original_matrix_is_unchanged():
    matrix = np.array(
        [
            [0, 1],
            [1, 2],
        ]
    )

    original_matrix = matrix.copy()

    reordered = reorder_connectivity_matrix(
        matrix,
        [1, 0],
    )

    assert np.array_equal(
        matrix,
        original_matrix,
    )
    assert not np.array_equal(
        reordered,
        matrix,
    )
