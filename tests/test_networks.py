import numpy as np
import pandas as pd
import pytest

from lemon_connectivity.networks import (
    calculate_network_means,
    calculate_network_statistics,
    calculate_system_segregation,
    create_network_masks,
    create_network_permutation,
    preprocess_connectivity_matrix,
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


# Test that network masks correctly classify unique within- and between-network edges
def test_create_network_masks():
    network_labels = np.array(["Network A", "Network A", "Network B", "Network B"])

    within_mask, between_mask = create_network_masks(
        network_labels,
    )

    assert within_mask.sum() == 2
    assert between_mask.sum() == 4

    assert not np.any(np.diag(within_mask))
    assert not np.any(np.diag(between_mask))

    assert not np.any(within_mask & between_mask)

    assert (within_mask | between_mask).sum() == 6


# Test that missing network labels are rejected
def test_create_network_masks_rejects_missing_labels():
    network_labels = np.array(
        ["Network A", None, "Network B"],
        dtype=object,
    )

    with pytest.raises(
        ValueError,
        match="network_labels must not contain missing values",
    ):
        create_network_masks(network_labels)


# Test that a network-label sequence with fewer than two ROIs is rejected
def test_create_network_masks_requires_at_least_two_rois():
    network_labels = np.array(["Network A"])

    with pytest.raises(
        ValueError,
        match="network_labels must contain at least two ROIs",
    ):
        create_network_masks(network_labels)


# Test Fisher preprocessing, diagonal exclusion, and input preservation
def test_preprocess_connectivity_matrix():
    matrix = np.array(
        [
            [1.0, 0.20, -0.20],
            [0.20, 1.0, 0.40],
            [-0.20, 0.40, 1.0],
        ]
    )

    original_matrix = matrix.copy()

    processed = preprocess_connectivity_matrix(matrix)

    assert processed.shape == matrix.shape
    assert np.allclose(processed, processed.T)

    assert np.isclose(processed[0, 1], np.arctanh(0.20))
    assert np.isclose(processed[1, 2], np.arctanh(0.40))
    assert processed[0, 2] == 0.0
    assert np.array_equal(np.diag(processed), np.zeros(3))

    assert np.array_equal(matrix, original_matrix)


# Test that a positive off-diagonal value of one is rejected
def test_preprocess_rejects_positive_one():
    matrix = np.eye(3)

    matrix[0, 1] = 1.0
    matrix[1, 0] = 1.0

    with pytest.raises(
        ValueError,
        match="strictly between -1 and 1",
    ):
        preprocess_connectivity_matrix(matrix)


# Test that a negative off-diagonal value of negative one is rejected
def test_preprocess_rejects_negative_one():
    matrix = np.eye(3)

    matrix[0, 1] = -1.0
    matrix[1, 0] = -1.0

    with pytest.raises(
        ValueError,
        match="strictly between -1 and 1",
    ):
        preprocess_connectivity_matrix(matrix)


# Test that pooled network means and system segregation are calculated correctly
def test_calculate_network_means_and_system_segregation():
    matrix = np.array(
        [
            [1.0, np.tanh(0.20), np.tanh(0.15), np.tanh(0.15)],
            [np.tanh(0.20), 1.0, np.tanh(0.15), np.tanh(0.15)],
            [np.tanh(0.15), np.tanh(0.15), 1.0, np.tanh(0.40)],
            [np.tanh(0.15), np.tanh(0.15), np.tanh(0.40), 1.0],
        ]
    )

    network_labels = np.array(["Network A", "Network A", "Network B", "Network B"])

    within_mask, between_mask = create_network_masks(
        network_labels,
    )

    processed_matrix = preprocess_connectivity_matrix(
        matrix,
    )

    within_mean, between_mean = calculate_network_means(
        processed_matrix,
        within_mask,
        between_mask,
    )

    segregation = calculate_system_segregation(
        within_mean,
        between_mean,
    )

    assert within_mask.sum() == 2
    assert between_mask.sum() == 4

    assert np.isclose(within_mean, 0.30)
    assert np.isclose(between_mean, 0.15)
    assert np.isclose(segregation, 0.50)


# Test that non-Boolean masks are rejected
def test_calculate_network_means_rejects_non_boolean_masks():
    matrix = np.ones((3, 3), dtype=float)
    within_mask = np.triu(np.ones((3, 3), dtype=int), k=1)
    between_mask = np.triu(np.ones((3, 3), dtype=int), k=1)

    with pytest.raises(
        ValueError,
        match="Within-network mask must contain Boolean values",
    ):
        calculate_network_means(
            matrix,
            within_mask,
            between_mask,
        )


# Test that non-finite connectivity values are rejected
def test_calculate_network_means_rejects_non_finite_matrix():
    matrix = np.ones((3, 3), dtype=float)
    matrix[0, 1] = np.nan

    within_mask = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )

    between_mask = np.array(
        [
            [False, False, True],
            [False, False, True],
            [False, False, False],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Connectivity matrix must contain only finite values",
    ):
        calculate_network_means(
            matrix,
            within_mask,
            between_mask,
        )


# Test that overlapping within- and between-network masks are rejected
def test_calculate_network_means_rejects_overlapping_masks():
    matrix = np.ones((3, 3), dtype=float)

    within_mask = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )

    between_mask = np.array(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )

    with pytest.raises(
        ValueError,
        match="must not overlap",
    ):
        calculate_network_means(
            matrix,
            within_mask,
            between_mask,
        )


# Test that masks must partition the complete upper triangle
def test_calculate_network_means_rejects_incomplete_mask_partition():
    matrix = np.ones((3, 3), dtype=float)

    within_mask = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )

    between_mask = np.zeros((3, 3), dtype=bool)

    with pytest.raises(
        ValueError,
        match="partition every upper-triangle edge exactly once",
    ):
        calculate_network_means(
            matrix,
            within_mask,
            between_mask,
        )


# Test that zero within-network mean is rejected while
# very small positive values remain valid
def test_zero_within_mean():
    with pytest.raises(
        ValueError,
        match="within_mean is zero",
    ):
        calculate_system_segregation(
            0.0,
            0.0,
        )

    small_positive_segregation = calculate_system_segregation(
        1e-13,
        0.0,
    )

    assert np.isclose(small_positive_segregation, 1.0)


# Test that segregation is unchanged when the matrix and labels are reordered together
def test_network_segregation_is_invariant_to_reordering():
    matrix = np.array(
        [
            [1.0, np.tanh(0.20), np.tanh(0.15), np.tanh(0.15)],
            [np.tanh(0.20), 1.0, np.tanh(0.15), np.tanh(0.15)],
            [np.tanh(0.15), np.tanh(0.15), 1.0, np.tanh(0.40)],
            [np.tanh(0.15), np.tanh(0.15), np.tanh(0.40), 1.0],
        ]
    )

    network_labels = np.array(["Network A", "Network A", "Network B", "Network B"])

    within_mask, between_mask = create_network_masks(network_labels)

    processed_matrix = preprocess_connectivity_matrix(matrix)

    within_mean, between_mean = calculate_network_means(
        processed_matrix,
        within_mask,
        between_mask,
    )

    original_segregation = calculate_system_segregation(
        within_mean,
        between_mean,
    )

    permutation = [2, 0, 3, 1]

    reordered_matrix = reorder_connectivity_matrix(
        matrix,
        permutation,
    )

    reordered_labels = network_labels[permutation]

    reordered_within_mask, reordered_between_mask = create_network_masks(
        reordered_labels,
    )

    reordered_processed_matrix = preprocess_connectivity_matrix(
        reordered_matrix,
    )

    reordered_within_mean, reordered_between_mean = calculate_network_means(
        reordered_processed_matrix,
        reordered_within_mask,
        reordered_between_mask,
    )

    reordered_segregation = calculate_system_segregation(
        reordered_within_mean,
        reordered_between_mean,
    )

    assert np.isclose(reordered_within_mean, within_mean)
    assert np.isclose(reordered_between_mean, between_mean)
    assert np.isclose(reordered_segregation, original_segregation)


# Test pooled-edge weighting with unequal network sizes and negative segregation
def test_pooled_edge_weighting():
    weighted_z_values = np.array(
        [
            [0.0, 0.10, 0.20, 0.40, 0.40],
            [0.10, 0.0, 0.30, 0.40, 0.40],
            [0.20, 0.30, 0.0, 0.40, 0.40],
            [0.40, 0.40, 0.40, 0.0, 0.80],
            [0.40, 0.40, 0.40, 0.80, 0.0],
        ]
    )

    matrix = np.tanh(weighted_z_values)
    np.fill_diagonal(matrix, 1.0)

    network_labels = np.array(
        [
            "Network A",
            "Network A",
            "Network A",
            "Network B",
            "Network B",
        ]
    )

    within_mask, between_mask = create_network_masks(
        network_labels,
    )

    processed_matrix = preprocess_connectivity_matrix(matrix)

    within_mean, between_mean = calculate_network_means(
        processed_matrix,
        within_mask,
        between_mask,
    )

    segregation = calculate_system_segregation(
        within_mean,
        between_mean,
    )

    assert within_mask.sum() == 4
    assert between_mask.sum() == 6

    assert np.isclose(within_mean, 0.35)
    assert np.isclose(between_mean, 0.40)
    assert np.isclose(segregation, -0.14285714285714285)


# Test the combined participant-level network statistics function
def test_calculate_network_statistics():
    matrix = np.array(
        [
            [1.0, np.tanh(0.20), np.tanh(0.15), np.tanh(0.15)],
            [np.tanh(0.20), 1.0, np.tanh(0.15), np.tanh(0.15)],
            [np.tanh(0.15), np.tanh(0.15), 1.0, np.tanh(0.40)],
            [np.tanh(0.15), np.tanh(0.15), np.tanh(0.40), 1.0],
        ]
    )

    network_labels = np.array(["Network A", "Network A", "Network B", "Network B"])

    result = calculate_network_statistics(
        matrix,
        network_labels,
    )

    assert np.isclose(result["within_mean"], 0.30)
    assert np.isclose(result["between_mean"], 0.15)
    assert np.isclose(result["system_segregation"], 0.50)
    assert result["within_edge_count"] == 2
    assert result["between_edge_count"] == 4


# Test that zeroed negative edges remain included in the network mean
def test_zeroed_negative_edges_remain_in_denominator():
    weighted_z_values = np.array(
        [
            [0.0, 0.40, 0.10, 0.10],
            [0.40, 0.0, 0.10, 0.10],
            [0.10, 0.10, 0.0, -0.20],
            [0.10, 0.10, -0.20, 0.0],
        ]
    )

    matrix = np.tanh(weighted_z_values)
    np.fill_diagonal(matrix, 1.0)

    network_labels = np.array(["Network A", "Network A", "Network B", "Network B"])

    within_mask, between_mask = create_network_masks(
        network_labels,
    )

    processed_matrix = preprocess_connectivity_matrix(
        matrix,
    )

    within_mean, between_mean = calculate_network_means(
        processed_matrix,
        within_mask,
        between_mask,
    )

    segregation = calculate_system_segregation(
        within_mean,
        between_mean,
    )

    assert np.isclose(within_mean, 0.20)
    assert np.isclose(between_mean, 0.10)
    assert np.isclose(segregation, 0.50)
