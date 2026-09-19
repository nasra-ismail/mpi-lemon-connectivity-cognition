import pandas as pd
import pytest

from lemon_connectivity.atlas import (
    parse_schaefer_label,
    validate_atlas_annotations,
)


# Test parsing of a standard Schaefer atlas label
def test_parse_schaefer_label():
    result = parse_schaefer_label(
        "7Networks_LH_Vis_1"
    )

    assert result == {
        "hemisphere": "LH",
        "network_id": "Vis",
        "parcel_name": "Vis_1",
    }


# Test that compound parcel names are preserved
def test_parse_schaefer_label_compound_name():
    result = parse_schaefer_label(
        "7Networks_LH_DorsAttn_Post_1"
    )

    assert result == {
        "hemisphere": "LH",
        "network_id": "DorsAttn",
        "parcel_name": "DorsAttn_Post_1",
    }


# Test that an invalid atlas prefix is rejected
def test_parse_schaefer_label_invalid_prefix():
    with pytest.raises(
        ValueError,
        match="Invalid Schaefer atlas prefix",
    ):
        parse_schaefer_label(
            "8Networks_LH_Vis_1"
        )


# Test that an invalid hemisphere is rejected
def test_parse_schaefer_label_invalid_hemisphere():
    with pytest.raises(
        ValueError,
        match="Invalid Schaefer hemisphere",
    ):
        parse_schaefer_label(
            "7Networks_XH_Vis_1"
        )


# Test that an invalid network identifier is rejected
def test_parse_schaefer_label_invalid_network():
    with pytest.raises(
        ValueError,
        match="Invalid Schaefer network identifier",
    ):
        parse_schaefer_label(
            "7Networks_LH_Invalid_1"
        )


# Test that a missing parcel name is rejected
def test_parse_schaefer_label_missing_parcel():
    with pytest.raises(
        ValueError,
        match="Missing Schaefer parcel name",
    ):
        parse_schaefer_label(
            "7Networks_LH_Vis_"
        )


# Test that a non-string label is rejected
def test_parse_schaefer_label_non_string():
    with pytest.raises(
        TypeError,
        match="Schaefer label must be a string",
    ):
        parse_schaefer_label(123)


# Test that required atlas columns are present
def test_validate_atlas_annotations_missing_column():
    roi_lut = pd.DataFrame(
        {
            "roi_index": range(200),
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                "Vis",
                "SomMot",
                "DorsAttn",
                "SalVentAttn",
                "Limbic",
                "Cont",
                "Default",
            ]
            * 28
            + ["Vis"] * 4,
            "parcel_name": [f"Vis_{i}" for i in range(200)],
        }
    )

    with pytest.raises(
        KeyError,
        match="Missing required atlas columns",
    ):
        validate_atlas_annotations(roi_lut)


# Test that exactly 200 ROIs are required
def test_validate_atlas_annotations_wrong_roi_count():
    roi_lut = pd.DataFrame(
        {
            "roi_index": range(199),
            "hemisphere": ["LH", "RH"] * 99 + ["LH"],
            "network_id": [
                "Vis",
                "SomMot",
                "DorsAttn",
                "SalVentAttn",
                "Limbic",
                "Cont",
                "Default",
            ]
            * 28
            + ["Vis"] * 3,
            "parcel_name": [f"Vis_{i}" for i in range(199)],
            "canonical_network": [
                "Visual",
                "Somatomotor",
                "Dorsal Attention",
                "Salience/Ventral Attention",
                "Limbic",
                "Control",
                "Default",
            ]
            * 28
            + ["Visual"] * 3,
        }
    )

    with pytest.raises(
        ValueError,
        match="Expected 200 ROIs",
    ):
        validate_atlas_annotations(roi_lut)


# Test that ROI indices must be exactly 0 through 199
def test_validate_atlas_annotations_invalid_roi_indices():
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    roi_lut = pd.DataFrame(
        {
            "roi_index": list(range(199)) + [200],
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                network_ids[i % 7]
                for i in range(200)
            ],
            "parcel_name": [
                f"{network_ids[i % 7]}_{i}"
                for i in range(200)
            ],
            "canonical_network": [
                canonical_networks[network_ids[i % 7]]
                for i in range(200)
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="ROI indices must be ordered exactly from 0 to 199",
    ):
        validate_atlas_annotations(roi_lut)


# Test that missing annotation values are rejected
def test_validate_atlas_annotations_missing_value():
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    roi_lut = pd.DataFrame(
        {
            "roi_index": range(200),
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                network_ids[i % 7]
                for i in range(200)
            ],
            "parcel_name": [
                f"{network_ids[i % 7]}_{i}"
                for i in range(200)
            ],
            "canonical_network": [
                canonical_networks[network_ids[i % 7]]
                for i in range(200)
            ],
        }
    )

    roi_lut.loc[0, "parcel_name"] = None

    with pytest.raises(
        ValueError,
        match="Required atlas annotations must not contain missing values",
    ):
        validate_atlas_annotations(roi_lut)


# Test that a complete valid 200-row atlas table passes validation
def test_validate_atlas_annotations_valid_table():
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    roi_lut = pd.DataFrame(
        {
            "roi_index": range(200),
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                network_ids[i % 7]
                for i in range(200)
            ],
            "parcel_name": [
                f"{network_ids[i % 7]}_{i}"
                for i in range(200)
            ],
            "canonical_network": [
                canonical_networks[network_ids[i % 7]]
                for i in range(200)
            ],
        }
    )

    validate_atlas_annotations(roi_lut)


# Test that ROI rows must remain in exact 0-to-199 order
def test_validate_atlas_annotations_shuffled_roi_order():
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    roi_lut = pd.DataFrame(
        {
            "roi_index": range(200),
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                network_ids[i % 7]
                for i in range(200)
            ],
            "parcel_name": [
                f"{network_ids[i % 7]}_{i}"
                for i in range(200)
            ],
            "canonical_network": [
                canonical_networks[network_ids[i % 7]]
                for i in range(200)
            ],
        }
    )

    roi_lut = roi_lut.iloc[
        [1, 0] + list(range(2, 200))
    ].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="ROI indices must be ordered exactly from 0 to 199",
    ):
        validate_atlas_annotations(roi_lut)


# Test that duplicate hemisphere-network-parcel annotations are rejected
def test_validate_atlas_annotations_duplicate_annotation():
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    roi_lut = pd.DataFrame(
        {
            "roi_index": range(200),
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                network_ids[i % 7]
                for i in range(200)
            ],
            "parcel_name": [
                f"{network_ids[i % 7]}_{i}"
                for i in range(200)
            ],
            "canonical_network": [
                canonical_networks[network_ids[i % 7]]
                for i in range(200)
            ],
        }
    )

    # Copy all annotation fields from row 0 to row 2
    # while keeping the ROI indices and hemisphere counts valid
    roi_lut.loc[2, "hemisphere"] = roi_lut.loc[
        0, "hemisphere"
    ]
    roi_lut.loc[2, "network_id"] = roi_lut.loc[
        0, "network_id"
    ]
    roi_lut.loc[2, "parcel_name"] = roi_lut.loc[
        0, "parcel_name"
    ]
    roi_lut.loc[2, "canonical_network"] = roi_lut.loc[
        0, "canonical_network"
    ]

    # The duplicate annotation should trigger the duplicate-validation branch
    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        validate_atlas_annotations(roi_lut)

# Test that network identifiers and canonical names must agree
def test_validate_atlas_annotations_network_mapping():
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    roi_lut = pd.DataFrame(
        {
            "roi_index": range(200),
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                network_ids[i % 7]
                for i in range(200)
            ],
            "parcel_name": [
                f"{network_ids[i % 7]}_{i}"
                for i in range(200)
            ],
            "canonical_network": [
                canonical_networks[network_ids[i % 7]]
                for i in range(200)
            ],
        }
    )

    # Change one Visual annotation to an incorrect canonical network
    roi_lut.loc[0, "canonical_network"] = "Control"

    with pytest.raises(
        ValueError,
        match="Incorrect canonical mapping for Vis",
    ):
        validate_atlas_annotations(roi_lut)


# Test that non-integer ROI indices are rejected
def test_validate_atlas_annotations_non_integer_roi_index():
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    roi_lut = pd.DataFrame(
        {
            "roi_index": range(200),
            "hemisphere": ["LH", "RH"] * 100,
            "network_id": [
                network_ids[i % 7]
                for i in range(200)
            ],
            "parcel_name": [
                f"{network_ids[i % 7]}_{i}"
                for i in range(200)
            ],
            "canonical_network": [
                canonical_networks[network_ids[i % 7]]
                for i in range(200)
            ],
        }
    )

    # Convert the ROI index column to object so it can hold a non-integer value
    roi_lut["roi_index"] = roi_lut["roi_index"].astype(object)

    # Replace one integer ROI index with a non-integer value
    roi_lut.loc[0, "roi_index"] = 0.5

    with pytest.raises(
        ValueError,
        match="ROI indices must be integers",
    ):
        validate_atlas_annotations(roi_lut)