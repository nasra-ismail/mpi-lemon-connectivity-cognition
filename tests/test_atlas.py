import pandas as pd
import pytest

# Import the Schaefer label parser and atlas validator
from lemon_connectivity.atlas import (
    parse_schaefer_label,
    validate_atlas_annotations,
)


# Test parsing of a simple visual-network Schaefer label
def test_parse_schaefer_label():
    # Define a representative Schaefer atlas label
    label = "7Networks_LH_Vis_1"

    # Parse the label into atlas annotations
    result = parse_schaefer_label(label)

    # Validate the extracted hemisphere
    assert result["hemisphere"] == "LH"

    # Validate the extracted network identifier
    assert result["network_id"] == "Vis"

    # Validate the complete parcel name
    assert result["parcel_name"] == "Vis_1"


# Test parsing of a Schaefer label with a compound parcel name
def test_parse_compound_schaefer_label():
    # Define a representative compound Schaefer atlas label
    label = "7Networks_LH_DorsAttn_Post_1"

    # Parse the label into atlas annotations
    result = parse_schaefer_label(label)

    # Validate the extracted hemisphere
    assert result["hemisphere"] == "LH"

    # Validate the extracted Yeo network identifier
    assert result["network_id"] == "DorsAttn"

    # Validate that the complete compound parcel name is preserved
    assert result["parcel_name"] == "DorsAttn_Post_1"


# Test that a malformed Schaefer label prefix is rejected
def test_parse_invalid_schaefer_prefix():
    # Define a label with an invalid atlas prefix
    label = "WrongAtlas_LH_Vis_1"

    # Validate that the invalid prefix raises a ValueError
    with pytest.raises(ValueError):
        parse_schaefer_label(label)


# Test that an invalid hemisphere identifier is rejected
def test_parse_invalid_schaefer_hemisphere():
    # Define a label with an invalid hemisphere
    label = "7Networks_XX_Vis_1"

    # Validate that the invalid hemisphere raises a ValueError
    with pytest.raises(ValueError):
        parse_schaefer_label(label)


# Test that an invalid network identifier is rejected
def test_parse_invalid_schaefer_network():
    # Define a label with an invalid network identifier
    label = "7Networks_LH_Unknown_1"

    # Validate that the invalid network raises a ValueError
    with pytest.raises(ValueError):
        parse_schaefer_label(label)


# Test that a non-string Schaefer label is rejected
def test_parse_non_string_label():
    # Define an invalid non-string label
    label = 123

    # Validate that the non-string input raises a TypeError
    with pytest.raises(TypeError):
        parse_schaefer_label(label)


# Test that atlas validation rejects a table with a missing required column
def test_validate_atlas_annotations_missing_column():
    # Create a minimal table with one required column missing
    roi_lut = pd.DataFrame(
        {
            "roi_index": [0],
            "hemisphere": ["LH"],
            "network_id": ["Vis"],
            "parcel_name": ["Vis_1"],
        }
    )

    # Validate that the missing canonical_network column is detected
    with pytest.raises(KeyError):
        validate_atlas_annotations(roi_lut)


# Test that atlas validation rejects a table with the wrong number of ROIs
def test_validate_atlas_annotations_wrong_roi_count():
    # Create a table with the required columns but only one ROI
    roi_lut = pd.DataFrame(
        {
            "roi_index": [0],
            "hemisphere": ["LH"],
            "network_id": ["Vis"],
            "parcel_name": ["Vis_1"],
            "canonical_network": ["Visual"],
        }
    )

    # Validate that the incorrect ROI count is detected
    with pytest.raises(ValueError):
        validate_atlas_annotations(roi_lut)


# Test that atlas validation rejects an incorrect ROI index set
def test_validate_atlas_annotations_invalid_roi_indices():
    # Create 200 rows with an invalid ROI index
    roi_lut = pd.DataFrame(
        {
            "roi_index": list(range(1, 201)),
            "hemisphere": ["LH"] * 100 + ["RH"] * 100,
            "network_id": ["Vis"] * 200,
            "parcel_name": [f"parcel_{i}" for i in range(200)],
            "canonical_network": ["Visual"] * 200,
        }
    )

    # Validate that the incorrect zero-based ROI indexing is detected
    with pytest.raises(ValueError):
        validate_atlas_annotations(roi_lut)


# Test that atlas validation rejects missing annotation values
def test_validate_atlas_annotations_missing_value():
    # Create 200 rows with one missing annotation
    roi_lut = pd.DataFrame(
        {
            "roi_index": list(range(200)),
            "hemisphere": ["LH"] * 100 + ["RH"] * 100,
            "network_id": ["Vis"] * 200,
            "parcel_name": [f"parcel_{i}" for i in range(200)],
            "canonical_network": ["Visual"] * 200,
        }
    )

    # Introduce one missing annotation value
    roi_lut.loc[0, "parcel_name"] = None

    # Validate that the missing annotation is detected
    with pytest.raises(ValueError):
        validate_atlas_annotations(roi_lut)


# Test that a complete valid 200-row atlas table passes validation
def test_validate_atlas_annotations_valid_table():
    # Define the seven Schaefer network identifiers
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    # Define the canonical name for each network identifier
    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    # Create 200 unique atlas annotations
    rows = []

    for roi_index in range(200):
        # Assign the first 100 ROIs to the left hemisphere
        hemisphere = "LH" if roi_index < 100 else "RH"

        # Assign networks cyclically across the 200 ROIs
        network_id = network_ids[roi_index % len(network_ids)]

        # Create a unique parcel name for each ROI
        parcel_name = f"{network_id}_parcel_{roi_index}"

        # Add one complete annotation row
        rows.append(
            {
                "roi_index": roi_index,
                "hemisphere": hemisphere,
                "network_id": network_id,
                "parcel_name": parcel_name,
                "canonical_network": canonical_networks[network_id],
            }
        )

    # Convert the annotation rows into a DataFrame
    roi_lut = pd.DataFrame(rows)

    # Validate the complete atlas table
    validate_atlas_annotations(roi_lut)


# Test that atlas validation rejects correctly indexed ROIs in the wrong order
def test_validate_atlas_annotations_shuffled_roi_order():
    # Create a valid 200-row atlas table
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    # Define the canonical name for each network identifier
    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    # Create 200 valid annotation rows
    rows = []

    for roi_index in range(200):
        # Assign the hemisphere based on the ROI index
        hemisphere = "LH" if roi_index < 100 else "RH"

        # Assign a valid network identifier
        network_id = network_ids[roi_index % len(network_ids)]

        # Create a unique parcel name
        parcel_name = f"{network_id}_parcel_{roi_index}"

        # Add the annotation row
        rows.append(
            {
                "roi_index": roi_index,
                "hemisphere": hemisphere,
                "network_id": network_id,
                "parcel_name": parcel_name,
                "canonical_network": canonical_networks[network_id],
            }
        )

    # Convert the rows into a DataFrame
    roi_lut = pd.DataFrame(rows)

    # Move the first ROI to the end to break the required ordering
    roi_lut = pd.concat(
        [
            roi_lut.iloc[1:],
            roi_lut.iloc[[0]],
        ],
        ignore_index=True,
    )

    # Validate that the incorrect ordering is rejected
    with pytest.raises(ValueError):
        validate_atlas_annotations(roi_lut)

# Test that atlas validation rejects duplicate annotations
def test_validate_atlas_annotations_duplicate_annotation():
    # Create a valid 200-row atlas table
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    # Define the canonical name for each network identifier
    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    # Create 200 valid annotation rows
    rows = []

    for roi_index in range(200):
        # Assign the hemisphere based on the ROI index
        hemisphere = "LH" if roi_index < 100 else "RH"

        # Assign a valid network identifier
        network_id = network_ids[roi_index % len(network_ids)]

        # Create a unique parcel name
        parcel_name = f"{network_id}_parcel_{roi_index}"

        # Add the annotation row
        rows.append(
            {
                "roi_index": roi_index,
                "hemisphere": hemisphere,
                "network_id": network_id,
                "parcel_name": parcel_name,
                "canonical_network": canonical_networks[network_id],
            }
        )

    # Convert the rows into a DataFrame
    roi_lut = pd.DataFrame(rows)

    # Duplicate one annotation while keeping the ROI indices unique
    roi_lut.loc[1, "hemisphere"] = roi_lut.loc[0, "hemisphere"]
    roi_lut.loc[1, "network_id"] = roi_lut.loc[0, "network_id"]
    roi_lut.loc[1, "parcel_name"] = roi_lut.loc[0, "parcel_name"]

    # Validate that the duplicate annotation is rejected
    with pytest.raises(ValueError):
        validate_atlas_annotations(roi_lut)

# Test that atlas validation rejects an incorrect network-to-canonical mapping
def test_validate_atlas_annotations_network_canonical_mismatch():
    # Create a valid 200-row atlas table
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    # Define the canonical name for each network identifier
    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    # Create 200 valid annotation rows
    rows = []

    for roi_index in range(200):
        # Assign the hemisphere based on the ROI index
        hemisphere = "LH" if roi_index < 100 else "RH"

        # Assign a valid network identifier
        network_id = network_ids[roi_index % len(network_ids)]

        # Create a unique parcel name
        parcel_name = f"{network_id}_parcel_{roi_index}"

        # Add the annotation row
        rows.append(
            {
                "roi_index": roi_index,
                "hemisphere": hemisphere,
                "network_id": network_id,
                "parcel_name": parcel_name,
                "canonical_network": canonical_networks[network_id],
            }
        )

    # Convert the rows into a DataFrame
    roi_lut = pd.DataFrame(rows)

    # Change one canonical network to an incorrect value
    roi_lut.loc[0, "canonical_network"] = "Control"

    # Validate that the incorrect network mapping is rejected
    with pytest.raises(ValueError):
        validate_atlas_annotations(roi_lut)

# Test that atlas validation rejects non-integer ROI indices
def test_validate_atlas_annotations_non_integer_roi_indices():
    # Create a valid 200-row atlas table
    network_ids = [
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    ]

    # Define the canonical name for each network identifier
    canonical_networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    # Create 200 valid annotation rows
    rows = []

    for roi_index in range(200):
        # Assign the hemisphere based on the ROI index
        hemisphere = "LH" if roi_index < 100 else "RH"

        # Assign a valid network identifier
        network_id = network_ids[roi_index % len(network_ids)]

        # Create a unique parcel name
        parcel_name = f"{network_id}_parcel_{roi_index}"

        # Add the annotation row
        rows.append(
            {
                "roi_index": roi_index,
                "hemisphere": hemisphere,
                "network_id": network_id,
                "parcel_name": parcel_name,
                "canonical_network": canonical_networks[network_id],
            }
        )

    # Convert the rows into a DataFrame
    roi_lut = pd.DataFrame(rows)

# Convert the ROI index column to object so it can hold a non-integer value
    roi_lut["roi_index"] = roi_lut["roi_index"].astype(object)

# Replace one integer ROI index with a non-integer value
    roi_lut.loc[0, "roi_index"] = 0.5

    # Validate that the non-integer ROI index is rejected
    with pytest.raises(ValueError):
        validate_atlas_annotations(roi_lut)