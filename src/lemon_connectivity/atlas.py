from numbers import Integral


# Parse and validate one Schaefer atlas label
def parse_schaefer_label(label: str) -> dict[str, str]:
    # Validate that the input is a string
    if not isinstance(label, str):
        raise TypeError("Schaefer label must be a string")

    # Split the label into underscore-separated components
    parts = label.split("_")

    # Validate that the label has enough components
    if len(parts) < 4:
        raise ValueError(
            f"Malformed Schaefer label: {label}"
        )

    # Validate the required Schaefer atlas prefix
    if parts[0] != "7Networks":
        raise ValueError(
            f"Invalid Schaefer atlas prefix: {label}"
        )

    # Validate the hemisphere identifier
    hemisphere = parts[1]

    if hemisphere not in {"LH", "RH"}:
        raise ValueError(
            f"Invalid Schaefer hemisphere: {hemisphere}"
        )

    # Extract the Yeo network identifier
    network_id = parts[2]

    # Define the seven valid Yeo network identifiers
    valid_network_ids = {
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    }

    # Validate the network identifier
    if network_id not in valid_network_ids:
        raise ValueError(
            f"Invalid Schaefer network identifier: {network_id}"
        )

    # Preserve the complete parcel name, including compound names
    parcel_name = "_".join(parts[2:])

    # Validate that the parcel name is present
    if not parcel_name:
        raise ValueError(
            f"Missing Schaefer parcel name: {label}"
        )

    # Return the validated atlas annotations
    return {
        "hemisphere": hemisphere,
        "network_id": network_id,
        "parcel_name": parcel_name,
    }


# Validate the complete Schaefer-200 atlas annotation table
def validate_atlas_annotations(roi_lut):
    # Define the expected zero-based FCM ROI indices
    expected_roi_indices = list(range(200))

    # Define the expected seven Yeo network identifiers
    expected_network_ids = {
        "Vis",
        "SomMot",
        "DorsAttn",
        "SalVentAttn",
        "Limbic",
        "Cont",
        "Default",
    }

    # Define the expected mapping from network identifiers
    # to canonical Yeo network names
    expected_network_mapping = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }

    # Define the required annotation columns
    required_columns = [
        "roi_index",
        "hemisphere",
        "network_id",
        "parcel_name",
        "canonical_network",
    ]

    # Identify any required columns that are missing
    missing_columns = [
        column
        for column in required_columns
        if column not in roi_lut.columns
    ]

    # Raise an explicit error if required columns are missing
    if missing_columns:
        raise KeyError(
            f"Missing required atlas columns: {missing_columns}"
        )

    # Validate that exactly 200 ROIs are present
    if len(roi_lut) != 200:
        raise ValueError(
            f"Expected 200 ROIs, found {len(roi_lut)}"
        )

    # Validate that every ROI index is unique
    if roi_lut["roi_index"].nunique() != 200:
        raise ValueError(
            "ROI indices must be unique"
        )

    # Validate that ROI indices are integers
    if not roi_lut["roi_index"].apply(
        lambda value: isinstance(value, Integral)
        and not isinstance(value, bool)
    ).all():
        raise ValueError(
            "ROI indices must be integers"
        )

    # Validate the exact ROI ordering used by the FCM matrices
    if roi_lut["roi_index"].tolist() != expected_roi_indices:
        raise ValueError(
            "ROI indices must be ordered exactly from 0 to 199"
        )

    # Validate that no required annotation value is missing
    if roi_lut[required_columns].isna().any().any():
        raise ValueError(
            "Required atlas annotations must not contain missing values"
        )

    # Validate that exactly 100 ROIs belong to each hemisphere
    hemisphere_counts = roi_lut["hemisphere"].value_counts().to_dict()

    if hemisphere_counts != {"LH": 100, "RH": 100}:
        raise ValueError(
            "Expected exactly 100 LH and 100 RH ROIs"
        )

    # Validate that only the seven expected network identifiers are present
    observed_network_ids = set(roi_lut["network_id"])

    if observed_network_ids != expected_network_ids:
        raise ValueError(
            "Unexpected Schaefer network identifiers"
        )

    # Validate that exactly the seven canonical networks are present
    expected_canonical_networks = set(
        expected_network_mapping.values()
    )

    observed_canonical_networks = set(
        roi_lut["canonical_network"]
    )

    if observed_canonical_networks != expected_canonical_networks:
        raise ValueError(
            "Unexpected canonical Yeo network names"
        )

    # Validate that each network identifier maps to the correct
    # canonical Yeo network name
    for network_id, canonical_network in expected_network_mapping.items():
        observed_mapping = set(
            roi_lut.loc[
                roi_lut["network_id"] == network_id,
                "canonical_network",
            ]
        )

        if observed_mapping != {canonical_network}:
            raise ValueError(
                f"Incorrect canonical mapping for {network_id}"
            )

    # Validate that each hemisphere-network-parcel combination is unique
    if roi_lut.duplicated(
        subset=[
            "hemisphere",
            "network_id",
            "parcel_name",
        ]
    ).any():
        raise ValueError(
            "Duplicate hemisphere-network-parcel annotations found"
        )