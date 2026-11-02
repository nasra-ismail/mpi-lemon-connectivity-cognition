import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from lemon_connectivity.networks import calculate_network_statistics
from lemon_connectivity.sensitivity import fcm_manifest_checksum, sha256_file

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK = "notebooks/09_segregation_sensitivity.ipynb"


def make_synthetic_project(root):
    """Create 16 artificial 200-ROI edge lists; no real participant data needed."""
    for filename in [
        "pyproject.toml",
        "docs/ANALYSIS_PLAN.md",
        "src/lemon_connectivity/networks.py",
        "src/lemon_connectivity/sensitivity.py",
        NOTEBOOK,
    ]:
        target = root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / filename, target)
    data_dir = root / "data/external/Curvature-FCN-Aging/DATA"
    fcm_dir = data_dir / "FCM"
    fcm_dir.mkdir(parents=True)
    interim = root / "data/interim"
    processed = root / "data/processed"
    interim.mkdir(parents=True)
    processed.mkdir(parents=True)
    networks = {
        "Vis": "Visual",
        "SomMot": "Somatomotor",
        "DorsAttn": "Dorsal Attention",
        "SalVentAttn": "Salience/Ventral Attention",
        "Limbic": "Limbic",
        "Cont": "Control",
        "Default": "Default",
    }
    atlas_rows = []
    for hemisphere in ["LH", "RH"]:
        for (network_id, canonical), count in zip(
            networks.items(), [15, 15, 15, 15, 15, 15, 10], strict=True
        ):
            for number in range(1, count + 1):
                atlas_rows.append(
                    {
                        "roi_index": len(atlas_rows),
                        "hemisphere": hemisphere,
                        "parcel_name": f"{network_id}_{number}",
                        "canonical_network": canonical,
                    }
                )
    atlas = pd.DataFrame(atlas_rows)
    atlas_path = interim / "schaefer200_7networks_labels.tsv"
    atlas.to_csv(atlas_path, sep="\t", index=False)
    labels = atlas["canonical_network"].to_numpy()
    i, j = np.triu_indices(200, k=1)
    same_network = labels[i] == labels[j]
    rng = np.random.default_rng(14)
    primary, cohort, alignment, inputs = [], [], [], []
    for number in range(1, 17):
        participant_id = f"sub-{number:06d}"
        elderly = number > 8
        male = (number - 1) % 8 >= 4
        values = rng.normal(0.06 + 0.005 * elderly, 0.06, len(i))
        values[same_network] = rng.normal(
            0.25 - 0.025 * elderly + 0.005 * male, 0.08, int(same_network.sum())
        )
        if number == 1:
            # Primary eligible, but signed W is negative: excluded only from common.
            values[same_network] = rng.normal(-0.04, 0.05, int(same_network.sum()))
        if number == 2:
            # Primary zero denominator: retained in every ledger, never the model.
            values[same_network] = -0.02
        matrix = np.eye(200)
        matrix[i, j] = values
        matrix[j, i] = values
        path = fcm_dir / f"fcm_sub_{number:05d}.txt"
        np.savetxt(
            path,
            np.column_stack([i, j, values]),
            delimiter="\t",
            fmt=["%d", "%d", "%.18e"],
        )
        primary.append(
            {
                "canonical_id": participant_id,
                **calculate_network_statistics(matrix, labels),
            }
        )
        cohort.append(
            {
                "sub_id": f"{number:05d}",
                "cohort": "elderly" if elderly else "young",
                "sex": "male" if male else "female",
            }
        )
        alignment.append(
            {
                "canonical_id": participant_id,
                "matrix_filename": path.name,
                "alignment_status": "matched",
            }
        )
        inputs.append({"canonical_id": participant_id, "matrix_path": str(path)})
    pd.DataFrame(primary).to_csv(
        processed / "network_segregation_results.tsv", sep="\t", index=False
    )
    pd.DataFrame(cohort).to_csv(
        data_dir / "cohort_information.tsv", sep="\t", index=False
    )
    pd.DataFrame(alignment).to_csv(
        interim / "participant_alignment_table.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "edge_policy_version": "edge-policy-v1",
                "segregation_definition": "seg_z_positive",
                "within_mean_denominator_rule": "W <= 1e-12 is undefined",
                "n_rois": 200,
                "n_retained_participants": 16,
                "n_processed_fcm_files": 16,
                "atlas_sha256": sha256_file(atlas_path),
                "fcm_manifest_sha256": fcm_manifest_checksum(
                    pd.DataFrame(inputs), root
                ),
            }
        ]
    ).to_csv(processed / "network_segregation_provenance.tsv", sep="\t", index=False)


def test_complete_sensitivity_notebook_with_synthetic_200_roi_cohort(
    tmp_path, monkeypatch
):
    make_synthetic_project(tmp_path)
    inputs_before = {
        path: sha256_file(path)
        for path in (tmp_path / "data").rglob("*")
        if path.is_file()
    }
    monkeypatch.chdir(tmp_path / "notebooks")
    notebook = json.loads((tmp_path / NOTEBOOK).read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"<issue14-notebook-cell-{index}>", "exec"), namespace)  # noqa: S102  

    processed = tmp_path / "data/processed"
    generated = tmp_path / "results/generated"
    ledger = pd.read_csv(processed / "segregation_sensitivity_analysis.tsv", sep="\t")
    assert len(ledger) == 64
    assert ledger.groupby("canonical_id").size().eq(4).all()
    common = ledger.loc[ledger["in_common_sample"]]
    assert common["canonical_id"].nunique() == 14
    assert not common["canonical_id"].isin(["sub-000001", "sub-000002"]).any()
    zero = ledger.loc[
        ledger["canonical_id"].eq("sub-000002")
        & ledger["definition"].eq("seg_z_positive")
    ]
    assert zero["within_mean"].item() == 0
    assert np.isfinite(zero["between_mean"].item())
    assert zero["system_segregation"].isna().all()
    models = pd.read_csv(generated / "segregation_sensitivity_models.tsv", sep="\t")
    assert models["model_status"].eq("estimated").all()
    assert models.loc[models["sample"].eq("common"), "n_total"].tolist() == [14] * 4
    assert (
        models.loc[models["sample"].eq("primary_full_reference"), "n_total"].item()
        == 15
    )
    assert (generated / "segregation_sensitivity_forest.png").stat().st_size > 1000
    provenance = json.loads(
        (processed / "segregation_sensitivity_provenance.json").read_text()
    )
    assert provenance["n_common_eligible"] == 14
    assert provenance["primary_replication"]["passed"]
    assert provenance["covariance"] == "HC3"
    assert provenance["use_t"] is True
    for path, expected_hash in inputs_before.items():
        assert sha256_file(path) == expected_hash
    for filename, expected_hash in provenance["output_sha256"].items():
        assert sha256_file(tmp_path / filename) == expected_hash
