# Data management

## Sources and roles

| Source | Role | Tracked in Git? |
|---|---|---:|
| Yadav `DATA/cohort_information.tsv` | Learning-pass metadata | No |
| Yadav `DATA/FCM/` | Complete Schaefer-200 edge lists | No |
| Yadav `DATA/FCN/` | Thresholded graph sensitivity/reference | No |
| `Behavioural_Data_MPILMBB_LEMON.zip` | LEMON cognition and covariates | No |
| Preprocessed LEMON fMRI derivatives | Independent pilot | No |
| Source code, tests, documentation | Reproducible workflow | Yes |
| Permitted small aggregate outputs | Portfolio evidence | Case-by-case |

## Participant identifiers

- Keep IDs as strings.
- Never renumber participants to remove gaps.
- Convert Yadav numeric IDs to canonical form only with an explicit, tested rule, for
  example `32301` to `sub-032301`.
- Require one-to-one merge validation and report unmatched IDs in both directions.

## Provenance record

For every external source, record filename, URL, release/version, download date,
checksum, license/citation, and any local renaming. Do not modify files in
`data/external/`.

## Derived data

Store intermediate outputs under `data/interim/` and final participant-level analysis
tables under `data/processed/`; both remain local. Before sharing any derived table,
check identifiability, license terms, and whether an aggregate figure would suffice.

## Recovery and reproducibility

Source datasets must be re-downloadable from their documented locations. Analysis
products must be reproducible from committed code and a locally recorded environment.
Data are not backed up by pushing them into this repository.
