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


### LEMON behavioral archive — Issue #15

| Field | Record |
|---|---|
| Release name | MPILMBB Behavioural Data LEMON |
| Archive filename | `Behavioural_Data_MPILMBB_LEMON.zip` |
| Listed release date | 2018-04-05 |
| Official release page | https://www.nitrc.org/frs/?group_id=1184 |
| Direct download URL | https://www.nitrc.org/frs/download.php/10585/Behavioural_Data_MPILMBB_LEMON.zip |
| Actual local download date | 2026-08-15 |
| File size | 144654 bytes |
| SHA-256 | `19D95A7E8E639EC6C0CE3161BDC8827F74794DAB19FE0DFCA0B99F5FCE47FF0A` |
| Local path | `data/external/lemon_behavioral/Behavioural_Data_MPILMBB_LEMON.zip` |
| Local renaming | None; original filename preserved |
| ZIP integrity | Passed (`ZipFile.testzip()` returned `None`) |
| Git status | Ignored and not tracked |

The SHA-256 value is a locally calculated fingerprint. No publisher-provided checksum was identified in the NITRC release listing.

The archive contents were inspected without extraction. The ZIP contains 84 entries,
including CSV data files and accompanying information files, organized into cognitive
test, emotion and personality, and medical data directories. The archive includes
measures such as TAP working memory, TMT, CVLT, questionnaires, anthropometry,
blood pressure, and participant metadata.

**Dataset citation**

Babayan, A., Erbey, M., Kumral, D., et al. (2019).
A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults.
Scientific Data, 6, 180308.
https://doi.org/10.1038/sdata.2018.308

**Data-use conditions**

Access to the behavioral data is subject to the NITRC data-use terms. Users must agree not to attempt to identify individuals from the phenotypic data. The publication's CC BY license applies to the article; it should not be assumed to constitute a separate license for this behavioral archive.

**Complementary release**

`behavioral_data_MPILMBB_v3.zip` (MPILMBB Behavioral Data 3.0, listed release date 2018-08-15) is a separate release. It was not found in the project repository during the Issue #15 checks and is not retained alongside the LEMON-specific archive.

Issue reference: #15.

## Derived data

Store intermediate outputs under `data/interim/` and final participant-level analysis
tables under `data/processed/`; both remain local. Before sharing any derived table,
check identifiability, license terms, and whether an aggregate figure would suffice.

## Recovery and reproducibility

Source datasets must be re-downloadable from their documented locations. Analysis
products must be reproducible from committed code and a locally recorded environment.
Data are not backed up by pushing them into this repository.
