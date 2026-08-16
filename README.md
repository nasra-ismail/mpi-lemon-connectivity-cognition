# MPI-LEMON Connectivity & Cognition

[![Tests](https://github.com/xamdoo/mpi-lemon-connectivity-cognition/actions/workflows/tests.yml/badge.svg)](https://github.com/xamdoo/mpi-lemon-connectivity-cognition/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible, learning-first analysis of resting-state functional connectivity,
brain-network organization, aging, and fluid reasoning using MPI-LEMON.

> **Status:** project scaffold. No scientific results are reported yet.

## Scientific question

Does functional-network segregation explain or predict fluid-reasoning performance
beyond age group, sex, education, and head motion?

Secondary aims are to:

- reproduce younger–older differences in network organization;
- examine prespecified graph measures;
- compare demographic-only, brain-only, and combined prediction; and
- learn the full path from preprocessed fMRI to parcel time series and connectivity.

The design is cross-sectional. Group differences cannot be interpreted as individual
aging or causal effects.

## Two analysis passes

1. **Learning pass:** use Yadav et al.'s ready-made Schaefer-200 correlation edge
   lists to learn matrix validation, network segregation, graph theory, statistics,
   and prediction.
2. **Independent pilot:** extract Schaefer-100 time series from a small set of
   preprocessed LEMON fMRI images, build new matrices with a documented confound
   strategy, and scale only after quality control.

The complete `DATA/FCM` matrices are used first. The thresholded `DATA/FCN` files are
not interchangeable with them and are not the default analysis input.

## Data sources

- [MPI-LEMON dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6371893/)
- [Official MPI-LEMON page](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html)
- [LEMON behavioral releases](https://www.nitrc.org/frs/?group_id=1184)
- [Yadav et al. Curvature-FCN-Aging repository](https://github.com/asamallab/Curvature-FCN-Aging)

Use the LEMON-specific behavioral archive, `Behavioural_Data_MPILMBB_LEMON.zip`.
The complementary `behavioral_data_MPILMBB_v3.zip` release is kept separate and is
not merged as though it were the LEMON cohort.

## Quick start

```bash
git clone git@github.com:xamdoo/mpi-lemon-connectivity-cognition.git
cd mpi-lemon-connectivity-cognition
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

Then follow [the learning path](docs/LEARNING_PATH.md) and work through the
[issue-based roadmap](docs/ROADMAP.md) in order.

## Local data layout

```text
data/
├── external/
│   ├── Curvature-FCN-Aging/
│   └── lemon_behavioral/
├── interim/
└── processed/
```

All contents under `data/` are ignored except `data/README.md`. Raw MRI,
behavioral archives, connectivity matrices, participant-level sensitive data, and
credentials must never be committed. See [data management](docs/DATA_MANAGEMENT.md).

## Repository map

| Path | Purpose |
|---|---|
| `docs/` | Scientific plan, decisions, learning sequence, and roadmap |
| `notebooks/` | Numbered, narrative analyses |
| `src/lemon_connectivity/` | Reusable and tested functions |
| `tests/` | Unit tests for data loading and analysis helpers |
| `results/` | README plus selected shareable final outputs |
| `data/` | Local-only data location; contents are ignored |

## Task management

- One GitHub issue represents one auditable task.
- Titles use phase prefixes `[P0]` through `[P7]`.
- A task is complete only when its definition-of-done checklist is satisfied and
  its code, notebook, figures, and pull request are linked.
- Start with [issue #3](https://github.com/xamdoo/mpi-lemon-connectivity-cognition/issues/3),
  then [issue #2](https://github.com/xamdoo/mpi-lemon-connectivity-cognition/issues/2).

## Reproducibility principles

- Freeze the confirmatory analysis plan after the data audit and before outcome
  modeling.
- Keep transformations inside cross-validation folds.
- Report effect sizes and uncertainty, including null or unstable results.
- Treat missing individual motion as a major limitation of any ready-matrix result.
- Record data versions, checksums, software versions, random seeds, exclusions, and
  every material analysis decision.

## License

Project code and original documentation are released under the [MIT License](LICENSE).
External datasets, atlases, and code retain their original licenses and citation
requirements.
