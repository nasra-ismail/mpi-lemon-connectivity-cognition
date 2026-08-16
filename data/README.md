# Local data directory

The contents of this directory are intentionally excluded from Git.

Recommended layout:

```text
data/
├── external/
│   ├── Curvature-FCN-Aging/
│   └── lemon_behavioral/
│       └── Behavioural_Data_MPILMBB_LEMON.zip
├── interim/
└── processed/
```

`external/` contains unchanged downloads, `interim/` contains auditable intermediate
files, and `processed/` contains analysis-ready local tables or matrices.

Do not place the complementary `behavioral_data_MPILMBB_v3.zip` archive in the LEMON
folder. If retained for another project, store it under a clearly separate path.

Before every commit, run:

```bash
git status --short
git check-ignore -v data/external/example_file
```

Only provenance, checksums, code, and permitted non-sensitive derived summaries
belong in the repository.
