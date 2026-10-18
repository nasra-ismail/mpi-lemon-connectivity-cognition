# Notebooks

Use one focused, numbered notebook per issue:

```text
01_cohort_audit.ipynb
02_matrix_alignment.ipynb
03_single_matrix_qc.ipynb
04_batch_matrix_qc.ipynb
05_network_labels.ipynb
06_network_ordering.ipynb
07_network_segregation.ipynb
08_age_group_comparison.ipynb
09_segregation_sensitivity.ipynb
10_behavioral_inventory.ipynb
11_outcome_covariates.ipynb
12_participant_merge.ipynb
13_missingness_audit.ipynb
```

Each notebook should state:

- linked GitHub issue;
- scientific or technical question;
- input and output paths;
- data and software versions;
- assumptions and analysis decisions;
- quality checks;
- concise conclusion and next dependency.

Keep reusable functions in `src/lemon_connectivity/`, not duplicated across notebooks.