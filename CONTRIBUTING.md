# Contributing

This is a learning-first scientific repository. Clarity and traceability matter more
than speed.

## Workflow

1. Choose the next unblocked issue from `docs/ROADMAP.md`.
2. Comment on the issue with the intended notebook, code, and output files.
3. Create a branch named `issue-<number>-short-description`.
4. Keep reusable logic in `src/lemon_connectivity/`; notebooks should call it.
5. Run `python -m pytest -q` and `python -m ruff check .`.
6. Open a pull request that links the issue and describes data and analysis decisions.
7. Merge only when the issue's definition of done is satisfied.

## Notebook rules

- Use numbered filenames and run notebooks from top to bottom.
- Begin with the question, inputs, outputs, and software/data versions.
- Use repository-relative paths through `pathlib.Path`.
- Never hide warnings without explaining them.
- Restart the kernel and run all cells before review.
- Do not place reusable algorithms only inside a notebook.

## Data rules

Never commit participant-level source data, MRI files, behavioral archives, Yadav
connectivity files, secrets, or credentials. Small derived tables may be shared only
after their license, privacy risk, and necessity have been checked.

## Scientific rules

- Do not choose covariates through univariable p-value screening.
- Do not tune graph construction or feature selection on the final evaluation data.
- Distinguish explanatory inference from out-of-sample prediction.
- Label post-hoc analyses as exploratory.
- Avoid causal or longitudinal claims from cross-sectional data.
