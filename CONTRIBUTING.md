# Contributing

This is a learning-first scientific repository. Clarity and traceability matter more
than speed.

## Workflow

1. Choose the next unblocked issue from `docs/ROADMAP.md`.
2. Comment on the issue with the intended notebook, code, and output files.
3. Create a branch named `issue-<number>-short-description`.
4. Keep reusable logic in `src/lemon_connectivity/`; notebooks should call it.
5. Install the Git hooks once with `pre-commit install`.
6. Commit changes; the hooks run Ruff, pytest, and commit-message validation.
7. Run `pre-commit run --all-files` before opening a pull request.
8. Open a pull request that links the issue and describes data and analysis decisions.
9. Merge only when the issue's definition of done is satisfied.

Use the complete [pull-request and review workflow](docs/CONTRIBUTION_WORKFLOW.md)
before merging any change into `main`.

## Automated local checks

Install the development dependencies and both Git hook types once after cloning:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Keep the project virtual environment activated when committing so the pytest hook
uses the dependencies installed for this repository.

Before each commit, pre-commit runs Ruff with automatic fixes and then runs the
pytest suite. If Ruff changes a file, review the change, stage it, and commit again.
The commit remains blocked while Ruff or pytest reports a failure.

Run all pre-commit checks manually when needed:

```bash
pre-commit run --all-files
```

GitHub Actions repeats Ruff and pytest for every pull request and every push to
`main`. Do not routinely bypass local hooks with `--no-verify`; centralized checks
remain the repository-level safeguard.

## Commit messages

Commit messages follow the Conventional Commits structure:

```text
type(optional-scope): short imperative description
```

Common types are `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `chore`, and
`ci`. Keep the scope short and omit it when it does not add useful information.

Examples:

```text
feat(identifiers): add participant ID canonicalization
refactor(notebooks): use shared alignment pipeline
test(alignment): cover duplicate canonical IDs
style: resolve Ruff findings
```

Use `cz commit` for an interactive prompt, or use `git commit` normally. The
Commitizen `commit-msg` hook rejects messages that do not follow the convention.

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
