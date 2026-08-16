## Summary

<!-- In 2–5 bullets, explain what changed and what the reviewer should focus on. -->

-

## Linked issue

Closes #

## Type of change

- [ ] Bug fix
- [ ] Analysis or statistical-method change
- [ ] New feature or reusable function
- [ ] Data/QC change
- [ ] Documentation only
- [ ] Refactor or maintenance with no intended result change

## Scope for review

<!-- Name the files, decisions, or outputs that need the closest review. -->

## Scientific and data impact

- Data source and version:
- Participant inclusion/exclusion changed: No / Yes — explain below
- Outcome, exposure, covariate, atlas, confound, or model changed: No / Yes — explain below
- Analysis status: Primary / Secondary / Exploratory / Not applicable
- Expected result change: None / Yes / Unknown

## Evidence

<!-- Link or attach non-sensitive tables, figures, screenshots, or logs. Never attach participant-level data. -->

## Author self-review

- [ ] The change is limited to the linked issue
- [ ] I reviewed the complete **Files changed** tab
- [ ] `python -m pytest -q` passes
- [ ] `python -m ruff check .` passes
- [ ] Modified notebooks restart and run from top to bottom
- [ ] New reusable logic has tests
- [ ] No raw MRI, behavioral, connectivity, participant-level, credential, or oversized files are tracked
- [ ] Data versions, exclusions, random seeds, and analysis decisions are recorded where relevant
- [ ] Figures and tables show labels, denominators, uncertainty, and reproducible source code
- [ ] Documentation and the decision log are updated where relevant

## Reviewer checklist

<!-- The reviewer completes this section before approval. Use N/A where appropriate. -->

- [ ] The change matches the linked issue and does not add unexplained scope
- [ ] Code and notebook logic are understandable and reproducible
- [ ] Data merges, exclusions, and dimensions have explicit checks
- [ ] Statistical choices were not selected only because of favorable results
- [ ] Cross-validation transformations remain inside training folds
- [ ] Conclusions match the design, effect sizes, uncertainty, and limitations
- [ ] No sensitive or prohibited data are exposed
- [ ] Automated checks pass and review conversations are resolved

## Limitations and follow-up

<!-- State unresolved scientific, data, motion, implementation, or interpretation limitations. -->

## Merge gate

- [ ] Pull request is no longer a draft
- [ ] Required checks pass
- [ ] Required review is complete, when another reviewer is available
- [ ] All review conversations are resolved
- [ ] The branch is up to date with `main` if GitHub requires it
