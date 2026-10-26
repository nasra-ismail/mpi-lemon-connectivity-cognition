# Project plan

## Primary objective

Estimate whether resting-state functional-network segregation is associated with fluid-reasoning performance in MPI-LEMON after accounting for age group, sex, education, and head motion when an individual motion measure is available.

## Secondary objectives

1. Estimate the younger–older difference in network segregation.
2. Examine strength, modularity, global efficiency, and participation coefficient.
3. Compare demographic-only, brain-only, and combined prediction.
4. Learn and validate the transformation from preprocessed fMRI to connectivity.

## Boundaries

- This is a cross-sectional observational study.
- The Yadav Schaefer-200 matrices form the initial learning pass.
- The Yadav `FCN` density-thresholded graphs are not the default input.
- The final pilot will use independently constructed Schaefer-100 matrices and a documented denoising strategy.
- If individual-level motion measurements are unavailable for the ready-made matrices, related inference remains exploratory; group-level motion values do not resolve this limitation.

## Phases

| Phase | Output |
|---|---|
| P0 — Foundation | Working environment, protected data layout, tested scaffold |
| P1 — Matrix audit | Validated cohort and 225 FCM files |
| P2 — Segregation | Atlas mapping and participant segregation scores |
| P3 — Behavior | Validated LPS-2/covariate table and participant flow |
| P4 — Statistics | Descriptives, explanatory models, and graph analyses |
| P5 — Prediction | Leakage-safe nested-CV model comparison |
| P6 — fMRI pilot | Independently constructed Schaefer-100 matrices |
| P7 — Portfolio | Final figures, report, reproducibility audit, and release |

## Decision log

Record material choices before seeing the result they could affect.

| Date | Decision | Rationale | Issue |
|---|---|---|---|
| 2026-08-16 | Begin with complete Yadav `FCM` edge lists | Avoid silently inheriting density thresholding | #1 |
| 2026-08-16 | Use the LEMON-specific behavioral archive | The MPILMBB v3 archive is a complementary protocol | #1 |
| 2026-08-16 | Treat graph and ML analyses as secondary | Keep one clear explanatory primary question | #1 |
| 2026-09-26 | Propose `edge-policy-v1`: positive Fisher-z segregation; positive raw-correlation strength, modularity, and weighted efficiency; no self-connections or additional density threshold | Make metric-specific edge choices explicit before calculation; adoption occurs on merge of the #11 PR | [#11](https://github.com/nasra-ismail/mpi-lemon-connectivity-cognition/issues/11) |
| 2026-09-26 | Specify pooled edge means and three segregation sensitivities (raw positive, Fisher signed, raw signed); graph sensitivities use positive Fisher weights | Expose sign and transformation dependence without selecting results by significance; keep fixed masks, zero-edge denominators, and failure records | [#11](https://github.com/nasra-ismail/mpi-lemon-connectivity-cognition/issues/11) |
| 2026-10-18 | For Issue #13, compare whole-brain `system_segregation` using `seg_z_positive` under `edge-policy-v1` between elderly and young participants, estimating the elderly-minus-young difference with OLS adjusted for recorded sex (young and female reference categories), using HC3 robust standard errors, a t-based 95% confidence interval, and a two-sided t-test | Specify the model and inference before examining age-group comparison results. This is a cross-sectional observational comparison, not evidence of causal or within-person aging effects; if individual-level motion adjustment is unavailable, interpret the findings as exploratory | [#13](https://github.com/nasra-ismail/mpi-lemon-connectivity-cognition/issues/13) |

Full definitions, rationale, and implementation boundaries are recorded in the
[analysis-plan edge policy](ANALYSIS_PLAN.md#edge-handling-policy--issue-11).