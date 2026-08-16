# Project plan

## Primary objective

Estimate whether resting-state functional-network segregation is associated with,
and adds out-of-sample predictive information for, fluid-reasoning performance in
MPI-LEMON after accounting for age group, sex, education, and head motion when an
individual motion measure is available.

## Secondary objectives

1. Reproduce a younger–older difference in network segregation.
2. Examine strength, modularity, global efficiency, and participation coefficient.
3. Compare demographic-only, brain-only, and combined prediction.
4. Learn and validate the transformation from preprocessed fMRI to connectivity.

## Boundaries

- This is a cross-sectional observational study.
- The Yadav Schaefer-200 matrices form the initial learning pass.
- The Yadav `FCN` density-thresholded graphs are not the default input.
- The final independent pilot uses Schaefer-100 and a documented denoising strategy.
- If individual motion is unavailable for the ready-made matrices, related inference
  remains exploratory and the limitation is not repaired by group-level motion values.

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
