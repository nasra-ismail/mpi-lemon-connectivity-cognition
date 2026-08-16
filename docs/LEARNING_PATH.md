# Learning path

Use each task to learn only the concepts needed for the next artifact.

| Phase | Learn | Prove it by producing |
|---|---|---|
| P0 | environments, paths, Git safety | passing tests and ignored local data |
| P1 | IDs, edge lists, matrices, QC | cohort audit and 225-file QC table |
| P2 | atlases, networks, masks, segregation | labeled heatmap and segregation table |
| P3 | dictionaries, merging, missingness | analytic sample and participant flow |
| P4 | regression, uncertainty, graph metrics | descriptive and adjusted results |
| P5 | leakage, ridge, nested CV, permutation | honest out-of-fold model comparison |
| P6 | NIfTI, confounds, parcellation | 5–10 validated Schaefer-100 matrices |
| P7 | reporting and reproducibility | coherent public portfolio release |

## Immediate checkpoint

Complete issues #3 and #2, then #4 through #8. Do not start Ricci curvature, raw fMRI
preprocessing, or cognition prediction before the complete matrices and IDs pass QC.

At the end of P1 you should be able to explain:

- why 225 participants do not imply a continuous ID sequence;
- why a Schaefer-200 undirected edge list has 19,900 rows;
- how one edge list becomes a symmetric 200 × 200 matrix;
- why the diagonal is excluded from network summaries; and
- why complete FCM and density-thresholded FCN products answer different questions.
