# Provisional analysis plan

This plan is provisional until the behavioral and missing-data audit is complete. It
must be frozen and versioned in issue #20 before outcome modeling.

## Primary question

Is whole-brain system segregation associated with LPS-2 fluid reasoning after
prespecified adjustment?

Provisional model:

```text
LPS-2 = β0 + β1(segregation) + β2(age group) + β3(sex)
        + β4(education) + β5(mean framewise displacement) + ε
```

The exact LPS-2 score, education coding, transformation, missing-data handling, and
motion variable will be selected from the dictionaries and audit—not from p-values.
If individual motion is unavailable for the ready-made matrices, the learning-pass
model is explicitly exploratory.

## Primary network feature

The finalized whole-brain system segregation definition is:

```text
S = (mean within-network FC - mean between-network FC)
    / mean within-network FC
```

The connectivity preprocessing follows the policy finalized in issue #11:

* diagonal entries are excluded;
* off-diagonal Pearson correlations are transformed using the Fisher \(z\)-transformation;
* negative Fisher \(z\)-values are replaced with zero;
* zeroed values remain included in the corresponding network means; and
* each unique undirected edge contributes equally to the pooled within-network and between-network means.

Within-network and between-network edges are defined from the validated Schaefer-200 canonical network assignments using the upper triangle of the connectivity matrix. The two edge groups are mutually exclusive and together cover all unique ROI-to-ROI connections.

If the pooled within-network mean is zero,

$$
W = 0
$$

the system segregation score is undefined. Such cases remain in the participant-level analysis ledger with an explicit calculation status and reason rather than being assigned an arbitrary value.

The primary implementation is recorded as `edge-policy-v1` with segregation definition `seg_z_positive`.

## Secondary explanatory analyses

* younger versus older segregation difference adjusted for sex;
* strength, modularity, global efficiency, and participation coefficient;
* graph-feature associations with age group and LPS-2;
* prespecified multiplicity control across each family of secondary tests.

## Prediction

Compare dummy, demographic-only, brain-only, and combined models using identical
participants and outer folds. Ridge regression is the primary high-dimensional model.
All imputation, scaling, feature reduction, and hyperparameter selection occur inside
the training data of repeated nested cross-validation. Report MAE, R², prediction
correlation, uncertainty across repeats, and a full-pipeline permutation test.

## Diagnostics and sensitivity

* participant and matrix QC before modeling;
* residual, influence, nonlinearity, heteroskedasticity, and collinearity checks;
* prespecified alternative segregation and graph-construction rules;
* inclusion/exclusion and motion sensitivity where data permit;
* complete reporting of null, negative, and unstable results.

## Interpretation limits

No analysis establishes causality, within-person aging, treatment effects, or broad
generalizability beyond this healthy German volunteer sample.
