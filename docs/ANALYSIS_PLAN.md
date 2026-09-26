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

The whole-brain segregation definition is:

```text
S = (mean within-network FC - mean between-network FC) / mean within-network FC
```

The edge definitions, denominators, and alternatives below govern implementation
in issue #12 and sensitivity analyses in issue #14.

## Edge-handling policy — issue #11

Policy identifier: `edge-policy-v1`. Proposed on 2026-09-26; adoption is recorded by
merging the pull request linked to [issue #11](https://github.com/nasra-ismail/mpi-lemon-connectivity-cognition/issues/11).
Agree these definitions before calculating metrics or testing age/cognition
associations. The wider analysis plan remains provisional until issue #20. This
record does not claim formal preregistration or that earlier results were unseen.

### Inputs and common rules

- Begin with the complete, QC-approved Schaefer-200 `DATA/FCM` correlation matrices
  and the validated seven-network atlas labels. Yadav's thresholded `DATA/FCN`
  products are not inputs to these definitions.
- Keep the original signed correlation matrix `R`, including its reconstructed
  unit diagonal, unchanged. Create separate working copies for each definition.
  Use the same retained inventory and ROI-to-network correspondence throughout.
- Exclude self-connections from every metric. Set working-matrix diagonals to zero,
  and explicitly exclude them from edge masks and averaging denominators. A zero
  diagonal alone is not sufficient to exclude self-connections from a mean.
- Preserve signed correlations in QC distributions; count each off-diagonal pair
  once. A negative correlation is not, by itself, a QC failure or evidence of an
  inhibitory anatomical connection.
- Do not convert the complete matrix to absolute values, binarize it, rescale each
  participant by their maximum weight, or impute invalid edges as zeros.

### Weight definitions and Fisher transformation

For distinct ROIs `i` and `j`, define:

```text
R_positive[i, j] = max(R[i, j], 0)
Z[i, j] = arctanh(R[i, j])
Z_positive[i, j] = max(Z[i, j], 0)
```

All working diagonals are zero. Apply Fisher transformation once, to individual
off-diagonal correlations before averaging. Never transform the diagonal, which
contains `r = 1`, or apply Fisher transformation to an already transformed matrix.
Fisher z values are weights on a different scale; they are not bounded by -1 and 1.

Input correlations must be finite and within [-1, 1], as checked in matrix QC.
Fisher-based definitions additionally require every off-diagonal value to lie
strictly inside (-1, 1). Exact endpoints, including `r = -1` that would later be
zeroed, require an explicit transformation-failure record and source investigation.
Do not silently clip endpoints or replace infinities. Raw-correlation definitions
remain mathematically valid at the endpoints; record their distinct eligibility
and use the common eligible sample for direct sensitivity comparisons.

### Definitions by measure

"Primary" for a graph measure identifies its main construction rule; graph
analyses remain secondary to the project's segregation question.

| Measure | Primary weights | Prespecified sensitivity weights | Self-connections |
|---|---|---|---|
| Whole-brain segregation | `Z_positive` | `R_positive`, signed `Z`, signed `R` | Excluded from both edge masks and denominators |
| Weighted strength | `R_positive` | `Z_positive` | Excluded from each node's weight sum |
| Weighted modularity | `R_positive` | `Z_positive`, with the same objective and search settings | No self-loops in the input graph |
| Weighted global efficiency | `R_positive`, converted to lengths as below | `Z_positive`, with the same length rule | Only pairs with `i != j` contribute |

For primary segregation, Fisher transformation followed by zeroing negative
weights follows the construction described by Chan et al. (2014) [1]. It measures
segregation of positive functional relationships. Keeping signed alternatives
allows the effect of discarding negative weights to be examined explicitly.

For the graph measures, positive raw correlations retain a bounded weight scale
and give the selected positive-weight algorithms valid inputs. These are project
choices, not a claim that one edge policy is optimal for every graph measure.
The Fisher-weighted alternatives isolate the effect of nonlinear weight scaling
while retaining the same positive-edge topology. Their absolute strength and
efficiency values are on different scales; compare their associations and
uncertainty accordingly, not as interchangeable numerical measurements.

### Segregation means and denominators

Let `g[i]` be the canonical network assigned to ROI `i`. Use unique upper-triangle
pairs, excluding the diagonal:

```text
within_pairs  = pairs (i, j) with i < j and g[i] == g[j]
between_pairs = pairs (i, j) with i < j and g[i] != g[j]
W = mean(A[i, j] over within_pairs)
B = mean(A[i, j] over between_pairs)
S = (W - B) / W
```

Here `A` is the working matrix for the selected definition. Each selected edge has
equal weight. This is a pooled whole-brain definition: larger networks contribute
more within-network pairs. It is not an unweighted average of seven network means
or seven network-specific segregation ratios. This pooling rule is a project
specification rather than a claim to reproduce every detail of Chan et al.

If network `k` contains `n_k` ROIs, document:

```text
N_within  = sum(n_k * (n_k - 1) / 2)
N_between = 200 * 199 / 2 - N_within
N_within + N_between = 19,900
```

The two masks must be disjoint and cover all unique off-diagonal pairs. Masks
depend on labels, not signs or nonzero status. Zeroed negative edges remain in
their original masks and denominators; taking the mean of only positive edges
would be a different definition. Require both masks to be nonempty.

Require a positive within-network baseline and use a prespecified near-zero guard
`W <= 1e-12` for every segregation definition. Record the score as unavailable
(`NaN`) with a reason when this guard is triggered, or if required values are
nonfinite. In signed alternatives, division by a negative `W` is mathematically
possible, but this policy does not interpret it as segregation relative to a
positive within-network baseline; distinguish that case from a zero or near-zero
denominator in the status record. Keep the participant's row rather than
substituting zero or silently removing it. This is a denominator rule, not an edge
threshold. A defined score can be negative when `B > W`. Signed alternatives can
exceed one when `B < 0`; do not clip segregation to [0, 1].

### Segregation sensitivity analyses — issue #14

Run all three alternatives below alongside the primary definition, with identical
ROI labels, masks, pooling, and denominator guards. Report eligibility and failure
counts for each definition and compare estimates on the common eligible sample.

| Definition ID | Transformation | Negative edges | Role |
|---|---|---|---|
| `seg_z_positive` | Fisher z before averaging | Set to zero; retain in denominator | Primary |
| `seg_r_positive` | None | Set to zero; retain in denominator | Isolates Fisher transformation choice |
| `seg_z_signed` | Fisher z before averaging | Retain signed values | Isolates negative-edge choice |
| `seg_r_signed` | None | Retain signed values | Joint alternative |

Report every predefined result with its uncertainty. Do not promote an alternative
because its association is larger or its p-value is smaller. The primary
definition remains primary when results are null or disagree across definitions.

### Graph-specific requirements — issues #24 and #25

- **Strength:** node strength is the sum of incident weights over all other ROIs,
  including zeros. A whole-brain mean strength averages those sums over all 200
  ROIs. Isolated nodes contribute zero; the Fisher-weighted sensitivity uses the
  same sum and averaging convention.
- **Modularity:** use the undirected weighted positive modularity objective with
  resolution `gamma = 1` [2]. Negative edges become absent edges, and zero-weight
  pairs are not inserted as graph edges. Keep all 200 nodes, including isolates.
  A graph with zero total edge weight has undefined modularity and must be flagged.
  Issue #24 must record the community-search algorithm, software version, fixed
  seed schedule, number of runs, and rule for summarizing runs before calculation;
  apply the same settings to both weight definitions. Signed modularity requires
  a separately documented signed objective and a prospective policy amendment.
- **Global efficiency:** for each strictly positive weight `w`, set edge length
  to `1 / w`. Zero-weight pairs are absent connections, never zero-length edges.
  Calculate shortest-path distances using these lengths and average `1 / d(i, j)`
  over all `N * (N - 1)` ordered distinct-node pairs, with `N = 200`. Unreachable
  pairs contribute zero. Retain isolates and disconnected components in the
  denominator; an edgeless graph therefore has efficiency zero. Use this rule for
  both raw and Fisher positive weights. NetworkX's `global_efficiency` ignores
  weights [3], so it does not implement this weighted definition.

The participation coefficient planned in #25 must also use inputs with no
self-loops. Its module definition and sensitivity rules belong in #24 before it
is calculated.

### Thresholding, provenance, and amendments

Neither the primary definitions nor the alternatives above apply an additional
absolute-correlation cutoff or proportional-density threshold. Positive-weight
definitions explicitly truncate negatives at zero; this does not equalize graph
density across participants. Retain all positive weights, including weak ones,
and record the resulting positive-edge density. Avoid an arbitrary density choice
for this initial analysis and avoid silently inheriting Yadav's `FCN` settings.

If density-based sensitivity analyses are later needed under #24, commit their
exact grid, ranking/tie rules, and disconnected-graph handling before running them.
They are not already specified or authorized as part of this policy. Further
methods may be proposed prospectively; changes after viewing affected results
must be labeled exploratory and cannot silently replace the primary definition.

Downstream outputs must identify the policy version, definition ID, source-data
version or checksum, atlas mapping, and software versions. Record metric-specific
failures without changing the earlier QC inventory silently. Keep participant
records under ignored local data directories; share aggregate summaries only as
permitted by the data-management rules. Log material amendments in the
[project decision log](PROJECT_PLAN.md#decision-log).

Issue #11 specifies the method. Issue #12 implements and tests segregation, #14
runs its sensitivities, and #24/#25 finalize and implement secondary graphs. No
new notebook or metric computation is required for this documentation issue.

### Method references

1. Chan MY et al. (2014). Decreased segregation of brain systems across the healthy
   adult lifespan. *PNAS*, 111, E4997-E5006.
   [Full text and methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4246293/).
2. NetworkX: [weighted modularity definition](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html)
   and [Louvain settings and self-loop behavior](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html).
   Record the installed version when selecting the implementation in #24.
3. NetworkX: [global efficiency definition and unweighted implementation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.efficiency_measures.global_efficiency.html).
   See its reference to Latora and Marchiori (2001),
   [Efficient behavior of small-world networks](https://doi.org/10.1103/PhysRevLett.87.198701).

## Secondary explanatory analyses

- younger versus older segregation difference adjusted for sex;
- strength, modularity, global efficiency, and participation coefficient;
- graph-feature associations with age group and LPS-2;
- prespecified multiplicity control across each family of secondary tests.

## Prediction

Compare dummy, demographic-only, brain-only, and combined models using identical
participants and outer folds. Ridge regression is the primary high-dimensional model.
All imputation, scaling, feature reduction, and hyperparameter selection occur inside
the training data of repeated nested cross-validation. Report MAE, R², prediction
correlation, uncertainty across repeats, and a full-pipeline permutation test.

## Diagnostics and sensitivity

- participant and matrix QC before modeling;
- residual, influence, nonlinearity, heteroskedasticity, and collinearity checks;
- prespecified alternative segregation and graph-construction rules;
- inclusion/exclusion and motion sensitivity where data permit;
- complete reporting of null, negative, and unstable results.

## Interpretation limits

No analysis establishes causality, within-person aging, treatment effects, or broad
generalizability beyond this healthy German volunteer sample.
