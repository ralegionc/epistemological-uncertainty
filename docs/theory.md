# The theory behind each stratum

This document explains what each stratum models, why it is modeled that way, and
where the model is a simplification. The goal is that a reader can decide whether
the abstraction is appropriate for their problem, and replace it where it is not.

## Why strata rather than a single error term

The usual move in uncertainty quantification is to collect every source of doubt
into one distribution over outcomes and report its spread. That is coherent when
the sources are exchangeable and independent. The sources here are neither. They
are ordered by how far they sit from the raw physical interaction, and each one
is conditional on the layers outside it.

Model uncertainty is uncertainty given the data. But the data is only meaningful
relative to the sampling process that produced it. The sampling process is only
interpretable relative to the labels applied to it. And the labels are only
coherent given the measurements they discretize. Ignoring an outer layer does
not remove it. It silently inflates the estimates of the inner layers while
presenting them as clean.

This is why the strata compound multiplicatively in the didactic proxy and why
the propagation nests them in order. The order is measurement, then labeling,
then sampling, then model. You label the disturbed measurement, you sample the
labeled population, and you fit the model to the sample.

## Stratum 1: measurement coupling (Δ_m)

**What it models.** The gap between the latent quantity you care about and the
quantity your instrument actually records. Every measurement is a physical
interaction. A thermometer exchanges heat with the thing it measures. A survey
question changes the respondent's state by asking it. An fMRI voxel reports a
blood-oxygen signal that is two or three causal steps removed from the neural
activity of interest, and integrates over roughly a hundred thousand neurons.

**How it is modeled.** As a systematic per-feature coupling plus independent
per-cell noise:

```
x_obs = x_true * (1 + b) + a + e
```

Here `b` is a multiplicative gain error and `a` is an additive offset. Both are
drawn once per Monte Carlo iteration and shared across every row. `e` is
per-cell noise drawn independently.

**Why this way.** The essential property is that `b` and `a` are systematic. A
biased instrument is biased for every reading, so collecting more rows does not
average the bias away. This is what distinguishes an outer-stratum problem from
ordinary noise, and it is modeled explicitly by sharing the draw across rows.
The per-cell term `e` is included as a contrast, and the test suite verifies
that a pure-`e` perturbation shrinks with sample size while a pure-`b`
perturbation does not.

**Where it simplifies.** Real coupling is rarely a clean affine map. Instruments
saturate, have dead zones, respond nonlinearly, and interact with the thing
they measure in ways that depend on the measured value itself. If you know the
transfer function of your instrument, replace the perturbation with it.

## Stratum 2: labeling ontology (Δ_l)

**What it models.** The fact that a label is a choice, not an observation.
"Employed," "depressed," "high risk," and "toxic" are all discretizations of
something continuous or contested. The same underlying reality, coded under a
different but equally defensible definition, is a different dataset with
different conclusions.

**How it is modeled.** Two modes.

In pluralism mode, you supply several alternative label vectors, each
representing a defensible coding scheme, and each draw samples among them. This
is the right model when the alternatives are discrete and namable, such as two
diagnostic manuals or two annotation guidelines.

In boundary mode, you supply a continuous latent score and a nominal threshold,
and the stratum jitters the threshold. Rows near the moving line flip. This is
the right model when the label is a cutoff on a continuum and the disagreement is
about where the line goes, such as a risk score dichotomized into high and low.

**Why this way.** Both modes make the same point in different shapes: the label
is doing work that the data does not justify, and the amount of that work is
measurable by how much the output moves when the labeling is varied. If
predictions are stable across schemes, the labeling contributes little. If they
are not, the label is carrying the conclusion, and that must be disclosed.

**Where it simplifies.** Label disagreement in practice is structured. Annotators
disagree systematically, not randomly, and the disagreement often correlates
with the features. A threshold jitter treats the boundary as uniformly uncertain,
which understates disagreement in some regions and overstates it in others. If
you have multiple annotators, model their actual disagreement.

## Stratum 3: generative process (Δ_s)

**What it models.** The mechanism that determined who ended up in the data.
Participation is never random. It depends on who was reachable, who consented,
what the incentive to participate was, what platform was used, what time of day,
and how the request was framed. The sample's joint distribution can differ from
the target population's in ways that are invisible from inside the sample.

**How it is modeled.** As a tilt of the per-row weights along an observed
coverage covariate:

```
w_i  ∝  exp(t * z_i)
```

`z` is a standardized covariate along which the sample may be unrepresentative,
and `t` is a tilt strength drawn each iteration from a zero-mean distribution.
The tilt can run in either direction, because the direction of a coverage gap is
usually unknown. Alternatively you can supply explicit reweighting schemes toward
named target populations, and the stratum samples among them.

**Why this way.** Reweighting is the standard tool for correcting known coverage
gaps. The move here is to treat the correction itself as uncertain: rather than
apply one reweighting and trust it, sample a range of plausible reweightings and
propagate the range. The zero-mean tilt encodes genuine ignorance about which
way the gap runs.

**Where it simplifies, and the hard limit.** The tilt can only express gaps along
covariates you have measured. The gaps along unmeasured covariates are exactly
the ones this method cannot reach. This is not a defect of the implementation. It
is the structure of the problem. The honest response is to treat an undocumented
generative process as a source of unquantified uncertainty rather than to fill
the blank with zero, and the reporting layer flags high outer-stratum dominance
partly for this reason.

## Stratum 4: model uncertainty (Δ_θ)

**What it models.** The familiar inner layer. Given the data, how much would the
estimate move under a different draw of the same process? This is finite-sample
uncertainty, and it is what the bootstrap, the posterior, and conformal
prediction quantify.

**How it is modeled.** As a bootstrap. Rows are resampled with replacement and
the estimand is re-evaluated. Severity below one holds a fraction of rows fixed,
shrinking the resampling variance toward zero, which is useful mainly for
isolating the other strata.

**Why this way.** The bootstrap is model-agnostic, which matches the library's
model-agnostic estimand interface. Where a proper posterior is available it is
better, and you can express it by writing an estimand that samples from the
posterior internally.

**Its real role in the framework.** This stratum is deliberately placed last and
named the innermost layer. Its purpose in the framework is less to be computed
accurately and more to be put in its place. It is the layer that conventional
practice treats as the whole of uncertainty, and the framework's contribution is
to show how small its share can be once the outer three are admitted.

## The composition

The propagation runs the full stack once, then each stratum in isolation. The
full-stack spread is the honest interval. The isolated spreads give a first-order
variance decomposition, and the gap between the full-stack variance and the sum
of isolated variances is reported as interaction. Positive interaction means the
strata amplify each other, which is common: a labeling choice and a coverage gap
that point the same way compound rather than cancel.

The didactic proxy in the web page uses `EUQ = 1 - product(1 - delta_i)` because
it is the simplest closed form that has the right qualitative behavior, namely
that reducing any single layer has diminishing returns when the others are large.
The Python library does not use this formula. It measures the compounding
directly by propagation, which is the honest version of the same idea.
