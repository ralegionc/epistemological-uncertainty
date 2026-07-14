# Epistemological Uncertainty Quantification

Honest uncertainty that includes not just statistical noise, but the uncertainty
about whether your data means what you think it means.

Most uncertainty quantification answers one question: given this data, how much
would my estimate move if I collected a different sample of the same size? That
is a real and important question, and it is the one that confidence intervals,
posteriors, and the bootstrap are built to answer. It is also the innermost of
at least four questions that separate a number in a table from a fact about the
world.

This project treats uncertainty as having four strata. Each stratum is a source
of doubt that sits outside the usual statistical machinery, and each one is
propagated through to the model's output rather than assumed away.

| Symbol | Stratum | The doubt it captures |
| --- | --- | --- |
| Δ_m | Measurement coupling | The instrument disturbs what it measures. The recorded value is not the latent quantity of interest. |
| Δ_l | Labeling ontology | Labels are theory-laden choices. The same phenomenon coded under a different but defensible definition gives a different dataset. |
| Δ_s | Generative process | Who appears in the data, and why. The sampling mechanism produces a distribution that may differ from the target population in ways you cannot see from inside the sample. |
| Δ_θ | Model uncertainty | Classical epistemic and aleatoric noise. Parameter posteriors, conformal intervals, the bootstrap. This is what error bars usually report. |

The central claim is that these strata do not add. They compound. A Bayesian
neural network applied to a biased survey will produce an extremely precise
estimate of what the biased sample believes, and call it an estimate of the
population. The precision is real. It is also beside the point, because it is
conditioned on assumptions the error bar never mentions.

## What the library does

`euq` wraps any estimand you can write as a function and propagates the four
strata through it by nested Monte Carlo. On each draw it perturbs the features
(measurement), the labels (labeling), the sampling weights (generative process),
and the rows (model), then re-evaluates your estimand. The spread of the
resulting distribution is the propagated epistemological uncertainty, and it is
decomposed so you can see which stratum owns which share.

The estimand signature is deliberately minimal:

```python
def estimand(X, y, sample_weight) -> float | np.ndarray:
    ...
```

Anything that respects `sample_weight` works: a fitted model's flag rate, a
group difference, a metric, a coefficient. Respecting the weights is what lets
the sampling stratum bite. An estimand that ignores them will report zero
sampling contribution, which is itself an honest answer.

## Install

```bash
git clone https://github.com/ralegionc/epistemological-uncertainty.git
cd epistemological-uncertainty
pip install -e ".[examples]"
```

The core depends only on NumPy. The worked examples add scikit-learn.

## Quickstart

```python
import numpy as np
from euq import (
    EUQPipeline, MeasurementStratum, LabelingStratum,
    SamplingStratum, ModelStratum, format_report,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(800, 4))
latent = X @ np.array([1.2, -0.8, 0.5, 0.3]) + rng.normal(scale=0.6, size=800)
threshold = np.percentile(latent, 70)
y = (latent >= threshold).astype(int)
coverage = X[:, 0] + rng.normal(scale=0.5, size=800)

def prevalence(X, y, w):
    return float(np.average(y, weights=w))

pipe = EUQPipeline(
    prevalence,
    strata=[
        MeasurementStratum(severity=0.5),
        LabelingStratum(severity=0.6, latent_score=latent, threshold=threshold),
        SamplingStratum(severity=0.6, coverage_covariate=coverage),
        ModelStratum(severity=1.0),
    ],
    n_draws=400,
    random_state=42,
)

result = pipe.run(X, y)
print(format_report(result, label="prevalence"))
```

The report states the point estimate, the full-stack interval, the share of
variance owned by each stratum, the degree of cross-stratum interaction, and a
plain-language verdict on whether the statistical error bar can be trusted on
its own.

## A worked result

`examples/01_screening_classifier.py` fits a logistic regression to a binary
"high risk" label and asks what fraction of the population it flags. The nominal
answer is about 0.29. Under all four strata:

```
point estimate      : +0.2863
full-stack 95% band : [+0.0310, +0.7256]

uncertainty budget (first-order variance share)
  Δ_m  measurement coupling   ·······················    0.0%
  Δ_l  labeling ontology      █████████████···········  52.4%
  Δ_s  generative process     ███████████·············  46.6%
  Δ_θ  model uncertainty      ························    1.0%

VERDICT: the model-only error bar is cosmetic.
```

The bootstrap alone gives a band of width 0.075. The honest band is width 0.695,
about nine times wider, and nearly all of that width comes from the labeling and
sampling strata. The conventional error bar was not wrong about the sampling
noise. It was silent about everything else.

`examples/02_group_rate_gap.py` shows the opposite case, where the outer strata
are present but do not dominate, and the verdict correctly reports that the
statistical band is a reasonable summary. The point of the framework is not to
always widen the interval. It is to make the widening honest when it is
warranted and to say so when it is not.

## The interactive explainer

`web/index.html` is a self-contained page (no build step, no dependencies) that
lets you drag each stratum's severity and watch the honest band swallow the
conventional one. It includes preset scenarios for a happiness survey, an fMRI
study, a hiring model, and a climate proxy reconstruction, each with a different
exposure profile. Open the file directly or serve it with GitHub Pages.

## How the strata are modeled

Each stratum is a small, inspectable perturbation. The design choices matter
more than the parameter values, so they are documented in `docs/theory.md`. The
one property worth stating up front is that outer-stratum perturbations are
systematic, not independent noise. A measurement-coupling bias is drawn once per
Monte Carlo iteration and applied to every row, so it does not average away as
the sample grows. This is the difference between "my thermometer is noisy" and
"my thermometer reads two degrees high," and it is the reason more data does not
fix an outer-stratum problem. The test suite checks this directly.

## Limitations

This is a framework and a reference implementation, not a finished measurement
of any real system. Several things are worth being explicit about.

The severity parameters are not estimated from data. They encode a judgment
about how badly each assumption might fail, and that judgment is itself
uncertain. The right way to use them is as the input to a sensitivity analysis:
vary them, and report how conclusions move.

The perturbation models are parametric and simple by choice. Real measurement
coupling is rarely a clean gain-and-offset error, real label disagreement is
rarely a threshold jitter, and real coverage gaps are rarely a single-covariate
tilt. The library is structured so you can supply your own perturbations, which
is where domain knowledge should enter.

The variance decomposition is first-order. It attributes variance to each
stratum in isolation and reports the leftover interaction as a separate number,
but it does not compute full Sobol indices. When the interaction term is large,
the isolated shares should be read as a lower bound.

Finally, the sampling stratum can only express uncertainty about coverage gaps
along covariates you have observed. The gaps you cannot see are, by
construction, the ones this method cannot quantify. That the unquantifiable
remainder exists is part of the honest answer, and it is why the framework
treats an undocumented generative process as a red flag rather than a blank to
be filled with zero.

## Repository layout

```
euq/            core library (strata, propagation, reporting, synthetic data)
examples/       runnable worked cases
tests/          property tests for the framework's meaning, not just its plumbing
web/            self-contained interactive explainer
docs/           the theory behind each stratum's model
```

## A note on how this was built

The framework, the code, and this write-up were developed with the assistance of
a large language model, used as a thinking and drafting partner. The design
decisions, the choice of what to model and what to leave out, and the framing are
mine, and any errors are as well.

## License

MIT. See `LICENSE`.
