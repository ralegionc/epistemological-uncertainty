"""Example 1 — a screening classifier under all four strata.

We fit a logistic regression to predict a binary 'high risk' label, and ask a
simple downstream question: what fraction of the population does the model flag?
That single number carries all four strata:

    Δ_m  the features are noisy, systematically biased measurements
    Δ_l  the 'high risk' cutoff is a convention, not a fact
    Δ_s  the sample over-covers the high-access tail
    Δ_θ  finite training data

Run:
    python examples/01_screening_classifier.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from euq import (
    EUQPipeline,
    LabelingStratum,
    MeasurementStratum,
    ModelStratum,
    SamplingStratum,
    format_report,
)
from euq.datasets import make_screening_dataset


def main() -> None:
    X, y, latent, threshold, coverage, alt_labels = make_screening_dataset(
        n=800, random_state=0
    )

    def flag_rate(X, y, w):
        """Weighted fraction the fitted model flags as high risk."""
        if len(np.unique(y)) < 2:
            return float(np.average(y, weights=w))
        model = LogisticRegression(max_iter=200)
        model.fit(X, y, sample_weight=w)
        pred = model.predict(X)
        return float(np.average(pred, weights=w))

    strata = [
        MeasurementStratum(severity=0.5, gain_sd=0.15, offset_sd=0.10, noise_sd=0.10),
        LabelingStratum(
            severity=0.6, latent_score=latent, threshold=threshold, threshold_sd=0.6
        ),
        SamplingStratum(severity=0.6, coverage_covariate=coverage, tilt_sd=1.0),
        ModelStratum(severity=1.0),
    ]

    pipe = EUQPipeline(flag_rate, strata, n_draws=400, random_state=42)
    result = pipe.run(X, y)

    print(format_report(result, label="fraction flagged high-risk"))
    print()
    print("Contrast — model-only error bar (Δ_θ alone):")
    model_only = EUQPipeline(
        flag_rate, [ModelStratum(1.0)], n_draws=400, random_state=42
    ).run(X, y)
    lo, hi = model_only.interval()
    lo_f, hi_f = result.interval()
    print(f"  model-only 95% band : [{lo[0]:.3f}, {hi[0]:.3f}]  width {hi[0]-lo[0]:.3f}")
    print(f"  full-stack 95% band : [{lo_f[0]:.3f}, {hi_f[0]:.3f}]  width {hi_f[0]-lo_f[0]:.3f}")
    ratio = (hi_f[0] - lo_f[0]) / max(hi[0] - lo[0], 1e-9)
    print(f"  honest band is {ratio:.1f}x wider than the conventional one.")


if __name__ == "__main__":
    main()
