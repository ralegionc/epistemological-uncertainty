"""Tests for the EUQ package.

These check the properties that make the framework *mean* something, not just
that the code runs:

  * an inert stratum (severity 0) contributes essentially no uncertainty;
  * measurement coupling does NOT average away as n grows (it is systematic);
  * aleatoric-only noise DOES shrink with n (sanity contrast);
  * the sampling stratum only bites when the estimand respects weights;
  * variance shares are a valid probability vector;
  * the full-stack interval contains the baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

from euq import (
    EUQPipeline,
    LabelingStratum,
    MeasurementStratum,
    ModelStratum,
    SamplingStratum,
)
from euq.strata import PipelineState


def prevalence(X, y, w):
    return float(np.average(y, weights=w))


def mean_feature0(X, y, w):
    return float(np.average(X[:, 0], weights=w))


def make_data(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_inert_stratum_is_quiet():
    X, y = make_data(300)
    pipe = EUQPipeline(
        prevalence, [MeasurementStratum(severity=0.0)], n_draws=200, random_state=0
    )
    res = pipe.run(X, y)
    assert res.total_std()[0] < 1e-9


def test_measurement_coupling_does_not_average_away():
    """Systematic coupling std should NOT shrink much with n."""
    stds = {}
    for n in (200, 4000):
        X, y = make_data(n, seed=1)
        pipe = EUQPipeline(
            mean_feature0,
            [MeasurementStratum(severity=0.6, gain_sd=0.2, offset_sd=0.2, noise_sd=0.0)],
            n_draws=300,
            random_state=3,
        )
        stds[n] = pipe.run(X, y).total_std()[0]
    # With only systematic terms, larger n should not collapse the spread.
    assert stds[4000] > 0.5 * stds[200]


def test_pure_aleatoric_noise_shrinks_with_n():
    """Contrast: per-cell noise on a mean should shrink ~1/sqrt(n)."""
    stds = {}
    for n in (200, 4000):
        X, y = make_data(n, seed=2)
        pipe = EUQPipeline(
            mean_feature0,
            [MeasurementStratum(severity=0.6, gain_sd=0.0, offset_sd=0.0, noise_sd=0.5)],
            n_draws=300,
            random_state=4,
        )
        stds[n] = pipe.run(X, y).total_std()[0]
    assert stds[4000] < stds[200]


def test_sampling_needs_weights():
    """A weight-ignoring estimand shows ~zero sampling contribution."""
    X, y = make_data(400)
    coverage = X[:, 0].copy()

    def ignores_weights(X, y, w):
        return float(y.mean())

    pipe = EUQPipeline(
        ignores_weights,
        [SamplingStratum(severity=0.8, coverage_covariate=coverage)],
        n_draws=200,
        random_state=0,
    )
    assert pipe.run(X, y).total_std()[0] < 1e-9

    def respects_weights(X, y, w):
        return float(np.average(y, weights=w))

    pipe2 = EUQPipeline(
        respects_weights,
        [SamplingStratum(severity=0.8, coverage_covariate=coverage)],
        n_draws=200,
        random_state=0,
    )
    assert pipe2.run(X, y).total_std()[0] > 1e-4


def test_variance_shares_are_a_distribution():
    X, y = make_data(400)
    cov = X[:, 1].copy()
    strata = [
        MeasurementStratum(0.4),
        SamplingStratum(0.4, coverage_covariate=cov),
        ModelStratum(1.0),
    ]
    res = EUQPipeline(prevalence, strata, n_draws=200, random_state=0).run(X, y)
    shares = res.variance_share()
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in shares.values())


def test_interval_contains_baseline():
    X, y = make_data(500)
    res = EUQPipeline(
        prevalence, [ModelStratum(1.0)], n_draws=400, random_state=0
    ).run(X, y)
    lo, hi = res.interval(0.95)
    assert lo[0] <= res.baseline[0] <= hi[0]


def test_labeling_pluralism_runs():
    X, y = make_data(300)
    alt = [1 - y, y]  # two label schemes
    res = EUQPipeline(
        prevalence,
        [LabelingStratum(0.5, alternative_labels=alt)],
        n_draws=200,
        random_state=0,
    ).run(X, y)
    assert res.total_std()[0] > 0.0


def test_severity_validation():
    with pytest.raises(ValueError):
        MeasurementStratum(severity=1.5)


def test_pipeline_state_roundtrip():
    X, y = make_data(50)
    st = PipelineState.initial(X, y)
    st2 = st.copy()
    st2.X[0, 0] = 999.0
    assert st.X[0, 0] != 999.0  # copy is independent
