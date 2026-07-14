"""Synthetic datasets for the worked examples.

Each generator returns features, labels, and any auxiliary arrays (latent
scores, coverage covariates, alternative labelings) needed to configure the
strata. They are deliberately simple and self-contained so the examples run in
seconds on CPU with no downloads.
"""
from __future__ import annotations

import numpy as np

__all__ = ["make_screening_dataset", "make_group_gap_dataset"]


def make_screening_dataset(n: int = 800, random_state: int = 0):
    """A binary screening task with a genuinely contested cutoff.

    A latent 'risk' score drives a continuous outcome. The nominal label
    thresholds the score at its 70th percentile ('high risk'), but that cutoff
    is a convention — hence a natural home for the labeling stratum. A coverage
    covariate (call it 'access') is correlated with who ends up in the sample,
    feeding the sampling stratum.

    Returns
    -------
    X : (n, 4) features
    y : (n,) nominal binary label
    latent : (n,) continuous risk score
    threshold : float nominal cutpoint on latent
    coverage : (n,) coverage covariate
    alt_labels : list of two alternative label vectors (stricter / looser cutoff)
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, 4))
    coef = np.array([1.2, -0.8, 0.5, 0.3])
    latent = X @ coef + rng.normal(scale=0.6, size=n)

    threshold = float(np.percentile(latent, 70))
    y = (latent >= threshold).astype(int)

    # Coverage covariate correlated with feature 0 (e.g. access to screening).
    coverage = X[:, 0] + rng.normal(scale=0.5, size=n)

    # Two equally defensible alternative cutoffs.
    strict = (latent >= np.percentile(latent, 80)).astype(int)
    loose = (latent >= np.percentile(latent, 60)).astype(int)
    alt_labels = [strict, loose]

    return X, y, latent, threshold, coverage, alt_labels


def make_group_gap_dataset(n: int = 1000, gap: float = 0.15, random_state: int = 1):
    """A two-group dataset for estimating a rate difference between groups.

    Group membership is in column 0 (0/1). The 'true' positive-rate gap is
    ``gap``. A coverage covariate makes one group over-sampled, so an unweighted
    estimate is biased unless the sampling stratum is allowed to reweight.

    Returns
    -------
    X : (n, 3) features, column 0 is group membership
    y : (n,) binary outcome
    coverage : (n,) coverage covariate driving over-sampling of group 1
    """
    rng = np.random.default_rng(random_state)
    group = rng.integers(0, 2, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    base_rate = 0.35
    p = base_rate + gap * group + 0.05 * x1
    p = np.clip(p, 0.02, 0.98)
    y = (rng.random(n) < p).astype(int)

    X = np.column_stack([group.astype(float), x1, x2])
    # Group 1 is easier to reach -> over-represented sample.
    coverage = group + rng.normal(scale=0.4, size=n)
    return X, y, coverage
