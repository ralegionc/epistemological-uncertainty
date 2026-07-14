"""Example 2 — a group rate difference dominated by the sampling stratum.

We estimate the difference in positive-outcome rates between two groups. The
true gap is 0.15, but group 1 is over-sampled, so an unweighted estimate is
biased. Crucially, the *direction and size* of the correction depends on how we
believe the target population differs from the sample — which we do not know.
The sampling stratum turns that ignorance into an interval.

This example shows a case where the outer strata dominate and the conventional
error bar is actively misleading.

Run:
    python examples/02_group_rate_gap.py
"""
from __future__ import annotations

import numpy as np

from euq import (
    EUQPipeline,
    ModelStratum,
    SamplingStratum,
    format_report,
)
from euq.datasets import make_group_gap_dataset


def main() -> None:
    X, y, coverage = make_group_gap_dataset(n=1000, gap=0.15, random_state=1)
    group = X[:, 0]

    def rate_gap(X, y, w):
        g = X[:, 0]
        w1 = w[g == 1]
        w0 = w[g == 0]
        r1 = np.average(y[g == 1], weights=w1) if w1.sum() > 0 else 0.0
        r0 = np.average(y[g == 0], weights=w0) if w0.sum() > 0 else 0.0
        return float(r1 - r0)

    strata = [
        SamplingStratum(severity=0.7, coverage_covariate=coverage, tilt_sd=1.2),
        ModelStratum(severity=1.0),
    ]
    pipe = EUQPipeline(rate_gap, strata, n_draws=500, random_state=7)
    result = pipe.run(X, y)

    print(format_report(result, label="group positive-rate gap (r1 - r0)"))
    print()
    print("True generating gap was 0.150.")
    print("Notice the interval is wide and asymmetric: we cannot rule out a")
    print("substantially smaller or larger gap without external knowledge of the")
    print("target population — exactly the uncertainty a bootstrap alone hides.")


if __name__ == "__main__":
    main()
