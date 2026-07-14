"""Epistemological Uncertainty Quantification (EUQ).

Honest uncertainty that includes not just statistical noise but the uncertainty
about whether your data means what you think it means.

Four strata, propagated through to model outputs:

    Δ_m  measurement coupling   the instrument disturbs what it measures
    Δ_l  labeling ontology      labels are theory-laden discretizations
    Δ_s  generative process     who appears in the data, and why
    Δ_θ  model uncertainty      classical finite-sample noise (the inner layer)

Quickstart
----------
>>> import numpy as np
>>> from euq import EUQPipeline, MeasurementStratum, ModelStratum
>>> X = np.random.default_rng(0).normal(size=(200, 3))
>>> y = (X[:, 0] > 0).astype(int)
>>> estimand = lambda X, y, w: np.average(y, weights=w)
>>> pipe = EUQPipeline(estimand,
...                    [MeasurementStratum(0.4), ModelStratum(1.0)],
...                    n_draws=300, random_state=0)
>>> result = pipe.run(X, y)
>>> from euq import format_report
>>> print(format_report(result, "prevalence"))  # doctest: +SKIP
"""
from __future__ import annotations

from .pipeline import EUQPipeline, EUQResult
from .report import epistemic_dominance, format_report
from .strata import (
    LabelingStratum,
    MeasurementStratum,
    ModelStratum,
    PipelineState,
    SamplingStratum,
    Stratum,
)

__version__ = "0.1.0"

__all__ = [
    "EUQPipeline",
    "EUQResult",
    "Stratum",
    "PipelineState",
    "MeasurementStratum",
    "LabelingStratum",
    "SamplingStratum",
    "ModelStratum",
    "format_report",
    "epistemic_dominance",
    "__version__",
]
