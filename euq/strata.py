"""Stratum definitions for Epistemological Uncertainty Quantification.

Each stratum is a source of uncertainty that sits *outside* the usual
statistical machinery. A stratum does not report a number directly; it knows
how to *perturb* the pipeline state (features, labels, sampling weights, or the
resampling of rows) in a way that reflects one honest-but-unverifiable
assumption about what the data means.

The propagation engine (``euq.pipeline``) draws perturbations from each active
stratum, re-runs the downstream estimand, and collects the resulting
distribution of outputs. The spread of that distribution *is* the propagated
epistemological uncertainty.

Design note
-----------
The critical property we preserve is that outer-stratum perturbations are
*systematic*, not *independent noise*. A measurement-coupling bias is drawn
once per Monte Carlo iteration and applied to every row. That is what makes it
irreducible: collecting more rows does not average it away, because every row
shares the same draw. Contrast with aleatoric noise, which is drawn per cell
and shrinks with n. Getting this distinction right is the whole point of the
framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "PipelineState",
    "Stratum",
    "MeasurementStratum",
    "LabelingStratum",
    "SamplingStratum",
    "ModelStratum",
]


@dataclass
class PipelineState:
    """The mutable state passed through the stratum stack on each MC draw.

    Attributes
    ----------
    X : np.ndarray of shape (n, d)
        Feature / measurement matrix.
    y : np.ndarray of shape (n,)
        Labels or targets.
    weight : np.ndarray of shape (n,)
        Per-row sampling weights. Start uniform; the sampling stratum tilts
        these to reweight toward alternative target populations.
    index : np.ndarray of shape (n,)
        Row indices into the original data. The model stratum resamples these
        with replacement to express finite-sample (bootstrap) uncertainty.
    """

    X: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    index: np.ndarray

    @classmethod
    def initial(cls, X: np.ndarray, y: np.ndarray) -> "PipelineState":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = X.shape[0]
        return cls(
            X=X.copy(),
            y=y.copy(),
            weight=np.ones(n, dtype=float),
            index=np.arange(n),
        )

    def copy(self) -> "PipelineState":
        return replace(
            self,
            X=self.X.copy(),
            y=self.y.copy(),
            weight=self.weight.copy(),
            index=self.index.copy(),
        )


class Stratum:
    """Base class. A stratum perturbs a :class:`PipelineState` in place-safe way.

    Subclasses implement :meth:`perturb`. ``severity`` is a 0..1 scalar that
    scales how aggressively the stratum's assumption is stressed; 0 means the
    stratum is effectively inert (the identity perturbation).
    """

    name: str = "stratum"
    symbol: str = "Δ"

    def __init__(self, severity: float = 0.5, active: bool = True):
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        self.severity = float(severity)
        self.active = bool(active)

    def perturb(self, state: PipelineState, rng: np.random.Generator) -> PipelineState:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(severity={self.severity:.2f}, active={self.active})"


class MeasurementStratum(Stratum):
    """Stratum 1 — the instrument disturbs what it measures.

    Models the gap between the latent quantity of interest and what the
    instrument records as a *systematic* per-feature coupling plus independent
    per-cell noise:

        x_obs = x_true * (1 + b) + a + e

    where ``b`` (multiplicative gain error) and ``a`` (additive offset) are
    drawn once per MC iteration and shared across all rows (systematic, does
    not average away), and ``e`` is per-cell aleatoric noise (shrinks with n).

    Parameters
    ----------
    gain_sd : float
        Std of the multiplicative coupling ``b`` at severity 1.0.
    offset_sd : float
        Std of the additive offset ``a`` (in units of each feature's own std)
        at severity 1.0.
    noise_sd : float
        Std of per-cell noise ``e`` (in units of each feature's std) at
        severity 1.0.
    feature_mask : sequence of bool, optional
        Which feature columns the coupling applies to. Defaults to all.
    """

    name = "measurement coupling"
    symbol = "Δ_m"

    def __init__(
        self,
        severity: float = 0.5,
        gain_sd: float = 0.15,
        offset_sd: float = 0.10,
        noise_sd: float = 0.10,
        feature_mask: Sequence[bool] | None = None,
        active: bool = True,
    ):
        super().__init__(severity=severity, active=active)
        self.gain_sd = gain_sd
        self.offset_sd = offset_sd
        self.noise_sd = noise_sd
        self.feature_mask = feature_mask

    def perturb(self, state: PipelineState, rng: np.random.Generator) -> PipelineState:
        X = state.X
        n, d = X.shape
        col_sd = X.std(axis=0)
        col_sd[col_sd == 0] = 1.0

        mask = (
            np.ones(d, dtype=bool)
            if self.feature_mask is None
            else np.asarray(self.feature_mask, dtype=bool)
        )

        s = self.severity
        # Systematic: one draw per feature, shared across all rows.
        b = rng.normal(0.0, self.gain_sd * s, size=d) * mask
        a = rng.normal(0.0, self.offset_sd * s, size=d) * col_sd * mask
        # Aleatoric: independent per cell.
        e = rng.normal(0.0, self.noise_sd * s, size=(n, d)) * col_sd * mask

        state.X = X * (1.0 + b) + a + e
        return state


class LabelingStratum(Stratum):
    """Stratum 2 — labels are theory-laden discretizations.

    Two modes:

    ``pluralism``
        You supply a list of alternative, equally defensible label vectors
        (e.g. the same phenomenon coded under different definitions). Each MC
        draw samples one scheme. Severity blends between the original labels and
        the sampled alternative: at severity 1.0 the alternative is used whole,
        at 0.5 roughly half the rows adopt it.

    ``boundary``
        You supply a continuous latent score and a nominal threshold. The
        stratum expresses uncertainty about *where the line goes* by jittering
        the threshold; rows near the moving boundary flip. This is the honest
        model for "the cutoff between diagnosed and not-diagnosed is a
        convention, not a fact."

    Only one mode is active per instance.
    """

    name = "labeling ontology"
    symbol = "Δ_l"

    def __init__(
        self,
        severity: float = 0.5,
        alternative_labels: Sequence[np.ndarray] | None = None,
        latent_score: np.ndarray | None = None,
        threshold: float | None = None,
        threshold_sd: float = 0.5,
        active: bool = True,
    ):
        super().__init__(severity=severity, active=active)
        if alternative_labels is not None and latent_score is not None:
            raise ValueError("choose one mode: alternative_labels OR latent_score")
        self.alternative_labels = (
            [np.asarray(a) for a in alternative_labels]
            if alternative_labels is not None
            else None
        )
        self.latent_score = (
            np.asarray(latent_score, dtype=float) if latent_score is not None else None
        )
        self.threshold = threshold
        self.threshold_sd = threshold_sd

    def perturb(self, state: PipelineState, rng: np.random.Generator) -> PipelineState:
        s = self.severity
        if self.alternative_labels is not None:
            alt = self.alternative_labels[rng.integers(len(self.alternative_labels))]
            alt = alt[state.index] if alt.shape[0] != state.y.shape[0] else alt
            # Blend: each row adopts the alternative with prob = severity.
            adopt = rng.random(state.y.shape[0]) < s
            new_y = state.y.copy()
            new_y[adopt] = alt[adopt]
            state.y = new_y
            return state

        if self.latent_score is not None and self.threshold is not None:
            score = self.latent_score[state.index]
            score_sd = score.std() or 1.0
            jitter = rng.normal(0.0, self.threshold_sd * s * score_sd)
            state.y = (score >= (self.threshold + jitter)).astype(state.y.dtype)
            return state

        # No mode configured -> identity.
        return state


class SamplingStratum(Stratum):
    """Stratum 3 — who appears in the data, and why.

    Expresses uncertainty about the divergence between the sample's generative
    process and the target population by tilting per-row weights along an
    observed covariate (the "coverage axis"). If the sample over-represents,
    say, urban and young respondents, the true target might up-weight the
    under-covered tail; we do not know by how much, so we sample the tilt.

        w_i  ∝  exp(t * z_i)

    where ``z`` is a standardized coverage covariate and ``t`` is a tilt
    strength drawn each iteration from N(0, tilt_sd * severity). ``t`` can be
    positive or negative because we genuinely do not know which direction the
    coverage gap runs.

    Alternatively, supply ``target_weight_schemes`` — explicit reweightings
    toward named alternative populations — and the stratum samples among them.
    """

    name = "generative process"
    symbol = "Δ_s"

    def __init__(
        self,
        severity: float = 0.5,
        coverage_covariate: np.ndarray | None = None,
        tilt_sd: float = 1.0,
        target_weight_schemes: Sequence[np.ndarray] | None = None,
        active: bool = True,
    ):
        super().__init__(severity=severity, active=active)
        self.coverage_covariate = (
            np.asarray(coverage_covariate, dtype=float)
            if coverage_covariate is not None
            else None
        )
        self.tilt_sd = tilt_sd
        self.target_weight_schemes = (
            [np.asarray(w, dtype=float) for w in target_weight_schemes]
            if target_weight_schemes is not None
            else None
        )

    def perturb(self, state: PipelineState, rng: np.random.Generator) -> PipelineState:
        s = self.severity
        n = state.weight.shape[0]

        if self.target_weight_schemes is not None:
            scheme = self.target_weight_schemes[
                rng.integers(len(self.target_weight_schemes))
            ]
            scheme = scheme[state.index] if scheme.shape[0] != n else scheme
            # Blend uniform <-> scheme by severity.
            blended = (1.0 - s) * state.weight + s * scheme
            state.weight = blended / blended.mean()
            return state

        if self.coverage_covariate is not None:
            z = self.coverage_covariate[state.index]
            z = (z - z.mean()) / (z.std() or 1.0)
            t = rng.normal(0.0, self.tilt_sd * s)
            w = np.exp(t * z)
            state.weight = state.weight * (w / w.mean())
            return state

        return state


class ModelStratum(Stratum):
    """Stratum 4 — classical finite-sample (epistemic) uncertainty.

    The familiar inner layer: resample rows with replacement (the bootstrap) so
    that the estimand is refit on a perturbed sample. Positioned *last* so the
    framework makes explicit that this is the only layer conventional error bars
    usually capture.

    Severity scales the effective bootstrap intensity: at severity 1.0 it is a
    standard bootstrap; below that, a fraction of rows are held at their
    original position, shrinking the resampling variance toward zero.
    """

    name = "model uncertainty"
    symbol = "Δ_θ"

    def __init__(self, severity: float = 0.5, active: bool = True):
        super().__init__(severity=severity, active=active)

    def perturb(self, state: PipelineState, rng: np.random.Generator) -> PipelineState:
        n = state.index.shape[0]
        if self.severity <= 0:
            return state
        n_resample = int(round(self.severity * n))
        keep = n - n_resample
        boot = rng.integers(0, n, size=n_resample)
        take = np.concatenate([np.arange(keep), boot])
        state.X = state.X[take]
        state.y = state.y[take]
        state.weight = state.weight[take]
        state.index = state.index[take]
        return state
