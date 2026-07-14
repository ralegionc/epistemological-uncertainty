"""The propagation engine.

An :class:`EUQPipeline` wraps a user-supplied *estimand* (any function that
maps a weighted sample to a scalar or vector of interest) together with a stack
of strata. Running it draws perturbations from each active stratum, re-evaluates
the estimand, and returns the distribution of outputs alongside a per-stratum
variance decomposition.

The estimand signature is::

    estimand(X, y, sample_weight) -> float | np.ndarray

It may fit a model internally, compute a group difference, evaluate a metric —
anything, as long as it respects ``sample_weight``. Respecting the weights is
what lets the sampling stratum bite; an estimand that ignores them will simply
show zero sampling contribution, which is itself honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .strata import PipelineState, Stratum

__all__ = ["EUQResult", "EUQPipeline"]

Estimand = Callable[[np.ndarray, np.ndarray, np.ndarray], "float | np.ndarray"]


@dataclass
class EUQResult:
    """Outcome of a propagation run.

    Attributes
    ----------
    baseline : np.ndarray
        The estimand on the untouched data (all strata inert).
    samples : np.ndarray of shape (n_draws, k)
        Estimand values under the full active stratum stack.
    per_stratum : dict[str, np.ndarray]
        For each stratum symbol, the estimand samples when *only* that stratum
        is active (isolated contribution).
    stratum_names : dict[str, str]
        Symbol -> human-readable name.
    """

    baseline: np.ndarray
    samples: np.ndarray
    per_stratum: dict[str, np.ndarray]
    stratum_names: dict[str, str]

    # ---- aggregate summaries -------------------------------------------------
    def total_std(self) -> np.ndarray:
        """Std of the full-stack output distribution (the honest error bar)."""
        return self.samples.std(axis=0)

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Empirical central interval of the full-stack distribution."""
        lo = (1 - level) / 2 * 100
        hi = (1 + level) / 2 * 100
        return (
            np.percentile(self.samples, lo, axis=0),
            np.percentile(self.samples, hi, axis=0),
        )

    def stratum_std(self) -> dict[str, np.ndarray]:
        """Isolated std contributed by each stratum (all else inert)."""
        return {sym: s.std(axis=0) for sym, s in self.per_stratum.items()}

    def variance_share(self) -> dict[str, float]:
        """Fraction of total isolated variance attributable to each stratum.

        Uses isolated (one-at-a-time) variances, normalized to sum to 1. This
        is a first-order attribution; interactions are reported separately via
        :meth:`interaction_share`.
        """
        var = {sym: float(np.mean(s.var(axis=0))) for sym, s in self.per_stratum.items()}
        total = sum(var.values())
        if total == 0:
            return {sym: 0.0 for sym in var}
        return {sym: v / total for sym, v in var.items()}

    def interaction_share(self) -> float:
        """How much full-stack variance exceeds the sum of isolated variances.

        A positive value means the strata amplify each other (the coupling is
        super-additive); near zero means they are roughly independent. Reported
        as a fraction of the full-stack variance.
        """
        full = float(np.mean(self.samples.var(axis=0)))
        iso = sum(float(np.mean(s.var(axis=0))) for s in self.per_stratum.values())
        if full == 0:
            return 0.0
        return (full - iso) / full

    def summary(self) -> dict:
        lo, hi = self.interval()
        return {
            "baseline": np.round(self.baseline, 6).tolist(),
            "total_std": np.round(self.total_std(), 6).tolist(),
            "interval_95": [np.round(lo, 6).tolist(), np.round(hi, 6).tolist()],
            "variance_share": {k: round(v, 4) for k, v in self.variance_share().items()},
            "interaction_share": round(self.interaction_share(), 4),
            "stratum_std": {
                k: np.round(v, 6).tolist() for k, v in self.stratum_std().items()
            },
        }


class EUQPipeline:
    """Compose strata around an estimand and propagate epistemological uncertainty.

    Parameters
    ----------
    estimand : callable
        ``estimand(X, y, sample_weight) -> float | np.ndarray``.
    strata : sequence of Stratum
        Applied in order on each draw. Order matters: measurement coupling
        should precede labeling (you label the disturbed measurement), which
        precedes sampling, which precedes the model bootstrap — mirroring the
        nesting of the framework.
    n_draws : int
        Monte Carlo iterations.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        estimand: Estimand,
        strata: Sequence[Stratum],
        n_draws: int = 500,
        random_state: int | None = None,
    ):
        self.estimand = estimand
        self.strata = list(strata)
        self.n_draws = int(n_draws)
        self.random_state = random_state

    # ---- internals -----------------------------------------------------------
    def _eval(self, state: PipelineState) -> np.ndarray:
        out = self.estimand(state.X, state.y, state.weight)
        return np.atleast_1d(np.asarray(out, dtype=float))

    def _run_stack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        active: Sequence[Stratum],
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = []
        for _ in range(self.n_draws):
            state = PipelineState.initial(X, y)
            for st in active:
                state = st.perturb(state, rng)
            out.append(self._eval(state))
        return np.asarray(out)

    # ---- public API ----------------------------------------------------------
    def run(self, X: np.ndarray, y: np.ndarray) -> EUQResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        baseline = self._eval(PipelineState.initial(X, y))

        active = [s for s in self.strata if s.active]
        samples = self._run_stack(X, y, active, rng)

        per_stratum: dict[str, np.ndarray] = {}
        names: dict[str, str] = {}
        for st in active:
            per_stratum[st.symbol] = self._run_stack(X, y, [st], rng)
            names[st.symbol] = st.name

        return EUQResult(
            baseline=baseline,
            samples=samples,
            per_stratum=per_stratum,
            stratum_names=names,
        )
