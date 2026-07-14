"""Honest reporting.

The output of EUQ is deliberately *not* a single confidence interval. It is a
report that states the point estimate, the full-stack interval, the share of
uncertainty owned by each stratum, and a plain-language verdict on whether the
statistical error bar can be trusted on its own.

The verdict thresholds are conventions, chosen to be conservative. They are
exposed as module constants so a user can override them for their domain — and
the fact that they are conventions is itself an instance of Stratum 2.
"""
from __future__ import annotations

import numpy as np

from .pipeline import EUQResult

__all__ = ["format_report", "epistemic_dominance"]

# The share of total variance owned by the outer three strata above which the
# conventional (model-only) error bar is judged misleading. A convention.
OUTER_DOMINANCE_WARN = 0.40
OUTER_DOMINANCE_CRIT = 0.65

_OUTER = {"Δ_m", "Δ_l", "Δ_s"}


def epistemic_dominance(result: EUQResult) -> float:
    """Fraction of first-order variance owned by the outer three strata.

    High values mean the data-meaning gap dominates statistical noise, so a
    model-only error bar understates true uncertainty.
    """
    shares = result.variance_share()
    return sum(v for k, v in shares.items() if k in _OUTER)


def _bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def format_report(result: EUQResult, label: str = "estimand") -> str:
    """Render a plain-text epistemic report suitable for logs or a README."""
    lines: list[str] = []
    base = result.baseline
    lo, hi = result.interval(0.95)
    tstd = result.total_std()

    scalar = base.shape[0] == 1

    lines.append("=" * 66)
    lines.append("EPISTEMOLOGICAL UNCERTAINTY REPORT")
    lines.append("=" * 66)
    lines.append(f"estimand: {label}")
    lines.append("")

    if scalar:
        lines.append(f"point estimate      : {base[0]:+.4f}")
        lines.append(f"full-stack 95% band : [{lo[0]:+.4f}, {hi[0]:+.4f}]")
        lines.append(f"full-stack std      : {tstd[0]:.4f}")
    else:
        lines.append(f"point estimate (vec): {np.round(base, 4).tolist()}")
        lines.append(f"full-stack std (vec): {np.round(tstd, 4).tolist()}")
    lines.append("")

    lines.append("uncertainty budget (first-order variance share)")
    lines.append("-" * 66)
    shares = result.variance_share()
    sstd = result.stratum_std()
    for sym in ["Δ_m", "Δ_l", "Δ_s", "Δ_θ"]:
        if sym not in shares:
            continue
        name = result.stratum_names.get(sym, sym)
        frac = shares[sym]
        std_i = float(np.mean(sstd[sym]))
        lines.append(f"  {sym}  {name:<22} {_bar(frac)} {frac*100:5.1f}%  (σ={std_i:.4f})")
    lines.append("")

    inter = result.interaction_share()
    lines.append(f"cross-stratum interaction : {inter*100:+.1f}% of full-stack variance")
    if inter > 0.05:
        lines.append("    strata amplify each other; treat isolated shares as a lower bound.")
    lines.append("")

    dom = epistemic_dominance(result)
    lines.append(f"outer-stratum dominance   : {dom*100:.1f}%")
    lines.append("-" * 66)
    if dom < OUTER_DOMINANCE_WARN:
        lines.append("VERDICT: statistical error bar is a reasonable summary.")
        lines.append("  Outer strata are documented and small relative to model noise.")
    elif dom < OUTER_DOMINANCE_CRIT:
        lines.append("VERDICT: qualify the error bar.")
        lines.append("  A meaningful share of uncertainty comes from what the data means,")
        lines.append("  not how much of it you have. Report the assumption exposure.")
    else:
        lines.append("VERDICT: the model-only error bar is cosmetic.")
        lines.append("  The data-meaning gap dominates. Run sensitivity analysis across")
        lines.append("  labeling and sampling assumptions before drawing conclusions.")
    lines.append("=" * 66)
    return "\n".join(lines)
