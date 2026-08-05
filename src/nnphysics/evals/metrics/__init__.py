"""The metrics and the registry that names them.

Importing this module registers every metric, so a name in a suite resolves without the
caller knowing which module implements it.

The regime gap the plan asks for is not here. It consumes two sets of results rather than
a rollout, so it cannot be a `Metric` without lying about what a metric is; it lives in
`nnphysics.evals.runner` and is computed over whatever metrics the suite ran.
"""

from __future__ import annotations

from nnphysics.evals.metrics.base import (
    DEFAULT_ERROR_THRESHOLDS,
    METRICS,
    MetricContext,
    MetricFactory,
    build_metrics,
    relative_error,
)
from nnphysics.evals.metrics.distribution import QUANTILES, DistributionDrift
from nnphysics.evals.metrics.error import NEVER_REACHED, OneStepError, RolloutErrorGrowth
from nnphysics.evals.metrics.invariants import REFERENCE_MARGIN, InvariantDrift
from nnphysics.evals.metrics.symmetry import SymmetryViolation

__all__ = [
    "DEFAULT_ERROR_THRESHOLDS",
    "DEFAULT_METRICS",
    "METRICS",
    "NEVER_REACHED",
    "QUANTILES",
    "REFERENCE_MARGIN",
    "DistributionDrift",
    "InvariantDrift",
    "MetricContext",
    "MetricFactory",
    "OneStepError",
    "RolloutErrorGrowth",
    "SymmetryViolation",
    "build_metrics",
    "relative_error",
]

METRICS.add("one_step_error", OneStepError)
METRICS.add("rollout_error", RolloutErrorGrowth)
METRICS.add("invariant_drift", InvariantDrift)
METRICS.add("symmetry_violation", SymmetryViolation)
METRICS.add("distribution_drift", DistributionDrift)

DEFAULT_METRICS = (
    "one_step_error",
    "rollout_error",
    "invariant_drift",
    "symmetry_violation",
    "distribution_drift",
)
"""Every metric, in the order a report reads best: what one step says, what a rollout
says, and then the three that say what a rollout cannot."""
