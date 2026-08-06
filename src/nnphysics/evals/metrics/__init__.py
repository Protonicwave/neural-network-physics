"""The metrics and the registry that names them.

Importing this module registers every metric, so a name in a suite resolves without the
caller knowing which module implements it.

The regime gap the plan asks for is not here. It consumes two sets of results rather than
a rollout, so it cannot be a `Metric` without lying about what a metric is; it lives in
`nnphysics.evals.runner` and is computed over whatever metrics the suite ran.
"""

from __future__ import annotations

from nnphysics.evals.metrics.base import (
    DEFAULT_CALIBRATION_LEVELS,
    DEFAULT_ERROR_THRESHOLDS,
    DEFAULT_TRUST_THRESHOLD,
    METRICS,
    MetricContext,
    MetricFactory,
    build_metrics,
    normalised_magnitude,
    relative_error,
)
from nnphysics.evals.metrics.calibration import (
    NOT_DETERMINED,
    ONE_SIGMA_COVERAGE,
    UNDEFINED_CORRELATION,
    Calibration,
)
from nnphysics.evals.metrics.distribution import QUANTILES, DistributionDrift
from nnphysics.evals.metrics.error import NEVER_REACHED, OneStepError, RolloutErrorGrowth
from nnphysics.evals.metrics.invariants import REFERENCE_MARGIN, InvariantDrift
from nnphysics.evals.metrics.resolution import ResolutionGeneralisation
from nnphysics.evals.metrics.symmetry import SymmetryViolation

__all__ = [
    "DEFAULT_CALIBRATION_LEVELS",
    "DEFAULT_ERROR_THRESHOLDS",
    "DEFAULT_METRICS",
    "DEFAULT_TRUST_THRESHOLD",
    "METRICS",
    "NEVER_REACHED",
    "NOT_DETERMINED",
    "ONE_SIGMA_COVERAGE",
    "QUANTILES",
    "REFERENCE_MARGIN",
    "UNDEFINED_CORRELATION",
    "Calibration",
    "DistributionDrift",
    "InvariantDrift",
    "MetricContext",
    "MetricFactory",
    "OneStepError",
    "ResolutionGeneralisation",
    "RolloutErrorGrowth",
    "SymmetryViolation",
    "build_metrics",
    "normalised_magnitude",
    "relative_error",
]

METRICS.add("one_step_error", OneStepError)
METRICS.add("rollout_error", RolloutErrorGrowth)
METRICS.add("invariant_drift", InvariantDrift)
METRICS.add("symmetry_violation", SymmetryViolation)
METRICS.add("distribution_drift", DistributionDrift)
METRICS.add("resolution_generalisation", ResolutionGeneralisation)
METRICS.add("calibration", Calibration)

DEFAULT_METRICS = (
    "one_step_error",
    "rollout_error",
    "invariant_drift",
    "symmetry_violation",
    "distribution_drift",
    "resolution_generalisation",
    "calibration",
)
"""Every metric, in the order a report reads best: what one step says, what a rollout
says, and then the five that say what a rollout cannot."""
