"""Error against ground truth, over one step and over a horizon.

The two are separated deliberately. One step error is the number a training loss reports
and the number a paper quotes, and on its own it is close to meaningless: a predictor can
be excellent over one step and produce a different physical system after two hundred.
Keeping both in the same suite is what lets that be shown rather than argued.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.types import MetricResult
from nnphysics.evals.metrics.base import MetricContext, relative_error

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray, Rollout

__all__ = ["NEVER_REACHED", "OneStepError", "RolloutErrorGrowth"]

NEVER_REACHED = -1.0
"""Reported as a horizon when the rollout never reaches a threshold. Negative rather than
infinite so that a result file stays plain JSON, and so that a reader cannot mistake it
for a time."""


@dataclass(frozen=True, slots=True)
class OneStepError:
    """Normalised error after a single step.

    Attributes:
        context: What the runner assembled. Nothing here needs it, and it is carried so
            that every metric is built the same way.
    """

    context: MetricContext = field(default_factory=MetricContext)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "one_step_error"

    def compute(self, rollout: Rollout) -> MetricResult:
        """Score the first step of a rollout.

        Args:
            rollout: The predicted and reference trajectories.

        Returns:
            The aggregate error and the error of each field.
        """
        per_field, aggregate = relative_error(rollout.predicted, rollout.reference)
        step = min(1, len(rollout) - 1)
        scalars = {"error": float(aggregate[step])}
        scalars.update({f"error.{name}": float(values[step]) for name, values in per_field.items()})
        return MetricResult(name=self.name, scalars=scalars)


@dataclass(frozen=True, slots=True)
class RolloutErrorGrowth:
    """Normalised error as a function of horizon, and where it crosses given levels.

    The curve is the answer and the crossings are the summary. A horizon is reported in
    simulated time rather than in steps, so that two predictors stepping at different
    intervals can be compared, and it is the time at which the error *first* exceeds a
    level rather than the last, since a predictor that recovers by luck has not earned
    the longer horizon.

    Attributes:
        context: What the runner assembled, for the thresholds.
    """

    context: MetricContext = field(default_factory=MetricContext)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "rollout_error"

    def compute(self, rollout: Rollout) -> MetricResult:
        """Score the whole rollout.

        Args:
            rollout: The predicted and reference trajectories.

        Returns:
            Summary errors, one horizon per threshold, and the error curve as plot data.
        """
        per_field, aggregate = relative_error(rollout.predicted, rollout.reference)
        times = rollout.times
        scalars = {
            "error.final": float(aggregate[-1]),
            "error.max": float(np.max(aggregate)),
            "error.mean": float(np.mean(aggregate)),
            "duration": float(times[-1] - times[0]),
        }
        for threshold in self.context.thresholds:
            scalars[f"horizon.{threshold:g}"] = _first_crossing(aggregate, times, threshold)
        series: dict[str, FloatArray] = {"error": aggregate, "time": times}
        series.update({f"error.{name}": values for name, values in per_field.items()})
        return MetricResult(name=self.name, scalars=scalars, series=series)


def _first_crossing(error: FloatArray, times: FloatArray, threshold: float) -> float:
    """Time at which the error first exceeds a threshold, or `NEVER_REACHED`."""
    exceeded = np.flatnonzero(error > threshold)
    if exceeded.size == 0:
        return NEVER_REACHED
    return float(times[exceeded[0]] - times[0])
