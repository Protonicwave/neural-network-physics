"""Whether a rollout still looks like the right system, once it has stopped tracking it.

Error against ground truth answers a question that stops being interesting at long
horizon. Two chaotic trajectories from nearby starts diverge whatever produced them, so a
large error late in a rollout says little on its own. What still matters is whether the
predicted states are drawn from the same distribution as the true ones: a cluster that is
still a cluster, a flow whose vorticity still has the same spread and tails.

That is measured here by comparing the distribution of each field's values over a window
at the end of the rollout, through its quantiles. Quantiles rather than moments because
a mean and a variance can agree while the tails do not, and the tails are what a noisy
predictor ruins first.

This is the system agnostic stand in for the summaries a physicist would reach for, an
energy spectrum for the fluid and a radial distribution for the N-body cluster. Those are
sharper, and either would require the metric to know which system it was looking at, or
would require a system to declare its own summary statistics. The second is the honest
fix if this proves too blunt, and it belongs in the abstraction rather than here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.types import MetricResult
from nnphysics.evals.metrics.base import MetricContext

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray, Rollout, Trajectory

__all__ = ["QUANTILES", "DistributionDrift"]

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
"""Fixed rather than configurable, so that two runs are comparable. The pair at one and
ninety nine per cent is what makes the measure sensitive to tails."""

_FLOOR = 1.0e-300
"""Only present so that a field that is identically zero divides rather than raises."""


@dataclass(frozen=True, slots=True)
class DistributionDrift:
    """Distance between the value distributions of predicted and true states.

    Measured over a window at the end of the rollout, where the trajectories have long
    since parted and only the statistics are still comparable.

    Attributes:
        context: What the runner assembled, for the window length.
    """

    context: MetricContext = field(default_factory=MetricContext)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "distribution_drift"

    def compute(self, rollout: Rollout) -> MetricResult:
        """Compare the two trajectories' late distributions.

        Args:
            rollout: The predicted and reference trajectories.

        Returns:
            Per field quantile distance and spread ratio, the worst distance over all
            fields, and both sets of quantiles as plot data.
        """
        window = _window_length(len(rollout), self.context.distribution_window)
        scalars: dict[str, float] = {"window": float(window)}
        series: dict[str, FloatArray] = {"quantiles": np.asarray(QUANTILES, dtype=np.float64)}
        worst = 0.0

        for name in sorted(rollout.reference.names):
            predicted = _late_quantiles(rollout.predicted, name, window)
            reference = _late_quantiles(rollout.reference, name, window)
            scale = _scale(reference)
            distance = float(np.mean(np.abs(predicted - reference))) / scale
            scalars[f"{name}.quantile_distance"] = distance
            scalars[f"{name}.spread_ratio"] = float(predicted[-1] - predicted[0]) / scale
            series[f"{name}.predicted"] = predicted
            series[f"{name}.reference"] = reference
            worst = max(worst, distance)
        scalars["worst"] = worst
        return MetricResult(name=self.name, scalars=scalars, series=series)


def _scale(quantiles: FloatArray) -> float:
    """What a distance between two sets of quantiles is measured relative to.

    The wider of the reference's spread and its magnitude. A field that is constant, as a
    list of masses is, has no spread to compare against, and dividing by one would report
    any perturbation of it as infinite; comparing it against its own size instead says
    the useful thing, which is by how much the constant moved.
    """
    spread = float(quantiles[-1] - quantiles[0])
    magnitude = float(np.max(np.abs(quantiles)))
    return max(spread, magnitude, _FLOOR)


def _window_length(n_steps: int, fraction: float) -> int:
    """How many states at the end of a rollout the distributions are taken over."""
    return max(1, min(n_steps, round(n_steps * fraction)))


def _late_quantiles(trajectory: Trajectory, field: str, window: int) -> FloatArray:
    """Quantiles of one field's values over the last `window` states.

    Every element of every state in the window is pooled. A field with structure loses
    that structure here, which is the price of a measure that does not know whether it is
    looking at a grid or at a list of bodies.
    """
    values = np.asarray(trajectory.fields[field][-window:], dtype=np.float64).reshape(-1)
    quantiles: FloatArray = np.quantile(values, QUANTILES)
    return quantiles
