"""Drift in the quantities a system says it conserves, or says it drains.

The metric the harness exists for. A predictor can track ground truth closely for a while
and still be adding energy every step, and that shows up here long before it shows up in
an error curve. It shows up in the other direction too: a viscous fluid must lose energy,
so a predictor whose energy climbs is unphysical even if its error is small.

Nothing here knows what energy is. The system declares a named scalar, whether it should
hold still or fall, and how far it may stray; this reads those declarations and measures
against them.

What is measured is the gap between the predictor's invariant and the reference's at the
same instant, not the predictor's excursion from its own starting value. The difference
matters. A quantity a system conserves stays near its initial value along any trajectory,
so two trajectories that have long since parted still carry comparable invariants, and
the gap between them is energy the predictor added or lost that the physics did not.
Measuring the excursion instead would blame a predictor for the solver's own integration
error, and would call a viscous flow that barely decays innocent because its energy went
down rather than up.

Scaling the judgement is the delicate part. A relative gap is undefined for a quantity
whose reference value is zero, which is exactly the case for the linear momentum of a
cluster generated at rest, and a fixed absolute tolerance would mean nothing across two
systems in different units. So a predictor is judged against the larger of two yardsticks:
the tolerance the invariant declares, and a multiple of the reference solver's own drift.
The second is what keeps a predictor that holds an invariant perfectly still, persistence
being the extreme case, from being blamed for the wobble in the solver it is compared to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.protocols import Conservation
from nnphysics.core.types import MetricResult
from nnphysics.evals.metrics.base import MetricContext

if TYPE_CHECKING:
    from nnphysics.core.protocols import Invariant
    from nnphysics.core.types import FloatArray, Rollout, Trajectory

__all__ = ["REFERENCE_MARGIN", "InvariantDrift"]

REFERENCE_MARGIN = 10.0
"""How much of the reference solver's own drift is allowed to excuse a predictor's. Ten,
so that a predictor which merely holds an invariant still lands an order of magnitude
below the flagging level rather than just under it."""

_FLOOR = 1.0e-300
"""Only present so that a quantity that is identically zero divides rather than raises."""


@dataclass(frozen=True, slots=True)
class InvariantDrift:
    """How far a predictor took every declared invariant from where the physics took it.

    A violation above one means the predictor broke a declaration the system made. For a
    quantity declared conserved that is a gap in either direction; for one declared
    decaying it is an upward gap alone, since a decaying quantity that falls faster than
    the reference is over dissipative rather than unphysical, and that is a different
    fault reported as its own number.

    The reference solver scores exactly zero here, because it is compared with itself.
    That is worth more than a tolerance chosen until it passed.

    Attributes:
        context: What the runner assembled, for the declared invariants.
    """

    context: MetricContext = field(default_factory=MetricContext)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "invariant_drift"

    def compute(self, rollout: Rollout) -> MetricResult:
        """Measure every declared invariant along both trajectories.

        Args:
            rollout: The predicted and reference trajectories.

        Returns:
            Per invariant drift, gap, shortfall and violation, the worst violation over
            all of them, and both invariant curves as plot data.
        """
        scalars: dict[str, float] = {}
        series: dict[str, FloatArray] = {}
        worst = 0.0
        for invariant in self.context.invariants:
            predicted = _evaluate(invariant, rollout.predicted)
            reference = _evaluate(invariant, rollout.reference)
            gap = predicted - reference
            violation = _violation(invariant, gap, reference)
            scalars[f"{invariant.name}.drift"] = _drift(predicted)
            scalars[f"{invariant.name}.reference_drift"] = _drift(reference)
            scalars[f"{invariant.name}.scale"] = _scale(reference)
            scalars[f"{invariant.name}.excess"] = _rise(gap)
            scalars[f"{invariant.name}.shortfall"] = _rise(-gap)
            scalars[f"{invariant.name}.violation"] = violation
            series[f"{invariant.name}.predicted"] = predicted
            series[f"{invariant.name}.reference"] = reference
            worst = max(worst, violation)
        scalars["worst_violation"] = worst
        return MetricResult(name=self.name, scalars=scalars, series=series)


def _evaluate(invariant: Invariant, trajectory: Trajectory) -> FloatArray:
    """Compute an invariant at every state of a trajectory."""
    values: FloatArray = np.asarray(
        [invariant.evaluate(state) for state in trajectory], dtype=np.float64
    )
    return values


def _drift(values: FloatArray) -> float:
    """Largest excursion from the starting value, in the invariant's own units."""
    return float(np.max(np.abs(values - values[0])))


def _rise(values: FloatArray) -> float:
    """Largest positive value, zero if none is positive."""
    return float(max(np.max(values), 0.0))


def _scale(values: FloatArray) -> float:
    """Largest magnitude the invariant reaches, which is what a tolerance is relative to."""
    return float(np.max(np.abs(values)))


def _violation(invariant: Invariant, gap: FloatArray, reference: FloatArray) -> float:
    """How far past its declared behaviour a predictor took an invariant.

    One is the flagging level. Below one the gap is within either the declared tolerance
    or a small multiple of the reference solver's own drift; above it, it is not.
    """
    if invariant.conservation is Conservation.DECAYING:
        # A decaying quantity has nowhere legitimate to go but down, so the reference has
        # no upward wobble to excuse a predictor with, and the declared tolerance stands
        # alone.
        measured = _rise(gap)
        excused = _rise(reference - reference[0])
    else:
        measured = float(np.max(np.abs(gap)))
        excused = _drift(reference)
    allowed = max(invariant.rtol * _scale(reference), REFERENCE_MARGIN * excused, _FLOOR)
    return measured / allowed
