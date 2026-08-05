"""Equivariance: does the predictor commute with the transformations the system declares.

Transform the initial condition, roll forward, undo the transformation, and compare with
the rollout that was never transformed. A predictor that respects the symmetry gives the
same answer either way. Nothing in this file knows what the transformation is, only that
the system declared it and can undo it.

Every declared symmetry is tested and the worst is reported. That is not thoroughness for
its own sake: a predictor whose fault is a fixed rotation each step is perfectly
equivariant under rotation, and only shows itself under the ones that do not commute with
it. Testing a single symmetry would make catching such a fault a matter of luck.

The comparison is taken over the whole tested horizon rather than at its end, because a
violation can come and go. A fault that is a quarter turn each step vanishes every fourth
step, and a metric that only looked at the last state could be handed a rollout whose
length hid it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.types import MetricResult, Trajectory
from nnphysics.evals.metrics.base import MetricContext, relative_error
from nnphysics.evals.rollout import roll_out

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor, Symmetry
    from nnphysics.core.types import FloatArray, Rollout, State

__all__ = ["SymmetryViolation"]


@dataclass(frozen=True, slots=True)
class SymmetryViolation:
    """Equivariance error of a predictor under every symmetry a system declares.

    Attributes:
        context: What the runner assembled, for the symmetries and the predictor.
    """

    context: MetricContext = field(default_factory=MetricContext)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "symmetry_violation"

    def compute(self, rollout: Rollout) -> MetricResult:
        """Roll the predictor out again from transformed initial conditions.

        Args:
            rollout: The predicted and reference trajectories. Only the predicted one is
                used: equivariance is a property of the predictor, not of its accuracy,
                and a predictor can be equivariant and wrong or accurate and not
                equivariant.

        Returns:
            Per symmetry violation, the worst over all of them, and the violation curves
            as plot data.
        """
        predictor = self.context.require_predictor(self.name)
        steps = min(self.context.symmetry_steps, len(rollout) - 1)
        scalars: dict[str, float] = {}
        series: dict[str, FloatArray] = {}
        worst = 0.0

        if steps < 1:
            return MetricResult(name=self.name, scalars={"worst": 0.0, "steps": 0.0})

        base = _prefix(rollout.predicted, steps + 1)
        initial = base[0]
        for symmetry in self.context.symmetries:
            violation = self._violation(predictor, symmetry, initial, base)
            scalars[f"{symmetry.name}.max"] = float(np.max(violation))
            scalars[f"{symmetry.name}.final"] = float(violation[-1])
            scalars[f"{symmetry.name}.mean"] = float(np.mean(violation))
            scalars[f"{symmetry.name}.steps"] = float(violation.size - 1)
            series[f"{symmetry.name}"] = violation
            worst = max(worst, float(np.max(violation)))
        scalars["worst"] = worst
        scalars["steps"] = float(steps)
        return MetricResult(name=self.name, scalars=scalars, series=series)

    def _violation(
        self, predictor: Predictor, symmetry: Symmetry, initial: State, base: Trajectory
    ) -> FloatArray:
        """Equivariance error at every step of the tested horizon.

        A rollout that stops early is compared over the steps it managed, so that a
        predictor which diverges only once transformed still reports what it did before
        it went.
        """
        result = roll_out(
            predictor,
            symmetry.apply(initial),
            len(base) - 1,
            divergence_factor=self.context.divergence_factor,
        )
        undone = Trajectory.from_states(
            [symmetry.apply_inverse(state) for state in result.trajectory]
        )
        common = min(len(undone), len(base))
        _, aggregate = relative_error(_prefix(undone, common), _prefix(base, common))
        return aggregate


def _prefix(trajectory: Trajectory, length: int) -> Trajectory:
    """The first `length` states of a trajectory."""
    if length >= len(trajectory):
        return trajectory
    return Trajectory(
        fields={name: array[:length] for name, array in trajectory.fields.items()},
        times=trajectory.times[:length],
    )
