"""Resolution generalisation: does the predictor mean the same thing on a finer grid.

A neural operator is claimed to be a map between function spaces rather than between
arrays, so that weights fitted on one discretisation apply to another. It is a strong
claim and this is where it is checked. Refine the initial condition to the finer
representation the system declares, roll the predictor forward there, coarsen the result
back, and compare.

Two numbers come out of that and they answer different questions.

**Consistency** compares the refined path against the predictor's own native path. It
needs no ground truth, because it is not asking whether the predictor is right: it is
asking whether the predictor is the same operator at both resolutions. A model whose
weights are indexed by wavenumber should read near zero here whatever its accuracy, and a
model whose weights are indexed by grid point should not.

**Degradation** compares the refined path against ground truth, and subtracts what the
native path scored on the same ground truth. It is the number a user cares about, and it
is the looser of the two: ground truth is stored at the coarse resolution, so a refined
rollout is charged for the modes the finer grid legitimately develops as well as for its
own error. Consistency is reported beside it for exactly that reason.

Nothing here knows what a resolution is. The system declares a transformation and its
inverse, and this file applies them, the same way the equivariance metric applies a
symmetry it cannot name. A system that declares none is not tested and says so with a
step count of zero, rather than with a score that would read as a pass.

A predictor tied to the grid it was built for fails when handed a refined state. That is
recorded as zero steps rather than propagated: the reference solver is such a predictor,
and its inability to run on a grid it was not constructed for is a fact about the solver
worth reading in the table, not a reason to abandon the suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import NNPhysicsError
from nnphysics.core.types import MetricResult, Trajectory
from nnphysics.evals.metrics.base import MetricContext, relative_error
from nnphysics.evals.rollout import roll_out

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor, Refinement
    from nnphysics.core.types import FloatArray, Rollout

__all__ = ["ResolutionGeneralisation"]


@dataclass(frozen=True, slots=True)
class _Attempt:
    """What one refinement produced, or nothing if the predictor could not run there."""

    consistency: FloatArray
    refined: FloatArray
    native: FloatArray


@dataclass(frozen=True, slots=True)
class ResolutionGeneralisation:
    """Agreement between a predictor at its native resolution and at a finer one.

    Attributes:
        context: What the runner assembled, for the refinements and the predictor.
    """

    context: MetricContext = field(default_factory=MetricContext)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "resolution_generalisation"

    def compute(self, rollout: Rollout) -> MetricResult:
        """Roll the predictor out again from a refined initial condition.

        Args:
            rollout: The predicted and reference trajectories. Both are used: the
                predicted one is what consistency is measured against, and the reference
                one is what both paths are scored on.

        Returns:
            Per refinement consistency, refined error, native error and degradation, the
            worst of each over all of them, and the consistency curves as plot data.
        """
        steps = min(self.context.resolution_steps, len(rollout) - 1)
        if steps < 1 or not self.context.refinements:
            return MetricResult(name=self.name, scalars={"steps": 0.0})

        predictor = self.context.require_predictor(self.name)
        base = _prefix(rollout.predicted, steps + 1)
        truth = _prefix(rollout.reference, steps + 1)

        scalars: dict[str, float] = {}
        series: dict[str, FloatArray] = {}
        consistencies: list[float] = []
        degradations: list[float] = []

        for refinement in self.context.refinements:
            attempt = self._attempt(predictor, refinement, base, truth)
            if attempt is None:
                scalars[f"{refinement.name}.steps"] = 0.0
                continue
            consistency = float(np.max(attempt.consistency))
            refined = float(attempt.refined[-1])
            native = float(attempt.native[-1])
            scalars[f"{refinement.name}.consistency"] = consistency
            scalars[f"{refinement.name}.refined_error"] = refined
            scalars[f"{refinement.name}.native_error"] = native
            scalars[f"{refinement.name}.degradation"] = refined - native
            scalars[f"{refinement.name}.steps"] = float(attempt.consistency.size - 1)
            series[refinement.name] = attempt.consistency
            consistencies.append(consistency)
            degradations.append(refined - native)

        scalars["steps"] = float(steps)
        if consistencies:
            scalars["worst_consistency"] = max(consistencies)
            scalars["worst_degradation"] = max(degradations)
        return MetricResult(name=self.name, scalars=scalars, series=series)

    def _attempt(
        self,
        predictor: Predictor,
        refinement: Refinement,
        base: Trajectory,
        truth: Trajectory,
    ) -> _Attempt | None:
        """Run one refinement, or return nothing if the predictor could not run there.

        A rollout that stops early is compared over the steps it managed, so a predictor
        that survives the change of resolution for a while still reports what it did
        before it stopped.
        """
        try:
            initial = refinement.refine(base[0])
        except NNPhysicsError:
            # A system may refuse to refine a state it cannot represent at the finer
            # resolution. That is the system's judgement about the state, not a failure
            # of the predictor, and either way there is nothing to measure.
            return None
        result = roll_out(
            predictor,
            initial,
            len(base) - 1,
            divergence_factor=self.context.divergence_factor,
        )
        if result.steps_completed < 1:
            return None
        coarsened = Trajectory.from_states(
            [refinement.coarsen(state) for state in result.trajectory]
        )
        common = min(len(coarsened), len(base))
        shortened = _prefix(coarsened, common)
        _, consistency = relative_error(shortened, _prefix(base, common))
        _, refined = relative_error(shortened, _prefix(truth, common))
        _, native = relative_error(_prefix(base, common), _prefix(truth, common))
        return _Attempt(consistency=consistency, refined=refined, native=native)


def _prefix(trajectory: Trajectory, length: int) -> Trajectory:
    """The first `length` states of a trajectory."""
    if length >= len(trajectory):
        return trajectory
    return Trajectory(
        fields={name: array[:length] for name, array in trajectory.fields.items()},
        times=trajectory.times[:length],
    )
