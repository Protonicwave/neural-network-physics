"""The system's own solver, seen as a predictor like any other.

A solver steps at whatever interval keeps it stable, and a surrogate learns the interval
the data is stored at. Comparing the two means putting them on the same interval, which
is all this wrapper does: it folds the solver's substeps into one step so that a rollout
of a hundred steps means the same span of simulated time whichever predictor produced it.

Folding the substeps in also makes the wall clock the honest one. The solver's cost per
stored interval is what a surrogate has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor
    from nnphysics.core.types import State

__all__ = ["REFERENCE_NAME", "Substepped"]

REFERENCE_NAME = "reference"
"""What ground truth is called in a result file, whichever system produced it."""


@dataclass(frozen=True, slots=True)
class Substepped:
    """A predictor that advances by several steps of another one.

    Attributes:
        inner: The predictor being folded, normally a system's reference solver.
        substeps: Inner steps taken per outer step.
        label: Name reported for this predictor.
    """

    inner: Predictor
    substeps: int
    label: str = REFERENCE_NAME

    def __post_init__(self) -> None:
        if self.substeps < 1:
            raise ValidationError(f"substeps must be positive, got {self.substeps}")
        if not self.label:
            raise ValidationError("a predictor must have a name")

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return self.label

    @property
    def dt(self) -> float:
        """The stored interval: one outer step's worth of simulated time."""
        return self.inner.dt * self.substeps

    def step(self, state: State) -> State:
        """Advance one stored interval.

        Args:
            state: The current state.

        Returns:
            The state one stored interval later.
        """
        for _ in range(self.substeps):
            state = self.inner.step(state)
        return state
