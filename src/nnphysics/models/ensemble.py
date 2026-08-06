"""A deep ensemble: several models of the same shape, trained from different seeds.

The simplest uncertainty estimate that works, and the reason it is the one the plan picks
is that it needs nothing the training loop does not already do. Train the same
configuration a few times from different initialisations, and where the members agree the
answer is probably determined by the data, while where they disagree it is determined by
which minimum each run happened to fall into. That disagreement is the estimate.

Two decisions here decide what the spread means.

**Every member keeps its own trajectory.** The alternative, averaging after each step and
feeding the mean back to everyone, gives a spread that measures one step of disagreement
and never grows. Letting members run their own rollouts and reporting the spread across
them is what makes the spread say something about a horizon, which is the question the
phase actually asks: does the disagreement grow before the error becomes unacceptable.

**The ensemble's own prediction is the mean of the members.** So the trajectory scored is
the trajectory the spread describes, rather than one member's path with somebody else's
error bars around it.

The members are advanced in sequence rather than in parallel. On eight cores the members
are already competing for those cores inside their own matrix products, and a process per
member would fight the threading rather than add to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import Prediction, State

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray
    from nnphysics.models.base import SurrogateModel

__all__ = ["ENSEMBLE_NAME", "Ensemble"]

ENSEMBLE_NAME = "ensemble"
"""Name the ensemble is reported under, whatever its members are called."""

_MINIMUM_MEMBERS = 2
"""One model has nothing to disagree with, so it has no spread to report."""

_INTERVAL_RTOL = 1.0e-9
_TIME_RTOL = 1.0e-9
_TIME_ATOL = 1.0e-12


class Ensemble:
    """Several trained models advanced together, reporting their mean and their spread.

    A predictor to the harness and nothing else. It is not a `SurrogateModel`: there is
    nothing here to train, and a class that pretended otherwise would have a checkpoint
    format and an optimiser state for weights that live in its members' own files.

    Like the other predictors that carry state between calls, a rollout is detected as new
    when the state handed in is not the one this returned last. That is what lets each
    member keep its own path through a rollout while the harness sees one predictor.

    Args:
        members: The trained models, at least two, all stepping by the same interval.

    Raises:
        ValidationError: If there are fewer than two members, or they disagree about the
            interval they advance by.
    """

    def __init__(self, members: Sequence[SurrogateModel]) -> None:
        if len(members) < _MINIMUM_MEMBERS:
            raise ValidationError(
                f"an ensemble needs at least {_MINIMUM_MEMBERS} members to have a spread, "
                f"got {len(members)}"
            )
        first = members[0].dt
        for member in members:
            if abs(member.dt - first) > _INTERVAL_RTOL * first:
                raise ValidationError(
                    f"ensemble members step by different intervals: {first:g} and {member.dt:g}"
                )
        self._members = tuple(members)
        self._states: tuple[State, ...] | None = None
        self._last: State | None = None

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return ENSEMBLE_NAME

    @property
    def dt(self) -> float:
        """The stored interval every member advances by."""
        return self._members[0].dt

    @property
    def members(self) -> tuple[SurrogateModel, ...]:
        """The models being averaged."""
        return self._members

    @property
    def n_parameters(self) -> int:
        """Trainable parameters over every member, which is what the ensemble cost."""
        return sum(member.n_parameters for member in self._members)

    def reset(self) -> None:
        """Forget every member's path, so the next step starts a rollout."""
        self._states = None
        self._last = None

    def predict(self, state: State) -> Prediction:
        """Advance every member and report their mean and their disagreement.

        Args:
            state: The current state. When it is the state this returned last, each member
                continues from its own path; otherwise every member is restarted from it.

        Returns:
            The mean state and the population standard deviation across members, per
            element.
        """
        if self._states is None or not self._continues(state):
            self._states = tuple(state for _ in self._members)
        advanced = tuple(
            member.step(current)
            for member, current in zip(self._members, self._states, strict=True)
        )
        self._states = advanced

        mean: dict[str, FloatArray] = {}
        spread: dict[str, FloatArray] = {}
        for name in advanced[0].fields:
            stacked = np.stack([entry.fields[name] for entry in advanced])
            mean[name] = np.mean(stacked, axis=0)
            # The population standard deviation rather than the sample one. Four members
            # is a small ensemble and the Bessel correction would inflate the spread by a
            # sixth, which would be a claim about the estimator rather than about the
            # models.
            spread[name] = np.std(stacked, axis=0)
        result = State(fields=mean, time=advanced[0].time)
        self._last = result
        return Prediction(state=result, spread=spread)

    def step(self, state: State) -> State:
        """Advance one step, discarding the spread.

        Args:
            state: The current state.

        Returns:
            The mean of what the members produced.
        """
        return self.predict(state).state

    def _continues(self, state: State) -> bool:
        """Whether this state is the one the previous call returned."""
        if self._last is None:
            return False
        if state is self._last:
            return True
        return bool(
            np.isclose(state.time, self._last.time, rtol=_TIME_RTOL, atol=_TIME_ATOL)
        ) and all(
            np.array_equal(array, self._last.fields[name]) for name, array in state.fields.items()
        )
