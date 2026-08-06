"""Driving a predictor forward and recording what happened while it ran.

Two things matter more here than throughput. A rollout that goes wrong must stop and say
why, because a trajectory of NaN scores as an enormous error when the honest answer is
that the predictor stopped producing states at step 40. And the wall clock must be
recorded on the stepping alone, since the whole speedup claim of a surrogate rests on it.

Divergence is judged against the initial state rather than against ground truth. The
driver never sees ground truth: it is handed a predictor and a state, which is all a
predictor needs, and comparing against the reference is the metrics' job.

A predictor that declares an uncertainty is asked for it here rather than by a metric
running the rollout a second time. The spread belongs to the states this rollout actually
produced, so recording it beside them costs nothing and keeps the two from disagreeing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import NNPhysicsError, ValidationError
from nnphysics.core.protocols import UncertainPredictor
from nnphysics.core.types import State, Trajectory

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nnphysics.core.protocols import Predictor
    from nnphysics.core.types import FloatArray

__all__ = [
    "DEFAULT_DIVERGENCE_FACTOR",
    "RolloutResult",
    "StopReason",
    "roll_out",
    "roll_out_many",
]

DEFAULT_DIVERGENCE_FACTOR = 1.0e3
"""How far past the initial scale a state may go before the rollout is abandoned.

Three orders of magnitude is loose enough that a physically growing quantity survives and
tight enough that a state on its way to overflow is caught while its numbers are still
finite, which is what lets the reason be recorded rather than inferred from NaN.
"""

_SCALE_FLOOR = 1.0e-12
"""Keeps a state of all zeros from making every later state look divergent."""


class StopReason(StrEnum):
    """Why a rollout stopped."""

    COMPLETED = "completed"
    """The requested number of steps was taken."""

    NON_FINITE = "non_finite"
    """A predicted state held a NaN or an infinity."""

    DIVERGED = "diverged"
    """A predicted state grew far past the scale of the state it started from."""

    NOT_ADVANCING = "not_advancing"
    """A predicted state did not move forward in time, so no trajectory can be built."""

    FAILED = "failed"
    """The predictor raised. A predictor that hands a system a state the system refuses
    has stopped predicting, and that is worth recording rather than propagating: the
    rollouts either side of it are still worth scoring."""


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """One predictor's trajectory and the record of producing it.

    Attributes:
        trajectory: States produced, the initial one included. Shorter than requested if
            the rollout stopped early.
        predictor: Name of the predictor that produced it.
        seconds: Wall clock spent stepping, excluding the checks around each step.
        steps_requested: Steps that were asked for.
        stop_reason: Why the rollout ended.
        detail: What the predictor said, when it stopped by raising. Empty otherwise.
        spread: The predictor's own uncertainty at each recorded state, or `None` if it
            declared none. Zero at the initial state, which was handed in rather than
            predicted.
    """

    trajectory: Trajectory
    predictor: str
    seconds: float
    steps_requested: int
    stop_reason: StopReason
    detail: str = ""
    spread: Trajectory | None = None

    @property
    def steps_completed(self) -> int:
        """Steps actually taken."""
        return len(self.trajectory) - 1

    @property
    def completed(self) -> bool:
        """Whether every requested step was taken."""
        return self.stop_reason is StopReason.COMPLETED

    @property
    def seconds_per_step(self) -> float:
        """Mean wall clock per step taken, or zero if no step was taken."""
        return self.seconds / self.steps_completed if self.steps_completed else 0.0


def roll_out(
    predictor: Predictor,
    initial: State,
    steps: int,
    *,
    divergence_factor: float = DEFAULT_DIVERGENCE_FACTOR,
) -> RolloutResult:
    """Advance a predictor from one initial state, stopping early if it goes wrong.

    Args:
        predictor: The predictor to drive.
        initial: State to start from, which becomes the first recorded state.
        steps: Steps to take. The trajectory holds one more state than this.
        divergence_factor: How far past the initial scale a state may go. Zero or less
            turns the check off, which is only wanted when a test is measuring the
            unbounded behaviour itself.

    Returns:
        The trajectory produced, the time it took and why it ended.

    Raises:
        ValidationError: If fewer than one step was asked for.
    """
    if steps < 1:
        raise ValidationError(f"a rollout must take at least one step, got {steps}")

    limit = divergence_factor * max(_state_scale(initial), _SCALE_FLOOR)
    uncertain: UncertainPredictor | None = (
        predictor if isinstance(predictor, UncertainPredictor) else None
    )
    states = [initial]
    spreads = [_no_spread(initial)] if uncertain is not None else []
    reason = StopReason.COMPLETED
    detail = ""
    elapsed = 0.0

    for _ in range(steps):
        started = time.perf_counter()
        try:
            if uncertain is not None:
                prediction = uncertain.predict(states[-1])
                state, spread = prediction.state, prediction.spread
            else:
                state, spread = predictor.step(states[-1]), {}
        except NNPhysicsError as error:
            elapsed += time.perf_counter() - started
            reason = StopReason.FAILED
            detail = str(error)
            break
        elapsed += time.perf_counter() - started

        if not _is_finite(state):
            reason = StopReason.NON_FINITE
            break
        if state.time <= states[-1].time:
            reason = StopReason.NOT_ADVANCING
            break
        if divergence_factor > 0.0 and _state_scale(state) > limit:
            reason = StopReason.DIVERGED
            break
        states.append(state)
        if uncertain is not None:
            spreads.append(State(fields=dict(spread), time=state.time))

    return RolloutResult(
        trajectory=Trajectory.from_states(states),
        predictor=predictor.name,
        seconds=elapsed,
        steps_requested=steps,
        stop_reason=reason,
        detail=detail,
        spread=Trajectory.from_states(spreads) if uncertain is not None else None,
    )


def roll_out_many(
    predictor: Predictor,
    initials: Sequence[State],
    steps: int,
    *,
    divergence_factor: float = DEFAULT_DIVERGENCE_FACTOR,
) -> list[RolloutResult]:
    """Advance a predictor from each of several initial states.

    A batch is a batch of initial conditions rather than a batched step, because a
    predictor advances one state and nothing in the interface promises more. Anything
    that can step a batch faster does so behind that interface.

    Args:
        predictor: The predictor to drive.
        initials: The initial states.
        steps: Steps to take from each.
        divergence_factor: Passed through to each rollout.

    Returns:
        One result per initial state, in the order given.

    Raises:
        ValidationError: If no initial states were given, or fewer than one step was
            asked for.
    """
    if not initials:
        raise ValidationError("cannot roll out from an empty set of initial conditions")
    return [
        roll_out(predictor, initial, steps, divergence_factor=divergence_factor)
        for initial in initials
    ]


def _no_spread(state: State) -> State:
    """Zero uncertainty in every field, at the time of a state that was handed in."""
    fields: Mapping[str, FloatArray] = {
        name: np.zeros_like(array) for name, array in state.fields.items()
    }
    return State(fields=fields, time=state.time)


def _is_finite(state: State) -> bool:
    """Whether every field of a state holds finite numbers."""
    return all(bool(np.isfinite(array).all()) for array in state.fields.values())


def _state_scale(state: State) -> float:
    """One number standing for the size of a state.

    The largest field magnitude rather than a norm over all of them, so that a field that
    blows up is not hidden by averaging against a field that does not.
    """
    return max(float(np.max(np.abs(array))) for array in state.fields.values())
