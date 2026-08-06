"""Faults with a known true cause, so that a diagnosis can be marked.

An agent that produces a fluent explanation of a run nobody broke on purpose cannot be
told apart from one that produces a fluent explanation of anything. So the runs it is
asked about are broken deliberately, one fault at a time, and what was broken is written
down before the agent is asked.

The catalogue is declarative. A fault is a name, the cause it is an instance of, a
transformation of the run configuration and, where the configuration has no knob for it,
the point in the pipeline where something has to be interfered with instead. Nothing here
trains anything or writes a file: the runner that does lives at the command line edge,
which is where every other side effect in this repository lives.

Four of the seven are pure configuration changes, and the split is the interesting part of
the design. A fault visible in the configuration diff is one a diagnoser could name by
reading two lines, so those four test whether it reads the evidence at all. The other
three, the wrong normalisation statistics, the broken symmetry and the lost optimiser
state, appear nowhere in the configuration: the only trace they leave is in the numbers,
which is the harder and more realistic case.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from nnphysics.agent.causes import Cause
from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.data.normalisation import FieldStats, Normalisation

if TYPE_CHECKING:
    from nnphysics.core.config import RunConfig
    from nnphysics.core.protocols import Predictor
    from nnphysics.core.types import State

__all__ = [
    "FAULTS",
    "Fault",
    "Injection",
    "RenamedPredictor",
    "corrupt_normalisation",
    "fault",
    "fault_names",
    "interrupted",
]

_LEARNING_RATE_FACTOR = 200.0
"""How much too high the excessive learning rate is. Two orders of magnitude, so the fault
is a run that cannot converge rather than one that converges slightly worse."""

_NORMALISATION_SCALE = 10.0
"""How much the wrong statistics inflate every field's scale."""

_NORMALISATION_SHIFT = 3.0
"""How far the wrong statistics move every field's centre, in units of that field's own
scale. Applied on top of the inflation so that a constant field, which has no scale of its
own to inflate, is still moved."""

_UNSTABLE_SUBSTEPS = 1
"""Solver steps per stored interval under the unstable integrator. One, so that the
solver takes the whole stored interval in a single step whatever the configuration asked
for."""


class Injection(StrEnum):
    """Where a fault has to interfere with the pipeline, beyond the configuration."""

    NONE = "none"
    """The configuration transformation is the whole fault."""

    NORMALISATION = "normalisation"
    """The model is built with statistics that do not describe its data."""

    SYMMETRY = "symmetry"
    """The trained model is wrapped so that it stops commuting with a declared symmetry."""

    OPTIMISER_STATE = "optimiser_state"
    """Training is interrupted and resumed from a checkpoint whose optimiser state has
    been thrown away."""


@dataclass(frozen=True, slots=True)
class Fault:
    """One thing that can be broken on purpose, and what it is an instance of.

    Attributes:
        name: Short identifier, used on the command line and in the scored table.
        cause: The cause a correct diagnosis has to name.
        summary: What was broken, in one line, for the results table. Never shown to the
            diagnoser: it is the answer.
        injection: Where the pipeline has to be interfered with, or `NONE`.
        transform: How the run configuration is changed. Identity for a fault that lives
            entirely in an injection.
    """

    name: str
    cause: Cause
    summary: str
    injection: Injection
    transform: Callable[[RunConfig], RunConfig]

    def apply(self, config: RunConfig) -> RunConfig:
        """The configuration this fault runs under.

        Args:
            config: The known good configuration.

        Returns:
            The faulty configuration, which is the same object's value when the fault
            lives entirely in an injection.
        """
        return self.transform(config)

    @property
    def visible_in_config(self) -> bool:
        """Whether the fault leaves a trace in the configuration diff.

        The three that do not are the harder half of the set: nothing a reader could
        compare says what happened, so the diagnosis has to come from the numbers alone.
        """
        return self.transform is not _same


def _same(config: RunConfig) -> RunConfig:
    """Leave a configuration alone, for a fault that lives entirely in an injection."""
    return config


def _high_learning_rate(config: RunConfig) -> RunConfig:
    """Raise the learning rate far past what the optimiser can follow."""
    training = config.training.model_copy(
        update={"learning_rate": config.training.learning_rate * _LEARNING_RATE_FACTOR}
    )
    return config.model_copy(update={"training": training})


def _no_curriculum(config: RunConfig) -> RunConfig:
    """Train on single steps only, so nothing ever teaches the model to stay stable."""
    training = config.training.model_copy(update={"curriculum": (1,), "curriculum_epochs": (0,)})
    return config.model_copy(update={"training": training})


def _wrong_regime(config: RunConfig) -> RunConfig:
    """Train on the regimes that were held out and hold out the ones that were trained on.

    Not a relabelling: the splits are drawn from the trained regimes, so this genuinely
    fits the model to one part of the configuration space and then asks it about another.

    Raises:
        ValidationError: If the configuration names no held out regime to swap in.
    """
    if not config.data.held_out_regimes:
        raise ValidationError("cannot swap regimes in a configuration that holds none out")
    data = config.data.model_copy(
        update={
            "regimes": config.data.held_out_regimes,
            "held_out_regimes": config.data.regimes,
        }
    )
    return config.model_copy(update={"data": data})


def _unstable_integrator(config: RunConfig) -> RunConfig:
    """Give the reference solver one step per stored interval, past where it stays stable.

    Raises:
        ValidationError: If the configuration already takes a single substep, in which
            case there is nothing to break.
    """
    if config.data.substeps <= _UNSTABLE_SUBSTEPS:
        raise ValidationError(
            f"the configuration already takes {config.data.substeps} substep per stored "
            f"interval, so the integrator cannot be made coarser"
        )
    data = config.data.model_copy(update={"substeps": _UNSTABLE_SUBSTEPS})
    return config.model_copy(update={"data": data})


FAULTS: tuple[Fault, ...] = (
    Fault(
        name="wrong_normalisation",
        cause=Cause.NORMALISATION_STATISTICS,
        summary=(
            f"The model is built with statistics whose scale is {_NORMALISATION_SCALE:g} "
            f"times too large and whose centre is {_NORMALISATION_SHIFT:g} scales off."
        ),
        injection=Injection.NORMALISATION,
        transform=_same,
    ),
    Fault(
        name="broken_symmetry",
        cause=Cause.MODEL_SYMMETRY,
        summary="The trained model has a declared symmetry applied after every step.",
        injection=Injection.SYMMETRY,
        transform=_same,
    ),
    Fault(
        name="high_learning_rate",
        cause=Cause.LEARNING_RATE,
        summary=f"The learning rate is {_LEARNING_RATE_FACTOR:g} times the configured one.",
        injection=Injection.NONE,
        transform=_high_learning_rate,
    ),
    Fault(
        name="wrong_regime",
        cause=Cause.TRAINING_REGIME,
        summary="Training and held out regimes are swapped, so the model is fitted to the "
        "configuration class it was supposed to be tested on.",
        injection=Injection.NONE,
        transform=_wrong_regime,
    ),
    Fault(
        name="no_curriculum",
        cause=Cause.ROLLOUT_CURRICULUM,
        summary="The rollout curriculum is reduced to a single step, so the model is never "
        "trained over a window.",
        injection=Injection.NONE,
        transform=_no_curriculum,
    ),
    Fault(
        name="unstable_integrator",
        cause=Cause.INTEGRATOR_STEP_SIZE,
        summary=f"The reference solver takes {_UNSTABLE_SUBSTEPS} substep per stored "
        f"interval, so the ground truth is itself inaccurate.",
        injection=Injection.NONE,
        transform=_unstable_integrator,
    ),
    Fault(
        name="no_optimiser_state",
        cause=Cause.OPTIMISER_STATE,
        summary="Training is interrupted partway and resumed from a checkpoint whose "
        "optimiser moments have been discarded.",
        injection=Injection.OPTIMISER_STATE,
        transform=_same,
    ),
)
"""The fault set. One per cause the brief names, and no two share a cause: a scored table
in which two faults had the same answer would be measuring one of them twice."""

_BY_NAME: Mapping[str, Fault] = {entry.name: entry for entry in FAULTS}


def fault_names() -> tuple[str, ...]:
    """Every fault, in catalogue order.

    Returns:
        The names.
    """
    return tuple(_BY_NAME)


def fault(name: str) -> Fault:
    """Look up one fault.

    Args:
        name: The fault name.

    Returns:
        The fault.

    Raises:
        UnknownNameError: If no fault has that name.
    """
    if name not in _BY_NAME:
        raise UnknownNameError(f"no fault named {name!r}. Known faults: {list(_BY_NAME)}")
    return _BY_NAME[name]


def corrupt_normalisation(
    normalisation: Normalisation,
    *,
    scale: float = _NORMALISATION_SCALE,
    shift: float = _NORMALISATION_SHIFT,
) -> Normalisation:
    """Statistics that no longer describe the data they were fitted to.

    Args:
        normalisation: The statistics fitted to the training split.
        scale: Factor every field's spread is multiplied by.
        shift: How far every field's centre moves, in units of that field's own scale.

    Returns:
        The corrupted statistics, covering exactly the same fields.

    Raises:
        ValidationError: If the factor is not positive, which would invert or flatten
            every field rather than mis-scale it.
    """
    if scale <= 0.0:
        raise ValidationError(f"a normalisation scale factor must be positive, got {scale}")
    return Normalisation(
        {
            name: FieldStats(
                mean=stats.mean + shift * stats.scale,
                std=stats.std * scale,
                count=stats.count,
            )
            for name, stats in normalisation.stats.items()
        }
    )


def interrupted(config: RunConfig) -> RunConfig:
    """The same run, stopped partway, so that it can be resumed from a damaged checkpoint.

    The stopping point is past the last curriculum stage and past the warmup, because a
    first half that never reached the schedule the run was configured with would be a
    different run rather than the first half of this one.

    Args:
        config: The run configuration.

    Returns:
        The configuration of the first half.

    Raises:
        ValidationError: If the run is too short to have a first half that satisfies its
            own schedule.
    """
    training = config.training
    floor = max(training.curriculum_epochs[-1] + 1, training.warmup_epochs + 1)
    epochs = max(floor, training.epochs // 2)
    if epochs >= training.epochs:
        raise ValidationError(
            f"a run of {training.epochs} epochs whose last curriculum stage starts at "
            f"{training.curriculum_epochs[-1]} cannot be interrupted and resumed"
        )
    return config.model_copy(update={"training": training.model_copy(update={"epochs": epochs})})


@dataclass(frozen=True, slots=True)
class RenamedPredictor:
    """A predictor reported under a name other than its own.

    The symmetry fault wraps a trained model in a transformation that already has a name
    of its own, and that name belongs to one of the suite's permanent fixtures. Reporting
    the wrapped model under it would collide with the fixture, and reporting it under
    anything other than the model's own name would leave the comparison with nothing on
    the baseline side to line it up against.

    Attributes:
        inner: The predictor being renamed.
        name: What the harness reports it as.
    """

    inner: Predictor
    name: str

    @property
    def dt(self) -> float:
        """Size of the step this predictor advances by."""
        return self.inner.dt

    def step(self, state: State) -> State:
        """Advance one step.

        Args:
            state: The current state.

        Returns:
            Whatever the wrapped predictor returns.
        """
        return self.inner.step(state)
