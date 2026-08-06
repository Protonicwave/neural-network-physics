"""State, trajectory and rollout containers, and the specifications that describe them.

These are the only data shapes the evaluation harness ever sees. A state is a mapping
from field name to array, so a system can declare any set of fields without any outer
layer knowing what those fields mean.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import NumericalError, ValidationError
from nnphysics.core.units import DIMENSIONLESS, Dimension

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:
    FloatArray = np.ndarray

__all__ = [
    "FieldSpec",
    "FloatArray",
    "MetricResult",
    "Prediction",
    "Regime",
    "Rollout",
    "State",
    "StateSpec",
    "Trajectory",
]


def _freeze_fields(fields: Mapping[str, FloatArray], *, context: str) -> Mapping[str, FloatArray]:
    """Copy a field mapping into an immutable one, checking names and dtypes."""
    if not fields:
        raise ValidationError(f"{context} has no fields")
    frozen: dict[str, FloatArray] = {}
    for name, array in fields.items():
        if not name:
            raise ValidationError(f"{context} has a field with an empty name")
        if array.dtype != np.float64:
            raise ValidationError(
                f"{context} field {name!r} has dtype {array.dtype}, expected float64"
            )
        frozen[name] = array
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """The declared shape and dimension of one field of a state.

    Attributes:
        name: Field name, unique within a state.
        shape: Shape of the field for a single state. `None` marks a free extent, for
            example a count that varies between trajectories.
        dimension: Physical dimension of the field.
        description: What the field means, for report readers.
    """

    name: str
    shape: tuple[int | None, ...]
    dimension: Dimension = DIMENSIONLESS
    description: str = ""

    def matches(self, shape: tuple[int, ...]) -> bool:
        """Whether a concrete array shape satisfies this specification."""
        if len(shape) != len(self.shape):
            return False
        return all(
            declared is None or declared == actual
            for declared, actual in zip(self.shape, shape, strict=True)
        )


@dataclass(frozen=True, slots=True)
class StateSpec:
    """The full set of fields a system's states carry.

    Attributes:
        fields: Field specifications, in declaration order.
    """

    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValidationError("a state specification must declare at least one field")
        names = [spec.name for spec in self.fields]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValidationError(f"duplicate field names in a state: {sorted(duplicates)}")

    @property
    def names(self) -> tuple[str, ...]:
        """Declared field names, in declaration order."""
        return tuple(spec.name for spec in self.fields)

    def validate(self, state: State) -> None:
        """Check a state against this specification.

        Raises:
            ValidationError: If a field is missing, unexpected or the wrong shape.
        """
        expected = set(self.names)
        actual = set(state.fields)
        if missing := sorted(expected - actual):
            raise ValidationError(f"state is missing fields {missing}")
        if unexpected := sorted(actual - expected):
            raise ValidationError(f"state has undeclared fields {unexpected}")
        for spec in self.fields:
            shape = state.fields[spec.name].shape
            if not spec.matches(shape):
                raise ValidationError(
                    f"field {spec.name!r} has shape {shape}, expected {spec.shape}"
                )


@dataclass(frozen=True, slots=True)
class Regime:
    """A named point or region of a system's parameter space.

    Attributes:
        name: Identifier used in configuration and in reports.
        parameters: Parameter values that define the regime.
    """

    name: str
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("a regime must have a name")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class State:
    """One instantaneous state of a system.

    Attributes:
        fields: Field name to array. Arrays are float64 and are not copied, so callers
            must not mutate an array after handing it over.
        time: Simulation time of this state.
    """

    fields: Mapping[str, FloatArray]
    time: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze_fields(self.fields, context="state"))
        object.__setattr__(self, "time", float(self.time))

    @property
    def names(self) -> tuple[str, ...]:
        """Field names present in this state."""
        return tuple(self.fields)

    def require_finite(self) -> None:
        """Raise if any field holds a non finite value.

        Raises:
            NumericalError: If a field contains a NaN or an infinity.
        """
        for name, array in self.fields.items():
            if not np.isfinite(array).all():
                raise NumericalError(f"state field {name!r} at t={self.time} is not finite")


@dataclass(frozen=True, slots=True)
class Prediction:
    """A predicted state together with how uncertain the predictor is about it.

    A `Predictor` returns a state and says nothing about its own confidence. Some can say
    more: an ensemble knows how far its members disagree. This is how that reaches the
    harness without any metric learning what an ensemble is, and without the plain
    predictor interface growing a field that most predictors would have to fake.

    The spread is a standard deviation in the field's own units, one number per element,
    so it can be compared against a residual directly. Every field of the state carries
    one, including a field the predictor merely copies: the honest spread there is zero,
    and leaving it out would make a missing claim indistinguishable from a confident one.

    Attributes:
        state: The predicted state.
        spread: Field name to the predictive standard deviation of that field.
    """

    state: State
    spread: Mapping[str, FloatArray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "spread", _freeze_fields(self.spread, context="prediction spread"))
        missing = sorted(set(self.state.fields) - set(self.spread))
        if missing:
            raise ValidationError(f"prediction carries no spread for fields {missing}")
        unexpected = sorted(set(self.spread) - set(self.state.fields))
        if unexpected:
            raise ValidationError(f"prediction carries a spread for undeclared fields {unexpected}")
        for name, array in self.spread.items():
            if array.shape != self.state.fields[name].shape:
                raise ValidationError(
                    f"prediction spread for {name!r} has shape {array.shape}, expected "
                    f"{self.state.fields[name].shape}"
                )
            # A comparison against a NaN is false, so this rejects a negative spread and
            # lets a non finite one through to the rollout driver, whose job it is to stop
            # and say which step stopped producing numbers.
            if bool(np.any(array < 0.0)):
                raise ValidationError(f"prediction spread for {name!r} has a negative value")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A time ordered sequence of states, stored as stacked arrays.

    Each field array has shape `(n_steps, *field_shape)` and `times` has shape
    `(n_steps,)`. Stacking rather than holding a list of states keeps metrics
    vectorised over time.

    Attributes:
        fields: Field name to stacked array.
        times: Strictly increasing simulation times.
    """

    fields: Mapping[str, FloatArray]
    times: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze_fields(self.fields, context="trajectory"))
        if self.times.ndim != 1:
            raise ValidationError(
                f"trajectory times must be one dimensional, got {self.times.ndim}"
            )
        if self.times.size == 0:
            raise ValidationError("trajectory has no steps")
        if not np.all(np.diff(self.times) > 0):
            raise ValidationError("trajectory times must be strictly increasing")
        for name, array in self.fields.items():
            if array.shape[:1] != self.times.shape:
                raise ValidationError(
                    f"trajectory field {name!r} has leading extent {array.shape[:1]}, "
                    f"expected {self.times.shape}"
                )

    def __len__(self) -> int:
        return int(self.times.size)

    def __getitem__(self, index: int) -> State:
        return State(
            fields={name: array[index] for name, array in self.fields.items()},
            time=float(self.times[index]),
        )

    def __iter__(self) -> Iterator[State]:
        for index in range(len(self)):
            yield self[index]

    @property
    def names(self) -> tuple[str, ...]:
        """Field names present in this trajectory."""
        return tuple(self.fields)

    @classmethod
    def from_states(cls, states: Sequence[State]) -> Trajectory:
        """Stack a sequence of states into a trajectory.

        Args:
            states: At least one state, all carrying the same field names. The order the
                names were inserted in does not matter: a state is a mapping, and two
                states built by different code paths carry the same physics whichever
                order their fields were written in.

        Returns:
            The stacked trajectory.

        Raises:
            ValidationError: If the sequence is empty or the field names differ.
        """
        if not states:
            raise ValidationError("cannot build a trajectory from an empty sequence of states")
        names = states[0].names
        expected = set(names)
        for position, state in enumerate(states):
            if set(state.names) != expected:
                raise ValidationError(
                    f"state {position} has fields {state.names}, expected {names}"
                )
        return cls(
            fields={name: np.stack([state.fields[name] for state in states]) for name in names},
            times=np.asarray([state.time for state in states], dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class Rollout:
    """A predicted trajectory paired with the reference trajectory it is scored against.

    Attributes:
        predicted: The predictor's trajectory.
        reference: Ground truth over the same times.
        predictor: Name of the predictor that produced `predicted`.
        system: Name of the system the trajectories belong to.
        spread: The predictor's own uncertainty at each step, or `None` for a predictor
            that claims none. Same fields and length as `predicted`, with zero at the
            initial state, which is known exactly because it was handed in.
    """

    predicted: Trajectory
    reference: Trajectory
    predictor: str
    system: str
    spread: Trajectory | None = None

    def __post_init__(self) -> None:
        if len(self.predicted) != len(self.reference):
            raise ValidationError(
                f"rollout lengths differ: predicted {len(self.predicted)}, "
                f"reference {len(self.reference)}"
            )
        if self.predicted.names != self.reference.names:
            raise ValidationError(
                f"rollout fields differ: predicted {self.predicted.names}, "
                f"reference {self.reference.names}"
            )
        if not np.allclose(self.predicted.times, self.reference.times):
            raise ValidationError("rollout times differ between predicted and reference")
        if self.spread is not None:
            if len(self.spread) != len(self.predicted):
                raise ValidationError(
                    f"rollout spread has {len(self.spread)} steps, predicted has "
                    f"{len(self.predicted)}"
                )
            if set(self.spread.names) != set(self.predicted.names):
                raise ValidationError(
                    f"rollout spread has fields {self.spread.names}, predicted has "
                    f"{self.predicted.names}"
                )

    def __len__(self) -> int:
        return len(self.reference)

    @property
    def times(self) -> FloatArray:
        """Times shared by both trajectories."""
        return self.reference.times


@dataclass(frozen=True, slots=True)
class MetricResult:
    """What a metric returns: named scalars and optional series for plotting.

    Attributes:
        name: Name of the metric that produced this result.
        scalars: Named scalar values, the numbers that go in a report table.
        series: Named one dimensional arrays over rollout steps, for plots.
    """

    name: str
    scalars: Mapping[str, float]
    series: Mapping[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("a metric result must carry the metric name")
        object.__setattr__(self, "scalars", MappingProxyType(dict(self.scalars)))
        object.__setattr__(self, "series", MappingProxyType(dict(self.series)))
