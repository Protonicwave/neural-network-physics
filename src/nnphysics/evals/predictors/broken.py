"""Predictors that are wrong on purpose.

These are permanent fixtures, not throwaway code. Every metric in the suite earns its
place by catching at least one of them, and the ones a metric does not catch are as
informative as the ones it does: persistence conserves every invariant perfectly, and a
predictor that rotates the world each step conserves energy and momentum too. A suite
built only against a working model would never learn either fact.

None of them knows which system it is breaking. Where a fixture needs to single out a
kind of field, it reads the dimension the system declared for that field rather than its
name, and where it needs a transformation it uses one the system declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import State

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor, Symmetry
    from nnphysics.core.types import FloatArray, StateSpec

__all__ = [
    "EnergyInjection",
    "LinearExtrapolation",
    "NoiseInjection",
    "Persistence",
    "SymmetryBreak",
    "rate_fields",
]

_SCALE_FLOOR = 1.0e-30
"""Keeps a field of all zeros from making a relative perturbation undefined."""


@dataclass(frozen=True, slots=True)
class Persistence:
    """Returns the state it was given, one interval later.

    The weakest predictor that is still well behaved. It conserves every invariant
    exactly, which is why a suite that only measured invariant drift would rate it
    perfect, and it is the floor any surrogate has to clear.

    Attributes:
        dt: The interval it claims to advance by.
    """

    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValidationError(f"step size must be positive, got {self.dt}")

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "persistence"

    def step(self, state: State) -> State:
        """Advance the clock and nothing else.

        Args:
            state: The current state.

        Returns:
            The same fields, one interval later.
        """
        return State(fields=dict(state.fields), time=state.time + self.dt)


class LinearExtrapolation:
    """Continues in a straight line through the last two states.

    Better than persistence over a short horizon, because a straight line through two
    states carries the rate of change, and worse over a long one, because nothing bends
    it back. That ordering is the reason both are here.

    The first step has no previous state to work from and is taken with the reference
    solver, which is what stops the scheme collapsing: extrapolating from two identical
    states reproduces them forever, so a self started version would be persistence under
    another name.

    This is the one predictor in the suite that carries state between calls. A rollout is
    detected as new when the state it is handed does not follow the last one it returned,
    so the same instance may be used for several rollouts.

    Args:
        reference: Solver used for the first step of each rollout.
    """

    def __init__(self, reference: Predictor) -> None:
        self._reference = reference
        self._previous: State | None = None
        self._expected_time: float | None = None

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "linear_extrapolation"

    @property
    def dt(self) -> float:
        """Size of the step this predictor advances by."""
        return self._reference.dt

    def reset(self) -> None:
        """Forget the previous state, so the next step starts a rollout."""
        self._previous = None
        self._expected_time = None

    def step(self, state: State) -> State:
        """Advance one step, extrapolating once there is something to extrapolate from.

        Args:
            state: The current state.

        Returns:
            The state one step later.
        """
        continues = self._expected_time is not None and np.isclose(
            state.time, self._expected_time, rtol=1e-9, atol=1e-12
        )
        if not continues or self._previous is None:
            following = self._reference.step(state)
        else:
            following = State(
                fields={
                    name: 2.0 * array - self._previous.fields[name]
                    for name, array in state.fields.items()
                },
                time=state.time + self.dt,
            )
        self._previous = state
        self._expected_time = following.time
        return following


@dataclass(frozen=True, slots=True)
class NoiseInjection:
    """The true solver with Gaussian noise added to every field at every step.

    The noise is scaled by each field's own root mean square, so one setting means the
    same thing for a position and for a vorticity. Errors of this shape are what a
    surrogate makes, and the interesting question is what they do over a long rollout
    rather than over one step, which is the question distribution drift asks.

    The draw has its own mean removed before it is added. A system may place a constraint
    on a field's mean, as the fluid does because a periodic streamfunction cannot
    represent a mean vorticity, and a fixture whose first step produced a state the
    system refuses would test the harness rather than the metric. Removing one mode out
    of a field's worth costs the fixture nothing.

    Attributes:
        inner: The solver the noise is added to.
        scale: Noise standard deviation as a fraction of a field's magnitude.
        rng: Generator every draw comes from. The only stochastic predictor here, and it
            is handed a generator rather than reaching for one.
    """

    inner: Predictor
    scale: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        if self.scale < 0.0:
            raise ValidationError(f"noise scale must not be negative, got {self.scale}")

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "noise"

    @property
    def dt(self) -> float:
        """Size of the step this predictor advances by."""
        return self.inner.dt

    def step(self, state: State) -> State:
        """Advance one step and corrupt the result.

        Args:
            state: The current state.

        Returns:
            The solver's state with noise added.
        """
        stepped = self.inner.step(state)
        return State(
            fields={
                name: array + self.scale * _magnitude(array) * self._draw(array.shape)
                for name, array in stepped.fields.items()
            },
            time=stepped.time,
        )

    def _draw(self, shape: tuple[int, ...]) -> FloatArray:
        """Standard normal noise with its own mean removed."""
        noise: FloatArray = self.rng.standard_normal(shape)
        return noise - np.mean(noise)


@dataclass(frozen=True, slots=True)
class EnergyInjection:
    """The true solver with a small constant amplification of every rate of change.

    The fixture the harness exists for. One step is accurate to the amplification factor,
    which a one step error metric will call excellent, while the energy it adds compounds
    over a rollout into a system that is not the one it started as. A metric that cannot
    separate those two statements is not measuring physics.

    Which fields to amplify is read off the declared dimensions rather than off names:
    a field whose dimension carries a power of time is a rate, and a system's energy is
    quadratic in its rates. Amplifying every field instead would leave a virialised
    cluster almost unchanged, because scaling positions and velocities together moves
    kinetic and potential energy in opposite directions.

    Attributes:
        inner: The solver being amplified.
        fields: Names of the fields amplified.
        factor: Amount added per step, so each field is scaled by `1 + factor`.
    """

    inner: Predictor
    fields: tuple[str, ...]
    factor: float

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValidationError("energy injection needs at least one field to amplify")
        if self.factor <= -1.0:
            raise ValidationError(f"an amplification of {self.factor} would invert a field")

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "energy_injection"

    @property
    def dt(self) -> float:
        """Size of the step this predictor advances by."""
        return self.inner.dt

    def step(self, state: State) -> State:
        """Advance one step and amplify the rates.

        Args:
            state: The current state.

        Returns:
            The solver's state with its rate fields scaled up.
        """
        stepped = self.inner.step(state)
        gain = 1.0 + self.factor
        return State(
            fields={
                name: array * gain if name in self.fields else array
                for name, array in stepped.fields.items()
            },
            time=stepped.time,
        )


@dataclass(frozen=True, slots=True)
class SymmetryBreak:
    """The true solver with a declared symmetry applied after every step.

    Subtle in a way worth stating, because it decides how the symmetry metric has to be
    written. The transformation applied here commutes with the dynamics, so this
    predictor conserves every invariant, and its equivariance under the very symmetry it
    applies is perfect. What it breaks is equivariance under the system's *other*
    symmetries, since a rotation composed with a translation is not a translation
    composed with a rotation. A metric that tested one symmetry would therefore have to
    be lucky. Testing every declared symmetry and taking the worst is what makes it not
    a matter of luck.

    Attributes:
        inner: The solver being twisted.
        symmetry: The transformation applied after each step.
    """

    inner: Predictor
    symmetry: Symmetry

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "symmetry_break"

    @property
    def dt(self) -> float:
        """Size of the step this predictor advances by."""
        return self.inner.dt

    def step(self, state: State) -> State:
        """Advance one step and transform the result.

        Args:
            state: The current state.

        Returns:
            The solver's state, transformed.
        """
        return self.symmetry.apply(self.inner.step(state))


def rate_fields(state_spec: StateSpec) -> tuple[str, ...]:
    """Names of the fields whose declared dimension carries a power of time.

    Args:
        state_spec: The system's declared fields.

    Returns:
        The field names, in declaration order.

    Raises:
        ValidationError: If the system declares no such field, in which case the
            dimensions cannot say where its energy lives.
    """
    names = tuple(spec.name for spec in state_spec.fields if spec.dimension.time != 0)
    if not names:
        raise ValidationError(
            "no declared field carries a power of time, so no field can be read as a rate"
        )
    return names


def _magnitude(array: FloatArray) -> float:
    """Root mean square of a field, floored so a field of zeros stays well defined."""
    return max(float(np.sqrt(np.mean(array**2))), _SCALE_FLOOR)
