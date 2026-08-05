"""Two integrators: one for production, one to prove in tests what the first one buys.

Velocity Verlet is symplectic, so its energy error oscillates within a bound set by the
step size instead of accumulating. Fourth order Runge-Kutta has the smaller error over a
short interval and no such bound, so it drifts. Only the first is used for ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError
from nnphysics.systems.nbody.state import make_state, unpack

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray, State
    from nnphysics.systems.nbody.dynamics import NBodyDynamics

__all__ = ["RungeKutta4", "VelocityVerlet"]


@dataclass(frozen=True, slots=True)
class VelocityVerlet:
    """Symplectic second order integrator, the reference solver for this system.

    Stability. For motion of angular frequency `w` the scheme is stable only while
    `dt < 2 / w`, and the energy error scales as `dt^2`, so a useful step is far below
    that limit: `dt` of a fiftieth of the shortest orbital period or less. The shortest
    period is set by the closest pair, and with Plummer softening `e` it is bounded below
    by roughly `2 pi sqrt(e^3 / (G M))`. A step chosen from the widest orbit alone will
    be unstable in a system with a tight binary.

    Attributes:
        dynamics: The force law.
        dt: Step size.
    """

    dynamics: NBodyDynamics
    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValidationError(f"step size must be positive, got {self.dt}")

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "nbody-verlet"

    def step(self, state: State) -> State:
        """Advance one step.

        Args:
            state: The current state.

        Returns:
            The state one step later.
        """
        position, velocity, mass = unpack(state)
        acceleration = self.dynamics.accelerations(position, mass)
        next_position = position + velocity * self.dt + 0.5 * acceleration * self.dt**2
        next_acceleration = self.dynamics.accelerations(next_position, mass)
        next_velocity = velocity + 0.5 * (acceleration + next_acceleration) * self.dt
        return make_state(next_position, next_velocity, mass, time=state.time + self.dt)


@dataclass(frozen=True, slots=True)
class RungeKutta4:
    """Classical fourth order Runge-Kutta, present to be compared against, not used.

    It is not symplectic. Over a long integration its energy error grows without bound
    even where velocity Verlet stays flat, which is the point the tests make.

    Attributes:
        dynamics: The force law.
        dt: Step size.
    """

    dynamics: NBodyDynamics
    dt: float

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValidationError(f"step size must be positive, got {self.dt}")

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "nbody-rk4"

    def step(self, state: State) -> State:
        """Advance one step.

        Args:
            state: The current state.

        Returns:
            The state one step later.
        """
        position, velocity, mass = unpack(state)
        half = 0.5 * self.dt

        position_1: FloatArray = velocity
        velocity_1 = self.dynamics.accelerations(position, mass)

        position_2: FloatArray = velocity + half * velocity_1
        velocity_2 = self.dynamics.accelerations(position + half * position_1, mass)

        position_3: FloatArray = velocity + half * velocity_2
        velocity_3 = self.dynamics.accelerations(position + half * position_2, mass)

        position_4: FloatArray = velocity + self.dt * velocity_3
        velocity_4 = self.dynamics.accelerations(position + self.dt * position_3, mass)

        weight = self.dt / 6.0
        next_position = position + weight * (
            position_1 + 2.0 * position_2 + 2.0 * position_3 + position_4
        )
        next_velocity = velocity + weight * (
            velocity_1 + 2.0 * velocity_2 + 2.0 * velocity_3 + velocity_4
        )
        return make_state(next_position, next_velocity, mass, time=state.time + self.dt)
