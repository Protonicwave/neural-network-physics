"""The three quantities gravitational dynamics conserve.

Each is a scalar read off a single state, which is all the evaluation harness is allowed
to know. Energy is approximate because any integrator makes an error in it. Linear and
angular momentum are exact: velocity Verlet conserves both to round off for central
pairwise forces, so any visible drift in them is a bug, not a step size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.protocols import Conservation
from nnphysics.core.units import ENERGY, LENGTH, MOMENTUM, Dimension
from nnphysics.systems.nbody.dynamics import kinetic_energy
from nnphysics.systems.nbody.state import unpack

if TYPE_CHECKING:
    from nnphysics.core.types import State
    from nnphysics.systems.nbody.dynamics import NBodyDynamics

__all__ = ["AngularMomentum", "LinearMomentum", "TotalEnergy"]

ANGULAR_MOMENTUM = MOMENTUM * LENGTH


@dataclass(frozen=True, slots=True)
class TotalEnergy:
    """Kinetic plus potential energy.

    The potential matches the softening the dynamics use, so this is conserved by the
    continuous softened system exactly and by the integrator to within its step error.

    Attributes:
        dynamics: The force law whose potential this energy belongs to.
        rtol: Relative drift over a reference rollout that is still acceptable.
    """

    dynamics: NBodyDynamics
    rtol: float = 1e-6

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "energy"

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of the scalar."""
        return ENERGY

    @property
    def conservation(self) -> Conservation:
        """Energy is bounded but not exact under any integrator."""
        return Conservation.APPROXIMATE

    def evaluate(self, state: State) -> float:
        """Compute the total energy of one state.

        Args:
            state: The state to measure.

        Returns:
            The total energy.
        """
        position, velocity, mass = unpack(state)
        return kinetic_energy(velocity, mass) + self.dynamics.potential_energy(position, mass)


@dataclass(frozen=True, slots=True)
class LinearMomentum:
    """Magnitude of the total linear momentum.

    Initial conditions are generated with the centre of mass at rest, so this is normally
    zero and drift in it must be read as an absolute quantity, not a relative one.

    Attributes:
        rtol: Relative drift over a reference rollout that is still acceptable.
    """

    rtol: float = 1e-12

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "linear_momentum"

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of the scalar."""
        return MOMENTUM

    @property
    def conservation(self) -> Conservation:
        """Internal forces cancel in pairs, so this is exact up to round off."""
        return Conservation.EXACT

    def evaluate(self, state: State) -> float:
        """Compute the momentum magnitude of one state.

        Args:
            state: The state to measure.

        Returns:
            The magnitude of the summed momentum.
        """
        _, velocity, mass = unpack(state)
        return float(np.linalg.norm(np.sum(mass[:, np.newaxis] * velocity, axis=0)))


@dataclass(frozen=True, slots=True)
class AngularMomentum:
    """Total angular momentum about the origin.

    In two dimensions this has one component, so it is signed rather than a magnitude and
    a sign flip is caught as drift.

    Attributes:
        rtol: Relative drift over a reference rollout that is still acceptable.
    """

    rtol: float = 1e-10

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "angular_momentum"

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of the scalar."""
        return ANGULAR_MOMENTUM

    @property
    def conservation(self) -> Conservation:
        """Central forces exert no net torque, so this is exact up to round off."""
        return Conservation.EXACT

    def evaluate(self, state: State) -> float:
        """Compute the angular momentum of one state.

        Args:
            state: The state to measure.

        Returns:
            The out of plane component of the total angular momentum.
        """
        position, velocity, mass = unpack(state)
        cross = position[:, 0] * velocity[:, 1] - position[:, 1] * velocity[:, 0]
        return float(np.sum(mass * cross))
