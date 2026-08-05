"""Gravitational N-body dynamics in two dimensions.

The reference solver is velocity Verlet. Everything an outer layer needs is reachable
through `NBodySystem`, which implements the `System` protocol.
"""

from nnphysics.systems.nbody.dynamics import NBodyDynamics, kinetic_energy
from nnphysics.systems.nbody.integrators import RungeKutta4, VelocityVerlet
from nnphysics.systems.nbody.state import NBODY_STATE_SPEC, make_state, unpack

__all__ = [
    "NBODY_STATE_SPEC",
    "NBodyDynamics",
    "RungeKutta4",
    "VelocityVerlet",
    "kinetic_energy",
    "make_state",
    "unpack",
]
