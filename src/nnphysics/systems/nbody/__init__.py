"""Gravitational N-body dynamics in two dimensions.

The reference solver is velocity Verlet. Everything an outer layer needs is reachable
through `NBodySystem`, which implements the `System` protocol.
"""

from nnphysics.systems.nbody.dynamics import NBodyDynamics, kinetic_energy
from nnphysics.systems.nbody.initial_conditions import NBODY_REGIMES, initial_state
from nnphysics.systems.nbody.integrators import RungeKutta4, VelocityVerlet
from nnphysics.systems.nbody.invariants import AngularMomentum, LinearMomentum, TotalEnergy
from nnphysics.systems.nbody.state import NBODY_STATE_SPEC, make_state, unpack
from nnphysics.systems.nbody.symmetries import GalileanBoost, Rotation, Translation
from nnphysics.systems.nbody.system import NBodySystem, build_nbody

__all__ = [
    "NBODY_REGIMES",
    "NBODY_STATE_SPEC",
    "AngularMomentum",
    "GalileanBoost",
    "LinearMomentum",
    "NBodyDynamics",
    "NBodySystem",
    "Rotation",
    "RungeKutta4",
    "TotalEnergy",
    "Translation",
    "VelocityVerlet",
    "build_nbody",
    "initial_state",
    "kinetic_energy",
    "make_state",
    "unpack",
]
