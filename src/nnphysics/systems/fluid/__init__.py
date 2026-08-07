"""Two dimensional incompressible flow, in vorticity and streamfunction form.

The reference solver is pseudo-spectral in space and integrating factor Runge-Kutta in
time. Everything an outer layer needs is reachable through `FluidSystem`, which
implements the `System` protocol.
"""

from nnphysics.systems.fluid.dynamics import FluidDynamics
from nnphysics.systems.fluid.grid import (
    VORTICITY,
    VORTICITY_FIELD,
    FluidGrid,
    SpectralOperators,
)
from nnphysics.systems.fluid.initial_conditions import (
    FLUID_REGIMES,
    characteristic_length,
    initial_state,
    taylor_green_decay_rate,
    taylor_green_vorticity,
    viscosity,
)
from nnphysics.systems.fluid.integrators import DEFAULT_CFL_NUMBER, IntegratingFactorRK4
from nnphysics.systems.fluid.invariants import ENSTROPHY, SPECIFIC_ENERGY, Enstrophy, KineticEnergy
from nnphysics.systems.fluid.refinement import SpectralRefinement
from nnphysics.systems.fluid.symmetries import QuarterTurn, Translation
from nnphysics.systems.fluid.system import FluidSystem, build_fluid

__all__ = [
    "DEFAULT_CFL_NUMBER",
    "ENSTROPHY",
    "FLUID_REGIMES",
    "SPECIFIC_ENERGY",
    "VORTICITY",
    "VORTICITY_FIELD",
    "Enstrophy",
    "FluidDynamics",
    "FluidGrid",
    "FluidSystem",
    "IntegratingFactorRK4",
    "KineticEnergy",
    "QuarterTurn",
    "SpectralOperators",
    "SpectralRefinement",
    "Translation",
    "build_fluid",
    "characteristic_length",
    "initial_state",
    "taylor_green_decay_rate",
    "taylor_green_vorticity",
    "viscosity",
]
