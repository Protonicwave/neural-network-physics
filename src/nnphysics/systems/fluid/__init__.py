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

__all__ = [
    "DEFAULT_CFL_NUMBER",
    "FLUID_REGIMES",
    "VORTICITY",
    "VORTICITY_FIELD",
    "FluidDynamics",
    "FluidGrid",
    "IntegratingFactorRK4",
    "SpectralOperators",
    "characteristic_length",
    "initial_state",
    "taylor_green_decay_rate",
    "taylor_green_vorticity",
    "viscosity",
]
