"""The fluid system: the object every outer layer sees.

The grid is a system level parameter and the Reynolds number is a regime one. That split
follows what each thing is: the grid is a statement about how well the flow is resolved,
the Reynolds number is a statement about which flow it is. Holding a regime out of
training therefore means holding out a Reynolds number, which is the generalisation
question worth asking of a fluid surrogate.

The viscosity that follows from a regime decides whether energy and enstrophy are
conserved or drained, which is why the invariants are asked for by regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nnphysics.core.protocols import Invariant, Predictor, Refinement, Symmetry
from nnphysics.systems.base import (
    SystemParameters,
    check_parameter_names,
    float_parameter,
    int_parameter,
)
from nnphysics.systems.fluid.dynamics import FluidDynamics
from nnphysics.systems.fluid.grid import TWO_PI, FluidGrid
from nnphysics.systems.fluid.initial_conditions import FLUID_REGIMES, initial_state, viscosity
from nnphysics.systems.fluid.integrators import IntegratingFactorRK4
from nnphysics.systems.fluid.invariants import Enstrophy, KineticEnergy
from nnphysics.systems.fluid.refinement import SpectralRefinement
from nnphysics.systems.fluid.symmetries import QuarterTurn, Translation

if TYPE_CHECKING:
    import numpy as np

    from nnphysics.core.types import Regime, State, StateSpec

__all__ = ["FLUID_SYSTEM_PARAMETERS", "FluidSystem", "build_fluid"]

FLUID_SYSTEM_PARAMETERS = ("grid_size", "domain_length")

_DEFAULT_GRID_SIZE = 64
"""Large enough for the declared regimes, small enough to roll out on eight CPU cores."""


@dataclass(frozen=True, slots=True)
class FluidSystem:
    """Two dimensional incompressible flow in vorticity and streamfunction form.

    Attributes:
        grid: The doubly periodic domain and its resolution.
    """

    grid: FluidGrid = field(default_factory=FluidGrid)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "fluid"

    @property
    def state_spec(self) -> StateSpec:
        """Shape and meaning of the fields a state carries."""
        return self.grid.state_spec

    @property
    def regimes(self) -> tuple[Regime, ...]:
        """Named regions of parameter space this system declares."""
        return FLUID_REGIMES

    @property
    def symmetries(self) -> tuple[Symmetry, ...]:
        """Transformations that map the grid onto itself exactly.

        The shifts are whole numbers of cells and are fixed rather than drawn, so that an
        equivariance test is reproducible and does not have to invent them.
        """
        return (
            Translation(self.grid, (self.grid.size // 4, -(self.grid.size // 8))),
            QuarterTurn(self.grid, 1),
        )

    @property
    def refinements(self) -> tuple[Refinement, ...]:
        """The same flow on twice the grid, in each direction.

        One factor rather than several. The claim a neural operator makes is that it is
        resolution independent, and doubling either finds it false or does not; a third
        grid costs another rollout of a field four times the size and says the same
        thing again.
        """
        return (SpectralRefinement(self.grid, 2),)

    def dynamics(self, regime: Regime) -> FluidDynamics:
        """Build the vorticity equation for a regime.

        Args:
            regime: The regime, whose Reynolds number fixes the viscosity.

        Returns:
            The dynamics.

        Raises:
            ValidationError: If the Reynolds number is missing or not positive.
        """
        return FluidDynamics(self.grid, viscosity(regime, self.grid))

    def invariants(self, regime: Regime) -> tuple[Invariant, ...]:
        """Energy and enstrophy, declared as conserved or draining by the regime.

        Args:
            regime: The regime the invariants will be measured in.

        Returns:
            The two invariants.

        Raises:
            ValidationError: If the Reynolds number is missing or not positive.
        """
        dynamics = self.dynamics(regime)
        return (KineticEnergy(dynamics), Enstrophy(dynamics))

    def initial_state(self, regime: Regime, rng: np.random.Generator) -> State:
        """Draw one initial condition from a regime.

        Args:
            regime: The regime to draw from.
            rng: Generator every random draw comes from.

        Returns:
            The initial state, at time zero.

        Raises:
            UnknownNameError: If the regime name has no generator.
            ValidationError: If a regime parameter is unknown or out of range.
        """
        return initial_state(regime, rng, self.dynamics(regime))

    def reference_predictor(self, regime: Regime, dt: float) -> Predictor:
        """Build the solver that produces ground truth for a regime.

        Args:
            regime: The regime the solver will be run in, which fixes the viscosity.
            dt: Step size.

        Returns:
            The reference solver.

        Raises:
            ValidationError: If the step size is not positive, or the regime is invalid.
        """
        return IntegratingFactorRK4(self.dynamics(regime), dt)


def build_fluid(parameters: SystemParameters) -> FluidSystem:
    """Build the fluid system from system level parameters.

    Args:
        parameters: Accepts `grid_size` and `domain_length`.

    Returns:
        The system.

    Raises:
        ValidationError: If a parameter is unknown or out of range.
    """
    context = "fluid system"
    check_parameter_names(parameters, FLUID_SYSTEM_PARAMETERS, context=context)
    return FluidSystem(
        FluidGrid(
            size=int_parameter(parameters, "grid_size", _DEFAULT_GRID_SIZE, context=context),
            length=float_parameter(parameters, "domain_length", TWO_PI, context=context),
        )
    )
