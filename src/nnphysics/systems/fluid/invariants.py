"""The two quantities a 2D flow lives or dies by.

In two dimensions both energy and enstrophy are conserved by the inviscid equations,
which is what makes the flow so different from a three dimensional one. Viscosity drains
both, and it drains enstrophy faster, because dissipation goes as `k^2` against each and
enstrophy is the one weighted towards high wavenumbers.

Neither is declared as conserved when there is viscosity. A quantity that must decrease
is a statement a metric can test just as sharply as conservation, and declaring a
draining quantity as conserved with a loose tolerance would throw that statement away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nnphysics.core.protocols import Conservation
from nnphysics.core.units import VELOCITY, Dimension
from nnphysics.systems.fluid.grid import VORTICITY

if TYPE_CHECKING:
    from nnphysics.core.types import State
    from nnphysics.systems.fluid.dynamics import FluidDynamics

__all__ = ["ENSTROPHY", "SPECIFIC_ENERGY", "Enstrophy", "KineticEnergy"]

SPECIFIC_ENERGY = VELOCITY**2
"""Energy per unit mass: the fluid has no density in these units."""

ENSTROPHY = VORTICITY**2


def _conservation(viscosity: float) -> Conservation:
    """What a quadratic invariant of this flow does at a given viscosity."""
    return Conservation.APPROXIMATE if viscosity == 0.0 else Conservation.DECAYING


@dataclass(frozen=True, slots=True)
class KineticEnergy:
    """Kinetic energy per unit area, `<|u|^2> / 2`.

    Attributes:
        dynamics: The flow whose viscosity decides what this quantity does.
        rtol: Relative drift, or relative increase once the flow is viscous, that is
            still acceptable.
    """

    dynamics: FluidDynamics
    rtol: float = 1e-8

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "energy"

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of the scalar."""
        return SPECIFIC_ENERGY

    @property
    def conservation(self) -> Conservation:
        """Conserved by the inviscid equations, drained by viscosity."""
        return _conservation(self.dynamics.viscosity)

    def evaluate(self, state: State) -> float:
        """Compute the energy of one state.

        Args:
            state: The state to measure.

        Returns:
            The kinetic energy per unit area.
        """
        return self.dynamics.energy(self.dynamics.grid.unpack(state))


@dataclass(frozen=True, slots=True)
class Enstrophy:
    """Enstrophy per unit area, `<omega^2> / 2`.

    Attributes:
        dynamics: The flow whose viscosity decides what this quantity does.
        rtol: Relative drift, or relative increase once the flow is viscous, that is
            still acceptable.
    """

    dynamics: FluidDynamics
    rtol: float = 1e-8

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "enstrophy"

    @property
    def dimension(self) -> Dimension:
        """Physical dimension of the scalar."""
        return ENSTROPHY

    @property
    def conservation(self) -> Conservation:
        """Conserved by the inviscid equations, drained by viscosity faster than energy."""
        return _conservation(self.dynamics.viscosity)

    def evaluate(self, state: State) -> float:
        """Compute the enstrophy of one state.

        Args:
            state: The state to measure.

        Returns:
            The enstrophy per unit area.
        """
        return self.dynamics.enstrophy(self.dynamics.grid.unpack(state))
