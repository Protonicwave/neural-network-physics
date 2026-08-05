"""Vorticity and streamfunction dynamics for a 2D incompressible flow.

Vorticity is the only state variable. The velocity is recovered from it through the
streamfunction, `omega = -laplacian(psi)` and `u = (dpsi/dy, -dpsi/dx)`, which is
divergence free by construction, so incompressibility is exact rather than enforced.

The nonlinear term is evaluated pseudo-spectrally: derivatives in Fourier space, the
products in physical space. That is only stable with dealiasing, so the two thirds rule
is applied to the term's input as well as to its output. Truncating the input is the
part that matters: a product of two retained modes can alias only into modes that are
discarded anyway, whereas a product involving an unretained mode can fold straight back
into the resolved band and pass for physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.systems.fluid.grid import FluidGrid, SpectralOperators

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray
    from nnphysics.systems.fluid.grid import ComplexArray

__all__ = ["FluidDynamics"]


@dataclass(frozen=True, slots=True)
class FluidDynamics:
    """The vorticity equation on one grid, at one viscosity.

    Attributes:
        grid: The domain and its Fourier conventions.
        viscosity: Kinematic viscosity. Zero gives the inviscid Euler limit.
        dealias: Whether the two thirds rule is applied. Only ever false in the test
            that shows what it buys.
        operators: Cached wavenumber arrays, derived from the grid.
    """

    grid: FluidGrid = field(default_factory=FluidGrid)
    viscosity: float = 0.0
    dealias: bool = True
    operators: SpectralOperators = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.viscosity) or self.viscosity < 0.0:
            raise ValidationError(
                f"viscosity must be finite and not negative, got {self.viscosity}"
            )
        object.__setattr__(self, "operators", SpectralOperators.build(self.grid))

    def streamfunction_spectrum(self, vorticity_spectrum: ComplexArray) -> ComplexArray:
        """Solve `-laplacian(psi) = omega` in Fourier space.

        The mean mode is left at zero. It is undetermined, and a solve that forgets to
        set it leaves an arbitrary constant in the streamfunction that no test of the
        velocity would ever see.

        Args:
            vorticity_spectrum: Half spectrum of the vorticity.

        Returns:
            Half spectrum of the streamfunction.
        """
        spectrum: ComplexArray = vorticity_spectrum * self.operators.inverse_squared_wavenumber
        return spectrum

    def velocity_from_spectrum(
        self, vorticity_spectrum: ComplexArray
    ) -> tuple[FloatArray, FloatArray]:
        """Recover the velocity field from a vorticity spectrum.

        Args:
            vorticity_spectrum: Half spectrum of the vorticity.

        Returns:
            The two velocity components, each shape `(n, n)`.
        """
        streamfunction = self.streamfunction_spectrum(vorticity_spectrum)
        along_x = self.grid.inverse(1j * self.operators.wavenumber_y * streamfunction)
        along_y = self.grid.inverse(-1j * self.operators.wavenumber_x * streamfunction)
        return along_x, along_y

    def velocity(self, vorticity: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Recover the velocity field from a vorticity field.

        Args:
            vorticity: Vorticity at each grid point.

        Returns:
            The two velocity components, each shape `(n, n)`.
        """
        return self.velocity_from_spectrum(self.grid.forward(vorticity))

    def nonlinear(self, vorticity_spectrum: ComplexArray) -> ComplexArray:
        """The advection term `-(u . grad) omega`, in Fourier space.

        Args:
            vorticity_spectrum: Half spectrum of the vorticity.

        Returns:
            Half spectrum of the term, dealiased unless dealiasing is switched off.
        """
        resolved = self._truncate(vorticity_spectrum)
        along_x, along_y = self.velocity_from_spectrum(resolved)
        gradient_x = self.grid.inverse(1j * self.operators.wavenumber_x * resolved)
        gradient_y = self.grid.inverse(1j * self.operators.wavenumber_y * resolved)
        advection = self.grid.forward(along_x * gradient_x + along_y * gradient_y)
        return self._truncate(-advection)

    def energy(self, vorticity: FloatArray) -> float:
        """Kinetic energy per unit area, `<|u|^2> / 2`.

        Args:
            vorticity: Vorticity at each grid point.

        Returns:
            The energy.
        """
        along_x, along_y = self.velocity(vorticity)
        return float(0.5 * np.mean(along_x**2 + along_y**2))

    def enstrophy(self, vorticity: FloatArray) -> float:
        """Enstrophy per unit area, `<omega^2> / 2`.

        Args:
            vorticity: Vorticity at each grid point.

        Returns:
            The enstrophy.
        """
        return float(0.5 * np.mean(vorticity**2))

    def maximum_speed(self, vorticity_spectrum: ComplexArray) -> float:
        """Largest `|u| + |v|` on the grid, which is what sets the advective step limit.

        Args:
            vorticity_spectrum: Half spectrum of the vorticity.

        Returns:
            The largest sum of component magnitudes at any point.
        """
        along_x, along_y = self.velocity_from_spectrum(vorticity_spectrum)
        return float(np.max(np.abs(along_x) + np.abs(along_y)))

    def _truncate(self, spectrum: ComplexArray) -> ComplexArray:
        """Apply the two thirds rule, unless dealiasing is switched off."""
        if not self.dealias:
            return spectrum
        truncated: ComplexArray = spectrum * self.operators.retained
        return truncated
