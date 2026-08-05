import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import FloatArray
from nnphysics.systems.fluid import FluidDynamics, FluidGrid, taylor_green_vorticity


def _divergence(dynamics: FluidDynamics, along_x: FloatArray, along_y: FloatArray) -> FloatArray:
    grid = dynamics.grid
    wavenumber_x, wavenumber_y = grid.wavenumbers()
    return grid.inverse(
        1j * wavenumber_x * grid.forward(along_x) + 1j * wavenumber_y * grid.forward(along_y)
    )


class TestConstruction:
    def test_a_negative_viscosity_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="viscosity"):
            FluidDynamics(FluidGrid(16), -1e-3)

    def test_the_inviscid_limit_is_allowed(self) -> None:
        assert FluidDynamics(FluidGrid(16), 0.0).viscosity == 0.0


class TestVelocity:
    def test_the_velocity_is_divergence_free(self) -> None:
        """Incompressibility is exact here because velocity comes from a streamfunction."""
        grid = FluidGrid(32)
        dynamics = FluidDynamics(grid)
        rng = np.random.default_rng(0)
        vorticity = grid.inverse(grid.forward(rng.standard_normal((32, 32))) * grid.dealias_mask())
        vorticity = vorticity - vorticity.mean()

        along_x, along_y = dynamics.velocity(vorticity)

        assert _divergence(dynamics, along_x, along_y) == pytest.approx(0.0, abs=1e-12)

    def test_the_curl_of_the_velocity_is_the_vorticity_again(self) -> None:
        grid = FluidGrid(32)
        dynamics = FluidDynamics(grid)
        vorticity = taylor_green_vorticity(grid, 3)
        wavenumber_x, wavenumber_y = grid.wavenumbers()

        along_x, along_y = dynamics.velocity(vorticity)
        curl = grid.inverse(
            1j * wavenumber_x * grid.forward(along_y) - 1j * wavenumber_y * grid.forward(along_x)
        )

        assert curl == pytest.approx(vorticity)

    def test_the_streamfunction_has_no_mean(self) -> None:
        """The mean mode is a free constant, and a solve that keeps it keeps a lie."""
        grid = FluidGrid(16)
        dynamics = FluidDynamics(grid)
        spectrum = grid.forward(taylor_green_vorticity(grid, 2))

        assert dynamics.streamfunction_spectrum(spectrum)[0, 0] == 0.0


class TestInvariantQuantities:
    def test_the_enstrophy_of_a_taylor_green_mode_is_analytic(self) -> None:
        grid = FluidGrid(32)
        vorticity = taylor_green_vorticity(grid, 2)

        # The mean of cos^2 cos^2 over the periodic domain is a quarter.
        assert FluidDynamics(grid).enstrophy(vorticity) == pytest.approx(0.125)

    def test_energy_and_enstrophy_are_related_by_the_wavenumber(self) -> None:
        """For a single mode the enstrophy is `k^2` times the energy, which pins both."""
        grid = FluidGrid(32)
        dynamics = FluidDynamics(grid)
        vorticity = taylor_green_vorticity(grid, 3)
        squared_wavenumber = 2.0 * 3.0**2

        assert dynamics.enstrophy(vorticity) == pytest.approx(
            squared_wavenumber * dynamics.energy(vorticity)
        )


class TestNonlinearTerm:
    def test_it_vanishes_for_a_laplacian_eigenfunction(self) -> None:
        """Streamfunction parallel to vorticity means no self advection, exactly."""
        grid = FluidGrid(32)
        spectrum = grid.forward(taylor_green_vorticity(grid, 2))

        term = FluidDynamics(grid).nonlinear(spectrum)

        assert np.abs(term).max() < 1e-12

    def test_it_conserves_energy_and_enstrophy_instantaneously(self) -> None:
        """Advection only moves the two quadratic invariants between modes."""
        grid = FluidGrid(32)
        dynamics = FluidDynamics(grid)
        rng = np.random.default_rng(1)
        vorticity = grid.inverse(grid.forward(rng.standard_normal((32, 32))) * grid.dealias_mask())
        vorticity = vorticity - vorticity.mean()
        spectrum = grid.forward(vorticity)

        rate = grid.inverse(dynamics.nonlinear(spectrum))
        streamfunction = grid.inverse(dynamics.streamfunction_spectrum(spectrum))

        # d(enstrophy)/dt is <omega * domega/dt>, d(energy)/dt is <psi * domega/dt>.
        assert float(np.mean(vorticity * rate)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.mean(streamfunction * rate)) == pytest.approx(0.0, abs=1e-12)


class TestDealiasing:
    """A quadratic term folds unresolved modes back onto resolved ones unless it is cut.

    The initial condition here sits above the two thirds cutoff and below the Nyquist
    mode, so the grid can carry it but the scheme must not advect it. Its own product
    lands beyond Nyquist and folds back to a low wavenumber, where it is indistinguishable
    from physics.
    """

    SIZE = 32
    LOW_MODE = 6

    def _low_wavenumber_amplitude(self, dealias: bool) -> float:
        grid = FluidGrid(self.SIZE)
        along_x, along_y = grid.coordinates()
        vorticity = np.cos(14.0 * along_x + 2.0 * along_y) + np.cos(15.0 * along_x - along_y)
        vorticity = vorticity - vorticity.mean()

        term = FluidDynamics(grid, dealias=dealias).nonlinear(grid.forward(vorticity))

        modes_x = np.abs(np.fft.fftfreq(self.SIZE, d=1.0 / self.SIZE)).astype(np.int64)
        modes_y = np.arange(self.SIZE // 2 + 1)
        low = (modes_x[:, np.newaxis] <= self.LOW_MODE) & (modes_y[np.newaxis, :] <= self.LOW_MODE)
        return float(np.abs(term * low).max())

    def test_without_it_the_low_wavenumbers_are_contaminated(self) -> None:
        assert self._low_wavenumber_amplitude(dealias=False) > 1.0

    def test_with_it_the_low_wavenumbers_stay_empty(self) -> None:
        assert self._low_wavenumber_amplitude(dealias=True) < 1e-20
