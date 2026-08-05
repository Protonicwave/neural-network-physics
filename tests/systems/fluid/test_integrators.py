from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest

from nnphysics.core.errors import NumericalError, ValidationError
from nnphysics.core.protocols import Predictor
from nnphysics.core.types import FloatArray, State
from nnphysics.systems.fluid import (
    FluidDynamics,
    FluidGrid,
    IntegratingFactorRK4,
    taylor_green_decay_rate,
    taylor_green_vorticity,
)

SmoothVorticity = Callable[[FluidGrid], FloatArray]


def _advance(dynamics: FluidDynamics, vorticity: FloatArray, n_steps: int, dt: float) -> State:
    """Roll a field forward, returning the state so its own time can be read back."""
    state = dynamics.grid.make_state(vorticity)
    predictor = IntegratingFactorRK4(dynamics, dt)
    for _ in range(n_steps):
        state = predictor.step(state)
    return state


def _truncate_onto(field: FloatArray, fine: FluidGrid, coarse: FluidGrid) -> FloatArray:
    """Drop a fine solution onto a coarse grid by discarding the modes it cannot hold.

    Sampling the fine field at the coarse points instead would alias its unresolved
    modes onto the coarse ones, at exactly the size of the error being measured.
    """
    spectrum = fine.forward(field)
    half = coarse.size // 2
    truncated = np.zeros((coarse.size, half + 1), dtype=np.complex128)
    truncated[: half + 1, : half + 1] = spectrum[: half + 1, : half + 1]
    truncated[half + 1 :, : half + 1] = spectrum[fine.size - half + 1 :, : half + 1]
    # The Nyquist mode of the coarse grid has no unambiguous counterpart on the fine one.
    truncated[half, :] = 0.0
    truncated[:, half] = 0.0
    return coarse.inverse(truncated * (coarse.size / fine.size) ** 2)


class TestConstruction:
    @pytest.mark.parametrize("dt", [0.0, -0.1])
    def test_a_step_size_that_is_not_positive_is_rejected(self, dt: float) -> None:
        with pytest.raises(ValidationError, match="step size"):
            IntegratingFactorRK4(FluidDynamics(FluidGrid(16)), dt)

    def test_a_cfl_number_that_is_not_positive_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="CFL number"):
            IntegratingFactorRK4(FluidDynamics(FluidGrid(16)), 0.01, cfl_number=0.0)

    def test_it_is_a_predictor(self) -> None:
        predictor = IntegratingFactorRK4(FluidDynamics(FluidGrid(16)), 0.01)
        assert isinstance(predictor, Predictor)
        assert predictor.name == "fluid-ifrk4"
        assert predictor.dt == 0.01


class TestTaylorGreen:
    """The anchor test for this phase.

    A Taylor-Green mode is an eigenfunction of the Laplacian, so its advection vanishes
    identically and the whole solution is `exp(-2 nu k^2 t)` times the initial field. Any
    error in the Fourier normalisation, the streamfunction solve or the integrating
    factor moves that rate, and nothing else in the solver can hide it.
    """

    GRID = FluidGrid(32)
    WAVENUMBER = 2
    VISCOSITY = 0.05
    STEPS = 200
    STEP = 0.01

    def test_the_field_matches_the_analytic_solution(self) -> None:
        dynamics = FluidDynamics(self.GRID, self.VISCOSITY)
        initial = taylor_green_vorticity(self.GRID, self.WAVENUMBER, (0.3, -0.7))
        rate = taylor_green_decay_rate(self.GRID, self.WAVENUMBER, self.VISCOSITY)

        final = _advance(dynamics, initial, self.STEPS, self.STEP)

        assert self.GRID.unpack(final) == pytest.approx(
            initial * np.exp(-rate * final.time), rel=1e-10, abs=1e-12
        )

    def test_the_measured_decay_rate_matches_the_analytic_one(self) -> None:
        dynamics = FluidDynamics(self.GRID, self.VISCOSITY)
        initial = taylor_green_vorticity(self.GRID, self.WAVENUMBER)
        expected = taylor_green_decay_rate(self.GRID, self.WAVENUMBER, self.VISCOSITY)

        final = _advance(dynamics, initial, self.STEPS, self.STEP)
        # Enstrophy is quadratic in the field, so it decays at twice the rate.
        ratio = dynamics.enstrophy(self.GRID.unpack(final)) / dynamics.enstrophy(initial)
        measured = -0.5 * np.log(ratio) / final.time

        assert measured == pytest.approx(expected, rel=1e-9)

    def test_the_inviscid_mode_does_not_move_at_all(self) -> None:
        initial = taylor_green_vorticity(self.GRID, self.WAVENUMBER)

        final = _advance(FluidDynamics(self.GRID, 0.0), initial, self.STEPS, self.STEP)

        assert self.GRID.unpack(final) == pytest.approx(initial, abs=1e-13)


REFINEMENT_SIZES = (20, 28, 36)
REFINEMENT_REFERENCE = FluidGrid(96)
REFINEMENT_VISCOSITY = 0.01
REFINEMENT_STEPS = 200
REFINEMENT_STEP = 0.002


@pytest.fixture(scope="module")
def refinement_errors(smooth_vorticity: SmoothVorticity) -> list[float]:
    """Error against a resolved reference at each of a sequence of grid sizes."""
    reference = REFINEMENT_REFERENCE.unpack(
        _advance(
            FluidDynamics(REFINEMENT_REFERENCE, REFINEMENT_VISCOSITY),
            smooth_vorticity(REFINEMENT_REFERENCE),
            REFINEMENT_STEPS,
            REFINEMENT_STEP,
        )
    )
    errors = []
    for size in REFINEMENT_SIZES:
        grid = FluidGrid(size)
        final = grid.unpack(
            _advance(
                FluidDynamics(grid, REFINEMENT_VISCOSITY),
                smooth_vorticity(grid),
                REFINEMENT_STEPS,
                REFINEMENT_STEP,
            )
        )
        difference = final - _truncate_onto(reference, REFINEMENT_REFERENCE, grid)
        errors.append(float(np.abs(difference).max()))
    return errors


class TestSpectralConvergence:
    """Refinement must beat any fixed polynomial order.

    Second order would divide the error by less than two for each of these refinements.
    The scheme divides it by more than eight, and by more than a hundred across the pair,
    which is what exponential convergence looks like on a smooth solution.
    """

    def test_each_refinement_beats_second_order_by_a_wide_margin(
        self, refinement_errors: list[float]
    ) -> None:
        errors = pairwise(refinement_errors)
        sizes = pairwise(REFINEMENT_SIZES)

        for (earlier, later), (coarse, fine) in zip(errors, sizes, strict=True):
            second_order = (fine / coarse) ** 2
            assert earlier / later > 8.0
            assert earlier / later > 4.0 * second_order

    def test_the_error_falls_by_two_orders_over_the_refinement(
        self, refinement_errors: list[float]
    ) -> None:
        assert refinement_errors[0] / refinement_errors[-1] > 100.0


class TestStepSizeLimit:
    def test_a_step_that_crosses_more_than_a_cell_is_refused(
        self, smooth_vorticity: SmoothVorticity
    ) -> None:
        """An unstable spectral solution looks plausible for a while, so it must not run."""
        grid = FluidGrid(32)
        dynamics = FluidDynamics(grid, 0.0)
        state = grid.make_state(smooth_vorticity(grid))

        with pytest.raises(NumericalError, match="CFL"):
            IntegratingFactorRK4(dynamics, dt=1.0).step(state)

    def test_the_limit_scales_with_the_cfl_number(self, smooth_vorticity: SmoothVorticity) -> None:
        grid = FluidGrid(32)
        dynamics = FluidDynamics(grid, 0.0)
        state = grid.make_state(smooth_vorticity(grid))
        speed = dynamics.maximum_speed(grid.forward(grid.unpack(state)))
        limit = 0.5 * grid.spacing / speed

        IntegratingFactorRK4(dynamics, dt=0.99 * limit).step(state)
        with pytest.raises(NumericalError, match="CFL"):
            IntegratingFactorRK4(dynamics, dt=1.01 * limit).step(state)

    def test_a_flow_at_rest_has_no_limit(self) -> None:
        grid = FluidGrid(16)
        state = grid.make_state(np.zeros((16, 16)))

        stepped = IntegratingFactorRK4(FluidDynamics(grid, 0.0), dt=100.0).step(state)

        assert stepped.time == pytest.approx(100.0)

    def test_viscosity_places_no_limit_on_the_step(self) -> None:
        """The point of the integrating factor: an explicit scheme would need `dx^2 / nu`."""
        grid = FluidGrid(32)
        viscosity = 10.0
        initial = taylor_green_vorticity(grid, 2)
        explicit_limit = grid.spacing**2 / viscosity

        final = _advance(FluidDynamics(grid, viscosity), initial, 4, 20.0 * explicit_limit)

        rate = taylor_green_decay_rate(grid, 2, viscosity)
        assert grid.unpack(final) == pytest.approx(
            initial * np.exp(-rate * final.time), rel=1e-9, abs=1e-15
        )


class TestStepping:
    def test_time_advances_by_exactly_one_step(self, smooth_vorticity: SmoothVorticity) -> None:
        grid = FluidGrid(16)
        state = grid.make_state(smooth_vorticity(grid), time=1.5)

        stepped = IntegratingFactorRK4(FluidDynamics(grid, 0.01), 0.01).step(state)

        assert stepped.time == pytest.approx(1.51)

    def test_a_state_from_another_grid_is_refused(self) -> None:
        predictor = IntegratingFactorRK4(FluidDynamics(FluidGrid(16)), 0.01)

        with pytest.raises(ValidationError, match="shape"):
            predictor.step(FluidGrid(32).make_state(np.zeros((32, 32))))
