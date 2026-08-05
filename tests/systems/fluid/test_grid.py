import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import State
from nnphysics.systems.fluid import FluidGrid, SpectralOperators
from nnphysics.systems.fluid.grid import TWO_PI


class TestConstruction:
    def test_it_declares_one_field_of_the_right_shape(self) -> None:
        grid = FluidGrid(16)
        assert grid.state_spec.names == ("vorticity",)
        assert grid.state_spec.fields[0].shape == (16, 16)

    def test_the_spacing_covers_the_domain(self) -> None:
        grid = FluidGrid(16, 4.0)
        assert grid.spacing * grid.size == pytest.approx(4.0)

    @pytest.mark.parametrize("size", [4, 6, 18, 30])
    def test_a_size_that_breaks_the_discrete_symmetries_is_rejected(self, size: int) -> None:
        with pytest.raises(ValidationError, match="grid size"):
            FluidGrid(size)

    @pytest.mark.parametrize("length", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_bad_domain_length_is_rejected(self, length: float) -> None:
        with pytest.raises(ValidationError, match="domain length"):
            FluidGrid(16, length)


class TestDealiasing:
    @pytest.mark.parametrize("size", [16, 32, 48, 64, 96])
    def test_the_cutoff_leaves_no_room_for_an_alias(self, size: int) -> None:
        """Products reach twice the cutoff and fold back to `n - 2 M`, which must miss it."""
        grid = FluidGrid(size)
        assert size - 2 * grid.cutoff_mode > grid.cutoff_mode

    def test_the_mask_keeps_the_mean_mode_and_drops_the_highest(self) -> None:
        mask = FluidGrid(32).dealias_mask()
        assert mask[0, 0]
        assert not mask[16, 0]
        assert not mask[0, 16]

    def test_the_mask_is_symmetric_between_the_axes(self) -> None:
        """A quarter turn swaps the axes, so an asymmetric mask would break equivariance."""
        grid = FluidGrid(32)
        mask = grid.dealias_mask()
        half = grid.size // 2 + 1
        assert np.array_equal(mask[:half, :], mask[:half, :].T)


class TestTransforms:
    def test_a_field_survives_a_round_trip(self) -> None:
        grid = FluidGrid(16)
        rng = np.random.default_rng(0)
        field = rng.standard_normal((16, 16))
        assert grid.inverse(grid.forward(field)) == pytest.approx(field)

    def test_the_wavenumbers_differentiate_exactly(self) -> None:
        """The convention test. A wrong `2 pi` here would look like a wrong viscosity."""
        grid = FluidGrid(32, TWO_PI)
        along_x, along_y = grid.coordinates()
        field = np.sin(3.0 * along_x) * np.cos(2.0 * along_y)
        wavenumber_x, _ = grid.wavenumbers()

        derivative = grid.inverse(1j * wavenumber_x * grid.forward(field))

        assert derivative == pytest.approx(3.0 * np.cos(3.0 * along_x) * np.cos(2.0 * along_y))

    def test_the_wavenumbers_scale_with_the_domain(self) -> None:
        grid = FluidGrid(32, 1.0)
        along_x, _ = grid.coordinates()
        field = np.tile(np.sin(TWO_PI * along_x), (1, 32))
        wavenumber_x, _ = grid.wavenumbers()

        derivative = grid.inverse(1j * wavenumber_x * grid.forward(field))

        assert derivative == pytest.approx(np.tile(TWO_PI * np.cos(TWO_PI * along_x), (1, 32)))


class TestSpectralOperators:
    def test_the_mean_mode_has_no_inverse_laplacian(self) -> None:
        """Forgetting this leaves an undetermined constant in every streamfunction."""
        operators = SpectralOperators.build(FluidGrid(16))
        assert operators.inverse_squared_wavenumber[0, 0] == 0.0

    def test_the_inverse_undoes_the_laplacian_everywhere_else(self) -> None:
        operators = SpectralOperators.build(FluidGrid(16))
        product = operators.squared_wavenumber * operators.inverse_squared_wavenumber
        assert product[1:, 1:] == pytest.approx(1.0)


class TestStates:
    def test_a_field_becomes_a_state(self) -> None:
        grid = FluidGrid(16)
        along_x, along_y = grid.coordinates()
        field = np.sin(along_x) * np.cos(along_y)

        state = grid.make_state(field, time=0.5)

        assert state.time == 0.5
        assert grid.unpack(state) == pytest.approx(field)

    def test_a_mean_vorticity_is_rejected(self) -> None:
        """A periodic streamfunction cannot carry one, so it must not reach a solver."""
        grid = FluidGrid(16)
        with pytest.raises(ValidationError, match="mean"):
            grid.make_state(np.ones((16, 16)))

    def test_a_field_of_the_wrong_shape_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="shape"):
            FluidGrid(16).make_state(np.zeros((16, 8)))

    def test_a_state_from_another_grid_is_rejected(self) -> None:
        foreign = FluidGrid(32).make_state(np.zeros((32, 32)))
        with pytest.raises(ValidationError, match="shape"):
            FluidGrid(16).unpack(foreign)

    def test_a_state_with_the_wrong_field_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="missing fields"):
            FluidGrid(16).unpack(State(fields={"velocity": np.zeros((16, 16))}))
