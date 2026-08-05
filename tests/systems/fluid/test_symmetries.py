from collections.abc import Callable

import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import Predictor, Symmetry
from nnphysics.core.types import FloatArray, Regime, State, Trajectory
from nnphysics.systems.fluid import (
    FluidDynamics,
    FluidGrid,
    IntegratingFactorRK4,
    QuarterTurn,
    Translation,
    initial_state,
)

SmoothVorticity = Callable[[FluidGrid], FloatArray]
RollOut = Callable[[Predictor, State, int], Trajectory]

GRID = FluidGrid(32)
REGIME = Regime("decaying_turbulence", {"reynolds": 200.0, "peak_wavenumber": 4.0})


def _symmetries() -> tuple[Symmetry, ...]:
    return (Translation(GRID, (8, -4)), QuarterTurn(GRID, 1))


class TestConstruction:
    def test_they_satisfy_the_protocol(self) -> None:
        assert all(isinstance(symmetry, Symmetry) for symmetry in _symmetries())

    def test_they_are_named(self) -> None:
        assert [symmetry.name for symmetry in _symmetries()] == ["translation", "rotation"]

    @pytest.mark.parametrize("turns", [0, 4, -8])
    def test_a_whole_number_of_full_turns_is_rejected(self, turns: int) -> None:
        """The identity is not a symmetry test, and silently accepting it would pass one."""
        with pytest.raises(ValidationError, match="identity"):
            QuarterTurn(GRID, turns)


class TestTransformations:
    def test_each_one_undoes_itself_exactly(self, smooth_vorticity: SmoothVorticity) -> None:
        """Both are permutations of the grid, so this is exact rather than accurate."""
        state = GRID.make_state(smooth_vorticity(GRID))

        for symmetry in _symmetries():
            restored = symmetry.apply_inverse(symmetry.apply(state))
            assert np.array_equal(GRID.unpack(restored), GRID.unpack(state))

    def test_a_translation_moves_the_field_by_whole_cells(self) -> None:
        along_x, _ = GRID.coordinates()
        field = np.tile(np.sin(along_x), (1, GRID.size))
        state = GRID.make_state(field)

        shifted = Translation(GRID, (4, 0)).apply(state)

        expected = np.tile(np.sin(along_x - 4 * GRID.spacing), (1, GRID.size))
        assert GRID.unpack(shifted) == pytest.approx(expected)

    def test_a_quarter_turn_matches_the_rotated_analytic_field(self) -> None:
        """A rotation sends the sample at `(x, y)` to `(-y, x)`, so the field reads `f(y, -x)`."""
        along_x, along_y = GRID.coordinates()
        field = np.sin(along_x) * np.cos(2.0 * along_y) + 0.3 * np.sin(3.0 * along_x - along_y)
        rotated = np.sin(along_y) * np.cos(2.0 * along_x) + 0.3 * np.sin(3.0 * along_y + along_x)

        turned = QuarterTurn(GRID, 1).apply(GRID.make_state(field - field.mean()))

        assert GRID.unpack(turned) == pytest.approx(rotated - rotated.mean())

    def test_four_quarter_turns_return_the_field(self, smooth_vorticity: SmoothVorticity) -> None:
        state = GRID.make_state(smooth_vorticity(GRID))
        turn = QuarterTurn(GRID, 1)

        for _ in range(4):
            state = turn.apply(state)

        assert np.array_equal(GRID.unpack(state), smooth_vorticity(GRID))

    def test_time_is_carried_through(self, smooth_vorticity: SmoothVorticity) -> None:
        state = GRID.make_state(smooth_vorticity(GRID), time=2.5)

        for symmetry in _symmetries():
            assert symmetry.apply(state).time == 2.5


class TestEquivariance:
    """Transforming and then stepping must equal stepping and then transforming.

    This is the property the evaluation harness will test a surrogate against, so the
    reference solver has to have it to round off, not merely to plotting accuracy.
    """

    @pytest.mark.parametrize("index", [0, 1])
    def test_rolling_out_a_transformed_state_gives_the_transformed_rollout(
        self, index: int, roll_out: RollOut
    ) -> None:
        dynamics = FluidDynamics(GRID, 0.005)
        predictor = IntegratingFactorRK4(dynamics, 0.005)
        state = initial_state(REGIME, np.random.default_rng(1), dynamics)
        symmetry = _symmetries()[index]
        n_steps = 200

        direct = roll_out(predictor, state, n_steps)
        transformed = roll_out(predictor, symmetry.apply(state), n_steps)

        for step in range(n_steps + 1):
            assert GRID.unpack(transformed[step]) == pytest.approx(
                GRID.unpack(symmetry.apply(direct[step])), abs=1e-11
            )

    def test_the_invariants_do_not_notice_a_symmetry(
        self, smooth_vorticity: SmoothVorticity
    ) -> None:
        dynamics = FluidDynamics(GRID, 0.005)
        state = GRID.make_state(smooth_vorticity(GRID))

        for symmetry in _symmetries():
            moved = GRID.unpack(symmetry.apply(state))
            assert dynamics.energy(moved) == pytest.approx(dynamics.energy(GRID.unpack(state)))
            assert dynamics.enstrophy(moved) == pytest.approx(
                dynamics.enstrophy(GRID.unpack(state))
            )
