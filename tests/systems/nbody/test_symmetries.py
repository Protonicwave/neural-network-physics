import numpy as np
import pytest
from conftest import RollOut, TwoBody

from nnphysics.core.protocols import Symmetry
from nnphysics.core.types import State
from nnphysics.systems.nbody import (
    GalileanBoost,
    NBodyDynamics,
    Rotation,
    Translation,
    VelocityVerlet,
    make_state,
    unpack,
)

EXACT = NBodyDynamics(gravitational_constant=1.0, softening=0.0)
DECLARED = [Translation((1.5, -0.75)), Rotation(0.7), GalileanBoost((0.3, -0.2))]


def max_difference(left: State, right: State) -> float:
    return max(
        float(np.max(np.abs(first - second)))
        for first, second in zip(unpack(left), unpack(right), strict=True)
    )


@pytest.mark.parametrize("symmetry", DECLARED, ids=lambda item: item.name)
class TestDeclaredSymmetries:
    def test_it_satisfies_the_protocol(self, symmetry: Symmetry) -> None:
        assert isinstance(symmetry, Symmetry)
        assert symmetry.name

    def test_the_inverse_undoes_the_transformation(
        self, symmetry: Symmetry, two_body: TwoBody
    ) -> None:
        state, _ = two_body(EXACT, eccentricity=0.3)
        assert max_difference(symmetry.apply_inverse(symmetry.apply(state)), state) < 1e-14

    def test_the_transformation_preserves_the_time(
        self, symmetry: Symmetry, two_body: TwoBody
    ) -> None:
        state, _ = two_body(EXACT)
        assert symmetry.apply(state).time == state.time

    def test_rolling_out_a_transformed_state_gives_the_transformed_rollout(
        self, symmetry: Symmetry, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT, eccentricity=0.3)
        predictor = VelocityVerlet(EXACT, period / 200)
        n_steps = 400

        direct = roll_out(predictor, state, n_steps)
        transformed = roll_out(predictor, symmetry.apply(state), n_steps)

        for step in range(n_steps + 1):
            assert max_difference(transformed[step], symmetry.apply(direct[step])) < 1e-10


class TestTransformations:
    def test_translation_moves_positions_and_leaves_velocities(self, two_body: TwoBody) -> None:
        state, _ = two_body(EXACT)
        moved = Translation((1.0, 2.0)).apply(state)

        assert np.allclose(unpack(moved)[0] - unpack(state)[0], np.array([1.0, 2.0]))
        assert np.array_equal(unpack(moved)[1], unpack(state)[1])

    def test_rotation_preserves_lengths(self, two_body: TwoBody) -> None:
        state, _ = two_body(EXACT, eccentricity=0.4)
        turned = Rotation(1.1).apply(state)

        assert np.allclose(
            np.linalg.norm(unpack(turned)[0], axis=-1), np.linalg.norm(unpack(state)[0], axis=-1)
        )

    def test_a_boost_displaces_positions_in_proportion_to_time(self) -> None:
        """The displacement is what makes a boost commute with the dynamics."""
        state = make_state(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2), time=3.0)
        boosted = GalileanBoost((1.0, -2.0)).apply(state)

        assert np.allclose(unpack(boosted)[0], np.array([3.0, -6.0]))
        assert np.allclose(unpack(boosted)[1], np.array([1.0, -2.0]))
