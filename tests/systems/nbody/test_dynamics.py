import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import FloatArray
from nnphysics.systems.nbody import NBodyDynamics, kinetic_energy

EXACT = NBodyDynamics(gravitational_constant=1.0, softening=0.0)
SOFTENED = NBodyDynamics(gravitational_constant=1.0, softening=0.1)


def naive_accelerations(
    position: FloatArray, mass: FloatArray, dynamics: NBodyDynamics
) -> FloatArray:
    """The same law written as an unvectorised double loop, for comparison only."""
    acceleration = np.zeros_like(position)
    for i in range(position.shape[0]):
        for j in range(position.shape[0]):
            if i == j:
                continue
            offset = position[j] - position[i]
            distance = np.sqrt(float(offset @ offset) + dynamics.softening**2)
            acceleration[i] += dynamics.gravitational_constant * mass[j] * offset / distance**3
    return acceleration


def random_configuration(seed: int, n_bodies: int = 6) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_bodies, 2)), rng.uniform(0.5, 2.0, size=n_bodies)


class TestAccelerations:
    def test_the_vectorised_law_matches_a_naive_double_loop(self) -> None:
        position, mass = random_configuration(seed=1)
        expected = naive_accelerations(position, mass, SOFTENED)
        assert np.allclose(SOFTENED.accelerations(position, mass), expected, rtol=1e-12)

    def test_a_two_body_pair_attracts_at_the_newtonian_strength(self) -> None:
        position = np.array([[0.0, 0.0], [2.0, 0.0]])
        mass = np.array([3.0, 1.0])
        acceleration = EXACT.accelerations(position, mass)

        assert acceleration[0] == pytest.approx([1.0 / 4.0, 0.0])
        assert acceleration[1] == pytest.approx([-3.0 / 4.0, 0.0])

    def test_internal_forces_cancel(self) -> None:
        position, mass = random_configuration(seed=2)
        force = mass[:, np.newaxis] * SOFTENED.accelerations(position, mass)
        # Measured against the size of the forces being cancelled, not against one.
        residual = np.linalg.norm(np.sum(force, axis=0)) / float(np.sum(np.abs(force)))
        assert residual < 1e-14

    def test_softening_weakens_the_force_at_short_range(self) -> None:
        position = np.array([[0.0, 0.0], [0.05, 0.0]])
        mass = np.array([1.0, 1.0])
        assert abs(SOFTENED.accelerations(position, mass)[0, 0]) < abs(
            EXACT.accelerations(position, mass)[0, 0]
        )

    def test_softening_survives_a_coincident_pair(self) -> None:
        position = np.zeros((2, 2))
        mass = np.array([1.0, 1.0])
        assert np.all(np.isfinite(SOFTENED.accelerations(position, mass)))

    def test_a_lone_body_does_not_accelerate_itself(self) -> None:
        acceleration = EXACT.accelerations(np.array([[1.0, -2.0]]), np.array([1.0]))
        assert np.array_equal(acceleration, np.zeros((1, 2)))


class TestEnergy:
    def test_potential_energy_of_a_pair_is_the_newtonian_value(self) -> None:
        position = np.array([[0.0, 0.0], [2.0, 0.0]])
        mass = np.array([3.0, 1.0])
        assert EXACT.potential_energy(position, mass) == pytest.approx(-1.5)

    def test_kinetic_energy_sums_over_bodies(self) -> None:
        velocity = np.array([[3.0, 4.0], [0.0, 1.0]])
        mass = np.array([2.0, 6.0])
        assert kinetic_energy(velocity, mass) == pytest.approx(0.5 * (2.0 * 25.0 + 6.0 * 1.0))

    def test_the_acceleration_is_the_gradient_of_the_potential(self) -> None:
        """A softened force that does not match its potential would leak energy silently."""
        position, mass = random_configuration(seed=3, n_bodies=4)
        acceleration = SOFTENED.accelerations(position, mass)
        delta = 1e-6

        for body in range(position.shape[0]):
            for axis in range(2):
                shifted = position.copy()
                shifted[body, axis] += delta
                forward = SOFTENED.potential_energy(shifted, mass)
                shifted[body, axis] -= 2.0 * delta
                backward = SOFTENED.potential_energy(shifted, mass)
                gradient = (forward - backward) / (2.0 * delta)
                assert -gradient / mass[body] == pytest.approx(acceleration[body, axis], abs=1e-7)


class TestValidation:
    def test_a_non_positive_gravitational_constant_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="gravitational constant must be positive"):
            NBodyDynamics(gravitational_constant=0.0)

    def test_a_negative_softening_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="softening length"):
            NBodyDynamics(softening=-1.0)
