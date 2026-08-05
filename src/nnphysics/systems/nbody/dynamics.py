"""Gravitational dynamics: softened pairwise accelerations and the energies.

Plummer softening replaces the Newtonian potential with `-G m_i m_j / sqrt(r^2 + e^2)`.
The acceleration used here is the exact gradient of that potential, so the softened
system is still Hamiltonian and its energy is a genuine invariant rather than an
approximation to one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray

__all__ = ["NBodyDynamics", "kinetic_energy"]


@dataclass(frozen=True, slots=True)
class NBodyDynamics:
    """The force law, in natural units.

    Attributes:
        gravitational_constant: Strength of the interaction, `G`.
        softening: Plummer softening length. Zero gives the exact Newtonian law and is
            safe only when no two bodies come close.
    """

    gravitational_constant: float = 1.0
    softening: float = 0.0

    def __post_init__(self) -> None:
        if self.gravitational_constant <= 0.0:
            raise ValidationError(
                f"gravitational constant must be positive, got {self.gravitational_constant}"
            )
        if self.softening < 0.0:
            raise ValidationError(f"softening length must not be negative, got {self.softening}")

    def accelerations(self, position: FloatArray, mass: FloatArray) -> FloatArray:
        """Acceleration of every body under the softened gravity of all the others.

        Args:
            position: Positions, shape `(n_bodies, 2)`.
            mass: Masses, shape `(n_bodies,)`.

        Returns:
            Accelerations, shape `(n_bodies, 2)`.
        """
        separation = position[np.newaxis, :, :] - position[:, np.newaxis, :]
        # The pair matrix is symmetric, so it is built once and contracted twice: once
        # against the masses to get accelerations, once against the diagonal mask.
        inverse_cube = self._inverse_cube_distance(separation)
        weights: FloatArray = inverse_cube * mass[np.newaxis, :]
        acceleration: FloatArray = self.gravitational_constant * np.einsum(
            "ij,ijd->id", weights, separation
        )
        return acceleration

    def potential_energy(self, position: FloatArray, mass: FloatArray) -> float:
        """Total gravitational potential energy of a configuration.

        Args:
            position: Positions, shape `(n_bodies, 2)`.
            mass: Masses, shape `(n_bodies,)`.

        Returns:
            The potential energy, negative for any bound or unbound configuration.
        """
        separation = position[np.newaxis, :, :] - position[:, np.newaxis, :]
        inverse_distance = self._inverse_distance(separation)
        pair_mass: FloatArray = mass[:, np.newaxis] * mass[np.newaxis, :]
        # Each unordered pair appears twice in the matrix.
        return float(-0.5 * self.gravitational_constant * np.sum(pair_mass * inverse_distance))

    def _inverse_distance(self, separation: FloatArray) -> FloatArray:
        """`1 / sqrt(r^2 + e^2)` for every pair, with self pairs zeroed."""
        squared = self._softened_square_distance(separation)
        inverse: FloatArray = 1.0 / np.sqrt(squared)
        np.fill_diagonal(inverse, 0.0)
        return inverse

    def _inverse_cube_distance(self, separation: FloatArray) -> FloatArray:
        """`(r^2 + e^2)^(-3/2)` for every pair, with self pairs zeroed."""
        squared = self._softened_square_distance(separation)
        inverse: FloatArray = squared**-1.5
        np.fill_diagonal(inverse, 0.0)
        return inverse

    def _softened_square_distance(self, separation: FloatArray) -> FloatArray:
        """`r^2 + e^2` for every pair, with the diagonal set to one.

        The diagonal is unsoftened when the softening length is zero, so it is replaced
        before any negative power is taken. Self terms are dropped afterwards.
        """
        squared: FloatArray = np.einsum("ijd,ijd->ij", separation, separation)
        squared = squared + self.softening**2
        np.fill_diagonal(squared, 1.0)
        return squared


def kinetic_energy(velocity: FloatArray, mass: FloatArray) -> float:
    """Total kinetic energy.

    Args:
        velocity: Velocities, shape `(n_bodies, 2)`.
        mass: Masses, shape `(n_bodies,)`.

    Returns:
        The kinetic energy.
    """
    return float(0.5 * np.sum(mass * np.einsum("id,id->i", velocity, velocity)))
