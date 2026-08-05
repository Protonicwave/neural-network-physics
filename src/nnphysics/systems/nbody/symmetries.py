"""Transformations the gravitational dynamics commute with.

The Galilean boost is defined with the state's own time in it, `x -> x + u t`, not as a
shift of velocity alone. Only the full transformation commutes with the dynamics, and a
velocity only version would fail the equivariance test for the right reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.systems.nbody.state import make_state, unpack

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray, State

__all__ = ["GalileanBoost", "Rotation", "Translation"]


@dataclass(frozen=True, slots=True)
class Translation:
    """Rigid shift of every position.

    Attributes:
        shift: Displacement applied to each body, length two.
    """

    shift: tuple[float, float]

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "translation"

    def apply(self, state: State) -> State:
        """Shift a state.

        Args:
            state: The state to transform.

        Returns:
            The shifted state.
        """
        return self._shift_by(state, 1.0)

    def apply_inverse(self, state: State) -> State:
        """Undo `apply`.

        Args:
            state: The state to transform.

        Returns:
            The state shifted back.
        """
        return self._shift_by(state, -1.0)

    def _shift_by(self, state: State, sign: float) -> State:
        position, velocity, mass = unpack(state)
        offset = sign * np.asarray(self.shift, dtype=np.float64)
        return make_state(position + offset, velocity, mass, time=state.time)


@dataclass(frozen=True, slots=True)
class Rotation:
    """Rotation of positions and velocities about the origin.

    Attributes:
        angle: Rotation angle in radians, anticlockwise.
    """

    angle: float

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "rotation"

    def apply(self, state: State) -> State:
        """Rotate a state.

        Args:
            state: The state to transform.

        Returns:
            The rotated state.
        """
        return self._rotate_by(state, self.angle)

    def apply_inverse(self, state: State) -> State:
        """Undo `apply`.

        Args:
            state: The state to transform.

        Returns:
            The state rotated back.
        """
        return self._rotate_by(state, -self.angle)

    def _rotate_by(self, state: State, angle: float) -> State:
        position, velocity, mass = unpack(state)
        matrix = _rotation_matrix(angle)
        return make_state(position @ matrix.T, velocity @ matrix.T, mass, time=state.time)


@dataclass(frozen=True, slots=True)
class GalileanBoost:
    """Change to a frame moving at constant velocity.

    Attributes:
        velocity: Velocity of the new frame relative to the old one, length two.
    """

    velocity: tuple[float, float]

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "galilean_boost"

    def apply(self, state: State) -> State:
        """Boost a state.

        Args:
            state: The state to transform.

        Returns:
            The boosted state.
        """
        return self._boost_by(state, 1.0)

    def apply_inverse(self, state: State) -> State:
        """Undo `apply`.

        Args:
            state: The state to transform.

        Returns:
            The state boosted back.
        """
        return self._boost_by(state, -1.0)

    def _boost_by(self, state: State, sign: float) -> State:
        position, velocity, mass = unpack(state)
        boost = sign * np.asarray(self.velocity, dtype=np.float64)
        return make_state(position + boost * state.time, velocity + boost, mass, time=state.time)


def _rotation_matrix(angle: float) -> FloatArray:
    """Anticlockwise rotation matrix for a plane angle in radians."""
    if not np.isfinite(angle):
        raise ValidationError(f"rotation angle must be finite, got {angle}")
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix: FloatArray = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return matrix
