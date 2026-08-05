"""Transformations the vorticity equation commutes with on a periodic square.

Both are chosen to map the grid onto itself exactly. A shift by a whole number of cells
is a permutation of the grid points, and so is a quarter turn, so equivariance here is
exact up to round off rather than accurate to the order of an interpolation. A test that
cannot tell a broken predictor from an interpolation error is not worth running.

Vorticity is a pseudoscalar in two dimensions: it transforms as a scalar under rotations
and changes sign under reflections. Only rotations are declared, so the field is simply
carried around, never negated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray, State
    from nnphysics.systems.fluid.grid import FluidGrid

__all__ = ["QuarterTurn", "Translation"]

_QUARTER_TURNS = 4


@dataclass(frozen=True, slots=True)
class Translation:
    """Shift of the field by a whole number of cells along each axis.

    Attributes:
        grid: The grid the field lives on.
        shift: Cells shifted along `x` and along `y`.
    """

    grid: FluidGrid
    shift: tuple[int, int]

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
        return self._shift_by(state, self.shift)

    def apply_inverse(self, state: State) -> State:
        """Undo `apply`.

        Args:
            state: The state to transform.

        Returns:
            The state shifted back.
        """
        return self._shift_by(state, (-self.shift[0], -self.shift[1]))

    def _shift_by(self, state: State, shift: tuple[int, int]) -> State:
        vorticity = self.grid.unpack(state)
        rolled: FloatArray = np.roll(vorticity, shift, axis=(0, 1))
        return self.grid.make_state(rolled, time=state.time)


@dataclass(frozen=True, slots=True)
class QuarterTurn:
    """Rotation by a multiple of ninety degrees about the origin.

    Attributes:
        grid: The grid the field lives on.
        turns: Number of anticlockwise quarter turns.
    """

    grid: FluidGrid
    turns: int = 1

    def __post_init__(self) -> None:
        if self.turns % _QUARTER_TURNS == 0:
            raise ValidationError(
                f"a quarter turn count that is a multiple of four is the identity, got {self.turns}"
            )

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
        return self._rotate_by(state, self.turns)

    def apply_inverse(self, state: State) -> State:
        """Undo `apply`.

        Args:
            state: The state to transform.

        Returns:
            The state rotated back.
        """
        return self._rotate_by(state, -self.turns)

    def _rotate_by(self, state: State, turns: int) -> State:
        vorticity = self.grid.unpack(state)
        for _ in range(turns % _QUARTER_TURNS):
            vorticity = _rotate_once(vorticity)
        return self.grid.make_state(vorticity, time=state.time)


def _rotate_once(field: FloatArray) -> FloatArray:
    """One anticlockwise quarter turn about the grid origin.

    A rotation of the domain sends the sample at `(x, y)` to `(-y, x)`, so the rotated
    field reads `field[j, -i]`. NumPy's own quarter turn reads `field[j, -1 - i]`,
    because it rotates about the centre of the array rather than about the sample at the
    origin, so it is corrected by one cell.
    """
    rotated: FloatArray = np.roll(np.rot90(field), 1, axis=0)
    return rotated
