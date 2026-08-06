"""Expressing the same flow on a finer grid, and reading it back.

Both directions are done in Fourier space, which for a periodic field is not one
interpolation scheme among several: a band limited field has exactly one continuous
representative, and padding its spectrum with zeros samples that representative on the
finer grid. Nothing is invented and nothing is smoothed. Reading back truncates the
spectrum to the coarse band, which is the projection the coarse grid can represent.

Two consequences are worth stating, because the resolution metric is only meaningful
given them.

**Coarsening a refined field returns it exactly.** Padding with zeros and then discarding
those zeros is the identity on the retained modes, to round off. So a disagreement the
metric measures is the predictor's, never the pair of transformations'.

**Refining a coarsened field does not.** Modes above the coarse band are gone and no
transform brings them back. That asymmetry is real and is the reason the refined path is
compared against coarse ground truth rather than the other way round.

The Nyquist row and column are dropped rather than split between the positive and
negative wavenumbers they stand for. On this system they are already zero: the solver
dealiases at two thirds of the grid, well below Nyquist, so every state the data holds
is band limited far inside it. A field that did carry them would be refined wrongly by a
half, so the state is checked rather than assumed.

Neither direction validates the physics of what it is handed, beyond the shape. This is a
change of representation and nothing else, and a predictor whose output has drifted to
carry a mean vorticity must be measured for that rather than corrected on the way through:
subtracting the mean here would have quietly improved the refined path relative to the
native one, and the number the resolution metric reports would have been partly this
file's doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import State
from nnphysics.systems.fluid.grid import VORTICITY_FIELD, FluidGrid

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray
    from nnphysics.systems.fluid.grid import ComplexArray

__all__ = ["SpectralRefinement"]

_MINIMUM_FACTOR = 2

_NYQUIST_RTOL = 1e-6
"""How much energy the Nyquist band may hold, relative to the field, before refining it
would be a guess rather than a resampling. Loose enough to pass round off from a field
that has been through single precision, which a surrogate's output has, and tight enough
that any real content at that wavenumber is caught: a field that genuinely carries the
Nyquist mode carries a fraction of order one there, not a millionth."""


@dataclass(frozen=True, slots=True)
class SpectralRefinement:
    """Zero padded spectral resampling between a grid and an integer multiple of it.

    Attributes:
        grid: The grid the data is stored on.
        factor: How many times finer the refined grid is, along each axis.
    """

    grid: FluidGrid
    factor: int = 2

    def __post_init__(self) -> None:
        if self.factor < _MINIMUM_FACTOR:
            raise ValidationError(
                f"a refinement must make the grid finer, got a factor of {self.factor}"
            )

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return f"grid_x{self.factor}"

    @property
    def fine_grid(self) -> FluidGrid:
        """The grid a refined state lives on, over the same domain."""
        return FluidGrid(size=self.grid.size * self.factor, length=self.grid.length)

    def refine(self, state: State) -> State:
        """Sample the same flow on the finer grid.

        Args:
            state: A state on the stored grid.

        Returns:
            The state on the finer grid, at the same time.

        Raises:
            ValidationError: If the state is not on the stored grid, or holds energy in
                the Nyquist band, which cannot be resampled without choosing how to split
                it.
        """
        coarse = self.grid.unpack(state)
        spectrum = self.grid.forward(coarse)
        _check_nyquist(spectrum, coarse)
        fine = self.fine_grid
        padded: ComplexArray = np.zeros((fine.size, fine.size // 2 + 1), dtype=np.complex128)
        half = self.grid.size // 2
        # The inverse transform divides by the number of points, so the padded spectrum
        # is scaled by the ratio of the two to leave the field's values unchanged.
        scale = (fine.size / self.grid.size) ** 2
        padded[:half, :half] = spectrum[:half, :half] * scale
        padded[-half + 1 :, :half] = spectrum[-half + 1 :, :half] * scale
        return State(fields={VORTICITY_FIELD: fine.inverse(padded)}, time=state.time)

    def coarsen(self, state: State) -> State:
        """Project a state on the finer grid back onto the stored one.

        Args:
            state: A state on the finer grid.

        Returns:
            The state on the stored grid, at the same time.

        Raises:
            ValidationError: If the state is not on the finer grid.
        """
        fine = self.fine_grid
        spectrum = fine.forward(fine.unpack(state))
        half = self.grid.size // 2
        truncated: ComplexArray = np.zeros(
            (self.grid.size, self.grid.size // 2 + 1), dtype=np.complex128
        )
        scale = (self.grid.size / fine.size) ** 2
        truncated[:half, :half] = spectrum[:half, :half] * scale
        truncated[-half + 1 :, :half] = spectrum[-half + 1 :, :half] * scale
        return State(fields={VORTICITY_FIELD: self.grid.inverse(truncated)}, time=state.time)


def _check_nyquist(spectrum: ComplexArray, field: FloatArray) -> None:
    """Reject a field whose Nyquist band carries energy this cannot resample."""
    half = field.shape[0] // 2
    nyquist = float(np.abs(spectrum[half, :]).max()) + float(np.abs(spectrum[:, half]).max())
    total = max(float(np.abs(spectrum).max()), 1.0)
    if nyquist > _NYQUIST_RTOL * total:
        raise ValidationError(
            f"the field carries {nyquist / total:.3e} of its amplitude at the Nyquist "
            f"wavenumber, which cannot be refined without choosing how to split it "
            f"between the wavenumbers it stands for"
        )
