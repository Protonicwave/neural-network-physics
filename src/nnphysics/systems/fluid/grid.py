"""The doubly periodic grid: the state it carries and the Fourier conventions it fixes.

Real transforms are used throughout, so the spectrum of an `n` by `n` field has shape
`(n, n // 2 + 1)` and its second axis holds only non negative wavenumbers. The forward
transform is unnormalised and the inverse divides by `n * n`, which is what NumPy does
and is the one convention in a spectral code that must not be guessed at: an error here
looks like a wrong viscosity rather than like a bug.

Fields are indexed `field[i, j]`, with `i` running over `x` and `j` over `y`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import FieldSpec, FloatArray, State, StateSpec
from nnphysics.core.units import DIMENSIONLESS, TIME

if TYPE_CHECKING:
    from numpy.typing import NDArray

    BoolArray = NDArray[np.bool_]
    ComplexArray = NDArray[np.complex128]
else:
    BoolArray = np.ndarray
    ComplexArray = np.ndarray

__all__ = [
    "VORTICITY",
    "VORTICITY_FIELD",
    "BoolArray",
    "ComplexArray",
    "FluidGrid",
    "SpectralOperators",
]

VORTICITY_FIELD = "vorticity"

VORTICITY = DIMENSIONLESS / TIME

TWO_PI = 2.0 * np.pi

_MINIMUM_SIZE = 8
_SIZE_DIVISOR = 4
"""A quarter turn and the shear layer both need the half and quarter grid to be exact."""

_MEAN_RTOL = 1e-9
"""A periodic streamfunction cannot represent a mean vorticity, so there must not be one."""


@dataclass(frozen=True, slots=True, eq=False)
class SpectralOperators:
    """The wavenumber arrays a solver on one grid needs, built once.

    Equality is identity: these are caches, and comparing them elementwise would be both
    meaningless and a source of ambiguous truth values.

    Attributes:
        wavenumber_x: Angular wavenumbers along `x`, shape `(n, 1)`.
        wavenumber_y: Angular wavenumbers along `y`, shape `(1, n // 2 + 1)`.
        squared_wavenumber: `kx^2 + ky^2`, shape `(n, n // 2 + 1)`.
        inverse_squared_wavenumber: Its reciprocal, with the mean mode set to zero.
        retained: Two thirds rule mask, true for the modes a product may keep.
    """

    wavenumber_x: FloatArray
    wavenumber_y: FloatArray
    squared_wavenumber: FloatArray
    inverse_squared_wavenumber: FloatArray
    retained: BoolArray

    @classmethod
    def build(cls, grid: FluidGrid) -> SpectralOperators:
        """Build the operators for a grid.

        Args:
            grid: The grid to build for.

        Returns:
            The operators.
        """
        wavenumber_x, wavenumber_y = grid.wavenumbers()
        squared: FloatArray = wavenumber_x**2 + wavenumber_y**2
        # The mean mode has no inverse. It is divided by one and then discarded, which
        # is the gauge choice that leaves the streamfunction with zero mean.
        divisor = squared.copy()
        divisor[0, 0] = 1.0
        inverse: FloatArray = 1.0 / divisor
        inverse[0, 0] = 0.0
        return cls(wavenumber_x, wavenumber_y, squared, inverse, grid.dealias_mask())


@dataclass(frozen=True, slots=True)
class FluidGrid:
    """A square doubly periodic domain sampled uniformly.

    Attributes:
        size: Number of points along each axis. Even, at least eight, and a multiple of
            four so that the discrete symmetries are exact.
        length: Side length of the domain.
    """

    size: int = 64
    length: float = TWO_PI

    def __post_init__(self) -> None:
        if self.size < _MINIMUM_SIZE:
            raise ValidationError(f"grid size must be at least {_MINIMUM_SIZE}, got {self.size}")
        if self.size % _SIZE_DIVISOR != 0:
            raise ValidationError(f"grid size must be a multiple of four, got {self.size}")
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValidationError(f"domain length must be positive and finite, got {self.length}")

    @property
    def spacing(self) -> float:
        """Distance between neighbouring points."""
        return self.length / self.size

    @property
    def cutoff_mode(self) -> int:
        """Highest mode index a nonlinear term may keep under the two thirds rule.

        A product of two retained modes reaches `2 M`, and anything past the Nyquist mode
        folds back to `n - 2 M`, so aliases miss the retained band only while `M < n / 3`.
        Strictly less: on a grid whose size divides by three, keeping the mode at exactly
        `n / 3` lets every alias land on it, and the resulting drift in the inviscid
        invariants does not shrink when the step size does.
        """
        return (self.size - 1) // 3

    @property
    def state_spec(self) -> StateSpec:
        """The single field a fluid state carries."""
        return StateSpec(
            fields=(
                FieldSpec(
                    VORTICITY_FIELD,
                    (self.size, self.size),
                    VORTICITY,
                    "out of plane vorticity at each grid point",
                ),
            )
        )

    def coordinates(self) -> tuple[FloatArray, FloatArray]:
        """Grid coordinates, shaped to broadcast against a field.

        Returns:
            `x` of shape `(n, 1)` and `y` of shape `(1, n)`.
        """
        axis: FloatArray = np.arange(self.size, dtype=np.float64) * self.spacing
        return axis[:, np.newaxis], axis[np.newaxis, :]

    def wavenumbers(self) -> tuple[FloatArray, FloatArray]:
        """Angular wavenumbers, shaped to broadcast against a spectrum.

        Returns:
            `kx` of shape `(n, 1)` and `ky` of shape `(1, n // 2 + 1)`.
        """
        along_x: FloatArray = TWO_PI * np.fft.fftfreq(self.size, d=self.spacing)
        along_y: FloatArray = TWO_PI * np.fft.rfftfreq(self.size, d=self.spacing)
        return along_x[:, np.newaxis], along_y[np.newaxis, :]

    def dealias_mask(self) -> BoolArray:
        """Which modes survive the two thirds rule.

        Returns:
            A boolean array over the half spectrum, true where the mode is retained.
        """
        # Built from integer mode indices rather than from wavenumbers, so that the
        # comparison at the cutoff is exact rather than a matter of rounding.
        along_x = np.abs(np.fft.fftfreq(self.size, d=1.0 / self.size)).astype(np.int64)
        along_y = np.arange(self.size // 2 + 1, dtype=np.int64)
        mask: BoolArray = (along_x[:, np.newaxis] <= self.cutoff_mode) & (
            along_y[np.newaxis, :] <= self.cutoff_mode
        )
        return mask

    def forward(self, field: FloatArray) -> ComplexArray:
        """Transform a real field to its half spectrum.

        Args:
            field: A real field, shape `(n, n)`.

        Returns:
            The spectrum, shape `(n, n // 2 + 1)`.
        """
        spectrum: ComplexArray = np.fft.rfft2(field)
        return spectrum

    def inverse(self, spectrum: ComplexArray) -> FloatArray:
        """Transform a half spectrum back to a real field.

        Args:
            spectrum: A half spectrum, shape `(n, n // 2 + 1)`.

        Returns:
            The field, shape `(n, n)`.
        """
        field: FloatArray = np.fft.irfft2(spectrum, s=(self.size, self.size))
        return field

    def make_state(self, vorticity: FloatArray, time: float = 0.0) -> State:
        """Build a validated fluid state.

        Args:
            vorticity: Vorticity at each grid point, shape `(n, n)`, with zero mean.
            time: Simulation time of the state.

        Returns:
            The state.

        Raises:
            ValidationError: If the shape is wrong or the field carries a mean vorticity.
        """
        field = np.asarray(vorticity, dtype=np.float64)
        state = State(fields={VORTICITY_FIELD: field}, time=time)
        self.state_spec.validate(state)
        mean = float(np.mean(field))
        scale = max(float(np.sqrt(np.mean(field**2))), 1.0)
        if abs(mean) > _MEAN_RTOL * scale:
            raise ValidationError(
                f"vorticity has mean {mean:.3e}, which a periodic streamfunction cannot "
                f"represent: subtract it before building a state"
            )
        return state

    def unpack(self, state: State) -> FloatArray:
        """Read the vorticity out of a state.

        Args:
            state: A state carrying the fluid field.

        Returns:
            The vorticity, shape `(n, n)`.

        Raises:
            ValidationError: If the state does not match this grid's specification.
        """
        self.state_spec.validate(state)
        return state.fields[VORTICITY_FIELD]
