"""Seeded initial conditions for the named regimes.

Every generator takes a generator object and draws from it in a fixed order, so a seed
determines a trajectory exactly. Nothing here touches a global random function.

Every field is finished the same way: truncated to the band the two thirds rule retains,
stripped of any mean vorticity, and scaled to unit root mean square speed. The scaling is
what gives the Reynolds number a meaning. With a characteristic speed of one and a
characteristic length fixed by the domain, a regime's Reynolds number determines the
viscosity, and two regimes at the same Reynolds number are comparable flows rather than
merely flows with the same label.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import NumericalError, UnknownNameError, ValidationError
from nnphysics.core.types import Regime, State
from nnphysics.systems.base import check_parameter_names, float_parameter, int_parameter
from nnphysics.systems.fluid.grid import TWO_PI

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray
    from nnphysics.systems.fluid.dynamics import FluidDynamics
    from nnphysics.systems.fluid.grid import FluidGrid

__all__ = [
    "FLUID_REGIMES",
    "characteristic_length",
    "initial_state",
    "taylor_green_decay_rate",
    "taylor_green_vorticity",
    "viscosity",
]

TAYLOR_GREEN = Regime(
    name="taylor_green",
    parameters={"reynolds": 100.0, "wavenumber": 2.0},
)
"""A single Laplacian eigenmode. It has an analytic solution, so it anchors the solver."""

DECAYING_TURBULENCE = Regime(
    name="decaying_turbulence",
    parameters={"reynolds": 500.0, "peak_wavenumber": 6.0},
)
"""Random phases on a prescribed energy spectrum. The broadest test of the solver."""

SHEAR_LAYER = Regime(
    name="shear_layer",
    parameters={"reynolds": 200.0, "layer_width": 0.05, "perturbation": 0.05},
)
"""Two opposed layers that roll up into vortices. Sharp gradients, so it is the hardest
of the three on a fixed grid."""

FLUID_REGIMES = (TAYLOR_GREEN, DECAYING_TURBULENCE, SHEAR_LAYER)

_TAYLOR_GREEN_PARAMETERS = ("reynolds", "wavenumber")
_TURBULENCE_PARAMETERS = ("reynolds", "peak_wavenumber")
_SHEAR_PARAMETERS = ("reynolds", "layer_width", "perturbation")

_MAXIMUM_LAYER_WIDTH = 0.2
"""Wider than this and the two layers overlap rather than being distinct."""


def characteristic_length(grid: FluidGrid) -> float:
    """Length the Reynolds number is defined against.

    The domain holds `2 pi` of it, so the default domain has a characteristic length of
    one and a Reynolds number that reads as the inverse viscosity.

    Args:
        grid: The domain.

    Returns:
        The characteristic length.
    """
    return grid.length / TWO_PI


def viscosity(regime: Regime, grid: FluidGrid) -> float:
    """Kinematic viscosity implied by a regime's Reynolds number.

    Initial conditions are scaled to unit root mean square speed, so the characteristic
    speed is one and `nu = L / Re`. An infinite Reynolds number is the inviscid limit and
    is spelled that way rather than through a separate flag.

    Args:
        regime: The regime to read.
        grid: The domain, which fixes the characteristic length.

    Returns:
        The viscosity, zero when the Reynolds number is infinite.

    Raises:
        ValidationError: If the Reynolds number is missing, not positive, or not a number.
    """
    context = f"fluid regime {regime.name!r}"
    reynolds = float_parameter(regime.parameters, "reynolds", context=context)
    if np.isnan(reynolds) or reynolds <= 0.0:
        raise ValidationError(f"{context} Reynolds number must be positive, got {reynolds}")
    return characteristic_length(grid) / reynolds


def taylor_green_vorticity(
    grid: FluidGrid, wavenumber: int, phase: tuple[float, float] = (0.0, 0.0)
) -> FloatArray:
    """The Taylor-Green vorticity field, of unit amplitude.

    This is an eigenfunction of the Laplacian, so the streamfunction is proportional to
    the vorticity and the advection term vanishes identically. What is left is pure
    diffusion, and the flow decays in place at a rate viscosity alone decides.

    Args:
        grid: The domain.
        wavenumber: Mode index along each axis, at least one.
        phase: Position of the pattern along each axis, in radians of the mode.

    Returns:
        The vorticity field, shape `(n, n)`.
    """
    along_x, along_y = grid.coordinates()
    scale = TWO_PI * wavenumber / grid.length
    field: FloatArray = np.cos(scale * along_x - phase[0]) * np.cos(scale * along_y - phase[1])
    return field


def taylor_green_decay_rate(grid: FluidGrid, wavenumber: int, kinematic_viscosity: float) -> float:
    """Exponential rate at which a Taylor-Green mode decays.

    The mode has squared wavenumber `2 k^2` because it is a product along both axes, so
    the vorticity decays as `exp(-2 nu k^2 t)`.

    Args:
        grid: The domain.
        wavenumber: Mode index along each axis.
        kinematic_viscosity: The viscosity.

    Returns:
        The decay rate, positive.
    """
    scale = TWO_PI * wavenumber / grid.length
    return 2.0 * kinematic_viscosity * scale**2


def initial_state(regime: Regime, rng: np.random.Generator, dynamics: FluidDynamics) -> State:
    """Draw one initial condition from a named regime.

    Args:
        regime: The regime to draw from. Its name selects the generator and its
            parameters configure it.
        rng: Generator every random draw comes from.
        dynamics: The flow the field is being built for, which supplies the grid and the
            energy used to scale it.

    Returns:
        The initial state, at time zero.

    Raises:
        UnknownNameError: If the regime name has no generator.
        ValidationError: If a parameter is unknown or out of range.
    """
    generators: Mapping[str, Callable[[Regime, np.random.Generator, FluidDynamics], FloatArray]] = {
        TAYLOR_GREEN.name: _taylor_green,
        DECAYING_TURBULENCE.name: _decaying_turbulence,
        SHEAR_LAYER.name: _shear_layer,
    }
    if regime.name not in generators:
        known = ", ".join(sorted(generators))
        raise UnknownNameError(f"unknown fluid regime {regime.name!r}, registered: {known}")
    vorticity = _finalise(dynamics, generators[regime.name](regime, rng, dynamics))
    return dynamics.grid.make_state(vorticity)


def _taylor_green(regime: Regime, rng: np.random.Generator, dynamics: FluidDynamics) -> FloatArray:
    """One Laplacian eigenmode, placed at a random phase."""
    context = f"fluid regime {regime.name!r}"
    check_parameter_names(regime.parameters, _TAYLOR_GREEN_PARAMETERS, context=context)
    grid = dynamics.grid
    wavenumber = int_parameter(regime.parameters, "wavenumber", 2, context=context)
    _check_wavenumber(wavenumber, grid, context=context, name="wavenumber")
    phase = rng.uniform(0.0, TWO_PI, size=2)
    return taylor_green_vorticity(grid, wavenumber, (float(phase[0]), float(phase[1])))


def _decaying_turbulence(
    regime: Regime, rng: np.random.Generator, dynamics: FluidDynamics
) -> FloatArray:
    """Random phases shaped to an energy spectrum that peaks at a chosen wavenumber.

    The shape is McWilliams': the shell energy goes as `k^2 / (1 + (k / k_p)^4)`, which
    rises steeply, peaks near `k_p` and falls off fast enough to leave the smallest
    scales resolved.

    The phases come from transforming a real white noise field rather than from drawing
    them mode by mode. Multiplying a real field's spectrum by a real function of `|k|`
    leaves it the spectrum of a real field, so the Hermitian symmetry is preserved by
    construction instead of being imposed afterwards and possibly wrongly.
    """
    context = f"fluid regime {regime.name!r}"
    check_parameter_names(regime.parameters, _TURBULENCE_PARAMETERS, context=context)
    grid = dynamics.grid
    peak = int_parameter(regime.parameters, "peak_wavenumber", 6, context=context)
    _check_wavenumber(peak, grid, context=context, name="peak wavenumber")

    noise: FloatArray = rng.standard_normal((grid.size, grid.size))
    magnitude = np.sqrt(dynamics.operators.squared_wavenumber)
    peak_magnitude = TWO_PI * peak / grid.length
    amplitude = magnitude**1.5 / np.sqrt(1.0 + (magnitude / peak_magnitude) ** 4)
    return grid.inverse(grid.forward(noise) * amplitude)


def _shear_layer(regime: Regime, rng: np.random.Generator, dynamics: FluidDynamics) -> FloatArray:
    """Two opposed hyperbolic tangent layers, perturbed so that they roll up.

    Two layers rather than one: a single layer cannot be periodic. They sit a half domain
    apart and are exact translates of each other, so their vorticity cancels to round off
    and the field carries no mean.
    """
    context = f"fluid regime {regime.name!r}"
    check_parameter_names(regime.parameters, _SHEAR_PARAMETERS, context=context)
    grid = dynamics.grid
    width = float_parameter(regime.parameters, "layer_width", 0.05, context=context)
    perturbation = float_parameter(regime.parameters, "perturbation", 0.05, context=context)
    if not 0.0 < width <= _MAXIMUM_LAYER_WIDTH:
        raise ValidationError(
            f"{context} layer width must lie in (0, {_MAXIMUM_LAYER_WIDTH}] as a fraction "
            f"of the domain, got {width}"
        )
    if not 0.0 <= perturbation < 1.0:
        raise ValidationError(f"{context} perturbation must lie in [0, 1), got {perturbation}")
    if width * grid.length < grid.spacing:
        raise ValidationError(
            f"{context} layer of width {width * grid.length:.3g} is thinner than a cell "
            f"of width {grid.spacing:.3g}"
        )

    along_x, along_y = grid.coordinates()
    thickness = width * grid.length
    lower = np.cosh((along_y - 0.25 * grid.length) / thickness) ** -2
    upper = np.cosh((along_y - 0.75 * grid.length) / thickness) ** -2
    layers = (upper - lower) / thickness
    phase = float(rng.uniform(0.0, TWO_PI))
    envelope = 1.0 + perturbation * np.sin(TWO_PI * along_x / grid.length - phase)
    field: FloatArray = layers * envelope
    return field


def _check_wavenumber(value: int, grid: FluidGrid, *, context: str, name: str) -> None:
    """Reject a mode the grid cannot carry through the dealiasing cutoff.

    Raises:
        ValidationError: If the mode is below one or above the retained band.
    """
    if not 1 <= value <= grid.cutoff_mode:
        raise ValidationError(
            f"{context} {name} must lie between 1 and the dealiasing cutoff "
            f"{grid.cutoff_mode} for a grid of {grid.size}, got {value}"
        )


def _finalise(dynamics: FluidDynamics, vorticity: FloatArray) -> FloatArray:
    """Truncate to the resolved band, remove the mean, and scale to unit speed.

    Raises:
        NumericalError: If the field carries no energy to scale against.
    """
    grid = dynamics.grid
    spectrum = grid.forward(vorticity) * dynamics.operators.retained
    spectrum[0, 0] = 0.0
    field = grid.inverse(spectrum)
    energy = dynamics.energy(field)
    if energy <= 0.0:
        raise NumericalError("cannot scale a fluid initial condition that carries no energy")
    # Unit root mean square speed is an energy of one half.
    scaled: FloatArray = field * np.sqrt(0.5 / energy)
    return scaled
