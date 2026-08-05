"""Seeded initial conditions for the named regimes.

Every generator takes a generator object and draws from it in a fixed order, so a seed
determines a trajectory exactly. Nothing here touches a global random function.

All three regimes are produced with the centre of mass at the origin and at rest, which
makes the momentum invariants read zero and their drift easy to interpret.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import NumericalError, UnknownNameError, ValidationError
from nnphysics.core.types import Regime, State
from nnphysics.systems.base import check_parameter_names, float_parameter, int_parameter
from nnphysics.systems.nbody.dynamics import kinetic_energy
from nnphysics.systems.nbody.state import make_state
from nnphysics.systems.nbody.symmetries import Rotation

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray
    from nnphysics.systems.nbody.dynamics import NBodyDynamics

__all__ = ["NBODY_REGIMES", "initial_state"]

COLD_COLLAPSE = Regime(
    name="cold_collapse",
    parameters={"n_bodies": 32.0, "radius": 1.0, "mass_spread": 1.0, "virial_ratio": 0.0},
)
"""A cloud released from rest. It falls together, so it is the hardest on a solver."""

VIRIALISED_CLUSTER = Regime(
    name="virialised_cluster",
    parameters={"n_bodies": 32.0, "radius": 1.0, "mass_spread": 4.0, "virial_ratio": 1.0},
)
"""A cluster in equilibrium, `2T = |U|`, which holds its size over a long rollout."""

HIERARCHICAL_PAIR = Regime(
    name="hierarchical_pair",
    parameters={"inner_separation": 0.2, "outer_separation": 2.0, "total_mass": 1.0},
)
"""Two tight binaries orbiting each other. Two separated timescales in one trajectory."""

NBODY_REGIMES = (COLD_COLLAPSE, VIRIALISED_CLUSTER, HIERARCHICAL_PAIR)

_CLUSTER_PARAMETERS = ("n_bodies", "radius", "mass_spread", "virial_ratio")
_PAIR_PARAMETERS = ("inner_separation", "outer_separation", "total_mass")
_MINIMUM_BODIES = 2
_MINIMUM_SEPARATION_RATIO = 4.0
"""Below this the outer orbit is not a perturbation of the binaries and the name lies."""


def initial_state(regime: Regime, rng: np.random.Generator, dynamics: NBodyDynamics) -> State:
    """Draw one initial condition from a named regime.

    Args:
        regime: The regime to draw from. Its name selects the generator and its
            parameters configure it.
        rng: Generator every random draw comes from.
        dynamics: Force law, needed to set velocities from a virial ratio.

    Returns:
        The initial state, at time zero.

    Raises:
        UnknownNameError: If the regime name has no generator.
        ValidationError: If a parameter is unknown or out of range.
    """
    generators: Mapping[str, Callable[[Regime, np.random.Generator, NBodyDynamics], State]] = {
        COLD_COLLAPSE.name: _cluster,
        VIRIALISED_CLUSTER.name: _cluster,
        HIERARCHICAL_PAIR.name: _hierarchical_pair,
    }
    if regime.name not in generators:
        known = ", ".join(sorted(generators))
        raise UnknownNameError(f"unknown nbody regime {regime.name!r}, registered: {known}")
    return generators[regime.name](regime, rng, dynamics)


def _cluster(regime: Regime, rng: np.random.Generator, dynamics: NBodyDynamics) -> State:
    """A disc of bodies with velocities scaled to a target virial ratio."""
    context = f"nbody regime {regime.name!r}"
    check_parameter_names(regime.parameters, _CLUSTER_PARAMETERS, context=context)
    n_bodies = int_parameter(regime.parameters, "n_bodies", 32, context=context)
    radius = float_parameter(regime.parameters, "radius", 1.0, context=context)
    mass_spread = float_parameter(regime.parameters, "mass_spread", 1.0, context=context)
    virial_ratio = float_parameter(regime.parameters, "virial_ratio", 1.0, context=context)
    if n_bodies < _MINIMUM_BODIES:
        raise ValidationError(f"{context} needs at least two bodies, got {n_bodies}")
    if radius <= 0.0:
        raise ValidationError(f"{context} radius must be positive, got {radius}")
    if mass_spread < 1.0:
        raise ValidationError(f"{context} mass spread must be at least one, got {mass_spread}")
    if virial_ratio < 0.0:
        raise ValidationError(f"{context} virial ratio must not be negative, got {virial_ratio}")

    mass = _sample_masses(n_bodies, mass_spread, rng)
    position = _sample_disc(n_bodies, radius, rng)
    direction = _sample_directions(n_bodies, rng)
    position = position - _centre_of_mass(position, mass)
    velocity = _scale_to_virial_ratio(position, direction, mass, dynamics, virial_ratio)
    return make_state(position, velocity, mass)


def _hierarchical_pair(
    regime: Regime,
    rng: np.random.Generator,
    dynamics: NBodyDynamics,
) -> State:
    """Two circular binaries on a circular orbit about their common centre of mass."""
    context = f"nbody regime {regime.name!r}"
    check_parameter_names(regime.parameters, _PAIR_PARAMETERS, context=context)
    inner = float_parameter(regime.parameters, "inner_separation", 0.2, context=context)
    outer = float_parameter(regime.parameters, "outer_separation", 2.0, context=context)
    total_mass = float_parameter(regime.parameters, "total_mass", 1.0, context=context)
    if inner <= 0.0 or outer <= 0.0:
        raise ValidationError(f"{context} separations must be positive, got {inner} and {outer}")
    if outer <= _MINIMUM_SEPARATION_RATIO * inner:
        raise ValidationError(
            f"{context} is only hierarchical if the outer separation exceeds four times the "
            f"inner one, got {outer} and {inner}"
        )
    if total_mass <= 0.0:
        raise ValidationError(f"{context} total mass must be positive, got {total_mass}")

    body_mass = 0.25 * total_mass
    binary_mass = 2.0 * body_mass
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)

    offset, drift = _circular_pair(binary_mass, binary_mass, outer, phases[0], dynamics)
    positions = []
    velocities = []
    for index in range(2):
        pair_position, pair_velocity = _circular_pair(
            body_mass, body_mass, inner, phases[index + 1], dynamics
        )
        positions.append(pair_position + offset[index])
        velocities.append(pair_velocity + drift[index])

    position = np.concatenate(positions)
    velocity = np.concatenate(velocities)
    mass = np.full(4, body_mass)
    position = position - _centre_of_mass(position, mass)
    velocity = velocity - _centre_of_mass(velocity, mass)
    return Rotation(float(rng.uniform(0.0, 2.0 * np.pi))).apply(
        make_state(position, velocity, mass)
    )


def _circular_pair(
    mass_a: float, mass_b: float, separation: float, phase: float, dynamics: NBodyDynamics
) -> tuple[FloatArray, FloatArray]:
    """Positions and velocities of two bodies on a circular orbit about their barycentre.

    The softened force law is used, so the orbit is circular for the dynamics actually
    integrated rather than for the unsoftened ones.
    """
    total = mass_a + mass_b
    softened = separation**2 + dynamics.softening**2
    angular_speed = np.sqrt(dynamics.gravitational_constant * total / softened**1.5)
    radial = separation * np.array([np.cos(phase), np.sin(phase)])
    tangential = angular_speed * separation * np.array([-np.sin(phase), np.cos(phase)])
    weights = np.array([[-mass_b / total], [mass_a / total]])
    position: FloatArray = weights * radial[np.newaxis, :]
    velocity: FloatArray = weights * tangential[np.newaxis, :]
    return position, velocity


def _sample_masses(n_bodies: int, spread: float, rng: np.random.Generator) -> FloatArray:
    """Masses log uniform over a spread, normalised so the total mass is one."""
    exponent = rng.uniform(0.0, np.log(spread) if spread > 1.0 else 0.0, size=n_bodies)
    mass: FloatArray = np.exp(exponent)
    return mass / float(np.sum(mass))


def _sample_disc(n_bodies: int, radius: float, rng: np.random.Generator) -> FloatArray:
    """Positions uniform over a disc. The square root keeps the density flat."""
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_bodies)
    distance = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_bodies))
    position: FloatArray = np.stack([distance * np.cos(angle), distance * np.sin(angle)], axis=1)
    return position


def _sample_directions(n_bodies: int, rng: np.random.Generator) -> FloatArray:
    """Unit vectors with uniformly distributed direction."""
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_bodies)
    direction: FloatArray = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return direction


def _centre_of_mass(quantity: FloatArray, mass: FloatArray) -> FloatArray:
    """Mass weighted mean of a per body vector quantity."""
    centre: FloatArray = np.sum(mass[:, np.newaxis] * quantity, axis=0) / float(np.sum(mass))
    return centre


def _scale_to_virial_ratio(
    position: FloatArray,
    direction: FloatArray,
    mass: FloatArray,
    dynamics: NBodyDynamics,
    virial_ratio: float,
) -> FloatArray:
    """Scale unit velocity directions so that `2T / |U|` hits a target.

    Raises:
        NumericalError: If the configuration has no potential energy to scale against.
    """
    if virial_ratio == 0.0:
        return np.zeros_like(direction)
    velocity: FloatArray = direction - _centre_of_mass(direction, mass)
    unscaled = kinetic_energy(velocity, mass)
    potential = abs(dynamics.potential_energy(position, mass))
    if unscaled <= 0.0 or potential <= 0.0:
        raise NumericalError("cannot scale velocities to a virial ratio for this configuration")
    return velocity * float(np.sqrt(0.5 * virial_ratio * potential / unscaled))
