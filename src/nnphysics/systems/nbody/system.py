"""The N-body system: the object every outer layer sees.

Softening is a system level parameter rather than a regime one. It changes the potential
the energy invariant measures, and a regime is a statement about initial conditions
rather than about the force law, so a softening that varied by regime would mean two
regimes of the same system were not measuring the same energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nnphysics.core.protocols import Invariant, Predictor, Symmetry
from nnphysics.systems.base import SystemParameters, check_parameter_names, float_parameter
from nnphysics.systems.nbody.dynamics import NBodyDynamics
from nnphysics.systems.nbody.initial_conditions import NBODY_REGIMES, initial_state
from nnphysics.systems.nbody.integrators import VelocityVerlet
from nnphysics.systems.nbody.invariants import AngularMomentum, LinearMomentum, TotalEnergy
from nnphysics.systems.nbody.state import NBODY_STATE_SPEC
from nnphysics.systems.nbody.symmetries import GalileanBoost, Rotation, Translation

if TYPE_CHECKING:
    import numpy as np

    from nnphysics.core.types import Regime, State, StateSpec

__all__ = ["NBODY_SYSTEM_PARAMETERS", "NBodySystem", "build_nbody"]

NBODY_SYSTEM_PARAMETERS = ("gravitational_constant", "softening")

_DEFAULT_SOFTENING = 0.01
"""Small against the tightest separation any declared regime produces."""


@dataclass(frozen=True, slots=True)
class NBodySystem:
    """Gravitational N-body dynamics in two dimensions.

    Attributes:
        dynamics: The force law, shared by the solver and the energy invariant.
    """

    dynamics: NBodyDynamics = field(
        default_factory=lambda: NBodyDynamics(softening=_DEFAULT_SOFTENING)
    )

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "nbody"

    @property
    def state_spec(self) -> StateSpec:
        """Shape and meaning of the fields a state carries."""
        return NBODY_STATE_SPEC

    @property
    def regimes(self) -> tuple[Regime, ...]:
        """Named regions of parameter space this system declares."""
        return NBODY_REGIMES

    def invariants(self, regime: Regime) -> tuple[Invariant, ...]:  # noqa: ARG002
        """Energy, linear momentum and angular momentum.

        No regime changes what these three do, because softening is a system level
        parameter rather than a regime one.

        Args:
            regime: The regime the invariants will be measured in.

        Returns:
            The three invariants.
        """
        return (TotalEnergy(self.dynamics), LinearMomentum(), AngularMomentum())

    @property
    def symmetries(self) -> tuple[Symmetry, ...]:
        """Representative transformations the dynamics commute with.

        The parameters are arbitrary but fixed, so that an equivariance test is
        reproducible and does not have to invent them.
        """
        return (
            Translation((1.5, -0.75)),
            Rotation(0.7),
            GalileanBoost((0.3, -0.2)),
        )

    def initial_state(self, regime: Regime, rng: np.random.Generator) -> State:
        """Draw one initial condition from a regime.

        Args:
            regime: The regime to draw from.
            rng: Generator every random draw comes from.

        Returns:
            The initial state, at time zero.

        Raises:
            UnknownNameError: If the regime name has no generator.
            ValidationError: If a regime parameter is unknown or out of range.
        """
        return initial_state(regime, rng, self.dynamics)

    def reference_predictor(self, regime: Regime, dt: float) -> Predictor:  # noqa: ARG002
        """Build the solver that produces ground truth.

        The regime does not change the solver: velocity Verlet with the system's own
        force law is ground truth everywhere, and the step size is the caller's choice.

        Args:
            regime: The regime the solver will be run in.
            dt: Step size.

        Returns:
            The reference solver.

        Raises:
            ValidationError: If the step size is not positive.
        """
        return VelocityVerlet(self.dynamics, dt)


def build_nbody(parameters: SystemParameters) -> NBodySystem:
    """Build the N-body system from system level parameters.

    Args:
        parameters: Accepts `gravitational_constant` and `softening`.

    Returns:
        The system.

    Raises:
        ValidationError: If a parameter is unknown or out of range.
    """
    context = "nbody system"
    check_parameter_names(parameters, NBODY_SYSTEM_PARAMETERS, context=context)
    return NBodySystem(
        NBodyDynamics(
            gravitational_constant=float_parameter(
                parameters, "gravitational_constant", 1.0, context=context
            ),
            softening=float_parameter(parameters, "softening", _DEFAULT_SOFTENING, context=context),
        )
    )
