from collections.abc import Callable

import numpy as np
import pytest

from nnphysics.core.protocols import Conservation, Invariant, Predictor
from nnphysics.core.types import State, Trajectory
from nnphysics.core.units import ENERGY, MOMENTUM
from nnphysics.systems.nbody import (
    AngularMomentum,
    LinearMomentum,
    NBodyDynamics,
    TotalEnergy,
    VelocityVerlet,
    make_state,
    unpack,
)

EXACT = NBodyDynamics(gravitational_constant=1.0, softening=0.0)


SOFTENED = NBodyDynamics(gravitational_constant=1.0, softening=0.05)

RollOut = Callable[[Predictor, State, int], Trajectory]
TwoBody = Callable[..., tuple[State, float]]

STEPS = 2000

MOMENTUM_ROUND_OFF = STEPS * float(np.finfo(np.float64).eps)
"""Ceiling on relative linear momentum drift over `STEPS` steps.

The pairwise forces are antisymmetric, so in exact arithmetic velocity Verlet leaves the
total momentum unchanged and every observed drift is accumulated round off. One step can
lose an epsilon, so the run can lose `STEPS` of them, and that product is the honest bound.

A hand picked constant here is a statement about one machine rather than about the solver.
This was 1e-14, which held on Windows at 4.9e-15 and failed in CI at 1.1e-14: the same seed
and the same code, differing only in the order the sums were accumulated in. A broken force
law drifts nine orders above this bound, so the headroom costs the test nothing.
"""


def cluster(seed: int, n_bodies: int = 8) -> State:
    rng = np.random.default_rng(seed)
    return make_state(
        rng.normal(size=(n_bodies, 2)),
        0.2 * rng.normal(size=(n_bodies, 2)),
        rng.uniform(0.5, 1.5, size=n_bodies),
    )


class TestDeclarations:
    @pytest.mark.parametrize("invariant", [TotalEnergy(EXACT), LinearMomentum(), AngularMomentum()])
    def test_each_invariant_satisfies_the_protocol(self, invariant: Invariant) -> None:
        assert isinstance(invariant, Invariant)
        assert invariant.name
        assert invariant.rtol > 0.0

    def test_energy_is_declared_approximate(self) -> None:
        energy = TotalEnergy(EXACT)
        assert energy.conservation is Conservation.APPROXIMATE
        assert energy.dimension == ENERGY

    @pytest.mark.parametrize("invariant", [LinearMomentum(), AngularMomentum()])
    def test_the_momenta_are_declared_exact(self, invariant: Invariant) -> None:
        assert invariant.conservation is Conservation.EXACT

    def test_linear_momentum_carries_the_momentum_dimension(self) -> None:
        assert LinearMomentum().dimension == MOMENTUM


class TestValues:
    def test_energy_is_kinetic_plus_potential(self) -> None:
        state = make_state(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            np.array([[0.0, 1.0], [0.0, -1.0]]),
            np.array([3.0, 1.0]),
        )
        assert TotalEnergy(EXACT).evaluate(state) == pytest.approx(0.5 * 4.0 - 1.5)

    def test_linear_momentum_is_zero_for_a_balanced_state(self) -> None:
        state = make_state(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            np.array([[0.0, 1.0], [0.0, -3.0]]),
            np.array([3.0, 1.0]),
        )
        assert LinearMomentum().evaluate(state) == pytest.approx(0.0)

    def test_linear_momentum_catches_a_drifting_state(self) -> None:
        state = make_state(np.zeros((2, 2)), np.ones((2, 2)), np.array([1.0, 3.0]))
        assert LinearMomentum().evaluate(state) == pytest.approx(4.0 * np.sqrt(2.0))

    def test_angular_momentum_is_signed(self) -> None:
        clockwise = make_state(np.array([[1.0, 0.0]]), np.array([[0.0, -2.0]]), np.array([3.0]))
        assert AngularMomentum().evaluate(clockwise) == pytest.approx(-6.0)


def momentum_scale(state: State) -> float:
    """Size of the individual momenta, which the total is a near exact cancellation of."""
    _, velocity, mass = unpack(state)
    return float(np.sum(mass[:, np.newaxis] * np.abs(velocity)))


class TestConservationUnderTheSolver:
    """Momentum drift is a bug in the force law, not a consequence of the step size."""

    def test_linear_momentum_is_conserved_to_machine_precision(
        self, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT, eccentricity=0.4)
        trajectory = roll_out(VelocityVerlet(EXACT, period / 500), state, STEPS)
        momentum = np.array([LinearMomentum().evaluate(step) for step in trajectory])

        drift = float(np.max(np.abs(momentum - momentum[0]))) / momentum_scale(state)
        assert drift < MOMENTUM_ROUND_OFF

    def test_angular_momentum_is_conserved_to_machine_precision(
        self, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT, eccentricity=0.4)
        trajectory = roll_out(VelocityVerlet(EXACT, period / 500), state, STEPS)
        angular = np.array([AngularMomentum().evaluate(step) for step in trajectory])

        assert float(np.max(np.abs(angular / angular[0] - 1.0))) < 1e-13

    def test_both_momenta_are_conserved_for_a_many_body_cluster(self, roll_out: RollOut) -> None:
        state = cluster(seed=7)
        trajectory = roll_out(VelocityVerlet(SOFTENED, 1e-3), state, STEPS)
        linear = np.array([LinearMomentum().evaluate(step) for step in trajectory])
        angular = np.array([AngularMomentum().evaluate(step) for step in trajectory])

        assert (
            float(np.max(np.abs(linear - linear[0]))) / momentum_scale(state) < MOMENTUM_ROUND_OFF
        )
        # Angular momentum is a cross product, so its round off carries the position scale
        # as well and lands an order above the linear bound rather than at it.
        assert float(np.max(np.abs(angular / angular[0] - 1.0))) < 10.0 * MOMENTUM_ROUND_OFF
