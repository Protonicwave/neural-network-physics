from collections.abc import Callable
from itertools import pairwise

import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import Predictor
from nnphysics.core.types import FloatArray, State, Trajectory
from nnphysics.systems.nbody import (
    NBodyDynamics,
    RungeKutta4,
    VelocityVerlet,
    kinetic_energy,
    unpack,
)

EXACT = NBodyDynamics(gravitational_constant=1.0, softening=0.0)

Integrator = type[VelocityVerlet] | type[RungeKutta4]
RollOut = Callable[[Predictor, State, int], Trajectory]
TwoBody = Callable[..., tuple[State, float]]


def max_position_error(left: State, right: State) -> float:
    return float(np.max(np.abs(unpack(left)[0] - unpack(right)[0])))


def total_energy(state: State, dynamics: NBodyDynamics) -> float:
    """Energy read straight off the dynamics, so this test does not lean on `invariants`."""
    position, velocity, mass = unpack(state)
    return kinetic_energy(velocity, mass) + dynamics.potential_energy(position, mass)


def relative_energy_error(trajectory: Trajectory, dynamics: NBodyDynamics) -> FloatArray:
    values = np.array([total_energy(state, dynamics) for state in trajectory])
    error: FloatArray = (values - values[0]) / abs(values[0])
    return error


class TestConvergenceOrder:
    """The test that catches an integrator that is wrong but plausible looking.

    A circular orbit returns exactly to its initial state after one period, so the global
    error after one period needs no reference solution.
    """

    @pytest.mark.parametrize(
        ("integrator", "expected_order"),
        [(VelocityVerlet, 2.0), (RungeKutta4, 4.0)],
    )
    def test_global_error_falls_at_the_expected_order(
        self,
        integrator: Integrator,
        expected_order: float,
        roll_out: RollOut,
        two_body: TwoBody,
    ) -> None:
        state, period = two_body(EXACT)
        errors = []
        # Fine enough that both schemes are in their asymptotic regime, coarse enough
        # that fourth order error stays well clear of round off.
        for n_steps in (256, 512, 1024):
            predictor = integrator(EXACT, period / n_steps)
            final = roll_out(predictor, state, n_steps)[-1]
            errors.append(max_position_error(final, state))

        orders = [np.log2(coarse / fine) for coarse, fine in pairwise(errors)]
        assert all(abs(order - expected_order) < 0.2 for order in orders), (
            f"errors {errors} give orders {orders}, expected {expected_order}"
        )


class TestCircularOrbit:
    def test_radius_and_speed_hold_over_many_orbits(
        self, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT)
        n_orbits = 10
        # The symplectic radius error is bounded and second order in the step, so the
        # tolerance below is a statement about the step size, not about the run length.
        steps_per_orbit = 2000
        trajectory = roll_out(
            VelocityVerlet(EXACT, period / steps_per_orbit), state, n_orbits * steps_per_orbit
        )
        position = trajectory.fields["position"]
        velocity = trajectory.fields["velocity"]
        radius = np.linalg.norm(position[:, 1] - position[:, 0], axis=-1)
        speed = np.linalg.norm(velocity[:, 1] - velocity[:, 0], axis=-1)

        assert np.allclose(radius, radius[0], rtol=1e-5)
        assert np.allclose(speed, speed[0], rtol=1e-5)

    def test_measured_period_matches_the_analytic_one(
        self, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT)
        steps_per_orbit = 2000
        dt = period / steps_per_orbit
        trajectory = roll_out(VelocityVerlet(EXACT, dt), state, int(1.5 * steps_per_orbit))
        position = trajectory.fields["position"]
        # The orbit starts on the positive x axis, so the period is the next crossing of
        # the axis in the same direction.
        offset = position[:, 1, 1] - position[:, 0, 1]
        after_half = steps_per_orbit // 2
        crossings = np.flatnonzero(
            (offset[after_half:-1] < 0.0) & (offset[after_half + 1 :] >= 0.0)
        )
        assert crossings.size >= 1

        index = int(crossings[0]) + after_half
        fraction = -offset[index] / (offset[index + 1] - offset[index])
        measured = float(trajectory.times[index] + fraction * dt)
        assert measured == pytest.approx(period, rel=1e-5)


class TestKeplerEllipse:
    def test_an_eccentric_orbit_does_not_precess(
        self, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT, eccentricity=0.5)
        steps_per_orbit = 4000
        trajectory = roll_out(
            VelocityVerlet(EXACT, period / steps_per_orbit), state, 8 * steps_per_orbit
        )
        angle = laplace_runge_lenz_angle(trajectory, EXACT.gravitational_constant)

        drift = np.abs(np.unwrap(angle) - angle[0])
        assert float(np.max(drift)) < 1e-4


def laplace_runge_lenz_angle(trajectory: Trajectory, gravitational_constant: float) -> FloatArray:
    """Direction of the Runge-Lenz vector of the relative orbit, in radians.

    The vector points at pericentre and is conserved only for an exact inverse square
    force, so its direction drifting is precession.
    """
    position = trajectory.fields["position"]
    velocity = trajectory.fields["velocity"]
    mass = trajectory.fields["mass"][0]
    separation = position[:, 1] - position[:, 0]
    relative_velocity = velocity[:, 1] - velocity[:, 0]
    distance = np.linalg.norm(separation, axis=-1)
    angular_momentum = (
        separation[:, 0] * relative_velocity[:, 1] - separation[:, 1] * relative_velocity[:, 0]
    )
    mu = gravitational_constant * float(np.sum(mass))
    vector = np.stack(
        [
            relative_velocity[:, 1] * angular_momentum - mu * separation[:, 0] / distance,
            -relative_velocity[:, 0] * angular_momentum - mu * separation[:, 1] / distance,
        ],
        axis=-1,
    )
    angle: FloatArray = np.arctan2(vector[:, 1], vector[:, 0])
    return angle


class TestEnergyBehaviour:
    """Symplectic and non symplectic schemes fail differently, and both failures matter."""

    steps_per_orbit = 100
    n_orbits = 400

    def windowed_drift(self, error: FloatArray, n_windows: int = 10) -> FloatArray:
        usable = error[: error.size - error.size % n_windows]
        drift: FloatArray = np.abs(usable.reshape(n_windows, -1).mean(axis=1))
        return drift

    @pytest.mark.slow
    def test_verlet_energy_error_is_bounded_and_oscillatory(
        self, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT, eccentricity=0.5)
        dt = period / self.steps_per_orbit
        trajectory = roll_out(
            VelocityVerlet(EXACT, dt), state, self.n_orbits * self.steps_per_orbit
        )
        error = relative_energy_error(trajectory, EXACT)
        drift = self.windowed_drift(error)

        # The peak is reached at the first pericentre, where the step resolves the orbit
        # worst, and no later pass exceeds it.
        assert float(np.max(np.abs(error))) < 2e-2
        # Bounded, not merely small: the last window is no worse than the first.
        assert drift[-1] < 2.0 * drift[0]
        assert float(np.max(error)) > 0.0 > float(np.min(error))

    @pytest.mark.slow
    def test_runge_kutta_energy_error_drifts_monotonically(
        self, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT, eccentricity=0.5)
        dt = period / self.steps_per_orbit
        trajectory = roll_out(RungeKutta4(EXACT, dt), state, self.n_orbits * self.steps_per_orbit)
        error = relative_energy_error(trajectory, EXACT)
        drift = self.windowed_drift(error)

        assert np.all(np.diff(drift) > 0.0)
        assert drift[-1] > 10.0 * drift[0]
        # The energy loss is one sided, which is what unbounded drift looks like.
        assert float(np.max(np.abs(error))) == pytest.approx(abs(float(error[-1])), rel=1e-6)


class TestStepValidation:
    @pytest.mark.parametrize("integrator", [VelocityVerlet, RungeKutta4])
    @pytest.mark.parametrize("dt", [0.0, -0.1])
    def test_a_non_positive_step_is_rejected(self, integrator: Integrator, dt: float) -> None:
        with pytest.raises(ValidationError, match="step size must be positive"):
            integrator(EXACT, dt)

    @pytest.mark.parametrize("integrator", [VelocityVerlet, RungeKutta4])
    def test_an_integrator_is_a_predictor(self, integrator: Integrator) -> None:
        predictor = integrator(EXACT, 0.01)
        assert isinstance(predictor, Predictor)
        assert predictor.dt == 0.01
        assert predictor.name


class TestDeterminism:
    @pytest.mark.parametrize("integrator", [VelocityVerlet, RungeKutta4])
    def test_the_same_input_gives_bitwise_identical_trajectories(
        self, integrator: Integrator, roll_out: RollOut, two_body: TwoBody
    ) -> None:
        state, period = two_body(EXACT, eccentricity=0.3)
        predictor = integrator(EXACT, period / 128)
        first = roll_out(predictor, state, 256)
        second = roll_out(predictor, state, 256)

        for name in first.names:
            assert np.array_equal(first.fields[name], second.fields[name])
        assert np.array_equal(first.times, second.times)
