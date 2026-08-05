from collections.abc import Callable

import numpy as np
import pytest

from nnphysics.core.protocols import Predictor
from nnphysics.core.types import State, Trajectory
from nnphysics.systems.nbody import NBodyDynamics, make_state

RollOut = Callable[[Predictor, State, int], Trajectory]
TwoBody = Callable[..., tuple[State, float]]


def _roll_out(predictor: Predictor, state: State, n_steps: int) -> Trajectory:
    states = [state]
    for _ in range(n_steps):
        states.append(predictor.step(states[-1]))
    return Trajectory.from_states(states)


def _two_body(
    dynamics: NBodyDynamics,
    *,
    mass_a: float = 0.6,
    mass_b: float = 0.4,
    apoapsis: float = 1.0,
    eccentricity: float = 0.0,
) -> tuple[State, float]:
    """A bound two body orbit started at apoapsis, with its analytic period.

    Only valid for unsoftened dynamics, which is what the analytic tests use.
    """
    total = mass_a + mass_b
    mu = dynamics.gravitational_constant * total
    semi_major = apoapsis / (1.0 + eccentricity)
    speed = np.sqrt(mu * (1.0 - eccentricity) / (semi_major * (1.0 + eccentricity)))
    separation = np.array([apoapsis, 0.0])
    relative_velocity = np.array([0.0, speed])
    weights = np.array([[-mass_b / total], [mass_a / total]])
    period = float(2.0 * np.pi * np.sqrt(semi_major**3 / mu))
    state = make_state(
        weights * separation[np.newaxis, :],
        weights * relative_velocity[np.newaxis, :],
        np.array([mass_a, mass_b]),
    )
    return state, period


@pytest.fixture
def roll_out() -> RollOut:
    return _roll_out


@pytest.fixture
def two_body() -> TwoBody:
    return _two_body
