from collections.abc import Callable

import numpy as np
import pytest

from nnphysics.core.protocols import Predictor
from nnphysics.core.types import FloatArray, State, Trajectory
from nnphysics.systems.fluid import FluidGrid

RollOut = Callable[[Predictor, State, int], Trajectory]
SmoothVorticity = Callable[[FluidGrid], FloatArray]


def _roll_out(predictor: Predictor, state: State, n_steps: int) -> Trajectory:
    states = [state]
    for _ in range(n_steps):
        states.append(predictor.step(states[-1]))
    return Trajectory.from_states(states)


def _smooth_vorticity(grid: FluidGrid) -> FloatArray:
    """A few low modes, written analytically so it is the same flow on any grid.

    It is smooth and band limited but not an eigenfunction, so it develops structure at
    every scale the grid can carry, which is what a convergence test needs.
    """
    along_x, along_y = grid.coordinates()
    field: FloatArray = (
        np.sin(along_x) * np.cos(2.0 * along_y)
        + 0.5 * np.cos(3.0 * along_x + along_y)
        + 0.25 * np.sin(2.0 * along_x - 3.0 * along_y)
    )
    return field - float(np.mean(field))


@pytest.fixture(scope="session")
def roll_out() -> RollOut:
    return _roll_out


@pytest.fixture(scope="session")
def smooth_vorticity() -> SmoothVorticity:
    return _smooth_vorticity
