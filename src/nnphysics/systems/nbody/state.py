"""The N-body state: what its fields are and how one is built and read.

Masses are carried in the state rather than held by the system because a predictor is
handed a state and nothing else, so anything the dynamics need must travel with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import FieldSpec, State, StateSpec
from nnphysics.core.units import LENGTH, MASS, VELOCITY

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray

__all__ = [
    "MASS_FIELD",
    "NBODY_STATE_SPEC",
    "POSITION_FIELD",
    "VELOCITY_FIELD",
    "make_state",
    "unpack",
]

POSITION_FIELD = "position"
VELOCITY_FIELD = "velocity"
MASS_FIELD = "mass"

DIMENSIONS = 2
"""Two dimensions, not three: plots stay readable and pair costs stay low."""

NBODY_STATE_SPEC = StateSpec(
    fields=(
        FieldSpec(POSITION_FIELD, (None, DIMENSIONS), LENGTH, "positions of each body"),
        FieldSpec(VELOCITY_FIELD, (None, DIMENSIONS), VELOCITY, "velocities of each body"),
        FieldSpec(MASS_FIELD, (None,), MASS, "mass of each body, constant in time"),
    )
)


def make_state(
    position: FloatArray, velocity: FloatArray, mass: FloatArray, time: float = 0.0
) -> State:
    """Build a validated N-body state.

    Args:
        position: Positions, shape `(n_bodies, 2)`.
        velocity: Velocities, shape `(n_bodies, 2)`.
        mass: Masses, shape `(n_bodies,)`, all strictly positive.
        time: Simulation time of the state.

    Returns:
        The state.

    Raises:
        ValidationError: If a shape is wrong, the three fields disagree on the number of
            bodies, there are no bodies, or a mass is not positive.
    """
    state = State(
        fields={
            POSITION_FIELD: np.asarray(position, dtype=np.float64),
            VELOCITY_FIELD: np.asarray(velocity, dtype=np.float64),
            MASS_FIELD: np.asarray(mass, dtype=np.float64),
        },
        time=time,
    )
    NBODY_STATE_SPEC.validate(state)
    counts = {name: state.fields[name].shape[0] for name in state.names}
    if len(set(counts.values())) != 1:
        raise ValidationError(f"N-body fields disagree on the number of bodies: {counts}")
    if next(iter(counts.values())) == 0:
        raise ValidationError("an N-body state must have at least one body")
    if not np.all(state.fields[MASS_FIELD] > 0.0):
        raise ValidationError("every N-body mass must be strictly positive")
    return state


def unpack(state: State) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Read positions, velocities and masses out of a state.

    Args:
        state: A state carrying the N-body fields.

    Returns:
        Positions, velocities and masses.

    Raises:
        ValidationError: If the state does not match the N-body specification.
    """
    NBODY_STATE_SPEC.validate(state)
    return (
        state.fields[POSITION_FIELD],
        state.fields[VELOCITY_FIELD],
        state.fields[MASS_FIELD],
    )
