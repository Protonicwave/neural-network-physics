import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import State
from nnphysics.systems.nbody import NBODY_STATE_SPEC, make_state, unpack


def test_a_state_round_trips_through_unpack() -> None:
    position = np.array([[0.0, 1.0], [2.0, 3.0]])
    velocity = np.array([[1.0, 0.0], [0.0, 1.0]])
    mass = np.array([1.0, 2.0])
    unpacked = unpack(make_state(position, velocity, mass, time=0.5))

    assert np.array_equal(unpacked[0], position)
    assert np.array_equal(unpacked[1], velocity)
    assert np.array_equal(unpacked[2], mass)


def test_the_specification_names_the_three_fields() -> None:
    assert NBODY_STATE_SPEC.names == ("position", "velocity", "mass")


def test_integer_input_is_promoted_rather_than_rejected() -> None:
    state = make_state(np.zeros((2, 2), dtype=int), np.zeros((2, 2)), np.ones(2))
    assert state.fields["position"].dtype == np.float64


def test_fields_disagreeing_on_the_number_of_bodies_are_rejected() -> None:
    with pytest.raises(ValidationError, match="disagree on the number of bodies"):
        make_state(np.zeros((3, 2)), np.zeros((2, 2)), np.ones(3))


def test_a_three_dimensional_state_is_rejected() -> None:
    with pytest.raises(ValidationError, match="expected"):
        make_state(np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2))


def test_an_empty_state_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one body"):
        make_state(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))


@pytest.mark.parametrize("mass", [0.0, -1.0])
def test_a_non_positive_mass_is_rejected(mass: float) -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        make_state(np.zeros((2, 2)), np.zeros((2, 2)), np.array([1.0, mass]))


def test_unpacking_a_foreign_state_is_rejected() -> None:
    with pytest.raises(ValidationError, match="missing fields"):
        unpack(State(fields={"vorticity": np.zeros((4, 4))}))
