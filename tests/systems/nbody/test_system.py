import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import Invariant, Predictor, Symmetry, System
from nnphysics.systems.nbody import NBodySystem, VelocityVerlet, build_nbody

SYSTEM = build_nbody({})


def test_the_system_satisfies_the_protocol() -> None:
    assert isinstance(SYSTEM, System)
    assert SYSTEM.name == "nbody"


def test_it_declares_its_state_specification() -> None:
    assert SYSTEM.state_spec.names == ("position", "velocity", "mass")


def test_it_declares_three_regimes() -> None:
    assert [regime.name for regime in SYSTEM.regimes] == [
        "cold_collapse",
        "virialised_cluster",
        "hierarchical_pair",
    ]


def test_it_declares_the_three_invariants() -> None:
    assert [invariant.name for invariant in SYSTEM.invariants] == [
        "energy",
        "linear_momentum",
        "angular_momentum",
    ]
    assert all(isinstance(invariant, Invariant) for invariant in SYSTEM.invariants)


def test_it_declares_the_three_symmetries() -> None:
    assert [symmetry.name for symmetry in SYSTEM.symmetries] == [
        "translation",
        "rotation",
        "galilean_boost",
    ]
    assert all(isinstance(symmetry, Symmetry) for symmetry in SYSTEM.symmetries)


def test_the_energy_invariant_uses_the_system_softening() -> None:
    """A mismatch here would report drift that the solver is not responsible for."""
    system = build_nbody({"softening": 0.25})
    cold = system.regimes[0]
    state = system.initial_state(cold, np.random.default_rng(0))
    # The cold collapse starts at rest, so its energy is its potential energy alone.
    expected = system.dynamics.potential_energy(state.fields["position"], state.fields["mass"])

    assert system.invariants[0].evaluate(state) == pytest.approx(expected)


def test_the_reference_predictor_is_velocity_verlet() -> None:
    predictor = SYSTEM.reference_predictor(SYSTEM.regimes[0], 0.01)

    assert isinstance(predictor, VelocityVerlet)
    assert isinstance(predictor, Predictor)
    assert predictor.dt == 0.01


def test_an_initial_state_can_be_stepped_by_the_reference_predictor() -> None:
    regime = SYSTEM.regimes[1]
    state = SYSTEM.initial_state(regime, np.random.default_rng(0))
    stepped = SYSTEM.reference_predictor(regime, 0.001).step(state)

    stepped.require_finite()
    assert stepped.time == pytest.approx(0.001)


def test_parameters_reach_the_dynamics() -> None:
    system = build_nbody({"gravitational_constant": 2.0, "softening": 0.5})

    assert system.dynamics.gravitational_constant == 2.0
    assert system.dynamics.softening == 0.5


def test_an_unknown_system_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown nbody system parameters"):
        build_nbody({"viscosity": 0.1})


def test_a_bad_parameter_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be a number"):
        build_nbody({"softening": "small"})


def test_the_default_system_softens_close_encounters() -> None:
    assert NBodySystem().dynamics.softening > 0.0
