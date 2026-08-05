import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import Conservation, Invariant, Predictor, Symmetry, System
from nnphysics.core.types import Regime
from nnphysics.systems.fluid import (
    FluidSystem,
    IntegratingFactorRK4,
    QuarterTurn,
    Translation,
    build_fluid,
)

SYSTEM = build_fluid({"grid_size": 32})


def test_the_system_satisfies_the_protocol() -> None:
    assert isinstance(SYSTEM, System)
    assert SYSTEM.name == "fluid"


def test_it_declares_its_state_specification() -> None:
    assert SYSTEM.state_spec.names == ("vorticity",)
    assert SYSTEM.state_spec.fields[0].shape == (32, 32)


def test_it_declares_three_regimes() -> None:
    assert [regime.name for regime in SYSTEM.regimes] == [
        "taylor_green",
        "decaying_turbulence",
        "shear_layer",
    ]


def test_it_declares_the_two_invariants() -> None:
    invariants = SYSTEM.invariants(SYSTEM.regimes[0])

    assert [invariant.name for invariant in invariants] == ["energy", "enstrophy"]
    assert all(isinstance(invariant, Invariant) for invariant in invariants)


def test_it_declares_the_two_symmetries() -> None:
    assert [symmetry.name for symmetry in SYSTEM.symmetries] == ["translation", "rotation"]
    assert all(isinstance(symmetry, Symmetry) for symmetry in SYSTEM.symmetries)


def test_the_regime_decides_what_the_invariants_do() -> None:
    """The reason the protocol asks for invariants by regime rather than as a property."""
    viscous = Regime("taylor_green", {"reynolds": 100.0, "wavenumber": 2.0})
    inviscid = Regime("taylor_green", {"reynolds": float("inf"), "wavenumber": 2.0})

    assert all(
        invariant.conservation is Conservation.DECAYING for invariant in SYSTEM.invariants(viscous)
    )
    assert all(
        invariant.conservation is Conservation.APPROXIMATE
        for invariant in SYSTEM.invariants(inviscid)
    )


def test_the_invariants_measure_the_viscosity_the_solver_uses() -> None:
    """A mismatch here would report drift the solver is not responsible for."""
    regime = SYSTEM.regimes[1]
    predictor = SYSTEM.reference_predictor(regime, 0.005)

    assert isinstance(predictor, IntegratingFactorRK4)
    assert predictor.dynamics.viscosity == SYSTEM.dynamics(regime).viscosity


def test_the_reference_predictor_is_the_spectral_solver() -> None:
    predictor = SYSTEM.reference_predictor(SYSTEM.regimes[0], 0.005)

    assert isinstance(predictor, Predictor)
    assert predictor.name == "fluid-ifrk4"
    assert predictor.dt == 0.005


def test_an_initial_state_can_be_stepped_by_the_reference_predictor() -> None:
    regime = SYSTEM.regimes[2]
    state = SYSTEM.initial_state(regime, np.random.default_rng(0))

    stepped = SYSTEM.reference_predictor(regime, 0.005).step(state)

    stepped.require_finite()
    assert stepped.time == pytest.approx(0.005)


def test_a_seed_and_a_regime_determine_a_whole_trajectory() -> None:
    """Nothing in the path from seed to trajectory reaches for a global generator."""
    regime = SYSTEM.regimes[1]
    predictor = SYSTEM.reference_predictor(regime, 0.005)

    finals = []
    for _ in range(2):
        state = SYSTEM.initial_state(regime, np.random.default_rng(5))
        for _ in range(20):
            state = predictor.step(state)
        finals.append(state.fields["vorticity"])

    assert np.array_equal(finals[0], finals[1])


def test_the_symmetries_fit_the_grid() -> None:
    """Shifts are whole cells, so a translation is a permutation rather than a fit."""
    translation, rotation = SYSTEM.symmetries
    assert isinstance(translation, Translation)
    assert isinstance(rotation, QuarterTurn)

    assert translation.shift == (8, -4)
    assert rotation.turns == 1


def test_parameters_reach_the_grid() -> None:
    system = build_fluid({"grid_size": 16, "domain_length": 4.0})

    assert system.grid.size == 16
    assert system.grid.length == 4.0


def test_an_unknown_system_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown fluid system parameters"):
        build_fluid({"viscosity": 0.1})


def test_a_bad_parameter_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be a number"):
        build_fluid({"grid_size": "large"})


def test_a_fractional_grid_size_is_rejected() -> None:
    with pytest.raises(ValidationError, match="whole number"):
        build_fluid({"grid_size": 32.5})


def test_the_default_grid_runs_on_the_target_machine() -> None:
    assert FluidSystem().grid.size == 64
