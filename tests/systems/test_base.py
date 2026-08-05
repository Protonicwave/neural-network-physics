import numpy as np
import pytest

from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.protocols import System
from nnphysics.systems import (
    SYSTEMS,
    build_system,
    check_parameter_names,
    float_parameter,
    int_parameter,
)


class TestRegistration:
    @pytest.mark.parametrize("name", ["nbody", "fluid"])
    def test_importing_the_layer_registers_both_systems(self, name: str) -> None:
        assert name in SYSTEMS

    @pytest.mark.parametrize("name", ["nbody", "fluid"])
    def test_a_registered_name_builds_a_system(self, name: str) -> None:
        assert isinstance(build_system(name), System)

    @pytest.mark.parametrize("name", ["nbody", "fluid"])
    def test_both_systems_are_driven_through_the_protocol_alone(self, name: str) -> None:
        """Nothing here knows which system it built, which is the whole point of the layer."""
        system = build_system(name)
        regime = system.regimes[0]
        state = system.initial_state(regime, np.random.default_rng(0))
        system.state_spec.validate(state)

        stepped = system.reference_predictor(regime, 1e-3).step(state)

        stepped.require_finite()
        assert [invariant.evaluate(stepped) for invariant in system.invariants(regime)]
        assert all(
            symmetry.apply_inverse(symmetry.apply(state)) is not None
            for symmetry in system.symmetries
        )

    def test_parameters_reach_the_factory(self) -> None:
        system = build_system("nbody", {"softening": 0.2})
        assert system.invariants(system.regimes[0])[0].evaluate is not None
        assert system.name == "nbody"

    def test_an_unknown_name_is_rejected(self) -> None:
        with pytest.raises(UnknownNameError, match="unknown system"):
            build_system("quantum")


class TestParameterReading:
    def test_a_known_name_passes(self) -> None:
        check_parameter_names({"a": 1.0}, ("a", "b"), context="test")

    def test_an_unknown_name_is_reported_with_the_accepted_ones(self) -> None:
        with pytest.raises(ValidationError, match=r"unknown test parameters \['c'\]"):
            check_parameter_names({"c": 1.0}, ("a", "b"), context="test")

    def test_a_float_is_read(self) -> None:
        assert float_parameter({"a": 2}, "a", context="test") == 2.0

    def test_a_missing_float_falls_back_to_its_default(self) -> None:
        assert float_parameter({}, "a", 3.5, context="test") == 3.5

    def test_a_missing_required_float_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires the parameter"):
            float_parameter({}, "a", context="test")

    @pytest.mark.parametrize("value", [True, "1.0", None])
    def test_a_non_numeric_value_is_rejected(self, value: object) -> None:
        with pytest.raises(ValidationError, match="must be a number"):
            float_parameter({"a": value}, "a", context="test")

    def test_an_integral_float_is_read_as_an_int(self) -> None:
        assert int_parameter({"a": 4.0}, "a", context="test") == 4

    def test_a_fractional_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="whole number"):
            int_parameter({"a": 4.5}, "a", context="test")

    def test_a_missing_int_falls_back_to_its_default(self) -> None:
        assert int_parameter({}, "a", 7, context="test") == 7
