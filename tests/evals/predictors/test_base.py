from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.types import FieldSpec, State, StateSpec
from nnphysics.core.units import VELOCITY
from nnphysics.evals.predictors import (
    BROKEN_PREDICTORS,
    PREDICTORS,
    REFERENCE_NAME,
    PredictorContext,
    build_predictor,
    parse_spec,
)

SPEC = StateSpec(fields=(FieldSpec("q", (4,), VELOCITY),))


@dataclass(frozen=True, slots=True)
class Doubling:
    """Stands in for a reference solver: cheap, exact and obviously not the identity."""

    dt: float = 0.5

    @property
    def name(self) -> str:
        return "doubling"

    def step(self, state: State) -> State:
        return State(
            fields={key: array * 2.0 for key, array in state.fields.items()},
            time=state.time + self.dt,
        )


@dataclass(frozen=True, slots=True)
class Negation:
    """A stand in symmetry, so that a predictor which needs one can be built."""

    @property
    def name(self) -> str:
        return "negation"

    def apply(self, state: State) -> State:
        return State(fields={key: -array for key, array in state.fields.items()}, time=state.time)

    def apply_inverse(self, state: State) -> State:
        return self.apply(state)


def context(seed: int = 0, stream: str = "test") -> PredictorContext:
    return PredictorContext(
        reference=Doubling(),
        state_spec=SPEC,
        symmetries=(Negation(),),
        seed=seed,
        stream=stream,
    )


def state(value: float = 1.0) -> State:
    return State(fields={"q": np.full(4, value)}, time=0.0)


class TestParsing:
    def test_a_bare_name_carries_no_parameters(self) -> None:
        spec = parse_spec("persistence")

        assert spec.name == "persistence"
        assert spec.parameters == {}
        assert spec.text == "persistence"

    def test_parameters_are_read(self) -> None:
        spec = parse_spec("noise:scale=0.02,factor=3")

        assert spec.name == "noise"
        assert spec.parameters == {"scale": 0.02, "factor": 3.0}

    def test_the_text_is_kept_as_written(self) -> None:
        """A result file records what was asked for, not a re-rendering of it."""
        assert parse_spec("noise:scale=0.02").text == "noise:scale=0.02"

    def test_an_empty_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="names no predictor"):
            parse_spec(":scale=1")

    def test_a_parameter_without_a_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="malformed parameter"):
            parse_spec("noise:scale")

    def test_a_repeated_parameter_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="twice"):
            parse_spec("noise:scale=1,scale=2")

    def test_a_non_numeric_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non numeric"):
            parse_spec("noise:scale=loud")


class TestRegistry:
    def test_every_predictor_the_plan_names_is_registered(self) -> None:
        assert set(PREDICTORS.names()) == {REFERENCE_NAME, *BROKEN_PREDICTORS}

    @pytest.mark.parametrize("name", [REFERENCE_NAME, *BROKEN_PREDICTORS])
    def test_a_registered_name_builds_something_that_steps(self, name: str) -> None:
        built = build_predictor(name, context())

        assert built.dt > 0.0
        assert built.step(state()).time > 0.0

    def test_the_reference_is_the_solver_it_was_handed(self) -> None:
        assert build_predictor(REFERENCE_NAME, context()).name == "doubling"

    def test_an_unknown_name_is_rejected(self) -> None:
        with pytest.raises(UnknownNameError, match="unknown predictor"):
            build_predictor("oracle", context())

    def test_an_unknown_parameter_is_rejected(self) -> None:
        """A misspelled setting must not leave a fixture quietly less broken than intended."""
        with pytest.raises(ValidationError, match="does not accept the parameters"):
            build_predictor("noise:sale=0.1", context())

    def test_a_parameter_reaches_the_predictor(self) -> None:
        quiet = build_predictor("noise:scale=0.0", context())
        loud = build_predictor("noise:scale=0.5", context())
        truth = Doubling().step(state())

        assert np.allclose(quiet.step(state()).fields["q"], truth.fields["q"])
        assert not np.allclose(loud.step(state()).fields["q"], truth.fields["q"])

    def test_a_system_declaring_no_symmetry_cannot_have_one_broken(self) -> None:
        bare = PredictorContext(
            reference=Doubling(), state_spec=SPEC, symmetries=(), seed=0, stream="test"
        )

        with pytest.raises(ValidationError, match="declares a symmetry"):
            build_predictor("symmetry_break", bare)

    def test_a_symmetry_can_be_chosen_by_index(self) -> None:
        assert build_predictor("symmetry_break:index=0", context()).name == "symmetry_break"

    def test_an_index_outside_the_declared_symmetries_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="whole number in"):
            build_predictor("symmetry_break:index=5", context())


class TestDeterminism:
    def test_the_same_seed_and_stream_give_the_same_noise(self) -> None:
        first = build_predictor("noise", context(seed=7)).step(state())
        second = build_predictor("noise", context(seed=7)).step(state())

        assert np.array_equal(first.fields["q"], second.fields["q"])

    def test_a_different_seed_gives_different_noise(self) -> None:
        first = build_predictor("noise", context(seed=7)).step(state())
        second = build_predictor("noise", context(seed=8)).step(state())

        assert not np.array_equal(first.fields["q"], second.fields["q"])

    def test_a_different_stream_gives_different_noise(self) -> None:
        """Two rollouts of the same predictor must not see the same draw."""
        first = build_predictor("noise", context(stream="a")).step(state())
        second = build_predictor("noise", context(stream="b")).step(state())

        assert not np.array_equal(first.fields["q"], second.fields["q"])
