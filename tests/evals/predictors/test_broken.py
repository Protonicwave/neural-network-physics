from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import FieldSpec, State, StateSpec
from nnphysics.core.units import LENGTH, MASS, VELOCITY
from nnphysics.evals.predictors import (
    EnergyInjection,
    LinearExtrapolation,
    NoiseInjection,
    Persistence,
    Substepped,
    SymmetryBreak,
    rate_fields,
)

SPEC = StateSpec(
    fields=(
        FieldSpec("x", (4,), LENGTH),
        FieldSpec("v", (4,), VELOCITY),
        FieldSpec("m", (4,), MASS),
    )
)


@dataclass(frozen=True, slots=True)
class Drifting:
    """Moves `x` at constant `v`, so linear extrapolation of it is exact."""

    dt: float = 1.0

    @property
    def name(self) -> str:
        return "drifting"

    def step(self, state: State) -> State:
        return State(
            fields={
                "x": state.fields["x"] + state.fields["v"] * self.dt,
                "v": state.fields["v"],
                "m": state.fields["m"],
            },
            time=state.time + self.dt,
        )


@dataclass(frozen=True, slots=True)
class Offset:
    """A stand in symmetry: shift `x` by a constant."""

    amount: float = 1.0

    @property
    def name(self) -> str:
        return "offset"

    def apply(self, state: State) -> State:
        return self._shift(state, self.amount)

    def apply_inverse(self, state: State) -> State:
        return self._shift(state, -self.amount)

    def _shift(self, state: State, amount: float) -> State:
        fields = dict(state.fields)
        fields["x"] = fields["x"] + amount
        return State(fields=fields, time=state.time)


def make_state(time: float = 0.0) -> State:
    return State(
        fields={
            "x": np.array([0.0, 1.0, 2.0, 3.0]),
            "v": np.array([1.0, 1.0, 1.0, 1.0]),
            "m": np.array([1.0, 1.0, 1.0, 1.0]),
        },
        time=time,
    )


class TestPersistence:
    def test_the_fields_come_back_unchanged(self) -> None:
        stepped = Persistence(0.5).step(make_state())

        assert np.array_equal(stepped.fields["x"], make_state().fields["x"])

    def test_only_the_clock_moves(self) -> None:
        assert Persistence(0.5).step(make_state(time=2.0)).time == pytest.approx(2.5)

    def test_a_step_of_zero_is_rejected(self) -> None:
        """A predictor that does not advance time produces no trajectory at all."""
        with pytest.raises(ValidationError, match="step size must be positive"):
            Persistence(0.0)


class TestLinearExtrapolation:
    def test_the_first_step_comes_from_the_reference(self) -> None:
        """Self starting from one state would reproduce it forever, which is persistence."""
        predictor = LinearExtrapolation(Drifting())

        stepped = predictor.step(make_state())

        assert np.allclose(stepped.fields["x"], [1.0, 2.0, 3.0, 4.0])

    def test_a_straight_line_is_continued_exactly(self) -> None:
        predictor = LinearExtrapolation(Drifting())
        first = make_state()
        second = predictor.step(first)

        third = predictor.step(second)

        assert np.allclose(third.fields["x"], [2.0, 3.0, 4.0, 5.0])
        assert third.time == pytest.approx(2.0)

    def test_it_does_not_collapse_into_persistence(self) -> None:
        predictor = LinearExtrapolation(Drifting())
        state = make_state()
        for _ in range(4):
            state = predictor.step(state)

        assert np.allclose(state.fields["x"], [4.0, 5.0, 6.0, 7.0])

    def test_a_new_rollout_is_detected_and_started_afresh(self) -> None:
        """The same instance may score several initial conditions without carrying memory."""
        predictor = LinearExtrapolation(Drifting())
        predictor.step(predictor.step(make_state()))

        restarted = predictor.step(make_state())

        assert np.allclose(restarted.fields["x"], [1.0, 2.0, 3.0, 4.0])

    def test_resetting_forgets_the_previous_state(self) -> None:
        predictor = LinearExtrapolation(Drifting())
        second = predictor.step(make_state())
        predictor.reset()

        third = predictor.step(second)

        assert np.allclose(third.fields["x"], [2.0, 3.0, 4.0, 5.0])


class TestNoiseInjection:
    def test_noise_is_added_to_every_field(self) -> None:
        predictor = NoiseInjection(Drifting(), 0.1, np.random.default_rng(0))

        stepped = predictor.step(make_state())

        assert not np.allclose(stepped.fields["x"], Drifting().step(make_state()).fields["x"])
        assert not np.allclose(stepped.fields["m"], make_state().fields["m"])

    def test_the_draw_has_no_mean(self) -> None:
        """So that a system which forbids a mean offset still accepts the state."""
        predictor = NoiseInjection(Drifting(), 1.0, np.random.default_rng(0))

        stepped = predictor.step(make_state())

        added = stepped.fields["m"] - make_state().fields["m"]
        assert float(np.mean(added)) == pytest.approx(0.0, abs=1e-12)

    def test_a_scale_of_zero_leaves_the_solver_alone(self) -> None:
        predictor = NoiseInjection(Drifting(), 0.0, np.random.default_rng(0))

        stepped = predictor.step(make_state())

        assert np.allclose(stepped.fields["x"], Drifting().step(make_state()).fields["x"])

    def test_a_negative_scale_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be negative"):
            NoiseInjection(Drifting(), -0.1, np.random.default_rng(0))


class TestEnergyInjection:
    def test_only_the_declared_rates_are_amplified(self) -> None:
        predictor = EnergyInjection(Drifting(), rate_fields(SPEC), 0.5)

        stepped = predictor.step(make_state())

        assert np.allclose(stepped.fields["v"], make_state().fields["v"] * 1.5)
        assert np.allclose(stepped.fields["m"], make_state().fields["m"])
        assert np.allclose(stepped.fields["x"], Drifting().step(make_state()).fields["x"])

    def test_a_rate_is_recognised_by_its_declared_dimension(self) -> None:
        assert rate_fields(SPEC) == ("v",)

    def test_a_system_with_no_rate_field_is_rejected(self) -> None:
        """Fail loudly: the dimensions cannot say where such a system's energy lives."""
        spec = StateSpec(fields=(FieldSpec("x", (2,), LENGTH),))

        with pytest.raises(ValidationError, match="no declared field carries a power of time"):
            rate_fields(spec)

    def test_amplification_compounds_over_a_rollout(self) -> None:
        """One step is accurate to the factor; many steps are not accurate at all."""
        predictor = EnergyInjection(Drifting(), ("v",), 0.1)
        state = make_state()
        for _ in range(10):
            state = predictor.step(state)

        assert float(state.fields["v"][0]) == pytest.approx(1.1**10)

    def test_an_amplification_that_would_invert_a_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="would invert a field"):
            EnergyInjection(Drifting(), ("v",), -1.5)

    def test_no_field_to_amplify_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one field"):
            EnergyInjection(Drifting(), (), 0.1)


class TestSymmetryBreak:
    def test_the_transformation_is_applied_after_the_step(self) -> None:
        predictor = SymmetryBreak(Drifting(), Offset(2.0))

        stepped = predictor.step(make_state())

        assert np.allclose(stepped.fields["x"], Drifting().step(make_state()).fields["x"] + 2.0)

    def test_it_accumulates(self) -> None:
        predictor = SymmetryBreak(Drifting(), Offset(2.0))
        state = predictor.step(predictor.step(make_state()))

        assert np.allclose(state.fields["x"], [6.0, 7.0, 8.0, 9.0])


class TestSubstepped:
    def test_the_outer_step_is_the_inner_one_times_the_count(self) -> None:
        assert Substepped(Drifting(0.25), 4).dt == pytest.approx(1.0)

    def test_it_takes_every_inner_step(self) -> None:
        stepped = Substepped(Drifting(1.0), 3).step(make_state())

        assert np.allclose(stepped.fields["x"], [3.0, 4.0, 5.0, 6.0])
        assert stepped.time == pytest.approx(3.0)

    def test_it_is_named_for_the_role_it_plays(self) -> None:
        assert Substepped(Drifting(), 2).name == "reference"

    def test_a_substep_count_below_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="substeps must be positive"):
            Substepped(Drifting(), 0)
