from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nnphysics.core.errors import NumericalError, ValidationError
from nnphysics.core.types import State
from nnphysics.evals.rollout import StopReason, roll_out, roll_out_many


def make_state(value: float = 1.0, time: float = 0.0) -> State:
    return State(fields={"q": np.full(3, value)}, time=time)


@dataclass(frozen=True, slots=True)
class Scaling:
    """Multiplies its state by a factor each step, so a rollout can be steered."""

    factor: float
    dt: float = 1.0

    @property
    def name(self) -> str:
        return "scaling"

    def step(self, state: State) -> State:
        return State(
            fields={key: array * self.factor for key, array in state.fields.items()},
            time=state.time + self.dt,
        )


@dataclass(frozen=True, slots=True)
class Adding:
    """Adds a fixed amount each step, so how far a rollout gets depends on where it began."""

    amount: float
    dt: float = 1.0

    @property
    def name(self) -> str:
        return "adding"

    def step(self, state: State) -> State:
        return State(
            fields={key: array + self.amount for key, array in state.fields.items()},
            time=state.time + self.dt,
        )


@dataclass(frozen=True, slots=True)
class Stuck:
    """Advances nothing, the clock included."""

    dt: float = 1.0

    @property
    def name(self) -> str:
        return "stuck"

    def step(self, state: State) -> State:
        return state


@dataclass(frozen=True, slots=True)
class Raising:
    """Refuses to predict after a given number of steps."""

    after: int
    dt: float = 1.0

    @property
    def name(self) -> str:
        return "raising"

    def step(self, state: State) -> State:
        if state.time >= self.after:
            raise NumericalError("the state stopped being representable")
        return State(fields=dict(state.fields), time=state.time + self.dt)


class TestCompletion:
    def test_a_well_behaved_rollout_completes(self) -> None:
        result = roll_out(Scaling(1.0), make_state(), 5)

        assert result.stop_reason is StopReason.COMPLETED
        assert result.completed
        assert result.steps_completed == 5
        assert len(result.trajectory) == 6

    def test_the_initial_state_is_the_first_recorded_one(self) -> None:
        result = roll_out(Scaling(2.0), make_state(3.0), 2)

        assert result.trajectory.fields["q"][0] == pytest.approx(3.0)
        assert result.trajectory.fields["q"][2] == pytest.approx(12.0)

    def test_the_predictor_is_named(self) -> None:
        assert roll_out(Scaling(1.0), make_state(), 1).predictor == "scaling"

    def test_time_is_measured_and_not_negative(self) -> None:
        result = roll_out(Scaling(1.0), make_state(), 4)

        assert result.seconds >= 0.0
        assert result.seconds_per_step == pytest.approx(result.seconds / 4)

    def test_a_rollout_that_took_no_step_reports_no_cost_per_step(self) -> None:
        assert roll_out(Stuck(), make_state(), 3).seconds_per_step == 0.0

    def test_at_least_one_step_is_required(self) -> None:
        with pytest.raises(ValidationError, match="at least one step"):
            roll_out(Scaling(1.0), make_state(), 0)


class TestStoppingEarly:
    def test_divergence_is_caught_before_the_numbers_stop_being_finite(self) -> None:
        result = roll_out(Scaling(10.0), make_state(), 20, divergence_factor=1e3)

        assert result.stop_reason is StopReason.DIVERGED
        assert not result.completed
        assert result.steps_completed == 3
        assert np.isfinite(result.trajectory.fields["q"]).all()

    def test_a_non_finite_state_is_caught(self) -> None:
        result = roll_out(Scaling(np.inf), make_state(), 4)

        assert result.stop_reason is StopReason.NON_FINITE
        assert result.steps_completed == 0

    def test_a_predictor_that_does_not_advance_the_clock_is_caught(self) -> None:
        result = roll_out(Stuck(), make_state(), 4)

        assert result.stop_reason is StopReason.NOT_ADVANCING
        assert len(result.trajectory) == 1

    def test_a_predictor_that_raises_is_recorded_rather_than_propagated(self) -> None:
        result = roll_out(Raising(after=3), make_state(), 10)

        assert result.stop_reason is StopReason.FAILED
        assert result.steps_completed == 3
        assert "representable" in result.detail

    def test_the_divergence_check_can_be_turned_off(self) -> None:
        result = roll_out(Scaling(10.0), make_state(), 6, divergence_factor=0.0)

        assert result.stop_reason is StopReason.COMPLETED

    def test_a_state_of_zeros_does_not_look_divergent(self) -> None:
        result = roll_out(Scaling(1.0), make_state(0.0), 4)

        assert result.stop_reason is StopReason.COMPLETED


class TestBatches:
    def test_every_initial_condition_is_rolled_out_in_order(self) -> None:
        results = roll_out_many(Scaling(2.0), [make_state(1.0), make_state(3.0)], 2)

        assert [float(result.trajectory.fields["q"][-1][0]) for result in results] == [4.0, 12.0]

    def test_an_empty_batch_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty set of initial conditions"):
            roll_out_many(Scaling(1.0), [], 3)

    def test_one_bad_initial_condition_does_not_stop_the_others(self) -> None:
        """Divergence is judged against the state a rollout started from, not an absolute."""
        results = roll_out_many(Adding(1.0), [make_state(1.0), make_state(1e-9)], 4)

        assert results[0].stop_reason is StopReason.COMPLETED
        assert results[1].stop_reason is StopReason.DIVERGED
