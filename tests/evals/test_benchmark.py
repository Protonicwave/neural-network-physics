from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest

from nnphysics.core.errors import NumericalError, ValidationError
from nnphysics.core.types import State
from nnphysics.evals.benchmark import STABILITY_LIMIT, benchmark_predictor

STEPS = 8


def state() -> State:
    return State(fields={"q": np.ones(4)}, time=0.0)


@dataclass
class Counting:
    """A predictor that records how many times it was stepped."""

    dt: float = 0.5
    calls: int = 0
    delay: float = 0.0

    @property
    def name(self) -> str:
        return "counting"

    def step(self, current: State) -> State:
        self.calls += 1
        if self.delay:
            deadline = time.perf_counter() + self.delay
            while time.perf_counter() < deadline:
                pass
        return State(fields=dict(current.fields), time=current.time + self.dt)


@dataclass(frozen=True, slots=True)
class Failing:
    """A predictor that refuses the first step it is given."""

    dt: float = 0.5

    @property
    def name(self) -> str:
        return "failing"

    def step(self, current: State) -> State:
        raise NumericalError(f"nothing to time at t={current.time}")


class TestWhatIsTimed:
    def test_the_warmup_runs_and_is_not_counted(self) -> None:
        predictor = Counting()

        timing = benchmark_predictor(predictor, state(), steps_per_trial=STEPS, trials=3, warmup=2)

        assert predictor.calls == STEPS * 5
        assert timing.trials == 3
        assert timing.warmup == 2

    def test_a_slower_predictor_is_measured_as_slower(self) -> None:
        """The measure has to respond to the thing it measures, not merely produce a number."""
        quick = benchmark_predictor(Counting(), state(), steps_per_trial=STEPS, trials=3, warmup=1)
        slow = benchmark_predictor(
            Counting(delay=0.001), state(), steps_per_trial=STEPS, trials=3, warmup=1
        )

        assert slow.seconds_per_step > quick.seconds_per_step

    def test_the_time_is_per_step_rather_than_per_trial(self) -> None:
        short = benchmark_predictor(
            Counting(delay=0.001), state(), steps_per_trial=2, trials=3, warmup=1
        )
        long = benchmark_predictor(
            Counting(delay=0.001), state(), steps_per_trial=8, trials=3, warmup=1
        )

        assert long.seconds_per_step == pytest.approx(short.seconds_per_step, rel=0.5)


class TestTheSpread:
    def test_a_steady_predictor_reads_as_stable(self) -> None:
        timing = benchmark_predictor(
            Counting(delay=0.002), state(), steps_per_trial=STEPS, trials=7, warmup=2
        )

        assert timing.relative_spread <= STABILITY_LIMIT
        assert timing.stable

    def test_the_median_sits_between_the_fastest_and_the_slowest_trial(self) -> None:
        """A best of seven would be a measurement of how quiet the machine briefly was."""
        timing = benchmark_predictor(
            Counting(delay=0.001), state(), steps_per_trial=STEPS, trials=7, warmup=1
        )

        assert timing.fastest <= timing.seconds_per_step <= timing.slowest

    def test_the_thread_count_is_recorded_as_given(self) -> None:
        timing = benchmark_predictor(
            Counting(), state(), steps_per_trial=STEPS, trials=2, warmup=0, threads=3
        )

        assert timing.threads == 3
        assert timing.batch_size == 1


class TestItRefusesWhatItCannotMeasure:
    def test_a_predictor_that_takes_no_step_is_an_error_rather_than_the_best_time(self) -> None:
        """Dividing by zero steps would put a predictor that fails at the top of the table."""
        with pytest.raises(ValidationError, match="completed no step"):
            benchmark_predictor(Failing(), state(), steps_per_trial=STEPS, trials=2, warmup=0)

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"trials": 0}, "at least one trial"),
            ({"steps_per_trial": 0}, "at least one step"),
            ({"warmup": -1}, "must not be negative"),
            ({"threads": 0}, "thread count must be positive"),
        ],
    )
    def test_a_setting_that_makes_no_sense_is_refused(
        self, settings: dict[str, int], message: str
    ) -> None:
        arguments: dict[str, int] = {"steps_per_trial": STEPS, "trials": 2, "warmup": 0}

        with pytest.raises(ValidationError, match=message):
            benchmark_predictor(Counting(), state(), **{**arguments, **settings})
