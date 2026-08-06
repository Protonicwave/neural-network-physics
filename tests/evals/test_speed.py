from __future__ import annotations

import numpy as np
import pytest

from nnphysics.core.errors import NumericalError, ValidationError
from nnphysics.core.types import Regime, State, Trajectory
from nnphysics.data.manifest import Split
from nnphysics.evals.runner import EvaluationCase
from nnphysics.evals.speed import (
    NEVER_PAYS,
    MatchedSpeedup,
    SpeedPoint,
    cost_accounting,
    matched_speedup,
    measure_point,
    substep_ladder,
)


class Refusing:
    """A solver run so far past its stability limit that it cannot take one step."""

    dt = 0.5

    @property
    def name(self) -> str:
        return "refusing"

    def step(self, current: State) -> State:
        raise NumericalError(f"unstable at t={current.time}")


class Stalling:
    """Takes a few steps and then refuses, which is what a diverging surrogate does."""

    dt = 0.5

    def __init__(self, after: int) -> None:
        self._after = after
        self._taken = 0

    @property
    def name(self) -> str:
        return "stalling"

    def step(self, current: State) -> State:
        if self._taken >= self._after:
            self._taken = 0
            raise NumericalError(f"stopped at t={current.time}")
        self._taken += 1
        return State(
            fields={name: array * 1.5 for name, array in current.fields.items()},
            time=current.time + self.dt,
        )


def case() -> EvaluationCase:
    """One initial condition with four steps of ground truth after it."""
    times = np.arange(5, dtype=np.float64) * Refusing.dt
    return EvaluationCase(
        trajectory_id="hot/00000",
        regime=Regime(name="hot", parameters={}),
        split=Split.TEST,
        reference=Trajectory(fields={"q": np.ones((5, 3))}, times=times),
    )


def point(substeps: int, error: float, seconds: float, *, kind: str = "solver") -> SpeedPoint:
    """One measured point, with the fields a comparison does not read left benign."""
    return SpeedPoint(
        label=f"{kind}:{substeps}",
        predictor="reference" if kind == "solver" else "surrogate",
        kind=kind,
        substeps=substeps,
        error=error,
        seconds_per_step=seconds,
        iqr=0.0,
        relative_spread=0.0,
        stable=True,
        completed=True,
    )


# A solver that is cheaper and worse as it is coarsened, and exact at the setting that
# produced ground truth. That shape is what a matched comparison reads off.
LADDER = (
    point(1, 0.8, 0.001),
    point(2, 0.4, 0.002),
    point(5, 0.1, 0.005),
    point(10, 0.0, 0.010),
)


class TestTheLadder:
    def test_it_is_every_divisor_of_the_substeps_the_data_used(self) -> None:
        assert substep_ladder(10) == (1, 2, 5, 10)
        assert substep_ladder(8) == (1, 2, 4, 8)

    def test_a_prime_count_gives_the_two_rungs_it_has(self) -> None:
        assert substep_ladder(7) == (1, 7)

    def test_one_substep_is_a_ladder_of_one_rung(self) -> None:
        assert substep_ladder(1) == (1,)

    def test_a_count_below_one_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            substep_ladder(0)


class TestARungThatCannotRun:
    def test_a_setting_that_takes_no_step_is_reported_as_nothing_rather_than_a_number(
        self,
    ) -> None:
        """A solver several times past its stability limit is not a worse setting.

        It is one that does not run, and putting an accuracy against it would invent one.
        """
        measured = measure_point(
            [Refusing()],
            [case()],
            steps=4,
            label="reference:substeps=1",
            kind="solver",
            substeps=1,
            threads=1,
            trials=1,
            warmup=0,
            divergence_factor=1.0e3,
        )

        assert measured is None


class TestARolloutThatStopsPartWay:
    def test_it_is_scored_over_the_steps_it_managed(self) -> None:
        """A diverging surrogate is the ordinary case on the fluid, not an edge one.

        The error is taken over the prefix that ran, and the point records that it did
        not finish. Truncating ground truth to match is the step this exercises.
        """
        measured = measure_point(
            [Stalling(after=2)],
            [case()],
            steps=4,
            label="stalling",
            kind="surrogate",
            substeps=0,
            threads=1,
            trials=1,
            warmup=0,
            steps_per_trial=4,
            divergence_factor=1.0e3,
        )

        assert measured is not None
        assert not measured.completed
        assert measured.error > 0.0


class TestMatchedAccuracy:
    def test_the_cheapest_rung_that_is_accurate_enough_is_the_one_compared_against(
        self,
    ) -> None:
        """Not the setting the data was generated with.

        That is exactly the comparison the phase exists to replace.
        """
        matched = matched_speedup(point(0, 0.3, 0.004, kind="surrogate"), LADDER)

        assert matched.matched_substeps == 5
        assert matched.matched_error == 0.1
        assert matched.speedup == pytest.approx(1.25)

    def test_a_surrogate_slower_than_its_matched_solver_reads_below_one(self) -> None:
        matched = matched_speedup(point(0, 0.3, 0.020, kind="surrogate"), LADDER)

        assert matched.speedup < 1.0

    def test_an_inaccurate_surrogate_is_bracketed_by_the_coarse_end_of_the_ladder(
        self,
    ) -> None:
        matched = matched_speedup(point(0, 0.5, 0.004, kind="surrogate"), LADDER)

        assert matched.matched_substeps == 2
        assert matched.bracketed

    def test_a_surrogate_worse_than_the_coarsest_rung_is_not_bracketed(self) -> None:
        """The one case the number is an upper bound rather than a measurement.

        The solver could be run coarser than anything measured and still match, so the
        cost being compared against is higher than the true matched cost.
        """
        matched = matched_speedup(point(0, 0.9, 0.004, kind="surrogate"), LADDER)

        assert matched.matched_substeps == 1
        assert not matched.bracketed

    def test_a_surrogate_more_accurate_than_every_coarsening_matches_the_top_rung(
        self,
    ) -> None:
        matched = matched_speedup(point(0, 0.05, 0.004, kind="surrogate"), LADDER)

        assert matched.matched_substeps == 10
        assert matched.bracketed

    def test_a_ladder_that_never_reaches_the_accuracy_is_refused(self) -> None:
        """Cannot happen with a real ladder, whose top rung produced ground truth.

        Worth refusing rather than guessing at if a ladder is assembled by hand.
        """
        with pytest.raises(ValidationError, match="no rung"):
            matched_speedup(point(0, 0.05, 0.004, kind="surrogate"), (point(1, 0.8, 0.001),))

    def test_an_empty_ladder_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="needs a solver ladder"):
            matched_speedup(point(0, 0.3, 0.004, kind="surrogate"), ())


def matched(solver: float, surrogate: float) -> MatchedSpeedup:
    """A comparison carrying only the two costs the accounting reads."""
    return MatchedSpeedup(
        predictor="surrogate",
        error=0.3,
        seconds_per_step=surrogate,
        matched_substeps=5,
        matched_error=0.1,
        matched_seconds_per_step=solver,
        speedup=solver / surrogate,
        bracketed=True,
    )


class TestCostAccounting:
    def test_the_break_even_count_repays_both_one_off_costs(self) -> None:
        cost = cost_accounting(
            matched(0.010, 0.005),
            training_seconds=100.0,
            generation_seconds=50.0,
            steps_per_rollout=100,
        )

        assert cost.saving_per_rollout == pytest.approx(0.5)
        assert cost.break_even_rollouts == 300.0

    def test_it_rounds_up_because_a_part_rollout_repays_nothing(self) -> None:
        cost = cost_accounting(
            matched(0.010, 0.005),
            training_seconds=100.25,
            generation_seconds=0.0,
            steps_per_rollout=100,
        )

        assert cost.break_even_rollouts == 201.0

    def test_a_surrogate_slower_than_the_solver_never_pays(self) -> None:
        cost = cost_accounting(
            matched(0.005, 0.010),
            training_seconds=100.0,
            generation_seconds=0.0,
            steps_per_rollout=100,
        )

        assert cost.saving_per_rollout < 0.0
        assert cost.break_even_rollouts == NEVER_PAYS

    def test_a_surrogate_exactly_as_fast_never_pays_either(self) -> None:
        """Zero saving repaid into any one off cost is not a large number, it is no number."""
        cost = cost_accounting(
            matched(0.005, 0.005),
            training_seconds=100.0,
            generation_seconds=0.0,
            steps_per_rollout=100,
        )

        assert cost.break_even_rollouts == NEVER_PAYS

    def test_a_free_surrogate_breaks_even_on_its_first_rollout(self) -> None:
        cost = cost_accounting(
            matched(0.010, 0.005),
            training_seconds=0.0,
            generation_seconds=0.0,
            steps_per_rollout=100,
        )

        assert cost.break_even_rollouts == 0.0

    @pytest.mark.parametrize(
        ("settings", "message"),
        [
            ({"training_seconds": -1.0}, "cannot be negative"),
            ({"generation_seconds": -1.0}, "cannot be negative"),
            ({"steps_per_rollout": 0}, "at least one step"),
        ],
    )
    def test_a_cost_that_makes_no_sense_is_refused(
        self, settings: dict[str, float], message: str
    ) -> None:
        arguments: dict[str, float] = {
            "training_seconds": 1.0,
            "generation_seconds": 1.0,
            "steps_per_rollout": 10,
        }

        with pytest.raises(ValidationError, match=message):
            cost_accounting(matched(0.010, 0.005), **{**arguments, **settings})  # type: ignore[arg-type]
