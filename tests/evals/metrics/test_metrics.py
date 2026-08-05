"""Unit level behaviour of each metric, on trajectories built by hand.

The sentinel tests say the metrics catch what they must on real systems. These say what
each number means, on inputs small enough to work out on paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import Conservation
from nnphysics.core.types import Rollout, State, Trajectory
from nnphysics.core.units import DIMENSIONLESS, Dimension
from nnphysics.evals.metrics import (
    NEVER_REACHED,
    QUANTILES,
    DistributionDrift,
    InvariantDrift,
    MetricContext,
    OneStepError,
    RolloutErrorGrowth,
    SymmetryViolation,
)


@dataclass(frozen=True, slots=True)
class Sum:
    """A stand in invariant: the total of a field."""

    conservation: Conservation = Conservation.EXACT
    rtol: float = 1e-6

    @property
    def name(self) -> str:
        return "total"

    @property
    def dimension(self) -> Dimension:
        return DIMENSIONLESS

    def evaluate(self, state: State) -> float:
        return float(np.sum(state.fields["q"]))


@dataclass(frozen=True, slots=True)
class Negation:
    """A stand in symmetry that is its own inverse."""

    @property
    def name(self) -> str:
        return "negation"

    def apply(self, state: State) -> State:
        return State(fields={"q": -state.fields["q"]}, time=state.time)

    def apply_inverse(self, state: State) -> State:
        return self.apply(state)


@dataclass(frozen=True, slots=True)
class Halving:
    """Equivariant under negation: scaling commutes with a sign change."""

    dt: float = 1.0

    @property
    def name(self) -> str:
        return "halving"

    def step(self, state: State) -> State:
        return State(fields={"q": state.fields["q"] * 0.5}, time=state.time + self.dt)


@dataclass(frozen=True, slots=True)
class Shifting:
    """Not equivariant under negation: adding a constant does not commute with it."""

    dt: float = 1.0

    @property
    def name(self) -> str:
        return "shifting"

    def step(self, state: State) -> State:
        return State(fields={"q": state.fields["q"] + 1.0}, time=state.time + self.dt)


def make_trajectory(rows: list[list[float]]) -> Trajectory:
    array = np.asarray(rows, dtype=np.float64)
    return Trajectory(fields={"q": array}, times=np.arange(array.shape[0], dtype=np.float64))


def make_rollout(predicted: list[list[float]], reference: list[list[float]]) -> Rollout:
    return Rollout(make_trajectory(predicted), make_trajectory(reference), "test", "toy")


class TestOneStepError:
    def test_it_reads_the_first_step_and_not_the_rest(self) -> None:
        rollout = make_rollout([[1.0], [1.0], [99.0]], [[1.0], [2.0], [2.0]])

        scalars = OneStepError().compute(rollout).scalars

        assert scalars["error"] == pytest.approx(0.5)

    def test_it_reports_each_field(self) -> None:
        rollout = make_rollout([[1.0], [1.0]], [[1.0], [2.0]])

        assert "error.q" in OneStepError().compute(rollout).scalars

    def test_a_rollout_that_took_no_step_does_not_raise(self) -> None:
        """A predictor can fail on its first step, and the suite still has to report."""
        rollout = make_rollout([[1.0]], [[1.0]])

        assert OneStepError().compute(rollout).scalars["error"] == 0.0


class TestRolloutErrorGrowth:
    def test_the_curve_is_returned_as_plot_data(self) -> None:
        rollout = make_rollout([[1.0], [1.0], [1.0]], [[1.0], [2.0], [4.0]])

        result = RolloutErrorGrowth().compute(rollout)

        assert result.series["error"].shape == (3,)
        assert result.series["time"].shape == (3,)

    def test_the_summary_numbers_describe_the_curve(self) -> None:
        rollout = make_rollout([[1.0], [1.0], [1.0]], [[1.0], [2.0], [4.0]])

        scalars = RolloutErrorGrowth().compute(rollout).scalars

        assert scalars["error.final"] == pytest.approx(0.75)
        assert scalars["error.max"] == pytest.approx(0.75)
        assert scalars["duration"] == pytest.approx(2.0)

    def test_a_horizon_is_the_time_a_threshold_is_first_exceeded(self) -> None:
        rollout = make_rollout([[1.0], [1.0], [1.0]], [[1.0], [2.0], [4.0]])

        scalars = RolloutErrorGrowth(MetricContext(thresholds=(0.6,))).compute(rollout).scalars

        assert scalars["horizon.0.6"] == pytest.approx(2.0)

    def test_a_threshold_never_reached_is_reported_as_such(self) -> None:
        """Reporting the rollout's own length instead would read as a real horizon."""
        rollout = make_rollout([[1.0], [1.0]], [[1.0], [1.0]])

        scalars = RolloutErrorGrowth(MetricContext(thresholds=(0.5,))).compute(rollout).scalars

        assert scalars["horizon.0.5"] == NEVER_REACHED

    def test_recovering_by_luck_does_not_earn_a_longer_horizon(self) -> None:
        rollout = make_rollout([[9.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]])

        scalars = RolloutErrorGrowth(MetricContext(thresholds=(1.0,))).compute(rollout).scalars

        assert scalars["horizon.1"] == pytest.approx(0.0)


class TestInvariantDrift:
    def context(self, conservation: Conservation, rtol: float = 1e-6) -> MetricContext:
        return MetricContext(invariants=(Sum(conservation, rtol),))

    def test_a_trajectory_against_itself_scores_zero(self) -> None:
        rollout = make_rollout([[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]])

        result = InvariantDrift(self.context(Conservation.EXACT)).compute(rollout)

        assert result.scalars["worst_violation"] == 0.0

    def test_a_gap_from_the_reference_is_what_is_measured(self) -> None:
        rollout = make_rollout([[1.0], [1.5]], [[1.0], [1.0]])

        scalars = InvariantDrift(self.context(Conservation.EXACT)).compute(rollout).scalars

        assert scalars["total.excess"] == pytest.approx(0.5)
        assert scalars["total.violation"] > 1.0

    def test_a_predictor_is_not_blamed_for_the_solver_wobble(self) -> None:
        """Holding an invariant still while the reference wanders must not be a violation."""
        rollout = make_rollout([[1.0], [1.0], [1.0]], [[1.0], [1.001], [0.999]])

        scalars = InvariantDrift(self.context(Conservation.EXACT)).compute(rollout).scalars

        assert scalars["total.violation"] <= 1.0

    def test_a_decaying_quantity_may_fall_faster_than_the_reference(self) -> None:
        rollout = make_rollout([[1.0], [0.1]], [[1.0], [0.5]])

        scalars = InvariantDrift(self.context(Conservation.DECAYING)).compute(rollout).scalars

        assert scalars["total.violation"] == 0.0
        assert scalars["total.shortfall"] > 0.0

    def test_a_decaying_quantity_may_not_fall_more_slowly(self) -> None:
        rollout = make_rollout([[1.0], [0.9]], [[1.0], [0.5]])

        scalars = InvariantDrift(self.context(Conservation.DECAYING)).compute(rollout).scalars

        assert scalars["total.violation"] > 1.0

    def test_the_declared_tolerance_is_what_the_gap_is_measured_against(self) -> None:
        rollout = make_rollout([[1.0], [1.0001]], [[1.0], [1.0]])
        tight = InvariantDrift(self.context(Conservation.EXACT, 1e-8)).compute(rollout)
        loose = InvariantDrift(self.context(Conservation.EXACT, 1e-2)).compute(rollout)

        assert tight.scalars["total.violation"] > 1.0
        assert loose.scalars["total.violation"] <= 1.0

    def test_both_curves_are_returned_as_plot_data(self) -> None:
        rollout = make_rollout([[1.0], [2.0]], [[1.0], [1.0]])

        series = InvariantDrift(self.context(Conservation.EXACT)).compute(rollout).series

        assert np.allclose(series["total.predicted"], [1.0, 2.0])
        assert np.allclose(series["total.reference"], [1.0, 1.0])

    def test_a_system_declaring_no_invariant_reports_nothing_to_violate(self) -> None:
        rollout = make_rollout([[1.0], [9.0]], [[1.0], [1.0]])

        assert InvariantDrift().compute(rollout).scalars == {"worst_violation": 0.0}


class TestSymmetryViolation:
    def context(self, predictor: object) -> MetricContext:
        return MetricContext(symmetries=(Negation(),), predictor=predictor, symmetry_steps=3)  # type: ignore[arg-type]

    def test_an_equivariant_predictor_scores_zero(self) -> None:
        rollout = make_rollout([[1.0], [0.5], [0.25], [0.125]], [[1.0], [0.5], [0.25], [0.125]])

        result = SymmetryViolation(self.context(Halving())).compute(rollout)

        assert result.scalars["worst"] == pytest.approx(0.0, abs=1e-12)

    def test_a_predictor_that_does_not_commute_is_caught(self) -> None:
        rollout = make_rollout([[1.0], [2.0], [3.0], [4.0]], [[1.0], [1.0], [1.0], [1.0]])

        result = SymmetryViolation(self.context(Shifting())).compute(rollout)

        assert result.scalars["worst"] > 0.1
        assert result.scalars["negation.max"] == result.scalars["worst"]

    def test_the_violation_curve_is_returned(self) -> None:
        rollout = make_rollout([[1.0], [2.0], [3.0], [4.0]], [[1.0], [1.0], [1.0], [1.0]])

        result = SymmetryViolation(self.context(Shifting())).compute(rollout)

        assert result.series["negation"].shape == (4,)

    def test_the_tested_horizon_is_capped_by_the_setting(self) -> None:
        rollout = make_rollout([[1.0]] * 6, [[1.0]] * 6)

        result = SymmetryViolation(self.context(Halving())).compute(rollout)

        assert result.scalars["steps"] == 3.0

    def test_a_metric_with_no_predictor_says_what_it_needs(self) -> None:
        rollout = make_rollout([[1.0], [1.0]], [[1.0], [1.0]])

        with pytest.raises(ValidationError, match="needs the predictor"):
            SymmetryViolation(MetricContext(symmetries=(Negation(),))).compute(rollout)


class TestDistributionDrift:
    def test_a_trajectory_against_itself_scores_zero(self) -> None:
        rollout = make_rollout([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])

        assert DistributionDrift().compute(rollout).scalars["worst"] == 0.0

    def test_a_wider_distribution_is_caught_even_when_the_mean_agrees(self) -> None:
        """A mean and a variance can agree while the tails do not, so quantiles are read."""
        rollout = make_rollout([[-4.0, 4.0]], [[-1.0, 1.0]])

        assert DistributionDrift().compute(rollout).scalars["worst"] > 0.5

    def test_only_the_end_of_the_rollout_is_looked_at(self) -> None:
        """Where two trajectories have parted and only their statistics still compare."""
        early_only = make_rollout([[9.0, 9.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]])

        scored = DistributionDrift(MetricContext(distribution_window=0.5)).compute(early_only)

        assert scored.scalars["worst"] == 0.0

    def test_a_constant_field_is_compared_against_its_own_size(self) -> None:
        """Dividing by a spread of zero would report any perturbation of it as infinite."""
        rollout = make_rollout([[2.2, 2.2]], [[2.0, 2.0]])

        assert DistributionDrift().compute(rollout).scalars["q.quantile_distance"] == (
            pytest.approx(0.1)
        )

    def test_the_quantiles_are_returned_as_plot_data(self) -> None:
        rollout = make_rollout([[1.0, 2.0]], [[1.0, 2.0]])

        result = DistributionDrift().compute(rollout)

        assert result.series["q.predicted"].shape == (len(QUANTILES),)
        assert np.allclose(result.series["quantiles"], QUANTILES)
