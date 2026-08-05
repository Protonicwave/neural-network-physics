from __future__ import annotations

import numpy as np
import pytest

from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.types import Trajectory
from nnphysics.evals.metrics import DEFAULT_METRICS, METRICS, MetricContext, build_metrics
from nnphysics.evals.metrics.base import relative_error


def trajectory(values: list[list[float]], scale: float = 1.0) -> Trajectory:
    array = np.asarray(values, dtype=np.float64) * scale
    return Trajectory(fields={"q": array}, times=np.arange(array.shape[0], dtype=np.float64))


class TestRegistry:
    def test_every_metric_the_plan_names_is_registered(self) -> None:
        assert set(METRICS.names()) == set(DEFAULT_METRICS)

    def test_names_are_resolved_in_the_order_given(self) -> None:
        built = build_metrics(["rollout_error", "one_step_error"], MetricContext())

        assert [metric.name for metric in built] == ["rollout_error", "one_step_error"]

    def test_an_unknown_metric_is_rejected(self) -> None:
        with pytest.raises(UnknownNameError, match="unknown metric"):
            build_metrics(["accuracy"], MetricContext())

    def test_an_empty_suite_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one metric"):
            build_metrics([], MetricContext())

    def test_a_repeated_metric_is_rejected(self) -> None:
        """Two entries would be averaged into one column, which reads as one run of it."""
        with pytest.raises(ValidationError, match="more than once"):
            build_metrics(["rollout_error", "rollout_error"], MetricContext())


class TestContext:
    def test_a_metric_that_needs_the_predictor_says_so(self) -> None:
        with pytest.raises(ValidationError, match="needs the predictor"):
            MetricContext().require_predictor("symmetry_violation")

    def test_a_non_positive_threshold_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="thresholds must be positive"):
            MetricContext(thresholds=(0.1, 0.0))

    def test_a_symmetry_horizon_below_one_step_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one step"):
            MetricContext(symmetry_steps=0)

    @pytest.mark.parametrize("window", [0.0, 1.5])
    def test_a_window_outside_the_unit_interval_is_rejected(self, window: float) -> None:
        with pytest.raises(ValidationError, match="fraction in"):
            MetricContext(distribution_window=window)


class TestRelativeError:
    def test_a_trajectory_against_itself_is_exactly_zero(self) -> None:
        reference = trajectory([[1.0, 2.0], [3.0, 4.0]])

        per_field, aggregate = relative_error(reference, reference)

        assert np.array_equal(aggregate, np.zeros(2))
        assert np.array_equal(per_field["q"], np.zeros(2))

    def test_the_error_is_normalised_by_the_reference(self) -> None:
        """An error equal to the signal reads as one, whatever the units are."""
        reference = trajectory([[3.0, 4.0]])
        predicted = trajectory([[6.0, 8.0]])

        _, aggregate = relative_error(predicted, reference)

        assert aggregate[0] == pytest.approx(1.0)

    def test_the_same_relative_error_reads_the_same_at_any_scale(self) -> None:
        small = relative_error(trajectory([[2.0]]), trajectory([[1.0]]))[1]
        large = relative_error(trajectory([[2.0]], 1e6), trajectory([[1.0]], 1e6))[1]

        assert small[0] == pytest.approx(large[0])

    def test_fields_are_weighted_equally(self) -> None:
        reference = Trajectory(
            fields={"a": np.ones((1, 2)), "b": np.ones((1, 2))}, times=np.zeros(1)
        )
        predicted = Trajectory(
            fields={"a": np.ones((1, 2)) * 2.0, "b": np.ones((1, 2))}, times=np.zeros(1)
        )

        per_field, aggregate = relative_error(predicted, reference)

        assert per_field["a"][0] == pytest.approx(1.0)
        assert per_field["b"][0] == pytest.approx(0.0)
        assert aggregate[0] == pytest.approx(np.sqrt(0.5))

    def test_trajectories_of_different_lengths_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot compare trajectories of"):
            relative_error(trajectory([[1.0]]), trajectory([[1.0], [2.0]]))

    def test_trajectories_with_different_fields_are_rejected(self) -> None:
        other = Trajectory(fields={"r": np.ones((1, 1))}, times=np.zeros(1))

        with pytest.raises(ValidationError, match="cannot compare trajectories with fields"):
            relative_error(trajectory([[1.0]]), other)
