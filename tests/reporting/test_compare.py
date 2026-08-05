from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.evals.metrics import NEVER_REACHED
from nnphysics.reporting.compare import Delta, Verdict, compare_records, compare_series
from nnphysics.reporting.record import RunRecord

Factory = Callable[..., RunRecord]

FINAL_ERROR = ("test", "reference", "rollout_error", "error.final")


def delta_for(deltas: Sequence[Delta], split: str, predictor: str, metric: str, key: str) -> Delta:
    """The one delta a test is about, named rather than indexed."""
    for delta in deltas:
        if (delta.split, delta.predictor, delta.metric, delta.key) == (
            split,
            predictor,
            metric,
            key,
        ):
            return delta
    raise AssertionError(f"no delta for {metric}.{key}")


class TestThreeCases:
    def test_a_run_with_smaller_errors_is_an_improvement(self, make_record: Factory) -> None:
        comparison = compare_records(make_record(), make_record(scale=0.5))

        delta = delta_for(comparison.deltas, *FINAL_ERROR)

        assert delta.verdict is Verdict.IMPROVED
        assert delta.change < 0.0

    def test_a_run_with_larger_errors_is_a_regression(self, make_record: Factory) -> None:
        comparison = compare_records(make_record(), make_record(scale=2.0))

        delta = delta_for(comparison.deltas, *FINAL_ERROR)

        assert delta.verdict is Verdict.REGRESSED
        assert delta.regressed

    def test_a_run_with_the_same_numbers_is_no_change(self, make_record: Factory) -> None:
        comparison = compare_records(make_record(), make_record())

        assert comparison.of(Verdict.IMPROVED) == ()
        assert comparison.of(Verdict.REGRESSED) == ()
        assert delta_for(comparison.deltas, *FINAL_ERROR).verdict is Verdict.UNCHANGED

    def test_a_regression_is_counted_and_listed_worst_first(self, make_record: Factory) -> None:
        comparison = compare_records(make_record(), make_record(scale=3.0))

        assert comparison.summary[Verdict.REGRESSED] == len(comparison.regressions)
        relatives = [abs(delta.relative) for delta in comparison.regressions]
        assert relatives == sorted(relatives, reverse=True)


class TestThreshold:
    def test_a_change_under_the_threshold_is_called_unchanged(self, make_record: Factory) -> None:
        comparison = compare_records(make_record(), make_record(scale=1.02), threshold=0.05)

        assert delta_for(comparison.deltas, *FINAL_ERROR).verdict is Verdict.UNCHANGED

    def test_the_same_change_over_a_tighter_threshold_is_a_regression(
        self, make_record: Factory
    ) -> None:
        comparison = compare_records(make_record(), make_record(scale=1.02), threshold=0.001)

        assert delta_for(comparison.deltas, *FINAL_ERROR).verdict is Verdict.REGRESSED

    def test_a_negative_threshold_is_refused(self, make_record: Factory) -> None:
        with pytest.raises(ValidationError, match="cannot be negative"):
            compare_records(make_record(), make_record(), threshold=-0.1)


class TestDirection:
    def test_a_scalar_with_no_direction_is_reported_without_a_verdict(
        self, make_record: Factory
    ) -> None:
        comparison = compare_records(make_record(), make_record(scale=2.0))

        delta = delta_for(comparison.deltas, "test", "reference", "rollout_error", "duration")

        assert delta.verdict is Verdict.UNDIRECTED

    def test_a_slower_predictor_is_a_regression_even_with_the_same_errors(
        self, make_record: Factory
    ) -> None:
        comparison = compare_records(make_record(), make_record(seconds=0.02))

        delta = delta_for(comparison.deltas, "test", "reference", "timing", "seconds_per_step")

        assert delta.verdict is Verdict.REGRESSED


class TestSentinels:
    def test_a_horizon_that_is_never_reached_beats_one_that_is(self, make_record: Factory) -> None:
        baseline = make_record()
        candidate = make_record()
        comparison = compare_records(baseline, candidate)

        delta = delta_for(comparison.deltas, "test", "reference", "rollout_error", "horizon.1")

        assert delta.baseline == NEVER_REACHED
        assert delta.verdict is Verdict.UNCHANGED
        assert delta.change == 0.0

    def test_losing_a_horizon_that_was_never_crossed_is_a_regression(
        self, make_record: Factory
    ) -> None:
        baseline = make_record()
        candidate = make_record()
        entry = candidate.evaluation.results[0]
        metric = entry.metrics[1]
        crossed = entry.model_copy(
            update={
                "metrics": (
                    entry.metrics[0],
                    metric.model_copy(update={"scalars": {**metric.scalars, "horizon.1": 0.02}}),
                    *entry.metrics[2:],
                )
            }
        )
        worse = candidate.model_copy(
            update={
                "evaluation": candidate.evaluation.model_copy(
                    update={"results": (crossed, *candidate.evaluation.results[1:])}
                )
            }
        )

        comparison = compare_records(baseline, worse)
        delta = delta_for(comparison.deltas, "test", "reference", "rollout_error", "horizon.1")

        assert delta.verdict is Verdict.REGRESSED
        assert delta.change == -math.inf


class TestScope:
    def test_only_scalars_both_runs_carry_are_compared(self, make_record: Factory) -> None:
        comparison = compare_records(
            make_record(predictors=("reference", "persistence")),
            make_record(predictors=("reference",)),
        )

        assert {delta.predictor for delta in comparison.deltas} == {"reference"}

    def test_two_systems_cannot_be_compared(self, make_record: Factory) -> None:
        other = make_record()
        elsewhere = other.model_copy(
            update={"evaluation": other.evaluation.model_copy(update={"system": "fluid"})}
        )

        with pytest.raises(ValidationError, match="cannot compare a run on"):
            compare_records(make_record(), elsewhere)

    def test_it_records_which_runs_it_compared(self, make_record: Factory) -> None:
        comparison = compare_records(make_record(run_id="aaaa"), make_record(run_id="bbbb"))

        assert comparison.baseline == "aaaa"
        assert comparison.candidate == "bbbb"


class TestSeries:
    def test_every_run_is_compared_against_the_first(self, make_record: Factory) -> None:
        records = [
            make_record(run_id="aaaa"),
            make_record(run_id="bbbb", scale=0.5),
            make_record(run_id="cccc", scale=2.0),
        ]

        comparisons = compare_series(records)

        assert [entry.baseline for entry in comparisons] == ["aaaa", "aaaa"]
        assert [entry.candidate for entry in comparisons] == ["bbbb", "cccc"]

    def test_a_single_run_is_not_a_comparison(self, make_record: Factory) -> None:
        with pytest.raises(ValidationError, match="at least two runs"):
            compare_series([make_record()])
