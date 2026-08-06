from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from nnphysics.agent.context import (
    IGNORED_CONFIG_PATHS,
    MAX_REGRESSIONS,
    build_context,
)
from nnphysics.core.errors import ValidationError
from nnphysics.reporting.record import RunRecord

from .conftest import CONFIG


def _with(**changes: Any) -> dict[str, Any]:
    """The fixture configuration with some nested keys replaced."""
    config = copy.deepcopy(CONFIG)
    for path, value in changes.items():
        section, _, key = path.partition(".")
        if key:
            config.setdefault(section, {})[key] = value
        else:
            config[section] = value
    return config


class TestConfigDifferences:
    def test_a_changed_leaf_is_reported_with_both_values(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        baseline = make_record()
        candidate = make_record(
            run_id="1",
            config=_with(**{"training.curriculum": [1], "training.curriculum_epochs": [0]}),
        )

        context = build_context(baseline, candidate)

        paths = {
            entry.path: (entry.baseline, entry.candidate) for entry in context.config_differences
        }
        assert paths["training.curriculum"] == ("[1, 2]", "[1]")

    def test_a_sequence_is_one_change_rather_than_one_per_index(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """Reading a curriculum as three indices would bury it in the indices moving."""
        candidate = make_record(
            run_id="1",
            config=_with(**{"training.curriculum": [1], "training.curriculum_epochs": [0]}),
        )

        context = build_context(make_record(), candidate)

        assert not [
            entry
            for entry in context.config_differences
            if entry.path.startswith("training.curriculum.")
        ]

    def test_paths_that_say_where_files_went_are_left_out(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """They differ between machines and never between a working run and a broken one."""
        candidate = make_record(run_id="1", config=_with(**{"data.workers": 3, "name": "other"}))

        context = build_context(make_record(), candidate)

        assert not [
            entry for entry in context.config_differences if entry.path in IGNORED_CONFIG_PATHS
        ]

    def test_identical_configurations_produce_no_differences(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """Three injected faults change nothing here, and the empty diff is the evidence."""
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        assert context.config_differences == ()


class TestChanges:
    def test_a_worse_scalar_is_a_regression_carrying_its_direction(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        worse = [
            change for change in context.regressions if change.scalar == "rollout_error.error.final"
        ]
        assert worse
        assert worse[0].candidate > worse[0].baseline
        assert worse[0].direction == "lower is better"

    def test_regressions_are_ordered_worst_first(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        relatives = [abs(change.relative) for change in context.regressions]
        assert relatives == sorted(relatives, reverse=True)

    def test_the_list_is_capped_and_says_how_many_were_dropped(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """A silent truncation would read as a run with fewer regressions than it had."""
        context = build_context(make_record(), make_record(run_id="1", scale=100.0))

        assert len(context.regressions) <= MAX_REGRESSIONS
        if context.regressions_dropped:
            assert len(context.regressions) == MAX_REGRESSIONS

    def test_improvements_are_kept_too(self, make_record: Callable[..., RunRecord]) -> None:
        """A fault that improves one number while ruining another is what this catches."""
        context = build_context(make_record(scale=4.0), make_record(run_id="1"))

        assert context.improvements

    def test_the_context_is_json_serialisable(self, make_record: Callable[..., RunRecord]) -> None:
        """A baseline of zero makes a relative change infinite, which JSON cannot carry."""
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        payload = json.loads(context.model_dump_json())

        assert payload["system"] == "toy"

    def test_a_run_on_another_system_is_refused(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        baseline = make_record()
        other = make_record(run_id="1").model_copy(
            update={
                "evaluation": make_record(run_id="1").evaluation.model_copy(
                    update={"system": "elsewhere"}
                )
            }
        )

        with pytest.raises(ValidationError, match="cannot compare"):
            build_context(baseline, other)


class TestCurves:
    def test_the_initial_condition_is_dropped_from_the_curve(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """Every predictor is exactly right at the initial condition.

        Keeping that point would make the first error zero for everything and the growth ratio
        undefined.
        """
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        for curve in context.curves:
            assert curve.first > 0.0
            assert curve.growth > 0.0

    def test_a_rising_curve_is_described_as_rising(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        curve = context.curves[0]
        assert curve.rising_fraction == pytest.approx(1.0)
        assert curve.worst_position == pytest.approx(1.0)


class TestTraining:
    def test_both_runs_training_is_summarised(self, make_record: Callable[..., RunRecord]) -> None:
        context = build_context(
            make_record(), make_record(run_id="1", curriculum=(1,), validation=0.9)
        )

        assert context.baseline_training is not None
        assert context.candidate_training is not None
        assert context.baseline_training.curriculum_stages == (1, 2)
        assert context.candidate_training.curriculum_stages == (1,)

    def test_a_run_that_trained_nothing_says_so(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """A run that scores only the solver and the fixtures trains nothing, and is a run."""
        context = build_context(make_record(trained=False), make_record(run_id="1"))

        assert context.baseline_training is None


class TestRollouts:
    def test_a_rollout_that_did_not_finish_is_reported(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", completed=1))

        assert context.rollouts
        assert context.rollouts[0].completed == 1
        assert "diverged" in context.rollouts[0].stop_reasons

    def test_nothing_is_reported_when_everything_finished(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1"))

        assert context.rollouts == ()


class TestRender:
    def test_rendering_twice_gives_the_same_text(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        """A recorded reply describes the call it came from only if that call never moves."""
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        assert context.render() == context.render()

    def test_every_section_is_present(self, make_record: Callable[..., RunRecord]) -> None:
        text = build_context(make_record(), make_record(run_id="1", scale=4.0)).render()

        for heading in (
            "## Configuration differences",
            "## Regressions",
            "## Improvements",
            "## Error curve shapes",
            "## Training",
            "## Rollouts that did not finish",
        ):
            assert heading in text

    def test_no_raw_array_reaches_the_text(self, make_record: Callable[..., RunRecord]) -> None:
        """The reduction is the whole point.

        A curve sent verbatim would be the run record again, only harder to read.
        """
        text = build_context(make_record(), make_record(run_id="1", scale=4.0)).render()

        assert "0.25, 0.4" not in text
        assert len(text) < 20_000

    def test_an_unbounded_change_is_named_rather_than_printed(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        baseline = make_record(seconds=0.0)
        text = build_context(baseline, make_record(run_id="1", seconds=0.5)).render()

        assert "unbounded, the baseline was zero" in text

    def test_the_regressed_metrics_are_ordered_by_how_far_they_moved(
        self, make_record: Callable[..., RunRecord]
    ) -> None:
        context = build_context(make_record(), make_record(run_id="1", scale=4.0))

        assert context.regressed_metrics
        assert "rollout_error" in context.regressed_metrics
