from __future__ import annotations

import re
from collections.abc import Callable
from html import escape
from pathlib import Path

import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.evals.result import RolloutRecord
from nnphysics.reporting.explain import METRIC_SUMMARIES, explanations
from nnphysics.reporting.plots import PlotRecord, render_plots
from nnphysics.reporting.record import RunRecord
from nnphysics.reporting.render import build_document, render_html, render_markdown

Factory = Callable[..., RunRecord]

EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["'](?!data:)""", re.IGNORECASE)
"""Anything the browser would have to fetch. A data URI is the only source allowed."""


def with_plots(record: RunRecord, directory: Path) -> tuple[PlotRecord, ...]:
    return render_plots(record.evaluation, None, directory)


class TestDeterminism:
    def test_the_same_record_renders_byte_identical_markdown(self, record: RunRecord) -> None:
        assert render_markdown(record).encode() == render_markdown(record).encode()

    def test_two_records_built_the_same_way_render_the_same(self, make_record: Factory) -> None:
        assert render_markdown(make_record()) == render_markdown(make_record())

    def test_a_changed_number_changes_the_report(self, make_record: Factory) -> None:
        assert render_markdown(make_record()) != render_markdown(make_record(scale=2.0))

    def test_the_html_is_stable_too(self, record: RunRecord, tmp_path: Path) -> None:
        plots = with_plots(record, tmp_path)

        first = render_html(record, plots, plot_dir=tmp_path)
        second = render_html(record, plots, plot_dir=tmp_path)

        assert first == second


class TestContent:
    def test_it_names_the_run_and_the_system(self, record: RunRecord) -> None:
        text = render_markdown(record)

        assert record.run_id in text
        assert record.evaluation.system in text

    def test_it_records_what_produced_the_numbers(self, record: RunRecord) -> None:
        text = render_markdown(record)

        for value in (record.commit, record.created, record.environment.python):
            assert value in text

    def test_it_states_the_timings(self, record: RunRecord) -> None:
        assert "evaluation (seconds)" in render_markdown(record)

    def test_every_metric_appears_with_its_summary(self, record: RunRecord) -> None:
        text = render_markdown(record)

        for metric in record.evaluation.settings.metrics:
            assert metric in text
            assert METRIC_SUMMARIES[metric] in text

    def test_every_scalar_appears_with_units_and_an_explanation(self, record: RunRecord) -> None:
        text = render_markdown(record)

        for explanation in explanations(record.evaluation):
            assert explanation.key in text
            assert explanation.units in text
            assert explanation.text in text

    def test_a_horizon_that_was_never_reached_says_so_rather_than_showing_minus_one(
        self, record: RunRecord
    ) -> None:
        text = render_markdown(record)

        assert "never reached" in text
        assert "| -1 " not in text

    def test_each_split_gets_its_own_section(self, record: RunRecord) -> None:
        text = render_markdown(record)

        assert "Results on the test split" in text
        assert "Results on the held_out split" in text

    def test_the_regime_gap_is_reported_with_a_direction(self, record: RunRecord) -> None:
        text = render_markdown(record)

        assert "Outside the training regimes" in text
        assert "held out is" in text
        assert "worse" in text

    def test_a_run_with_no_gap_leaves_the_section_out(self, make_record: Factory) -> None:
        text = render_markdown(make_record(splits=("test",), regime_gap={}))

        assert "Outside the training regimes" not in text

    def test_a_complete_run_says_that_nothing_stopped_early(self, record: RunRecord) -> None:
        assert "Every predictor reached the requested horizon" in render_markdown(record)

    def test_a_rollout_that_stopped_early_is_named(self, make_record: Factory) -> None:
        record = make_record()
        entry = record.evaluation.results[0]
        stopped = entry.model_copy(
            update={
                "rollouts": (
                    RolloutRecord(
                        trajectory="test-0",
                        regime="hot",
                        split="test",
                        steps_requested=4,
                        steps_completed=2,
                        stop_reason="diverged",
                        seconds=0.01,
                    ),
                )
            }
        )
        broken = record.model_copy(
            update={
                "evaluation": record.evaluation.model_copy(
                    update={"results": (stopped, *record.evaluation.results[1:])}
                )
            }
        )

        text = render_markdown(broken)

        assert "diverged" in text
        assert "2 of 4" in text

    def test_a_figure_is_referenced_beside_the_report(
        self, record: RunRecord, tmp_path: Path
    ) -> None:
        plots = with_plots(record, tmp_path)

        text = render_markdown(record, plots)

        assert "![" in text
        assert "plots/test-error.png" in text


class TestSelfContainedHtml:
    def test_it_references_nothing_outside_itself(self, record: RunRecord, tmp_path: Path) -> None:
        plots = with_plots(record, tmp_path)

        text = render_html(record, plots, plot_dir=tmp_path)

        assert not EXTERNAL.search(text)
        assert "http://" not in text
        assert "https://" not in text

    def test_every_figure_is_embedded(self, record: RunRecord, tmp_path: Path) -> None:
        plots = with_plots(record, tmp_path)

        text = render_html(record, plots, plot_dir=tmp_path)

        assert text.count("data:image/png;base64,") == len(plots)

    def test_a_missing_figure_is_a_failure_rather_than_a_silent_gap(
        self, record: RunRecord, tmp_path: Path
    ) -> None:
        plots = (PlotRecord("test-absent.png", "missing", "nothing here"),)

        with pytest.raises(ValidationError, match="cannot embed the figure"):
            render_html(record, plots, plot_dir=tmp_path)

    def test_it_carries_the_same_numbers_as_the_markdown(
        self, record: RunRecord, tmp_path: Path
    ) -> None:
        plots = with_plots(record, tmp_path)
        markdown = render_markdown(record, plots)
        html = render_html(record, plots, plot_dir=tmp_path)

        for explanation in explanations(record.evaluation):
            assert explanation.text in markdown
            # The HTML escapes what Markdown does not, so the same sentence is compared
            # in the form each format writes it.
            assert escape(explanation.text) in html


class TestDocument:
    def test_the_two_formats_come_from_one_document(self, record: RunRecord) -> None:
        assert build_document(record) == build_document(record)

    def test_an_unexplained_scalar_stops_the_report(self, make_record: Factory) -> None:
        record = make_record()
        entry = record.evaluation.results[0]
        metric = entry.metrics[0]
        odd = entry.model_copy(
            update={
                "metrics": (
                    metric.model_copy(update={"scalars": {**metric.scalars, "novelty": 1.0}}),
                    *entry.metrics[1:],
                )
            }
        )
        broken = record.model_copy(
            update={
                "evaluation": record.evaluation.model_copy(
                    update={"results": (odd, *record.evaluation.results[1:])}
                )
            }
        )

        with pytest.raises(Exception, match="nothing explains"):
            build_document(broken)
