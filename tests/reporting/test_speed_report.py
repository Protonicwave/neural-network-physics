from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from nnphysics.evals.result import SuiteResult
from nnphysics.reporting.document import Table, to_markdown
from nnphysics.reporting.plots import reliability_plot, render_plots, speed_plot, warning_plot
from nnphysics.reporting.record import (
    RECORD_SCHEMA_VERSION,
    RunRecord,
    read_record,
    write_record,
)
from nnphysics.reporting.render import build_document, render_html, render_markdown

if TYPE_CHECKING:
    from collections.abc import Callable

    from nnphysics.evals.speed import SpeedReport


def tables(record: RunRecord) -> list[Table]:
    return [block for block in build_document(record).blocks if isinstance(block, Table)]


class TestTheSpeedSection:
    def test_a_run_that_was_never_benchmarked_says_so_rather_than_claiming_nothing(
        self, record: RunRecord
    ) -> None:
        """Silence would read as a run with no speedup.

        That is a different statement from a run that was never timed.
        """
        assert "was not benchmarked" in render_markdown(record)

    def test_the_ladder_and_the_surrogate_are_both_in_the_report(
        self, benchmarked: RunRecord
    ) -> None:
        rendered = render_markdown(benchmarked)

        assert "Speed at matched accuracy" in rendered
        assert "reference:substeps=1" in rendered
        assert "operator" in rendered

    def test_the_speedup_is_stated_against_the_setting_it_was_matched_to(
        self, benchmarked: RunRecord
    ) -> None:
        captions = [table.caption for table in tables(benchmarked)]

        assert any("faster than the solver" in caption for caption in captions)

    def test_the_break_even_count_is_stated(self, benchmarked: RunRecord) -> None:
        rendered = render_markdown(benchmarked)

        assert "break even (rollouts)" in rendered

    def test_a_surrogate_that_never_pays_says_so_in_words(
        self, record: RunRecord, make_speed_report: Callable[[], SpeedReport]
    ) -> None:
        """A break even count of minus one in a table would read as a measurement."""
        report = make_speed_report()
        never = report.costs[0].model_copy(
            update={"saving_per_rollout": -1.0, "break_even_rollouts": -1.0}
        )
        attached = record.model_copy(
            update={"benchmark": report.model_copy(update={"costs": (never,)})}
        )

        assert "never: it is the slower of the two" in render_markdown(attached)

    def test_an_unstable_timing_is_named_rather_than_quietly_quoted(
        self, record: RunRecord, make_speed_report: Callable[[], SpeedReport]
    ) -> None:
        """A noisy measurement is named rather than dropped.

        A benchmark that discarded them would report a precision it did not have.
        """
        report = make_speed_report()
        attached = record.model_copy(
            update={"benchmark": report.model_copy(update={"unstable": ("reference:substeps=1",)})}
        )

        rendered = render_markdown(attached)

        assert "varied by more than" in rendered
        assert "reference:substeps=1" in rendered

    def test_an_upper_bound_is_marked_as_one(
        self, record: RunRecord, make_speed_report: Callable[[], SpeedReport]
    ) -> None:
        report = make_speed_report()
        loose = report.matched[0].model_copy(update={"bracketed": False})
        attached = record.model_copy(
            update={"benchmark": report.model_copy(update={"matched": (loose,)})}
        )

        rows = [row for table in tables(attached) for row in table.rows if "operator" in row]

        assert any("no" in row for row in rows)


class TestTheRecordCarriesIt:
    def test_a_benchmark_survives_being_written_and_read(
        self, benchmarked: RunRecord, tmp_path: Path
    ) -> None:
        path = tmp_path / "record.json"
        write_record(path, benchmarked)

        assert read_record(path).benchmark == benchmarked.benchmark

    def test_a_record_written_before_benchmarks_existed_still_reads(
        self, record: RunRecord, tmp_path: Path
    ) -> None:
        """A run whose numbers can no longer be read is a run that never happened."""
        path = tmp_path / "record.json"
        payload = record.model_dump(mode="json")
        payload["schema_version"] = 1
        payload.pop("benchmark")
        path.write_text(json.dumps(payload), encoding="utf-8")

        recovered = read_record(path)

        assert recovered.schema_version == RECORD_SCHEMA_VERSION
        assert recovered.benchmark is None


class TestTheCalibrationNumbersAreExplained:
    def test_every_scalar_the_metric_produces_reaches_a_report(
        self, record: RunRecord, uncertain_result: SuiteResult
    ) -> None:
        """The guard that stops a metric growing a number nobody has described."""
        attached = record.model_copy(update={"evaluation": uncertain_result})

        rendered = render_markdown(attached)

        assert "Average gap between the coverage" in rendered
        assert "what a correctly sized Gaussian delivers" in rendered
        assert "how long before the error became unacceptable" in rendered.lower()

    def test_a_predictor_that_never_warned_reads_as_never_reached(
        self, record: RunRecord, uncertain_result: SuiteResult
    ) -> None:
        """The sentinel written out in words, because minus one is not a time."""
        attached = record.model_copy(update={"evaluation": uncertain_result})

        rendered = render_markdown(attached)

        assert "never reached" in rendered
        assert "not determined" in rendered


class TestTheFigures:
    def test_the_reliability_diagram_is_drawn_for_a_predictor_that_states_a_spread(
        self, uncertain_result: SuiteResult, tmp_path: Path
    ) -> None:
        drawn = reliability_plot(uncertain_result, "test", tmp_path / "reliability.png")

        assert drawn is not None
        assert (tmp_path / "reliability.png").is_file()

    def test_it_is_not_drawn_when_nobody_stated_one(
        self, make_result: Callable[..., SuiteResult], tmp_path: Path
    ) -> None:
        assert reliability_plot(make_result(), "test", tmp_path / "reliability.png") is None

    def test_the_uncertainty_against_error_figure_is_drawn(
        self, uncertain_result: SuiteResult, tmp_path: Path
    ) -> None:
        drawn = warning_plot(uncertain_result, "test", tmp_path / "uncertainty.png")

        assert drawn is not None
        assert (tmp_path / "uncertainty.png").is_file()

    def test_the_accuracy_against_wall_clock_curve_is_drawn(
        self, make_speed_report: Callable[[], SpeedReport], tmp_path: Path
    ) -> None:
        drawn = speed_plot(make_speed_report(), tmp_path / "speed.png")

        assert drawn.name == "speed.png"
        assert (tmp_path / "speed.png").is_file()

    def test_the_speed_figure_is_rendered_only_when_there_is_a_benchmark(
        self,
        make_result: Callable[..., SuiteResult],
        make_speed_report: Callable[[], SpeedReport],
        tmp_path: Path,
    ) -> None:
        without = render_plots(make_result(), None, tmp_path)
        with_benchmark = render_plots(make_result(), None, tmp_path, make_speed_report())

        assert not any(plot.name == "speed.png" for plot in without)
        assert any(plot.name == "speed.png" for plot in with_benchmark)

    def test_the_html_report_embeds_the_speed_figure(
        self, benchmarked: RunRecord, tmp_path: Path
    ) -> None:
        plots = render_plots(benchmarked.evaluation, None, tmp_path, benchmarked.benchmark)

        html = render_html(benchmarked, plots, plot_dir=tmp_path)

        assert "data:image/png;base64," in html
        assert "speed.png" not in html


class TestRenderingStaysDeterministic:
    def test_the_same_record_renders_the_same_report_twice(self, benchmarked: RunRecord) -> None:
        assert to_markdown(build_document(benchmarked)) == to_markdown(build_document(benchmarked))


class TestWhatCouldNotBeMeasured:
    def test_a_solver_setting_that_could_not_run_is_named(
        self, record: RunRecord, make_speed_report: Callable[[], SpeedReport]
    ) -> None:
        """The ladder stops rather than being extended with an invented number.

        And the report says where it stopped.
        """
        report = make_speed_report()
        attached = record.model_copy(
            update={"benchmark": report.model_copy(update={"unusable_substeps": (1,)})}
        )

        rendered = render_markdown(attached)

        assert "Missing from the table: 1 substeps" in rendered
        assert "could not take a single step" in rendered

    def test_a_predictor_that_could_not_run_is_named_the_same_way(
        self, record: RunRecord, make_speed_report: Callable[[], SpeedReport]
    ) -> None:
        report = make_speed_report()
        attached = record.model_copy(
            update={"benchmark": report.model_copy(update={"unmeasurable": ("operator",)})}
        )

        assert "Missing from the table: operator" in render_markdown(attached)
