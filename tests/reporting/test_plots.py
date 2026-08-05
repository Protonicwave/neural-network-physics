from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from nnphysics.evals.result import SuiteResult
from nnphysics.evals.snapshots import Snapshot, SnapshotSet
from nnphysics.reporting.plots import (
    distribution_plot,
    drift_plot,
    error_plot,
    overlay_plot,
    render_plots,
    snapshot_plot,
)

Factory = Callable[..., SuiteResult]

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def make_snapshot(shape: tuple[int, ...], predictor: str = "persistence") -> Snapshot:
    """A snapshot of one field with a given per state shape."""
    generator = np.random.default_rng(0)
    stacked = generator.normal(size=(3, *shape))
    return Snapshot(
        predictor=predictor,
        split="test",
        trajectory="test-0",
        steps=(0, 1, 2),
        times=(0.0, 0.1, 0.2),
        predicted={"field": stacked},
        reference={"field": stacked * 0.5},
    )


class TestErrorPlot:
    def test_it_writes_a_png(self, make_result: Factory, tmp_path: Path) -> None:
        record = error_plot(make_result(), "test", tmp_path / "e.png")

        assert record is not None
        assert (tmp_path / "e.png").read_bytes().startswith(_PNG_MAGIC)

    def test_it_names_the_split_in_its_title(self, make_result: Factory, tmp_path: Path) -> None:
        record = error_plot(make_result(), "test", tmp_path / "e.png")

        assert record is not None
        assert "test" in record.title

    def test_a_split_with_no_curve_draws_nothing(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        assert error_plot(make_result(), "validation", tmp_path / "e.png") is None
        assert not (tmp_path / "e.png").exists()

    def test_the_same_result_gives_byte_identical_output(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        result = make_result()

        error_plot(result, "test", tmp_path / "one.png")
        error_plot(result, "test", tmp_path / "two.png")

        assert (tmp_path / "one.png").read_bytes() == (tmp_path / "two.png").read_bytes()


class TestDriftPlot:
    def test_it_draws_one_panel_per_declared_invariant(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        record = drift_plot(make_result(), "test", tmp_path / "d.png")

        assert record is not None
        assert "energy" not in record.title
        assert (tmp_path / "d.png").is_file()

    def test_it_says_the_band_is_the_declared_tolerance(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        record = drift_plot(make_result(), "test", tmp_path / "d.png")

        assert record is not None
        assert "tolerance the system declared" in record.caption


class TestDistributionPlot:
    def test_it_draws_the_spread_across_trajectories(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        record = distribution_plot(make_result(), "test", tmp_path / "s.png")

        assert record is not None
        assert (tmp_path / "s.png").is_file()

    def test_a_result_without_per_rollout_scalars_draws_nothing(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        result = make_result()
        stripped = result.model_dump()
        for entry in stripped["results"]:
            for rollout in entry["rollouts"]:
                rollout["scalars"] = {}

        assert (
            distribution_plot(SuiteResult.model_validate(stripped), "test", tmp_path / "s.png")
            is None
        )


class TestSnapshotPlot:
    def test_a_field_of_plane_vectors_is_drawn_in_the_plane(self, tmp_path: Path) -> None:
        record = snapshot_plot(make_snapshot((6, 2)), tmp_path / "q.png")

        assert record is not None
        assert "filled black" in record.caption
        assert (tmp_path / "q.png").is_file()

    def test_a_two_dimensional_field_is_drawn_as_an_image(self, tmp_path: Path) -> None:
        record = snapshot_plot(make_snapshot((8, 8)), tmp_path / "q.png")

        assert record is not None
        assert "top row is the truth" in record.caption
        assert "own colour scale" in record.caption

    def test_a_field_of_one_value_per_element_is_drawn_as_a_line(self, tmp_path: Path) -> None:
        record = snapshot_plot(make_snapshot((6,)), tmp_path / "q.png")

        assert record is not None
        assert "over the element index" in record.caption

    def test_a_state_with_nothing_but_scalars_is_not_drawn(self, tmp_path: Path) -> None:
        assert snapshot_plot(make_snapshot(()), tmp_path / "q.png") is None

    def test_it_names_the_trajectory_it_came_from(self, tmp_path: Path) -> None:
        record = snapshot_plot(make_snapshot((6, 2)), tmp_path / "q.png")

        assert record is not None
        assert "test-0" in record.caption

    def test_the_same_snapshot_gives_byte_identical_output(self, tmp_path: Path) -> None:
        snapshot = make_snapshot((6, 2))

        snapshot_plot(snapshot, tmp_path / "one.png")
        snapshot_plot(snapshot, tmp_path / "two.png")

        assert (tmp_path / "one.png").read_bytes() == (tmp_path / "two.png").read_bytes()


class TestOverlay:
    def test_it_draws_one_curve_per_run(self, tmp_path: Path) -> None:
        curves = [
            ("run one", np.asarray([0.1, 0.2, 0.3]), np.asarray([0.0, 0.1, 0.2])),
            ("run two", np.asarray([0.1, 0.15, 0.2]), None),
        ]

        record = overlay_plot(curves, "reference on the test split", tmp_path / "o.png")

        assert record.title == "reference on the test split"
        assert (tmp_path / "o.png").is_file()


class TestRenderPlots:
    def test_it_draws_every_figure_for_every_split(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        records = render_plots(make_result(), None, tmp_path)

        names = {record.name for record in records}
        assert names == {
            "test-error.png",
            "test-invariant-drift.png",
            "test-spread.png",
            "held_out-error.png",
            "held_out-invariant-drift.png",
            "held_out-spread.png",
        }

    def test_it_draws_a_qualitative_figure_for_each_snapshot(
        self, make_result: Factory, tmp_path: Path
    ) -> None:
        snapshots = SnapshotSet((make_snapshot((6, 2), predictor="persistence"),))

        records = render_plots(make_result(), snapshots, tmp_path)

        assert "test-state-persistence.png" in {record.name for record in records}

    def test_every_figure_it_names_is_on_disk(self, make_result: Factory, tmp_path: Path) -> None:
        records = render_plots(make_result(), None, tmp_path)

        for record in records:
            assert (tmp_path / record.name).is_file()

    def test_every_figure_carries_a_caption(self, make_result: Factory, tmp_path: Path) -> None:
        for record in render_plots(make_result(), None, tmp_path):
            assert record.caption.endswith(".")


@pytest.mark.parametrize("shape", [(6, 2), (8, 8), (4, 3)])
def test_a_snapshot_of_any_shape_is_drawn_without_knowing_the_system(
    shape: tuple[int, ...], tmp_path: Path
) -> None:
    record = snapshot_plot(make_snapshot(shape), tmp_path / "q.png")

    assert record is not None
    assert (tmp_path / "q.png").is_file()
