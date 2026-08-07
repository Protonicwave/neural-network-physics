from __future__ import annotations

import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.reporting.draw import (
    Bar,
    Dot,
    Label,
    Linear,
    Log,
    Stroke,
    canvas,
    decade_ticks,
    ticks,
)

TRACK = 500.0
"""The length of the axis the geometry is asserted against, chosen so that a tenth of it
is a round number."""


class TestLinear:
    def test_the_smallest_value_sits_on_the_axis(self) -> None:
        axis = Linear(lo=0.0, hi=255.0, start=150.0, end=650.0)

        assert axis.at(0.0) == 150.0

    def test_the_largest_value_sits_at_the_far_end(self) -> None:
        axis = Linear(lo=0.0, hi=255.0, start=150.0, end=650.0)

        assert axis.at(255.0) == 650.0

    def test_a_tenth_of_the_axis_is_a_tenth_of_the_track(self) -> None:
        axis = Linear(lo=0.0, hi=100.0, start=150.0, end=150.0 + TRACK)

        assert axis.length(10.0) == pytest.approx(TRACK / 10.0)

    def test_an_axis_can_run_upwards_while_the_viewport_runs_downwards(self) -> None:
        """A vertical axis has its largest value at the smallest coordinate."""
        axis = Linear(lo=0.0, hi=2.0, start=350.0, end=30.0)

        assert axis.at(0.0) == 350.0
        assert axis.at(2.0) == 30.0
        assert axis.length(1.0) == pytest.approx(160.0)

    def test_a_value_above_the_axis_raises(self) -> None:
        axis = Linear(lo=0.0, hi=100.0, start=0.0, end=TRACK)

        with pytest.raises(ValidationError, match="outside the axis"):
            axis.at(101.0)

    def test_a_value_below_the_axis_raises(self) -> None:
        axis = Linear(lo=0.0, hi=100.0, start=0.0, end=TRACK)

        with pytest.raises(ValidationError, match="outside the axis"):
            axis.at(-0.5)

    def test_a_value_that_is_not_a_number_raises(self) -> None:
        axis = Linear(lo=0.0, hi=100.0, start=0.0, end=TRACK)

        with pytest.raises(ValidationError, match="cannot place"):
            axis.at(float("nan"))

    def test_an_empty_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="lo < hi"):
            Linear(lo=1.0, hi=1.0, start=0.0, end=TRACK)


class TestLog:
    def test_the_smallest_value_sits_on_the_axis(self) -> None:
        axis = Log(lo=0.1, hi=10.0, start=60.0, end=690.0)

        assert axis.at(0.1) == 60.0

    def test_a_decade_is_the_same_distance_wherever_it_falls(self) -> None:
        axis = Log(lo=0.01, hi=10.0, start=0.0, end=300.0)

        assert axis.at(0.1) == pytest.approx(100.0)
        assert axis.at(1.0) == pytest.approx(200.0)

    def test_a_value_outside_the_axis_raises(self) -> None:
        axis = Log(lo=0.1, hi=10.0, start=0.0, end=300.0)

        with pytest.raises(ValidationError, match="outside the axis"):
            axis.at(11.0)

    def test_an_axis_reaching_zero_raises(self) -> None:
        """A logarithm has no value there, so the axis has no origin to draw from."""
        with pytest.raises(ValidationError, match="lo > 0"):
            Log(lo=0.0, hi=10.0, start=0.0, end=300.0)


class TestTicks:
    def test_a_linear_axis_is_labelled_in_readable_steps(self) -> None:
        assert ticks(0.0, 1.68, 4) == pytest.approx((0.0, 0.5, 1.0, 1.5))

    def test_the_step_grows_with_the_range(self) -> None:
        assert ticks(0.0, 100.0, 4) == pytest.approx((0.0, 20.0, 40.0, 60.0, 80.0, 100.0))

    def test_a_range_with_no_round_value_in_it_has_no_ticks(self) -> None:
        assert ticks(1.01, 1.09, 1) == ()

    def test_an_empty_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="lo < hi"):
            ticks(1.0, 1.0, 4)

    def test_a_logarithmic_axis_is_labelled_once_per_decade_and_a_half(self) -> None:
        assert decade_ticks(0.05, 12.0) == pytest.approx((0.05, 0.1, 0.5, 1.0, 5.0, 10.0))

    def test_a_narrow_logarithmic_axis_is_labelled_more_finely(self) -> None:
        assert decade_ticks(0.9, 12.0) == pytest.approx((1.0, 2.0, 5.0, 10.0))

    def test_a_logarithmic_axis_reaching_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="lo > 0"):
            decade_ticks(0.0, 10.0)


class TestMarks:
    def test_a_bar_names_its_colour_rather_than_writing_one(self) -> None:
        mark = Bar(x=0.0, y=0.0, width=10.0, height=7.0, colour="nbody").draw()

        assert 'fill="var(--nbody)"' in mark
        assert "#" not in mark

    def test_a_misspelled_colour_raises(self) -> None:
        """A colour that resolves to nothing draws an invisible mark."""
        with pytest.raises(ValidationError, match="no theme token"):
            Bar(x=0.0, y=0.0, width=10.0, height=7.0, colour="nbdoy").draw()

    def test_a_point_carries_a_ring_so_two_points_stay_two_points(self) -> None:
        mark = Dot(x=1.0, y=2.0, colour="fluid").draw()

        assert 'stroke="var(--surface)"' in mark

    def test_a_stroke_joins_its_points_in_order(self) -> None:
        mark = Stroke(points=((0.0, 0.0), (10.0, 5.0)), colour="axis").draw()

        assert 'd="M0 0 L10 5"' in mark

    def test_a_dashed_stroke_says_so(self) -> None:
        mark = Stroke(points=((0.0, 0.0), (1.0, 1.0)), colour="fail", dashed=True).draw()

        assert "stroke-dasharray" in mark

    def test_a_stroke_with_one_point_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least two points"):
            Stroke(points=((0.0, 0.0),), colour="axis")

    def test_a_label_is_escaped(self) -> None:
        mark = Label(x=0.0, y=0.0, text="<script>", colour="ink").draw()

        assert "&lt;script&gt;" in mark
        assert "<script>" not in mark

    def test_a_label_can_be_turned_on_its_side(self) -> None:
        mark = Label(x=18.0, y=190.0, text="error", rotate=-90.0).draw()

        assert 'transform="rotate(-90 18 190)"' in mark

    def test_a_number_is_set_in_the_mono_family(self) -> None:
        assert 'class="num"' in Label(x=0.0, y=0.0, text="3.8", mono=True).draw()

    def test_a_coordinate_is_written_short(self) -> None:
        """The same chart on two machines must give the same bytes."""
        assert 'x="7.35"' in Bar(x=7.3456, y=0.0, width=1.0, height=1.0, colour="ink").draw()


class TestCanvas:
    def test_the_chart_carries_its_text_alternative(self) -> None:
        drawn = canvas(720.0, 200.0, (Bar(x=0, y=0, width=1, height=1, colour="ink").draw(),), "It")

        assert 'role="img"' in drawn
        assert 'aria-label="It"' in drawn

    def test_the_text_alternative_is_escaped(self) -> None:
        drawn = canvas(720.0, 200.0, ("",), 'a "quoted" <claim>')

        assert "&quot;quoted&quot;" in drawn
        assert "<claim>" not in drawn

    def test_a_chart_without_a_text_alternative_raises(self) -> None:
        with pytest.raises(ValidationError, match="alternative text"):
            canvas(720.0, 200.0, ("",), "  ")
