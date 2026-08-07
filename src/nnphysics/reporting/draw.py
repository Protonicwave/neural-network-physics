"""The drawing vocabulary the landing page charts are built from.

The charts have to follow the reader's theme, scale with the width of the browser and
reference no host, so they are generated SVG rather than rendered images. This module is
the whole of the vocabulary: two scales, a bar, a point, a stroke and a label. It knows
nothing about runs, predictors or costs.

It is deliberately not a plotting library. There is no automatic layout, no legend and no
default anybody has to look up: a chart states every coordinate itself. Anything a chart
needs twice belongs in `charts`, and a chart that wants a concept this module does not
have is asking for a plot rather than a drawing.

Colour is named, never written. Every mark takes a theme token name and resolves it
through `theme.css_var`, so a misspelled colour raises here rather than drawing an
invisible mark, and no colour value appears outside the theme module.

A value outside its scale raises. Drawing it would put a mark outside the frame, which a
reader sees as a chart with a different axis rather than as a mistake.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Literal, Protocol

from nnphysics.core.errors import ValidationError
from nnphysics.reporting.theme import css_var

__all__ = [
    "Bar",
    "Dot",
    "Label",
    "Linear",
    "Log",
    "Scale",
    "Stroke",
    "canvas",
    "decade_ticks",
    "ticks",
]

type Anchor = Literal["start", "middle", "end"]
"""Where a label sits relative to its own coordinate."""

_STEPS = (1.0, 2.0, 5.0)
"""The tick steps a reader reads without arithmetic. A chart asks for a step near the one
it wants and gets the nearest of these, so that no axis is ever labelled in sevenths."""


class Scale(Protocol):
    """A mapping from a range of values onto a range of pixels."""

    def at(self, value: float) -> float:
        """Place one value.

        Args:
            value: A value in the scale's range.

        Returns:
            Where it sits, in viewport units.
        """
        ...


@dataclass(frozen=True, slots=True)
class Linear:
    """A linear scale.

    Attributes:
        lo: Smallest value the axis shows.
        hi: Largest value the axis shows.
        start: Where `lo` sits, in viewport units.
        end: Where `hi` sits. May be smaller than `start`, which is how a vertical axis
            is drawn: values grow upwards while viewport coordinates grow downwards.
    """

    lo: float
    hi: float
    start: float
    end: float

    def __post_init__(self) -> None:
        """Reject a range no value can sit in.

        Raises:
            ValidationError: If the range is empty or inverted.
        """
        if not self.lo < self.hi:
            raise ValidationError(f"a scale needs lo < hi, got lo={self.lo}, hi={self.hi}")

    def at(self, value: float) -> float:
        """Place one value.

        Args:
            value: A value between `lo` and `hi`.

        Returns:
            Where it sits, in viewport units.

        Raises:
            ValidationError: If the value is outside the axis.
        """
        _inside(value, self.lo, self.hi)
        return self.start + (value - self.lo) / (self.hi - self.lo) * (self.end - self.start)

    def length(self, value: float) -> float:
        """How long a bar reaching this value is.

        Args:
            value: A value between `lo` and `hi`.

        Returns:
            The distance from the start of the axis, in viewport units, never negative.

        Raises:
            ValidationError: If the value is outside the axis.
        """
        return abs(self.at(value) - self.start)


@dataclass(frozen=True, slots=True)
class Log:
    """A logarithmic scale, for a quantity spanning decades.

    Wall clock per step is the one such quantity on the page: the solver's cheapest
    setting and the slowest surrogate are two orders of magnitude apart, and a linear axis
    would put every solver setting on top of the origin.

    Attributes:
        lo: Smallest value the axis shows. Must be positive.
        hi: Largest value the axis shows.
        start: Where `lo` sits, in viewport units.
        end: Where `hi` sits.
    """

    lo: float
    hi: float
    start: float
    end: float

    def __post_init__(self) -> None:
        """Reject a range a logarithm is not defined on.

        Raises:
            ValidationError: If the range is empty, inverted or reaches zero.
        """
        if self.lo <= 0.0:
            raise ValidationError(f"a logarithmic scale needs lo > 0, got lo={self.lo}")
        if not self.lo < self.hi:
            raise ValidationError(f"a scale needs lo < hi, got lo={self.lo}, hi={self.hi}")

    def at(self, value: float) -> float:
        """Place one value.

        Args:
            value: A value between `lo` and `hi`.

        Returns:
            Where it sits, in viewport units.

        Raises:
            ValidationError: If the value is outside the axis.
        """
        _inside(value, self.lo, self.hi)
        decades = math.log10(self.hi) - math.log10(self.lo)
        return self.start + (math.log10(value) - math.log10(self.lo)) / decades * (
            self.end - self.start
        )


@dataclass(frozen=True, slots=True)
class Bar:
    """A rectangle, used for a bar and for the track it is drawn on.

    Attributes:
        x: Left edge.
        y: Top edge.
        width: Length along the axis.
        height: Thickness across it.
        colour: Theme token name.
        radius: Corner radius.
    """

    x: float
    y: float
    width: float
    height: float
    colour: str
    radius: float = 2.0

    def draw(self) -> str:
        """The mark, as SVG."""
        return (
            f'<rect x="{_n(self.x)}" y="{_n(self.y)}" width="{_n(self.width)}" '
            f'height="{_n(self.height)}" rx="{_n(self.radius)}" fill="{css_var(self.colour)}"/>'
        )


@dataclass(frozen=True, slots=True)
class Dot:
    """A measured point.

    Attributes:
        x: Where it sits across the frame.
        y: Where it sits up the frame.
        colour: Theme token name of the fill.
        radius: How big it is drawn.
        edge: Theme token name of the ring around it, which is what keeps two overlapping
            points readable as two points.
    """

    x: float
    y: float
    colour: str
    radius: float = 5.0
    edge: str = "surface"

    def draw(self) -> str:
        """The mark, as SVG."""
        return (
            f'<circle cx="{_n(self.x)}" cy="{_n(self.y)}" r="{_n(self.radius)}" '
            f'fill="{css_var(self.colour)}" stroke="{css_var(self.edge)}" stroke-width="2"/>'
        )


@dataclass(frozen=True, slots=True)
class Stroke:
    """A line through one or more points, used for an axis, a rule and a joined series.

    Attributes:
        points: The corners, in order. Two points make a straight line.
        colour: Theme token name.
        width: How thick it is drawn.
        dashed: Whether it is drawn as a dashed line, which the page uses for a
            comparison it has made rather than a quantity it has measured.
    """

    points: tuple[tuple[float, float], ...]
    colour: str
    width: float = 1.0
    dashed: bool = False

    def __post_init__(self) -> None:
        """Reject a line with nothing to join.

        Raises:
            ValidationError: If fewer than two points were given.
        """
        if len(self.points) < 2:  # noqa: PLR2004 - a line joins at least two points
            raise ValidationError(f"a stroke needs at least two points, got {len(self.points)}")

    def draw(self) -> str:
        """The mark, as SVG."""
        head, *rest = self.points
        path = f"M{_n(head[0])} {_n(head[1])}" + "".join(f" L{_n(x)} {_n(y)}" for x, y in rest)
        dash = ' stroke-dasharray="5 4"' if self.dashed else ""
        return (
            f'<path d="{path}" fill="none" stroke="{css_var(self.colour)}" '
            f'stroke-width="{_n(self.width)}" stroke-linejoin="round"{dash}/>'
        )


@dataclass(frozen=True, slots=True)
class Label:
    """A piece of text on a chart.

    Attributes:
        x: Where it sits across the frame.
        y: Its baseline.
        text: What it says. Escaped when drawn, so a run name is safe here.
        colour: Theme token name.
        size: Type size, in viewport units.
        anchor: Which end of the text sits at `x`.
        bold: Whether it is the label of a series rather than of an axis.
        rotate: Degrees to turn it about its own position, for an axis title up the side
            of a chart. `None` leaves it horizontal.
        mono: Whether it is a number. A column of numbers is set in the mono family with
            tabular figures, so that the digits line up down the column.
    """

    x: float
    y: float
    text: str
    colour: str = "muted"
    size: float = 11.0
    anchor: Anchor = "start"
    bold: bool = False
    rotate: float | None = None
    mono: bool = False

    def draw(self) -> str:
        """The mark, as SVG."""
        weight = ' font-weight="600"' if self.bold else ""
        family = ' class="num"' if self.mono else ""
        turn = (
            f' transform="rotate({_n(self.rotate)} {_n(self.x)} {_n(self.y)})"'
            if self.rotate is not None
            else ""
        )
        return (
            f'<text x="{_n(self.x)}" y="{_n(self.y)}" font-size="{_n(self.size)}" '
            f'fill="{css_var(self.colour)}" text-anchor="{self.anchor}"'
            f"{weight}{family}{turn}>{html.escape(self.text, quote=True)}</text>"
        )


def canvas(width: float, height: float, marks: tuple[str, ...], alt: str) -> str:
    """Wrap the marks of one chart in a viewport.

    Args:
        width: Viewport width, which the page scales to its own.
        height: Viewport height.
        marks: The drawn marks, in painting order.
        alt: What the chart shows and what to conclude from it, for a reader who cannot
            see it. It is the chart's only text alternative, so it states the conclusion
            rather than naming the chart.

    Returns:
        The chart, as one SVG element ending in a newline.

    Raises:
        ValidationError: If the alternative text is empty. A chart nobody can read is a
            chart that reports nothing to half its readers.
    """
    if not alt.strip():
        raise ValidationError("a chart needs alternative text stating what it shows")
    body = "\n".join(f"  {mark}" for mark in marks)
    return (
        f'<svg viewBox="0 0 {_n(width)} {_n(height)}" role="img" '
        f'aria-label="{html.escape(alt, quote=True)}">\n{body}\n</svg>\n'
    )


def ticks(lo: float, hi: float, wanted: int) -> tuple[float, ...]:
    """Choose the values an axis is labelled at.

    Args:
        lo: Smallest value the axis shows.
        hi: Largest value it shows.
        wanted: Roughly how many intervals to divide it into.

    Returns:
        The tick values inside the range, in order, spaced by one, two or five times a
        power of ten. Empty if the range holds no such value.

    Raises:
        ValidationError: If the range is empty or `wanted` is not positive.
    """
    if not lo < hi:
        raise ValidationError(f"ticks need lo < hi, got lo={lo}, hi={hi}")
    if wanted < 1:
        raise ValidationError(f"ticks need a positive count, got {wanted}")
    rough = (hi - lo) / wanted
    decade = 10.0 ** math.floor(math.log10(rough))
    # The nearest readable step rather than the next one up, because rounding 25 up to 50
    # halves the number of ticks and leaves an axis labelled twice.
    step = min(
        (decade * unit for unit in (*_STEPS, 10.0)),
        key=lambda candidate: abs(math.log10(candidate / rough)),
    )
    first = math.ceil(lo / step)
    last = math.floor(hi / step)
    # Multiplying is what keeps the values exact: adding the step repeatedly turns 0.3
    # into 0.30000000000000004 and prints it.
    return tuple(index * step for index in range(first, last + 1))


def decade_ticks(lo: float, hi: float) -> tuple[float, ...]:
    """Choose the values a logarithmic axis is labelled at.

    Args:
        lo: Smallest value the axis shows. Must be positive.
        hi: Largest value it shows.

    Returns:
        The tick values inside the range, in order: one, two and five times each power of
        ten while the range is under two decades, and one and five beyond that, so that a
        wide axis is not labelled every eight millimetres.

    Raises:
        ValidationError: If the range is empty, inverted or reaches zero.
    """
    if lo <= 0.0:
        raise ValidationError(f"a logarithmic axis needs lo > 0, got lo={lo}")
    if not lo < hi:
        raise ValidationError(f"ticks need lo < hi, got lo={lo}, hi={hi}")
    wide = 2.0
    units = _STEPS if math.log10(hi / lo) < wide else (1.0, 5.0)
    first = math.floor(math.log10(lo))
    last = math.ceil(math.log10(hi))
    return tuple(
        value
        for power in range(first, last + 1)
        for unit in units
        if lo <= (value := unit * 10.0**power) <= hi
    )


def _inside(value: float, lo: float, hi: float) -> None:
    """Check one value against its axis.

    Raises:
        ValidationError: If the value is outside the axis, which is the only way a mark
            can leave the frame.
    """
    if not math.isfinite(value):
        raise ValidationError(f"cannot place {value} on an axis")
    if not lo <= value <= hi:
        raise ValidationError(f"{value} is outside the axis {lo} to {hi}")


def _n(value: float) -> str:
    """A coordinate, short enough to read in the markup and stable across platforms."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text
