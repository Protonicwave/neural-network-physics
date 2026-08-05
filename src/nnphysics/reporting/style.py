"""Plot styling, defined once.

Every figure in every report goes through here, so that two plots of the same quantity
look the same and a reader can compare them without first working out which is which. The
palette is the Okabe and Ito set, which stays distinguishable under the common forms of
colour blindness and in greyscale print.

The backend is fixed to Agg at import. A report is rendered to files, often on a machine
with no display and often from a test, and choosing the backend by environment would make
the output depend on where it was produced. Figures are saved with their software stamp
suppressed for the same reason: a report has to be byte identical when nothing about the
run has changed, and a version string in an image header would break that for no gain.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "DPI",
    "FIGURE_SIZE",
    "PALETTE",
    "REFERENCE_COLOUR",
    "colour",
    "figure",
    "save",
    "style",
]

PALETTE = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#56b4e9",
    "#e69f00",
    "#8c564b",
    "#666666",
)
"""Series colours, in the order they are handed out."""

REFERENCE_COLOUR = "#000000"
"""Ground truth is black everywhere. It is not one series among others: it is the thing
the others are measured against."""

FIGURE_SIZE = (7.2, 4.2)
"""Inches. Wide enough for a legend beside the curves, short enough to sit in a report."""

DPI = 110
"""Enough to read a label on a screen without making an embedded image heavy."""

# Typed loosely on purpose: the stub for rc_context takes a mapping keyed by a literal
# union of every setting matplotlib has, which no hand written dictionary can satisfy.
_RC: Any = {
    "figure.figsize": FIGURE_SIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.4,
    "image.cmap": "RdBu_r",
}
"""The whole of the house style. Set through a context rather than globally, so that
importing this module does not change how somebody else's plot looks."""


def colour(index: int) -> str:
    """The colour of one series.

    Args:
        index: Position of the series. Wraps around the palette.

    Returns:
        A hexadecimal colour.
    """
    return PALETTE[index % len(PALETTE)]


@contextmanager
def style() -> Iterator[None]:
    """Apply the house style for the duration of a block."""
    with plt.rc_context(_RC):
        yield


@contextmanager
def figure(
    nrows: int = 1, ncols: int = 1, *, size: tuple[float, float] | None = None
) -> Iterator[tuple[Figure, Any]]:
    """Create a styled figure and close it afterwards.

    Closing matters more than it looks: a report renders many figures, and a figure that
    is not closed stays alive in the pyplot registry until the process ends.

    Args:
        nrows: Rows of axes.
        ncols: Columns of axes.
        size: Figure size in inches, defaulting to the house size scaled by the grid.

    Yields:
        The figure and its axes, as `subplots` returns them.
    """
    with style():
        width, height = size or (FIGURE_SIZE[0], FIGURE_SIZE[1] * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
        try:
            yield fig, axes
        finally:
            plt.close(fig)


def save(fig: Figure, path: Path) -> None:
    """Write a figure, reproducibly.

    Args:
        fig: The figure.
        path: File to write. Its parent must exist.
    """
    fig.savefig(path, format="png", metadata={"Software": None})


def label_axes(axes: Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the three labels every axes in a report carries.

    Args:
        axes: The axes.
        title: What it shows.
        xlabel: Horizontal label, units included.
        ylabel: Vertical label, units included.
    """
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
