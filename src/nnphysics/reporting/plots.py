"""The figures a report carries.

Four of them, each answering a question a table cannot. How fast does the error grow.
Does the predictor drift out of the band the system declared. Is the average error a fair
summary or is one trajectory of four carrying it. And what does being wrong actually look
like.

Nothing here knows which system it is looking at. The qualitative plot chooses how to
draw a field from the shape of the field: a list of two component vectors is drawn in the
plane, a two dimensional array is drawn as an image, anything else is drawn as a line.
That is the same trick the metrics use, one layer further out. A plot that branched on
whether it was N-body or fluid would put system knowledge in the outermost layer of all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.evals.predictors import REFERENCE_NAME
from nnphysics.reporting.explain import explain
from nnphysics.reporting.style import REFERENCE_COLOUR, colour, figure, label_axes, save

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.core.types import FloatArray
    from nnphysics.evals.result import InvariantRecord, PredictorResult, SuiteResult
    from nnphysics.evals.snapshots import Snapshot, SnapshotSet

__all__ = [
    "DISTRIBUTION_SCALAR",
    "PlotRecord",
    "distribution_plot",
    "drift_plot",
    "error_plot",
    "overlay_plot",
    "render_plots",
    "snapshot_plot",
]

DISTRIBUTION_SCALAR = "rollout_error.error.final"
"""The per rollout scalar the spread across trajectories is drawn from. The error at the
end of the rollout, because that is the number a claim about a surrogate rests on."""

_PLANE_COMPONENTS = 2
"""A field with this many components per element is drawn as points in the plane."""

_JITTER = 0.11
"""Horizontal separation between the trajectories of one predictor, so that two equal
values do not hide each other."""


@dataclass(frozen=True, slots=True)
class PlotRecord:
    """One rendered figure.

    Attributes:
        name: File name, relative to the plot directory.
        title: Heading it appears under.
        caption: What it shows and how to read it.
    """

    name: str
    title: str
    caption: str


def error_plot(result: SuiteResult, split: str, path: Path) -> PlotRecord | None:
    """Draw the error of every predictor against the horizon.

    Args:
        result: The evaluation result.
        split: Split to draw.
        path: File to write.

    Returns:
        The record, or `None` if no predictor on that split carries an error curve.
    """
    drawn: list[tuple[PredictorResult, FloatArray, FloatArray | None]] = []
    for entry in _on_split(result, split):
        error = _series(entry, "rollout_error", "error")
        if error is not None:
            drawn.append((entry, error, _series(entry, "rollout_error", "time")))
    if not drawn:
        return None

    flat: list[str] = []
    with figure() as (fig, axes):
        target = axes[0][0]
        for index, (entry, error, time) in enumerate(drawn):
            horizon = time if time is not None else np.arange(len(error), dtype=np.float64)
            if not np.any(error > 0.0):
                flat.append(entry.predictor)
                continue
            target.plot(
                horizon,
                np.where(error > 0.0, error, np.nan),
                color=_colour(entry.predictor, index),
                label=entry.predictor,
            )
        target.set_yscale("log")
        label_axes(
            target,
            title=f"Error growth, {split} split",
            xlabel="simulated time",
            ylabel="error, relative to the size of the true state",
        )
        target.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        save(fig, path)

    caption = (
        "Normalised error against horizon, averaged over the initial conditions. One "
        "means the predictor is as wrong as the signal is large."
    )
    if flat:
        verb = "is" if len(flat) == 1 else "are"
        caption += (
            f" {_join(flat)} {verb} exactly right at every step, which a logarithmic "
            f"axis cannot show."
        )
    return PlotRecord(path.name, f"Error growth on the {split} split", caption)


def drift_plot(result: SuiteResult, split: str, path: Path) -> PlotRecord | None:
    """Draw how far each predictor takes every invariant from the truth.

    Args:
        result: The evaluation result.
        split: Split to draw.
        path: File to write.

    Returns:
        The record, or `None` if no invariant curve was recorded on that split.
    """
    entries = _on_split(result, split)
    declared = _declared(result)
    names = sorted(
        {
            key[: -len(".predicted")]
            for entry in entries
            for record in entry.metrics
            if record.name == "invariant_drift"
            for key in record.series
            if key.endswith(".predicted")
        }
    )
    if not names:
        return None

    with figure(nrows=len(names)) as (fig, axes):
        for row, name in enumerate(names):
            target = axes[row][0]
            invariant = declared.get(name)
            tolerance = invariant.rtol if invariant is not None else 1.0
            for index, entry in enumerate(entries):
                gap = _gap(entry, name)
                if gap is None:
                    continue
                target.plot(
                    np.arange(len(gap), dtype=np.float64),
                    gap,
                    color=_colour(entry.predictor, index),
                    label=entry.predictor if row == 0 else None,
                )
            target.axhspan(-tolerance, tolerance, color="#000000", alpha=0.08)
            target.set_yscale("symlog", linthresh=tolerance)
            units = invariant.dimension if invariant is not None else "declared units"
            label_axes(
                target,
                title=f"{name}, declared {_conservation(invariant)} in {units}",
                xlabel="step",
                ylabel="gap from the truth,\nrelative to the scale of the invariant",
            )
            if row == 0:
                target.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        save(fig, path)

    return PlotRecord(
        path.name,
        f"Invariant drift on the {split} split",
        "How far each predictor takes a declared invariant from where the physics takes "
        "it, as a fraction of the size of that invariant. The shaded band is the "
        "tolerance the system declared, and the vertical axis is linear inside it and "
        "logarithmic outside.",
    )


def distribution_plot(
    result: SuiteResult, split: str, path: Path, *, scalar: str = DISTRIBUTION_SCALAR
) -> PlotRecord | None:
    """Draw one point per trajectory, so a mean cannot hide an outlier.

    Args:
        result: The evaluation result.
        split: Split to draw.
        path: File to write.
        scalar: Per rollout scalar to draw, keyed `metric.scalar`.

    Returns:
        The record, or `None` if no rollout on that split recorded that scalar. Results
        written before rollouts carried their own scalars have none.
    """
    entries = [
        entry
        for entry in _on_split(result, split)
        if any(scalar in rollout.scalars for rollout in entry.rollouts)
    ]
    if not entries:
        return None

    metric, _, key = scalar.partition(".")
    explanation = explain(metric, key, _declared(result))
    flat: list[str] = []
    with figure() as (fig, axes):
        target = axes[0][0]
        for index, entry in enumerate(entries):
            values = [
                rollout.scalars[scalar] for rollout in entry.rollouts if scalar in rollout.scalars
            ]
            if not any(value > 0.0 for value in values):
                flat.append(entry.predictor)
                continue
            offsets = _offsets(len(values))
            target.scatter(
                [index + offset for offset in offsets],
                values,
                color=_colour(entry.predictor, index),
                s=26,
                zorder=3,
            )
            target.plot(
                [index - 0.25, index + 0.25],
                [float(np.mean(values))] * 2,
                color=_colour(entry.predictor, index),
                linewidth=2.0,
            )
        target.set_xticks(range(len(entries)))
        target.set_xticklabels([entry.predictor for entry in entries], rotation=30, ha="right")
        target.set_yscale("log")
        label_axes(
            target,
            title=f"Spread across trajectories, {split} split",
            xlabel="",
            ylabel=f"{explanation.title} ({explanation.units})",
        )
        save(fig, path)

    caption = (
        f"One point per initial condition, with the mean as a bar. A predictor whose "
        f"points are spread over decades has a {explanation.title} that its mean does "
        f"not describe."
    )
    if flat:
        verb = "is" if len(flat) == 1 else "are"
        caption += f" {_join(flat)} {verb} exactly zero, which a logarithmic axis cannot show."
    return PlotRecord(path.name, f"Spread across trajectories on the {split} split", caption)


def snapshot_plot(snapshot: Snapshot, path: Path) -> PlotRecord | None:
    """Draw the predicted state beside the true one at several horizons.

    Args:
        snapshot: The kept states.
        path: File to write.

    Returns:
        The record, or `None` if no field of the state can be drawn.
    """
    name = _primary_field(snapshot)
    if name is None:
        return None
    predicted = np.asarray(snapshot.predicted[name], dtype=np.float64)
    reference = np.asarray(snapshot.reference[name], dtype=np.float64)
    shape = predicted.shape[1:]
    columns = predicted.shape[0]

    if len(shape) == _PLANE_COMPONENTS and shape[-1] == _PLANE_COMPONENTS:
        drawn = _plane_panels(predicted, reference, snapshot, path, name)
    elif len(shape) == _PLANE_COMPONENTS:
        drawn = _image_panels(predicted, reference, snapshot, path, name)
    else:
        drawn = _line_panels(predicted, reference, snapshot, path, name)

    return PlotRecord(
        path.name,
        f"{snapshot.predictor} against the truth",
        f"The field {name} at {columns} horizons of one {snapshot.split} trajectory, "
        f"{snapshot.trajectory}. {drawn}",
    )


def overlay_plot(
    curves: Sequence[tuple[str, FloatArray, FloatArray | None]], title: str, path: Path
) -> PlotRecord:
    """Draw one labelled error curve per run, on shared axes.

    Args:
        curves: Label, error curve and time axis for each run. A missing time axis is
            replaced by the step index.
        title: Heading and axes title.
        path: File to write.

    Returns:
        The record.
    """
    with figure() as (fig, axes):
        target = axes[0][0]
        for index, (label, error, time) in enumerate(curves):
            horizon = time if time is not None else np.arange(len(error), dtype=np.float64)
            target.plot(
                horizon,
                np.where(error > 0.0, error, np.nan),
                color=colour(index),
                label=label,
            )
        target.set_yscale("log")
        label_axes(
            target,
            title=title,
            xlabel="simulated time",
            ylabel="error, relative to the size of the true state",
        )
        target.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        save(fig, path)
    return PlotRecord(
        path.name,
        title,
        "The same error curve from each run, so that a change of shape is visible and "
        "not only a change of the final number.",
    )


def render_plots(
    result: SuiteResult, snapshots: SnapshotSet | None, directory: Path
) -> tuple[PlotRecord, ...]:
    """Draw every figure a report needs.

    Args:
        result: The evaluation result.
        snapshots: Kept states, or `None` if none were captured.
        directory: Directory to write into. It must exist.

    Returns:
        One record per figure written, in the order a report should show them.
    """
    records: list[PlotRecord] = []
    for split in _splits(result):
        drawn = (
            error_plot(result, split, directory / f"{split}-error.png"),
            drift_plot(result, split, directory / f"{split}-invariant-drift.png"),
            distribution_plot(result, split, directory / f"{split}-spread.png"),
        )
        records.extend(record for record in drawn if record is not None)
        for snapshot in snapshots.snapshots if snapshots else ():
            if snapshot.split != split:
                continue
            record = snapshot_plot(snapshot, directory / f"{split}-state-{snapshot.predictor}.png")
            if record is not None:
                records.append(record)
    return tuple(records)


def _plane_panels(
    predicted: FloatArray, reference: FloatArray, snapshot: Snapshot, path: Path, name: str
) -> str:
    """Points in the plane, predicted over true, one panel per horizon."""
    columns = predicted.shape[0]
    limit = float(np.max(np.abs(reference))) * 1.15 or 1.0
    with figure(ncols=columns, size=(2.4 * columns, 2.6)) as (fig, axes):
        for column in range(columns):
            target = axes[0][column]
            target.scatter(
                reference[column, :, 0], reference[column, :, 1], s=14, color=REFERENCE_COLOUR
            )
            target.scatter(
                predicted[column, :, 0],
                predicted[column, :, 1],
                s=26,
                facecolors="none",
                edgecolors=colour(0),
            )
            target.set_xlim(-limit, limit)
            target.set_ylim(-limit, limit)
            target.set_aspect("equal")
            target.set_title(f"t = {snapshot.times[column]:.3g}")
        save(fig, path)
    return f"True {name} is filled black, predicted {name} is open blue."


def _image_panels(
    predicted: FloatArray, reference: FloatArray, snapshot: Snapshot, path: Path, name: str
) -> str:
    """Two rows of images, the truth above and the prediction below."""
    columns = predicted.shape[0]
    with figure(nrows=2, ncols=columns, size=(2.4 * columns, 5.0)) as (fig, axes):
        for column in range(columns):
            # One scale per column, set by the truth at that time. A single scale across
            # the row would leave a decaying field as a blank panel, which says only that
            # it decayed and nothing about whether it decayed into the right shape.
            limit = float(np.max(np.abs(reference[column]))) or 1.0
            for row, values in ((0, reference), (1, predicted)):
                target = axes[row][column]
                target.imshow(values[column], vmin=-limit, vmax=limit, origin="lower")
                target.set_xticks([])
                target.set_yticks([])
                target.grid(visible=False)
            axes[0][column].set_title(f"t = {snapshot.times[column]:.3g}")
        axes[0][0].set_ylabel("true")
        axes[1][0].set_ylabel(snapshot.predictor)
        save(fig, path)
    return (
        f"The top row is the truth and the bottom row is the prediction. Each column has "
        f"its own colour scale, set by the true {name} at that time, so a field that "
        f"decays still shows its structure."
    )


def _line_panels(
    predicted: FloatArray, reference: FloatArray, snapshot: Snapshot, path: Path, name: str
) -> str:
    """One panel per horizon, the two states as lines over the element index."""
    columns = predicted.shape[0]
    with figure(ncols=columns, size=(2.4 * columns, 2.6)) as (fig, axes):
        for column in range(columns):
            target = axes[0][column]
            index = np.arange(predicted.shape[1], dtype=np.float64)
            target.plot(index, reference[column].reshape(-1), color=REFERENCE_COLOUR)
            target.plot(index, predicted[column].reshape(-1), color=colour(0))
            target.set_title(f"t = {snapshot.times[column]:.3g}")
        save(fig, path)
    return f"The true {name} is black and the predicted one is blue, over the element index."


def _primary_field(snapshot: Snapshot) -> str | None:
    """The field a qualitative plot shows: the largest structure the state carries.

    Highest rank first, then most elements, then the name, so that the choice does not
    depend on the order a system happened to declare its fields in.
    """
    candidates = [
        (
            -np.asarray(snapshot.predicted[name]).ndim,
            -np.asarray(snapshot.predicted[name]).size,
            name,
        )
        for name in snapshot.names
        if np.asarray(snapshot.predicted[name]).ndim > 1
    ]
    return min(candidates)[2] if candidates else None


def _on_split(result: SuiteResult, split: str) -> list[PredictorResult]:
    """Every predictor result on one split, in suite order."""
    return [entry for entry in result.results if entry.split == split]


def _splits(result: SuiteResult) -> list[str]:
    """Splits present, in the order they were evaluated."""
    seen: list[str] = []
    for entry in result.results:
        if entry.split not in seen:
            seen.append(entry.split)
    return seen


def _series(entry: PredictorResult, metric: str, key: str) -> FloatArray | None:
    """One recorded curve, or `None` if it was not recorded."""
    for record in entry.metrics:
        if record.name == metric and key in record.series:
            return np.asarray(record.series[key], dtype=np.float64)
    return None


def _gap(entry: PredictorResult, invariant: str) -> FloatArray | None:
    """A predictor's invariant minus the truth's, as a fraction of the invariant's size."""
    predicted = _series(entry, "invariant_drift", f"{invariant}.predicted")
    reference = _series(entry, "invariant_drift", f"{invariant}.reference")
    if predicted is None or reference is None or len(predicted) != len(reference):
        return None
    try:
        scale = entry.scalar("invariant_drift", f"{invariant}.scale")
    except KeyError:
        return None
    gap: FloatArray = (predicted - reference) / (scale or 1.0)
    return gap


def _declared(result: SuiteResult) -> dict[str, InvariantRecord]:
    """Every declared invariant by name, taking the first declaration of each."""
    declared: dict[str, InvariantRecord] = {}
    for records in result.invariants.values():
        for record in records:
            declared.setdefault(record.name, record)
    return declared


def _conservation(invariant: InvariantRecord | None) -> str:
    """How the system said the invariant behaves."""
    return invariant.conservation if invariant is not None else "undeclared"


def _colour(predictor: str, index: int) -> str:
    """Ground truth is black, everything else takes the next palette colour."""
    return REFERENCE_COLOUR if predictor == REFERENCE_NAME else colour(index)


def _offsets(count: int) -> list[float]:
    """Horizontal offsets that spread `count` points around zero, deterministically."""
    return [(position - (count - 1) / 2.0) * _JITTER for position in range(count)]


def _join(names: Sequence[str]) -> str:
    """Join names for a sentence: `a`, `a and b`, `a, b and c`."""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"
