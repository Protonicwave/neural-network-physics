"""Turning a run record into a report someone can read.

The record is built into a document once and written out twice, as Markdown and as a
single self contained HTML file. Rendering is a pure function of the record: no clock, no
environment, no dictionary that iterates in whatever order it feels like. The same record
gives byte identical output, which is what makes a diff between two reports mean
something.

Every number in the report is explained. That is not decoration. A reader who cannot tell
whether 0.4 is good has not been given a result, and the explanations are the part of
this repository a reader meets first.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError
from nnphysics.reporting.compare import Verdict
from nnphysics.reporting.document import (
    Block,
    Bullets,
    Document,
    Figure,
    Heading,
    Paragraph,
    Table,
    to_html,
    to_markdown,
)
from nnphysics.reporting.explain import TIMING_METRIC, Direction, explain, metric_summary
from nnphysics.reporting.layout import PLOTS_DIR

if TYPE_CHECKING:
    from nnphysics.evals.result import InvariantRecord, PredictorResult, SuiteResult
    from nnphysics.reporting.compare import Comparison, Delta
    from nnphysics.reporting.explain import Explanation
    from nnphysics.reporting.plots import PlotRecord
    from nnphysics.reporting.record import RunRecord

__all__ = [
    "HEADLINE",
    "MISSING",
    "build_comparison_document",
    "build_document",
    "format_value",
    "render_comparison_html",
    "render_comparison_markdown",
    "render_html",
    "render_markdown",
]

HEADLINE = (
    ("one_step_error", "error"),
    ("rollout_error", "error.final"),
    ("rollout_error", "error.max"),
    ("invariant_drift", "worst_violation"),
    ("symmetry_violation", "worst"),
    ("distribution_drift", "worst"),
    (TIMING_METRIC, "seconds_per_step"),
)
"""One number per metric for the table a reader looks at first. Everything else is below
it, and everything without exception is in the record."""

MISSING = "n/a"
"""Written where a predictor has no value for a scalar, so that a gap in a table is
visibly a gap rather than a zero."""

_DIGITS = 4
"""Significant figures. Enough to see a change, few enough to read a row."""


def format_value(value: float, explanation: Explanation | None = None) -> str:
    """Format one number for a table.

    Args:
        value: The number.
        explanation: What the number means, used to write out a sentinel in words rather
            than as the number that stands for it.

    Returns:
        The formatted cell.
    """
    if (
        explanation is not None
        and explanation.sentinel is not None
        and value == explanation.sentinel
    ):
        return explanation.sentinel_text or MISSING
    return f"{value:.{_DIGITS}g}"


def build_document(record: RunRecord, plots: Sequence[PlotRecord] = ()) -> Document:
    """Assemble the whole report.

    Args:
        record: The run record.
        plots: Figures already rendered, in the order they should appear. A figure whose
            name does not begin with a split name is shown in the closing section.

    Returns:
        The document.

    Raises:
        UnknownNameError: If the result carries a scalar nobody has explained.
    """
    result = record.evaluation
    declared = _declared(result)
    blocks: list[Block] = [
        Paragraph(_opening(record)),
        *_provenance(record),
        *_settings(result),
        *_training(record),
        *_reading(result, declared),
    ]
    for split in record.splits:
        blocks.extend(_split_section(result, split, declared, plots))
    blocks.extend(_gap_section(result, declared))
    blocks.extend(_incident_section(result))
    return Document(
        title=f"{record.name}: {result.system} on the {result.settings.name} suite",
        blocks=tuple(blocks),
    )


def render_markdown(record: RunRecord, plots: Sequence[PlotRecord] = ()) -> str:
    """Render the Markdown report.

    Args:
        record: The run record.
        plots: Figures already rendered.

    Returns:
        The Markdown, which references the plot files beside it.
    """
    return to_markdown(build_document(record, plots))


def render_html(record: RunRecord, plots: Sequence[PlotRecord] = (), *, plot_dir: Path) -> str:
    """Render the HTML report, with every image embedded.

    Args:
        record: The run record.
        plots: Figures already rendered.
        plot_dir: Directory the figures were written to, read so they can be embedded.

    Returns:
        A single file that references nothing outside itself.

    Raises:
        ValidationError: If a figure named in the document is not on disk. A report that
            silently dropped an image would be a report that quietly said less.
    """
    return to_html(build_document(record, plots), lambda source: _embed(plot_dir, source))


def build_comparison_document(
    comparisons: Sequence[Comparison],
    records: Mapping[str, RunRecord],
    plots: Sequence[PlotRecord] = (),
) -> Document:
    """Assemble a report comparing several runs against one baseline.

    Args:
        comparisons: One per candidate run, each against the same baseline.
        records: Run record by run identifier, for naming the runs.
        plots: Overlay figures already rendered.

    Returns:
        The document.

    Raises:
        ValidationError: If no comparisons were given.
    """
    if not comparisons:
        raise ValidationError("a comparison report needs at least one comparison")
    baseline = comparisons[0].baseline
    blocks: list[Block] = [
        Paragraph(
            f"Every run below is compared against {_label(records, baseline)}. Each row is "
            f"the candidate value minus the baseline value, together with the direction "
            f"that scalar counts as an improvement in. A change of "
            f"{comparisons[0].threshold:g} of the baseline or less is called unchanged."
        ),
        Heading("Summary"),
        Table(
            headers=("run", "improved", "regressed", "unchanged", "undirected"),
            rows=tuple(
                (
                    _label(records, comparison.candidate),
                    *(
                        str(comparison.summary[verdict])
                        for verdict in (
                            Verdict.IMPROVED,
                            Verdict.REGRESSED,
                            Verdict.UNCHANGED,
                            Verdict.UNDIRECTED,
                        )
                    ),
                )
                for comparison in comparisons
            ),
            caption="Scalars carried by both runs, counted by verdict.",
        ),
    ]
    for comparison in comparisons:
        blocks.extend(_comparison_section(comparison, records))
    blocks.extend(
        Figure(
            source=f"{PLOTS_DIR}/{plot.name}",
            alt=plot.title,
            caption=f"{plot.title}. {plot.caption}",
        )
        for plot in plots
    )
    return Document(title=f"Comparison against {_label(records, baseline)}", blocks=tuple(blocks))


def render_comparison_markdown(
    comparisons: Sequence[Comparison],
    records: Mapping[str, RunRecord],
    plots: Sequence[PlotRecord] = (),
) -> str:
    """Render the comparison as Markdown.

    Args:
        comparisons: One per candidate run.
        records: Run record by run identifier.
        plots: Overlay figures already rendered.

    Returns:
        The Markdown.
    """
    return to_markdown(build_comparison_document(comparisons, records, plots))


def render_comparison_html(
    comparisons: Sequence[Comparison],
    records: Mapping[str, RunRecord],
    plots: Sequence[PlotRecord] = (),
    *,
    plot_dir: Path,
) -> str:
    """Render the comparison as a single self contained HTML file.

    Args:
        comparisons: One per candidate run.
        records: Run record by run identifier.
        plots: Overlay figures already rendered.
        plot_dir: Directory the figures were written to.

    Returns:
        The HTML.
    """
    return to_html(
        build_comparison_document(comparisons, records, plots),
        lambda source: _embed(plot_dir, source),
    )


def _comparison_section(comparison: Comparison, records: Mapping[str, RunRecord]) -> list[Block]:
    """One candidate run against the baseline."""
    label = _label(records, comparison.candidate)
    blocks: list[Block] = [Heading(label)]
    regressions = comparison.regressions
    if regressions:
        blocks.append(
            Table(
                headers=("split", "predictor", "scalar", "baseline", "candidate", "change"),
                rows=tuple(_delta_row(delta) for delta in regressions),
                caption=(
                    f"Regressions: {len(regressions)} scalars moved the wrong way by more "
                    f"than {comparison.threshold:g} of the baseline."
                ),
            )
        )
    else:
        blocks.append(
            Paragraph(
                f"No scalar moved the wrong way by more than {comparison.threshold:g} of "
                f"the baseline."
            )
        )
    headline = [delta for delta in comparison.deltas if (delta.metric, delta.key) in set(HEADLINE)]
    blocks.append(
        Table(
            headers=(
                "split",
                "predictor",
                "scalar",
                "baseline",
                "candidate",
                "change",
                "better",
                "verdict",
            ),
            rows=tuple(
                (*_delta_row(delta), _direction(delta.direction), str(delta.verdict.value))
                for delta in headline
            ),
            caption="One scalar per metric. The record holds every other scalar compared.",
        )
    )
    return blocks


def _delta_row(delta: Delta) -> tuple[str, ...]:
    """The cells every delta table shares."""
    return (
        delta.split,
        delta.predictor,
        delta.scalar,
        format_value(delta.baseline),
        format_value(delta.candidate),
        format_value(delta.change),
    )


def _label(records: Mapping[str, RunRecord], run_id: str) -> str:
    """Name a run, falling back to its identifier when the record is not to hand."""
    record = records.get(run_id)
    return f"{record.name} ({run_id})" if record is not None else run_id


def _embed(plot_dir: Path, source: str) -> str:
    """Read one figure and return it as a data URI."""
    path = plot_dir / Path(source).name
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValidationError(f"cannot embed the figure {path}: {error}") from error
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _opening(record: RunRecord) -> str:
    """The paragraph that says what this run was."""
    return (
        f"Run {record.run_id} of the {record.evaluation.system} system, scored by the "
        f"{record.evaluation.settings.name} suite over {record.evaluation.settings.rollout_steps} "
        f"steps from {record.evaluation.settings.n_initial_conditions} initial conditions per "
        f"split. Every number below comes from the record written at {record.created}, and the "
        f"record is enough to reproduce it."
    )


def _provenance(record: RunRecord) -> list[Block]:
    """What produced the numbers."""
    packages = ", ".join(
        f"{name} {version}" for name, version in sorted(record.environment.packages.items())
    )
    rows = [
        ("run id", record.run_id),
        ("dataset id", record.evaluation.dataset_id),
        ("seed", str(record.evaluation.seed)),
        ("code version", record.code_version),
        ("commit", record.commit or MISSING),
        ("created", record.created),
        ("python", f"{record.environment.implementation} {record.environment.python}"),
        (
            "machine",
            f"{record.environment.platform}, {record.environment.machine}, "
            f"{record.environment.cpu_count} logical cores",
        ),
        ("libraries", packages),
    ]
    rows += [
        (f"{name} (seconds)", f"{seconds:.3f}") for name, seconds in sorted(record.timings.items())
    ]
    return [
        Heading("Provenance"),
        Table(
            headers=("what", "value"),
            rows=tuple((label, value) for label, value in rows),
            caption="Everything needed to say where these numbers came from.",
        ),
    ]


def _settings(result: SuiteResult) -> list[Block]:
    """The suite as it was configured."""
    settings = result.settings
    return [
        Heading("What was run"),
        Bullets(
            (
                f"Predictors: {', '.join(entry.spec for entry in _unique_specs(result))}.",
                f"Splits: {', '.join(_splits(result))}.",
                f"Metrics: {', '.join(settings.metrics)}.",
                f"Rollout of {settings.rollout_steps} steps from "
                f"{settings.n_initial_conditions} initial conditions per split.",
                f"Error thresholds a horizon is reported for: "
                f"{', '.join(f'{value:g}' for value in settings.error_thresholds)}.",
                f"Equivariance tested over {settings.symmetry_steps} steps; distributions "
                f"taken over the last {settings.distribution_window:g} of each rollout.",
            )
        ),
    ]


def _training(record: RunRecord) -> list[Block]:
    """How the model was trained, and what each stage of it bought.

    One row per curriculum stage rather than one per epoch. A reader wants to know
    whether lengthening the rollout helped, and forty rows of a falling loss answer that
    no better than three do. The full history is in the record for anything that does
    want every epoch.
    """
    history = record.training
    if history is None:
        return []
    stages: list[tuple[str, ...]] = []
    for steps in dict.fromkeys(entry.curriculum_steps for entry in history.epochs):
        during = [entry for entry in history.epochs if entry.curriculum_steps == steps]
        stages.append(
            (
                str(steps),
                f"{during[0].epoch}-{during[-1].epoch}",
                f"{during[0].validation_error:.4g}",
                f"{min(entry.validation_error for entry in during):.4g}",
                f"{sum(entry.seconds for entry in during):.0f}",
            )
        )
    stopped = (
        f"stopped early after {len(history.epochs)} of {record.config.training.epochs} epochs"
        if history.stopped_early
        else f"ran every one of {len(history.epochs)} epochs"
    )
    return [
        Heading("How it was trained"),
        Bullets(
            (
                f"Model {history.model} with {history.n_parameters} trainable parameters.",
                f"{history.train_windows} training windows per epoch, "
                f"{history.validation_windows} validation windows.",
                f"Selected on a {history.validation_steps} step validation rollout, the "
                f"same horizon in every epoch, and {stopped}.",
                f"Best at epoch {history.best_epoch} with a validation error of "
                f"{history.best_validation_error:.4g}, in {history.seconds:.0f} seconds.",
            )
        ),
        Table(
            headers=("rollout steps", "epochs", "validation at entry", "best", "seconds"),
            rows=tuple(stages),
            caption="One row per curriculum stage. A stage that lowers the best "
            "validation error earned its place; one that does not is capacity spent "
            "on nothing.",
        ),
    ]


def _reading(result: SuiteResult, declared: Mapping[str, InvariantRecord]) -> list[Block]:
    """The table that says what every number means."""
    blocks: list[Block] = [
        Heading("How to read these numbers"),
        Paragraph(
            "Every scalar the suite produced is described here, with its units and with "
            "the direction that counts as an improvement. A number whose meaning is not "
            "stated is not a result."
        ),
    ]
    for metric in _metrics(result):
        keys = sorted(_keys(result, metric))
        blocks.append(Heading(metric, level=3))
        blocks.append(Paragraph(metric_summary(metric)))
        described = [explain(metric, key, declared) for key in keys]
        blocks.append(
            Table(
                headers=("scalar", "units", "better", "meaning"),
                rows=tuple(
                    (entry.key, entry.units, _direction(entry.direction), entry.text)
                    for entry in described
                ),
            )
        )
    timing = explain(TIMING_METRIC, "seconds_per_step")
    blocks.append(Heading(TIMING_METRIC, level=3))
    blocks.append(Paragraph(metric_summary(TIMING_METRIC)))
    blocks.append(
        Table(
            headers=("scalar", "units", "better", "meaning"),
            rows=((timing.key, timing.units, _direction(timing.direction), timing.text),),
        )
    )
    return blocks


def _split_section(
    result: SuiteResult,
    split: str,
    declared: Mapping[str, InvariantRecord],
    plots: Sequence[PlotRecord],
) -> list[Block]:
    """Everything measured on one split."""
    entries = [entry for entry in result.results if entry.split == split]
    if not entries:
        return []
    blocks: list[Block] = [Heading(f"Results on the {split} split")]
    blocks.append(_headline_table(entries, declared))
    for metric in _metrics(result):
        keys = sorted(_keys(result, metric))
        if not keys:
            continue
        blocks.append(
            Table(
                headers=("predictor", *keys),
                rows=tuple(
                    (
                        entry.predictor,
                        *(_cell(entry, metric, key, declared) for key in keys),
                    )
                    for entry in entries
                ),
                caption=f"{metric}, averaged over the initial conditions.",
            )
        )
    blocks.extend(
        Figure(
            source=f"{PLOTS_DIR}/{plot.name}",
            alt=plot.title,
            caption=f"{plot.title}. {plot.caption}",
        )
        for plot in plots
        if plot.name.startswith(f"{split}-")
    )
    return blocks


def _headline_table(
    entries: Sequence[PredictorResult], declared: Mapping[str, InvariantRecord]
) -> Table:
    """One number per metric, for the reader who reads one table."""
    columns = [
        (metric, key)
        for metric, key in HEADLINE
        if metric == TIMING_METRIC
        or any(_value(entry, metric, key) is not None for entry in entries)
    ]
    return Table(
        headers=("predictor", *(f"{metric}.{key}" for metric, key in columns)),
        rows=tuple(
            (entry.predictor, *(_cell(entry, metric, key, declared) for metric, key in columns))
            for entry in entries
        ),
        caption="One number per metric. Lower is better everywhere here except the horizons.",
    )


def _gap_section(result: SuiteResult, declared: Mapping[str, InvariantRecord]) -> list[Block]:
    """How much worse everything is outside the training regimes."""
    if not result.regime_gap:
        return []
    rows: list[tuple[str, ...]] = []
    for name in sorted(result.regime_gap):
        predictor, _, scalar = name.partition(".")
        metric, _, key = scalar.partition(".")
        explanation = explain(metric, key, declared)
        if explanation.direction is Direction.NEUTRAL:
            continue
        rows.append(
            (
                predictor,
                scalar,
                format_value(result.regime_gap[name]),
                _worse(result.regime_gap[name], explanation.direction),
            )
        )
    return [
        Heading("Outside the training regimes"),
        Paragraph(
            "The held out regime is never seen during training or model selection. Each "
            "row is the held out value minus the in distribution value, so the sign is "
            "read the same way for every quantity: positive means the held out regimes "
            "are worse for anything a predictor wants small."
        ),
        Table(
            headers=("predictor", "scalar", "gap", "held out is"),
            rows=tuple(rows),
            caption="Scalars whose direction of improvement is stated. The record holds the rest.",
        ),
    ]


def _incident_section(result: SuiteResult) -> list[Block]:
    """Rollouts that did not finish, which a mean would otherwise hide."""
    rows = tuple(
        (
            entry.predictor,
            rollout.split,
            rollout.trajectory,
            rollout.regime,
            f"{rollout.steps_completed} of {rollout.steps_requested}",
            rollout.stop_reason,
            rollout.detail or "",
        )
        for entry in result.results
        for rollout in entry.rollouts
        if rollout.steps_completed != rollout.steps_requested
    )
    if not rows:
        return [
            Heading("Rollouts that stopped early"),
            Paragraph(
                "None. Every predictor reached the requested horizon from every initial condition."
            ),
        ]
    return [
        Heading("Rollouts that stopped early"),
        Paragraph(
            "A rollout that stops is recorded rather than scored to the end. The metrics "
            "for these predictors are computed over the steps that were taken, so they "
            "describe less of the horizon than the others."
        ),
        Table(
            headers=("predictor", "split", "trajectory", "regime", "steps", "reason", "detail"),
            rows=rows,
        ),
    ]


def _cell(
    entry: PredictorResult, metric: str, key: str, declared: Mapping[str, InvariantRecord]
) -> str:
    """One formatted number, or a visible gap."""
    value = _value(entry, metric, key)
    if value is None:
        return MISSING
    explanation = (
        explain(metric, key, declared) if metric != TIMING_METRIC else explain(metric, key)
    )
    return format_value(value, explanation)


def _value(entry: PredictorResult, metric: str, key: str) -> float | None:
    """One scalar, or `None` if this predictor has no such number."""
    if metric == TIMING_METRIC:
        return entry.seconds_per_step if key == "seconds_per_step" else None
    try:
        return entry.scalar(metric, key)
    except KeyError:
        return None


def _worse(gap: float, direction: Direction) -> str:
    """Whether a gap means the held out regime did worse, better or neither."""
    if gap == 0.0:
        return "the same"
    if direction is Direction.LOWER:
        return "worse" if gap > 0.0 else "better"
    return "better" if gap > 0.0 else "worse"


def _direction(direction: Direction) -> str:
    """The direction of improvement, for a table cell."""
    return str(direction.value)


def _metrics(result: SuiteResult) -> list[str]:
    """Metrics present, in the order the suite named them."""
    present = {record.name for entry in result.results for record in entry.metrics}
    ordered = [name for name in result.settings.metrics if name in present]
    return ordered + sorted(present - set(ordered))


def _keys(result: SuiteResult, metric: str) -> set[str]:
    """Every scalar name one metric produced anywhere in the result."""
    return {
        key
        for entry in result.results
        for record in entry.metrics
        if record.name == metric
        for key in record.scalars
    }


def _splits(result: SuiteResult) -> list[str]:
    """Splits present, in the order they were evaluated."""
    seen: list[str] = []
    for entry in result.results:
        if entry.split not in seen:
            seen.append(entry.split)
    return seen


def _unique_specs(result: SuiteResult) -> list[PredictorResult]:
    """One entry per predictor, the first time it appears."""
    seen: set[str] = set()
    unique: list[PredictorResult] = []
    for entry in result.results:
        if entry.predictor not in seen:
            seen.add(entry.predictor)
            unique.append(entry)
    return unique


def _declared(result: SuiteResult) -> dict[str, InvariantRecord]:
    """Every declared invariant by name, taking the first declaration of each."""
    declared: dict[str, InvariantRecord] = {}
    for records in result.invariants.values():
        for record in records:
            declared.setdefault(record.name, record)
    return declared
