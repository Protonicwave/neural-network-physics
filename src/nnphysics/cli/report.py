"""The `nnp report` subcommands.

Rendering, comparing and listing are all side effects, so they live here. What they call
is pure: a record goes in, a document comes out, and the only thing this module adds is
deciding which files to read and where to put the result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import numpy as np
import typer

from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError
from nnphysics.evals.snapshots import read_snapshots
from nnphysics.reporting.compare import DEFAULT_THRESHOLD, Verdict, compare_series
from nnphysics.reporting.index import KEY_SCALARS, index_runs
from nnphysics.reporting.landing import LANDING_NAME, render_landing
from nnphysics.reporting.layout import (
    HTML_NAME,
    MARKDOWN_NAME,
    PLOTS_DIR,
    RunPaths,
    find_record,
    run_paths,
)
from nnphysics.reporting.page import build_page
from nnphysics.reporting.plots import PlotRecord, overlay_plot, render_plots
from nnphysics.reporting.record import RunRecord, read_record
from nnphysics.reporting.render import (
    render_comparison_html,
    render_comparison_markdown,
    render_html,
    render_markdown,
)

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray

__all__ = ["app"]

app = typer.Typer(
    name="report",
    help="Render run reports, compare runs and list the history.",
    no_args_is_help=True,
)

RootOption = Annotated[
    Path,
    typer.Option("--root", help="Directory holding one subdirectory per run."),
]

_EXIT_FAILURE = 1
_EXIT_USAGE = 2

_DEFAULT_ROOT = Path("runs")
_COMPARISON_DIR = "comparison"
_MINIMUM_RUNS = 2

_SCORES = Path("docs/results/fault-scores.json")
"""The one committed data file the landing page reads. Looked for beside the working
directory, as the agent configuration is, so that building the page from a checkout picks
it up and building it from anywhere else leaves the diagnosis section out rather than
failing."""


@app.command()
def render(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c", help="YAML run configuration naming the run.", show_default=False
        ),
    ] = None,
    run: Annotated[
        str | None,
        typer.Option("--run", help="Run identifier, looked up under the root.", show_default=False),
    ] = None,
    root: RootOption = _DEFAULT_ROOT,
) -> None:
    """Render the Markdown and HTML reports for one run."""
    record_path = _locate(config, run, root)
    try:
        record = read_record(record_path)
        paths = RunPaths(record_path.parent)
        paths.ensure()
        snapshots = read_snapshots(paths.snapshots) if paths.snapshots.is_file() else None
        plots = render_plots(record.evaluation, snapshots, paths.plots, record.benchmark)
        paths.markdown.write_text(render_markdown(record, plots), encoding="utf-8")
        paths.html.write_text(render_html(record, plots, plot_dir=paths.plots), encoding="utf-8")
    except NNPhysicsError as error:
        _fail(str(error))

    typer.echo(f"Rendered {len(plots)} plots for run {record.run_id}.")
    typer.echo(f"Wrote {paths.markdown} and {paths.html}.")


@app.command()
def page(
    root: RootOption = _DEFAULT_ROOT,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=f"Where to write the page. Defaults to {LANDING_NAME} in the runs root.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Build the landing page over every run under the root."""
    scores = _SCORES if _SCORES.is_file() else None
    destination = output if output is not None else root / LANDING_NAME
    try:
        model = build_page(root, scores=scores)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_landing(model), encoding="utf-8")
    except NNPhysicsError as error:
        _fail(str(error))

    if scores is None:
        typer.echo(f"No {_SCORES}, so the page carries no diagnosis section.")
    typer.echo(f"Wrote {destination} from {len(model.reported)} runs.")


@app.command()
def compare(
    runs: Annotated[
        list[str],
        typer.Argument(help="Two or more run identifiers. The first is the baseline."),
    ],
    root: RootOption = _DEFAULT_ROOT,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Relative change below which a scalar is unchanged."),
    ] = DEFAULT_THRESHOLD,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Directory to write the comparison into.", show_default=False
        ),
    ] = None,
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression", help="Exit non zero if any scalar regressed."),
    ] = False,
) -> None:
    """Compare two or more runs against the first of them."""
    if len(runs) < _MINIMUM_RUNS:
        typer.echo("a comparison needs at least two run identifiers", err=True)
        raise typer.Exit(code=_EXIT_USAGE)

    directory = output or root / _COMPARISON_DIR
    try:
        records = [read_record(find_record(root, identifier)) for identifier in runs]
        comparisons = compare_series(records, threshold=threshold)
        by_id = {record.run_id: record for record in records}
        (directory / PLOTS_DIR).mkdir(parents=True, exist_ok=True)
        plots = _overlays(records, directory / PLOTS_DIR)
        (directory / MARKDOWN_NAME).write_text(
            render_comparison_markdown(comparisons, by_id, plots), encoding="utf-8"
        )
        (directory / HTML_NAME).write_text(
            render_comparison_html(comparisons, by_id, plots, plot_dir=directory / PLOTS_DIR),
            encoding="utf-8",
        )
    except NNPhysicsError as error:
        _fail(str(error))

    regressions = 0
    for comparison in comparisons:
        counts = comparison.summary
        regressions += counts[Verdict.REGRESSED]
        typer.echo(
            f"{comparison.candidate} against {comparison.baseline}: "
            f"{counts[Verdict.IMPROVED]} improved, {counts[Verdict.REGRESSED]} regressed, "
            f"{counts[Verdict.UNCHANGED]} unchanged."
        )
        for delta in comparison.regressions[:5]:
            typer.echo(
                f"  {delta.split} {delta.predictor} {delta.scalar}: "
                f"{delta.baseline:.4g} to {delta.candidate:.4g}"
            )
    typer.echo(f"Wrote {directory / MARKDOWN_NAME} and {directory / HTML_NAME}.")
    if fail_on_regression and regressions:
        raise typer.Exit(code=_EXIT_FAILURE)


@app.command("list")
def list_runs(
    root: RootOption = _DEFAULT_ROOT,
    predictor: Annotated[
        str | None,
        typer.Option("--predictor", "-p", help="Show only this predictor.", show_default=False),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option("--split", "-s", help="Show only this split.", show_default=False),
    ] = None,
) -> None:
    """List every run under the root with its key metrics."""
    try:
        summaries = index_runs(root, predictor=predictor, split=split)
    except NNPhysicsError as error:
        _fail(str(error))

    if not summaries:
        typer.echo(f"No runs under {root}.")
        return
    keys = [f"{metric}.{key}" for metric, key in KEY_SCALARS]
    typer.echo(
        f"{'run':<28} {'system':<8} {'split':<9} {'predictor':<20} {'usable':>8} "
        + "  ".join(f"{key:>28}" for key in keys)
        + f"  {'verdict':<28}"
    )
    for summary in summaries:
        numbers = "  ".join(
            f"{summary.values[key]:>28.4g}" if key in summary.values else f"{'n/a':>28}"
            for key in keys
        )
        usable = f"{summary.usable:>8.4g}" if summary.usable is not None else f"{'n/a':>8}"
        typer.echo(
            f"{summary.label:<28} {summary.system:<8} {summary.split:<9} "
            f"{summary.predictor:<20} {usable} {numbers}  {summary.verdict:<28}"
        )


def _overlays(records: list[RunRecord], directory: Path) -> tuple[PlotRecord, ...]:
    """One error curve overlay per split every run measured."""
    common = [
        split
        for split in records[0].splits
        if all(split in record.splits for record in records[1:])
    ]
    plots: list[PlotRecord] = []
    for split in common:
        for name in records[0].predictors:
            curves = _curves(records, split, name)
            if len(curves) == len(records):
                plots.append(
                    overlay_plot(
                        curves,
                        f"{name} on the {split} split",
                        directory / f"{split}-{name}-error.png",
                    )
                )
    return tuple(plots)


def _curves(
    records: list[RunRecord], split: str, predictor: str
) -> list[tuple[str, FloatArray, FloatArray | None]]:
    """The error curve of one predictor on one split, from every run that has it."""
    curves: list[tuple[str, FloatArray, FloatArray | None]] = []
    for record in records:
        for entry in record.evaluation.results:
            if entry.predictor != predictor or entry.split != split:
                continue
            for metric in entry.metrics:
                if metric.name == "rollout_error" and "error" in metric.series:
                    time = metric.series.get("time")
                    curves.append(
                        (
                            f"{record.name} ({record.run_id})",
                            np.asarray(metric.series["error"], dtype=np.float64),
                            np.asarray(time, dtype=np.float64) if time is not None else None,
                        )
                    )
    return curves


def _locate(config: Path | None, run: str | None, root: Path) -> Path:
    """Find the record to render, from a configuration or from a run identifier."""
    if config is not None:
        return run_paths(_resolve(config)).record
    if run is not None:
        try:
            return find_record(root, run)
        except NNPhysicsError as error:
            _fail(str(error))
    typer.echo("pass --config or --run to say which run to render", err=True)
    raise typer.Exit(code=_EXIT_USAGE)


def _resolve(config: Path) -> RunConfig:
    """Load a configuration, reporting failures as a clean exit."""
    try:
        return load_run_config(config, os.environ)
    except NNPhysicsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=_EXIT_USAGE) from error


def _fail(message: str) -> NoReturn:
    """Report a failure from the reporting layer and exit."""
    typer.echo(message, err=True)
    raise typer.Exit(code=_EXIT_FAILURE)
