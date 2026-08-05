"""The `nnp eval` subcommands.

The other place side effects belong. This is also the only place a system is built by
name: the evaluation layer is handed the built system through the protocol and never
learns which one it got.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError
from nnphysics.data.layout import MANIFEST_NAME, dataset_dir
from nnphysics.data.manifest import Split, read_manifest
from nnphysics.evals.metrics import METRICS
from nnphysics.evals.predictors import PREDICTORS
from nnphysics.evals.result import SuiteResult, write_result
from nnphysics.evals.runner import run_suite
from nnphysics.systems import build_system

__all__ = ["app"]

app = typer.Typer(
    name="eval",
    help="Roll predictors out and score them against a suite.",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="YAML run configuration.", show_default=False),
]

_EXIT_FAILURE = 1
_EXIT_USAGE = 2

_SUMMARY_SCALARS = (
    ("one_step_error", "error"),
    ("rollout_error", "error.final"),
    ("invariant_drift", "worst_violation"),
    ("symmetry_violation", "worst"),
    ("distribution_drift", "worst"),
)
"""One number per metric for the terminal. The result file carries the rest."""


@app.command()
def run(
    config: ConfigOption,
    predictor: Annotated[
        list[str] | None,
        typer.Option(
            "--predictor",
            "-p",
            help="Predictor specification, `name` or `name:key=value`. Repeatable. "
            "Overrides the suite's own list.",
            show_default=False,
        ),
    ] = None,
    split: Annotated[
        list[Split] | None,
        typer.Option(
            "--split",
            "-s",
            help="Split to evaluate on. Repeatable. Defaults to the test split and the "
            "held out one.",
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the result file.", show_default=False),
    ] = None,
) -> None:
    """Evaluate predictors against a suite and write a structured result file."""
    resolved = _resolve(config)
    directory = dataset_dir(resolved)
    destination = output or resolved.run_dir / f"evaluation-{resolved.evaluation.name}.json"
    started = time.perf_counter()
    try:
        system = build_system(resolved.system.name, resolved.system.parameters)
        manifest = read_manifest(directory / MANIFEST_NAME)
        result = run_suite(
            system,
            directory,
            manifest,
            resolved.evaluation,
            seed=resolved.seed,
            run_id=resolved.run_id,
            predictors=predictor or None,
            splits=split or None,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_result(destination, result)
    except NNPhysicsError as error:
        _fail(str(error))

    _summarise(result)
    typer.echo(f"Wrote {destination} in {time.perf_counter() - started:.1f}s.")


@app.command("list")
def list_registered() -> None:
    """List every registered metric and predictor."""
    typer.echo("Metrics:")
    for name in METRICS.names():
        typer.echo(f"  {name}")
    typer.echo("Predictors:")
    for name in PREDICTORS.names():
        typer.echo(f"  {name}")


def _summarise(result: SuiteResult) -> None:
    """Print one line per predictor and split, with one number per metric."""
    typer.echo(
        f"System {result.system}, suite {result.settings.name}, "
        f"{result.settings.rollout_steps} steps from "
        f"{result.settings.n_initial_conditions} initial conditions."
    )
    for entry in result.results:
        numbers = []
        for metric, key in _SUMMARY_SCALARS:
            try:
                numbers.append(f"{metric.split('_')[0]}.{key}={entry.scalar(metric, key):.3g}")
            except KeyError:
                continue
        suffix = "" if entry.completed else "  [a rollout stopped early]"
        typer.echo(f"  {entry.split:>9} {entry.predictor:<20} {'  '.join(numbers)}{suffix}")
    if result.regime_gap:
        typer.echo(f"Regime gap recorded for {len(result.regime_gap)} scalars.")


def _resolve(config: Path) -> RunConfig:
    """Load a configuration, reporting failures as a clean exit."""
    try:
        return load_run_config(config, os.environ)
    except NNPhysicsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=_EXIT_USAGE) from error


def _fail(message: str) -> NoReturn:
    """Report a failure from the evaluation layer and exit."""
    typer.echo(message, err=True)
    raise typer.Exit(code=_EXIT_FAILURE)
