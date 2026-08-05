"""The `nnp eval` subcommands.

The other place side effects belong. This is also the only place a system is built by
name: the evaluation layer is handed the built system through the protocol and never
learns which one it got.

An evaluation writes three things: the result file, the states kept for the qualitative
plot, and the run record that ties them to the configuration, the machine and the clock.
The record is written here rather than at rendering time because the environment and the
timings are only true at the moment the run happens.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from nnphysics import __version__
from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError
from nnphysics.core.protocols import System
from nnphysics.data.layout import MANIFEST_NAME, dataset_dir
from nnphysics.data.manifest import Manifest, Split, read_manifest
from nnphysics.evals.metrics import METRICS
from nnphysics.evals.predictors import PREDICTORS
from nnphysics.evals.result import SuiteResult, write_result
from nnphysics.evals.runner import load_cases, run_suite
from nnphysics.evals.snapshots import (
    Snapshot,
    SnapshotSet,
    capture_snapshots,
    write_snapshots,
)
from nnphysics.reporting.environment import describe_environment, git_commit
from nnphysics.reporting.layout import RunPaths, run_paths
from nnphysics.reporting.record import RunRecord, write_record
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
    snapshots: Annotated[
        bool,
        typer.Option(
            "--snapshots/--no-snapshots",
            help="Keep a few states of each rollout for the qualitative plot. Each "
            "predictor costs one extra rollout.",
        ),
    ] = True,
) -> None:
    """Evaluate predictors against a suite and write a run record."""
    resolved = _resolve(config)
    directory = dataset_dir(resolved)
    paths = run_paths(resolved)
    destination = output or paths.result(resolved.evaluation.name)
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
        elapsed = time.perf_counter() - started
        kept = (
            _capture(system, directory, manifest, resolved, result) if snapshots else SnapshotSet()
        )
        paths.ensure()
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_result(destination, result)
        write_snapshots(paths.snapshots, kept)
        write_record(
            paths.record,
            _record(resolved, result, paths, destination, seconds=elapsed),
        )
    except NNPhysicsError as error:
        _fail(str(error))

    _summarise(result)
    typer.echo(f"Wrote {destination} and {paths.record} in {time.perf_counter() - started:.1f}s.")


@app.command("list")
def list_registered() -> None:
    """List every registered metric and predictor."""
    typer.echo("Metrics:")
    for name in METRICS.names():
        typer.echo(f"  {name}")
    typer.echo("Predictors:")
    for name in PREDICTORS.names():
        typer.echo(f"  {name}")


def _capture(
    system: System,
    directory: Path,
    manifest: Manifest,
    config: RunConfig,
    result: SuiteResult,
) -> SnapshotSet:
    """Keep a few states of each predictor, on every split the suite evaluated.

    The cases are loaded exactly as the runner loaded them, so the first of them is the
    same trajectory the numbers above came from.
    """
    specs = tuple(dict.fromkeys(entry.spec for entry in result.results))
    kept: list[Snapshot] = []
    for name in dict.fromkeys(entry.split for entry in result.results):
        cases = load_cases(
            directory,
            manifest,
            system,
            split=Split(name),
            count=config.evaluation.n_initial_conditions,
            steps=config.evaluation.rollout_steps,
        )
        kept.extend(
            capture_snapshots(
                system,
                cases,
                specs,
                config.evaluation,
                substeps=manifest.spec.substeps,
                seed=config.seed,
            ).snapshots
        )
    return SnapshotSet(tuple(kept))


def _record(
    config: RunConfig,
    result: SuiteResult,
    paths: RunPaths,
    destination: Path,
    *,
    seconds: float,
) -> RunRecord:
    """Tie the numbers to the configuration, the machine and the moment."""
    artefacts = {"snapshots": paths.relative(paths.snapshots)}
    if destination.is_relative_to(paths.root):
        artefacts["result"] = paths.relative(destination)
    else:
        # A result written outside the run directory is still worth naming, and naming it
        # absolutely is more honest than pretending it sits beside the record.
        artefacts["result"] = destination.as_posix()
    return RunRecord(
        run_id=config.run_id,
        name=config.name,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        code_version=__version__,
        commit=git_commit(),
        config=config,
        environment=describe_environment(),
        timings={"evaluation": seconds},
        evaluation=result,
        artefacts=artefacts,
    )


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
