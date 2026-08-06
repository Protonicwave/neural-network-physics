"""The `nnp benchmark` command.

Where the thread count is actually fixed. Every timing in a speed report is a function of
how many cores the arithmetic was allowed, so a report whose thread count drifted between
the solver and the network would be comparing thread counts. It is set here, at the edge,
before anything is built, and recorded in the report; the evaluation layer records the
number and never reaches for it.

The report is written twice, beside the run record and into it. Beside it so a benchmark
can be read without parsing a run, into it so the machine specification, the commit and
the configuration travel with the numbers. Where a record already exists the benchmark is
attached to it rather than replacing it: the run's own evaluation is what the benchmark is
a benchmark of.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import torch
import typer

from nnphysics.cli.predictors import ModelFactory, trained_members
from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError
from nnphysics.data.layout import MANIFEST_NAME, dataset_dir
from nnphysics.data.manifest import Split, read_manifest
from nnphysics.evals.benchmark import (
    DEFAULT_STEPS_PER_TRIAL,
    DEFAULT_TRIALS,
    DEFAULT_WARMUP,
    STABILITY_LIMIT,
)
from nnphysics.evals.runner import load_cases
from nnphysics.evals.speed import NEVER_PAYS, SpeedReport, build_speed_report
from nnphysics.models import Ensemble, load_model
from nnphysics.reporting.layout import RunPaths, ensemble_paths, run_paths
from nnphysics.reporting.record import read_record, write_record
from nnphysics.systems import build_system
from nnphysics.training import read_history

if TYPE_CHECKING:
    from nnphysics.evals.predictors import PredictorFactory

__all__ = ["benchmark", "summarise_speed"]

_EXIT_FAILURE = 1
_EXIT_USAGE = 2

_TRAINED_ROLE = "checkpoint"
"""Artefact a run record names its weights under, which is where a benchmark looks for a
trained predictor when it is not told one."""

_MILLISECONDS = 1.0e3


def benchmark(  # noqa: PLR0913, PLR0917
    # One required option and five knobs: one chooses what is timed beside the solver, the
    # rest decide whether the timings mean anything. Typer reads them positionally, so
    # they cannot be made keyword only.
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="YAML run configuration.", show_default=False),
    ],
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Model to time beside the solver. Defaults to the one this run trained.",
            show_default=False,
        ),
    ] = None,
    ensemble: Annotated[
        bool,
        typer.Option(
            "--ensemble/--no-ensemble",
            help="Also time the deep ensemble of this configuration, if its members have "
            "been trained.",
        ),
    ] = False,
    threads: Annotated[
        int | None,
        typer.Option(
            "--threads",
            help="Threads the arithmetic may use. Defaults to every logical core.",
            show_default=False,
        ),
    ] = None,
    trials: Annotated[
        int, typer.Option("--trials", help="Timed repeats of every point.")
    ] = DEFAULT_TRIALS,
    warmup: Annotated[
        int, typer.Option("--warmup", help="Repeats discarded before timing begins.")
    ] = DEFAULT_WARMUP,
    steps: Annotated[
        int,
        typer.Option(
            "--steps",
            help="Steps each trial averages over. Raise it where one step is short "
            "enough that the operating system's own jitter is a large part of it.",
        ),
    ] = DEFAULT_STEPS_PER_TRIAL,
) -> None:
    """Measure the speedup a surrogate delivers at matched accuracy."""
    resolved = _resolve(config)
    fixed = _fix_threads(threads)
    directory = dataset_dir(resolved)
    # An ensemble's numbers belong to the run that scored the ensemble. Writing them into
    # the directory of member zero, which is a different predictor, would attach a
    # benchmark to a record that never measured the thing benchmarked.
    paths = ensemble_paths(resolved) if ensemble else run_paths(resolved)

    started = time.perf_counter()
    try:
        system = build_system(resolved.system.name, resolved.system.parameters)
        manifest = read_manifest(directory / MANIFEST_NAME)
        cases = load_cases(
            directory,
            manifest,
            system,
            split=Split.TEST,
            count=resolved.evaluation.n_initial_conditions,
            steps=resolved.evaluation.rollout_steps,
        )
        factories, costs = _surrogates(resolved, run_paths(resolved), checkpoint, ensemble=ensemble)
        typer.echo(
            f"Timing the {resolved.system.name} solver from {manifest.spec.substeps} "
            f"substeps down, on {fixed} threads, {trials} trials of {steps} steps after "
            f"{warmup} warmups."
        )
        report = build_speed_report(
            system,
            cases,
            resolved.evaluation,
            dataset_substeps=manifest.spec.substeps,
            seed=resolved.run_seed,
            predictors=tuple(factories),
            factories=factories,
            training_seconds=costs,
            trajectories=len(manifest.trajectories),
            states_per_trajectory=manifest.spec.n_steps,
            threads=fixed,
            trials=trials,
            warmup=warmup,
            steps_per_trial=steps,
        )
        _write(paths, report, seconds=time.perf_counter() - started)
    except NNPhysicsError as error:
        _fail(str(error))

    summarise_speed(report)
    typer.echo(f"Wrote {paths.benchmark} in {time.perf_counter() - started:.1f}s.")


def summarise_speed(report: SpeedReport) -> None:
    """Print the ladder, the surrogates and what each one is worth.

    Args:
        report: What the benchmark measured.
    """
    typer.echo(
        f"System {report.system}, {report.split} split, {report.steps} steps from "
        f"{report.n_initial_conditions} initial conditions, {report.threads} threads."
    )
    for point in (*report.ladder, *report.surrogates):
        flag = "" if point.stable else f"  [unstable, above {STABILITY_LIMIT:.0%}]"
        typer.echo(
            f"  {point.label:<26} error={point.error:<10.4g} "
            f"{point.seconds_per_step * _MILLISECONDS:.3f} ms/step "
            f"+-{point.relative_spread:.1%}{flag}"
        )
    for matched, cost in zip(report.matched, report.costs, strict=True):
        bound = (
            ""
            if matched.bracketed
            else ", the cheapest setting that runs: nothing measured was as inaccurate"
        )
        typer.echo(
            f"  {matched.predictor}: {matched.speedup:.3g}x the solver at "
            f"{matched.matched_substeps} substeps{bound}."
        )
        if cost.break_even_rollouts == NEVER_PAYS:
            typer.echo(
                f"    Never pays for itself: {-cost.saving_per_rollout:.3g} s slower per "
                f"rollout than the solver it would replace."
            )
        else:
            typer.echo(
                f"    Saves {cost.saving_per_rollout:.3g} s per rollout against "
                f"{cost.training_seconds + cost.generation_seconds:.0f} s spent once, so "
                f"it breaks even after {cost.break_even_rollouts:.0f} rollouts."
            )


def _surrogates(
    config: RunConfig, paths: RunPaths, checkpoint: Path | None, *, ensemble: bool
) -> tuple[dict[str, PredictorFactory], dict[str, float]]:
    """Find what to time beside the solver, and what each of them cost to train.

    A benchmark of the solver alone is still a benchmark, so nothing here fails when a
    configuration has trained nothing yet. It says what it found and times that.
    """
    factories: dict[str, PredictorFactory] = {}
    costs: dict[str, float] = {}

    path = checkpoint if checkpoint is not None else _trained(paths)
    if path is not None:
        model = load_model(path)
        factories[model.name] = ModelFactory(model)
        costs[model.name] = _training_seconds(paths)
    if ensemble:
        members = trained_members(config)
        wanted = config.ensemble.members
        if len(members) < wanted:
            typer.echo(f"Found {len(members)} of {wanted} ensemble members trained.")
        if len(members) > 1:
            built = Ensemble(members)
            factories[built.name] = ModelFactory(built)
            costs[built.name] = sum(
                _training_seconds(run_paths(config.for_member(index)))
                for index in range(len(members))
            )
    if not factories:
        typer.echo("No trained predictor found, so only the solver is timed.")
    return factories, costs


def _trained(paths: RunPaths) -> Path | None:
    """The checkpoint this run's record names, if it has one on disk."""
    if not paths.record.is_file():
        return None
    relative = read_record(paths.record).artefacts.get(_TRAINED_ROLE)
    if relative is None:
        return None
    path = paths.root / relative
    return path if path.is_file() else None


def _training_seconds(paths: RunPaths) -> float:
    """How long a run trained, or zero if it never did.

    The history rather than the record's own timing, wherever there is one. A history
    sums the wall clock of every epoch, so it survives a run that was interrupted and
    resumed; the record's timing is what that one invocation took, which for a resumed run
    is only the tail of the training. An ensemble member evaluates nothing and writes no
    record at all, and its history on its own is where its cost went.

    A predictor accounted as free to train would break even on its first rollout, which is
    the one answer this must not give.
    """
    if paths.history.is_file():
        return read_history(paths.history).seconds
    if not paths.record.is_file():
        return 0.0
    record = read_record(paths.record)
    if record.training is not None:
        return record.training.seconds
    return record.timings.get("training", 0.0)


def _write(paths: RunPaths, report: SpeedReport, *, seconds: float) -> None:
    """Write the report beside the record, and attach it to the record if there is one."""
    paths.ensure()
    paths.benchmark.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if not paths.record.is_file():
        typer.echo(
            f"No run record at {paths.record}, so the benchmark stands alone. Run "
            f"`nnp eval run` or `nnp train` to give it one."
        )
        return
    existing = read_record(paths.record)
    write_record(
        paths.record,
        existing.model_copy(
            update={
                "benchmark": report,
                "timings": {**existing.timings, "benchmark": seconds},
            }
        ),
    )


def _resolve(config: Path) -> RunConfig:
    """Load a configuration, reporting failures as a clean exit."""
    try:
        return load_run_config(config, os.environ)
    except NNPhysicsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=_EXIT_USAGE) from error


def _fix_threads(threads: int | None) -> int:
    """Pin how many threads the arithmetic may use, and return the number pinned.

    Set rather than read, because a timing is only comparable against another timing if
    both were taken with the same number of cores doing the work.
    """
    chosen = threads if threads is not None else max(1, os.cpu_count() or 1)
    torch.set_num_threads(chosen)
    return chosen


def _fail(message: str) -> NoReturn:
    """Report a failure and exit."""
    typer.echo(message, err=True)
    raise typer.Exit(code=_EXIT_FAILURE)
