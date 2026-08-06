"""The `nnp train` command.

Training and evaluation happen in one command and produce one run record. That is
deliberate: a set of weights whose numbers were produced by a different invocation, on a
different day, against a suite nobody wrote down is not a result. Here the model is
trained, the best checkpoint is loaded back, and that checkpoint is scored by the same
phase 05 suite as the reference solver and the five baselines that are broken on purpose,
on the test split and on the held out regime, with the gap between them reported.

This module is also where the trained model meets the harness. The evaluation layer may
not import the models layer, and it does not: it is handed a factory, and it builds the
predictor through the same interface it builds a solver or a persistence baseline
through. Nothing is added to any registry, so one run's weights cannot leak into another.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from nnphysics import __version__
from nnphysics.cli.evals import summarise_suite
from nnphysics.cli.predictors import CHECKPOINT_DIR, ModelFactory
from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError
from nnphysics.core.seeding import make_deterministic
from nnphysics.data.fields import constant_fields
from nnphysics.data.layout import MANIFEST_NAME, NORMALISATION_NAME, dataset_dir
from nnphysics.data.manifest import Manifest, Split, read_manifest
from nnphysics.data.normalisation import (
    Normalisation,
    fit_normalisation,
    read_normalisation,
    write_normalisation,
)
from nnphysics.evals.result import write_result
from nnphysics.evals.runner import load_cases, run_suite
from nnphysics.evals.snapshots import Snapshot, SnapshotSet, capture_snapshots, write_snapshots
from nnphysics.models import ModelContext, build_model, load_model
from nnphysics.reporting.environment import describe_environment, git_commit
from nnphysics.reporting.layout import RunPaths, run_paths
from nnphysics.reporting.record import RunRecord, write_record
from nnphysics.systems import build_system
from nnphysics.training import CheckpointPaths, TrainingHistory, train_model, write_history

if TYPE_CHECKING:
    from nnphysics.core.protocols import System
    from nnphysics.evals.result import SuiteResult

__all__ = ["train"]

_EXIT_FAILURE = 1
_EXIT_USAGE = 2


def train(
    # One required option and four switches, each turning off a stage that costs time.
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="YAML run configuration.", show_default=False),
    ],
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Continue from the last checkpoint of this run rather than starting over.",
        ),
    ] = False,
    evaluate: Annotated[
        bool,
        typer.Option(
            "--evaluate/--no-evaluate",
            help="Score the best checkpoint with the configured suite and write a record.",
        ),
    ] = True,
    baselines: Annotated[
        bool,
        typer.Option(
            "--baselines/--no-baselines",
            help="Also score the suite's own predictors, so the model's numbers sit "
            "beside the reference solver and the deliberately broken ones.",
        ),
    ] = True,
    snapshots: Annotated[
        bool,
        typer.Option(
            "--snapshots/--no-snapshots",
            help="Keep a few states of each rollout for the qualitative plot.",
        ),
    ] = True,
) -> None:
    """Train a surrogate on generated trajectories and score it."""
    resolved = _resolve(config)
    make_deterministic(resolved.run_seed)
    directory = dataset_dir(resolved)
    paths = run_paths(resolved)
    paths.ensure()

    started = time.perf_counter()
    try:
        manifest = read_manifest(directory / MANIFEST_NAME)
        model = build_model(
            resolved.model.name,
            _context(directory, manifest, resolved),
            resolved.model.hyperparameters,
        )
        typer.echo(
            f"Training {model.name} on {resolved.system.name}: {model.n_parameters} "
            f"parameters, {resolved.training.epochs} epochs, stored step {model.dt:g}."
        )
        checkpoints = CheckpointPaths(paths.root / CHECKPOINT_DIR)
        history = train_model(
            model,
            resolved.training,
            directory,
            manifest,
            seed=resolved.run_seed,
            checkpoints=checkpoints,
            progress=typer.echo,
            resume=resume,
        )
        trained = time.perf_counter() - started
        write_history(paths.history, history)
        typer.echo(
            f"Best epoch {history.best_epoch} with validation error "
            f"{history.best_validation_error:.4g}, trained in {history.seconds:.1f}s."
        )
        if not evaluate:
            typer.echo(f"Wrote {checkpoints.best} and {checkpoints.last}.")
            return
        _evaluate(
            resolved,
            paths=paths,
            directory=directory,
            manifest=manifest,
            checkpoints=checkpoints,
            history=history,
            seconds=trained,
            baselines=baselines,
            snapshots=snapshots,
        )
    except NNPhysicsError as error:
        _fail(str(error))


def _evaluate(  # noqa: PLR0913
    # The run, where its artefacts go, where its data is, what was trained and what the
    # caller switched off. Every one of them varies independently, so they are named at
    # the call site rather than ordered.
    config: RunConfig,
    *,
    paths: RunPaths,
    directory: Path,
    manifest: Manifest,
    checkpoints: CheckpointPaths,
    history: TrainingHistory,
    seconds: float,
    baselines: bool,
    snapshots: bool,
) -> None:
    """Score the best checkpoint and write the run record."""
    started = time.perf_counter()
    system = build_system(config.system.name, config.system.parameters)
    best = load_model(checkpoints.best)
    factory = ModelFactory(best)
    names = (*config.evaluation.predictors, best.name) if baselines else (best.name,)

    result = run_suite(
        system,
        directory,
        manifest,
        config.evaluation,
        seed=config.run_seed,
        run_id=config.run_id,
        predictors=names,
        factories={best.name: factory},
    )
    elapsed = time.perf_counter() - started
    kept = (
        _capture(
            system,
            config,
            result,
            directory=directory,
            manifest=manifest,
            factory=factory,
            name=best.name,
        )
        if snapshots
        else SnapshotSet()
    )

    destination = paths.result(config.evaluation.name)
    write_result(destination, result)
    write_snapshots(paths.snapshots, kept)
    write_record(
        paths.record,
        RunRecord(
            run_id=config.run_id,
            name=config.name,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
            code_version=__version__,
            commit=git_commit(),
            config=config,
            environment=describe_environment(),
            timings={"training": seconds, "evaluation": elapsed},
            training=history,
            evaluation=result,
            artefacts={
                "checkpoint": paths.relative(checkpoints.best),
                "snapshots": paths.relative(paths.snapshots),
                "result": paths.relative(destination),
            },
        ),
    )
    summarise_suite(result)
    typer.echo(f"Wrote {destination} and {paths.record}.")


def _capture(  # noqa: PLR0913
    # The same arguments the evaluation needed, plus the factory and the name that
    # cannot be resolved through the registry.
    system: System,
    config: RunConfig,
    result: SuiteResult,
    *,
    directory: Path,
    manifest: Manifest,
    factory: ModelFactory,
    name: str,
) -> SnapshotSet:
    """Keep a few states of each predictor, on every split the suite evaluated."""
    specs = tuple(dict.fromkeys(entry.spec for entry in result.results))
    kept: list[Snapshot] = []
    for split in dict.fromkeys(entry.split for entry in result.results):
        cases = load_cases(
            directory,
            manifest,
            system,
            split=Split(split),
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
                seed=config.run_seed,
                factories={name: factory},
            ).snapshots
        )
    return SnapshotSet(tuple(kept))


def _context(directory: Path, manifest: Manifest, config: RunConfig) -> ModelContext:
    """Everything the model needs to know about the data it will be trained on."""
    return ModelContext(
        field_shapes=manifest.field_shapes(Split.TRAIN),
        static_fields=constant_fields(directory, manifest),
        normalisation=_normalisation(directory, manifest),
        dt=manifest.spec.dt,
        seed=config.run_seed,
    )


def _normalisation(directory: Path, manifest: Manifest) -> Normalisation:
    """The training split's statistics, fitted now if `nnp data stats` never was.

    Fitting here rather than failing keeps the pipeline to one command, and it cannot
    reach the wrong data: the fit reads the training split whichever path it is reached
    by, and the artefact it writes is the one `nnp data stats` would have written.
    """
    path = directory / NORMALISATION_NAME
    if path.is_file():
        return read_normalisation(path)
    typer.echo(f"No {NORMALISATION_NAME} beside the data, fitting it from the training split.")
    statistics = fit_normalisation(directory, manifest)
    write_normalisation(path, statistics)
    return statistics


def _resolve(config: Path) -> RunConfig:
    """Load a configuration, reporting failures as a clean exit."""
    try:
        return load_run_config(config, os.environ)
    except NNPhysicsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=_EXIT_USAGE) from error


def _fail(message: str) -> NoReturn:
    """Report a failure and exit."""
    typer.echo(message, err=True)
    raise typer.Exit(code=_EXIT_FAILURE)
