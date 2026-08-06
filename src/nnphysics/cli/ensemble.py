"""The `nnp ensemble` subcommands.

An ensemble is not a model, it is several runs of the same configuration and a way of
reading them together. So this command does not train anything the training loop cannot
already train: it trains the configuration once per member, each into its own run
directory with its own record, and then scores the ensemble of what came out.

Member zero is the plain run. Its configuration is byte for byte what `nnp train` would
resolve, so a configuration that has already been trained does not train it again, and the
member the whole repository's earlier numbers describe is the same member the ensemble is
built from. Only the initialisation and the shuffling differ between members; the dataset
identifier does not depend on the member index, and a test asserts that, because members
trained on different data would be measuring the data rather than the initialisation.

The members are trained in sequence. Each one already spreads its own arithmetic across
every core, so a process per member would contend for the same cores rather than add to
them, and the phase's own benchmark would then be measuring the contention.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from nnphysics import __version__
from nnphysics.cli.evals import summarise_suite
from nnphysics.cli.predictors import (
    CHECKPOINT_DIR,
    ModelFactory,
    member_checkpoint,
    trained_members,
)
from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError, ValidationError
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
from nnphysics.evals.runner import run_suite
from nnphysics.models import Ensemble, ModelContext, build_model
from nnphysics.reporting.environment import describe_environment, git_commit
from nnphysics.reporting.layout import ensemble_paths, run_paths
from nnphysics.reporting.record import RunRecord, write_record
from nnphysics.systems import build_system
from nnphysics.training import CheckpointPaths, train_model, write_history

__all__ = ["app"]

app = typer.Typer(
    name="ensemble",
    help="Train several models from different seeds and score them as one predictor.",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="YAML run configuration.", show_default=False),
]

_EXIT_FAILURE = 1
_EXIT_USAGE = 2


@app.command()
def train(
    config: ConfigOption,
    members: Annotated[
        int | None,
        typer.Option(
            "--members",
            help="Members to train. Defaults to what the configuration declares.",
            show_default=False,
        ),
    ] = None,
    retrain: Annotated[
        bool,
        typer.Option(
            "--retrain/--reuse",
            help="Train a member again even if its checkpoint is already on disk.",
        ),
    ] = False,
) -> None:
    """Train every member of a deep ensemble, one after another."""
    resolved = _resolve(config)
    wanted = members if members is not None else resolved.ensemble.members
    try:
        if not 0 < wanted <= resolved.ensemble.members:
            raise ValidationError(
                f"asked for {wanted} members of an ensemble the configuration declares "
                f"{resolved.ensemble.members} of"
            )
        for index in range(wanted):
            _train_member(resolved.for_member(index), index, wanted, retrain=retrain)
    except NNPhysicsError as error:
        _fail(str(error))
    typer.echo(f"Trained {wanted} members. Run `nnp ensemble run` to score them together.")


@app.command()
def run(
    config: ConfigOption,
    baselines: Annotated[
        bool,
        typer.Option(
            "--baselines/--no-baselines",
            help="Also score the suite's own predictors, so the ensemble's numbers sit "
            "beside the reference solver and the deliberately broken ones.",
        ),
    ] = True,
) -> None:
    """Score the trained members together as one predictor."""
    resolved = _resolve(config)
    directory = dataset_dir(resolved)
    paths = ensemble_paths(resolved)
    started = time.perf_counter()
    try:
        loaded = trained_members(resolved)
        if len(loaded) < resolved.ensemble.members:
            typer.echo(
                f"Only {len(loaded)} of {resolved.ensemble.members} members are trained; "
                f"scoring the ones that are."
            )
        built = Ensemble(loaded)
        system = build_system(resolved.system.name, resolved.system.parameters)
        manifest = read_manifest(directory / MANIFEST_NAME)
        typer.echo(
            f"Scoring an ensemble of {len(built.members)} {loaded[0].name} models, "
            f"{built.n_parameters} parameters in total."
        )
        factory = ModelFactory(built)
        names = (*resolved.evaluation.predictors, built.name) if baselines else (built.name,)
        result = run_suite(
            system,
            directory,
            manifest,
            resolved.evaluation,
            seed=resolved.run_seed,
            run_id=resolved.for_member(0).run_id,
            predictors=names,
            factories={built.name: factory},
        )
        elapsed = time.perf_counter() - started
        paths.ensure()
        destination = paths.result(resolved.evaluation.name)
        write_result(destination, result)
        write_record(
            paths.record,
            RunRecord(
                run_id=resolved.for_member(0).run_id,
                name=f"{resolved.name}-ensemble",
                created=datetime.now(UTC).isoformat(timespec="seconds"),
                code_version=__version__,
                commit=git_commit(),
                config=resolved.for_member(0),
                environment=describe_environment(),
                timings={"evaluation": elapsed},
                evaluation=result,
                artefacts={"result": paths.relative(destination)},
            ),
        )
    except NNPhysicsError as error:
        _fail(str(error))

    summarise_suite(result)
    typer.echo(f"Wrote {destination} and {paths.record}.")


def _train_member(config: RunConfig, index: int, total: int, *, retrain: bool) -> None:
    """Train one member, unless it is already trained and reuse was asked for."""
    paths = run_paths(config)
    if member_checkpoint(config).is_file() and not retrain:
        typer.echo(f"Member {index + 1} of {total} is already trained at {paths.root}.")
        return

    make_deterministic(config.run_seed)
    directory = dataset_dir(config)
    paths.ensure()
    manifest = read_manifest(directory / MANIFEST_NAME)
    model = build_model(
        config.model.name, _context(directory, manifest, config), config.model.hyperparameters
    )
    typer.echo(
        f"Member {index + 1} of {total}: {model.name}, {model.n_parameters} parameters, "
        f"seed {config.run_seed}, into {paths.root}."
    )
    history = train_model(
        model,
        config.training,
        directory,
        manifest,
        seed=config.run_seed,
        checkpoints=CheckpointPaths(paths.root / CHECKPOINT_DIR),
        progress=typer.echo,
    )
    # A member is trained without being evaluated, so it writes no run record. What it
    # cost is exactly what the cost accounting needs, and a member whose training time
    # went unwritten would be accounted as free.
    write_history(paths.history, history)
    typer.echo(
        f"Member {index + 1} best epoch {history.best_epoch} with validation error "
        f"{history.best_validation_error:.4g}, in {history.seconds:.0f}s."
    )


def _context(directory: Path, manifest: Manifest, config: RunConfig) -> ModelContext:
    """Everything the model needs to know about the data it will be trained on.

    The seed is the member's, so members initialise differently. Everything else is the
    dataset's, and the dataset is the same for every member.
    """
    return ModelContext(
        field_shapes=manifest.field_shapes(Split.TRAIN),
        static_fields=constant_fields(directory, manifest),
        normalisation=_normalisation(directory, manifest),
        dt=manifest.spec.dt,
        seed=config.run_seed,
    )


def _normalisation(directory: Path, manifest: Manifest) -> Normalisation:
    """The training split's statistics, fitted now if `nnp data stats` never was."""
    path = directory / NORMALISATION_NAME
    if path.is_file():
        return read_normalisation(path)
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
