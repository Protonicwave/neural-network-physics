"""The `nnp` subcommands.

Each command that a phase has not reached yet is a placeholder. It states what it will
do, and resolves the configuration if one is given, so the configuration boundary is
exercised from the outside from the first phase. Implemented commands live in their own
module and are attached here as a group.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from nnphysics.cli.data import app as data_app
from nnphysics.cli.evals import app as eval_app
from nnphysics.cli.report import app as report_app
from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.core.errors import NNPhysicsError

__all__ = ["register"]

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML run configuration.", show_default=False),
]


def _resolve(config: Path | None) -> RunConfig | None:
    """Load a configuration if one was given, reporting failures as a clean exit."""
    if config is None:
        return None
    try:
        return load_run_config(config, os.environ)
    except NNPhysicsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error


def _announce(action: str, config: Path | None) -> None:
    """Print what the command will do once it is implemented."""
    resolved = _resolve(config)
    if resolved is None:
        typer.echo(f"{action}. Not implemented yet. Pass --config to select a run.")
        return
    typer.echo(f"{action} for run {resolved.name} ({resolved.run_id}).")
    typer.echo(f"System {resolved.system.name}, model {resolved.model.name}, seed {resolved.seed}.")
    typer.echo(f"Artefacts would be written to {resolved.run_dir}.")
    typer.echo("Not implemented yet.")


def train(config: ConfigOption = None) -> None:
    """Train a surrogate on generated trajectories."""
    _announce("Train a surrogate", config)


def diagnose(config: ConfigOption = None) -> None:
    """Explain a regression in a run by reading its report."""
    _announce("Diagnose a regression", config)


def register(app: typer.Typer) -> None:
    """Attach every subcommand to an application.

    Args:
        app: The Typer application to register on.
    """
    app.add_typer(data_app)
    app.add_typer(eval_app)
    app.add_typer(report_app)
    for command in (train, diagnose):
        app.command()(command)
