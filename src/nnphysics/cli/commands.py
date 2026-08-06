"""The `nnp` subcommands.

Every command the plan names is now implemented. Each lives in its own module and is
attached here, either as a group of subcommands or as a single command.
"""

from __future__ import annotations

import typer

from nnphysics.cli.benchmark import benchmark
from nnphysics.cli.data import app as data_app
from nnphysics.cli.diagnose import app as diagnose_app
from nnphysics.cli.ensemble import app as ensemble_app
from nnphysics.cli.evals import app as eval_app
from nnphysics.cli.report import app as report_app
from nnphysics.cli.train import train

__all__ = ["register"]


def register(app: typer.Typer) -> None:
    """Attach every subcommand to an application.

    Args:
        app: The Typer application to register on.
    """
    app.add_typer(data_app)
    app.add_typer(eval_app)
    app.add_typer(ensemble_app)
    app.add_typer(report_app)
    app.add_typer(diagnose_app)
    for command in (train, benchmark):
        app.command()(command)
