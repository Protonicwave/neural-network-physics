"""The `nnp` entry point."""

from __future__ import annotations

from typing import Annotated

import typer

from nnphysics import __version__
from nnphysics.cli.commands import register

__all__ = ["app", "main"]

app = typer.Typer(
    name="nnp",
    help="Neural surrogates for physical simulation and the harness that scores them.",
    no_args_is_help=True,
    add_completion=False,
)
register(app)


def _version(value: bool) -> None:
    """Print the version and exit, before any other option is processed."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Run one stage of the surrogate pipeline."""


def main() -> None:
    """Console script entry point."""
    app()
