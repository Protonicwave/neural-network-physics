from pathlib import Path

import pytest
from typer.testing import CliRunner

from nnphysics import __version__
from nnphysics.cli.app import app

SUBCOMMANDS = ("diagnose",)
"""Commands still waiting on the phase that fills them in."""

GROUPS = ("data", "eval", "report", "train")
"""Commands a phase has implemented, each tested in its own module."""
EXAMPLE = Path(__file__).resolve().parents[2] / "configs" / "example.yaml"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_lists_every_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (*SUBCOMMANDS, *GROUPS):
        assert name in result.stdout


def test_version_prints_the_package_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_no_arguments_shows_usage(runner: CliRunner) -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.output


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_each_subcommand_exits_cleanly_without_a_config(runner: CliRunner, name: str) -> None:
    result = runner.invoke(app, [name])
    assert result.exit_code == 0
    assert "Not implemented yet" in result.stdout


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_each_subcommand_reports_the_run_it_would_act_on(runner: CliRunner, name: str) -> None:
    result = runner.invoke(app, [name, "--config", str(EXAMPLE)])
    assert result.exit_code == 0
    assert "run example" in result.stdout


def test_an_invalid_config_exits_with_a_message_not_a_traceback(
    runner: CliRunner, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: []\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(bad)])
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_missing_config_exits_with_a_message(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--config", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 2
