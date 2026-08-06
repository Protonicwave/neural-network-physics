from pathlib import Path

import pytest
from typer.testing import CliRunner

from nnphysics import __version__
from nnphysics.cli.app import app

COMMANDS = ("data", "eval", "ensemble", "report", "train", "benchmark", "diagnose")
"""Every command the plan names. All of them are implemented, so there is no longer a
placeholder to test: each is covered in its own module."""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_lists_every_subcommand(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.stdout


def test_version_prints_the_package_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_no_arguments_shows_usage(runner: CliRunner) -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.output


@pytest.mark.parametrize("name", COMMANDS)
def test_each_command_describes_itself(runner: CliRunner, name: str) -> None:
    """A command that cannot say what it does is a command nobody will find."""
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


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
