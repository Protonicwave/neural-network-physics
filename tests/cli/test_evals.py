from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nnphysics.cli.app import app
from nnphysics.core.config import RunConfig
from nnphysics.data.build import build_dataset
from nnphysics.evals.result import read_result

runner = CliRunner()

CONFIG: dict[str, Any] = {
    "name": "cli-evals",
    "seed": 2,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 4,
        "n_steps": 6,
        "dt": 0.02,
        "substeps": 2,
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 2,
    },
    "model": {"name": "placeholder"},
    "evaluation": {
        "name": "cli",
        "metrics": ["rollout_error", "invariant_drift"],
        "predictors": ["reference", "persistence"],
        "rollout_steps": 4,
        "n_initial_conditions": 2,
        "symmetry_steps": 2,
    },
}


@pytest.fixture
def written(tmp_path: Path) -> Path:
    """A configuration file whose dataset has already been generated."""
    config = dict(CONFIG)
    config["data"] = {**CONFIG["data"], "root": str(tmp_path / "data")}
    config["output_dir"] = str(tmp_path / "runs")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    build_dataset(RunConfig.model_validate(config))
    return path


class TestRun:
    def test_it_writes_a_result_file(self, written: Path, tmp_path: Path) -> None:
        output = tmp_path / "evaluation.json"

        result = runner.invoke(app, ["eval", "run", "-c", str(written), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert read_result(output).settings.name == "cli"

    def test_it_defaults_to_a_path_inside_the_run_directory(self, written: Path) -> None:
        result = runner.invoke(app, ["eval", "run", "-c", str(written)])

        assert result.exit_code == 0, result.output
        assert "evaluation-cli.json" in result.output

    def test_it_summarises_every_predictor(self, written: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["eval", "run", "-c", str(written), "-o", str(tmp_path / "e.json")]
        )

        assert "reference" in result.output
        assert "persistence" in result.output

    def test_a_predictor_can_be_given_on_the_command_line(
        self, written: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "evaluation.json"

        result = runner.invoke(
            app,
            ["eval", "run", "-c", str(written), "-p", "noise:scale=0.02", "-o", str(output)],
        )

        assert result.exit_code == 0, result.output
        assert {entry.spec for entry in read_result(output).results} == {"noise:scale=0.02"}

    def test_a_split_can_be_given_on_the_command_line(self, written: Path, tmp_path: Path) -> None:
        output = tmp_path / "evaluation.json"

        result = runner.invoke(
            app, ["eval", "run", "-c", str(written), "-s", "test", "-o", str(output)]
        )

        assert result.exit_code == 0, result.output
        assert {entry.split for entry in read_result(output).results} == {"test"}

    def test_a_missing_configuration_is_a_usage_error(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["eval", "run", "-c", str(tmp_path / "absent.yaml")])

        assert result.exit_code == 2

    def test_a_missing_dataset_is_a_failure_not_a_traceback(self, tmp_path: Path) -> None:
        config = dict(CONFIG)
        config["data"] = {**CONFIG["data"], "root": str(tmp_path / "empty")}
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")

        result = runner.invoke(app, ["eval", "run", "-c", str(path)])

        assert result.exit_code == 1
        assert "cannot read manifest" in result.output

    def test_an_unknown_predictor_is_a_failure_not_a_traceback(
        self, written: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, ["eval", "run", "-c", str(written), "-p", "oracle", "-o", str(tmp_path / "e.json")]
        )

        assert result.exit_code == 1
        assert "unknown predictor" in result.output


class TestList:
    def test_it_names_every_metric_and_predictor(self) -> None:
        result = runner.invoke(app, ["eval", "list"])

        assert result.exit_code == 0
        assert "invariant_drift" in result.output
        assert "energy_injection" in result.output


class TestHelp:
    def test_the_group_is_attached_to_the_application(self) -> None:
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "eval" in result.output
