from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nnphysics.cli.app import app
from nnphysics.data.build import build_dataset
from nnphysics.data.layout import NORMALISATION_NAME, dataset_dir
from nnphysics.models import MODELS
from nnphysics.reporting.layout import run_paths
from nnphysics.reporting.record import read_record

from nnphysics.core.config import RunConfig  # isort: skip

CONFIG: dict[str, Any] = {
    "name": "cli-train",
    "seed": 5,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 8,
        "n_steps": 10,
        "dt": 0.02,
        "substeps": 2,
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 4,
    },
    "model": {"name": "constant"},
    "training": {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.01,
        "warmup_epochs": 0,
        "validation_steps": 2,
        "window_stride": 4,
    },
    "evaluation": {
        "name": "cli-suite",
        "metrics": ["one_step_error", "rollout_error"],
        "predictors": ["reference", "persistence"],
        "rollout_steps": 5,
        "n_initial_conditions": 2,
        "symmetry_steps": 2,
    },
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, RunConfig]:
    """A configuration file whose dataset has already been generated."""
    payload = {
        **CONFIG,
        "data": {**CONFIG["data"], "root": str(tmp_path / "data")},
        "output_dir": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = RunConfig.model_validate(payload)
    build_dataset(config)
    return path, config


def test_every_model_a_run_may_train_is_discoverable(runner: CliRunner) -> None:
    """`eval list` names everything a suite might resolve, models included."""
    result = runner.invoke(app, ["eval", "list"])
    assert result.exit_code == 0, result.output
    assert "Models:" in result.output
    for name in MODELS.names():
        assert name in result.output


def test_training_writes_a_record_carrying_both_halves_of_the_run(
    runner: CliRunner, prepared: tuple[Path, RunConfig]
) -> None:
    path, config = prepared
    result = runner.invoke(app, ["train", "--config", str(path)])
    assert result.exit_code == 0, result.output

    record = read_record(run_paths(config).record)
    assert record.training is not None
    assert record.training.model == "constant"
    assert len(record.training.epochs) == config.training.epochs
    assert record.timings.keys() == {"training", "evaluation"}
    assert "constant" in record.predictors


def test_the_model_is_scored_beside_the_predictors_the_suite_names(
    runner: CliRunner, prepared: tuple[Path, RunConfig]
) -> None:
    """The harness sees a trained model through the same interface as a broken baseline."""
    path, config = prepared
    assert runner.invoke(app, ["train", "--config", str(path)]).exit_code == 0
    record = read_record(run_paths(config).record)
    assert set(record.predictors) == {"reference", "persistence", "constant"}
    assert record.evaluation.regime_gap


def test_the_baselines_can_be_left_out(runner: CliRunner, prepared: tuple[Path, RunConfig]) -> None:
    path, config = prepared
    result = runner.invoke(app, ["train", "--config", str(path), "--no-baselines"])
    assert result.exit_code == 0
    assert read_record(run_paths(config).record).predictors == ("constant",)


def test_the_checkpoint_is_recorded_as_an_artefact(
    runner: CliRunner, prepared: tuple[Path, RunConfig]
) -> None:
    path, config = prepared
    assert runner.invoke(app, ["train", "--config", str(path)]).exit_code == 0
    paths = run_paths(config)
    record = read_record(paths.record)
    assert (paths.root / record.artefacts["checkpoint"]).is_file()


def test_evaluation_can_be_skipped_and_then_no_record_is_written(
    runner: CliRunner, prepared: tuple[Path, RunConfig]
) -> None:
    path, config = prepared
    result = runner.invoke(app, ["train", "--config", str(path), "--no-evaluate"])
    assert result.exit_code == 0
    assert not run_paths(config).record.exists()


def test_the_statistics_are_fitted_when_no_one_has_fitted_them(
    runner: CliRunner, prepared: tuple[Path, RunConfig]
) -> None:
    path, config = prepared
    statistics = dataset_dir(config) / NORMALISATION_NAME
    assert not statistics.exists()
    assert runner.invoke(app, ["train", "--config", str(path), "--no-evaluate"]).exit_code == 0
    assert statistics.is_file()


def test_a_run_can_be_resumed_from_the_command_line(
    runner: CliRunner, prepared: tuple[Path, RunConfig]
) -> None:
    path, _ = prepared
    assert runner.invoke(app, ["train", "--config", str(path), "--no-evaluate"]).exit_code == 0
    result = runner.invoke(app, ["train", "--config", str(path), "--no-evaluate", "--resume"])
    assert result.exit_code == 0
    assert "Resuming" in result.output


def test_a_missing_dataset_is_reported_rather_than_raised(
    runner: CliRunner, tmp_path: Path
) -> None:
    payload = {**CONFIG, "data": {**CONFIG["data"], "root": str(tmp_path / "absent")}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(path)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_an_unregistered_model_is_reported_by_name(
    runner: CliRunner, prepared: tuple[Path, RunConfig]
) -> None:
    path, _ = prepared
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["model"] = {"name": "not-a-model"}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(path)])
    assert result.exit_code == 1
    assert "not-a-model" in result.output + (result.stderr or "")
