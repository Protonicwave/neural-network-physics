from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nnphysics.agent.faults import fault
from nnphysics.cli.app import app
from nnphysics.core.config import RunConfig
from nnphysics.reporting.layout import fault_paths
from nnphysics.reporting.record import read_record

CONFIG: dict[str, Any] = {
    "name": "cli-diagnose",
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
        "epochs": 4,
        "batch_size": 4,
        "learning_rate": 0.01,
        "warmup_epochs": 0,
        "curriculum": [1, 2],
        "curriculum_epochs": [0, 2],
        "validation_steps": 2,
        "window_stride": 4,
    },
    "evaluation": {
        "name": "cli-suite",
        "metrics": ["one_step_error", "rollout_error", "symmetry_violation"],
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
    """A configuration file small enough that the fault suite runs inside a test."""
    payload = {
        **CONFIG,
        "data": {**CONFIG["data"], "root": str(tmp_path / "data")},
        "output_dir": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path, RunConfig.model_validate(payload)


def _score(runner: CliRunner, path: Path, output: Path, *faults: str) -> Any:
    arguments = ["diagnose", "score", "--config", str(path), "--rule-based", "-o", str(output)]
    for name in faults:
        arguments += ["--fault", name]
    return runner.invoke(app, arguments)


class TestScore:
    def test_the_suite_writes_a_scored_table_and_the_numbers_behind_it(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> None:
        path, _ = prepared
        output = tmp_path / "results"

        result = _score(runner, path, output, "no_curriculum")

        assert result.exit_code == 0, result.output
        assert (output / "diagnosis.md").is_file()
        scores = json.loads((output / "fault-scores.json").read_text(encoding="utf-8"))
        assert scores["cards"][0]["source"] == "rule_based"
        assert scores["cards"][0]["outcomes"][0]["true_cause"] == "rollout_curriculum"

    def test_the_baseline_and_the_fault_get_their_own_directories(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> None:
        """Three faults hash to the baseline's identifier and would otherwise overwrite it."""
        path, config = prepared

        _score(runner, path, tmp_path / "results", "wrong_normalisation")

        baseline = fault_paths(config, "baseline")
        injected = fault_paths(config, "wrong_normalisation")
        assert baseline.root != injected.root
        assert baseline.record.is_file()
        assert injected.record.is_file()

    def test_a_fault_run_is_recorded_as_an_ordinary_run(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> None:
        """A recognisable fault run would be diagnosed from its label, not its numbers."""
        path, config = prepared

        _score(runner, path, tmp_path / "results", "no_curriculum")

        record = read_record(
            fault_paths(fault("no_curriculum").apply(config), "no_curriculum").record
        )
        assert record.training is not None
        assert record.evaluation.results

    def test_an_injected_fault_leaves_the_configuration_alone(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> None:
        path, config = prepared

        _score(runner, path, tmp_path / "results", "wrong_normalisation")

        record = read_record(fault_paths(config, "wrong_normalisation").record)
        assert record.config == config

    def test_a_second_run_reuses_what_the_first_trained(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> None:
        path, _ = prepared
        output = tmp_path / "results"
        _score(runner, path, output, "no_curriculum")

        again = _score(runner, path, output, "no_curriculum")

        assert again.exit_code == 0, again.output
        assert "reusing" in again.output

    def test_the_interrupted_run_loses_its_optimiser_state_and_still_finishes(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> None:
        """The one fault that leaves no trace anywhere except in the loss curve."""
        path, config = prepared

        result = _score(runner, path, tmp_path / "results", "no_optimiser_state")

        assert result.exit_code == 0, result.output
        assert "Discarded the optimiser moments" in result.output
        record = read_record(fault_paths(config, "no_optimiser_state").record)
        assert record.training is not None
        assert len(record.training.epochs) == config.training.epochs

    def test_an_unknown_fault_is_reported_by_name(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> None:
        path, _ = prepared

        result = _score(runner, path, tmp_path / "results", "nonsense")

        assert result.exit_code != 0
        assert "nonsense" in result.output


class TestDiagnose:
    @pytest.fixture
    def two_runs(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig], tmp_path: Path
    ) -> tuple[Path, str, str]:
        """A baseline and a faulty run, under a runs root, ready to be compared."""
        path, config = prepared
        _score(runner, path, tmp_path / "results", "no_curriculum")
        faulty = fault("no_curriculum").apply(config)
        return (
            config.output_dir,
            fault_paths(config, "baseline").root.name,
            fault_paths(faulty, "no_curriculum").root.name,
        )

    def test_the_context_can_be_printed_without_calling_anything(
        self, runner: CliRunner, two_runs: tuple[Path, str, str]
    ) -> None:
        root, baseline, candidate = two_runs

        result = runner.invoke(
            app,
            [
                "diagnose",
                "--root",
                str(root),
                "--baseline",
                baseline,
                "--candidate",
                candidate,
                "--context-only",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "## Configuration differences" in result.output
        assert "training.curriculum" in result.output

    def test_the_rule_based_diagnoser_needs_no_credential(
        self, runner: CliRunner, two_runs: tuple[Path, str, str]
    ) -> None:
        root, baseline, candidate = two_runs

        result = runner.invoke(
            app,
            [
                "diagnose",
                "--root",
                str(root),
                "--baseline",
                baseline,
                "--candidate",
                candidate,
                "--rule-based",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Diagnosed by rule_based" in result.output

    def test_two_runs_are_required(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["diagnose"])

        assert result.exit_code == 2
        assert "--baseline" in result.output

    def test_a_missing_agent_configuration_is_a_sentence(
        self, runner: CliRunner, two_runs: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        root, baseline, candidate = two_runs

        result = runner.invoke(
            app,
            [
                "diagnose",
                "--root",
                str(root),
                "--baseline",
                baseline,
                "--candidate",
                candidate,
                "--agent-config",
                str(tmp_path / "absent.yaml"),
            ],
        )

        assert result.exit_code == 1
        assert "--rule-based" in result.output

    def test_an_unknown_run_is_reported(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["diagnose", "--root", str(tmp_path), "--baseline", "absent", "--candidate", "gone"],
        )

        assert result.exit_code == 1
        assert "absent" in result.output
