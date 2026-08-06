from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nnphysics.cli.app import app
from nnphysics.core.config import RunConfig
from nnphysics.data.build import build_dataset
from nnphysics.evals.speed import SpeedReport
from nnphysics.reporting.layout import ensemble_paths, run_paths
from nnphysics.reporting.record import read_record

SUBSTEPS = 4
"""Divisible enough to give a ladder with rungs between its ends."""

CONFIG: dict[str, Any] = {
    "name": "cli-speed",
    "seed": 5,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 8,
        "n_steps": 10,
        "dt": 0.02,
        "substeps": SUBSTEPS,
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
    "ensemble": {"members": 2},
    "evaluation": {
        "name": "cli-suite",
        "metrics": ["one_step_error", "rollout_error", "calibration"],
        "predictors": ["reference", "persistence"],
        "rollout_steps": 5,
        "n_initial_conditions": 2,
        "symmetry_steps": 2,
    },
}

FAST = ["--trials", "2", "--warmup", "0", "--threads", "1"]
"""The smallest settings that still measure something. A benchmark in a test is checking
that the machinery works, not the machine."""


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


def report_of(config: RunConfig) -> SpeedReport:
    return SpeedReport.model_validate(
        json.loads(run_paths(config).benchmark.read_text(encoding="utf-8"))
    )


class TestTheSolverAlone:
    def test_a_configuration_that_trained_nothing_still_benchmarks(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """The solver's own curve is a result.

        It is the one everything else in the report is read against.
        """
        path, config = prepared

        result = runner.invoke(app, ["benchmark", "--config", str(path), *FAST])

        assert result.exit_code == 0, result.output
        assert "only the solver is timed" in result.output
        assert report_of(config).surrogates == ()

    def test_the_ladder_is_every_divisor_of_the_substeps_the_data_used(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        path, config = prepared
        assert runner.invoke(app, ["benchmark", "--config", str(path), *FAST]).exit_code == 0

        report = report_of(config)

        assert tuple(point.substeps for point in report.ladder) == (1, 2, 4)
        assert report.dataset_substeps == SUBSTEPS

    def test_the_rung_that_made_ground_truth_is_exact_and_the_coarse_ones_are_not(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """What anchors the ladder. Without it there is nothing to call accurate."""
        path, config = prepared
        assert runner.invoke(app, ["benchmark", "--config", str(path), *FAST]).exit_code == 0

        by_substeps = {point.substeps: point for point in report_of(config).ladder}

        assert by_substeps[SUBSTEPS].error == 0.0
        assert by_substeps[1].error > 0.0

    def test_coarsening_the_solver_makes_it_cheaper(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """The trade the whole comparison rests on."""
        path, config = prepared
        assert runner.invoke(app, ["benchmark", "--config", str(path), *FAST]).exit_code == 0

        by_substeps = {point.substeps: point for point in report_of(config).ladder}

        assert by_substeps[1].seconds_per_step < by_substeps[SUBSTEPS].seconds_per_step

    def test_the_thread_count_is_recorded(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """A speedup at an unrecorded thread count cannot be compared with another."""
        path, config = prepared
        assert runner.invoke(app, ["benchmark", "--config", str(path), *FAST]).exit_code == 0

        assert report_of(config).threads == 1


class TestATrainedModel:
    def test_the_model_this_run_trained_is_timed_without_being_named(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        path, config = prepared
        assert runner.invoke(app, ["train", "--config", str(path)]).exit_code == 0

        result = runner.invoke(app, ["benchmark", "--config", str(path), *FAST])

        assert result.exit_code == 0, result.output
        assert [point.predictor for point in report_of(config).surrogates] == ["constant"]

    def test_it_is_compared_against_a_rung_of_the_ladder_and_accounted_for(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        path, config = prepared
        assert runner.invoke(app, ["train", "--config", str(path)]).exit_code == 0
        assert runner.invoke(app, ["benchmark", "--config", str(path), *FAST]).exit_code == 0

        report = report_of(config)

        assert len(report.matched) == 1
        assert report.matched[0].matched_substeps in {point.substeps for point in report.ladder}
        assert report.costs[0].training_seconds > 0.0
        assert report.costs[0].generation_seconds > 0.0

    def test_the_benchmark_is_attached_to_the_record_that_already_existed(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """Into the record rather than only beside it.

        That is what makes the machine and the commit travel with the numbers.
        """
        path, config = prepared
        assert runner.invoke(app, ["train", "--config", str(path)]).exit_code == 0
        assert runner.invoke(app, ["benchmark", "--config", str(path), *FAST]).exit_code == 0

        record = read_record(run_paths(config).record)

        assert record.benchmark is not None
        assert record.benchmark.system == "nbody"
        assert record.training is not None
        assert "benchmark" in record.timings

    def test_a_checkpoint_can_be_named_instead(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        path, config = prepared
        assert runner.invoke(app, ["train", "--config", str(path)]).exit_code == 0
        checkpoint = run_paths(config).root / "checkpoints" / "best.pt"

        result = runner.invoke(
            app, ["benchmark", "--config", str(path), "--checkpoint", str(checkpoint), *FAST]
        )

        assert result.exit_code == 0, result.output
        assert len(report_of(config).surrogates) == 1


class TestTheEnsembleIsTimedToo:
    def test_an_ensemble_is_one_more_point_on_the_same_axes(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        path, config = prepared
        assert runner.invoke(app, ["ensemble", "train", "--config", str(path)]).exit_code == 0

        result = runner.invoke(app, ["benchmark", "--config", str(path), "--ensemble", *FAST])

        assert result.exit_code == 0, result.output
        assert "ensemble" in {point.predictor for point in report_of(config).surrogates}

    def test_its_training_cost_is_what_every_member_cost(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """A member evaluates nothing and writes no record.

        Its cost lives in its own history instead. An ensemble accounted as free to train
        would break even on its first rollout.
        """
        path, config = prepared
        assert runner.invoke(app, ["ensemble", "train", "--config", str(path)]).exit_code == 0
        assert (
            runner.invoke(app, ["benchmark", "--config", str(path), "--ensemble", *FAST]).exit_code
            == 0
        )

        costs = {cost.predictor: cost for cost in report_of(config).costs}
        members = [
            run_paths(config.for_member(index)).history for index in range(config.ensemble.members)
        ]

        assert all(path.is_file() for path in members)
        assert costs["ensemble"].training_seconds > 0.0


class TestTheEnsembleCommand:
    def test_it_trains_one_member_per_declared_seed(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        path, config = prepared

        result = runner.invoke(app, ["ensemble", "train", "--config", str(path)])

        assert result.exit_code == 0, result.output
        for index in range(config.ensemble.members):
            assert (run_paths(config.for_member(index)).root / "checkpoints" / "best.pt").is_file()

    def test_a_second_call_reuses_what_is_already_trained(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        path, _ = prepared
        assert runner.invoke(app, ["ensemble", "train", "--config", str(path)]).exit_code == 0

        result = runner.invoke(app, ["ensemble", "train", "--config", str(path)])

        assert result.exit_code == 0, result.output
        assert result.output.count("is already trained") == CONFIG["ensemble"]["members"]

    def test_the_ensemble_is_scored_into_a_directory_of_its_own(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """Member zero hashes to the same identifier.

        Without a label of its own the ensemble would write over one of the models it is
        made of.
        """
        path, config = prepared
        assert runner.invoke(app, ["ensemble", "train", "--config", str(path)]).exit_code == 0

        result = runner.invoke(app, ["ensemble", "run", "--config", str(path)])

        assert result.exit_code == 0, result.output
        paths = ensemble_paths(config)
        assert paths.root != run_paths(config.for_member(0)).root
        assert read_record(paths.record).predictors[-1] == "ensemble"

    def test_the_calibration_metric_scores_the_ensemble_and_nothing_else(
        self, runner: CliRunner, prepared: tuple[Path, RunConfig]
    ) -> None:
        """The point of the whole apparatus.

        The ensemble is the only predictor in the suite that states an uncertainty, and it
        is scored on that claim through the same interface as everything else.
        """
        path, config = prepared
        assert runner.invoke(app, ["ensemble", "train", "--config", str(path)]).exit_code == 0
        assert runner.invoke(app, ["ensemble", "run", "--config", str(path)]).exit_code == 0

        result = read_record(ensemble_paths(config).record).evaluation
        scored = {
            entry.predictor: entry.scalar("calibration", "steps")
            for entry in result.results
            if entry.split == "test"
        }

        assert scored["ensemble"] > 0.0
        assert scored["persistence"] == 0.0
        assert scored["reference"] == 0.0
