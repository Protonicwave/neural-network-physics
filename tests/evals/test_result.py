from __future__ import annotations

import json
from pathlib import Path

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ConfigurationError
from nnphysics.data.manifest import Manifest
from nnphysics.evals.result import (
    RESULT_SCHEMA_VERSION,
    MetricRecord,
    PredictorResult,
    RolloutRecord,
    SuiteResult,
    SuiteSettings,
    read_result,
    write_result,
)
from nnphysics.evals.runner import run_suite
from nnphysics.systems import build_system

Dataset = tuple[Path, Manifest, RunConfig]


def make_result() -> SuiteResult:
    return SuiteResult(
        code_version="0.1.0",
        run_id="abc",
        dataset_id="def",
        system="toy",
        seed=0,
        settings=SuiteSettings(
            name="suite",
            metrics=("rollout_error",),
            rollout_steps=4,
            n_initial_conditions=1,
            error_thresholds=(0.1,),
            symmetry_steps=2,
            distribution_window=0.25,
            divergence_factor=1e3,
        ),
        invariants={},
        results=(
            PredictorResult(
                predictor="persistence",
                spec="persistence",
                split="test",
                regimes=("one",),
                rollouts=(
                    RolloutRecord(
                        trajectory="one/00000",
                        regime="one",
                        split="test",
                        steps_requested=4,
                        steps_completed=4,
                        stop_reason="completed",
                        seconds=0.01,
                    ),
                ),
                metrics=(
                    MetricRecord(
                        name="rollout_error",
                        scalars={"error.final": 0.5},
                        series={"error": (0.0, 0.5)},
                    ),
                ),
                seconds_per_step=0.0025,
                completed=True,
            ),
        ),
    )


class TestRoundTrip:
    def test_a_result_survives_being_written_and_read(self, tmp_path: Path) -> None:
        path = tmp_path / "evaluation.json"

        write_result(path, make_result())

        assert read_result(path) == make_result()

    def test_plot_data_is_kept(self, tmp_path: Path) -> None:
        """Discarding it would force a re-run to draw a curve."""
        path = tmp_path / "evaluation.json"
        write_result(path, make_result())

        recovered = read_result(path)

        assert recovered.results[0].metrics[0].series["error"] == (0.0, 0.5)

    def test_the_file_is_plain_json(self, tmp_path: Path) -> None:
        path = tmp_path / "evaluation.json"
        write_result(path, make_result())

        assert json.loads(path.read_text(encoding="utf-8"))["system"] == "toy"


class TestLookups:
    def test_a_predictor_and_split_can_be_found(self) -> None:
        assert make_result().result("persistence", "test").spec == "persistence"

    def test_a_missing_result_says_what_was_asked_for(self) -> None:
        with pytest.raises(KeyError, match="persistence"):
            make_result().result("persistence", "held_out")

    def test_a_scalar_can_be_read_by_metric_and_name(self) -> None:
        assert make_result().results[0].scalar("rollout_error", "error.final") == 0.5

    def test_a_missing_metric_is_reported(self) -> None:
        with pytest.raises(KeyError, match="no metric"):
            make_result().results[0].scalar("invariant_drift", "worst_violation")


class TestReading:
    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="cannot read result"):
            read_result(tmp_path / "absent.json")

    def test_a_file_that_is_not_json_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "evaluation.json"
        path.write_text("{", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="not valid JSON"):
            read_result(path)

    def test_an_unsupported_schema_version_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "evaluation.json"
        path.write_text(json.dumps({"schema_version": RESULT_SCHEMA_VERSION + 1}), "utf-8")

        with pytest.raises(ConfigurationError, match="result schema version"):
            read_result(path)

    def test_a_malformed_result_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "evaluation.json"
        path.write_text(json.dumps({"schema_version": RESULT_SCHEMA_VERSION}), "utf-8")

        with pytest.raises(ConfigurationError, match="invalid result"):
            read_result(path)


class TestAgainstARealRun:
    def test_a_generated_result_reads_back(self, dataset: Dataset, tmp_path: Path) -> None:
        directory, manifest, config = dataset
        result = run_suite(
            build_system(config.system.name, config.system.parameters),
            directory,
            manifest,
            config.evaluation,
            seed=config.seed,
            run_id=config.run_id,
        )
        path = tmp_path / "evaluation.json"

        write_result(path, result)

        assert read_result(path) == result

    def test_it_records_what_produced_it(self, dataset: Dataset) -> None:
        _, manifest, config = dataset
        result = run_suite(
            build_system(config.system.name, config.system.parameters),
            dataset[0],
            manifest,
            config.evaluation,
            seed=config.seed,
            run_id=config.run_id,
        )

        assert result.run_id == config.run_id
        assert result.dataset_id == manifest.dataset_id
        assert result.system == config.system.name
        assert result.seed == config.seed
