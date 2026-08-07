from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nnphysics.core.errors import ConfigurationError
from nnphysics.evals.result import RESULT_SCHEMA_VERSION
from nnphysics.reporting.record import (
    RECORD_SCHEMA_VERSION,
    RunRecord,
    read_record,
    upgrade_record,
    write_record,
)


def written(path: Path, raw: dict[str, Any]) -> Path:
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def as_version_one(record: RunRecord) -> dict[str, Any]:
    """The record as the build before per rollout scalars would have written it."""
    raw = record.model_dump(mode="json")
    raw["evaluation"]["schema_version"] = 1
    for entry in raw["evaluation"]["results"]:
        for rollout in entry["rollouts"]:
            del rollout["scalars"]
    return raw


Factory = Callable[..., RunRecord]


class TestRoundTrip:
    def test_a_record_survives_being_written_and_read(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        record = make_record()

        write_record(tmp_path / "record.json", record)

        assert read_record(tmp_path / "record.json") == record

    def test_nothing_is_lost_from_the_embedded_evaluation(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        record = make_record()

        write_record(tmp_path / "record.json", record)
        again = read_record(tmp_path / "record.json")

        assert again.evaluation == record.evaluation
        assert again.config == record.config

    def test_a_record_written_twice_is_byte_identical(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        record = make_record()

        write_record(tmp_path / "one.json", record)
        write_record(tmp_path / "two.json", record)

        assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()

    def test_a_run_that_diverged_can_still_be_read_back(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        """A loss that went to infinity is a measurement, not a value to clean up.

        Found by the phase 10 fault suite, whose excessive learning rate produces exactly
        this. The default serialisation writes a non finite number as `null`, which then
        refuses to read back as a float, so the run that most needed explaining was the
        one whose record could not be opened.
        """
        diverged = make_record(scale=math.inf)

        write_record(tmp_path / "record.json", diverged)
        again = read_record(tmp_path / "record.json")

        assert math.isinf(again.evaluation.results[0].scalar("rollout_error", "error.final"))

    def test_a_number_that_is_not_a_number_survives_too(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        diverged = make_record(scale=math.nan)

        write_record(tmp_path / "record.json", diverged)
        again = read_record(tmp_path / "record.json")

        assert math.isnan(again.evaluation.results[0].scalar("rollout_error", "error.final"))


class TestMigration:
    def test_a_record_under_the_previous_result_schema_still_reads(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        path = written(tmp_path / "old.json", as_version_one(make_record()))

        record = read_record(path)

        assert record.evaluation.schema_version == RESULT_SCHEMA_VERSION

    def test_the_upgrade_leaves_the_old_rollouts_without_scalars(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        path = written(tmp_path / "old.json", as_version_one(make_record()))

        record = read_record(path)

        assert all(
            rollout.scalars == {}
            for entry in record.evaluation.results
            for rollout in entry.rollouts
        )

    def test_everything_else_survives_the_upgrade(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        original = make_record()
        path = written(tmp_path / "old.json", as_version_one(original))

        record = read_record(path)

        assert record.evaluation.results[0].metrics == original.evaluation.results[0].metrics
        assert record.run_id == original.run_id

    def test_an_upgraded_record_round_trips_again(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        path = written(tmp_path / "old.json", as_version_one(make_record()))
        record = read_record(path)

        write_record(tmp_path / "new.json", record)

        assert read_record(tmp_path / "new.json") == record

    def test_a_future_record_is_refused_rather_than_guessed_at(self, make_record: Factory) -> None:
        raw = make_record().model_dump(mode="json")
        raw["schema_version"] = RECORD_SCHEMA_VERSION + 1

        with pytest.raises(ConfigurationError, match="newer than the version"):
            upgrade_record(raw)

    def test_a_future_evaluation_is_refused_too(self, make_record: Factory) -> None:
        raw = make_record().model_dump(mode="json")
        raw["evaluation"]["schema_version"] = RESULT_SCHEMA_VERSION + 1

        with pytest.raises(ConfigurationError, match="newer than the version"):
            upgrade_record(raw)

    def test_a_record_without_a_version_is_refused(self, make_record: Factory) -> None:
        raw = make_record().model_dump(mode="json")
        del raw["schema_version"]

        with pytest.raises(ConfigurationError, match="no integer record schema version"):
            upgrade_record(raw)

    def test_a_record_without_an_evaluation_is_refused(self, make_record: Factory) -> None:
        raw = make_record().model_dump(mode="json")
        del raw["evaluation"]

        with pytest.raises(ConfigurationError, match="must carry an evaluation"):
            upgrade_record(raw)


class TestReading:
    def test_a_missing_file_is_a_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="cannot read run record"):
            read_record(tmp_path / "absent.json")

    def test_a_file_that_is_not_json_is_a_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="not valid JSON"):
            read_record(path)

    def test_a_json_list_is_a_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        path.write_text("[]", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="must hold a JSON object"):
            read_record(path)

    def test_an_invalid_record_names_the_file(self, make_record: Factory, tmp_path: Path) -> None:
        raw = make_record().model_dump(mode="json")
        raw["run_id"] = ""
        path = written(tmp_path / "record.json", raw)

        with pytest.raises(ConfigurationError, match="invalid run record"):
            read_record(path)


class TestDescription:
    def test_it_reports_the_splits_in_the_order_they_were_evaluated(
        self, make_record: Factory
    ) -> None:
        assert make_record().splits == ("test", "held_out")

    def test_it_reports_each_predictor_once(self, make_record: Factory) -> None:
        assert make_record().predictors == ("reference", "persistence")

    def test_it_reports_the_system_and_the_suite(self, make_record: Factory) -> None:
        record = make_record()

        assert record.system == "toy"
        assert record.suite == "standard"
