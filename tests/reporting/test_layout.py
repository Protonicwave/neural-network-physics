from __future__ import annotations

from pathlib import Path

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ValidationError
from nnphysics.reporting.layout import (
    RECORD_NAME,
    RunPaths,
    find_record,
    find_records,
    result_name,
    run_paths,
)

CONFIG = {
    "name": "example",
    "system": {"name": "toy"},
    "data": {
        "n_trajectories": 8,
        "regimes": ["hot"],
        "held_out_regimes": ["cold"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
    },
    "model": {"name": "placeholder"},
    "evaluation": {"metrics": ["rollout_error"]},
}


def make_run(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / RECORD_NAME).write_text("{}", encoding="utf-8")
    return directory


class TestPaths:
    def test_every_artefact_sits_in_the_run_directory(self, tmp_path: Path) -> None:
        paths = RunPaths(tmp_path / "run")

        for path in (paths.record, paths.snapshots, paths.markdown, paths.html, paths.plots):
            assert path.parent == paths.root

    def test_a_result_is_named_after_its_suite(self, tmp_path: Path) -> None:
        assert RunPaths(tmp_path).result("standard").name == result_name("standard")

    def test_it_creates_the_directories_a_report_needs(self, tmp_path: Path) -> None:
        paths = RunPaths(tmp_path / "run")

        paths.ensure()

        assert paths.plots.is_dir()

    def test_a_path_inside_the_run_is_recorded_relative_to_it(self, tmp_path: Path) -> None:
        paths = RunPaths(tmp_path / "run")

        assert paths.relative(paths.plots / "a.png") == "plots/a.png"

    def test_a_path_outside_the_run_is_refused(self, tmp_path: Path) -> None:
        paths = RunPaths(tmp_path / "run")

        with pytest.raises(ValidationError, match="not inside the run directory"):
            paths.relative(tmp_path / "elsewhere.png")

    def test_a_configuration_decides_where_its_run_lives(self) -> None:
        config = RunConfig.model_validate(CONFIG)

        assert run_paths(config).root == config.run_dir
        assert config.run_id in run_paths(config).root.name


class TestFinding:
    def test_a_root_that_does_not_exist_holds_no_runs(self, tmp_path: Path) -> None:
        assert find_records(tmp_path / "absent") == ()

    def test_a_directory_without_a_record_is_not_a_run(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-run").mkdir()

        assert find_records(tmp_path) == ()

    def test_it_finds_every_record_in_a_stable_order(self, tmp_path: Path) -> None:
        make_run(tmp_path, "b-2222")
        make_run(tmp_path, "a-1111")

        found = find_records(tmp_path)

        assert [path.parent.name for path in found] == ["a-1111", "b-2222"]

    def test_a_run_can_be_found_by_its_identifier(self, tmp_path: Path) -> None:
        directory = make_run(tmp_path, "example-1111")

        assert find_record(tmp_path, "1111").parent == directory

    def test_a_run_can_be_found_by_its_directory_name(self, tmp_path: Path) -> None:
        directory = make_run(tmp_path, "example-1111")

        assert find_record(tmp_path, "example-1111").parent == directory

    def test_an_unknown_identifier_is_refused(self, tmp_path: Path) -> None:
        make_run(tmp_path, "example-1111")

        with pytest.raises(ValidationError, match="no run under"):
            find_record(tmp_path, "2222")

    def test_an_ambiguous_identifier_is_refused_rather_than_picked(self, tmp_path: Path) -> None:
        make_run(tmp_path, "one-1111")
        make_run(tmp_path, "two-1111")

        with pytest.raises(ValidationError, match="2 runs"):
            find_record(tmp_path, "1111")
