from __future__ import annotations

import re
from pathlib import Path

from nnphysics.reporting.environment import (
    RECORDED_PACKAGES,
    UNKNOWN,
    describe_environment,
    git_commit,
)

COMMIT = "0" * 39 + "1"
_HEX = re.compile(r"^[0-9a-f]{40}$")


def make_repository(root: Path) -> Path:
    directory = root / ".git"
    directory.mkdir(parents=True)
    return directory


class TestEnvironment:
    def test_it_describes_the_interpreter(self) -> None:
        environment = describe_environment()

        assert environment.python.count(".") >= 1
        assert environment.implementation

    def test_it_counts_the_cores_a_timing_has_to_be_read_against(self) -> None:
        assert describe_environment().cpu_count >= 1

    def test_it_records_every_library_that_can_change_a_number(self) -> None:
        packages = describe_environment().packages

        assert set(packages) == set(RECORDED_PACKAGES)

    def test_an_installed_library_reports_a_version_rather_than_unknown(self) -> None:
        assert describe_environment().packages["numpy"] != UNKNOWN


class TestCommit:
    def test_it_reports_a_commit_or_nothing_and_never_anything_else(self, tmp_path: Path) -> None:
        # Whether a temporary directory has a checkout above it depends on the machine,
        # so what is asserted is the invariant that holds either way.
        for value in (git_commit(), git_commit(tmp_path)):
            assert value == "" or _HEX.match(value)

    def test_a_detached_head_is_read_straight_out_of_the_file(self, tmp_path: Path) -> None:
        make_repository(tmp_path).joinpath("HEAD").write_text(COMMIT, encoding="utf-8")

        assert git_commit(tmp_path) == COMMIT

    def test_a_branch_is_followed_to_its_loose_reference(self, tmp_path: Path) -> None:
        directory = make_repository(tmp_path)
        directory.joinpath("HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (directory / "refs" / "heads").mkdir(parents=True)
        (directory / "refs" / "heads" / "main").write_text(COMMIT, encoding="utf-8")

        assert git_commit(tmp_path) == COMMIT

    def test_a_branch_is_followed_into_the_packed_references(self, tmp_path: Path) -> None:
        directory = make_repository(tmp_path)
        directory.joinpath("HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        directory.joinpath("packed-refs").write_text(
            f"# pack-refs with: peeled\n{COMMIT} refs/heads/main\n", encoding="utf-8"
        )

        assert git_commit(tmp_path) == COMMIT

    def test_a_repository_with_no_commit_yet_reports_nothing(self, tmp_path: Path) -> None:
        directory = make_repository(tmp_path)
        directory.joinpath("HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

        assert git_commit(tmp_path) == ""

    def test_an_empty_head_reports_nothing(self, tmp_path: Path) -> None:
        make_repository(tmp_path).joinpath("HEAD").write_text("", encoding="utf-8")

        assert git_commit(tmp_path) == ""
