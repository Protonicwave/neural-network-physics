from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ConfigurationError, ValidationError
from nnphysics.data.manifest import Manifest, Split
from nnphysics.evals.runner import EvaluationCase, load_cases
from nnphysics.evals.snapshots import (
    DEFAULT_HORIZONS,
    Snapshot,
    SnapshotSet,
    capture_snapshots,
    read_snapshots,
    write_snapshots,
)
from nnphysics.systems import build_system

Dataset = tuple[Path, Manifest, RunConfig]

SPECS = ("reference", "persistence")


def make_snapshot(predictor: str = "reference", split: str = "test") -> Snapshot:
    values = np.arange(12.0).reshape(3, 2, 2)
    return Snapshot(
        predictor=predictor,
        split=split,
        trajectory="t-0",
        steps=(0, 2, 4),
        times=(0.0, 0.2, 0.4),
        predicted={"position": values},
        reference={"position": values + 1.0},
    )


def cases_of(
    dataset: Dataset, *, count: int | None = None, steps: int | None = None
) -> list[EvaluationCase]:
    """The initial conditions of the test split, as the runner would load them."""
    directory, manifest, config = dataset
    return load_cases(
        directory,
        manifest,
        build_system(config.system.name, config.system.parameters),
        split=Split.TEST,
        count=count or config.evaluation.n_initial_conditions,
        steps=steps or config.evaluation.rollout_steps,
    )


def capture(dataset: Dataset, horizons: int = DEFAULT_HORIZONS) -> SnapshotSet:
    """Capture snapshots of both predictors on the test split."""
    _, manifest, config = dataset
    return capture_snapshots(
        build_system(config.system.name, config.system.parameters),
        cases_of(dataset),
        SPECS,
        config.evaluation,
        substeps=manifest.spec.substeps,
        seed=config.seed,
        horizons=horizons,
    )


@pytest.fixture
def captured(dataset: Dataset) -> SnapshotSet:
    """Snapshots of both predictors on the test split of the shared dataset."""
    return capture(dataset)


class TestValidation:
    def test_the_two_states_must_carry_the_same_fields(self) -> None:
        values = np.zeros((2, 3))

        with pytest.raises(ValidationError, match="snapshot fields differ"):
            Snapshot(
                predictor="reference",
                split="test",
                trajectory="t-0",
                steps=(0, 1),
                times=(0.0, 0.1),
                predicted={"position": values},
                reference={"velocity": values},
            )

    def test_a_snapshot_must_keep_a_state(self) -> None:
        with pytest.raises(ValidationError, match="at least one state"):
            Snapshot(
                predictor="reference",
                split="test",
                trajectory="t-0",
                steps=(),
                times=(),
                predicted={"position": np.zeros((0, 3))},
                reference={"position": np.zeros((0, 3))},
            )


class TestStorage:
    def test_a_set_survives_being_written_and_read(self, tmp_path: Path) -> None:
        snapshots = SnapshotSet((make_snapshot(), make_snapshot("persistence")))

        write_snapshots(tmp_path / "s.npz", snapshots)
        again = read_snapshots(tmp_path / "s.npz")

        assert [entry.predictor for entry in again.snapshots] == ["reference", "persistence"]
        assert np.array_equal(
            again.snapshots[0].predicted["position"], snapshots.snapshots[0].predicted["position"]
        )

    def test_an_empty_set_still_writes_a_file(self, tmp_path: Path) -> None:
        write_snapshots(tmp_path / "s.npz", SnapshotSet())

        assert read_snapshots(tmp_path / "s.npz") == SnapshotSet()

    def test_a_missing_file_is_a_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="cannot read snapshots"):
            read_snapshots(tmp_path / "absent.npz")

    def test_a_file_that_is_not_an_archive_is_a_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "s.npz"
        path.write_bytes(b"not an archive")

        with pytest.raises(ConfigurationError):
            read_snapshots(path)

    def test_a_snapshot_can_be_found_by_predictor_and_split(self) -> None:
        snapshots = SnapshotSet((make_snapshot(), make_snapshot("persistence")))

        assert snapshots.find("persistence", "test") is not None
        assert snapshots.find("persistence", "held_out") is None


class TestCapture:
    def test_it_captures_one_snapshot_per_predictor(self, captured: SnapshotSet) -> None:
        assert [entry.predictor for entry in captured.snapshots] == list(SPECS)

    def test_it_keeps_the_first_and_the_last_state(self, captured: SnapshotSet) -> None:
        snapshot = captured.snapshots[0]

        assert snapshot.steps[0] == 0
        assert len(snapshot.steps) <= DEFAULT_HORIZONS

    def test_the_predicted_and_true_states_line_up(self, captured: SnapshotSet) -> None:
        snapshot = captured.snapshots[0]

        for name in snapshot.names:
            assert snapshot.predicted[name].shape == snapshot.reference[name].shape

    def test_the_reference_predictor_reproduces_the_truth(self, captured: SnapshotSet) -> None:
        snapshot = captured.snapshots[0]

        for name in snapshot.names:
            assert np.allclose(snapshot.predicted[name], snapshot.reference[name])

    def test_capturing_twice_gives_the_same_states(
        self, dataset: Dataset, captured: SnapshotSet
    ) -> None:
        again = capture(dataset)

        for one, two in zip(captured.snapshots, again.snapshots, strict=True):
            for name in one.names:
                assert np.array_equal(one.predicted[name], two.predicted[name])

    def test_it_needs_an_initial_condition(self, dataset: Dataset) -> None:
        _, _, config = dataset
        system = build_system(config.system.name, config.system.parameters)

        with pytest.raises(ValidationError, match="without an initial condition"):
            capture_snapshots(system, (), SPECS, config.evaluation, substeps=1, seed=0)

    def test_it_needs_a_horizon(self, dataset: Dataset) -> None:
        with pytest.raises(ValidationError, match="at least one state"):
            capture(dataset, horizons=0)
