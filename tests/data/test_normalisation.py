from pathlib import Path

import h5py
import numpy as np
import pytest

from nnphysics.core.errors import ConfigurationError, ValidationError
from nnphysics.data.manifest import Manifest, Split
from nnphysics.data.normalisation import (
    FieldStats,
    Normalisation,
    fit_normalisation,
    read_normalisation,
    write_normalisation,
)
from nnphysics.data.store import FIELDS_GROUP, ShardReader

Built = tuple[Path, Manifest, object]


def _corrupt(directory: Path, manifest: Manifest, splits: tuple[Split, ...]) -> None:
    """Replace the stored data of every trajectory in the given splits."""
    rows: dict[str, list[int]] = {}
    for split in splits:
        for identifier in manifest.split(split):
            record = manifest.trajectory(identifier)
            rows.setdefault(record.shard, []).append(record.row)
    assert rows, "the splits to corrupt hold no trajectories"
    for shard, targets in rows.items():
        with h5py.File(directory / shard, "r+") as handle:
            group = handle[FIELDS_GROUP]
            assert isinstance(group, h5py.Group)
            for name in group:
                dataset = group[name]
                assert isinstance(dataset, h5py.Dataset)
                for row in targets:
                    dataset[row] = np.full(dataset.shape[1:], 1234.5)


def test_no_leakage_from_validation_and_test_into_the_statistics(built_nbody: Built) -> None:
    """The test this phase exists for.

    Normalisation is fitted to the training split alone, so replacing every number in
    the validation and test splits must leave the statistics bit for bit unchanged. If
    this fails, every reported test error is optimistic by an unknown amount.
    """
    directory, manifest, _ = built_nbody
    before = fit_normalisation(directory, manifest)
    _corrupt(directory, manifest, (Split.VALIDATION, Split.TEST))
    after = fit_normalisation(directory, manifest)

    assert after.names == before.names
    for name in before.names:
        assert after.stats[name] == before.stats[name]


def test_the_leakage_test_can_fail(built_nbody: Built) -> None:
    """Changing the training data must move the statistics.

    Without this, a fit that read nothing at all would pass the test above.
    """
    directory, manifest, _ = built_nbody
    before = fit_normalisation(directory, manifest)
    _corrupt(directory, manifest, (Split.TRAIN,))
    after = fit_normalisation(directory, manifest)

    assert any(after.stats[name] != before.stats[name] for name in before.names)


def test_the_held_out_regime_does_not_reach_the_statistics(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    before = fit_normalisation(directory, manifest)
    _corrupt(directory, manifest, (Split.HELD_OUT,))
    assert fit_normalisation(directory, manifest) == before


def test_the_statistics_cover_every_field(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    normalisation = fit_normalisation(directory, manifest)
    assert normalisation.names == ("mass", "position", "velocity")
    assert all(stats.count > 0 for stats in normalisation.stats.values())


def test_the_statistics_match_a_direct_computation(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    gathered: list[np.ndarray] = []
    for identifier in manifest.split(Split.TRAIN):
        record = manifest.trajectory(identifier)
        with ShardReader(directory / record.shard) as reader:
            fields, _ = reader.window(record.row, 0, reader.n_steps)
        gathered.append(fields["position"].ravel())
    stacked = np.concatenate(gathered)

    stats = fit_normalisation(directory, manifest).stats["position"]
    assert stats.mean == pytest.approx(float(np.mean(stacked)))
    assert stats.std == pytest.approx(float(np.std(stacked)))
    assert stats.count == stacked.size


def test_applying_and_inverting_returns_the_original() -> None:
    normalisation = Normalisation({"a": FieldStats(mean=2.0, std=0.5, count=10)})
    values = {"a": np.linspace(-1.0, 1.0, 7)}
    restored = normalisation.invert(normalisation.apply(values))
    assert np.allclose(restored["a"], values["a"])


def test_applying_centres_and_scales() -> None:
    normalisation = Normalisation({"a": FieldStats(mean=2.0, std=0.5, count=10)})
    normalised = normalisation.apply({"a": np.array([2.0, 2.5])})
    assert np.allclose(normalised["a"], [0.0, 1.0])


def test_a_constant_field_is_scaled_by_one_and_says_so() -> None:
    stats = FieldStats(mean=3.0, std=0.0, count=10)
    assert stats.degenerate
    assert stats.scale == 1.0
    normalised = Normalisation({"a": stats}).apply({"a": np.array([3.0, 4.0])})
    assert np.allclose(normalised["a"], [0.0, 1.0])


def test_normalising_a_field_with_no_statistics_is_rejected() -> None:
    normalisation = Normalisation({"a": FieldStats(mean=0.0, std=1.0, count=1)})
    with pytest.raises(ValidationError, match="no normalisation statistics"):
        normalisation.apply({"b": np.zeros(3)})


def test_unusable_statistics_are_rejected() -> None:
    with pytest.raises(ValidationError):
        FieldStats(mean=0.0, std=-1.0, count=1)
    with pytest.raises(ValidationError):
        FieldStats(mean=float("nan"), std=1.0, count=1)
    with pytest.raises(ValidationError):
        FieldStats(mean=0.0, std=1.0, count=0)
    with pytest.raises(ValidationError):
        Normalisation({})


def test_fitting_from_an_empty_split_is_rejected(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    empty = manifest.model_copy(
        update={
            "splits": manifest.splits.model_copy(
                update={"members": {Split.TRAIN: manifest.split(Split.TRAIN)}}
            )
        }
    )
    with pytest.raises(ValidationError, match="empty split"):
        fit_normalisation(directory, empty, split=Split.VALIDATION)


def test_the_artefact_round_trips(tmp_path: Path) -> None:
    normalisation = Normalisation(
        {
            "a": FieldStats(mean=1.5, std=0.25, count=100),
            "b": FieldStats(mean=0.0, std=0.0, count=100),
        }
    )
    path = tmp_path / "normalisation.json"
    write_normalisation(path, normalisation)
    assert read_normalisation(path) == normalisation


@pytest.mark.parametrize(
    "text",
    ['{"schema_version": 99, "fields": {}}', '{"schema_version": 1, "fields": {}}', "not json"],
)
def test_a_malformed_artefact_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "normalisation.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        read_normalisation(path)


def test_a_missing_artefact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        read_normalisation(tmp_path / "absent.json")
