from pathlib import Path

import h5py
import numpy as np
import pytest

from nnphysics.core.errors import ConfigurationError, ValidationError
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import Manifest, write_manifest
from nnphysics.data.store import FIELDS_GROUP, content_hash
from nnphysics.data.verify import verify_dataset

Built = tuple[Path, Manifest, object]


def test_a_freshly_generated_dataset_verifies(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    report = verify_dataset(directory, sample=3)
    assert report.ok
    assert report.failures == ()
    assert report.shards_checked == len(manifest.shards)
    assert report.trajectories_checked == 3


def test_a_dataset_altered_by_one_byte_fails(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    shard = directory / manifest.shards[0].path
    data = bytearray(shard.read_bytes())
    # A byte in the middle of the file, so it lands in stored data rather than in the
    # superblock, which HDF5 would refuse to open at all.
    position = len(data) // 2
    data[position] ^= 0xFF
    shard.write_bytes(bytes(data))

    report = verify_dataset(directory, sample=0)
    assert not report.ok
    assert any(manifest.shards[0].path in failure for failure in report.failures)


def test_data_that_does_not_match_its_seed_fails(built_nbody: Built) -> None:
    """Rewriting a trajectory and its hash defeats the hash check, not the re-derivation."""
    directory, manifest, _ = built_nbody
    record = manifest.trajectories[0]
    with h5py.File(directory / record.shard, "r+") as handle:
        group = handle[FIELDS_GROUP]
        assert isinstance(group, h5py.Group)
        dataset = group["position"]
        assert isinstance(dataset, h5py.Dataset)
        dataset[record.row] = np.asarray(dataset[record.row]) + 1.0

    repaired = manifest.model_copy(
        update={
            "shards": tuple(
                entry.model_copy(update={"content_hash": content_hash(directory / entry.path)})
                if entry.path == record.shard
                else entry
                for entry in manifest.shards
            )
        }
    )
    write_manifest(directory / MANIFEST_NAME, repaired)

    hashes_only = verify_dataset(directory, sample=0)
    assert hashes_only.ok

    report = verify_dataset(directory, sample=len(manifest.trajectories))
    assert not report.ok
    assert any(record.id in failure and "position" in failure for failure in report.failures)


def test_a_missing_shard_is_reported(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    (directory / manifest.shards[0].path).unlink()
    report = verify_dataset(directory)
    assert not report.ok
    assert any("missing" in failure for failure in report.failures)


def test_the_same_trajectories_are_sampled_every_run(built_nbody: Built) -> None:
    directory, _, _ = built_nbody
    assert verify_dataset(directory, sample=2) == verify_dataset(directory, sample=2)


def test_checking_hashes_only_re_derives_nothing(built_nbody: Built) -> None:
    directory, _, _ = built_nbody
    report = verify_dataset(directory, sample=0)
    assert report.ok
    assert report.trajectories_checked == 0


def test_a_sample_larger_than_the_dataset_checks_all_of_it(built_nbody: Built) -> None:
    directory, manifest, _ = built_nbody
    report = verify_dataset(directory, sample=len(manifest.trajectories) + 10)
    assert report.ok
    assert report.trajectories_checked == len(manifest.trajectories)


def test_a_missing_manifest_cannot_be_checked_at_all(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        verify_dataset(tmp_path)


def test_a_negative_sample_is_rejected(built_nbody: Built) -> None:
    directory, _, _ = built_nbody
    with pytest.raises(ValidationError):
        verify_dataset(directory, sample=-1)
