from pathlib import Path

import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import FloatArray, Trajectory
from nnphysics.data.store import ShardReader, content_hash, write_shard

N_STEPS = 5


def _trajectory(seed: int, n_bodies: int = 3, n_steps: int = N_STEPS) -> Trajectory:
    rng = np.random.default_rng(seed)
    fields: dict[str, FloatArray] = {
        "position": rng.standard_normal((n_steps, n_bodies, 2)),
        "mass": rng.standard_normal((n_steps, n_bodies)),
    }
    return Trajectory(fields=fields, times=np.arange(n_steps, dtype=np.float64) * 0.1)


def test_a_shard_round_trips_exactly(tmp_path: Path) -> None:
    trajectories = [_trajectory(seed) for seed in range(3)]
    shapes = write_shard(tmp_path / "shard.h5", trajectories)
    assert shapes == {"position": (3, 2), "mass": (3,)}

    with ShardReader(tmp_path / "shard.h5") as reader:
        assert reader.n_trajectories == 3
        assert reader.n_steps == N_STEPS
        assert reader.names == ("mass", "position")
        for row, expected in enumerate(trajectories):
            stored = reader.trajectory(row)
            assert np.array_equal(stored.times, expected.times)
            for name, array in expected.fields.items():
                assert np.array_equal(stored.fields[name], array)


def test_writing_the_same_numbers_twice_gives_the_same_bytes(tmp_path: Path) -> None:
    trajectories = [_trajectory(seed) for seed in range(3)]
    write_shard(tmp_path / "one.h5", trajectories)
    write_shard(tmp_path / "two.h5", trajectories)
    assert content_hash(tmp_path / "one.h5") == content_hash(tmp_path / "two.h5")


def test_different_numbers_give_different_bytes(tmp_path: Path) -> None:
    write_shard(tmp_path / "one.h5", [_trajectory(0)])
    write_shard(tmp_path / "two.h5", [_trajectory(1)])
    assert content_hash(tmp_path / "one.h5") != content_hash(tmp_path / "two.h5")


@pytest.mark.parametrize("level", [0, 9])
def test_compression_does_not_change_what_is_read_back(tmp_path: Path, level: int) -> None:
    expected = _trajectory(7)
    path = tmp_path / f"level-{level}.h5"
    write_shard(path, [expected], compression_level=level)
    with ShardReader(path) as reader:
        assert np.array_equal(reader.trajectory(0).fields["position"], expected.fields["position"])


def test_a_window_reads_the_states_it_names(tmp_path: Path) -> None:
    expected = _trajectory(11)
    write_shard(tmp_path / "shard.h5", [expected])
    with ShardReader(tmp_path / "shard.h5") as reader:
        fields, times = reader.window(0, 1, 3)
        assert np.array_equal(times, expected.times[1:4])
        assert np.array_equal(fields["position"], expected.fields["position"][1:4])


@pytest.mark.parametrize(("row", "start", "length"), [(1, 0, 1), (0, 0, 0), (0, 3, 5), (0, -1, 2)])
def test_a_window_outside_the_shard_is_rejected(
    tmp_path: Path, row: int, start: int, length: int
) -> None:
    write_shard(tmp_path / "shard.h5", [_trajectory(0)])
    with ShardReader(tmp_path / "shard.h5") as reader, pytest.raises(ValidationError):
        reader.window(row, start, length)


def test_trajectories_that_cannot_be_stacked_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        write_shard(tmp_path / "shard.h5", [_trajectory(0), _trajectory(1, n_bodies=4)])
    with pytest.raises(ValidationError):
        write_shard(tmp_path / "shard.h5", [_trajectory(0), _trajectory(1, n_steps=N_STEPS + 1)])


def test_an_empty_shard_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        write_shard(tmp_path / "shard.h5", [])


def test_an_out_of_range_compression_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        write_shard(tmp_path / "shard.h5", [_trajectory(0)], compression_level=10)


def test_opening_something_that_is_not_a_shard_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "not-a-shard.h5").write_text("plain text", encoding="utf-8")
    with pytest.raises(ValidationError):
        ShardReader(tmp_path / "not-a-shard.h5")


def test_hashing_a_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        content_hash(tmp_path / "absent.h5")
