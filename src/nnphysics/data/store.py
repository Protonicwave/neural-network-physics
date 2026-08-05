"""Reading and writing shards.

One HDF5 file holds one regime's trajectories as stacked arrays, one dataset per field
with shape `(n_trajectories, n_steps, *field_shape)`, plus the times. Nothing else: no
names, no timestamps, no attributes. That is what lets the file's bytes be a function of
the numbers alone, so the same seed twice gives the same hash. Everything descriptive
lives in the manifest.

Chunking is chosen so that reading one short window costs one or two chunks rather than
a whole trajectory, because that is the access pattern the training loader has.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

import h5py
import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import FloatArray, Trajectory

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FIELDS_GROUP",
    "TIMES_DATASET",
    "ShardReader",
    "content_hash",
    "write_shard",
]

FIELDS_GROUP = "fields"
TIMES_DATASET = "times"

_TARGET_CHUNK_BYTES = 1 << 16
"""Chunks near this size keep the count low without making a window read decompress far
more than it needs."""

_HASH_BLOCK = 1 << 20

_MAX_GZIP_LEVEL = 9


def write_shard(
    path: Path,
    trajectories: Sequence[Trajectory],
    *,
    compression_level: int = 4,
) -> dict[str, tuple[int, ...]]:
    """Write one shard.

    Args:
        path: File to write. Overwritten if it exists.
        trajectories: Trajectories to stack, all with the same fields, shapes and times.
        compression_level: gzip level, zero for no compression.

    Returns:
        Field name to the shape of one state of that field.

    Raises:
        ValidationError: If there are no trajectories, they disagree on fields, shapes or
            length, or the compression level is out of range.
    """
    if not trajectories:
        raise ValidationError("cannot write a shard with no trajectories")
    if not 0 <= compression_level <= _MAX_GZIP_LEVEL:
        raise ValidationError(
            f"gzip level must lie in [0, {_MAX_GZIP_LEVEL}], got {compression_level}"
        )
    shapes = _check_uniform(trajectories)
    n_steps = len(trajectories[0])

    times = np.stack([trajectory.times for trajectory in trajectories])
    # track_times off throughout: HDF5 stamps object headers with a modification time by
    # default, which would make two runs of the same seed differ byte for byte.
    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_dataset(TIMES_DATASET, data=times, track_times=False)
        group = handle.create_group(FIELDS_GROUP, track_times=False)
        for name, shape in shapes.items():
            stacked = np.stack([trajectory.fields[name] for trajectory in trajectories])
            group.create_dataset(
                name,
                data=stacked,
                chunks=_chunk_shape(n_steps, shape, stacked.dtype.itemsize),
                compression="gzip" if compression_level > 0 else None,
                compression_opts=compression_level if compression_level > 0 else None,
                shuffle=compression_level > 0,
                track_times=False,
            )
    return shapes


def content_hash(path: Path) -> str:
    """Hash a file's bytes.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal digest.

    Raises:
        ValidationError: If the file cannot be read.
    """
    digest = hashlib.blake2b(digest_size=16)
    try:
        with path.open("rb") as handle:
            while block := handle.read(_HASH_BLOCK):
                digest.update(block)
    except OSError as error:
        raise ValidationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


class ShardReader:
    """Read access to one shard, holding the file open.

    Reads are sliced straight out of the file rather than loaded whole, so a dataset
    larger than memory can still be trained on.

    Args:
        path: The shard file.

    Raises:
        ValidationError: If the file cannot be opened or is not a shard.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            self._handle = h5py.File(path, "r")
            fields = self._handle[FIELDS_GROUP]
            if not isinstance(fields, h5py.Group):
                raise ValidationError(f"{path} has no field group")
            self._fields = {name: _dataset(fields, name) for name in sorted(fields)}
            self._times = _dataset(self._handle, TIMES_DATASET)
        except (OSError, KeyError) as error:
            raise ValidationError(f"cannot open shard {path}: {error}") from error

    @property
    def path(self) -> Path:
        """The file being read."""
        return self._path

    @property
    def names(self) -> tuple[str, ...]:
        """Field names in the shard, sorted."""
        return tuple(self._fields)

    @property
    def n_trajectories(self) -> int:
        """Rows in the shard."""
        return int(self._times.shape[0])

    @property
    def n_steps(self) -> int:
        """Recorded states per trajectory."""
        return int(self._times.shape[1])

    def window(self, row: int, start: int, length: int) -> tuple[dict[str, FloatArray], FloatArray]:
        """Read a contiguous run of states from one trajectory.

        Args:
            row: Row of the shard.
            start: First state to read.
            length: How many states to read.

        Returns:
            Field name to an array of shape `(length, *field_shape)`, and the times.

        Raises:
            ValidationError: If the window falls outside the shard.
        """
        self._check_window(row, start, length)
        stop = start + length
        fields = {
            name: np.asarray(dataset[row, start:stop], dtype=np.float64)
            for name, dataset in self._fields.items()
        }
        return fields, np.asarray(self._times[row, start:stop], dtype=np.float64)

    def trajectory(self, row: int) -> Trajectory:
        """Read one whole trajectory.

        Args:
            row: Row of the shard.

        Returns:
            The trajectory.

        Raises:
            ValidationError: If the row is out of range.
        """
        fields, times = self.window(row, 0, self.n_steps)
        return Trajectory(fields=fields, times=times)

    def close(self) -> None:
        """Close the file."""
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _check_window(self, row: int, start: int, length: int) -> None:
        """Raise if a window falls outside the shard."""
        if not 0 <= row < self.n_trajectories:
            raise ValidationError(
                f"{self._path} has {self.n_trajectories} trajectories, asked for row {row}"
            )
        if length < 1:
            raise ValidationError(f"a window must hold at least one state, got {length}")
        if start < 0 or start + length > self.n_steps:
            raise ValidationError(
                f"window [{start}, {start + length}) falls outside the {self.n_steps} "
                f"recorded states of {self._path}"
            )


def _dataset(group: h5py.Group | h5py.File, name: str) -> h5py.Dataset:
    """Read one dataset out of a group, failing if it is something else."""
    node = group[name]
    if not isinstance(node, h5py.Dataset):
        raise ValidationError(f"{name} is not a dataset")
    return node


def _check_uniform(trajectories: Sequence[Trajectory]) -> dict[str, tuple[int, ...]]:
    """Check that trajectories can be stacked, and return the shape of one state.

    Raises:
        ValidationError: If they disagree on fields, shapes or length.
    """
    first = trajectories[0]
    names = tuple(sorted(first.names))
    n_steps = len(first)
    shapes = {name: tuple(first.fields[name].shape[1:]) for name in names}
    for position, trajectory in enumerate(trajectories[1:], start=1):
        if tuple(sorted(trajectory.names)) != names:
            raise ValidationError(
                f"trajectory {position} has fields {tuple(sorted(trajectory.names))}, "
                f"expected {names}"
            )
        if len(trajectory) != n_steps:
            raise ValidationError(
                f"trajectory {position} has {len(trajectory)} states, expected {n_steps}"
            )
        for name in names:
            shape = tuple(trajectory.fields[name].shape[1:])
            if shape != shapes[name]:
                raise ValidationError(
                    f"trajectory {position} field {name!r} has state shape {shape}, "
                    f"expected {shapes[name]}"
                )
    return shapes


def _chunk_shape(n_steps: int, field_shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """Chunk one trajectory at a time, over as many states as fit a target chunk size."""
    per_state = itemsize * int(np.prod(field_shape, dtype=np.int64)) if field_shape else itemsize
    steps = max(1, min(n_steps, _TARGET_CHUNK_BYTES // max(per_state, 1)))
    return (1, steps, *field_shape)
