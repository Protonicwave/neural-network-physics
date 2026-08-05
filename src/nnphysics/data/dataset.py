"""The PyTorch view of a dataset: windows of consecutive states.

A sample is one input state and the `sequence_length` states that follow it, so a length
of one gives the single step pairs a one step model trains on and a longer length gives
the short rollouts that training against accumulated error needs. Phase 07 wants both,
and they differ only in this number.

Reads are sliced out of the HDF5 files rather than loaded up front, so the dataset a
model trains on may be larger than memory. File handles are opened lazily per process
and never pickled, because a handle inherited across a fork or sent to a spawned worker
is not usable.

Nothing here draws a random number. Sampling order is the loader's business, which is
what makes the epoch a worker count sees independent of how many workers there are.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from nnphysics.core.errors import ValidationError
from nnphysics.core.seeding import seed_sequence
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import Split, read_manifest
from nnphysics.data.store import ShardReader

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from nnphysics.data.manifest import Manifest
    from nnphysics.data.normalisation import Normalisation

__all__ = [
    "LOADER_SEED_STREAM",
    "Sample",
    "TrajectoryWindows",
    "WorkerInit",
    "make_worker_init",
]

LOADER_SEED_STREAM = "data.loader.worker"
"""Stream each loader worker's own generators are derived from."""

_MAX_UINT64 = (1 << 64) - 1


class Sample(TypedDict):
    """One training example.

    Attributes:
        inputs: Field name to the input state, shape `(*field_shape)`.
        targets: Field name to the states that follow, shape `(length, *field_shape)`.
        times: Times of the input state and its targets, shape `(length + 1,)`.
        trajectory: Index of the trajectory within the split, for tracing a sample back.
        start: Index of the input state within that trajectory.
    """

    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    times: torch.Tensor
    trajectory: int
    start: int


class TrajectoryWindows(Dataset[Sample]):
    """Windows of consecutive states drawn from one split of a dataset.

    Args:
        directory: The dataset directory.
        split: Which split to read.
        sequence_length: States to predict after the input state. One gives single step
            pairs.
        stride: Distance between the starts of consecutive windows. One uses every
            window, which is the default because dropping data needs a reason.
        normalisation: Statistics applied as samples are read, or `None` for raw units.
        manifest: The already read manifest, or `None` to read it from the directory.

    Raises:
        ConfigurationError: If the manifest is missing or invalid.
        ValidationError: If the split is empty, the split mixes state shapes, or the
            window does not fit in a trajectory.
    """

    def __init__(
        self,
        directory: Path,
        split: Split = Split.TRAIN,
        *,
        sequence_length: int = 1,
        stride: int = 1,
        normalisation: Normalisation | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        if sequence_length < 1:
            raise ValidationError(f"sequence length must be positive, got {sequence_length}")
        if stride < 1:
            raise ValidationError(f"stride must be positive, got {stride}")
        self._directory = directory
        self._split = split
        self._sequence_length = sequence_length
        self._normalisation = normalisation
        self._manifest = manifest if manifest is not None else read_manifest(
            directory / MANIFEST_NAME
        )
        # Raises if the split mixes shapes, which is the point: a batch could not be
        # collated from it, and finding that out here beats finding it out mid epoch.
        self._field_shapes = self._manifest.field_shapes(split)

        n_steps = self._manifest.spec.n_steps
        window = sequence_length + 1
        if window > n_steps:
            raise ValidationError(
                f"a window of {window} states does not fit in trajectories of {n_steps}"
            )
        self._members = self._manifest.split(split)
        self._starts = tuple(range(0, n_steps - window + 1, stride))
        self._readers: dict[str, ShardReader] = {}
        self._reader_pid: int | None = None

    @property
    def split(self) -> Split:
        """The split being read."""
        return self._split

    @property
    def sequence_length(self) -> int:
        """States predicted after the input state."""
        return self._sequence_length

    @property
    def field_shapes(self) -> dict[str, tuple[int, ...]]:
        """Shape of one state of each field."""
        return dict(self._field_shapes)

    @property
    def n_trajectories(self) -> int:
        """Trajectories in the split."""
        return len(self._members)

    @property
    def windows_per_trajectory(self) -> int:
        """Windows drawn from each trajectory."""
        return len(self._starts)

    def __len__(self) -> int:
        return self.n_trajectories * self.windows_per_trajectory

    def __getitem__(self, index: int) -> Sample:
        """Read one window.

        Args:
            index: Position in the split, laid out trajectory major.

        Returns:
            The sample.

        Raises:
            IndexError: If the index is out of range.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"sample {index} is outside a dataset of {len(self)}")
        trajectory, position = divmod(index, self.windows_per_trajectory)
        start = self._starts[position]
        record = self._manifest.trajectory(self._members[trajectory])
        fields, times = self._reader(record.shard).window(
            record.row, start, self._sequence_length + 1
        )
        if self._normalisation is not None:
            fields = self._normalisation.apply(fields)
        return Sample(
            inputs={name: _tensor(array[0]) for name, array in fields.items()},
            targets={name: _tensor(array[1:]) for name, array in fields.items()},
            # Times stay double precision. They index a rollout rather than feed a layer,
            # and single precision cannot hold a late step of a long one exactly.
            times=torch.from_numpy(np.ascontiguousarray(times)),
            trajectory=trajectory,
            start=start,
        )

    def close(self) -> None:
        """Close every open shard handle."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._reader_pid = None

    def _reader(self, shard: str) -> ShardReader:
        """Return this process's handle on a shard, opening it if needed."""
        pid = os.getpid()
        if self._reader_pid != pid:
            # Inherited across a fork: the handles belong to the parent, not to us.
            self._readers = {}
            self._reader_pid = pid
        if shard not in self._readers:
            self._readers[shard] = ShardReader(self._directory / shard)
        return self._readers[shard]

    def __getstate__(self) -> dict[str, Any]:
        """Drop open file handles, which cannot cross a process boundary."""
        state = dict(self.__dict__)
        state["_readers"] = {}
        state["_reader_pid"] = None
        return state


@dataclass(frozen=True, slots=True)
class WorkerInit:
    """Seeds one loader worker from the run's seed.

    A class rather than a closure because a spawned worker is handed this by pickle, and
    a closure does not pickle.

    Attributes:
        seed: Root seed of the run.
    """

    seed: int

    def __call__(self, worker_id: int) -> None:
        """Seed this worker's generators.

        Args:
            worker_id: Position of the worker within the loader.
        """
        state = seed_sequence(self.seed, f"{LOADER_SEED_STREAM}.{worker_id}")
        torch.manual_seed(int(state.generate_state(2, dtype=np.uint64)[0]) & _MAX_UINT64)


def make_worker_init(seed: int) -> WorkerInit:
    """Build a loader `worker_init_fn` that seeds each worker from the run's seed.

    The dataset itself draws nothing, so this exists for anything a worker calls that
    does, and to make sure two workers never share a stream.

    Args:
        seed: Root seed of the run.

    Returns:
        Something to pass as `worker_init_fn`.
    """
    return WorkerInit(seed)


def _tensor(array: np.ndarray[Any, np.dtype[np.float64]]) -> torch.Tensor:
    """Copy an array read from a shard into a tensor.

    Single precision, because that is what a model on this machine trains in. The stored
    data stays double precision, so the reference trajectory a rollout is scored against
    never inherits the model's precision.
    """
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
