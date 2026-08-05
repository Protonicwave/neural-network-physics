"""Which fields move and which only sit there.

A model should not be asked to predict a field that never changes. Mass is the example:
it differs from body to body and from trajectory to trajectory, so it is not constant in
the sense normalisation cares about, but it does not change along a trajectory, and a
model that predicted it would be spending capacity to reproduce its own input and would
be scored for succeeding.

The question is answered by reading the training split rather than by naming the field,
because naming it would put a fact about one system into a layer that must not know which
system it is looking at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.data.manifest import Split
from nnphysics.data.store import ShardReader

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.data.manifest import Manifest

__all__ = ["CONSTANCY_RTOL", "constant_fields"]

_MINIMUM_STATES = 2
"""One state cannot show a field moving, so a field is constant over it by default."""

CONSTANCY_RTOL = 1e-12
"""How far a field may move along a trajectory and still count as constant.

Near round off, because a field that is meant to be held fixed is written back unchanged
by the solver rather than recomputed. This is a check that a quantity is carried, not a
judgement about how slowly it varies.
"""


def constant_fields(
    directory: Path, manifest: Manifest, *, split: Split = Split.TRAIN
) -> tuple[str, ...]:
    """Fields that do not change along any trajectory of a split.

    Args:
        directory: The dataset directory.
        manifest: Its manifest.
        split: Split to read. The training split, because this decides what a model is
            built to predict and a model may not be shaped by data it is tested on.

    Returns:
        The field names, sorted. Empty if every field moves.

    Raises:
        ValidationError: If the split is empty.
    """
    members = manifest.split(split)
    if not members:
        raise ValidationError(f"cannot read field constancy from empty split {split.value!r}")

    by_shard: dict[str, list[int]] = {}
    for identifier in sorted(members):
        record = manifest.trajectory(identifier)
        by_shard.setdefault(record.shard, []).append(record.row)

    constant: set[str] | None = None
    for shard in sorted(by_shard):
        with ShardReader(directory / shard) as reader:
            for row in sorted(by_shard[shard]):
                fields, _ = reader.window(row, 0, reader.n_steps)
                still = {name for name, array in fields.items() if _is_constant(array)}
                constant = still if constant is None else constant & still
    return tuple(sorted(constant or ()))


def _is_constant(array: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> bool:
    """Whether a stacked field never moves away from its first state."""
    if array.shape[0] < _MINIMUM_STATES:
        return True
    scale = float(np.max(np.abs(array))) or 1.0
    return bool(np.max(np.abs(array - array[0])) <= CONSTANCY_RTOL * scale)
