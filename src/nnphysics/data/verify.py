"""Checking that a dataset is what its manifest says it is.

Two independent checks, because they catch different failures. Hashing every shard
catches a file that has changed since it was written, whether from a bad disk, a partial
copy or an edit. Re-deriving a sample of trajectories from their recorded seeds catches
the case where the file is intact but the data never matched the configuration, and is
also the only direct evidence that the dataset is reproducible.

Failures are collected rather than raised, so one run reports everything that is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import NNPhysicsError, ValidationError
from nnphysics.core.seeding import numpy_generator
from nnphysics.data.generation import find_regime, generate_trajectory
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import read_manifest
from nnphysics.data.spec import TrajectorySpec
from nnphysics.data.store import ShardReader, content_hash
from nnphysics.systems import build_system

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from nnphysics.core.types import FloatArray
    from nnphysics.data.manifest import Manifest

__all__ = ["VERIFY_SEED_STREAM", "VerificationReport", "verify_dataset"]

VERIFY_SEED_STREAM = "data.verify"
"""Stream the sample of trajectories is drawn from, so two runs check the same ones."""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """What a verification run found.

    Attributes:
        shards_checked: Shards whose bytes were hashed.
        trajectories_checked: Trajectories re-derived from their seeds.
        failures: One line per problem, empty if the dataset is sound.
    """

    shards_checked: int = 0
    trajectories_checked: int = 0
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether the dataset passed every check."""
        return not self.failures


def verify_dataset(directory: Path, *, sample: int = 4) -> VerificationReport:
    """Verify a dataset against its manifest.

    Args:
        directory: The dataset directory.
        sample: How many trajectories to re-derive from their seeds. Zero checks hashes
            only, which is fast enough to run on every dataset before training.

    Returns:
        The report.

    Raises:
        ConfigurationError: If the manifest is missing or unreadable, which is not a
            finding about the data but an inability to check it at all.
        ValidationError: If the sample size is negative.
    """
    if sample < 0:
        raise ValidationError(f"sample size must not be negative, got {sample}")
    manifest = read_manifest(directory / MANIFEST_NAME)

    failures: list[str] = []
    checked_shards = 0
    for record in manifest.shards:
        path = directory / record.path
        if not path.is_file():
            failures.append(f"shard {record.path} is missing")
            continue
        try:
            digest = content_hash(path)
        except NNPhysicsError as error:
            failures.append(f"shard {record.path} cannot be read: {error}")
            continue
        checked_shards += 1
        if digest != record.content_hash:
            failures.append(
                f"shard {record.path} hashes to {digest}, the manifest records "
                f"{record.content_hash}"
            )

    checked_trajectories = 0
    if sample > 0 and not failures:
        checked_trajectories, reproduction_failures = _check_reproduction(
            directory, manifest, sample
        )
        failures.extend(reproduction_failures)

    return VerificationReport(
        shards_checked=checked_shards,
        trajectories_checked=checked_trajectories,
        failures=tuple(failures),
    )


def _check_reproduction(directory: Path, manifest: Manifest, sample: int) -> tuple[int, list[str]]:
    """Re-derive a sample of trajectories and compare them with what was stored.

    Equality is exact. The same seed, the same code and the same machine produce the same
    floating point numbers, so anything else is a real difference rather than noise.
    """
    system = build_system(manifest.system, dict(manifest.system_parameters))
    spec = TrajectorySpec(
        n_steps=manifest.spec.n_steps, dt=manifest.spec.dt, substeps=manifest.spec.substeps
    )
    rng = numpy_generator(manifest.seed, VERIFY_SEED_STREAM)
    count = min(sample, len(manifest.trajectories))
    chosen = sorted(rng.choice(len(manifest.trajectories), size=count, replace=False).tolist())

    failures: list[str] = []
    checked = 0
    readers: dict[str, ShardReader] = {}
    try:
        for position in chosen:
            record = manifest.trajectories[position]
            if record.shard not in readers:
                readers[record.shard] = ShardReader(directory / record.shard)
            try:
                stored = readers[record.shard].trajectory(record.row)
                expected = generate_trajectory(
                    system,
                    find_regime(system, record.regime),
                    spec,
                    manifest.seed,
                    record.index,
                )
            except NNPhysicsError as error:
                failures.append(f"trajectory {record.id} could not be re-derived: {error}")
                continue
            checked += 1
            failures.extend(_compare(record.id, stored.fields, expected.fields, stored.times,
                                     expected.times))
    finally:
        for reader in readers.values():
            reader.close()
    return checked, failures


def _compare(
    identifier: str,
    stored: Mapping[str, FloatArray],
    expected: Mapping[str, FloatArray],
    stored_times: FloatArray,
    expected_times: FloatArray,
) -> list[str]:
    """Compare one re-derived trajectory with the stored one."""
    failures: list[str] = []
    if set(stored) != set(expected):
        return [
            f"trajectory {identifier} stores fields {sorted(stored)}, the system declares "
            f"{sorted(expected)}"
        ]
    if not np.array_equal(stored_times, expected_times):
        failures.append(f"trajectory {identifier} stores times that were not re-derived")
    for name in sorted(stored):
        if not np.array_equal(stored[name], expected[name]):
            difference = float(np.max(np.abs(stored[name] - expected[name])))
            failures.append(
                f"trajectory {identifier} field {name!r} differs from its re-derivation by "
                f"up to {difference:.3e}"
            )
    return failures
