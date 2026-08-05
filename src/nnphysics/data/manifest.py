"""The provenance record that makes a dataset readable and re-derivable.

Everything needed to regenerate the data, to know which regime a trajectory came from
and to know which split it belongs to lives here rather than in the shard files. Shards
hold numbers and nothing else, which keeps their bytes a function of the data alone.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nnphysics.core.errors import ConfigurationError
from nnphysics.core.errors import ValidationError as NNPValidationError

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "Manifest",
    "RegimeRecord",
    "RegimeRole",
    "ShardRecord",
    "Split",
    "SplitRecord",
    "SpecRecord",
    "TrajectoryRecord",
    "read_manifest",
    "write_manifest",
]

MANIFEST_SCHEMA_VERSION = 1
"""Bumped when the shape of this record changes incompatibly."""


class Split(StrEnum):
    """The four disjoint sets a trajectory can belong to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HELD_OUT = "held_out"
    """Drawn from a regime never seen in training or model selection."""


class RegimeRole(StrEnum):
    """Why a regime was generated."""

    TRAIN = "train"
    """Contributes to the training, validation and test splits."""

    HELD_OUT = "held_out"
    """Contributes to the held out split only."""


class Record(BaseModel):
    """Base for every manifest record: immutable and intolerant of unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SpecRecord(Record):
    """The two time intervals and the length of a trajectory.

    Attributes:
        n_steps: States recorded per trajectory, the initial one included.
        dt: Time between recorded states.
        substeps: Solver steps per recorded interval.
        solver_dt: Step the solver took, recorded rather than inferred so that a reader
            never has to divide.
    """

    n_steps: int = Field(ge=2)
    dt: float = Field(gt=0.0)
    substeps: int = Field(ge=1)
    solver_dt: float = Field(gt=0.0)


class RegimeRecord(Record):
    """One regime and what it contributed.

    Attributes:
        name: Regime name.
        parameters: Parameter values the trajectories were drawn with.
        role: Whether the regime is trained on or held out.
        n_trajectories: Trajectories generated from it.
    """

    name: str = Field(min_length=1)
    parameters: dict[str, float]
    role: RegimeRole
    n_trajectories: int = Field(ge=1)


class TrajectoryRecord(Record):
    """Where one trajectory sits and how it was seeded.

    Attributes:
        id: Trajectory identifier.
        regime: Regime it was drawn from.
        index: Position within that regime, which fixes its seed stream.
        shard: File name of the shard holding it.
        row: Row of that shard's stacked arrays.
    """

    id: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    index: int = Field(ge=0)
    shard: str = Field(min_length=1)
    row: int = Field(ge=0)


class ShardRecord(Record):
    """One shard file and the hash that proves it has not changed.

    Attributes:
        path: File name, relative to the dataset directory.
        regime: Regime the shard holds.
        n_trajectories: Rows in the shard.
        fields: Field name to the shape of one state of that field.
        content_hash: Hash of the file's bytes.
    """

    path: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    n_trajectories: int = Field(ge=1)
    fields: dict[str, tuple[int, ...]]
    content_hash: str = Field(min_length=1)


class SplitRecord(Record):
    """How trajectories were divided, and into what.

    Attributes:
        seed_stream: Stream the split permutation was drawn from.
        val_fraction: Share of each training regime used for validation.
        test_fraction: Share of each training regime used for testing.
        members: Split to the trajectory identifiers it holds, sorted.
    """

    seed_stream: str = Field(min_length=1)
    val_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    members: dict[Split, tuple[str, ...]]


class Manifest(Record):
    """Everything about a dataset except the numbers themselves.

    Attributes:
        schema_version: Version of this record's shape.
        code_version: Version of the package that generated the data.
        dataset_id: Hash of the configuration the data was generated from.
        system: System name.
        system_parameters: System level parameters it was built with.
        seed: Root seed of the run.
        seed_stream_template: How each trajectory's seed stream was derived.
        spec: The trajectory specification.
        regimes: Every regime generated, in generation order.
        shards: Every shard written, in generation order.
        trajectories: Every trajectory, in generation order.
        splits: How the trajectories were divided.
        dataset_hash: Hash over the shard hashes, in shard order.
    """

    schema_version: int = MANIFEST_SCHEMA_VERSION
    code_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    system: str = Field(min_length=1)
    system_parameters: dict[str, float | int | bool | str]
    seed: int = Field(ge=0)
    seed_stream_template: str = Field(min_length=1)
    spec: SpecRecord
    regimes: tuple[RegimeRecord, ...] = Field(min_length=1)
    shards: tuple[ShardRecord, ...] = Field(min_length=1)
    trajectories: tuple[TrajectoryRecord, ...] = Field(min_length=1)
    splits: SplitRecord
    dataset_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> Self:
        identifiers = [record.id for record in self.trajectories]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("the manifest lists a trajectory identifier more than once")
        known = set(identifiers)
        shards = {record.path for record in self.shards}
        for record in self.trajectories:
            if record.shard not in shards:
                raise ValueError(f"trajectory {record.id} names an unlisted shard {record.shard}")
        placed = [name for members in self.splits.members.values() for name in members]
        if len(set(placed)) != len(placed):
            raise ValueError("a trajectory appears in more than one split")
        if unknown := sorted(set(placed) - known):
            raise ValueError(f"splits name trajectories the manifest does not list: {unknown}")
        if unplaced := sorted(known - set(placed)):
            raise ValueError(f"trajectories belong to no split: {unplaced}")
        return self

    def split(self, split: Split) -> tuple[str, ...]:
        """Trajectory identifiers in one split.

        Args:
            split: The split to read.

        Returns:
            The identifiers, sorted. Empty if the split holds nothing.
        """
        return self.splits.members.get(split, ())

    def trajectory(self, identifier: str) -> TrajectoryRecord:
        """Look up one trajectory.

        Args:
            identifier: The trajectory identifier.

        Returns:
            Its record.

        Raises:
            ValidationError: If the manifest does not list it.
        """
        for record in self.trajectories:
            if record.id == identifier:
                return record
        raise NNPValidationError(f"the manifest does not list a trajectory {identifier!r}")

    def regime(self, name: str) -> RegimeRecord:
        """Look up one regime.

        Args:
            name: The regime name.

        Returns:
            Its record.

        Raises:
            ValidationError: If the manifest does not list it.
        """
        for record in self.regimes:
            if record.name == name:
                return record
        raise NNPValidationError(f"the manifest does not list a regime {name!r}")

    def field_shapes(self, split: Split) -> dict[str, tuple[int, ...]]:
        """Shape of one state of each field, for the trajectories in a split.

        Args:
            split: The split to describe.

        Returns:
            Field name to shape.

        Raises:
            ValidationError: If the split is empty, or its trajectories disagree on a
                shape, which no batch could then be built from.
        """
        members = self.split(split)
        if not members:
            raise NNPValidationError(f"split {split.value!r} holds no trajectories")
        by_path = {record.path: record for record in self.shards}
        shards = {self.trajectory(name).shard for name in members}
        distinct = {
            tuple(sorted((field, tuple(shape)) for field, shape in by_path[path].fields.items()))
            for path in shards
        }
        if len(distinct) != 1:
            raise NNPValidationError(
                f"split {split.value!r} mixes trajectories of different shapes: {sorted(distinct)}"
            )
        return {name: tuple(shape) for name, shape in next(iter(distinct))}


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Write a manifest.

    Args:
        path: File to write.
        manifest: The manifest.
    """
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> Manifest:
    """Read and validate a manifest.

    Args:
        path: File to read.

    Returns:
        The manifest.

    Raises:
        ConfigurationError: If the file is missing, is not valid JSON, is of an
            unsupported schema version, or fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read manifest {path}: {error}") from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{path} is not valid JSON: {error}") from error
    if isinstance(raw, dict) and raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ConfigurationError(
            f"{path} has manifest schema version {raw.get('schema_version')!r}, "
            f"this build reads version {MANIFEST_SCHEMA_VERSION}"
        )
    try:
        return Manifest.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid manifest in {path}: {error}") from error
