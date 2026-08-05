"""The run record: one object holding everything a run is.

A result file says what came out. A run record says what was asked for, what produced it,
how long it took and where the rest of the artefacts are. That is the difference between
a number and a comparable number, and comparison is what this layer exists for.

It carries a schema version and its reader upgrades an older record rather than refusing
it. The record embeds an evaluation result, which carries a schema version of its own, so
the upgrade chain has two links. Today only the second one has a step in it: a record
written before the evaluation result gained per rollout scalars still reads, and reads as
a record that simply has no spread to plot.

Nothing here reaches for the time, the environment or the commit. They are arguments, so
that a test builds a record that is byte for byte the same on every machine and on every
day, which is what makes rendering testable at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ConfigurationError
from nnphysics.evals.result import SuiteResult, upgrade_result
from nnphysics.reporting.environment import EnvironmentRecord
from nnphysics.training.history import TrainingHistory

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "RECORD_SCHEMA_VERSION",
    "RunRecord",
    "read_record",
    "upgrade_record",
    "write_record",
]

RECORD_SCHEMA_VERSION = 1
"""Bumped when the shape of this record changes incompatibly."""


class RunRecord(BaseModel):
    """Everything one run is, in one serialisable object.

    Attributes:
        schema_version: Version of this record's shape.
        run_id: Identifier derived from the resolved configuration and the seed.
        name: Human readable label from the configuration.
        created: When the run finished, as an ISO 8601 string. Supplied by the caller so
            that a record is a value rather than a reading of the clock.
        code_version: Version of the package that produced it.
        commit: Full git commit of the working tree, empty when unknown.
        config: The resolved configuration, in full.
        environment: The machine and the libraries.
        timings: Named wall clock durations in seconds, for example how long the
            evaluation took.
        training: What training recorded about itself, epoch by epoch, or `None` for a
            run that only evaluated. Optional rather than versioned away, because a run
            that scores the reference solver and the broken baselines trains nothing and
            is still a run.
        evaluation: Every metric output.
        artefacts: Role to path, relative to the run directory. Relative so that a run
            directory can be moved and still describe itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = RECORD_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    created: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    commit: str = ""
    config: RunConfig
    environment: EnvironmentRecord
    timings: dict[str, float] = {}
    training: TrainingHistory | None = None
    evaluation: SuiteResult
    artefacts: dict[str, str] = {}

    @property
    def system(self) -> str:
        """Name of the system this run evaluated."""
        return self.evaluation.system

    @property
    def suite(self) -> str:
        """Name of the evaluation suite."""
        return self.evaluation.settings.name

    @property
    def splits(self) -> tuple[str, ...]:
        """Splits evaluated on, in the order they were run."""
        seen: list[str] = []
        for entry in self.evaluation.results:
            if entry.split not in seen:
                seen.append(entry.split)
        return tuple(seen)

    @property
    def predictors(self) -> tuple[str, ...]:
        """Predictors evaluated, in the order they were run."""
        seen: list[str] = []
        for entry in self.evaluation.results:
            if entry.predictor not in seen:
                seen.append(entry.predictor)
        return tuple(seen)


def _evaluation_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade the embedded evaluation result, which versions itself."""
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ConfigurationError("a run record must carry an evaluation object")
    return {**raw, "evaluation": upgrade_result(evaluation, source="the embedded evaluation")}


_UPGRADES: Mapping[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
"""One entry per record version that can still be read, keyed by the version it upgrades
from. Empty while there has only ever been one shape of run record."""


def upgrade_record(raw: dict[str, Any], *, source: str = "record") -> dict[str, Any]:
    """Bring a serialised run record up to the current schema version.

    Args:
        raw: The parsed JSON object.
        source: What to name in an error message.

    Returns:
        An object at the current schema version, its evaluation payload upgraded too.
        The input is not modified.

    Raises:
        ConfigurationError: If either version is missing, is not an integer, is newer
            than this build reads, or is older than any upgrade covers.
    """
    version = raw.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigurationError(f"{source} carries no integer record schema version")
    if version > RECORD_SCHEMA_VERSION:
        raise ConfigurationError(
            f"{source} has record schema version {version}, newer than the version "
            f"{RECORD_SCHEMA_VERSION} this build reads"
        )
    while version < RECORD_SCHEMA_VERSION:
        upgrade = _UPGRADES.get(version)
        if upgrade is None:
            raise ConfigurationError(
                f"{source} has record schema version {version}, which this build can no "
                f"longer upgrade"
            )
        raw = upgrade(raw)
        moved = raw.get("schema_version")
        if not isinstance(moved, int) or moved <= version:
            raise ConfigurationError(
                f"the upgrade from record schema version {version} left it at {moved!r}"
            )
        version = moved
    return _evaluation_payload(raw)


def write_record(path: Path, record: RunRecord) -> None:
    """Write a run record.

    Args:
        path: File to write. Its parent must exist.
        record: The record.
    """
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_record(path: Path) -> RunRecord:
    """Read a run record, upgrading it if it was written by an older build.

    Args:
        path: File to read.

    Returns:
        The record, at the current schema version.

    Raises:
        ConfigurationError: If the file is missing, is not valid JSON, is not an object,
            is of an unsupported schema version, or fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read run record {path}: {error}") from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path} must hold a JSON object, got {type(raw).__name__}")
    try:
        return RunRecord.model_validate(upgrade_record(raw, source=str(path)))
    except ValidationError as error:
        raise ConfigurationError(f"invalid run record in {path}: {error}") from error
