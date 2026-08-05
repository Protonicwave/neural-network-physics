"""What an evaluation writes down.

A result file is the only thing later phases see of a run, so it records what was asked
for as well as what came out: the suite, the predictor specifications as they were
written, which trajectories were rolled out and whether each rollout finished. A number
without the settings that produced it cannot be compared with a number from another run,
and comparison is the entire point of phase 06.

Plot data is kept. Rendering it is not this layer's business, but discarding it would
force a re-run to draw a curve.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nnphysics.core.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "InvariantRecord",
    "MetricRecord",
    "PredictorResult",
    "RolloutRecord",
    "SuiteResult",
    "SuiteSettings",
    "read_result",
    "write_result",
]

RESULT_SCHEMA_VERSION = 1
"""Bumped when the shape of this record changes incompatibly."""


class Record(BaseModel):
    """Base for every result record: immutable and intolerant of unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SuiteSettings(Record):
    """The suite as it was configured, so a result can be reproduced or compared.

    Attributes:
        name: Suite name.
        metrics: Metrics run, in order.
        rollout_steps: Steps each predictor was rolled out for.
        n_initial_conditions: Initial conditions drawn from each split.
        error_thresholds: Normalised error levels horizons were reported for.
        symmetry_steps: Steps the equivariance test rolled out for.
        distribution_window: Fraction of the rollout the distributions were taken over.
        divergence_factor: How far past its initial scale a state could go.
    """

    name: str = Field(min_length=1)
    metrics: tuple[str, ...] = Field(min_length=1)
    rollout_steps: int = Field(ge=1)
    n_initial_conditions: int = Field(ge=1)
    error_thresholds: tuple[float, ...]
    symmetry_steps: int = Field(ge=1)
    distribution_window: float = Field(gt=0.0, le=1.0)
    divergence_factor: float


class InvariantRecord(Record):
    """One invariant as the system declared it.

    Recorded because a drift number means nothing without knowing whether the quantity
    was supposed to hold still, and how tightly.

    Attributes:
        name: Invariant name.
        conservation: Whether it is exact, approximate or decaying.
        rtol: The tolerance it declared.
        dimension: Its physical dimension, as a symbol.
    """

    name: str = Field(min_length=1)
    conservation: str = Field(min_length=1)
    rtol: float = Field(gt=0.0)
    dimension: str = Field(min_length=1)


class MetricRecord(Record):
    """What one metric returned, aggregated over the initial conditions.

    Attributes:
        name: Metric name.
        scalars: Named scalars, the mean over rollouts.
        series: Named curves, the mean over rollouts, kept for plotting.
    """

    name: str = Field(min_length=1)
    scalars: dict[str, float]
    series: dict[str, tuple[float, ...]] = {}


class RolloutRecord(Record):
    """One rollout of one predictor from one initial condition.

    Attributes:
        trajectory: Identifier of the dataset trajectory it started from.
        regime: Regime that trajectory was drawn from.
        split: Split it belongs to.
        steps_requested: Steps asked for.
        steps_completed: Steps taken before the rollout ended.
        stop_reason: Why it ended.
        detail: What the predictor said, when it stopped by raising.
        seconds: Wall clock spent stepping.
    """

    trajectory: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    split: str = Field(min_length=1)
    steps_requested: int = Field(ge=1)
    steps_completed: int = Field(ge=0)
    stop_reason: str = Field(min_length=1)
    detail: str = ""
    seconds: float = Field(ge=0.0)


class PredictorResult(Record):
    """Everything one predictor scored on one set of initial conditions.

    Attributes:
        predictor: Registered predictor name.
        spec: The specification as it was written.
        split: Split the initial conditions came from.
        regimes: Regimes represented, sorted.
        rollouts: One record per initial condition.
        metrics: One record per metric.
        seconds_per_step: Mean wall clock per step over every rollout, which is what a
            speedup claim is eventually measured from.
        completed: Whether every rollout ran to the requested horizon.
    """

    predictor: str = Field(min_length=1)
    spec: str = Field(min_length=1)
    split: str = Field(min_length=1)
    regimes: tuple[str, ...]
    rollouts: tuple[RolloutRecord, ...] = Field(min_length=1)
    metrics: tuple[MetricRecord, ...]
    seconds_per_step: float = Field(ge=0.0)
    completed: bool

    def scalar(self, metric: str, key: str) -> float:
        """Read one scalar out of one metric.

        Args:
            metric: Metric name.
            key: Scalar name within that metric.

        Returns:
            The value.

        Raises:
            KeyError: If the metric or the scalar is not present.
        """
        for record in self.metrics:
            if record.name == metric:
                return record.scalars[key]
        raise KeyError(f"this result carries no metric {metric!r}")


class SuiteResult(Record):
    """One evaluation: every predictor, on every split that was asked for.

    Attributes:
        schema_version: Version of this record's shape.
        code_version: Version of the package that produced it.
        run_id: Identifier of the run configuration.
        dataset_id: Identifier of the dataset evaluated on.
        system: System name.
        seed: Root seed of the run.
        settings: The suite as it was configured.
        invariants: The invariants the system declared, per regime.
        results: One entry per predictor and split.
        regime_gap: Held out value minus in distribution value, per predictor, metric and
            scalar. Empty if the dataset carries no held out split.
    """

    schema_version: int = RESULT_SCHEMA_VERSION
    code_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    system: str = Field(min_length=1)
    seed: int = Field(ge=0)
    settings: SuiteSettings
    invariants: dict[str, tuple[InvariantRecord, ...]]
    results: tuple[PredictorResult, ...] = Field(min_length=1)
    regime_gap: dict[str, float] = {}

    def result(self, predictor: str, split: str) -> PredictorResult:
        """Look up one predictor's result on one split.

        Args:
            predictor: Registered predictor name.
            split: Split name.

        Returns:
            The result.

        Raises:
            KeyError: If no such result was recorded.
        """
        for entry in self.results:
            if entry.predictor == predictor and entry.split == split:
                return entry
        raise KeyError(f"this suite carries no result for {predictor!r} on split {split!r}")


def write_result(path: Path, result: SuiteResult) -> None:
    """Write a suite result.

    Args:
        path: File to write. Its parent must exist.
        result: The result.
    """
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_result(path: Path) -> SuiteResult:
    """Read and validate a suite result.

    Args:
        path: File to read.

    Returns:
        The result.

    Raises:
        ConfigurationError: If the file is missing, is not valid JSON, is of an
            unsupported schema version, or fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read result {path}: {error}") from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{path} is not valid JSON: {error}") from error
    if isinstance(raw, dict) and raw.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ConfigurationError(
            f"{path} has result schema version {raw.get('schema_version')!r}, "
            f"this build reads version {RESULT_SCHEMA_VERSION}"
        )
    try:
        return SuiteResult.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid result in {path}: {error}") from error
