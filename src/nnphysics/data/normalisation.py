"""Statistics fitted to the training split, and their application at load time.

Fitting and applying are separate operations and separate types. Anything that can apply
statistics can be handed statistics it did not fit, which is what stops the validation
and test data from ever reaching the fit. The fit reads the training split only, and a
test asserts that changing the other splits leaves the numbers untouched.

One mean and one scale per field, reduced over every axis rather than per component. A
per component statistic would subtract a different number from `x` than from `y`, and a
system whose dynamics are rotation equivariant would stop being so once normalised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nnphysics.core.errors import ConfigurationError, ValidationError
from nnphysics.data.manifest import Split
from nnphysics.data.store import ShardReader

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nnphysics.core.types import FloatArray
    from nnphysics.data.manifest import Manifest

__all__ = [
    "DEGENERATE_ATOL",
    "FieldStats",
    "Normalisation",
    "fit_normalisation",
    "read_normalisation",
    "write_normalisation",
]

DEGENERATE_ATOL = 1e-12
"""Below this a field is constant across the training split and has no scale to divide
by. Mass is constant in every N-body trajectory, so this is the ordinary case and not an
error, but it is recorded rather than hidden."""

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FieldStats:
    """Centre and scale of one field over the training split.

    Attributes:
        mean: Mean over every element of every training state.
        std: Standard deviation over the same, which may be zero.
        count: Elements the statistics were computed from.
    """

    mean: float
    std: float
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValidationError(f"field statistics need at least one element, got {self.count}")
        if self.std < 0.0 or not np.isfinite([self.mean, self.std]).all():
            raise ValidationError(f"field statistics are not usable: mean {self.mean}, std "
                                  f"{self.std}")

    @property
    def degenerate(self) -> bool:
        """Whether the field is constant, so that it has no scale of its own."""
        return self.std <= DEGENERATE_ATOL

    @property
    def scale(self) -> float:
        """Divisor used when normalising. One for a constant field."""
        return 1.0 if self.degenerate else self.std


@dataclass(frozen=True, slots=True)
class Normalisation:
    """Statistics for every field of a system, and the transform they define.

    Attributes:
        stats: Field name to its statistics.
    """

    stats: Mapping[str, FieldStats]

    def __post_init__(self) -> None:
        if not self.stats:
            raise ValidationError("normalisation must cover at least one field")

    @property
    def names(self) -> tuple[str, ...]:
        """Fields covered, sorted."""
        return tuple(sorted(self.stats))

    def apply(self, fields: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        """Centre and scale every field.

        Args:
            fields: Field name to array.

        Returns:
            The normalised arrays.

        Raises:
            ValidationError: If a field has no statistics.
        """
        self._check(fields)
        return {
            name: (array - self.stats[name].mean) / self.stats[name].scale
            for name, array in fields.items()
        }

    def invert(self, fields: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        """Undo `apply`.

        Args:
            fields: Field name to normalised array.

        Returns:
            The arrays in physical units.

        Raises:
            ValidationError: If a field has no statistics.
        """
        self._check(fields)
        return {
            name: array * self.stats[name].scale + self.stats[name].mean
            for name, array in fields.items()
        }

    def _check(self, fields: Mapping[str, FloatArray]) -> None:
        """Raise if a field has no statistics."""
        if missing := sorted(set(fields) - set(self.stats)):
            raise ValidationError(
                f"no normalisation statistics for fields {missing}, have {list(self.names)}"
            )


def fit_normalisation(
    directory: Path, manifest: Manifest, *, split: Split = Split.TRAIN
) -> Normalisation:
    """Fit statistics from one split, the training split by default.

    Sums are accumulated in a fixed order over sorted identifiers, so the numbers do not
    depend on how the data was sharded.

    Args:
        directory: The dataset directory.
        manifest: Its manifest.
        split: Split to fit from. Anything but the training split makes a leaking
            dataset, and exists here only so a test can demonstrate the difference.

    Returns:
        The statistics.

    Raises:
        ValidationError: If the split is empty or a shard cannot be read.
    """
    members = manifest.split(split)
    if not members:
        raise ValidationError(f"cannot fit normalisation from empty split {split.value!r}")

    total: dict[str, float] = {}
    total_square: dict[str, float] = {}
    count: dict[str, int] = {}
    by_shard: dict[str, list[int]] = {}
    for identifier in sorted(members):
        record = manifest.trajectory(identifier)
        by_shard.setdefault(record.shard, []).append(record.row)

    for shard in sorted(by_shard):
        with ShardReader(directory / shard) as reader:
            for row in sorted(by_shard[shard]):
                fields, _ = reader.window(row, 0, reader.n_steps)
                for name, array in fields.items():
                    total[name] = total.get(name, 0.0) + float(np.sum(array))
                    total_square[name] = total_square.get(name, 0.0) + float(np.sum(array**2))
                    count[name] = count.get(name, 0) + array.size

    stats: dict[str, FieldStats] = {}
    for name in sorted(total):
        mean = total[name] / count[name]
        # Clamped at zero: the two sums are large and nearly equal for a constant field,
        # so cancellation can put the variance a rounding error below it.
        variance = max(total_square[name] / count[name] - mean**2, 0.0)
        stats[name] = FieldStats(mean=mean, std=float(np.sqrt(variance)), count=count[name])
    return Normalisation(stats)


def write_normalisation(path: Path, normalisation: Normalisation) -> None:
    """Write statistics as an artefact.

    Args:
        path: File to write.
        normalisation: The statistics.
    """
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "fields": {
            name: {
                "mean": stats.mean,
                "std": stats.std,
                "count": stats.count,
                "degenerate": stats.degenerate,
            }
            for name, stats in sorted(normalisation.stats.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_normalisation(path: Path) -> Normalisation:
    """Read statistics back.

    Args:
        path: File to read.

    Returns:
        The statistics.

    Raises:
        ConfigurationError: If the file is missing, malformed or of another version.
    """
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"cannot read normalisation {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
        raise ConfigurationError(
            f"{path} is not a version {_SCHEMA_VERSION} normalisation artefact"
        )
    fields = raw.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ConfigurationError(f"{path} carries no field statistics")
    try:
        stats = {
            str(name): FieldStats(
                mean=float(entry["mean"]), std=float(entry["std"]), count=int(entry["count"])
            )
            for name, entry in fields.items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"{path} has malformed field statistics: {error}") from error
    return Normalisation(stats)
