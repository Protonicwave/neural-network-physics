"""States kept from a rollout so that a report can show them next to the truth.

A number says how wrong a predictor is. It does not say what wrong looks like, and for a
surrogate that is often the more useful thing: a cluster that has quietly ejected a body,
a vortex sheet that has gone to noise. So a few states are kept from each rollout, at
spread out horizons, together with the true states at the same times.

They are captured by rolling the predictor out again rather than by holding every state
of the scoring pass in memory. That is affordable, one extra rollout per predictor and
split against the several the suite already runs, and it is exact: a predictor is seeded
from the run seed, its own name and the trajectory it starts from, so the second rollout
produces the same states as the first. Determinism is what makes the cheap option the
honest one.

Storage is a compressed `.npz` beside the result rather than more JSON. A single fluid
state is four thousand numbers, and a report that embedded them as text would be an order
of magnitude larger than the plots drawn from them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from nnphysics.core.errors import ConfigurationError, ValidationError
from nnphysics.evals.predictors import parse_spec
from nnphysics.evals.rollout import roll_out
from nnphysics.evals.runner import build_case_predictor

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.core.config import EvaluationConfig
    from nnphysics.core.protocols import System
    from nnphysics.core.types import FloatArray, Trajectory
    from nnphysics.evals.runner import EvaluationCase

__all__ = [
    "DEFAULT_HORIZONS",
    "Snapshot",
    "SnapshotSet",
    "capture_snapshots",
    "read_snapshots",
    "write_snapshots",
]

DEFAULT_HORIZONS = 4
"""States kept per rollout, the initial one included. Four fits a row of panels on a page
and still shows a beginning, a middle and an end."""

_INDEX_KEY = "index"
"""Name of the array holding the metadata, which travels with the arrays so that a file
describes itself."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A handful of predicted states and the true states at the same times.

    Attributes:
        predictor: Registered predictor name.
        split: Split the initial condition came from.
        trajectory: Identifier of the dataset trajectory.
        steps: Step index of each kept state within the rollout.
        times: Simulation time of each kept state.
        predicted: Field name to array of shape `(n_kept, *field_shape)`.
        reference: The same fields, from ground truth at the same times.
    """

    predictor: str
    split: str
    trajectory: str
    steps: tuple[int, ...]
    times: tuple[float, ...]
    predicted: Mapping[str, FloatArray]
    reference: Mapping[str, FloatArray]

    def __post_init__(self) -> None:
        if set(self.predicted) != set(self.reference):
            raise ValidationError(
                f"snapshot fields differ: predicted {sorted(self.predicted)}, "
                f"reference {sorted(self.reference)}"
            )
        if not self.steps:
            raise ValidationError("a snapshot must keep at least one state")

    @property
    def names(self) -> tuple[str, ...]:
        """Field names present, sorted."""
        return tuple(sorted(self.predicted))


@dataclass(frozen=True, slots=True)
class SnapshotSet:
    """Every snapshot of one evaluation.

    Attributes:
        snapshots: One per predictor and split, in the order they were captured.
    """

    snapshots: tuple[Snapshot, ...] = ()

    def __len__(self) -> int:
        return len(self.snapshots)

    def find(self, predictor: str, split: str) -> Snapshot | None:
        """Look up one snapshot.

        Args:
            predictor: Registered predictor name.
            split: Split name.

        Returns:
            The snapshot, or `None` if none was captured for that pair.
        """
        for snapshot in self.snapshots:
            if snapshot.predictor == predictor and snapshot.split == split:
                return snapshot
        return None


def capture_snapshots(  # noqa: PLR0913
    # The same independently varying arguments the runner takes, plus how many states to
    # keep. Grouping them would mean inventing a configuration object that exists only to
    # be passed here.
    system: System,
    cases: Sequence[EvaluationCase],
    specs: Sequence[str],
    config: EvaluationConfig,
    *,
    substeps: int,
    seed: int,
    horizons: int = DEFAULT_HORIZONS,
) -> SnapshotSet:
    """Roll each predictor out once more and keep a few states of each.

    The first case of the set is used rather than all of them. The purpose is qualitative,
    and one trajectory shown honestly says more than four shown too small to read.

    Args:
        system: The system, seen only through the protocol.
        cases: Initial conditions from one split. Only the first is used.
        specs: Predictor specifications, as written.
        config: The suite, for the horizon and the divergence factor.
        substeps: Solver steps per stored interval.
        seed: Root seed of the run.
        horizons: States to keep per rollout, the initial one included.

    Returns:
        One snapshot per predictor, in the order the specifications were given. A
        predictor that failed on its first step contributes only that state.

    Raises:
        ValidationError: If no cases were given or fewer than one horizon was asked for.
        UnknownNameError: If a predictor is not registered.
    """
    if not cases:
        raise ValidationError("cannot capture snapshots without an initial condition")
    if horizons < 1:
        raise ValidationError(f"a snapshot must keep at least one state, asked for {horizons}")

    case = cases[0]
    steps = min(config.rollout_steps, case.steps)
    captured: list[Snapshot] = []
    for text in specs:
        spec = parse_spec(text)
        predictor = build_case_predictor(system, case, spec, substeps=substeps, seed=seed)
        result = roll_out(
            predictor,
            case.initial,
            steps,
            divergence_factor=config.divergence_factor,
        )
        kept = _spread(len(result.trajectory), horizons)
        captured.append(
            Snapshot(
                predictor=spec.name,
                split=case.split.value,
                trajectory=case.trajectory_id,
                steps=kept,
                times=tuple(float(result.trajectory.times[step]) for step in kept),
                predicted=_take(result.trajectory, kept),
                reference=_take(case.reference, kept),
            )
        )
    return SnapshotSet(tuple(captured))


def write_snapshots(path: Path, snapshots: SnapshotSet) -> None:
    """Write a snapshot set to a compressed archive.

    Args:
        path: File to write. Its parent must exist.
        snapshots: The snapshots. An empty set still writes a file, so that a reader
            never has to distinguish absent from empty.
    """
    arrays: dict[str, FloatArray] = {}
    index: list[dict[str, Any]] = []
    for position, snapshot in enumerate(snapshots.snapshots):
        index.append(
            {
                "predictor": snapshot.predictor,
                "split": snapshot.split,
                "trajectory": snapshot.trajectory,
                "steps": list(snapshot.steps),
                "times": list(snapshot.times),
                "fields": list(snapshot.names),
            }
        )
        for name in snapshot.names:
            arrays[f"{position}.predicted.{name}"] = snapshot.predicted[name]
            arrays[f"{position}.reference.{name}"] = snapshot.reference[name]
    encoded = json.dumps(index, sort_keys=True, separators=(",", ":"))
    payload: dict[str, Any] = {_INDEX_KEY: np.asarray(encoded), **arrays}
    # Called through a name rather than directly: the stub declares a keyword of its own
    # alongside the arrays, so a mapping of array names cannot be spread into it.
    write: Callable[..., None] = np.savez_compressed
    write(path, **payload)


def read_snapshots(path: Path) -> SnapshotSet:
    """Read a snapshot set.

    Args:
        path: File to read.

    Returns:
        The snapshots, in the order they were written.

    Raises:
        ConfigurationError: If the file is missing, is not a snapshot archive, or names an
            array it does not carry.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _INDEX_KEY not in archive:
                raise ConfigurationError(f"{path} carries no snapshot index")
            index = json.loads(str(archive[_INDEX_KEY]))
            return SnapshotSet(
                tuple(_restore(archive, position, entry) for position, entry in enumerate(index))
            )
    except OSError as error:
        raise ConfigurationError(f"cannot read snapshots {path}: {error}") from error
    except (ValueError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"{path} is not a readable snapshot archive: {error}") from error


def _restore(archive: Mapping[str, FloatArray], position: int, entry: dict[str, Any]) -> Snapshot:
    """Rebuild one snapshot from the archive."""
    fields: list[str] = entry["fields"]
    return Snapshot(
        predictor=entry["predictor"],
        split=entry["split"],
        trajectory=entry["trajectory"],
        steps=tuple(int(step) for step in entry["steps"]),
        times=tuple(float(time) for time in entry["times"]),
        predicted={name: archive[f"{position}.predicted.{name}"] for name in fields},
        reference={name: archive[f"{position}.reference.{name}"] for name in fields},
    )


def _spread(length: int, horizons: int) -> tuple[int, ...]:
    """Step indices spread evenly over a trajectory, the first and last included."""
    if length <= horizons:
        return tuple(range(length))
    chosen = np.rint(np.linspace(0, length - 1, horizons))
    return tuple(sorted({int(value) for value in chosen}))


def _take(trajectory: Trajectory, steps: Sequence[int]) -> dict[str, FloatArray]:
    """Stack the chosen states of a trajectory, one array per field."""
    index = np.asarray(steps, dtype=np.intp)
    return {name: np.asarray(array)[index] for name, array in trajectory.fields.items()}
