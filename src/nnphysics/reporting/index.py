"""The history of every run, without opening a file.

A runs directory accumulates. Being able to see what is in it, sorted, with the numbers
that decide whether a run was any good, is what stops the history from being a set of
hashes nobody remembers the meaning of.

A run holds one row per predictor and split, because a run is not one number. Filtering
to a single predictor is what turns the listing back into one row per run.

What a run is worth, meaning how long it stayed usable and what its answer was, is derived
by the page model rather than again here. The listing and the landing page are two views of
the same runs, and two definitions of a run's key numbers would disagree the first time
either changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nnphysics.reporting.explain import TIMING_METRIC
from nnphysics.reporting.layout import find_records
from nnphysics.reporting.page import summarise_run
from nnphysics.reporting.record import read_record

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.reporting.record import RunRecord

__all__ = ["KEY_SCALARS", "RunSummary", "index_runs", "summarise"]

KEY_SCALARS = (
    ("rollout_error", "error.final"),
    ("invariant_drift", "worst_violation"),
    ("symmetry_violation", "worst"),
    (TIMING_METRIC, "seconds_per_step"),
)
"""What a listing shows. Accuracy, physicality, symmetry and cost: the four questions a
run is asked, one number each."""


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One predictor of one run, as a listing shows it.

    Attributes:
        run_id: Identifier of the run.
        name: Human readable label from the configuration.
        created: When the run finished.
        system: System evaluated.
        suite: Evaluation suite.
        split: Split the numbers were measured on.
        predictor: Predictor they belong to.
        values: Key scalar to value, absent where the run has no such number.
        usable: Stored steps this predictor stayed usable for on this split, or `None`
            where there is nothing to report.
        verdict: The run's answer in one phrase, the same phrase the landing page prints.
        directory: Directory the run's artefacts live in.
    """

    run_id: str
    name: str
    created: str
    system: str
    suite: str
    split: str
    predictor: str
    values: dict[str, float]
    usable: float | None
    verdict: str
    directory: Path

    @property
    def label(self) -> str:
        """How a listing names the run."""
        return f"{self.name} ({self.run_id})"


def summarise(
    record: RunRecord,
    directory: Path,
    *,
    predictor: str | None = None,
    split: str | None = None,
) -> tuple[RunSummary, ...]:
    """Summarise one run.

    Args:
        record: The run record.
        directory: Directory the record was read from.
        predictor: Keep only this predictor, or `None` for all of them.
        split: Keep only this split, or `None` for all of them.

    Returns:
        One summary per predictor and split kept, in the order they were evaluated.
    """
    card = summarise_run(record, directory.name)
    summaries: list[RunSummary] = []
    for entry in record.evaluation.results:
        if predictor is not None and entry.predictor != predictor:
            continue
        if split is not None and entry.split != split:
            continue
        values: dict[str, float] = {}
        for metric, key in KEY_SCALARS:
            if metric == TIMING_METRIC:
                values[f"{metric}.{key}"] = entry.seconds_per_step
                continue
            try:
                values[f"{metric}.{key}"] = entry.scalar(metric, key)
            except KeyError:
                continue
        horizon = card.horizon(entry.predictor, entry.split)
        summaries.append(
            RunSummary(
                run_id=record.run_id,
                name=record.name,
                created=record.created,
                system=record.evaluation.system,
                suite=record.evaluation.settings.name,
                split=entry.split,
                predictor=entry.predictor,
                values=values,
                usable=horizon.steps if horizon is not None else None,
                verdict=card.verdict.phrase,
                directory=directory,
            )
        )
    return tuple(summaries)


def index_runs(
    root: Path, *, predictor: str | None = None, split: str | None = None
) -> tuple[RunSummary, ...]:
    """Summarise every run under a runs root.

    Args:
        root: The runs root.
        predictor: Keep only this predictor, or `None` for all of them.
        split: Keep only this split, or `None` for all of them.

    Returns:
        The summaries, oldest run first and ties broken by identifier, so that a listing
        is stable whatever order the filesystem returns directories in.

    Raises:
        ConfigurationError: If a record cannot be read. A listing that skipped an
            unreadable run would be a listing that quietly lost one.
    """
    summaries: list[RunSummary] = []
    for path in find_records(root):
        record = read_record(path)
        summaries.extend(summarise(record, path.parent, predictor=predictor, split=split))
    return tuple(_sorted(summaries))


def _sorted(summaries: Sequence[RunSummary]) -> list[RunSummary]:
    """Oldest first, then by identifier, split and predictor."""
    return sorted(
        summaries,
        key=lambda entry: (entry.created, entry.run_id, entry.split, entry.predictor),
    )
