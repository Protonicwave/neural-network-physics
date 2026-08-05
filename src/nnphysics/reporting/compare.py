"""Comparing runs.

A single run cannot support a claim of improvement. Two can, and only if the comparison
says which way each number is supposed to move: an error that halves and a horizon that
halves are opposite news, and a table of signed differences with no direction attached is
a table a reader can read either way.

So every delta carries the direction its scalar declares, and a scalar that declares no
direction is reported without a verdict rather than guessed at. Regressions above a
threshold are flagged, because the point of a comparison is to be told, not to notice.

Only scalars both runs carry are compared. A metric added between two runs has nothing to
be compared against, and inventing a baseline for it would manufacture an improvement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from nnphysics.core.errors import ValidationError
from nnphysics.reporting.explain import TIMING_METRIC, Direction, explain

if TYPE_CHECKING:
    from nnphysics.evals.result import InvariantRecord, SuiteResult
    from nnphysics.reporting.explain import Explanation
    from nnphysics.reporting.record import RunRecord

__all__ = [
    "DEFAULT_THRESHOLD",
    "Comparison",
    "Delta",
    "Verdict",
    "compare_records",
    "compare_series",
]

DEFAULT_THRESHOLD = 0.05
"""Relative change below which two numbers are called the same. Five per cent: smaller
than any improvement worth reporting, larger than the noise between two runs of the same
configuration on a machine that is also doing something else."""

_MINIMUM_RUNS = 2
"""A comparison of one run against nothing is a report, not a comparison."""


class Verdict(StrEnum):
    """What a comparison makes of one change."""

    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    UNDIRECTED = "undirected"
    """The scalar declares no direction of improvement, so the change is reported and not
    judged."""


@dataclass(frozen=True, slots=True)
class Delta:
    """One scalar in two runs.

    Attributes:
        predictor: Predictor the scalar belongs to.
        split: Split it was measured on.
        metric: Metric that produced it.
        key: Scalar name within that metric.
        baseline: Value in the run being compared against.
        candidate: Value in the run being compared.
        change: Candidate minus baseline.
        relative: Change as a fraction of the size of the baseline, infinite when the
            baseline is zero and the candidate is not.
        direction: Which way is an improvement.
        verdict: What that makes of the change.
    """

    predictor: str
    split: str
    metric: str
    key: str
    baseline: float
    candidate: float
    change: float
    relative: float
    direction: Direction
    verdict: Verdict

    @property
    def scalar(self) -> str:
        """The scalar's full name, `metric.scalar`."""
        return f"{self.metric}.{self.key}"

    @property
    def regressed(self) -> bool:
        """Whether this change is a regression."""
        return self.verdict is Verdict.REGRESSED


@dataclass(frozen=True, slots=True)
class Comparison:
    """Two runs, scalar by scalar.

    Attributes:
        baseline: Run identifier of the run compared against.
        candidate: Run identifier of the run compared.
        threshold: Relative change below which a scalar was called unchanged.
        deltas: One per scalar both runs carry, ordered by split, predictor and scalar.
    """

    baseline: str
    candidate: str
    threshold: float
    deltas: tuple[Delta, ...]

    def of(self, verdict: Verdict) -> tuple[Delta, ...]:
        """Every delta with one verdict.

        Args:
            verdict: The verdict to select.

        Returns:
            The deltas, in order.
        """
        return tuple(delta for delta in self.deltas if delta.verdict is verdict)

    @property
    def regressions(self) -> tuple[Delta, ...]:
        """Every regression, worst relative change first."""
        return tuple(sorted(self.of(Verdict.REGRESSED), key=lambda delta: -abs(delta.relative)))

    @property
    def summary(self) -> dict[Verdict, int]:
        """How many scalars fell into each verdict."""
        return {verdict: len(self.of(verdict)) for verdict in Verdict}


def compare_records(
    baseline: RunRecord, candidate: RunRecord, *, threshold: float = DEFAULT_THRESHOLD
) -> Comparison:
    """Compare two runs, scalar by scalar.

    Args:
        baseline: The run compared against.
        candidate: The run compared.
        threshold: Relative change below which a scalar is called unchanged.

    Returns:
        The comparison. Only predictors, splits and scalars both runs carry appear.

    Raises:
        ValidationError: If the threshold is negative, or the two runs evaluated
            different systems, which would make every delta meaningless.
        UnknownNameError: If a scalar has no explanation, so no direction is known for it.
    """
    if threshold < 0.0:
        raise ValidationError(f"a comparison threshold cannot be negative, got {threshold}")
    if baseline.system != candidate.system:
        raise ValidationError(
            f"cannot compare a run on {baseline.system} with a run on {candidate.system}"
        )
    declared = _declared(candidate.evaluation)
    left = _scalars(baseline.evaluation)
    right = _scalars(candidate.evaluation)
    deltas = [
        _delta(key, left[key], right[key], declared, threshold)
        for key in sorted(set(left) & set(right))
    ]
    return Comparison(
        baseline=baseline.run_id,
        candidate=candidate.run_id,
        threshold=threshold,
        deltas=tuple(deltas),
    )


def compare_series(
    records: Sequence[RunRecord], *, threshold: float = DEFAULT_THRESHOLD
) -> tuple[Comparison, ...]:
    """Compare several runs against the first of them.

    Args:
        records: Two or more run records. The first is the baseline.
        threshold: Relative change below which a scalar is called unchanged.

    Returns:
        One comparison per run after the first, in the order given.

    Raises:
        ValidationError: If fewer than two records were given.
    """
    if len(records) < _MINIMUM_RUNS:
        raise ValidationError(f"a comparison needs at least two runs, got {len(records)}")
    return tuple(
        compare_records(records[0], candidate, threshold=threshold) for candidate in records[1:]
    )


def _delta(
    key: tuple[str, str, str, str],
    baseline: float,
    candidate: float,
    declared: Mapping[str, InvariantRecord],
    threshold: float,
) -> Delta:
    """One scalar's change, and what to make of it."""
    split, predictor, metric, scalar = key
    explanation = explain(metric, scalar, declared)
    left = _effective(baseline, explanation)
    right = _effective(candidate, explanation)
    change = 0.0 if left == right else right - left
    relative = _relative(left, change)
    return Delta(
        predictor=predictor,
        split=split,
        metric=metric,
        key=scalar,
        baseline=baseline,
        candidate=candidate,
        change=change,
        relative=relative,
        direction=explanation.direction,
        verdict=_verdict(change, relative, explanation.direction, threshold),
    )


def _effective(value: float, explanation: Explanation) -> float:
    """The value a comparison should use.

    A sentinel says the thing never happened, which for a horizon is the best possible
    outcome and not a small number. Substituting an infinity of the right sign is what
    keeps a predictor that never crosses a threshold from being called a regression
    against one that crosses it late.
    """
    if explanation.sentinel is None or value != explanation.sentinel:
        return value
    return -math.inf if explanation.direction is Direction.LOWER else math.inf


def _relative(baseline: float, change: float) -> float:
    """Change as a fraction of the size of the baseline."""
    if change == 0.0:
        return 0.0
    if not math.isfinite(baseline) or baseline == 0.0:
        return math.inf if change > 0.0 else -math.inf
    return change / abs(baseline)


def _verdict(change: float, relative: float, direction: Direction, threshold: float) -> Verdict:
    """Improved, regressed, unchanged or not for this function to say."""
    if direction is Direction.NEUTRAL:
        return Verdict.UNDIRECTED
    if change == 0.0 or abs(relative) <= threshold:
        return Verdict.UNCHANGED
    improved = change < 0.0 if direction is Direction.LOWER else change > 0.0
    return Verdict.IMPROVED if improved else Verdict.REGRESSED


def _scalars(result: SuiteResult) -> dict[tuple[str, str, str, str], float]:
    """Every scalar of a result, keyed by split, predictor, metric and scalar name."""
    values: dict[tuple[str, str, str, str], float] = {}
    for entry in result.results:
        for record in entry.metrics:
            for key, value in record.scalars.items():
                values[(entry.split, entry.predictor, record.name, key)] = value
        values[(entry.split, entry.predictor, TIMING_METRIC, "seconds_per_step")] = (
            entry.seconds_per_step
        )
    return values


def _declared(result: SuiteResult) -> dict[str, InvariantRecord]:
    """Every declared invariant by name, taking the first declaration of each."""
    declared: dict[str, InvariantRecord] = {}
    for records in result.invariants.values():
        for record in records:
            declared.setdefault(record.name, record)
    return declared
