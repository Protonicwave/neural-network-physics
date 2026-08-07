"""What every number in a report means, in one line of plain English.

The rule this module enforces is that a number without an explanation is not finished
work. Every scalar a metric can produce is described here, with its units and with the
direction that counts as an improvement, and a scalar that is not described raises rather
than being rendered bare. That is deliberate: a metric added later cannot reach a report
until someone has said what its numbers mean.

The vocabulary is fixed, the names are not. A scalar is either a fixed key, such as the
worst violation over every invariant, or a declared name followed by a fixed suffix, such
as `energy.violation` or `rotation.max`. The declared part comes from the system and this
module never has to know it, which is what keeps the explanations system agnostic while
still naming the right quantity in the right sentence.

These sentences are the first thing a reader of the repository sees, and phase 11 reuses
them in the README. They are worth the care.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from nnphysics.core.errors import UnknownNameError
from nnphysics.evals.metrics import NEVER_REACHED

if TYPE_CHECKING:
    from nnphysics.evals.result import InvariantRecord, SuiteResult

__all__ = [
    "TIMING_METRIC",
    "Direction",
    "Explanation",
    "explain",
    "explanations",
    "metric_summary",
]

TIMING_METRIC = "timing"
"""Pseudo metric under which a predictor's own timings are explained. Not a registered
metric: the wall clock is measured by the rollout driver rather than computed from a
trajectory, and it still belongs in the table with everything else."""

_NORMALISED = "1 (normalised)"
_TIME = "simulated time"
_STEPS = "steps"
_SECONDS = "seconds"
_DECLARED = "@declared"
"""Marker for units taken from the dimension the system declared for a named invariant."""


class Direction(StrEnum):
    """Which way a number has to move to count as an improvement."""

    LOWER = "lower is better"
    HIGHER = "higher is better"
    NEUTRAL = "no direction"
    """For a number that describes a run rather than judging it, and for one whose ideal
    is a particular value rather than an extreme. A comparison reports the change and
    declines to call it good or bad."""


@dataclass(frozen=True, slots=True)
class Explanation:
    """What one scalar means.

    Attributes:
        metric: Metric that produces it.
        key: Scalar name within that metric.
        title: Short label for a table row.
        units: Units, or `1 (normalised)` for a quantity measured against the signal.
        direction: Which way is an improvement.
        text: One line of plain English.
        sentinel: A value that means the thing did not happen rather than a measurement,
            or `None` if the scalar has no such value.
        sentinel_text: What the sentinel means, for a reader.
    """

    metric: str
    key: str
    title: str
    units: str
    direction: Direction
    text: str
    sentinel: float | None = None
    sentinel_text: str = ""

    @property
    def qualified(self) -> str:
        """The scalar's full name, `metric.scalar`."""
        return f"{self.metric}.{self.key}"


@dataclass(frozen=True, slots=True)
class _Rule:
    """A described scalar before the declared name is substituted into it."""

    title: str
    units: str
    direction: Direction
    text: str
    sentinel: float | None = None
    sentinel_text: str = ""

    def resolve(self, metric: str, key: str, name: str, units: str | None = None) -> Explanation:
        """Fill the declared name into the sentence and the title."""
        return Explanation(
            metric=metric,
            key=key,
            title=self.title.format(name=name),
            units=units if units is not None else self.units,
            direction=self.direction,
            text=self.text.format(name=name),
            sentinel=self.sentinel,
            sentinel_text=self.sentinel_text,
        )


METRIC_SUMMARIES: Mapping[str, str] = {
    "one_step_error": (
        "Error after a single step. The number a training loss reports, and on its own "
        "close to meaningless: a predictor can be excellent over one step and produce a "
        "different physical system after two hundred."
    ),
    "rollout_error": (
        "Error against ground truth as the rollout goes on, and how long the predictor "
        "stays within given levels of it."
    ),
    "invariant_drift": (
        "Whether the predictor respects the quantities the system says it conserves or "
        "drains. A predictor can track the truth closely and still be adding energy "
        "every step, and that shows here long before it shows in an error curve."
    ),
    "symmetry_violation": (
        "Whether the predictor commutes with the transformations the system declares. "
        "Transform the start, roll forward, undo the transformation: a predictor that "
        "respects the symmetry gives the same answer either way."
    ),
    "distribution_drift": (
        "Whether the predicted states still look like states of the right system once "
        "they have stopped tracking the true ones. Two chaotic trajectories part "
        "whatever produced them; a cluster that is no longer a cluster is a real fault."
    ),
    "resolution_generalisation": (
        "Whether the predictor means the same thing on a finer grid than the one it was "
        "fitted on. Refine the initial condition, roll forward there, coarsen back. A "
        "predictor whose parameters describe the physics should agree with itself; one "
        "whose parameters describe the grid should not. A system whose states are not a "
        "discretisation of anything is not tested, and reports zero steps."
    ),
    TIMING_METRIC: (
        "What the predictor cost to run. A surrogate earns its place by being faster "
        "than the solver at an accuracy someone is willing to accept."
    ),
}
"""One paragraph per metric, for the section heading a report puts its numbers under."""

_EXACT: Mapping[str, Mapping[str, _Rule]] = {
    "one_step_error": {
        "error": _Rule(
            "one step error",
            _NORMALISED,
            Direction.LOWER,
            "Error after a single step, measured against the size of the true state: one "
            "means the predictor is as wrong as the signal is large.",
        ),
    },
    "rollout_error": {
        "error.final": _Rule(
            "final error",
            _NORMALISED,
            Direction.LOWER,
            "Error at the end of the rollout, measured against the size of the true state.",
        ),
        "error.max": _Rule(
            "worst error",
            _NORMALISED,
            Direction.LOWER,
            "Largest error reached at any point of the rollout.",
        ),
        "error.mean": _Rule(
            "mean error",
            _NORMALISED,
            Direction.LOWER,
            "Error averaged over every step of the rollout.",
        ),
        "duration": _Rule(
            "duration",
            _TIME,
            Direction.NEUTRAL,
            "Simulated time the rollout covered, which is what every horizon below is "
            "measured against.",
        ),
    },
    "invariant_drift": {
        "worst_violation": _Rule(
            "worst violation",
            _NORMALISED,
            Direction.LOWER,
            "Largest violation over every invariant the system declares. Above one, the "
            "predictor broke a declaration the system made.",
        ),
    },
    "symmetry_violation": {
        "worst": _Rule(
            "worst violation",
            _NORMALISED,
            Direction.LOWER,
            "Largest equivariance error over every symmetry the system declares.",
        ),
        "steps": _Rule(
            "steps tested",
            _STEPS,
            Direction.NEUTRAL,
            "Steps the equivariance test rolled out for, shorter than the main rollout "
            "because each symmetry costs another rollout.",
        ),
    },
    "distribution_drift": {
        "worst": _Rule(
            "worst distance",
            _NORMALISED,
            Direction.LOWER,
            "Largest quantile distance over every field: how far the late predicted "
            "states have drifted from the distribution of the true ones.",
        ),
        "window": _Rule(
            "window",
            _STEPS,
            Direction.NEUTRAL,
            "States at the end of the rollout the distributions were taken over.",
        ),
    },
    "resolution_generalisation": {
        "worst_consistency": _Rule(
            "worst inconsistency",
            _NORMALISED,
            Direction.LOWER,
            "Largest disagreement, over every declared refinement, between the predictor "
            "run at its native resolution and the same predictor run at the finer one. "
            "Near zero means the predictor is the same operator on both grids.",
        ),
        "worst_degradation": _Rule(
            "worst degradation",
            _NORMALISED,
            Direction.LOWER,
            "Most the error against ground truth grew when the predictor was run at a "
            "finer resolution instead of its native one. Ground truth is stored at the "
            "native resolution, so this also charges the refined rollout for detail the "
            "finer grid legitimately resolves.",
        ),
        "steps": _Rule(
            "steps tested",
            _STEPS,
            Direction.NEUTRAL,
            "Steps the resolution test rolled out for, shorter than the main rollout "
            "because the refined rollout works on a state several times the size. Zero "
            "means the system declares no finer resolution, so nothing was tested.",
        ),
    },
    TIMING_METRIC: {
        "seconds_per_step": _Rule(
            "seconds per step",
            _SECONDS,
            Direction.LOWER,
            "Mean wall clock spent advancing one stored interval, which is what a speedup "
            "claim is eventually measured from.",
        ),
    },
}
"""Scalars whose whole name is fixed."""

_SUFFIXED: Mapping[str, Mapping[str, _Rule]] = {
    "invariant_drift": {
        "violation": _Rule(
            "{name}: violation",
            _NORMALISED,
            Direction.LOWER,
            "How far the predictor took {name} past what the system allows, counting "
            "both the declared tolerance and the reference solver's own drift. Above one "
            "is a violation.",
        ),
        "drift": _Rule(
            "{name}: drift",
            _DECLARED,
            Direction.LOWER,
            "Largest excursion of {name} from its starting value along the predicted trajectory.",
        ),
        "reference_drift": _Rule(
            "{name}: reference drift",
            _DECLARED,
            Direction.NEUTRAL,
            "The same excursion along the true trajectory. This is the solver's own "
            "drift, and part of the predictor's is excused by it.",
        ),
        "scale": _Rule(
            "{name}: scale",
            _DECLARED,
            Direction.NEUTRAL,
            "Largest magnitude {name} reaches along the true trajectory, which is what "
            "its declared tolerance is relative to.",
        ),
        "excess": _Rule(
            "{name}: excess",
            _DECLARED,
            Direction.LOWER,
            "Most the predictor's {name} ever stood above the true value: quantity the "
            "physics did not supply.",
        ),
        "shortfall": _Rule(
            "{name}: shortfall",
            _DECLARED,
            Direction.LOWER,
            "Most the predictor's {name} ever stood below the true value: over "
            "dissipation rather than injection.",
        ),
    },
    "symmetry_violation": {
        "max": _Rule(
            "{name}: worst",
            _NORMALISED,
            Direction.LOWER,
            "Largest disagreement under {name} between transforming and then predicting, "
            "and predicting and then transforming.",
        ),
        "final": _Rule(
            "{name}: final",
            _NORMALISED,
            Direction.LOWER,
            "That same disagreement at the end of the tested horizon.",
        ),
        "mean": _Rule(
            "{name}: mean",
            _NORMALISED,
            Direction.LOWER,
            "That same disagreement averaged over the tested horizon, which catches a "
            "fault that comes and goes.",
        ),
        "steps": _Rule(
            "{name}: steps",
            _STEPS,
            Direction.NEUTRAL,
            "Steps the test under {name} managed before it stopped.",
        ),
    },
    "resolution_generalisation": {
        "consistency": _Rule(
            "{name}: inconsistency",
            _NORMALISED,
            Direction.LOWER,
            "Largest disagreement under {name} between the predictor at its native "
            "resolution and the same predictor at the finer one, with no ground truth "
            "involved: this asks whether the predictor is the same operator, not "
            "whether it is right.",
        ),
        "refined_error": _Rule(
            "{name}: refined error",
            _NORMALISED,
            Direction.LOWER,
            "Error against ground truth of the rollout run at the resolution {name} "
            "declares, coarsened back for the comparison.",
        ),
        "native_error": _Rule(
            "{name}: native error",
            _NORMALISED,
            Direction.LOWER,
            "Error against ground truth over the same steps at the stored resolution, "
            "which is what the refined error is measured against.",
        ),
        "degradation": _Rule(
            "{name}: degradation",
            _NORMALISED,
            Direction.LOWER,
            "The refined error under {name} less the native error: what running at the "
            "finer resolution cost. Negative means it helped.",
        ),
        "steps": _Rule(
            "{name}: steps",
            _STEPS,
            Direction.NEUTRAL,
            "Steps the test under {name} managed. Zero means the predictor could not run "
            "at that resolution at all, which a solver built for one grid cannot.",
        ),
    },
    "distribution_drift": {
        "quantile_distance": _Rule(
            "{name}: quantile distance",
            _NORMALISED,
            Direction.LOWER,
            "Mean distance between the predicted and true quantiles of {name}, measured "
            "against the spread of the true ones.",
        ),
        "spread_ratio": _Rule(
            "{name}: spread ratio",
            _NORMALISED,
            Direction.NEUTRAL,
            "Spread of the predicted {name} measured against the true one. Far below one "
            "is a distribution that has collapsed, far above it is noise.",
        ),
    },
}
"""Scalars named `<declared>.<suffix>`, where the declared part is an invariant, a
symmetry or a field the system named and this module never has to recognise."""

_FIELD_ERROR = _Rule(
    "error in {name}",
    _NORMALISED,
    Direction.LOWER,
    "The same error for the field {name} alone, which says which part of the state the "
    "predictor is getting wrong.",
)

_HORIZON = _Rule(
    "horizon at {name}",
    _TIME,
    Direction.HIGHER,
    "Simulated time before the error first exceeds {name}. This is the number a claim "
    "about how far a surrogate can be trusted is made from.",
    sentinel=NEVER_REACHED,
    sentinel_text="never reached",
)


def explain(
    metric: str, key: str, invariants: Mapping[str, InvariantRecord] | None = None
) -> Explanation:
    """Describe one scalar.

    Args:
        metric: Metric name, or `timing` for a predictor's own measurements.
        key: Scalar name within that metric.
        invariants: Declared invariants by name, used to give a drift the units the
            system declared for it. Without them such a scalar is described as being in
            the invariant's own units.

    Returns:
        The explanation.

    Raises:
        UnknownNameError: If nothing describes that scalar, which means a metric has
            grown a number that nobody has explained yet.
    """
    exact = _EXACT.get(metric, {}).get(key)
    if exact is not None:
        return exact.resolve(metric, key, key)
    if metric == "rollout_error" and key.startswith("horizon."):
        return _HORIZON.resolve(metric, key, key[len("horizon.") :])
    prefix, _, suffix = key.rpartition(".")
    rule = _SUFFIXED.get(metric, {}).get(suffix)
    if rule is not None and prefix:
        return rule.resolve(metric, key, prefix, units=_units(rule, prefix, invariants))
    if metric in {"one_step_error", "rollout_error"} and key.startswith("error.") and prefix:
        return _FIELD_ERROR.resolve(metric, key, key[len("error.") :])
    raise UnknownNameError(
        f"nothing explains the scalar {metric}.{key}. A number without an explanation "
        f"cannot go in a report: describe it in nnphysics.reporting.explain."
    )


def explanations(result: SuiteResult) -> tuple[Explanation, ...]:
    """Describe every scalar a suite result carries.

    Args:
        result: The evaluation result.

    Returns:
        One explanation per distinct scalar, ordered by the suite's metric order and then
        by scalar name.

    Raises:
        UnknownNameError: If any scalar has no explanation.
    """
    declared = _declared_invariants(result)
    keys: dict[str, set[str]] = {}
    for entry in result.results:
        for record in entry.metrics:
            keys.setdefault(record.name, set()).update(record.scalars)
    ordered = [name for name in result.settings.metrics if name in keys]
    ordered += sorted(name for name in keys if name not in ordered)
    described = [
        explain(metric, key, declared) for metric in ordered for key in sorted(keys[metric])
    ]
    described.append(explain(TIMING_METRIC, "seconds_per_step"))
    return tuple(described)


def metric_summary(metric: str) -> str:
    """One paragraph saying what a metric is for.

    Args:
        metric: Metric name.

    Returns:
        The paragraph.

    Raises:
        UnknownNameError: If the metric has no summary.
    """
    summary = METRIC_SUMMARIES.get(metric)
    if summary is None:
        raise UnknownNameError(
            f"nothing summarises the metric {metric!r}. Describe it in nnphysics.reporting.explain."
        )
    return summary


def _units(rule: _Rule, name: str, invariants: Mapping[str, InvariantRecord] | None) -> str:
    """Resolve units that the system declares rather than this module."""
    if rule.units != _DECLARED:
        return rule.units
    record = (invariants or {}).get(name)
    return record.dimension if record is not None else f"units of {name}"


def _declared_invariants(result: SuiteResult) -> dict[str, InvariantRecord]:
    """Every invariant the result carries, by name.

    A system declares the same invariants for every regime it defines, so the first
    declaration of a name is the one used. A regime that declared a different dimension
    for the same name would be a fault in the system rather than something a report
    should paper over, and the layering guard is what would catch it.
    """
    declared: dict[str, InvariantRecord] = {}
    for records in result.invariants.values():
        for record in records:
            declared.setdefault(record.name, record)
    return declared
