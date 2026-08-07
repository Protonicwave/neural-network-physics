"""Two run records reduced to something worth sending.

The whole difficulty of this phase is here rather than in the client. A run record holds
several megabytes of curves; a diagnosis is a paragraph. What survives the reduction is
what the model gets to reason from, and anything dropped is a fault it cannot possibly
name.

Four things survive, and each answers a different question a person would ask.

What was changed: every leaf of the configuration that differs, with both values. Four of
the seven injected faults are visible here in principle, and it is the reason a diagnosis
is a comparison of two runs rather than a reading of one. The other three leave nothing
here at all, and the three sections below are the only evidence they leave anywhere.

What got worse: the scalar deltas, already carrying the direction each one is supposed to
move in, taken from the phase 06 comparison so that a diagnosis and a report cannot
disagree about what a regression is.

What shape the failure has: an error curve reduced to six numbers. A curve that rises
from the first step is a different fault from one that is flat and then explodes, and a
mean over the rollout cannot tell them apart.

What training did: the loss and the validation rollout at both ends of the run, the best
epoch, and whether patience ran out. A learning rate that is too high and a curriculum
that never lengthened both produce a bad model, and the training history is the only
place they look different.

No raw array is sent. Every number here is a scalar, and the caps below bound the prompt
whatever size the run was.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from nnphysics.reporting.compare import Comparison, Verdict, compare_records

if TYPE_CHECKING:
    from nnphysics.evals.result import PredictorResult, SuiteResult
    from nnphysics.reporting.record import RunRecord
    from nnphysics.training.history import TrainingHistory

__all__ = [
    "IGNORED_CONFIG_PATHS",
    "MAX_CONFIG_DIFFERENCES",
    "MAX_CURVES",
    "MAX_IMPROVEMENTS",
    "MAX_REGRESSIONS",
    "MAX_ROLLOUTS",
    "ConfigDifference",
    "CurveShape",
    "DiagnosisContext",
    "RolloutShape",
    "ScalarChange",
    "TrainingShape",
    "build_context",
]

MAX_CONFIG_DIFFERENCES = 40
"""Configuration leaves reported. Two runs of the same pipeline differ in a handful of
places, so a diff longer than this is two unrelated runs rather than a regression."""

MAX_REGRESSIONS = 24
"""Regressions reported, worst relative change first."""

MAX_IMPROVEMENTS = 8
"""Improvements reported. Fewer than the regressions, because what got better is context
for the diagnosis rather than the subject of it, but not none: a fault that improves one
number while ruining another is the kind a single sided summary hides."""

MAX_CURVES = 8
"""Error curves reduced. One per predictor and split, worst regression first."""

MAX_ROLLOUTS = 12
"""Rollout outcomes reported, taken from the ones that did not finish."""

IGNORED_CONFIG_PATHS = frozenset(
    {
        "name",
        "output_dir",
        "data.root",
        "data.workers",
        "data.shard_trajectories",
        "data.compression_level",
        "training.loader_workers",
    }
)
"""Configuration leaves that say where a run put its files or how many processes it used.
They differ between machines and never between a working run and a broken one, so leaving
them in the diff would be noise the model has to read past every time."""

_ERROR_METRIC = "rollout_error"
_ERROR_SERIES = "error"
_MINIMUM_CURVE = 3
"""Below three points a curve has no shape to describe."""

_UNBOUNDED = 1.0e12
"""What a non finite number is recorded as. JSON has no infinity, and a relative change is
infinite whenever the baseline was zero, which is every scalar of the reference solver. The
sign carries the meaning, so it is kept and the magnitude is replaced with a value large
enough that no measurement reaches it. The renderer says `unbounded` rather than printing
it."""


class ConfigDifference(BaseModel):
    """One configuration leaf that differs between the two runs.

    Attributes:
        path: Dotted path to the leaf, for example `training.learning_rate`.
        baseline: Its value in the baseline run, rendered.
        candidate: Its value in the candidate run, rendered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    baseline: str
    candidate: str


class ScalarChange(BaseModel):
    """One scalar in both runs, and what the comparison made of it.

    Attributes:
        split: Split it was measured on.
        predictor: Predictor it belongs to.
        scalar: Full scalar name, `metric.scalar`.
        baseline: Value in the baseline run.
        candidate: Value in the candidate run.
        relative: Change as a fraction of the size of the baseline.
        direction: Which way counts as an improvement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str = Field(min_length=1)
    predictor: str = Field(min_length=1)
    scalar: str = Field(min_length=1)
    baseline: float
    candidate: float
    relative: float
    direction: str = Field(min_length=1)


class CurveShape(BaseModel):
    """An error curve reduced to the numbers that describe its shape.

    Attributes:
        split: Split the rollout was measured on.
        predictor: Predictor that produced it.
        first: Error at the first predicted step.
        final: Error at the end of the rollout.
        worst: Largest error reached.
        worst_position: Where the worst error occurred, as a fraction of the rollout.
        rising_fraction: Share of the steps at which the error grew. Near one is a curve
            that never recovers; near a half is a curve that wanders.
        growth: Final error divided by the first, which separates a predictor that is
            uniformly inaccurate from one that starts well and diverges.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str = Field(min_length=1)
    predictor: str = Field(min_length=1)
    first: float
    final: float
    worst: float
    worst_position: float
    rising_fraction: float
    growth: float


class TrainingShape(BaseModel):
    """What a training run did, in the numbers a diagnosis needs.

    Attributes:
        model: Registered model name.
        n_parameters: Trainable parameters.
        epochs: Epochs actually run.
        best_epoch: Epoch with the lowest validation rollout error.
        best_validation_error: That error.
        stopped_early: Whether patience ran out.
        first_loss: Training objective in the first epoch.
        final_loss: Training objective in the last epoch.
        first_validation_error: Validation rollout error in the first epoch.
        final_validation_error: The same in the last epoch.
        peak_learning_rate: Largest rate the optimiser stepped at.
        max_gradient_norm: Largest mean gradient norm before clipping, which is what says
            whether the optimiser was being held back.
        curriculum_stages: Distinct rollout lengths trained against, in order.
        non_finite_epochs: Epochs whose loss was not a finite number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    n_parameters: int = Field(ge=0)
    epochs: int = Field(ge=1)
    best_epoch: int = Field(ge=0)
    best_validation_error: float
    stopped_early: bool
    first_loss: float
    final_loss: float
    first_validation_error: float
    final_validation_error: float
    peak_learning_rate: float
    max_gradient_norm: float
    curriculum_stages: tuple[int, ...]
    non_finite_epochs: int = Field(ge=0)


class RolloutShape(BaseModel):
    """How far one predictor got on one split.

    Attributes:
        split: Split the rollouts started from.
        predictor: Predictor that ran them.
        completed: Rollouts that reached the requested horizon.
        total: Rollouts attempted.
        stop_reasons: Distinct reasons the others ended, sorted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str = Field(min_length=1)
    predictor: str = Field(min_length=1)
    completed: int = Field(ge=0)
    total: int = Field(ge=1)
    stop_reasons: tuple[str, ...]


class DiagnosisContext(BaseModel):
    """Everything a diagnosis is allowed to reason from.

    Attributes:
        system: System both runs evaluated.
        suite: Evaluation suite both ran.
        baseline_id: Run identifier of the run compared against.
        candidate_id: Run identifier of the run compared.
        baseline_name: Human readable label of the baseline.
        candidate_name: The same for the candidate.
        threshold: Relative change below which a scalar was called unchanged.
        config_differences: Configuration leaves that differ.
        config_differences_dropped: Leaves that differ and did not fit the cap.
        regressions: Scalars that got worse, worst relative change first.
        regressions_dropped: Regressions that did not fit the cap.
        improvements: Scalars that got better, largest relative change first.
        unchanged: Scalars neither run moved beyond the threshold.
        undirected: Scalars whose ideal is neither extreme, so no verdict was reached.
        curves: Error curves reduced to their shape, candidate run.
        baseline_training: What the baseline's training did, or `None` if it trained
            nothing.
        candidate_training: The same for the candidate.
        rollouts: Rollouts that did not reach the requested horizon, candidate run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    baseline_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    baseline_name: str = Field(min_length=1)
    candidate_name: str = Field(min_length=1)
    threshold: float
    config_differences: tuple[ConfigDifference, ...]
    config_differences_dropped: int = Field(ge=0)
    regressions: tuple[ScalarChange, ...]
    regressions_dropped: int = Field(ge=0)
    improvements: tuple[ScalarChange, ...]
    unchanged: int = Field(ge=0)
    undirected: int = Field(ge=0)
    curves: tuple[CurveShape, ...]
    baseline_training: TrainingShape | None
    candidate_training: TrainingShape | None
    rollouts: tuple[RolloutShape, ...]

    @property
    def regressed_metrics(self) -> tuple[str, ...]:
        """Metrics that carry at least one regression, worst first.

        Returns:
            Metric names, ordered by the largest relative regression each one carries.
        """
        worst: dict[str, float] = {}
        for change in self.regressions:
            metric = change.scalar.split(".", 1)[0]
            worst[metric] = max(worst.get(metric, 0.0), abs(change.relative))
        return tuple(sorted(worst, key=lambda name: -worst[name]))

    def render(self) -> str:
        """The context as the text the model is shown.

        Returns:
            A block of plain text. Deterministic, so the same two records always produce
            the same prompt and a recorded reply keeps describing the call it came from.
        """
        return "\n".join(_render(self))


def build_context(
    baseline: RunRecord, candidate: RunRecord, *, threshold: float = 0.05
) -> DiagnosisContext:
    """Reduce two run records to what a diagnosis needs.

    Args:
        baseline: The run compared against, which is the one believed to be good.
        candidate: The run compared, which is the one being explained.
        threshold: Relative change below which a scalar is called unchanged.

    Returns:
        The context.

    Raises:
        ValidationError: If the two runs evaluated different systems.
        UnknownNameError: If a scalar has no explanation, so no direction is known for it.
    """
    comparison = compare_records(baseline, candidate, threshold=threshold)
    differences = _config_differences(baseline, candidate)
    regressions = _changes(comparison, Verdict.REGRESSED)
    improvements = _changes(comparison, Verdict.IMPROVED)
    return DiagnosisContext(
        system=candidate.system,
        suite=candidate.suite,
        baseline_id=baseline.run_id,
        candidate_id=candidate.run_id,
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        threshold=threshold,
        config_differences=tuple(differences[:MAX_CONFIG_DIFFERENCES]),
        config_differences_dropped=max(len(differences) - MAX_CONFIG_DIFFERENCES, 0),
        regressions=tuple(regressions[:MAX_REGRESSIONS]),
        regressions_dropped=max(len(regressions) - MAX_REGRESSIONS, 0),
        improvements=tuple(improvements[:MAX_IMPROVEMENTS]),
        unchanged=comparison.summary[Verdict.UNCHANGED],
        undirected=comparison.summary[Verdict.UNDIRECTED],
        curves=_curves(candidate.evaluation, regressions),
        baseline_training=_training(baseline.training),
        candidate_training=_training(candidate.training),
        rollouts=_rollouts(candidate.evaluation),
    )


def _config_differences(baseline: RunRecord, candidate: RunRecord) -> list[ConfigDifference]:
    """Every configuration leaf the two runs disagree about, in path order."""
    left = _flatten(baseline.config.model_dump(mode="json"))
    right = _flatten(candidate.config.model_dump(mode="json"))
    return [
        ConfigDifference(
            path=path,
            baseline=_render_value(left.get(path)),
            candidate=_render_value(right.get(path)),
        )
        for path in sorted(set(left) | set(right))
        if path not in IGNORED_CONFIG_PATHS and left.get(path) != right.get(path)
    ]


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:  # noqa: ANN401
    """Flatten nested mappings to dotted paths, leaving sequences whole.

    A sequence is a leaf rather than a set of numbered paths, because a curriculum that
    lengthened from `[1]` to `[1, 4, 8]` is one change and reading it as three would bury
    it in the noise of the indices moving.
    """
    if not isinstance(payload, Mapping):
        return {prefix: payload}
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(value, path))
    return flattened


def _render_value(value: Any) -> str:  # noqa: ANN401
    """Render a configuration leaf compactly, marking one that is absent."""
    if value is None:
        return "none"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "[" + ", ".join(_render_value(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _changes(comparison: Comparison, verdict: Verdict) -> list[ScalarChange]:
    """Every delta with one verdict, largest relative change first."""
    selected = sorted(comparison.of(verdict), key=lambda delta: -abs(delta.relative))
    return [
        ScalarChange(
            split=delta.split,
            predictor=delta.predictor,
            scalar=delta.scalar,
            baseline=_finite(delta.baseline),
            candidate=_finite(delta.candidate),
            relative=_finite(delta.relative),
            direction=str(delta.direction),
        )
        for delta in selected
    ]


def _finite(value: float) -> float:
    """Replace a non finite number with something JSON can carry.

    A relative change is infinite whenever the baseline was zero, which happens for every
    scalar of the reference solver. Serialising an infinity would produce JSON no strict
    reader accepts, and the sign of it is the part that carries the meaning.
    """
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return math.copysign(_UNBOUNDED, value)
    return value


def _curves(result: SuiteResult, regressions: Sequence[ScalarChange]) -> tuple[CurveShape, ...]:
    """Reduce the error curves, the predictors that regressed most first."""
    rank: dict[tuple[str, str], float] = {}
    for change in regressions:
        key = (change.split, change.predictor)
        rank[key] = max(rank.get(key, 0.0), abs(change.relative))
    shapes = [shape for entry in result.results if (shape := _curve(entry)) is not None]
    shapes.sort(key=lambda shape: -rank.get((shape.split, shape.predictor), 0.0))
    return tuple(shapes[:MAX_CURVES])


def _curve(entry: PredictorResult) -> CurveShape | None:
    """One predictor's error curve, reduced, or `None` if it has no usable one."""
    for record in entry.metrics:
        if record.name != _ERROR_METRIC:
            continue
        series = record.series.get(_ERROR_SERIES)
        if series is None or len(series) < _MINIMUM_CURVE:
            return None
        values = [_finite(value) for value in series]
        # The curve starts at the initial condition, where every predictor is exactly
        # right. Keeping that point would make the first error zero for everything and the
        # growth ratio undefined, which is the one number that separates a predictor that
        # is uniformly inaccurate from one that starts well and then diverges.
        if values[0] == 0.0 and len(values) > _MINIMUM_CURVE:
            values = values[1:]
        worst = max(values)
        rising = sum(1 for earlier, later in pairwise(values) if later > earlier)
        first = values[0]
        return CurveShape(
            split=entry.split,
            predictor=entry.predictor,
            first=first,
            final=values[-1],
            worst=worst,
            worst_position=values.index(worst) / (len(values) - 1),
            rising_fraction=rising / (len(values) - 1),
            growth=_finite(values[-1] / first) if first > 0.0 else 0.0,
        )
    return None


def _training(history: TrainingHistory | None) -> TrainingShape | None:
    """Reduce a training history, or pass `None` through for a run that trained nothing."""
    if history is None:
        return None
    epochs = history.epochs
    return TrainingShape(
        model=history.model,
        n_parameters=history.n_parameters,
        epochs=len(epochs),
        best_epoch=history.best_epoch,
        best_validation_error=_finite(history.best_validation_error),
        stopped_early=history.stopped_early,
        first_loss=_finite(epochs[0].loss),
        final_loss=_finite(epochs[-1].loss),
        first_validation_error=_finite(epochs[0].validation_error),
        final_validation_error=_finite(epochs[-1].validation_error),
        peak_learning_rate=max(record.learning_rate for record in epochs),
        max_gradient_norm=_finite(max(record.gradient_norm for record in epochs)),
        curriculum_stages=tuple(dict.fromkeys(record.curriculum_steps for record in epochs)),
        non_finite_epochs=sum(1 for record in epochs if not math.isfinite(record.loss)),
    )


def _rollouts(result: SuiteResult) -> tuple[RolloutShape, ...]:
    """Every predictor and split whose rollouts did not all reach the horizon."""
    shapes: list[RolloutShape] = []
    for entry in result.results:
        if entry.completed:
            continue
        finished = sum(
            1 for rollout in entry.rollouts if rollout.steps_completed >= rollout.steps_requested
        )
        shapes.append(
            RolloutShape(
                split=entry.split,
                predictor=entry.predictor,
                completed=finished,
                total=len(entry.rollouts),
                stop_reasons=tuple(sorted({rollout.stop_reason for rollout in entry.rollouts})),
            )
        )
    return tuple(shapes[:MAX_ROLLOUTS])


def _render(context: DiagnosisContext) -> list[str]:
    """Build the prompt block, section by section."""
    lines = [
        f"System: {context.system}. Suite: {context.suite}.",
        f"Baseline run: {context.baseline_name} ({context.baseline_id}).",
        f"Candidate run: {context.candidate_name} ({context.candidate_id}).",
        "",
        "## Configuration differences",
    ]
    if context.config_differences:
        lines += [
            f"- {entry.path}: {entry.baseline} -> {entry.candidate}"
            for entry in context.config_differences
        ]
    else:
        lines.append("- none. The two runs were configured identically.")
    if context.config_differences_dropped:
        lines.append(f"- and {context.config_differences_dropped} more, not shown.")

    lines += ["", "## Regressions", *_render_changes(context.regressions, "regression")]
    if context.regressions_dropped:
        lines.append(f"- and {context.regressions_dropped} more, not shown.")
    lines += ["", "## Improvements", *_render_changes(context.improvements, "improvement")]
    lines += [
        "",
        f"{context.unchanged} scalars did not move beyond the {context.threshold:.0%} "
        f"threshold. {context.undirected} have no direction of improvement and were not judged.",
    ]

    lines += ["", "## Error curve shapes, candidate run"]
    if context.curves:
        lines += [
            f"- {curve.split}/{curve.predictor}: first {curve.first:.4g}, final "
            f"{curve.final:.4g}, worst {curve.worst:.4g} at {curve.worst_position:.0%} of "
            f"the rollout, rising at {curve.rising_fraction:.0%} of steps, growth "
            f"{curve.growth:.4g}x"
            for curve in context.curves
        ]
    else:
        lines.append("- none recorded.")

    lines += ["", "## Training"]
    lines += _render_training("baseline", context.baseline_training)
    lines += _render_training("candidate", context.candidate_training)

    lines += ["", "## Rollouts that did not finish, candidate run"]
    if context.rollouts:
        lines += [
            f"- {shape.split}/{shape.predictor}: {shape.completed} of {shape.total} "
            f"completed, ended by {', '.join(shape.stop_reasons)}"
            for shape in context.rollouts
        ]
    else:
        lines.append("- none. Every rollout reached the requested horizon.")
    return lines


def _render_changes(changes: Sequence[ScalarChange], kind: str) -> list[str]:
    """Render a block of scalar changes."""
    if not changes:
        return [f"- none. No scalar was judged a {kind}."]
    return [
        f"- {change.split}/{change.predictor} {change.scalar}: {change.baseline:.4g} -> "
        f"{change.candidate:.4g} ({_render_relative(change.relative)}, {change.direction})"
        for change in changes
    ]


def _render_relative(relative: float) -> str:
    """Render a relative change, naming the one that has no size rather than printing it."""
    if abs(relative) >= _UNBOUNDED:
        return "unbounded, the baseline was zero"
    return f"{relative:+.1%}"


def _render_training(label: str, shape: TrainingShape | None) -> list[str]:
    """Render one run's training summary."""
    if shape is None:
        return [f"- {label}: trained nothing."]
    return [
        f"- {label}: {shape.model}, {shape.n_parameters} parameters, {shape.epochs} epochs, "
        f"curriculum stages {list(shape.curriculum_stages)}.",
        f"  loss {shape.first_loss:.4g} -> {shape.final_loss:.4g}, validation rollout "
        f"{shape.first_validation_error:.4g} -> {shape.final_validation_error:.4g}, best "
        f"{shape.best_validation_error:.4g} at epoch {shape.best_epoch}.",
        f"  peak learning rate {shape.peak_learning_rate:.4g}, largest gradient norm "
        f"{shape.max_gradient_norm:.4g}, {shape.non_finite_epochs} epochs with a non finite "
        f"loss, stopped early: {shape.stopped_early}.",
    ]
