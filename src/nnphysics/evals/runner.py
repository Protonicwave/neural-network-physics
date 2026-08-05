"""Turning a configuration and a dataset into a result file.

The runner is where the pieces meet: it reads initial conditions and ground truth from
the data layer, wraps the system's solver so that every predictor steps at the same
interval, builds the predictors a suite names, drives the rollouts and hands each metric
the context it needs.

It is also the only place that touches a `System`, and it touches it only through the
protocol. The system declares its invariants, its symmetries and its solver; the runner
passes those on and never looks inside them. That is what lets the same suite, with only
the system name changed, score an N-body cluster and a two dimensional flow.

Ground truth comes from the stored dataset rather than from a fresh integration. The
stored trajectory is what the solver produced at the stored interval, which is exactly
what a surrogate is asked to reproduce, and re-deriving it would only add a chance of
disagreeing with the data a model was trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics import __version__
from nnphysics.core.errors import ValidationError
from nnphysics.core.types import Rollout, Trajectory
from nnphysics.data.generation import find_regime
from nnphysics.data.manifest import Split
from nnphysics.data.store import ShardReader
from nnphysics.evals.metrics import MetricContext, build_metrics
from nnphysics.evals.predictors import PredictorContext, Substepped, build_predictor, parse_spec
from nnphysics.evals.result import (
    InvariantRecord,
    MetricRecord,
    PredictorResult,
    RolloutRecord,
    SuiteResult,
    SuiteSettings,
)
from nnphysics.evals.rollout import roll_out

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.core.config import EvaluationConfig
    from nnphysics.core.protocols import Predictor, System
    from nnphysics.core.types import FloatArray, MetricResult, Regime, State
    from nnphysics.data.manifest import Manifest
    from nnphysics.evals.predictors import PredictorSpec

__all__ = [
    "EVAL_SEED_STREAM",
    "EvaluationCase",
    "build_case_predictor",
    "evaluate_predictor",
    "load_cases",
    "regime_gap",
    "run_suite",
]

EVAL_SEED_STREAM = "evals.{predictor}.{trajectory}"
"""How a stochastic predictor's generator is derived. It depends on the predictor and the
trajectory alone, so the noise a predictor sees does not change when another predictor is
added to the suite."""

_IN_DISTRIBUTION = Split.TEST
"""The split held out regimes are compared against. Test rather than validation, because
validation is what a model may be selected on and a gap measured against it would be
measured against a number the model had already seen."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One initial condition and the ground truth it is scored against.

    Attributes:
        trajectory_id: Identifier of the dataset trajectory.
        regime: The regime it was drawn from.
        split: The split it belongs to.
        reference: Ground truth, truncated to the rollout horizon.
    """

    trajectory_id: str
    regime: Regime
    split: Split
    reference: Trajectory

    @property
    def initial(self) -> State:
        """The state every predictor starts from."""
        return self.reference[0]

    @property
    def steps(self) -> int:
        """Steps of ground truth available after the initial state."""
        return len(self.reference) - 1


def load_cases(  # noqa: PLR0913
    # Where the data is, what to read out of it and how much. The three keyword arguments
    # are the suite's, and passing the suite itself would tie reading a dataset to having
    # one, which is the opposite of what this function is for.
    directory: Path,
    manifest: Manifest,
    system: System,
    *,
    split: Split,
    count: int,
    steps: int,
) -> list[EvaluationCase]:
    """Read initial conditions and ground truth out of a dataset.

    Trajectories are taken in identifier order rather than drawn, so that two runs of the
    same suite evaluate the same states without needing a generator to say so, and they
    are taken in turn from each regime the split holds. A split that spans two regimes and
    was sampled from the first of them alphabetically would report a number for the split
    that described only half of it.

    Args:
        directory: The dataset directory.
        manifest: Its manifest.
        system: The system, used only to resolve regime names to declared regimes.
        split: Which split to draw from.
        count: How many trajectories to take. Taking more than the split holds is an
            error rather than a silent truncation.
        steps: Rollout horizon, which must fit inside a stored trajectory.

    Returns:
        The cases, in identifier order.

    Raises:
        ValidationError: If the split is empty, holds fewer trajectories than asked for,
            or its trajectories are shorter than the horizon.
        UnknownNameError: If the system declares no regime a trajectory names.
    """
    members = manifest.split(split)
    if not members:
        raise ValidationError(f"split {split.value!r} holds no trajectories to evaluate on")
    if count > len(members):
        raise ValidationError(
            f"split {split.value!r} holds {len(members)} trajectories, asked for {count}"
        )
    if steps + 1 > manifest.spec.n_steps:
        raise ValidationError(
            f"a rollout of {steps} steps needs {steps + 1} stored states, but the dataset "
            f"holds {manifest.spec.n_steps}"
        )

    cases: list[EvaluationCase] = []
    readers: dict[str, ShardReader] = {}
    try:
        for identifier in _spread_over_regimes(manifest, members, count):
            record = manifest.trajectory(identifier)
            if record.shard not in readers:
                readers[record.shard] = ShardReader(directory / record.shard)
            fields, times = readers[record.shard].window(record.row, 0, steps + 1)
            cases.append(
                EvaluationCase(
                    trajectory_id=identifier,
                    regime=find_regime(system, record.regime),
                    split=split,
                    reference=Trajectory(fields=fields, times=times),
                )
            )
    finally:
        for reader in readers.values():
            reader.close()
    return cases


def evaluate_predictor(  # noqa: PLR0913
    # The system, the cases, what to build, how to run it and what seeds it. Every one of
    # them varies independently, and the object that would group them is the run
    # configuration, which this function deliberately does not take so that a test can
    # evaluate a predictor without inventing one.
    system: System,
    cases: Sequence[EvaluationCase],
    spec: PredictorSpec | str,
    config: EvaluationConfig,
    *,
    substeps: int,
    seed: int,
) -> PredictorResult:
    """Roll one predictor out from every case and score it.

    Args:
        system: The system, seen only through the protocol.
        cases: Initial conditions and ground truth, all from the same split.
        spec: Predictor specification, parsed or as written.
        config: The suite.
        substeps: Solver steps per stored interval, from the dataset specification.
        seed: Root seed of the run.

    Returns:
        The aggregated result.

    Raises:
        ValidationError: If no cases were given, they mix splits, or a case is shorter
            than the configured horizon.
        UnknownNameError: If the predictor or a metric is not registered.
    """
    parsed = parse_spec(spec) if isinstance(spec, str) else spec
    if not cases:
        raise ValidationError("cannot evaluate a predictor without any initial conditions")
    splits = {case.split for case in cases}
    if len(splits) != 1:
        raise ValidationError(f"cases mix splits {sorted(split.value for split in splits)}")

    records: list[RolloutRecord] = []
    measured: list[tuple[MetricResult, ...]] = []
    seconds = 0.0
    taken = 0

    for case in cases:
        steps = min(config.rollout_steps, case.steps)
        predictor = build_case_predictor(system, case, parsed, substeps=substeps, seed=seed)
        result = roll_out(
            predictor,
            case.initial,
            steps,
            divergence_factor=config.divergence_factor,
        )
        rollout = Rollout(
            predicted=result.trajectory,
            reference=_prefix(case.reference, len(result.trajectory)),
            predictor=result.predictor,
            system=system.name,
        )
        metrics = build_metrics(config.metrics, _metric_context(system, case, predictor, config))
        scored = tuple(metric.compute(rollout) for metric in metrics)
        measured.append(scored)
        records.append(
            RolloutRecord(
                trajectory=case.trajectory_id,
                regime=case.regime.name,
                split=case.split.value,
                steps_requested=steps,
                steps_completed=result.steps_completed,
                stop_reason=result.stop_reason.value,
                detail=result.detail,
                seconds=result.seconds,
                scalars=_flatten(scored),
            )
        )
        seconds += result.seconds
        taken += result.steps_completed

    return PredictorResult(
        predictor=parsed.name,
        spec=parsed.text,
        split=cases[0].split.value,
        regimes=tuple(sorted({case.regime.name for case in cases})),
        rollouts=tuple(records),
        metrics=_aggregate(measured),
        seconds_per_step=seconds / taken if taken else 0.0,
        completed=all(record.steps_completed == record.steps_requested for record in records),
    )


def run_suite(  # noqa: PLR0913
    # As with a single predictor, these vary independently, and the run configuration
    # that would group them is the caller's to hold rather than this function's to demand.
    system: System,
    directory: Path,
    manifest: Manifest,
    config: EvaluationConfig,
    *,
    seed: int,
    run_id: str,
    predictors: Sequence[str] | None = None,
    splits: Sequence[Split] | None = None,
) -> SuiteResult:
    """Run a whole suite and assemble the result.

    Args:
        system: The system, seen only through the protocol.
        directory: The dataset directory.
        manifest: Its manifest.
        config: The suite.
        seed: Root seed of the run.
        run_id: Identifier of the run configuration.
        predictors: Predictor specifications, or `None` to use the suite's own.
        splits: Splits to evaluate on, or `None` for the test split and, if the dataset
            has one, the held out split.

    Returns:
        The result, including the regime gap when both splits were evaluated.

    Raises:
        ValidationError: If a split is empty or too small, or a specification is
            malformed.
        UnknownNameError: If a predictor, metric or regime is not registered.
    """
    chosen = tuple(predictors) if predictors else config.predictors
    wanted = tuple(splits) if splits else _default_splits(manifest)
    specs = tuple(parse_spec(text) for text in chosen)

    results: list[PredictorResult] = []
    invariants: dict[str, tuple[InvariantRecord, ...]] = {}
    for split in wanted:
        cases = load_cases(
            directory,
            manifest,
            system,
            split=split,
            count=config.n_initial_conditions,
            steps=config.rollout_steps,
        )
        for case in cases:
            invariants.setdefault(case.regime.name, _declared_invariants(system, case.regime))
        for spec in specs:
            results.append(
                evaluate_predictor(
                    system,
                    cases,
                    spec,
                    config,
                    substeps=manifest.spec.substeps,
                    seed=seed,
                )
            )

    return SuiteResult(
        code_version=__version__,
        run_id=run_id,
        dataset_id=manifest.dataset_id,
        system=system.name,
        seed=seed,
        settings=_settings(config),
        invariants=invariants,
        results=tuple(results),
        regime_gap=regime_gap(results),
    )


def regime_gap(results: Sequence[PredictorResult]) -> dict[str, float]:
    """How much worse every number is outside the training regimes.

    Reported as a difference rather than a ratio, so that a scalar which is legitimately
    zero in distribution does not produce an infinite gap, and so that the sign always
    reads the same way: positive means the held out regimes are worse for any quantity a
    predictor wants small.

    Args:
        results: Predictor results, from any mixture of splits.

    Returns:
        Held out value minus in distribution value, keyed
        `predictor.metric.scalar`. Empty if either split is missing.
    """
    in_distribution = {
        entry.predictor: entry for entry in results if entry.split == _IN_DISTRIBUTION.value
    }
    held_out = {entry.predictor: entry for entry in results if entry.split == Split.HELD_OUT.value}
    gaps: dict[str, float] = {}
    for predictor, outside in held_out.items():
        inside = in_distribution.get(predictor)
        if inside is None:
            continue
        for record in outside.metrics:
            for key, value in record.scalars.items():
                try:
                    reference = inside.scalar(record.name, key)
                except KeyError:
                    continue
                gaps[f"{predictor}.{record.name}.{key}"] = value - reference
    return gaps


def _spread_over_regimes(manifest: Manifest, members: Sequence[str], count: int) -> list[str]:
    """Take `count` identifiers, one from each regime in turn, in identifier order."""
    by_regime: dict[str, list[str]] = {}
    for identifier in members:
        by_regime.setdefault(manifest.trajectory(identifier).regime, []).append(identifier)
    chosen: list[str] = []
    for position in range(max(len(entries) for entries in by_regime.values())):
        for regime in sorted(by_regime):
            entries = by_regime[regime]
            if position < len(entries) and len(chosen) < count:
                chosen.append(entries[position])
    return sorted(chosen)


def _default_splits(manifest: Manifest) -> tuple[Split, ...]:
    """Test, and held out as well when the dataset carries one."""
    if manifest.split(Split.HELD_OUT):
        return (_IN_DISTRIBUTION, Split.HELD_OUT)
    return (_IN_DISTRIBUTION,)


def build_case_predictor(
    system: System,
    case: EvaluationCase,
    spec: PredictorSpec,
    *,
    substeps: int,
    seed: int,
) -> Predictor:
    """Build the predictor for one case, with the reference solver already folded.

    The seeding depends on the predictor and the trajectory alone, so building the same
    predictor for the same case twice gives a predictor that produces the same states.
    That is what lets a later pass reproduce a rollout instead of storing it.

    Args:
        system: The system, seen only through the protocol.
        case: The initial condition and its ground truth.
        spec: Parsed predictor specification.
        substeps: Solver steps per stored interval.
        seed: Root seed of the run.

    Returns:
        The predictor.

    Raises:
        UnknownNameError: If the predictor is not registered.
        ValidationError: If the specification carries a setting the predictor refuses.
    """
    return build_predictor(
        spec,
        PredictorContext(
            reference=_reference_predictor(system, case, substeps),
            state_spec=system.state_spec,
            symmetries=system.symmetries,
            seed=seed,
            stream=EVAL_SEED_STREAM.format(predictor=spec.name, trajectory=case.trajectory_id),
        ),
    )


def _reference_predictor(system: System, case: EvaluationCase, substeps: int) -> Predictor:
    """The system's solver, stepping one stored interval at a time."""
    stored_dt = float(case.reference.times[1] - case.reference.times[0])
    return Substepped(system.reference_predictor(case.regime, stored_dt / substeps), substeps)


def _metric_context(
    system: System, case: EvaluationCase, predictor: Predictor, config: EvaluationConfig
) -> MetricContext:
    """What the metrics for one case are allowed to know."""
    return MetricContext(
        invariants=system.invariants(case.regime),
        symmetries=system.symmetries,
        predictor=predictor,
        thresholds=config.error_thresholds,
        symmetry_steps=config.symmetry_steps,
        distribution_window=config.distribution_window,
        divergence_factor=config.divergence_factor,
    )


def _declared_invariants(system: System, regime: Regime) -> tuple[InvariantRecord, ...]:
    """Record what a system declared, so a drift number can be read later."""
    return tuple(
        InvariantRecord(
            name=invariant.name,
            conservation=invariant.conservation.value,
            rtol=invariant.rtol,
            dimension=invariant.dimension.symbol,
        )
        for invariant in system.invariants(regime)
    )


def _settings(config: EvaluationConfig) -> SuiteSettings:
    """Copy the suite into the result, so a number carries the settings that made it."""
    return SuiteSettings(
        name=config.name,
        metrics=config.metrics,
        rollout_steps=config.rollout_steps,
        n_initial_conditions=config.n_initial_conditions,
        error_thresholds=config.error_thresholds,
        symmetry_steps=config.symmetry_steps,
        distribution_window=config.distribution_window,
        divergence_factor=config.divergence_factor,
    )


def _prefix(trajectory: Trajectory, length: int) -> Trajectory:
    """The first `length` states of a trajectory."""
    if length >= len(trajectory):
        return trajectory
    return Trajectory(
        fields={name: array[:length] for name, array in trajectory.fields.items()},
        times=trajectory.times[:length],
    )


def _aggregate(measured: Sequence[tuple[MetricResult, ...]]) -> tuple[MetricRecord, ...]:
    """Average each metric's scalars and curves over the initial conditions.

    A scalar missing from some rollouts is averaged over the ones that have it, and a
    curve whose length varies between rollouts is dropped rather than padded: a rollout
    that stopped early has no values to contribute past where it stopped, and inventing
    them would smooth over exactly the failure worth seeing.
    """
    if not measured:
        return ()
    records: list[MetricRecord] = []
    for position, first in enumerate(measured[0]):
        across = [entry[position] for entry in measured]
        records.append(
            MetricRecord(
                name=first.name,
                scalars=_mean_scalars(across),
                series=_mean_series(across),
            )
        )
    return tuple(records)


def _flatten(results: Sequence[MetricResult]) -> dict[str, float]:
    """Every scalar of one rollout, keyed `metric.scalar`."""
    return {
        f"{result.name}.{key}": value
        for result in results
        for key, value in sorted(result.scalars.items())
    }


def _mean_scalars(results: Sequence[MetricResult]) -> dict[str, float]:
    """Mean of each named scalar, over the results that carry it."""
    collected: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.scalars.items():
            collected.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in sorted(collected.items())}


def _mean_series(results: Sequence[MetricResult]) -> dict[str, tuple[float, ...]]:
    """Mean of each named curve, over the results that carry it at a common length."""
    collected: dict[str, list[FloatArray]] = {}
    for result in results:
        for key, values in result.series.items():
            collected.setdefault(key, []).append(values)
    return {
        key: tuple(float(value) for value in np.mean(np.stack(arrays), axis=0))
        for key, arrays in sorted(collected.items())
        if len(arrays) == len(results) and len({array.shape for array in arrays}) == 1
    }
