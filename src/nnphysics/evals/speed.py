"""Speedup at matched accuracy, and the number of rollouts it takes to pay for itself.

The comparison a surrogate is usually reported with is the wrong one. Timing a network
against the solver at the settings the training data was generated with flatters the
network, because those settings were chosen to make ground truth rather than to be fast.
The solver has a knob, the number of substeps it folds into one stored interval, and
turning it down makes the solver cheaper and less accurate. The honest question is where
the surrogate sits against the solver run as coarsely as it can be while still being as
accurate as the surrogate is.

So the ladder is built first: the solver at every substep count that divides the one the
dataset used, each with its accuracy and its cost measured on the same states. A
surrogate is one more point on the same axes. The speedup is then read off the crossing,
and if no rung of the ladder is as inaccurate as the surrogate then the solver could be
run cheaper still than anything measured here, and the number is reported as the upper
bound it is rather than as a result.

Cost accounting is the other half. A surrogate that saves a second per rollout and cost
two hours to train has not saved anything until it has been run seven thousand times, and
that count is the useful number. Data generation is counted too, because a surrogate that
needs a dataset the solver did not need is a surrogate that spent the solver's time to
avoid spending it.

Accuracy here is the worst normalised error over the horizon rather than the error at the
end of it. A configuration is only as trustworthy as its worst moment, and a rollout that
went wrong and came back is not a rollout that stayed right.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nnphysics.core.errors import ValidationError
from nnphysics.evals.benchmark import (
    DEFAULT_STEPS_PER_TRIAL,
    DEFAULT_TRIALS,
    DEFAULT_WARMUP,
    Timing,
    benchmark_predictor,
)
from nnphysics.evals.metrics import relative_error
from nnphysics.evals.predictors import REFERENCE_NAME, parse_spec
from nnphysics.evals.rollout import roll_out
from nnphysics.evals.runner import build_case_predictor, substepped_reference

if TYPE_CHECKING:
    from nnphysics.core.config import EvaluationConfig
    from nnphysics.core.protocols import Predictor, System
    from nnphysics.core.types import Trajectory
    from nnphysics.evals.predictors import PredictorFactory
    from nnphysics.evals.runner import EvaluationCase

__all__ = [
    "NEVER_PAYS",
    "SPEED_SCHEMA_VERSION",
    "CostAccounting",
    "MatchedSpeedup",
    "SpeedPoint",
    "SpeedReport",
    "build_speed_report",
    "cost_accounting",
    "matched_speedup",
    "measure_point",
    "solver_ladder",
    "substep_ladder",
]

SPEED_SCHEMA_VERSION = 1
"""Bumped when the shape of a speed report changes incompatibly."""

NEVER_PAYS = -1.0
"""Reported as a break even count when the surrogate is not cheaper than the solver it
would have to replace, so no number of rollouts ever repays what it cost to train.
Negative because a count of rollouts cannot be, and so a reader cannot mistake it for
one."""

_SOLVER = "solver"
_SURROGATE = "surrogate"

_MINIMUM_STATES = 2
"""A trajectory of one state spans no interval, so it cost nothing to generate."""


class Record(BaseModel):
    """Base for every speed record: immutable and intolerant of unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SpeedPoint(Record):
    """One point on the accuracy against wall clock curve.

    Attributes:
        label: How the point is named in a table, for example `reference:substeps=2`.
        predictor: The predictor that produced it.
        kind: `solver` for a rung of the ladder, `surrogate` for anything else.
        substeps: Solver steps folded into one stored interval, or zero for a predictor
            that takes none.
        error: Worst normalised error over the horizon, averaged over initial conditions.
        seconds_per_step: Median wall clock to advance one stored interval.
        iqr: Interquartile range of that time over the trials.
        relative_spread: The interquartile range as a fraction of the median.
        stable: Whether that spread is small enough for the median to be worth quoting.
        completed: Whether every rollout reached the horizon. A point that did not is
            reporting an error over less of the rollout than the others.
    """

    label: str = Field(min_length=1)
    predictor: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    substeps: int = Field(ge=0)
    error: float
    seconds_per_step: float = Field(ge=0.0)
    iqr: float = Field(ge=0.0)
    relative_spread: float = Field(ge=0.0)
    stable: bool
    completed: bool


class MatchedSpeedup(Record):
    """What a surrogate is worth against the cheapest solver that is as accurate.

    Attributes:
        predictor: The surrogate.
        error: Its worst normalised error over the horizon.
        seconds_per_step: What it costs.
        matched_substeps: Substeps of the cheapest solver setting at least as accurate.
        matched_error: That setting's error.
        matched_seconds_per_step: What that setting costs.
        speedup: Solver cost divided by surrogate cost. Above one means the surrogate is
            faster at the same accuracy, which is the whole claim.
        bracketed: Whether a cheaper solver setting was measured that is genuinely worse
            than the surrogate. False means the ladder ran out before the surrogate's
            accuracy was reached from below, so the solver could be run cheaper still and
            the speedup is an upper bound rather than a measurement.
    """

    predictor: str = Field(min_length=1)
    error: float
    seconds_per_step: float = Field(ge=0.0)
    matched_substeps: int = Field(ge=1)
    matched_error: float
    matched_seconds_per_step: float = Field(ge=0.0)
    speedup: float
    bracketed: bool


class CostAccounting(Record):
    """What the surrogate cost once, against what it saves each time it is used.

    Attributes:
        predictor: The surrogate.
        training_seconds: Wall clock spent training it.
        generation_seconds: Wall clock the training data cost, measured rather than
            recalled: the number of trajectories times their length times what the solver
            costs per stored interval on this machine at these thread settings.
        steps_per_rollout: Stored intervals in one rollout, which is what a saving per
            step is multiplied by.
        saving_per_rollout: Seconds one rollout saves against the matched solver setting.
            Negative when the surrogate is the slower of the two.
        break_even_rollouts: Rollouts before the one off cost is repaid, or `NEVER_PAYS`.
    """

    predictor: str = Field(min_length=1)
    training_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    steps_per_rollout: int = Field(ge=1)
    saving_per_rollout: float
    break_even_rollouts: float


class SpeedReport(Record):
    """Everything one benchmark measured.

    Attributes:
        schema_version: Version of this record's shape.
        system: System name.
        split: Split the initial conditions came from.
        steps: Horizon accuracy was measured over.
        n_initial_conditions: Initial conditions accuracy was averaged over.
        threads: Threads the arithmetic was fixed to.
        trials: Timed repeats per point.
        warmup: Repeats discarded before timing.
        steps_per_trial: Steps each trial averaged over.
        dataset_substeps: Substeps the ground truth was generated with, which is the top
            of the ladder and the only rung whose error is zero by construction.
        ladder: The solver at each substep count, cheapest first.
        surrogates: Every other predictor measured, in the order asked for.
        matched: One entry per surrogate.
        costs: One entry per surrogate.
        unstable: Points whose timing spread was too large for the median to be quoted.
            Named rather than dropped, because a benchmark that quietly discarded its
            noisy measurements would report a precision it did not have.
        unusable_substeps: Solver settings that could not take a single step, so they have
            no accuracy to report. A step several times past what a solver's stability
            allows is not a worse setting, it is one that does not run, and the ladder
            simply stops there.
        unmeasurable: Predictors that could not take a single step either, named for the
            same reason.
    """

    schema_version: int = SPEED_SCHEMA_VERSION
    system: str = Field(min_length=1)
    split: str = Field(min_length=1)
    steps: int = Field(ge=1)
    n_initial_conditions: int = Field(ge=1)
    threads: int = Field(ge=1)
    trials: int = Field(ge=1)
    warmup: int = Field(ge=0)
    steps_per_trial: int = Field(ge=1)
    dataset_substeps: int = Field(ge=1)
    ladder: tuple[SpeedPoint, ...] = Field(min_length=1)
    surrogates: tuple[SpeedPoint, ...] = ()
    matched: tuple[MatchedSpeedup, ...] = ()
    costs: tuple[CostAccounting, ...] = ()
    unstable: tuple[str, ...] = ()
    unusable_substeps: tuple[int, ...] = ()
    unmeasurable: tuple[str, ...] = ()


def substep_ladder(substeps: int) -> tuple[int, ...]:
    """Substep counts to run the solver at, cheapest first.

    Every divisor of the count the dataset used. Divisors rather than an arbitrary
    sequence so that the top rung is the setting ground truth was generated with, whose
    error is zero by construction and which therefore anchors the ladder.

    Args:
        substeps: Solver steps per stored interval the dataset used.

    Returns:
        The ladder, ascending.

    Raises:
        ValidationError: If the count is not positive.
    """
    if substeps < 1:
        raise ValidationError(f"a substep count must be positive, got {substeps}")
    return tuple(value for value in range(1, substeps + 1) if substeps % value == 0)


def measure_point(  # noqa: PLR0913
    # What to measure, on what, how to time it and what to call it. The label and the kind
    # are the caller's because the same predictor is a rung of the ladder at one substep
    # count and a surrogate at none.
    predictor_for: Sequence[Predictor],
    cases: Sequence[EvaluationCase],
    *,
    steps: int,
    label: str,
    kind: str,
    substeps: int,
    threads: int,
    trials: int = DEFAULT_TRIALS,
    warmup: int = DEFAULT_WARMUP,
    steps_per_trial: int = DEFAULT_STEPS_PER_TRIAL,
    divergence_factor: float,
) -> SpeedPoint | None:
    """Measure one predictor's accuracy and its cost.

    Accuracy is averaged over every case and cost is timed on the first of them. Timing
    every case would multiply the cost of the benchmark by the number of initial
    conditions to measure a quantity that does not depend on which state it starts from,
    and the repeated trials already say how much the machine varies.

    Args:
        predictor_for: One predictor per case, already built. A separate instance per case
            because some predictors carry state between steps.
        cases: The initial conditions and their ground truth.
        steps: Horizon to measure accuracy over.
        label: How the point is named in a table.
        kind: `solver` or `surrogate`.
        substeps: Substeps this point folds, or zero.
        threads: Threads the caller has fixed the arithmetic to.
        trials: Timed repeats.
        warmup: Repeats discarded first.
        steps_per_trial: Steps each trial averages over.
        divergence_factor: Passed to every rollout.

    Returns:
        The point, or `None` if the predictor could not take a single step from one of the
        initial conditions. A rollout with no steps in it has no error curve, and a
        solver run far past what its stability allows is exactly that: the setting is not
        a slower or worse setting, it is one that does not run. The caller reports which
        ones those were rather than putting a number against them.

    Raises:
        ValidationError: If there are no cases, or the counts do not match.
    """
    if not cases:
        raise ValidationError("a speed point needs at least one initial condition")
    if len(predictor_for) != len(cases):
        raise ValidationError(
            f"got {len(predictor_for)} predictors for {len(cases)} cases, expected one each"
        )

    errors: list[float] = []
    completed = True
    for predictor, case in zip(predictor_for, cases, strict=True):
        horizon = min(steps, case.steps)
        result = roll_out(predictor, case.initial, horizon, divergence_factor=divergence_factor)
        completed = completed and result.completed
        if result.steps_completed < 1:
            return None
        _, curve = relative_error(
            result.trajectory, _prefix(case.reference, len(result.trajectory))
        )
        errors.append(float(np.max(curve)))

    timing = benchmark_predictor(
        predictor_for[0],
        cases[0].initial,
        steps_per_trial=min(steps_per_trial, cases[0].steps),
        trials=trials,
        warmup=warmup,
        threads=threads,
        divergence_factor=divergence_factor,
    )
    return _point(label, kind, substeps, errors, timing, completed=completed)


def solver_ladder(  # noqa: PLR0913
    # The system, the states, the horizon, the ladder to walk and how to time each rung.
    system: System,
    cases: Sequence[EvaluationCase],
    *,
    steps: int,
    ladder: Sequence[int],
    threads: int,
    trials: int = DEFAULT_TRIALS,
    warmup: int = DEFAULT_WARMUP,
    steps_per_trial: int = DEFAULT_STEPS_PER_TRIAL,
    divergence_factor: float,
) -> tuple[tuple[SpeedPoint, ...], tuple[int, ...]]:
    """Measure the reference solver at every substep count on the ladder.

    Args:
        system: The system, seen only through the protocol.
        cases: The initial conditions and their ground truth.
        steps: Horizon to measure accuracy over.
        ladder: Substep counts, cheapest first.
        threads: Threads the caller has fixed the arithmetic to.
        trials: Timed repeats per rung.
        warmup: Repeats discarded first.
        steps_per_trial: Steps each trial averages over.
        divergence_factor: Passed to every rollout.

    Returns:
        One point per rung that ran, in the order given, and the substep counts of the
        rungs that could not take a single step. A spectral solver run at a step several
        times its stability limit is one of those, and it is not a slower or a worse
        setting: it is a setting that does not run, and putting an accuracy against it
        would be inventing one.

    Raises:
        ValidationError: If the ladder is empty, names a count below one, or no rung of
            it ran at all.
    """
    if not ladder:
        raise ValidationError("a solver ladder needs at least one substep count")
    if any(substeps < 1 for substeps in ladder):
        raise ValidationError(f"every substep count must be positive, got {list(ladder)}")

    measured: list[SpeedPoint] = []
    unusable: list[int] = []
    for substeps in ladder:
        point = measure_point(
            [substepped_reference(system, case, substeps) for case in cases],
            cases,
            steps=steps,
            label=f"{REFERENCE_NAME}:substeps={substeps}",
            kind=_SOLVER,
            substeps=substeps,
            threads=threads,
            trials=trials,
            warmup=warmup,
            steps_per_trial=steps_per_trial,
            divergence_factor=divergence_factor,
        )
        if point is None:
            unusable.append(substeps)
        else:
            measured.append(point)
    if not measured:
        raise ValidationError(
            f"the reference solver took no step at any substep count in {list(ladder)}, "
            f"so there is nothing to compare a surrogate against"
        )
    return tuple(measured), tuple(unusable)


def matched_speedup(point: SpeedPoint, ladder: Sequence[SpeedPoint]) -> MatchedSpeedup:
    """Compare a surrogate against the cheapest solver setting that is as accurate.

    Args:
        point: The surrogate.
        ladder: The solver at each substep count, cheapest first.

    Returns:
        The comparison, carrying whether the surrogate's accuracy was genuinely bracketed
        by the ladder or merely bounded from one side.

    Raises:
        ValidationError: If the ladder is empty, or no rung is as accurate as the
            surrogate. The rung the data was generated with has an error of zero by
            construction, so the second cannot happen unless a ladder was assembled by
            hand.
    """
    if not ladder:
        raise ValidationError("a matched comparison needs a solver ladder")
    ordered = sorted(ladder, key=lambda rung: rung.substeps)
    qualifying = [rung for rung in ordered if rung.error <= point.error]
    if not qualifying:
        raise ValidationError(
            f"no rung of the solver ladder reaches the accuracy of {point.predictor!r}, "
            f"whose worst error is {point.error:g}"
        )
    matched = qualifying[0]
    # Bracketed means a cheaper rung was measured and was genuinely worse. Without one,
    # the solver could be run coarser than anything on the ladder and still match the
    # surrogate, so the cost being compared against is higher than the true matched cost.
    bracketed = any(rung.substeps < matched.substeps for rung in ordered)
    return MatchedSpeedup(
        predictor=point.predictor,
        error=point.error,
        seconds_per_step=point.seconds_per_step,
        matched_substeps=matched.substeps,
        matched_error=matched.error,
        matched_seconds_per_step=matched.seconds_per_step,
        speedup=_ratio(matched.seconds_per_step, point.seconds_per_step),
        bracketed=bracketed,
    )


def cost_accounting(
    matched: MatchedSpeedup,
    *,
    training_seconds: float,
    generation_seconds: float,
    steps_per_rollout: int,
) -> CostAccounting:
    """Work out how many rollouts it takes for a surrogate to repay what it cost.

    Args:
        matched: The surrogate against the solver setting it matches.
        training_seconds: Wall clock spent training it.
        generation_seconds: Wall clock its training data cost.
        steps_per_rollout: Stored intervals in one rollout.

    Returns:
        The accounting, with `NEVER_PAYS` where the surrogate is not the cheaper of the
        two and so never repays anything.

    Raises:
        ValidationError: If a cost is negative or the rollout has no steps.
    """
    if training_seconds < 0.0 or generation_seconds < 0.0:
        raise ValidationError("a one off cost cannot be negative")
    if steps_per_rollout < 1:
        raise ValidationError(f"a rollout needs at least one step, got {steps_per_rollout}")
    saving = (matched.matched_seconds_per_step - matched.seconds_per_step) * steps_per_rollout
    one_off = training_seconds + generation_seconds
    return CostAccounting(
        predictor=matched.predictor,
        training_seconds=training_seconds,
        generation_seconds=generation_seconds,
        steps_per_rollout=steps_per_rollout,
        saving_per_rollout=saving,
        break_even_rollouts=math.ceil(one_off / saving) if saving > 0.0 else NEVER_PAYS,
    )


def build_speed_report(  # noqa: PLR0913
    # The system and its states, the suite that fixes the horizon, what to measure beside
    # the solver, and the four settings that decide what a timing means.
    system: System,
    cases: Sequence[EvaluationCase],
    config: EvaluationConfig,
    *,
    dataset_substeps: int,
    seed: int,
    predictors: Sequence[str] = (),
    factories: Mapping[str, PredictorFactory] | None = None,
    training_seconds: Mapping[str, float] | None = None,
    trajectories: int = 0,
    states_per_trajectory: int = 0,
    threads: int,
    trials: int = DEFAULT_TRIALS,
    warmup: int = DEFAULT_WARMUP,
    steps_per_trial: int = DEFAULT_STEPS_PER_TRIAL,
) -> SpeedReport:
    """Measure the ladder, every surrogate asked for, and what each is worth.

    Args:
        system: The system, seen only through the protocol.
        cases: The initial conditions and their ground truth, all from one split.
        config: The suite, which fixes the horizon and the divergence factor.
        dataset_substeps: Substeps ground truth was generated with.
        seed: Root seed of the run.
        predictors: Specifications to measure beside the solver.
        factories: Predictor factories consulted before the registry, for a predictor
            that cannot be registered, such as a model loaded from a checkpoint.
        training_seconds: Wall clock spent training each named predictor. A predictor
            absent from it is accounted as having cost nothing to train, which is true of
            a baseline and would be a lie about a model.
        trajectories: Trajectories the dataset holds, for the generation cost.
        states_per_trajectory: States each one holds, for the same.
        threads: Threads the caller has fixed the arithmetic to.
        trials: Timed repeats per point.
        warmup: Repeats discarded first.
        steps_per_trial: Steps each trial averages over.

    Returns:
        The report.

    Raises:
        ValidationError: If there are no cases, or they mix splits.
        UnknownNameError: If a predictor is neither registered nor supplied.
    """
    if not cases:
        raise ValidationError("a speed report needs at least one initial condition")
    splits = {case.split for case in cases}
    if len(splits) != 1:
        raise ValidationError(f"cases mix splits {sorted(split.value for split in splits)}")

    steps = min(config.rollout_steps, *(case.steps for case in cases))
    ladder, unusable = solver_ladder(
        system,
        cases,
        steps=steps,
        ladder=substep_ladder(dataset_substeps),
        threads=threads,
        trials=trials,
        warmup=warmup,
        steps_per_trial=steps_per_trial,
        divergence_factor=config.divergence_factor,
    )

    measured: list[SpeedPoint] = []
    unmeasurable: list[str] = []
    for text in predictors:
        spec = parse_spec(text)
        point = measure_point(
            [
                build_case_predictor(
                    system,
                    case,
                    spec,
                    substeps=dataset_substeps,
                    seed=seed,
                    factories=factories,
                )
                for case in cases
            ],
            cases,
            steps=steps,
            label=spec.text,
            kind=_SURROGATE,
            substeps=0,
            threads=threads,
            trials=trials,
            warmup=warmup,
            steps_per_trial=steps_per_trial,
            divergence_factor=config.divergence_factor,
        )
        # A predictor that cannot take one step has no accuracy and no speed. Naming it
        # is the honest report; a number against it would be an invention.
        if point is None:
            unmeasurable.append(spec.text)
        else:
            measured.append(point)

    matched = tuple(matched_speedup(point, ladder) for point in measured)
    generation = _generation_seconds(ladder, dataset_substeps, trajectories, states_per_trajectory)
    costs = tuple(
        cost_accounting(
            entry,
            training_seconds=(training_seconds or {}).get(entry.predictor, 0.0),
            generation_seconds=generation,
            steps_per_rollout=steps,
        )
        for entry in matched
    )
    return SpeedReport(
        system=system.name,
        split=cases[0].split.value,
        steps=steps,
        n_initial_conditions=len(cases),
        threads=threads,
        trials=trials,
        warmup=warmup,
        steps_per_trial=steps_per_trial,
        dataset_substeps=dataset_substeps,
        ladder=ladder,
        surrogates=tuple(measured),
        matched=matched,
        costs=costs,
        unstable=tuple(point.label for point in (*ladder, *measured) if not point.stable),
        unusable_substeps=unusable,
        unmeasurable=tuple(unmeasurable),
    )


def _point(  # noqa: PLR0913
    # What the point is called, what it is, what it scored and what it cost. Assembling it
    # at the call site instead would repeat six field names in two places.
    label: str,
    kind: str,
    substeps: int,
    errors: Sequence[float],
    timing: Timing,
    *,
    completed: bool,
) -> SpeedPoint:
    """Assemble one point from what was measured of it."""
    return SpeedPoint(
        label=label,
        predictor=timing.predictor,
        kind=kind,
        substeps=substeps,
        error=float(np.mean(errors)),
        seconds_per_step=timing.seconds_per_step,
        iqr=timing.iqr,
        relative_spread=timing.relative_spread,
        stable=timing.stable,
        completed=completed,
    )


def _generation_seconds(
    ladder: Sequence[SpeedPoint], substeps: int, trajectories: int, states: int
) -> float:
    """What the dataset cost to make, measured on this machine rather than recalled.

    The wall clock generation actually took was spent across several worker processes on
    an unknown machine load, so it says as much about that day as about the work. What the
    solver costs per stored interval here, at the thread count everything else in this
    report was timed at, times the intervals the dataset holds, is the comparable number.
    """
    rung = next((point for point in ladder if point.substeps == substeps), None)
    if rung is None or trajectories < 1 or states < _MINIMUM_STATES:
        return 0.0
    return rung.seconds_per_step * trajectories * (states - 1)


def _ratio(solver: float, surrogate: float) -> float:
    """Solver cost over surrogate cost.

    A surrogate timed at zero seconds per step has not been timed, it has been found to
    be faster than the clock, and dividing by it would report an infinite speedup.
    """
    if surrogate <= 0.0:
        raise ValidationError(
            "a predictor timed at zero seconds per step cannot be compared: raise "
            "steps_per_trial until the clock can see it"
        )
    return solver / surrogate


def _prefix(trajectory: Trajectory, length: int) -> Trajectory:
    """The first `length` states of a trajectory."""
    if length >= len(trajectory):
        return trajectory
    return Trajectory(
        fields={name: array[:length] for name, array in trajectory.fields.items()},
        times=trajectory.times[:length],
    )
