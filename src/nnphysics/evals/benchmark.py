"""Timing a predictor fairly.

A speedup claim is the one number this whole repository exists to produce, and it is the
easiest number to get wrong. Four things are done here to make it mean something.

**Warmup runs are discarded.** The first call into a network allocates, dispatches and
fills a cache, and a solver's first transform plans one. None of that is paid again on
the millionth rollout, so counting it would measure the import rather than the model.

**The median is reported, not the best.** A best of ten on a laptop is a measurement of
how quiet the machine happened to be for a moment. The median is what a user would
actually experience, and the interquartile range beside it says how much to trust it.

**The spread is reported at all.** A timing without a spread is not a measurement, and
this is a shared laptop rather than a cluster. A benchmark whose interquartile range is a
third of its median has not measured anything and the number should say so rather than
being quoted.

**The thread count is recorded.** Every timing here is a function of how many cores the
arithmetic was allowed to use, and comparing a solver on eight threads against a network
on one would be comparing the thread counts.

Nothing here knows what a predictor is beyond the protocol, which is what lets the same
function time a spectral solver, a neural operator and a baseline that returns its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.evals.rollout import DEFAULT_DIVERGENCE_FACTOR, roll_out

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor
    from nnphysics.core.types import FloatArray, State

__all__ = [
    "DEFAULT_STEPS_PER_TRIAL",
    "DEFAULT_TRIALS",
    "DEFAULT_WARMUP",
    "STABILITY_LIMIT",
    "Timing",
    "benchmark_predictor",
]

DEFAULT_TRIALS = 7
"""Timed repeats. Odd, so the median is a measurement rather than an average of two, and
small enough that timing six predictors at several settings each stays affordable."""

DEFAULT_WARMUP = 2
"""Repeats run and thrown away first. One would be enough for allocation; two also covers
a library that plans a transform on its second distinct shape."""

DEFAULT_STEPS_PER_TRIAL = 16
"""Steps averaged over within one trial. Long enough that the clock's own resolution
disappears against the total, short enough that a predictor which diverges still
completes trials to be timed on."""

STABILITY_LIMIT = 0.1
"""Interquartile range, as a fraction of the median, above which a timing is called
unstable. A tenth is loose for a dedicated machine and tight for a laptop with a browser
open, which is the machine this is measured on."""

_QUARTILES = (25.0, 75.0)


@dataclass(frozen=True, slots=True)
class Timing:
    """What one predictor cost, and how sure of it we are.

    Attributes:
        predictor: Name of what was timed.
        seconds_per_step: Median wall clock to advance one stored interval.
        iqr: Interquartile range of the same quantity over the trials.
        fastest: Quickest trial, kept only so a reader can see how far the median sits
            above it.
        slowest: Slowest trial.
        trials: Timed repeats.
        warmup: Repeats discarded before timing began.
        steps_per_trial: Steps each trial averaged over.
        threads: Threads the arithmetic was allowed, as the caller fixed them.
        batch_size: States advanced per call. One everywhere today, recorded because a
            surrogate that only wins when batched must say so.
    """

    predictor: str
    seconds_per_step: float
    iqr: float
    fastest: float
    slowest: float
    trials: int
    warmup: int
    steps_per_trial: int
    threads: int
    batch_size: int = 1

    @property
    def relative_spread(self) -> float:
        """Interquartile range as a fraction of the median, or zero for a free predictor."""
        return self.iqr / self.seconds_per_step if self.seconds_per_step > 0.0 else 0.0

    @property
    def stable(self) -> bool:
        """Whether the spread is small enough for the median to be worth quoting."""
        return self.relative_spread <= STABILITY_LIMIT


def benchmark_predictor(  # noqa: PLR0913
    # What to time, what to time it on, and the four knobs that decide whether the number
    # means anything. They are named at the call site rather than ordered, because a
    # benchmark whose trial count was passed positionally is a benchmark nobody can read.
    predictor: Predictor,
    initial: State,
    *,
    steps_per_trial: int = DEFAULT_STEPS_PER_TRIAL,
    trials: int = DEFAULT_TRIALS,
    warmup: int = DEFAULT_WARMUP,
    threads: int = 1,
    divergence_factor: float = DEFAULT_DIVERGENCE_FACTOR,
) -> Timing:
    """Time one predictor over repeated rollouts from the same initial state.

    Every trial starts from the same state, so the trials differ only in what the machine
    was doing. Only the stepping is timed: the rollout driver excludes the checks it makes
    around each step, and this excludes everything outside the driver.

    Args:
        predictor: What to time.
        initial: State every trial starts from.
        steps_per_trial: Steps each trial takes, which the total is divided by.
        trials: Timed repeats, at least one.
        warmup: Repeats run and discarded first.
        threads: Threads the caller has fixed the arithmetic to, recorded rather than set.
            Setting it here would mean this module reaching into a library it does not
            otherwise depend on, and a thread count set halfway through a process is not
            reliably honoured anyway.
        divergence_factor: Passed to each rollout.

    Returns:
        The timing.

    Raises:
        ValidationError: If fewer than one trial or one step was asked for, the warmup is
            negative, the thread count is not positive, or the predictor cannot complete a
            single step of a trial.
    """
    if trials < 1:
        raise ValidationError(f"a benchmark needs at least one trial, got {trials}")
    if steps_per_trial < 1:
        raise ValidationError(f"a trial needs at least one step, got {steps_per_trial}")
    if warmup < 0:
        raise ValidationError(f"the warmup count must not be negative, got {warmup}")
    if threads < 1:
        raise ValidationError(f"the thread count must be positive, got {threads}")

    for _ in range(warmup):
        roll_out(predictor, initial, steps_per_trial, divergence_factor=divergence_factor)

    measured: list[float] = []
    for _ in range(trials):
        result = roll_out(predictor, initial, steps_per_trial, divergence_factor=divergence_factor)
        # A predictor that stops early is still timed, over the steps it managed. What is
        # not acceptable is a trial with nothing in it: dividing by zero steps would give
        # a predictor that fails immediately the best time in the table.
        if result.steps_completed < 1:
            raise ValidationError(
                f"predictor {predictor.name!r} completed no step to time: "
                f"{result.stop_reason.value}{f', {result.detail}' if result.detail else ''}"
            )
        measured.append(result.seconds / result.steps_completed)

    samples: FloatArray = np.asarray(measured, dtype=np.float64)
    low, high = (float(value) for value in np.percentile(samples, _QUARTILES))
    return Timing(
        predictor=predictor.name,
        seconds_per_step=float(np.median(samples)),
        iqr=high - low,
        fastest=float(np.min(samples)),
        slowest=float(np.max(samples)),
        trials=trials,
        warmup=warmup,
        steps_per_trial=steps_per_trial,
        threads=threads,
    )
