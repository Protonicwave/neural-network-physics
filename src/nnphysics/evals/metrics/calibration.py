"""Calibration: whether a predictor's own uncertainty is worth believing.

A surrogate that knows when it is wrong is worth more than a surrogate that is slightly
more accurate, because the first one can hand the problem back to the solver and the
second one cannot. This metric asks two separate questions about that claim, and they can
be answered differently.

**Is the spread the right size.** Treat the predictive spread as a standard deviation and
ask how often the truth actually falls inside the interval it implies. A predictor whose
spread is a hundred times too small will have almost nothing inside its one standard
deviation band, and the expected calibration error, the average gap between the coverage
claimed and the coverage delivered over a range of levels, is the one number that says
so. The reliability curve behind it is kept, because the average hides which end is
wrong.

**Does the spread grow when the error does.** A spread of the right size on average is
still useless if it is constant, since it cannot tell a good step from a bad one. The
correlation between the spread and the error along the rollout answers that, and the two
horizons answer the practical form of it: does the spread cross the level at which a
prediction stops being worth using before the error does. That difference is the warning
a user would actually act on, and it is reported whether it is positive or negative.

Sharpness is reported beside both, because calibration alone can be bought by claiming an
enormous uncertainty. A predictor that says it might be anywhere is never caught out and
has said nothing.

Nothing here knows what a field means. The spread arrives from the rollout driver in the
same shape as the state, and a predictor that declares none is not scored: it reports
zero steps rather than a number, the same way the resolution metric does for a system
that declares no finer grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.types import MetricResult
from nnphysics.evals.metrics.base import MetricContext, normalised_magnitude, relative_error
from nnphysics.evals.metrics.error import NEVER_REACHED

if TYPE_CHECKING:
    from nnphysics.core.types import FloatArray, Rollout, Trajectory

__all__ = ["NOT_DETERMINED", "ONE_SIGMA_COVERAGE", "UNDEFINED_CORRELATION", "Calibration"]

NOT_DETERMINED = -1.0e30
"""Reported as a warning lead when there was nothing to warn about, or when no warning
came. A dedicated value rather than the sentinel the horizons use, because a lead is
legitimately negative when a predictor warns too late and a reader must be able to tell
the two apart."""

UNDEFINED_CORRELATION = -2.0
"""Reported when the spread or the error does not vary along the rollout, so no
correlation between them exists. Outside the range a correlation can take, so it cannot
be mistaken for a measurement."""

ONE_SIGMA_COVERAGE = 0.6826894921370859
"""Coverage a correctly sized Gaussian interval of one standard deviation delivers. The
number `coverage` is read against."""

_SPREAD_FLOOR = 1.0e-12
"""Relative to a field's own largest spread, below which a claimed uncertainty is treated
as a claim of certainty rather than divided by."""

_HUGE_Z = 1.0e12
"""Where a residual is divided by a spread of zero. A predictor that was certain and
wrong is infinitely overconfident, and this stands for that without producing an infinity
a result file cannot hold."""

_MINIMUM_FOR_CORRELATION = 2
"""Points needed before two curves can be said to move together at all."""


@dataclass(frozen=True, slots=True)
class Calibration:
    """How well a predictor's stated uncertainty matches the error it actually makes.

    Attributes:
        context: What the runner assembled, for the coverage levels and the level at
            which a prediction stops being worth using.
    """

    context: MetricContext = field(default_factory=MetricContext)

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return "calibration"

    def compute(self, rollout: Rollout) -> MetricResult:
        """Score a rollout against the uncertainty its predictor claimed.

        Args:
            rollout: The predicted and reference trajectories, and the spread the
                predictor declared at each step. A rollout carrying no spread is not
                scored.

        Returns:
            The expected calibration error, the coverage at one standard deviation, the
            sharpness, the correlation between spread and error, the two horizons and the
            warning lead, with the reliability curve and both curves as plot data.
        """
        steps = len(rollout) - 1
        if rollout.spread is None or steps < 1:
            return MetricResult(name=self.name, scalars={"steps": 0.0})

        residuals = _standardised(rollout.predicted, rollout.reference, rollout.spread)
        levels = np.asarray(self.context.calibration_levels, dtype=np.float64)
        empirical = _coverage(residuals, levels)
        _, error = relative_error(rollout.predicted, rollout.reference)
        spread = normalised_magnitude(rollout.spread, rollout.reference)
        times = rollout.times

        error_horizon = _first_crossing(error, times, self.context.trust_threshold)
        spread_horizon = _first_crossing(spread, times, self.context.trust_threshold)
        scalars = {
            "steps": float(steps),
            "ece": float(np.mean(np.abs(empirical - levels))),
            "coverage": float(_coverage(residuals, np.asarray([ONE_SIGMA_COVERAGE]))[0]),
            "sharpness": float(np.mean(spread[1:])),
            "spread_error_correlation": _correlation(spread[1:], error[1:]),
            "horizon.error": error_horizon,
            "horizon.spread": spread_horizon,
            "warning_lead": _lead(error_horizon, spread_horizon),
        }
        series: dict[str, FloatArray] = {
            "reliability.nominal": levels,
            "reliability.empirical": empirical,
            "spread": spread,
            "error": error,
            "time": times,
        }
        return MetricResult(
            name=self.name,
            scalars=scalars,
            series=series,
            sentinels={
                "horizon.error": NEVER_REACHED,
                "horizon.spread": NEVER_REACHED,
                "warning_lead": NOT_DETERMINED,
                "spread_error_correlation": UNDEFINED_CORRELATION,
            },
        )


def _standardised(predicted: Trajectory, reference: Trajectory, spread: Trajectory) -> FloatArray:
    """Every residual of the rollout divided by the uncertainty claimed for it.

    The initial state is dropped: it was handed to the predictor rather than produced by
    it, so its residual and its spread are both zero and counting it would be counting a
    perfect prediction the predictor never made.

    Elements where the predictor claimed nothing and was right are dropped too. A field
    carried unchanged from the input, a mass that no model predicts, has a residual of
    zero and a spread of zero, and it is neither a success nor a failure of calibration.
    """
    collected: list[FloatArray] = []
    for name in sorted(reference.names):
        residual = np.abs(predicted.fields[name][1:] - reference.fields[name][1:])
        sigma = spread.fields[name][1:]
        floor = _SPREAD_FLOOR * float(np.max(sigma)) if sigma.size else 0.0
        claimed = sigma > floor
        standardised = np.where(claimed, residual / np.where(claimed, sigma, 1.0), _HUGE_Z)
        collected.append(np.where(claimed | (residual > 0.0), standardised, np.nan).ravel())
    stacked = np.concatenate(collected) if collected else np.zeros(0)
    kept: FloatArray = stacked[np.isfinite(stacked)]
    return kept


def _coverage(residuals: FloatArray, levels: FloatArray) -> FloatArray:
    """Fraction of standardised residuals inside the interval each level implies.

    A Gaussian is assumed, which is what a mean and a standard deviation amount to
    claiming. The claim being wrong in shape rather than in size is a limit of the
    measure worth knowing, and it is the same limit every reliability diagram has.
    """
    if residuals.size == 0:
        return np.zeros_like(levels)
    normal = NormalDist()
    widths = np.asarray(
        [normal.inv_cdf(0.5 * (1.0 + float(level))) for level in levels], dtype=np.float64
    )
    inside: FloatArray = np.mean(residuals[None, :] <= widths[:, None], axis=1)
    return inside


def _correlation(spread: FloatArray, error: FloatArray) -> float:
    """Pearson correlation between two curves, or `UNDEFINED_CORRELATION`."""
    if (
        spread.size < _MINIMUM_FOR_CORRELATION
        or float(np.std(spread)) == 0.0
        or float(np.std(error)) == 0.0
    ):
        return UNDEFINED_CORRELATION
    return float(np.corrcoef(spread, error)[0, 1])


def _first_crossing(values: FloatArray, times: FloatArray, threshold: float) -> float:
    """Time at which a curve first exceeds a threshold, or `NEVER_REACHED`."""
    exceeded = np.flatnonzero(values > threshold)
    if exceeded.size == 0:
        return NEVER_REACHED
    return float(times[exceeded[0]] - times[0])


def _lead(error_horizon: float, spread_horizon: float) -> float:
    """How long before the error became unacceptable the spread said so.

    Only defined when both happened. An error that never became unacceptable left nothing
    to warn about, and a spread that never grew gave no warning at all; the two horizons
    say which of those it was, and averaging either into a lead would hide it.
    """
    if NEVER_REACHED in (error_horizon, spread_horizon):
        return NOT_DETERMINED
    return error_horizon - spread_horizon
