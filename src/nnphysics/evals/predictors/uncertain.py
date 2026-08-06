"""Predictors that make a claim about their own uncertainty, one of them dishonestly.

The calibration metric measures a claim, so it needs a claim it already knows the answer
for. These two are that, and they are permanent fixtures for the same reason the broken
predictors are: an expected calibration error of 0.3 means nothing until it sits beside
the number a predictor that is telling the truth produces.

Both add Gaussian noise to the reference solver at every step, which is the error shape a
surrogate makes. What they differ in is what they say about it. The honest one states the
deviation that much noise accumulates to, which grows as the square root of the number of
steps taken because independent perturbations add in quadrature. The overconfident one
states a fraction of that, which is the failure mode the phase brief names: a model whose
ensemble members agree with each other and are all wrong together.

The square root law is a claim about a system that neither amplifies nor damps a
perturbation, and neither supported system is quite that. That is deliberate. The honest
fixture is honest about its noise, not about the dynamics, and how far the metric still
rates it as calibrated on a chaotic cluster against a decaying flow is a fact about those
systems worth reading rather than one worth engineering away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.core.types import Prediction, State

if TYPE_CHECKING:
    from nnphysics.core.protocols import Predictor
    from nnphysics.core.types import FloatArray

__all__ = ["CALIBRATED_NAME", "OVERCONFIDENT_NAME", "UncertainNoise"]

CALIBRATED_NAME = "calibrated"
"""The fixture that states the uncertainty it actually has."""

OVERCONFIDENT_NAME = "overconfident"
"""The fixture that states a fraction of it."""

_SCALE_FLOOR = 1.0e-30
"""Keeps a field of all zeros from making a relative perturbation undefined."""

_TIME_RTOL = 1.0e-9
_TIME_ATOL = 1.0e-12


class UncertainNoise:
    """The true solver with Gaussian noise, and a stated uncertainty about the result.

    Like `LinearExtrapolation` this carries state between calls, because the deviation it
    claims depends on how many steps of this rollout have been taken. A rollout is
    detected as new when the state handed in does not follow the last one returned, so the
    same instance may be used for several rollouts.

    Args:
        inner: The solver the noise is added to.
        scale: Noise standard deviation as a fraction of a field's magnitude, per step.
        rng: Generator every draw comes from.
        confidence: What the stated spread is divided by. One states the deviation the
            noise actually accumulates; a hundred states a hundredth of it; a tenth
            overstates it tenfold.
        name: Registered name, so the same class serves both fixtures.

    Raises:
        ValidationError: If the scale is negative or the confidence is not positive.
    """

    def __init__(
        # The solver, what is done to it, what seeds that, how the claim is distorted and
        # what the fixture is called. Only the last two differ between the two fixtures.
        self,
        inner: Predictor,
        scale: float,
        rng: np.random.Generator,
        confidence: float,
        name: str,
    ) -> None:
        if scale < 0.0:
            raise ValidationError(f"noise scale must not be negative, got {scale}")
        if confidence <= 0.0:
            raise ValidationError(f"confidence must be positive, got {confidence}")
        self._inner = inner
        self._scale = scale
        self._rng = rng
        self._confidence = confidence
        self._name = name
        self._taken = 0
        self._expected_time: float | None = None

    @property
    def name(self) -> str:
        """Identifier used in configuration and in reports."""
        return self._name

    @property
    def dt(self) -> float:
        """Size of the step this predictor advances by."""
        return self._inner.dt

    def reset(self) -> None:
        """Forget how far the current rollout has gone."""
        self._taken = 0
        self._expected_time = None

    def predict(self, state: State) -> Prediction:
        """Advance one step, corrupt the result and say how far off it probably is.

        Args:
            state: The current state.

        Returns:
            The noisy state and the deviation the noise has accumulated to by this step.
        """
        if self._expected_time is None or not np.isclose(
            state.time, self._expected_time, rtol=_TIME_RTOL, atol=_TIME_ATOL
        ):
            self.reset()
        stepped = self._inner.step(state)
        self._taken += 1
        self._expected_time = stepped.time

        fields: dict[str, FloatArray] = {}
        spread: dict[str, FloatArray] = {}
        for field_name, array in stepped.fields.items():
            deviation = self._scale * _magnitude(array)
            fields[field_name] = array + deviation * self._draw(array.shape)
            # Independent perturbations add in quadrature, so the deviation after n steps
            # is the per step one times the square root of n. The stated claim is that,
            # divided by however overconfident this fixture is.
            claimed = deviation * np.sqrt(self._taken) / self._confidence
            spread[field_name] = np.full(array.shape, claimed, dtype=np.float64)
        return Prediction(state=State(fields=fields, time=stepped.time), spread=spread)

    def step(self, state: State) -> State:
        """Advance one step, discarding the uncertainty.

        The `Predictor` half of the interface. It draws the same noise `predict` would,
        because it is the same call: a fixture whose two entry points produced different
        states would be scored on a trajectory it never claimed to make.

        Args:
            state: The current state.

        Returns:
            The noisy state.
        """
        return self.predict(state).state

    def _draw(self, shape: tuple[int, ...]) -> FloatArray:
        """Standard normal noise with its own mean removed.

        The mean is removed for the reason `NoiseInjection` removes it: a system may
        constrain a field's mean, and a fixture whose first step produced a state the
        system refuses would be testing the harness rather than the metric.
        """
        noise: FloatArray = self._rng.standard_normal(shape)
        return noise - np.mean(noise)


def _magnitude(array: FloatArray) -> float:
    """Root mean square of a field, floored so a field of zeros stays well defined."""
    return max(float(np.sqrt(np.mean(array**2))), _SCALE_FLOOR)
