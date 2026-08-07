from __future__ import annotations

import numpy as np
import pytest

from nnphysics.core.types import Rollout, Trajectory
from nnphysics.evals.metrics import NEVER_REACHED, MetricContext
from nnphysics.evals.metrics.calibration import (
    NOT_DETERMINED,
    ONE_SIGMA_COVERAGE,
    UNDEFINED_CORRELATION,
    Calibration,
)

STEPS = 64
ELEMENTS = 24
"""Enough residuals per step that a coverage fraction is a measurement rather than a
count of a handful of draws."""

OVERCONFIDENCE = 100.0
"""What the deliberately broken fixture divides its stated uncertainty by."""


def _truth(rng: np.random.Generator) -> Trajectory:
    """A wandering signal of unit scale, so a spread can be normalised against it."""
    walk = np.cumsum(rng.standard_normal((STEPS, ELEMENTS)), axis=0)
    return Trajectory(fields={"q": walk + 10.0}, times=np.arange(STEPS, dtype=np.float64) * 0.1)


def _rollout(
    truth: Trajectory, rng: np.random.Generator, *, confidence: float = 1.0, growth: bool = True
) -> Rollout:
    """A predictor that adds noise of a known size and states a spread for it.

    With `confidence` of one the stated spread is exactly the standard deviation of the
    noise added, so the claim is true by construction. Dividing it is the overconfident
    fixture. Turning off `growth` states a spread that never changes, which is a claim of
    the right size on average and of no use at any particular step.
    """
    scale = 0.02 * float(np.mean(np.abs(truth.fields["q"])))
    steps = np.arange(STEPS, dtype=np.float64)
    sigma = scale * (np.sqrt(steps) if growth else np.full(STEPS, np.sqrt(STEPS / 2.0)))
    actual = np.repeat(sigma[:, None], ELEMENTS, axis=1)
    noise = rng.standard_normal((STEPS, ELEMENTS)) * actual
    noise[0] = 0.0
    spread = actual / confidence
    spread[0] = 0.0
    return Rollout(
        predicted=Trajectory(fields={"q": truth.fields["q"] + noise}, times=truth.times),
        reference=truth,
        predictor="fixture",
        system="test",
        spread=Trajectory(fields={"q": spread}, times=truth.times),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260806)


@pytest.fixture
def truth(rng: np.random.Generator) -> Trajectory:
    return _truth(rng)


class TestTheHonestClaimIsNotFlagged:
    """Half of a sentinel: the metric must stay quiet about a claim that is true."""

    def test_its_calibration_error_is_small(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        scored = Calibration().compute(_rollout(truth, rng))

        assert scored.scalars["ece"] < 0.05

    def test_its_coverage_is_close_to_what_a_gaussian_delivers(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        scored = Calibration().compute(_rollout(truth, rng))

        assert abs(scored.scalars["coverage"] - ONE_SIGMA_COVERAGE) < 0.05

    def test_its_reliability_curve_tracks_the_diagonal(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        scored = Calibration().compute(_rollout(truth, rng))
        nominal = scored.series["reliability.nominal"]
        empirical = scored.series["reliability.empirical"]

        assert np.max(np.abs(np.asarray(empirical) - np.asarray(nominal))) < 0.1


class TestTheOverconfidentClaimIsCaught:
    """The other half, and the fixture the phase brief names."""

    def test_its_calibration_error_is_enormous_beside_the_honest_one(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        honest = Calibration().compute(_rollout(truth, rng))
        broken = Calibration().compute(_rollout(truth, rng, confidence=OVERCONFIDENCE))

        assert broken.scalars["ece"] > 0.4
        assert broken.scalars["ece"] > honest.scalars["ece"] * 10.0

    def test_almost_nothing_falls_inside_the_interval_it_claims(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        scored = Calibration().compute(_rollout(truth, rng, confidence=OVERCONFIDENCE))

        assert scored.scalars["coverage"] < 0.05

    def test_its_reliability_curve_sits_below_the_diagonal_everywhere(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        """Below, not merely away from it. Overconfidence has a direction."""
        scored = Calibration().compute(_rollout(truth, rng, confidence=OVERCONFIDENCE))
        nominal = np.asarray(scored.series["reliability.nominal"])
        empirical = np.asarray(scored.series["reliability.empirical"])

        assert np.all(empirical < nominal)

    def test_it_is_sharper_than_the_honest_one_which_is_the_whole_trick(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        """Sharpness alone would rank the broken fixture first, which is why it is not alone."""
        honest = Calibration().compute(_rollout(truth, rng))
        broken = Calibration().compute(_rollout(truth, rng, confidence=OVERCONFIDENCE))

        assert broken.scalars["sharpness"] < honest.scalars["sharpness"]


class TestUnderconfidenceIsCaughtToo:
    def test_a_spread_a_hundred_times_too_large_is_also_flagged(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        """A metric that only caught overconfidence would be measuring sharpness."""
        scored = Calibration().compute(_rollout(truth, rng, confidence=1.0 / OVERCONFIDENCE))

        assert scored.scalars["ece"] > 0.4
        assert scored.scalars["coverage"] > 0.99


class TestSpreadAgainstError:
    def test_a_growing_spread_tracks_a_growing_error(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        scored = Calibration().compute(_rollout(truth, rng))

        assert scored.scalars["spread_error_correlation"] > 0.5

    def test_a_constant_spread_is_reported_as_no_correlation(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        """A flat claim tracks the error worse than a growing one.

        The spread is constant in the field's own units, so against a wandering signal its
        normalised value still moves a little. What is asserted is the comparison rather
        than a number.
        """
        growing = Calibration().compute(_rollout(truth, rng))
        flat = Calibration().compute(_rollout(truth, rng, growth=False))

        assert (
            growing.scalars["spread_error_correlation"] > flat.scalars["spread_error_correlation"]
        )

    def test_a_spread_that_never_varies_at_all_is_undefined_rather_than_zero(self) -> None:
        times = np.arange(8, dtype=np.float64)
        truth = Trajectory(fields={"q": np.ones((8, 4))}, times=times)
        rollout = Rollout(
            predicted=Trajectory(fields={"q": np.full((8, 4), 1.5)}, times=times),
            reference=truth,
            predictor="flat",
            system="test",
            spread=Trajectory(fields={"q": np.full((8, 4), 0.5)}, times=times),
        )

        scored = Calibration().compute(rollout)

        assert scored.scalars["spread_error_correlation"] == UNDEFINED_CORRELATION


class TestTheWarning:
    def test_an_honest_spread_crosses_the_threshold_and_a_lead_is_reported(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        scored = Calibration().compute(_rollout(truth, rng))

        assert scored.scalars["horizon.error"] != NEVER_REACHED
        assert scored.scalars["horizon.spread"] != NEVER_REACHED
        assert scored.scalars["warning_lead"] != NOT_DETERMINED

    def test_the_overconfident_one_fails_silently_and_the_two_horizons_say_so(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        """The reading that matters: the error became unacceptable and nothing warned."""
        scored = Calibration().compute(_rollout(truth, rng, confidence=OVERCONFIDENCE))

        assert scored.scalars["horizon.error"] != NEVER_REACHED
        assert scored.scalars["horizon.spread"] == NEVER_REACHED
        assert scored.scalars["warning_lead"] == NOT_DETERMINED

    def test_the_threshold_is_the_one_the_context_was_given(
        self, truth: Trajectory, rng: np.random.Generator
    ) -> None:
        tight = Calibration(MetricContext(trust_threshold=0.005)).compute(_rollout(truth, rng))
        loose = Calibration(MetricContext(trust_threshold=0.05)).compute(_rollout(truth, rng))

        assert tight.scalars["horizon.error"] != NEVER_REACHED
        assert loose.scalars["horizon.error"] != NEVER_REACHED
        assert tight.scalars["horizon.error"] < loose.scalars["horizon.error"]


class TestWhatIsNotScored:
    def test_a_rollout_with_no_spread_reports_no_steps_rather_than_a_number(
        self, truth: Trajectory
    ) -> None:
        """A predictor that states nothing has made no claim to be wrong about."""
        rollout = Rollout(
            predicted=truth, reference=truth, predictor="silent", system="test", spread=None
        )

        scored = Calibration().compute(rollout)

        assert scored.scalars == {"steps": 0.0}

    def test_an_element_the_predictor_copied_exactly_is_not_counted(self) -> None:
        """A carried field is neither a pass nor a failure of calibration.

        It has no error and claims no spread. Counting it would let a model with one
        constant field buy any coverage it liked.
        """
        times = np.arange(4, dtype=np.float64)
        moving = np.arange(4, dtype=np.float64).reshape(4, 1) * np.ones((1, 6))
        static = np.ones((4, 6))
        rollout = Rollout(
            predicted=Trajectory(fields={"q": moving + 0.5, "m": static}, times=times),
            reference=Trajectory(fields={"q": moving, "m": static}, times=times),
            predictor="carrier",
            system="test",
            spread=Trajectory(
                fields={"q": np.full((4, 6), 0.5), "m": np.zeros((4, 6))}, times=times
            ),
        )

        scored = Calibration().compute(rollout)

        # Every counted residual is exactly one stated standard deviation, so the coverage
        # at one sigma is total. Had the carried field been counted it would be total for
        # a different reason, and the two would be indistinguishable.
        assert scored.scalars["coverage"] == 1.0
        assert scored.scalars["steps"] == 3.0

    def test_being_certain_and_wrong_is_counted_rather_than_skipped(self) -> None:
        """The one case a spread of zero must not be excused."""
        times = np.arange(4, dtype=np.float64)
        truth = Trajectory(fields={"q": np.ones((4, 6))}, times=times)
        rollout = Rollout(
            predicted=Trajectory(fields={"q": np.full((4, 6), 5.0)}, times=times),
            reference=truth,
            predictor="certain",
            system="test",
            spread=Trajectory(fields={"q": np.zeros((4, 6))}, times=times),
        )

        scored = Calibration().compute(rollout)

        assert scored.scalars["coverage"] == 0.0
        assert scored.scalars["ece"] > 0.4
