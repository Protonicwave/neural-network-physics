"""The core deliverable of phase 05.

Every metric here is asked to do two things: flag the predictors that are broken in the
way it exists to notice, and stay quiet about the reference solver and about predictors
whose faults lie elsewhere. The second half matters as much as the first, since a metric
that flags everything is not measuring anything either.

Each test runs against both systems unless it is making a point about one of them.
Nothing in this file mentions a position, a vorticity or a viscosity.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nnphysics.core.types import MetricResult
from nnphysics.evals.metrics import NEVER_REACHED, ONE_SIGMA_COVERAGE

Score = Callable[..., dict[str, MetricResult]]

STEPS = 32
"""Long enough that a fault which only shows over a rollout has shown, short enough that
the whole file runs in the default test run."""

EARLY = 4
"""Where a predictor that extrapolates is still better than one that copies."""

BROKEN = ("persistence", "linear_extrapolation", "noise", "energy_injection", "symmetry_break")


def _growth(scored: dict[str, MetricResult]) -> float:
    """How many times the error at the end of a rollout exceeds the error early in it."""
    curve = scored["rollout_error"].series["error"]
    return float(curve[-1]) / max(float(curve[EARLY]), 1e-15)


def _is_later(horizon: float, other: float) -> bool:
    """Whether one horizon comes after another, with `NEVER_REACHED` counting as latest."""
    if horizon == NEVER_REACHED:
        return True
    return other != NEVER_REACHED and horizon > other


class TestTheReferenceIsBenign:
    """Ground truth scored against itself must score clean on every metric."""

    def test_every_metric_returns_a_benign_value(self, score: Score) -> None:
        scored = score("reference", steps=STEPS)

        assert scored["one_step_error"].scalars["error"] == 0.0
        assert scored["rollout_error"].scalars["error.final"] == 0.0
        assert scored["invariant_drift"].scalars["worst_violation"] == 0.0
        assert scored["distribution_drift"].scalars["worst"] == 0.0
        # Equivariance is the one that cannot be exactly zero: it compares two separate
        # rollouts, and the two differ by the order the same arithmetic was done in.
        assert scored["symmetry_violation"].scalars["worst"] < 1e-9

    def test_no_error_horizon_is_ever_reached(self, score: Score) -> None:
        scalars = score("reference", steps=STEPS, metrics=["rollout_error"])[
            "rollout_error"
        ].scalars

        assert all(value == -1.0 for key, value in scalars.items() if key.startswith("horizon."))


class TestInvariantDriftAgainstOneStepError:
    """The contrast the harness exists for, asserted in one test because it is the point."""

    def test_energy_injection_is_invisible_to_one_step_error_and_obvious_to_drift(
        self, score: Score
    ) -> None:
        injected = score("energy_injection", steps=STEPS)
        persistence = score("persistence", steps=STEPS)

        one_step = injected["one_step_error"].scalars["error"]
        assert one_step < 1e-2
        assert one_step < persistence["one_step_error"].scalars["error"]
        # Error alone stays flattering over the whole rollout, too. Only the invariant
        # notices that the system being predicted is no longer the one it started as.
        assert injected["rollout_error"].scalars["error.final"] < 0.05

        assert injected["invariant_drift"].scalars["worst_violation"] > 1.0

    def test_the_injected_quantity_ends_up_above_the_true_one(self, score: Score) -> None:
        """The sign is half the statement: energy was added, not merely lost track of."""
        scalars = score("energy_injection", steps=STEPS, metrics=["invariant_drift"])[
            "invariant_drift"
        ].scalars

        excesses = {key: value for key, value in scalars.items() if key.endswith(".excess")}
        assert excesses
        assert max(excesses.values()) > 0.0


class TestInvariantDriftStaysQuiet:
    def test_persistence_conserves_everything_a_closed_system_conserves(
        self, score_nbody: Score
    ) -> None:
        """Worth asserting because it is the metric's most useful blind spot.

        A predictor that returns its input holds every conserved quantity perfectly still,
        so invariant drift rates it flawless while its error is enormous. No single metric
        is enough, and this is the cleanest demonstration of why.
        """
        scored = score_nbody("persistence", steps=STEPS)

        assert scored["invariant_drift"].scalars["worst_violation"] <= 1.0
        assert scored["rollout_error"].scalars["error.final"] > 0.1

    def test_persistence_does_not_get_away_with_it_in_a_dissipative_system(
        self, score_fluid: Score
    ) -> None:
        """The same blind spot closes when a quantity is declared decaying.

        A viscous flow must lose energy. Holding it still is as unphysical as adding it,
        and the declaration is what lets one metric say so without knowing what viscosity
        is.
        """
        scored = score_fluid("persistence", steps=STEPS)

        assert scored["invariant_drift"].scalars["worst_violation"] > 1.0

    def test_a_broken_symmetry_does_not_show_up_as_drift(self, score: Score) -> None:
        """A rotation each step conserves energy exactly, so only one metric sees it."""
        scored = score("symmetry_break", steps=STEPS)

        assert scored["invariant_drift"].scalars["worst_violation"] <= 1.0
        assert scored["symmetry_violation"].scalars["worst"] > 1e-3


class TestSymmetryViolation:
    def test_the_rotating_predictor_is_caught(self, score: Score) -> None:
        broken = score("symmetry_break", steps=STEPS, metrics=["symmetry_violation"])
        reference = score("reference", steps=STEPS, metrics=["symmetry_violation"])

        assert broken["symmetry_violation"].scalars["worst"] > 1e-3
        assert (
            broken["symmetry_violation"].scalars["worst"]
            > reference["symmetry_violation"].scalars["worst"] * 1e6
        )

    def test_it_is_caught_under_a_symmetry_it_did_not_break(self, score: Score) -> None:
        """The whole reason every declared symmetry is tested.

        The transformation this predictor applies commutes with the dynamics, so it is
        perfectly equivariant under itself. Only the symmetries it does not commute with
        reveal it, and a metric testing a single symmetry could be handed the wrong one.
        """
        scalars = score("symmetry_break", steps=STEPS, metrics=["symmetry_violation"])[
            "symmetry_violation"
        ].scalars

        per_symmetry = {
            key.removesuffix(".max"): value
            for key, value in scalars.items()
            if key.endswith(".max")
        }
        assert len(per_symmetry) > 1
        assert min(per_symmetry.values()) < 1e-6
        assert max(per_symmetry.values()) > 1e-3

    def test_persistence_is_ranked_far_below_the_symmetry_breaker(self, score: Score) -> None:
        """The metric has to separate the two, not merely fire on both."""
        persistence = score("persistence", steps=STEPS, metrics=["symmetry_violation"])
        broken = score("symmetry_break", steps=STEPS, metrics=["symmetry_violation"])

        assert (
            persistence["symmetry_violation"].scalars["worst"]
            < broken["symmetry_violation"].scalars["worst"] / 20.0
        )

    def test_persistence_is_exact_under_a_symmetry_that_does_not_involve_the_clock(
        self, score: Score
    ) -> None:
        scalars = score("persistence", steps=STEPS, metrics=["symmetry_violation"])[
            "symmetry_violation"
        ].scalars

        violations = [value for key, value in scalars.items() if key.endswith(".max")]
        assert min(violations) < 1e-12

    def test_persistence_is_not_exact_under_one_that_does(self, score_nbody: Score) -> None:
        """Worth pinning down, because it is easy to assume a frozen state is equivariant.

        A Galilean boost is a transformation of position by velocity times time, so undoing
        it at a later instant does not undo what applying it did at an earlier one. A
        predictor that advances the clock without moving anything therefore fails it, and
        the metric is right to say so. The N-body system is named here because it is the
        one that declares a time dependent symmetry.
        """
        scalars = score_nbody("persistence", steps=STEPS, metrics=["symmetry_violation"])[
            "symmetry_violation"
        ].scalars

        assert scalars["galilean_boost.max"] > 1e-3
        assert scalars["translation.max"] < 1e-12


class TestRolloutErrorGrowth:
    def test_extrapolation_beats_persistence_early(self, score: Score) -> None:
        """A straight line through two states carries the rate of change; a copy does not."""
        extrapolation = score("linear_extrapolation", steps=EARLY, metrics=["rollout_error"])
        persistence = score("persistence", steps=EARLY, metrics=["rollout_error"])

        assert (
            extrapolation["rollout_error"].scalars["error.final"]
            < persistence["rollout_error"].scalars["error.final"]
        )

    def test_extrapolation_degrades_faster_than_persistence(self, score: Score) -> None:
        """And nothing bends the line back, so it overshoots where a copy merely lags.

        Which of the two is worse at the end of a rollout depends on the system: a flow
        that decays smoothly is extrapolated well for a long time, while a cluster that
        collapses is not. What does not depend on the system is the shape of the two
        curves, one starting low and steepening, the other starting high and flattening.
        A metric reporting error at a single horizon would rank the pair either way
        depending on which horizon it picked, which is the reason the curve is the result
        and the summary numbers are the summary.
        """
        extrapolation = score("linear_extrapolation", steps=STEPS, metrics=["rollout_error"])
        persistence = score("persistence", steps=STEPS, metrics=["rollout_error"])

        assert _growth(extrapolation) > _growth(persistence)

    def test_the_horizon_summary_orders_them_the_same_way(self, score: Score) -> None:
        """Extrapolation reaches a given error later than persistence, or never reaches it."""
        extrapolation = score("linear_extrapolation", steps=STEPS, metrics=["rollout_error"])
        persistence = score("persistence", steps=STEPS, metrics=["rollout_error"])

        assert _is_later(
            extrapolation["rollout_error"].scalars["horizon.0.01"],
            persistence["rollout_error"].scalars["horizon.0.01"],
        )

    def test_error_grows_with_horizon(self, score: Score) -> None:
        curve = score("noise", steps=STEPS, metrics=["rollout_error"])["rollout_error"].series[
            "error"
        ]

        assert curve[0] == 0.0
        assert curve[-1] > curve[1]


class TestDistributionDrift:
    def test_noise_is_caught_and_the_reference_is_not(self, score: Score) -> None:
        noisy = score("noise:scale=0.05", steps=STEPS, metrics=["distribution_drift"])
        reference = score("reference", steps=STEPS, metrics=["distribution_drift"])

        assert reference["distribution_drift"].scalars["worst"] == 0.0
        assert noisy["distribution_drift"].scalars["worst"] > 0.0

    def test_more_noise_drifts_further(self, score: Score) -> None:
        """The measure has to respond to the fault, not merely be non zero in its presence."""
        loud = score("noise:scale=0.05", steps=STEPS, metrics=["distribution_drift"])
        quiet = score("noise:scale=0.005", steps=STEPS, metrics=["distribution_drift"])

        assert (
            loud["distribution_drift"].scalars["worst"]
            > quiet["distribution_drift"].scalars["worst"]
        )

    def test_a_stable_but_wrong_predictor_is_caught(self, score_nbody: Score) -> None:
        """What the metric is for: persistence never diverges, and never stops being wrong."""
        scored = score_nbody("persistence", steps=STEPS, metrics=["distribution_drift"])

        assert scored["distribution_drift"].scalars["worst"] > 0.01


class TestCalibration:
    """The two fixtures differ only in what they claim.

    So the metric can only be measuring the claim.
    """

    def test_the_overconfident_predictor_is_caught_and_the_honest_one_is_not(
        self, score: Score
    ) -> None:
        honest = score("calibrated", steps=STEPS, metrics=["calibration"])
        broken = score("overconfident", steps=STEPS, metrics=["calibration"])

        assert broken["calibration"].scalars["ece"] > 0.4
        assert honest["calibration"].scalars["ece"] < broken["calibration"].scalars["ece"] / 2.0

    def test_the_overconfident_predictor_covers_almost_nothing_it_claims(
        self, score: Score
    ) -> None:
        scored = score("overconfident", steps=STEPS, metrics=["calibration"])

        assert scored["calibration"].scalars["coverage"] < 0.05

    def test_the_honest_predictor_covers_far_more_of_what_it_claims(self, score: Score) -> None:
        """Not the Gaussian's own figure, and deliberately so.

        The fixtures add noise to a real system, whose dynamics amplify a perturbation
        faster than a random walk accumulates one, so an honest claim about the noise
        still understates the error. That gap is a fact about the systems, which is why
        this brackets the coverage rather than naming it.
        """
        scored = score("calibrated", steps=STEPS, metrics=["calibration"])

        assert scored["calibration"].scalars["coverage"] > 0.2
        assert scored["calibration"].scalars["coverage"] < ONE_SIGMA_COVERAGE * 1.5

    def test_the_claim_is_the_only_difference_the_metric_can_be_seeing(self, score: Score) -> None:
        """The point of the pair, made exactly.

        The same fixture at two confidences produces bit identical states, because the
        confidence changes only what is claimed about the noise and not the noise. Every
        metric that looks at states alone therefore rates the two the same, and the
        calibration error is the only number that moves.
        """
        honest = score("overconfident:factor=1", steps=STEPS)
        broken = score("overconfident:factor=100", steps=STEPS)

        for metric, key in (
            ("one_step_error", "error"),
            ("rollout_error", "error.final"),
            ("invariant_drift", "worst_violation"),
            ("distribution_drift", "worst"),
        ):
            assert honest[metric].scalars[key] == broken[metric].scalars[key]

        assert broken["calibration"].scalars["ece"] > honest["calibration"].scalars["ece"] * 2.0

    def test_a_predictor_that_states_nothing_is_not_scored(self, score: Score) -> None:
        """Every other predictor in the suite.

        Reporting zero steps rather than a perfect calibration is what stops the metric
        flattering six predictors that never made a claim.
        """
        scored = score("persistence", steps=STEPS, metrics=["calibration"])

        assert scored["calibration"].scalars == {"steps": 0.0}


class TestEveryBrokenPredictorIsCaughtBySomething:
    """No fixture may pass the whole suite, or it is not a fixture."""

    @pytest.mark.parametrize("spec", BROKEN)
    def test_at_least_one_metric_flags_it(self, score: Score, spec: str) -> None:
        scored = score(spec, steps=STEPS)
        reference = score("reference", steps=STEPS)

        flags = {
            "rollout_error": scored["rollout_error"].scalars["error.final"] > 1e-3,
            "invariant_drift": scored["invariant_drift"].scalars["worst_violation"] > 1.0,
            "symmetry_violation": scored["symmetry_violation"].scalars["worst"]
            > max(reference["symmetry_violation"].scalars["worst"] * 1e3, 1e-9),
            "distribution_drift": scored["distribution_drift"].scalars["worst"] > 1e-4,
        }
        assert any(flags.values()), f"{spec} passed every metric"
