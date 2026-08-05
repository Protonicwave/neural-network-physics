from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from nnphysics.core.errors import UnknownNameError
from nnphysics.core.types import Rollout
from nnphysics.evals.metrics import DEFAULT_METRICS, MetricContext, build_metrics
from nnphysics.evals.predictors import PredictorContext, Substepped, build_predictor, parse_spec
from nnphysics.evals.result import InvariantRecord, SuiteResult
from nnphysics.evals.rollout import roll_out
from nnphysics.reporting.explain import (
    TIMING_METRIC,
    Direction,
    explain,
    explanations,
    metric_summary,
)
from nnphysics.systems import build_system

SCENES = (("nbody", {"softening": 0.05}, 0.02), ("fluid", {"grid_size": 32}, 0.05))
"""Both systems, so that a scalar named after a field or an invariant one of them declares
cannot go unexplained."""

_STEPS = 8

ENERGY = InvariantRecord(name="energy", conservation="exact", rtol=1e-6, dimension="M L^2 T^-2")


def scalars_of(name: str, parameters: dict[str, Any], dt: float) -> dict[str, set[str]]:
    """Every scalar the default metrics actually produce for one system."""
    system = build_system(name, parameters)
    regime = system.regimes[0]
    reference = Substepped(system.reference_predictor(regime, dt / 4), 4)
    initial = system.initial_state(regime, np.random.default_rng(0))
    truth = roll_out(reference, initial, _STEPS).trajectory
    predictor = build_predictor(
        parse_spec("noise:scale=0.01"),
        PredictorContext(
            reference=reference,
            state_spec=system.state_spec,
            symmetries=system.symmetries,
            seed=0,
            stream="test",
        ),
    )
    result = roll_out(predictor, initial, _STEPS)
    rollout = Rollout(
        predicted=result.trajectory,
        reference=truth,
        predictor=result.predictor,
        system=system.name,
    )
    context = MetricContext(
        invariants=system.invariants(regime),
        symmetries=system.symmetries,
        predictor=predictor,
        symmetry_steps=4,
    )
    return {
        metric.name: set(metric.compute(rollout).scalars)
        for metric in build_metrics(DEFAULT_METRICS, context)
    }


@pytest.fixture(scope="module", params=SCENES, ids=[entry[0] for entry in SCENES])
def produced(request: pytest.FixtureRequest) -> dict[str, set[str]]:
    """The scalars a real run of every default metric produces."""
    name, parameters, dt = request.param
    return scalars_of(name, parameters, dt)


class TestCoverage:
    def test_every_scalar_a_metric_produces_is_explained(
        self, produced: dict[str, set[str]]
    ) -> None:
        for metric, keys in produced.items():
            for key in keys:
                explain(metric, key)

    def test_every_metric_has_a_summary(self, produced: dict[str, set[str]]) -> None:
        for metric in produced:
            assert metric_summary(metric)

    def test_every_explanation_carries_units_and_a_sentence(
        self, produced: dict[str, set[str]]
    ) -> None:
        for metric, keys in produced.items():
            for key in keys:
                explanation = explain(metric, key)
                assert explanation.units
                assert explanation.text.endswith(".")

    def test_a_scalar_nobody_has_described_is_refused(self) -> None:
        with pytest.raises(UnknownNameError, match="nothing explains"):
            explain("rollout_error", "recurrence_time")

    def test_a_declared_name_with_an_undescribed_suffix_is_refused(self) -> None:
        with pytest.raises(UnknownNameError, match="nothing explains"):
            explain("invariant_drift", "energy.curvature")

    def test_an_unknown_metric_is_refused(self) -> None:
        with pytest.raises(UnknownNameError, match="nothing explains"):
            explain("novelty", "worst")

    def test_a_metric_without_a_summary_is_refused(self) -> None:
        with pytest.raises(UnknownNameError, match="nothing summarises"):
            metric_summary("novelty")


class TestDirection:
    def test_an_error_is_better_when_it_is_lower(self) -> None:
        assert explain("rollout_error", "error.final").direction is Direction.LOWER

    def test_a_horizon_is_better_when_it_is_longer(self) -> None:
        assert explain("rollout_error", "horizon.0.01").direction is Direction.HIGHER

    def test_a_horizon_carries_the_value_that_means_it_was_never_reached(self) -> None:
        explanation = explain("rollout_error", "horizon.1")

        assert explanation.sentinel == -1.0
        assert explanation.sentinel_text == "never reached"

    def test_a_scalar_whose_ideal_is_neither_extreme_claims_no_direction(self) -> None:
        assert explain("distribution_drift", "x.spread_ratio").direction is Direction.NEUTRAL

    def test_a_description_of_the_run_claims_no_direction(self) -> None:
        assert explain("rollout_error", "duration").direction is Direction.NEUTRAL

    def test_time_per_step_is_better_when_it_is_lower(self) -> None:
        assert explain(TIMING_METRIC, "seconds_per_step").direction is Direction.LOWER


class TestUnits:
    def test_a_drift_takes_the_dimension_the_system_declared(self) -> None:
        explanation = explain("invariant_drift", "energy.drift", {"energy": ENERGY})

        assert explanation.units == "M L^2 T^-2"

    def test_a_drift_without_a_declaration_says_so_rather_than_inventing_units(self) -> None:
        assert explain("invariant_drift", "energy.drift").units == "units of energy"

    def test_a_violation_is_dimensionless_whatever_the_invariant_is(self) -> None:
        explanation = explain("invariant_drift", "energy.violation", {"energy": ENERGY})

        assert explanation.units == "1 (normalised)"

    def test_a_horizon_is_in_simulated_time(self) -> None:
        assert explain("rollout_error", "horizon.0.01").units == "simulated time"


class TestNaming:
    def test_the_declared_name_reaches_the_sentence(self) -> None:
        explanation = explain("symmetry_violation", "translation.max")

        assert "translation" in explanation.text
        assert explanation.title == "translation: worst"

    def test_a_field_error_names_the_field(self) -> None:
        assert "vorticity" in explain("rollout_error", "error.vorticity").text

    def test_a_threshold_reaches_the_horizon_sentence(self) -> None:
        assert "0.01" in explain("rollout_error", "horizon.0.01").text


class TestWholeResult:
    def test_it_describes_every_scalar_a_result_carries(
        self, make_result: Callable[..., SuiteResult]
    ) -> None:
        result = make_result()
        described = {entry.qualified for entry in explanations(result)}

        for entry in result.results:
            for record in entry.metrics:
                for key in record.scalars:
                    assert f"{record.name}.{key}" in described

    def test_it_describes_the_timing_the_metrics_do_not_produce(
        self, make_result: Callable[..., SuiteResult]
    ) -> None:
        described = {entry.qualified for entry in explanations(make_result())}

        assert f"{TIMING_METRIC}.seconds_per_step" in described

    def test_it_follows_the_order_the_suite_named_its_metrics(
        self, make_result: Callable[..., SuiteResult]
    ) -> None:
        described = [entry.metric for entry in explanations(make_result())]

        assert described.index("one_step_error") < described.index("rollout_error")
        assert described.index("rollout_error") < described.index("invariant_drift")
