from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch

from nnphysics.core.errors import ValidationError
from nnphysics.core.protocols import UncertainPredictor
from nnphysics.core.types import State
from nnphysics.data.normalisation import FieldStats, Normalisation
from nnphysics.evals.rollout import roll_out
from nnphysics.models import Carry, Ensemble, ModelContext, SurrogateModel

STEPS = 6
SIZE = 5


def context() -> ModelContext:
    return ModelContext(
        field_shapes={"q": (SIZE,)},
        static_fields=(),
        normalisation=Normalisation({"q": FieldStats(mean=0.0, std=1.0, count=SIZE)}),
        dt=0.5,
        seed=0,
    )


class Drifting(SurrogateModel):
    """Adds a fixed amount every step, so members separate at a rate a test can predict.

    Standing in for a trained model rather than being one. What the ensemble does with its
    members is the thing under test, and a real network would only make the arithmetic
    harder to check by hand.
    """

    def __init__(self, rate: float) -> None:
        super().__init__("drifting", context())
        self._rate = rate

    def advance(
        self, fields: Mapping[str, torch.Tensor], carry: Carry | None
    ) -> tuple[dict[str, torch.Tensor], Carry | None]:
        del carry
        return {"q": fields["q"] + self._rate}, None


def members(*rates: float) -> list[Drifting]:
    return [Drifting(rate) for rate in rates]


def state() -> State:
    """A state of unit scale.

    All zeros would make every later state look divergent to the rollout driver, which
    judges divergence against the scale it started from.
    """
    return State(fields={"q": np.ones(SIZE)}, time=0.0)


class TestWhatItPredicts:
    def test_it_returns_the_mean_of_its_members(self) -> None:
        built = Ensemble(members(1.0, 3.0))

        result = built.predict(state())

        assert np.allclose(result.state.fields["q"], 3.0)

    def test_it_reports_the_population_spread_of_its_members(self) -> None:
        built = Ensemble(members(1.0, 3.0))

        result = built.predict(state())

        assert np.allclose(result.spread["q"], 1.0)

    def test_members_that_agree_report_no_uncertainty(self) -> None:
        built = Ensemble(members(2.0, 2.0))

        result = built.predict(state())

        assert np.allclose(result.spread["q"], 0.0)

    def test_step_gives_the_same_state_predict_does(self) -> None:
        """The two halves of the interface must not disagree about what was predicted."""
        stepped = Ensemble(members(1.0, 3.0)).step(state())
        predicted = Ensemble(members(1.0, 3.0)).predict(state())

        assert np.array_equal(stepped.fields["q"], predicted.state.fields["q"])
        assert stepped.time == predicted.state.time


class TestTheSpreadOverARollout:
    def test_every_member_keeps_its_own_path_so_the_spread_grows(self) -> None:
        """The decision the spread's meaning rests on.

        Averaging after every step and feeding the mean back would give a spread that
        measured one step of disagreement and never grew, which cannot answer whether a
        model warns before it fails.
        """
        built = Ensemble(members(1.0, 3.0))

        result = roll_out(built, state(), STEPS)

        assert result.spread is not None
        curve = result.spread.fields["q"][:, 0]
        assert curve[0] == 0.0
        assert np.allclose(curve[1:], np.arange(1, STEPS + 1, dtype=np.float64))

    def test_the_mean_path_is_the_one_that_was_scored(self) -> None:
        built = Ensemble(members(1.0, 3.0))

        result = roll_out(built, state(), STEPS)

        assert np.allclose(result.trajectory.fields["q"][-1], 2.0 * STEPS + 1.0)

    def test_a_second_rollout_from_the_same_state_starts_over(self) -> None:
        """Members must not be carried from one rollout into the next.

        A predictor that did would report a spread with nothing to do with the states it
        was scored on.
        """
        built = Ensemble(members(1.0, 3.0))

        first = roll_out(built, state(), STEPS)
        second = roll_out(built, state(), STEPS)

        assert first.spread is not None
        assert second.spread is not None
        assert np.array_equal(first.spread.fields["q"], second.spread.fields["q"])


class TestItIsSeenAsUncertain:
    def test_the_harness_recognises_it_through_the_protocol(self) -> None:
        """How the spread reaches a metric without the metric knowing what an ensemble is."""
        assert isinstance(Ensemble(members(1.0, 3.0)), UncertainPredictor)

    def test_a_single_model_is_not_recognised_as_uncertain(self) -> None:
        assert not isinstance(Drifting(1.0), UncertainPredictor)

    def test_a_rollout_of_a_single_model_carries_no_spread(self) -> None:
        assert roll_out(Drifting(1.0), state(), STEPS).spread is None


class TestWhatItRefuses:
    def test_one_member_has_nothing_to_disagree_with(self) -> None:
        with pytest.raises(ValidationError, match="at least 2 members"):
            Ensemble(members(1.0))

    def test_no_members_at_all(self) -> None:
        with pytest.raises(ValidationError, match="at least 2 members"):
            Ensemble([])

    def test_members_that_step_by_different_intervals(self) -> None:
        mismatched = members(1.0, 2.0)
        mismatched[1]._context = ModelContext(
            field_shapes={"q": (SIZE,)},
            static_fields=(),
            normalisation=Normalisation({"q": FieldStats(mean=0.0, std=1.0, count=SIZE)}),
            dt=0.25,
            seed=0,
        )

        with pytest.raises(ValidationError, match="different intervals"):
            Ensemble(mismatched)


class TestWhatItReports:
    def test_it_is_named_for_what_it_is_rather_than_for_its_members(self) -> None:
        assert Ensemble(members(1.0, 3.0)).name == "ensemble"

    def test_its_parameter_count_is_what_the_whole_ensemble_cost(self) -> None:
        built = Ensemble(members(1.0, 3.0, 2.0))

        assert built.n_parameters == sum(member.n_parameters for member in built.members)

    def test_it_advances_by_the_interval_its_members_do(self) -> None:
        assert Ensemble(members(1.0, 3.0)).dt == context().dt
