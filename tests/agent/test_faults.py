from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nnphysics.agent.causes import Cause
from nnphysics.agent.faults import (
    FAULTS,
    Injection,
    RenamedPredictor,
    corrupt_normalisation,
    fault,
    fault_names,
    interrupted,
)
from nnphysics.core.config import RunConfig
from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.types import State
from nnphysics.data.normalisation import FieldStats, Normalisation

from .conftest import CONFIG

BRIEF_CAUSES = {
    Cause.NORMALISATION_STATISTICS,
    Cause.MODEL_SYMMETRY,
    Cause.LEARNING_RATE,
    Cause.TRAINING_REGIME,
    Cause.ROLLOUT_CURRICULUM,
    Cause.INTEGRATOR_STEP_SIZE,
    Cause.OPTIMISER_STATE,
}
"""The seven the phase brief names. Written out here rather than derived from the
catalogue, so that deleting a fault fails this test instead of quietly shrinking it."""


def _config(**changes: Any) -> RunConfig:
    payload = dict(CONFIG)
    payload.update(changes)
    return RunConfig.model_validate(payload)


class TestCatalogue:
    def test_every_cause_the_brief_names_has_a_fault(self) -> None:
        assert {entry.cause for entry in FAULTS} == BRIEF_CAUSES

    def test_no_two_faults_share_a_cause(self) -> None:
        """A table in which two faults shared an answer would measure one of them twice."""
        causes = [entry.cause for entry in FAULTS]

        assert len(set(causes)) == len(causes)

    def test_names_are_distinct(self) -> None:
        assert len(set(fault_names())) == len(FAULTS)

    def test_every_fault_says_what_it_broke(self) -> None:
        for entry in FAULTS:
            assert entry.summary.endswith(".")

    def test_three_faults_leave_no_trace_in_the_configuration(self) -> None:
        """The harder half of the set.

        Nothing a reader could compare says what happened, so the diagnosis has to come from the
        numbers alone.
        """
        hidden = {entry.name for entry in FAULTS if not entry.visible_in_config}

        assert hidden == {"wrong_normalisation", "broken_symmetry", "no_optimiser_state"}

    def test_a_fault_can_be_looked_up_by_name(self) -> None:
        assert fault("no_curriculum").cause is Cause.ROLLOUT_CURRICULUM

    def test_an_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(UnknownNameError, match="no_curriculum"):
            fault("nonsense")


class TestTransforms:
    def test_the_learning_rate_fault_raises_it(self) -> None:
        config = _config()

        faulty = fault("high_learning_rate").apply(config)

        assert faulty.training.learning_rate > config.training.learning_rate * 100

    def test_the_curriculum_fault_reduces_it_to_one_step(self) -> None:
        faulty = fault("no_curriculum").apply(_config())

        assert faulty.training.curriculum == (1,)
        assert faulty.training.curriculum_epochs == (0,)

    def test_the_regime_fault_swaps_what_is_trained_on_and_what_is_held_out(self) -> None:
        config = _config()

        faulty = fault("wrong_regime").apply(config)

        assert faulty.data.regimes == config.data.held_out_regimes
        assert faulty.data.held_out_regimes == config.data.regimes

    def test_the_integrator_fault_takes_one_substep(self) -> None:
        faulty = fault("unstable_integrator").apply(_config())

        assert faulty.data.substeps == 1

    def test_the_integrator_fault_refuses_a_configuration_it_cannot_coarsen(self) -> None:
        """Inventing a coarser setting than there is would invent a fault the run lacks."""
        config = _config(data={**dict(CONFIG["data"]), "substeps": 1})

        with pytest.raises(ValidationError, match="already takes"):
            fault("unstable_integrator").apply(config)

    def test_the_injection_only_faults_leave_the_configuration_alone(self) -> None:
        config = _config()

        for name in ("wrong_normalisation", "broken_symmetry", "no_optimiser_state"):
            assert fault(name).apply(config) == config

    def test_a_transform_never_changes_which_dataset_is_named_unless_it_means_to(
        self,
    ) -> None:
        """Two faults regenerate data and the rest must not, or the table spans datasets."""
        config = _config()
        regenerating = {"wrong_regime", "unstable_integrator"}

        for entry in FAULTS:
            changed = entry.apply(config).data != config.data
            assert changed is (entry.name in regenerating), entry.name


class TestCorruptNormalisation:
    def test_the_scale_is_inflated_and_the_centre_moved(self) -> None:
        original = Normalisation({"velocity": FieldStats(mean=1.0, std=2.0, count=10)})

        corrupted = corrupt_normalisation(original, scale=10.0, shift=3.0)

        assert corrupted.stats["velocity"].std == pytest.approx(20.0)
        assert corrupted.stats["velocity"].mean == pytest.approx(1.0 + 3.0 * 2.0)

    def test_a_constant_field_is_still_moved(self) -> None:
        """Mass is constant, so inflating its spread alone would leave it untouched."""
        original = Normalisation({"mass": FieldStats(mean=1.0, std=0.0, count=10)})

        corrupted = corrupt_normalisation(original, scale=10.0, shift=3.0)

        assert corrupted.stats["mass"].mean == pytest.approx(4.0)

    def test_the_same_fields_are_covered(self) -> None:
        original = Normalisation({"a": FieldStats(1.0, 2.0, 3), "b": FieldStats(0.0, 1.0, 3)})

        assert corrupt_normalisation(original).names == original.names

    def test_a_factor_that_would_flatten_every_field_is_refused(self) -> None:
        original = Normalisation({"a": FieldStats(1.0, 2.0, 3)})

        with pytest.raises(ValidationError, match="must be positive"):
            corrupt_normalisation(original, scale=0.0)


class TestInterrupted:
    def test_the_first_half_is_shorter_than_the_whole(self) -> None:
        config = _config(training={"epochs": 12, "curriculum": [1, 4], "curriculum_epochs": [0, 3]})

        first = interrupted(config)

        assert first.training.epochs < config.training.epochs

    def test_the_first_half_still_reaches_its_own_last_curriculum_stage(self) -> None:
        """A first half that never reached its own schedule is a different run."""
        config = _config(training={"epochs": 12, "curriculum": [1, 4], "curriculum_epochs": [0, 3]})

        first = interrupted(config)

        assert first.training.epochs > config.training.curriculum_epochs[-1]

    def test_a_run_too_short_to_interrupt_is_refused(self) -> None:
        config = _config(training={"epochs": 4, "curriculum": [1, 2], "curriculum_epochs": [0, 3]})

        with pytest.raises(ValidationError, match="cannot be interrupted"):
            interrupted(config)

    def test_nothing_else_about_the_run_moves(self) -> None:
        config = _config(training={"epochs": 12, "curriculum": [1, 4], "curriculum_epochs": [0, 3]})

        first = interrupted(config)

        assert first.data == config.data
        assert first.training.learning_rate == config.training.learning_rate


class _Doubling:
    """A predictor that doubles every field, so a wrapper is visibly passing through."""

    name = "inner"
    dt = 0.5

    def step(self, state: State) -> State:
        return State(
            fields={name: array * 2.0 for name, array in state.fields.items()},
            time=state.time + self.dt,
        )


class TestRenamedPredictor:
    def test_the_step_and_the_interval_pass_through(self) -> None:
        wrapped = RenamedPredictor(inner=_Doubling(), name="graph")
        state = State(fields={"position": np.ones((2, 2))}, time=0.0)

        stepped = wrapped.step(state)

        assert wrapped.dt == 0.5
        assert np.allclose(stepped.fields["position"], 2.0)
        assert stepped.time == 0.5

    def test_only_the_name_changes(self) -> None:
        """The model keeps its own name, or the comparison has nothing to line it up with."""
        assert RenamedPredictor(inner=_Doubling(), name="graph").name == "graph"


class TestInjection:
    def test_every_fault_declares_where_it_interferes(self) -> None:
        assert {entry.injection for entry in FAULTS} <= set(Injection)

    def test_the_configuration_only_faults_declare_no_injection(self) -> None:
        for entry in FAULTS:
            if entry.injection is Injection.NONE:
                assert entry.visible_in_config, entry.name
