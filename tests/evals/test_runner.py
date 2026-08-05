from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.protocols import System
from nnphysics.data.manifest import Manifest, Split
from nnphysics.evals.result import SuiteResult
from nnphysics.evals.runner import (
    EvaluationCase,
    evaluate_predictor,
    load_cases,
    regime_gap,
    run_suite,
)
from nnphysics.systems import build_system

Dataset = tuple[Path, Manifest, RunConfig]

STEPS = 6


def system_of(config: RunConfig) -> System:
    return build_system(config.system.name, config.system.parameters)


class TestLoadingCases:
    def test_ground_truth_comes_from_the_dataset(self, dataset: Dataset) -> None:
        directory, manifest, config = dataset

        cases = load_cases(
            directory, manifest, system_of(config), split=Split.TEST, count=2, steps=STEPS
        )

        assert len(cases) == 2
        assert all(case.steps == STEPS for case in cases)
        assert all(case.initial.time == 0.0 for case in cases)

    def test_the_same_trajectories_are_chosen_every_time(self, dataset: Dataset) -> None:
        """Reproducible without a generator: identifier order decides, not a draw."""
        directory, manifest, config = dataset
        system = system_of(config)

        first = load_cases(directory, manifest, system, split=Split.TEST, count=2, steps=STEPS)
        second = load_cases(directory, manifest, system, split=Split.TEST, count=2, steps=STEPS)

        assert [case.trajectory_id for case in first] == [case.trajectory_id for case in second]

    def test_the_regimes_of_a_split_are_all_represented(self, dataset: Dataset) -> None:
        """A number for a split that only described one of its regimes would be misleading."""
        directory, manifest, config = dataset

        cases = load_cases(
            directory, manifest, system_of(config), split=Split.TEST, count=2, steps=STEPS
        )

        assert {case.regime.name for case in cases} == set(config.data.regimes)

    def test_the_regime_is_resolved_from_the_manifest(self, dataset: Dataset) -> None:
        directory, manifest, config = dataset

        cases = load_cases(
            directory, manifest, system_of(config), split=Split.HELD_OUT, count=1, steps=STEPS
        )

        assert cases[0].regime.name == config.data.held_out_regimes[0]

    def test_asking_for_more_than_a_split_holds_is_an_error(self, dataset: Dataset) -> None:
        """Silently returning fewer would make two runs of one suite incomparable."""
        directory, manifest, config = dataset

        with pytest.raises(ValidationError, match="asked for"):
            load_cases(
                directory, manifest, system_of(config), split=Split.TEST, count=99, steps=STEPS
            )

    def test_a_horizon_longer_than_the_stored_trajectory_is_an_error(
        self, dataset: Dataset
    ) -> None:
        directory, manifest, config = dataset

        with pytest.raises(ValidationError, match="stored states"):
            load_cases(directory, manifest, system_of(config), split=Split.TEST, count=1, steps=999)


class TestEvaluatingOnePredictor:
    def cases(self, dataset: Dataset, split: Split = Split.TEST) -> list[EvaluationCase]:
        directory, manifest, config = dataset
        return load_cases(directory, manifest, system_of(config), split=split, count=2, steps=STEPS)

    def test_the_reference_reproduces_the_stored_trajectory(self, dataset: Dataset) -> None:
        """The dataset is the solver's own output, so rolling the solver out must match it."""
        _, manifest, config = dataset

        result = evaluate_predictor(
            system_of(config),
            self.cases(dataset),
            "reference",
            config.evaluation,
            substeps=manifest.spec.substeps,
            seed=config.seed,
        )

        assert result.scalar("rollout_error", "error.final") == pytest.approx(0.0, abs=1e-12)
        assert result.completed

    def test_every_rollout_is_recorded(self, dataset: Dataset) -> None:
        _, manifest, config = dataset

        result = evaluate_predictor(
            system_of(config),
            self.cases(dataset),
            "persistence",
            config.evaluation,
            substeps=manifest.spec.substeps,
            seed=config.seed,
        )

        assert len(result.rollouts) == 2
        assert {record.split for record in result.rollouts} == {"test"}
        assert all(record.stop_reason == "completed" for record in result.rollouts)

    def test_the_specification_is_recorded_as_written(self, dataset: Dataset) -> None:
        _, manifest, config = dataset

        result = evaluate_predictor(
            system_of(config),
            self.cases(dataset),
            "noise:scale=0.02",
            config.evaluation,
            substeps=manifest.spec.substeps,
            seed=config.seed,
        )

        assert result.predictor == "noise"
        assert result.spec == "noise:scale=0.02"

    def test_the_cost_per_step_is_recorded(self, dataset: Dataset) -> None:
        """What a speedup claim is eventually measured from."""
        _, manifest, config = dataset

        result = evaluate_predictor(
            system_of(config),
            self.cases(dataset),
            "reference",
            config.evaluation,
            substeps=manifest.spec.substeps,
            seed=config.seed,
        )

        assert result.seconds_per_step > 0.0

    def test_evaluating_nothing_is_rejected(self, dataset: Dataset) -> None:
        _, manifest, config = dataset

        with pytest.raises(ValidationError, match="without any initial conditions"):
            evaluate_predictor(
                system_of(config),
                [],
                "reference",
                config.evaluation,
                substeps=manifest.spec.substeps,
                seed=config.seed,
            )

    def test_mixing_splits_in_one_result_is_rejected(self, dataset: Dataset) -> None:
        """A number that averaged two splits would hide the very gap the suite reports."""
        _, manifest, config = dataset
        mixed = [*self.cases(dataset), *self.cases(dataset, Split.HELD_OUT)]

        with pytest.raises(ValidationError, match="mix splits"):
            evaluate_predictor(
                system_of(config),
                mixed,
                "reference",
                config.evaluation,
                substeps=manifest.spec.substeps,
                seed=config.seed,
            )


class TestRunningASuite:
    def run(self, dataset: Dataset, **overrides: Any) -> SuiteResult:
        directory, manifest, config = dataset
        return run_suite(
            system_of(config),
            directory,
            manifest,
            config.evaluation,
            seed=config.seed,
            run_id=config.run_id,
            **overrides,
        )

    def test_every_predictor_is_scored_on_every_split(self, dataset: Dataset) -> None:
        result = self.run(dataset)

        assert {(entry.predictor, entry.split) for entry in result.results} == {
            ("reference", "test"),
            ("reference", "held_out"),
            ("persistence", "test"),
            ("persistence", "held_out"),
        }

    def test_the_declared_invariants_are_recorded_per_regime(self, dataset: Dataset) -> None:
        """A drift number cannot be read later without knowing what was declared."""
        result = self.run(dataset)

        declared = result.invariants
        assert set(declared) == {"cold_collapse", "virialised_cluster", "hierarchical_pair"}
        assert {record.name for record in declared["cold_collapse"]} == {
            "energy",
            "linear_momentum",
            "angular_momentum",
        }
        assert all(record.conservation for record in declared["cold_collapse"])

    def test_the_settings_are_recorded_with_the_numbers(self, dataset: Dataset) -> None:
        _, _, config = dataset

        result = self.run(dataset)

        assert result.settings.name == config.evaluation.name
        assert result.settings.metrics == config.evaluation.metrics

    def test_a_predictor_given_on_the_command_line_overrides_the_suite(
        self, dataset: Dataset
    ) -> None:
        result = self.run(dataset, predictors=["noise:scale=0.01"])

        assert {entry.predictor for entry in result.results} == {"noise"}

    def test_running_one_split_records_no_gap(self, dataset: Dataset) -> None:
        result = self.run(dataset, splits=[Split.TEST])

        assert result.regime_gap == {}

    def test_the_run_is_reproducible(self, dataset: Dataset) -> None:
        """Same configuration, same seed, same numbers, noise included."""
        first = self.run(dataset, predictors=["noise"])
        second = self.run(dataset, predictors=["noise"])

        assert first.results[0].metrics == second.results[0].metrics

    def test_an_unknown_metric_is_rejected(self, dataset: Dataset) -> None:
        directory, manifest, config = dataset
        suite = config.evaluation.model_copy(update={"metrics": ("accuracy",)})

        with pytest.raises(UnknownNameError, match="unknown metric"):
            run_suite(
                system_of(config),
                directory,
                manifest,
                suite,
                seed=config.seed,
                run_id=config.run_id,
            )


class TestRegimeGap:
    def test_the_gap_is_held_out_minus_in_distribution(self, dataset: Dataset) -> None:
        directory, manifest, config = dataset

        result = run_suite(
            system_of(config),
            directory,
            manifest,
            config.evaluation,
            seed=config.seed,
            run_id=config.run_id,
            predictors=["persistence"],
        )

        key = "persistence.rollout_error.error.final"
        assert key in result.regime_gap
        assert result.regime_gap[key] == pytest.approx(
            result.result("persistence", "held_out").scalar("rollout_error", "error.final")
            - result.result("persistence", "test").scalar("rollout_error", "error.final")
        )

    def test_a_result_without_both_splits_produces_no_gap(self) -> None:
        assert regime_gap([]) == {}
