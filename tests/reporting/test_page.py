from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ConfigurationError
from nnphysics.evals.metrics import NEVER_REACHED
from nnphysics.evals.predictors import BROKEN_PREDICTORS
from nnphysics.evals.predictors.broken import Persistence
from nnphysics.evals.result import (
    MetricRecord,
    PredictorResult,
    RolloutRecord,
    SuiteResult,
    SuiteSettings,
)
from nnphysics.evals.speed import (
    NEVER_PAYS,
    CostAccounting,
    MatchedSpeedup,
    SpeedPoint,
    SpeedReport,
)
from nnphysics.reporting.environment import EnvironmentRecord
from nnphysics.reporting.layout import PLOTS_DIR, RECORD_NAME, find_records, state_plot_name
from nnphysics.reporting.page import (
    HARNESS_PREDICTORS,
    HELD_OUT_SPLIT,
    NOISE,
    PERSISTENCE,
    TRAINED_ON_SPLIT,
    USABLE_THRESHOLD,
    VIEWER_BASELINES,
    DriftViewer,
    PageModel,
    VerdictKind,
    build_page,
    build_viewer,
    read_fault_scores,
    summarise_run,
)
from nnphysics.reporting.record import RunRecord, read_record, write_record
from nnphysics.training.history import EpochRecord, TrainingHistory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DT = 0.01
"""The stored step the fixture configuration uses, which every usable stretch is counted
in."""

ROLLOUT_STEPS = 100

CONFIG: dict[str, Any] = {
    "name": "example",
    "seed": 5,
    "system": {"name": "toy"},
    "data": {
        "n_trajectories": 8,
        "n_steps": 8,
        "dt": DT,
        "regimes": ["hot"],
        "held_out_regimes": ["cold"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
    },
    "model": {"name": "placeholder"},
    "evaluation": {"name": "standard", "metrics": ["rollout_error"], "rollout_steps": 4},
}

ENVIRONMENT = EnvironmentRecord(
    python="3.12.0",
    implementation="CPython",
    platform="Linux 6.1",
    machine="x86_64",
    cpu_count=8,
    packages={"numpy": "2.1.0"},
)
"""Fixed rather than read from the machine, so that the same runs root gives the same page
everywhere."""


def _rollouts(*, stop: str = "completed", count: int = 4) -> tuple[RolloutRecord, ...]:
    """Rollouts that all ended the same way."""
    return tuple(
        RolloutRecord(
            trajectory=f"t{index}",
            regime="hot",
            split="test",
            steps_requested=ROLLOUT_STEPS,
            steps_completed=ROLLOUT_STEPS if stop == "completed" else 1,
            stop_reason=stop,
            seconds=0.01,
        )
        for index in range(count)
    )


def _predictor(
    predictor: str,
    split: str,
    *,
    horizon: float = 0.25,
    stop: str = "completed",
    metrics: bool = True,
) -> PredictorResult:
    """One predictor on one split, with its usable stretch chosen directly."""
    scalars = {"error.final": 1.0, f"horizon.{USABLE_THRESHOLD:g}": horizon}
    return PredictorResult(
        predictor=predictor,
        spec=predictor,
        split=split,
        regimes=("hot",),
        rollouts=_rollouts(stop=stop),
        metrics=(MetricRecord(name="rollout_error", scalars=scalars),) if metrics else (),
        seconds_per_step=0.002,
        completed=stop == "completed",
    )


def _result(results: Sequence[PredictorResult]) -> SuiteResult:
    """A suite result carrying nothing but the predictors handed to it."""
    return SuiteResult(
        code_version="0.1.0",
        run_id="0123456789abcdef",
        dataset_id="fedcba9876543210",
        system="toy",
        seed=5,
        settings=SuiteSettings(
            name="standard",
            metrics=("rollout_error",),
            rollout_steps=ROLLOUT_STEPS,
            n_initial_conditions=4,
            error_thresholds=(0.01, USABLE_THRESHOLD, 1.0),
            symmetry_steps=4,
            distribution_window=0.25,
            divergence_factor=1000.0,
        ),
        invariants={},
        results=tuple(results),
    )


def _history(*, model: str = "graph", epochs: int = 3) -> TrainingHistory:
    """A training history, which is what marks a run as having trained a model itself."""
    return TrainingHistory(
        model=model,
        n_parameters=1234,
        train_windows=8,
        validation_windows=4,
        validation_steps=2,
        epochs=tuple(
            EpochRecord(
                epoch=index,
                curriculum_steps=1,
                learning_rate=0.001,
                loss=1.0,
                one_step_error=1.0,
                multi_step_error=1.0,
                physics_penalty=0.0,
                gradient_norm=1.0,
                validation_error=1.0,
                seconds=1.0,
                improved=index == 0,
            )
            for index in range(epochs)
        ),
        best_epoch=0,
        best_validation_error=1.0,
        stopped_early=False,
        seconds=10.0,
    )


def _record(
    results: Sequence[PredictorResult],
    *,
    run_id: str = "0123456789abcdef",
    created: str = "2026-01-01T00:00:00+00:00",
    training: TrainingHistory | None = None,
    benchmark: SpeedReport | None = None,
) -> RunRecord:
    """A run record whose every varying field is fixed."""
    return RunRecord(
        run_id=run_id,
        name="example",
        created=created,
        code_version="0.1.0",
        commit="a" * 40,
        config=RunConfig.model_validate(CONFIG),
        environment=ENVIRONMENT,
        timings={"evaluation": 1.5},
        training=training,
        evaluation=_result(results),
        benchmark=benchmark,
    )


def _speed_report(
    *,
    speedup: float = 0.5,
    bracketed: bool = True,
    break_even: float = NEVER_PAYS,
    matched: bool = True,
) -> SpeedReport:
    """A benchmark whose matched comparison says what a test needs it to say."""
    solver = SpeedPoint(
        label="reference:substeps=1",
        predictor="reference",
        kind="solver",
        substeps=1,
        error=0.5,
        seconds_per_step=0.001,
        iqr=0.0001,
        relative_spread=0.1,
        stable=True,
        completed=True,
    )
    surrogate = SpeedPoint(
        label="graph",
        predictor="graph",
        kind="surrogate",
        substeps=0,
        error=0.5,
        seconds_per_step=0.002,
        iqr=0.0002,
        relative_spread=0.2,
        stable=True,
        completed=True,
    )
    comparison = MatchedSpeedup(
        predictor="graph",
        error=0.5,
        seconds_per_step=0.002,
        matched_substeps=1,
        matched_error=0.5,
        matched_seconds_per_step=0.001,
        speedup=speedup,
        bracketed=bracketed,
    )
    return SpeedReport(
        system="toy",
        split="test",
        steps=ROLLOUT_STEPS,
        n_initial_conditions=4,
        threads=1,
        trials=5,
        warmup=1,
        steps_per_trial=4,
        dataset_substeps=1,
        ladder=(solver,),
        surrogates=(surrogate,),
        matched=(comparison,) if matched else (),
        costs=(
            CostAccounting(
                predictor="graph",
                training_seconds=10.0,
                generation_seconds=1.0,
                steps_per_rollout=ROLLOUT_STEPS,
                saving_per_rollout=-0.1,
                break_even_rollouts=break_even,
            ),
        )
        if matched
        else (),
    )


def _healthy() -> tuple[PredictorResult, ...]:
    """A run whose learned model completes everywhere and clears the free baseline."""
    return (
        _predictor(PERSISTENCE, TRAINED_ON_SPLIT, horizon=0.1),
        _predictor(PERSISTENCE, HELD_OUT_SPLIT, horizon=0.1),
        _predictor("graph", TRAINED_ON_SPLIT, horizon=0.25),
        _predictor("graph", HELD_OUT_SPLIT, horizon=0.2),
    )


def _write(root: Path, directory: str, record: RunRecord) -> None:
    """Put one record where a runs root keeps it."""
    run = root / directory
    run.mkdir(parents=True)
    write_record(run / RECORD_NAME, record)


def _page(root: Path) -> PageModel:
    """Build a page from a runs root with no diagnosis section."""
    return build_page(root)


def _draw(root: Path, directory: str, *combinations: tuple[str, str]) -> None:
    """Put a state comparison plot where a run keeps it, for each combination named."""
    plots = root / directory / PLOTS_DIR
    plots.mkdir(parents=True, exist_ok=True)
    for split, predictor in combinations:
        (plots / state_plot_name(split, predictor)).write_bytes(b"")


def _both(predictor: str) -> tuple[tuple[str, str], ...]:
    """The combinations a predictor needs a plot for before the viewer will offer it."""
    return ((TRAINED_ON_SPLIT, predictor), (HELD_OUT_SPLIT, predictor))


def _viewer(root: Path) -> DriftViewer | None:
    """What a runs root's drift viewer can show.

    Built the way the page builds it, from the runs it reports on in the order it reports
    them, rather than through `build_page`, which would also demand the hero figures.
    """
    cards = sorted(
        (summarise_run(read_record(path), path.parent.name) for path in find_records(root)),
        key=lambda card: (card.created, card.run_id),
    )
    return build_viewer(tuple(card for card in cards if not card.fault), root)


def _offering(root: Path) -> tuple[tuple[str, str, str], ...]:
    """What a runs root's viewer offers, as one tuple per combination."""
    viewer = _viewer(root)
    if viewer is None:
        return ()
    return tuple((frame.system, frame.predictor, frame.split) for frame in viewer.frames)


class TestHarnessPredictors:
    def test_the_free_baseline_is_named_as_the_predictor_names_itself(self) -> None:
        assert Persistence(dt=1.0).name == PERSISTENCE

    def test_the_baseline_is_part_of_the_harness(self) -> None:
        assert PERSISTENCE in HARNESS_PREDICTORS

    def test_a_learned_model_is_not(self) -> None:
        assert "graph" not in HARNESS_PREDICTORS
        assert "operator" not in HARNESS_PREDICTORS


class TestUsableSteps:
    def test_a_horizon_becomes_stored_steps(self) -> None:
        card = summarise_run(_record([_predictor("graph", TRAINED_ON_SPLIT, horizon=0.25)]), "run")
        entry = card.horizon("graph", TRAINED_ON_SPLIT)

        assert entry is not None
        assert entry.steps == pytest.approx(0.25 / DT)

    def test_never_crossing_the_threshold_is_the_whole_rollout(self) -> None:
        card = summarise_run(
            _record([_predictor("reference", TRAINED_ON_SPLIT, horizon=NEVER_REACHED)]), "run"
        )
        entry = card.horizon("reference", TRAINED_ON_SPLIT)

        assert entry is not None
        assert entry.whole_rollout
        assert entry.steps == ROLLOUT_STEPS

    def test_a_model_that_refused_to_build_reports_nothing(self) -> None:
        card = summarise_run(
            _record(
                [_predictor("mlp", HELD_OUT_SPLIT, horizon=NEVER_REACHED, stop="failed")],
            ),
            "run",
        )
        entry = card.horizon("mlp", HELD_OUT_SPLIT)

        assert entry is not None
        assert entry.failed
        assert entry.steps is None
        assert not entry.whole_rollout

    def test_a_suite_that_never_measured_error_growth_reports_nothing(self) -> None:
        card = summarise_run(_record([_predictor("graph", TRAINED_ON_SPLIT, metrics=False)]), "run")
        entry = card.horizon("graph", TRAINED_ON_SPLIT)

        assert entry is not None
        assert entry.steps is None
        assert not entry.whole_rollout

    def test_a_diverged_rollout_is_counted_and_flagged(self) -> None:
        card = summarise_run(_record([_predictor("graph", HELD_OUT_SPLIT, stop="diverged")]), "run")
        entry = card.horizon("graph", HELD_OUT_SPLIT)

        assert entry is not None
        assert entry.diverged
        assert entry.completed == 0
        assert entry.rollouts == 4

    def test_a_predictor_is_marked_learned_or_not(self) -> None:
        card = summarise_run(_record(_healthy()), "run")

        assert card.horizon("graph", TRAINED_ON_SPLIT).learned  # type: ignore[union-attr]
        assert not card.horizon(PERSISTENCE, TRAINED_ON_SPLIT).learned  # type: ignore[union-attr]


class TestVerdict:
    def test_a_run_with_no_learned_predictor_is_a_harness_check(self) -> None:
        card = summarise_run(_record([_predictor(PERSISTENCE, TRAINED_ON_SPLIT)]), "run")

        assert card.verdict.kind is VerdictKind.HARNESS_CHECK
        assert card.verdict.phrase == "harness check"

    def test_a_model_that_cannot_start_on_the_unseen_split_says_so(self) -> None:
        card = summarise_run(
            _record(
                [
                    _predictor("mlp", TRAINED_ON_SPLIT),
                    _predictor("mlp", HELD_OUT_SPLIT, stop="failed"),
                ],
                training=_history(model="mlp"),
            ),
            "run",
        )

        assert card.verdict.kind is VerdictKind.CANNOT_RUN_UNSEEN
        assert card.verdict.phrase == "cannot run unseen"

    def test_diverging_only_on_the_unseen_split_names_that_split(self) -> None:
        card = summarise_run(
            _record(
                [
                    _predictor("graph", TRAINED_ON_SPLIT),
                    _predictor("graph", HELD_OUT_SPLIT, stop="diverged"),
                ],
                training=_history(),
            ),
            "run",
        )

        assert card.verdict.kind is VerdictKind.DIVERGES
        assert card.verdict.phrase == "diverges when unseen"

    def test_diverging_on_the_trained_on_split_too_is_stated_plainly(self) -> None:
        card = summarise_run(
            _record(
                [
                    _predictor("operator", TRAINED_ON_SPLIT, stop="diverged"),
                    _predictor("operator", HELD_OUT_SPLIT, stop="diverged"),
                ],
                training=_history(model="operator"),
            ),
            "run",
        )

        assert card.verdict.phrase == "diverges"

    def test_completing_everything_and_clearing_the_baseline_is_stable(self) -> None:
        card = summarise_run(_record(_healthy(), training=_history()), "run")

        assert card.verdict.kind is VerdictKind.STABLE
        assert card.verdict.phrase == "stable, beats persistence"

    def test_completing_everything_below_the_baseline_is_qualified(self) -> None:
        card = summarise_run(
            _record(
                [
                    _predictor(PERSISTENCE, TRAINED_ON_SPLIT, horizon=0.5),
                    _predictor("convolution", TRAINED_ON_SPLIT, horizon=0.25),
                ],
                training=_history(model="convolution"),
            ),
            "run",
        )

        assert card.verdict.phrase == "stable, inaccurate"

    def test_a_benchmark_of_models_it_did_not_train_is_read_off_the_speedup(self) -> None:
        card = summarise_run(
            _record(_healthy(), benchmark=_speed_report(speedup=0.5, break_even=NEVER_PAYS)),
            "run",
        )

        assert card.verdict.kind is VerdictKind.COST_BENCHMARK
        assert card.verdict.phrase == "never pays back"

    def test_an_unbracketed_comparison_is_undecided(self) -> None:
        card = summarise_run(
            _record(_healthy(), benchmark=_speed_report(speedup=1.33, bracketed=False)),
            "run",
        )

        assert card.verdict.phrase == "undecided"

    def test_a_surrogate_that_repays_its_training_says_when(self) -> None:
        card = summarise_run(
            _record(_healthy(), benchmark=_speed_report(speedup=2.0, break_even=500.0)),
            "run",
        )

        assert card.verdict.phrase == "pays back eventually"

    def test_a_benchmark_that_matched_nothing_measured_nothing(self) -> None:
        card = summarise_run(_record(_healthy(), benchmark=_speed_report(matched=False)), "run")

        assert card.verdict.phrase == "not measured"

    def test_a_run_that_trained_its_own_model_is_judged_on_its_rollouts(self) -> None:
        # A benchmark alone does not make a run a cost benchmark: this one trained the
        # model it benchmarked, so the accuracy question is still the one being asked.
        card = summarise_run(
            _record(_healthy(), training=_history(), benchmark=_speed_report()), "run"
        )

        assert card.verdict.kind is VerdictKind.STABLE


class TestRunCard:
    def test_it_carries_what_the_register_shows(self) -> None:
        card = summarise_run(_record(_healthy(), training=_history(epochs=40)), "nbody-0123")

        assert card.model == "graph"
        assert card.parameters == 1234
        assert card.epochs == 40
        assert card.run_id == "0123456789abcdef"
        assert card.commit == "a" * 40

    def test_a_run_that_trained_nothing_says_so_rather_than_guessing(self) -> None:
        card = summarise_run(_record(_healthy()), "run")

        assert card.model is None
        assert card.parameters is None
        assert card.epochs is None

    def test_the_links_are_relative_to_the_runs_root(self) -> None:
        card = summarise_run(_record(_healthy()), "nbody-0123456789abcdef")

        assert card.report == "nbody-0123456789abcdef/report.html"
        assert card.plots == "nbody-0123456789abcdef/plots"

    def test_a_fault_run_is_marked_by_its_directory(self) -> None:
        assert summarise_run(_record(_healthy()), "nbody-fault-wrong_regime-0123").fault
        assert not summarise_run(_record(_healthy()), "nbody-0123").fault

    def test_a_run_with_no_benchmark_still_summarises(self) -> None:
        card = summarise_run(_record(_healthy()), "run")

        assert card.cost is None

    def test_a_run_with_no_held_out_split_still_summarises(self) -> None:
        card = summarise_run(_record([_predictor("graph", TRAINED_ON_SPLIT)]), "run")

        assert card.horizon("graph", HELD_OUT_SPLIT) is None
        assert card.verdict.kind is VerdictKind.STABLE


class TestCostLadder:
    def test_the_points_keep_which_side_they_are_on(self) -> None:
        card = summarise_run(_record(_healthy(), benchmark=_speed_report()), "run")

        assert card.cost is not None
        assert all(point.solver for point in card.cost.solver)
        assert not any(point.solver for point in card.cost.surrogates)

    def test_a_slower_surrogate_carries_its_slowdown(self) -> None:
        card = summarise_run(_record(_healthy(), benchmark=_speed_report(speedup=0.05)), "run")

        assert card.cost is not None
        assert card.cost.matched[0].slowdown == pytest.approx(20.0)

    def test_a_faster_surrogate_carries_none(self) -> None:
        card = summarise_run(
            _record(_healthy(), benchmark=_speed_report(speedup=2.0, break_even=500.0)), "run"
        )

        assert card.cost is not None
        assert card.cost.matched[0].slowdown is None

    def test_never_paying_back_is_absent_rather_than_negative(self) -> None:
        card = summarise_run(_record(_healthy(), benchmark=_speed_report()), "run")

        assert card.cost is not None
        assert card.cost.matched[0].break_even_rollouts is None

    def test_a_break_even_count_survives(self) -> None:
        card = summarise_run(
            _record(_healthy(), benchmark=_speed_report(speedup=2.0, break_even=14079.0)), "run"
        )

        assert card.cost is not None
        assert card.cost.matched[0].break_even_rollouts == pytest.approx(14079.0)


class TestHeadlines:
    def test_the_longest_usable_stretch_wins(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-2",
            _record(
                [_predictor("operator", TRAINED_ON_SPLIT, horizon=0.5)],
                run_id="2" * 16,
                created="2026-01-02T00:00:00+00:00",
                training=_history(model="operator"),
            ),
        )
        _write(
            tmp_path,
            "c-3",
            _record(
                _healthy(),
                run_id="3" * 16,
                created="2026-01-03T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        page = _page(tmp_path)

        assert page.headlines.usable_predictor == "operator"
        assert page.headlines.usable_steps == pytest.approx(0.5 / DT)
        assert page.headlines.usable_of == ROLLOUT_STEPS

    def test_a_surrogate_that_did_not_complete_cannot_hold_the_headline(
        self, tmp_path: Path
    ) -> None:
        # Its horizon is a mean over rollouts, some of which blew up, so quoting it would
        # claim a stretch the run never sustained.
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-2",
            _record(
                [_predictor("operator", TRAINED_ON_SPLIT, horizon=0.9, stop="diverged")],
                run_id="2" * 16,
                created="2026-01-02T00:00:00+00:00",
                training=_history(model="operator"),
            ),
        )
        _write(
            tmp_path,
            "c-3",
            _record(
                _healthy(),
                run_id="3" * 16,
                created="2026-01-03T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        page = _page(tmp_path)

        assert page.headlines.usable_predictor == "graph"

    def test_a_fault_run_is_never_quoted(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-fault-wrong_regime-2",
            _record(
                [_predictor("graph", TRAINED_ON_SPLIT, horizon=0.99)],
                run_id="2" * 16,
                created="2026-01-02T00:00:00+00:00",
                training=_history(),
            ),
        )
        _write(
            tmp_path,
            "c-3",
            _record(
                _healthy(),
                run_id="3" * 16,
                created="2026-01-03T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        page = _page(tmp_path)

        assert page.headlines.usable_steps == pytest.approx(0.25 / DT)
        assert len(page.reported) == 2

    def test_the_worst_held_out_completion_wins(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-2",
            _record(
                [
                    _predictor("operator", TRAINED_ON_SPLIT),
                    _predictor("operator", HELD_OUT_SPLIT, stop="diverged"),
                ],
                run_id="2" * 16,
                created="2026-01-02T00:00:00+00:00",
                training=_history(model="operator"),
            ),
        )
        _write(
            tmp_path,
            "c-3",
            _record(
                _healthy(),
                run_id="3" * 16,
                created="2026-01-03T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        page = _page(tmp_path)

        assert page.headlines.held_out_completed == 0
        assert page.headlines.held_out_of == 4
        assert page.headlines.held_out_predictor == "operator"

    def test_the_largest_slowdown_wins(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "a-1",
            _record(
                _healthy(), run_id="1" * 16, benchmark=_speed_report(speedup=0.5), training=None
            ),
        )
        _write(
            tmp_path,
            "b-2",
            _record(
                _healthy(),
                run_id="2" * 16,
                created="2026-01-02T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        page = _page(tmp_path)

        assert page.headlines.slowdown == pytest.approx(20.0)

    def test_no_surrogate_at_all_is_an_error_rather_than_a_blank(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record([_predictor(PERSISTENCE, TRAINED_ON_SPLIT)]))

        with pytest.raises(ConfigurationError, match="completed its trained on split"):
            _page(tmp_path)

    def test_no_benchmark_at_all_is_an_error_rather_than_a_blank(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))

        with pytest.raises(ConfigurationError, match="slower than the solver"):
            _page(tmp_path)


class TestBuildPage:
    def test_an_empty_root_names_the_directory_it_looked_in(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match=str(tmp_path.name)):
            _page(tmp_path)

    def test_the_same_root_builds_the_same_model_twice(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-2",
            _record(
                _healthy(),
                run_id="2" * 16,
                created="2026-01-02T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        assert _page(tmp_path) == _page(tmp_path)

    def test_the_runs_are_ordered_by_when_they_ran(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "z-late",
            _record(
                _healthy(),
                run_id="9" * 16,
                created="2026-01-09T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )
        _write(
            tmp_path,
            "a-early",
            _record(
                _healthy(),
                run_id="1" * 16,
                created="2026-01-01T00:00:00+00:00",
                training=_history(),
            ),
        )

        assert [run.run_id for run in _page(tmp_path).runs] == ["1" * 16, "9" * 16]

    def test_two_runs_that_finished_together_are_ordered_by_identifier(
        self, tmp_path: Path
    ) -> None:
        _write(
            tmp_path,
            "b",
            _record(
                _healthy(),
                run_id="b" * 16,
                benchmark=_speed_report(speedup=0.05),
            ),
        )
        _write(tmp_path, "a", _record(_healthy(), run_id="a" * 16, training=_history()))

        assert [run.run_id for run in _page(tmp_path).runs] == ["a" * 16, "b" * 16]

    def test_the_reported_runs_leave_the_fault_runs_out(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "a-fault-wrong_regime-2",
            _record(_healthy(), run_id="2" * 16, created="2026-01-02T00:00:00+00:00"),
        )
        _write(
            tmp_path,
            "c-3",
            _record(
                _healthy(),
                run_id="3" * 16,
                created="2026-01-03T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        page = _page(tmp_path)

        assert len(page.runs) == 3
        assert [run.run_id for run in page.reported] == ["1" * 16, "3" * 16]

    def test_the_benchmarks_are_the_ladders_the_charts_draw(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-2",
            _record(
                _healthy(),
                run_id="2" * 16,
                created="2026-01-02T00:00:00+00:00",
                benchmark=_speed_report(speedup=0.05),
            ),
        )

        assert len(_page(tmp_path).benchmarks) == 1

    def test_a_page_built_without_scores_has_no_diagnosis(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), benchmark=_speed_report(speedup=0.05)))

        assert _page(tmp_path).diagnosis is None


SCORES: dict[str, Any] = {
    "created": "2026-01-01T00:00:00+00:00",
    "cards": [
        {
            "source": "agent",
            "model": "a model",
            "outcomes": [
                {"fault": "one", "true_cause": "learning_rate", "rank": 1},
                {"fault": "two", "true_cause": "optimiser_state", "rank": 3},
                {"fault": "three", "true_cause": "training_regime", "rank": None},
                {"fault": "four", "true_cause": "model_symmetry", "rank": 1},
            ],
        },
        {
            "source": "rule_based",
            "model": "worst regressed metric",
            "outcomes": [
                {"fault": "one", "true_cause": "learning_rate", "rank": None},
                {"fault": "two", "true_cause": "optimiser_state", "rank": 4},
                {"fault": "three", "true_cause": "training_regime", "rank": None},
                {"fault": "four", "true_cause": "model_symmetry", "rank": 2},
            ],
        },
    ],
}


def _scores(tmp_path: Path, payload: Any) -> Path:
    """Write a fault scores file."""
    path = tmp_path / "fault-scores.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestFaultScores:
    def test_the_rates_are_counted_over_the_faults(self, tmp_path: Path) -> None:
        diagnosis = read_fault_scores(_scores(tmp_path, SCORES))

        assert diagnosis.agent.faults == 4
        assert diagnosis.agent.top1_count == 2
        assert diagnosis.agent.top3_count == 3
        assert diagnosis.agent.top1 == pytest.approx(0.5)
        assert diagnosis.agent.top3 == pytest.approx(0.75)

    def test_a_cause_never_named_counts_against_neither_position(self, tmp_path: Path) -> None:
        diagnosis = read_fault_scores(_scores(tmp_path, SCORES))

        assert diagnosis.baseline.top1_count == 0
        assert diagnosis.baseline.top3_count == 1

    def test_the_ranks_keep_the_order_the_faults_were_run_in(self, tmp_path: Path) -> None:
        diagnosis = read_fault_scores(_scores(tmp_path, SCORES))

        assert [entry.fault for entry in diagnosis.agent.ranks] == ["one", "two", "three", "four"]

    def test_a_field_the_page_does_not_show_is_ignored(self, tmp_path: Path) -> None:
        payload = json.loads(json.dumps(SCORES))
        payload["cards"][0]["outcomes"][0]["confidence"] = 0.62
        payload["provenance"] = "how this was produced"

        assert read_fault_scores(_scores(tmp_path, payload)).agent.top1_count == 2

    def test_a_missing_baseline_card_is_an_error(self, tmp_path: Path) -> None:
        payload = {"cards": [SCORES["cards"][0]]}

        with pytest.raises(ConfigurationError, match="no rule_based card"):
            read_fault_scores(_scores(tmp_path, payload))

    def test_a_missing_file_names_it(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="cannot read fault scores"):
            read_fault_scores(tmp_path / "absent.json")

    def test_a_file_that_is_not_json_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "fault-scores.json"
        path.write_text("not json", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="not valid JSON"):
            read_fault_scores(path)

    def test_a_file_that_is_not_an_object_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="must hold a JSON object"):
            read_fault_scores(_scores(tmp_path, [1, 2, 3]))

    def test_a_card_with_no_outcomes_is_an_error(self, tmp_path: Path) -> None:
        payload = {"cards": [{"source": "agent", "model": "m", "outcomes": []}]}

        with pytest.raises(ConfigurationError, match="invalid fault scores"):
            read_fault_scores(_scores(tmp_path, payload))

    def test_the_page_carries_both_diagnosers(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), benchmark=_speed_report(speedup=0.05)))
        page = build_page(tmp_path, scores=_scores(tmp_path, SCORES))

        assert page.diagnosis is not None
        assert page.diagnosis.agent.source == "agent"
        assert page.diagnosis.baseline.source == "rule_based"


class TestDriftViewer:
    def test_the_sentinel_is_named_as_the_predictor_names_itself(self) -> None:
        assert NOISE in BROKEN_PREDICTORS
        assert set(VIEWER_BASELINES) <= HARNESS_PREDICTORS

    def test_a_predictor_with_a_plot_on_every_split_is_offered(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))
        _draw(tmp_path, "a-1", *_both("graph"), *_both(PERSISTENCE))

        assert _offering(tmp_path) == (
            ("toy", "graph", TRAINED_ON_SPLIT),
            ("toy", "graph", HELD_OUT_SPLIT),
            ("toy", PERSISTENCE, TRAINED_ON_SPLIT),
            ("toy", PERSISTENCE, HELD_OUT_SPLIT),
        )

    def test_a_predictor_with_a_plot_on_one_split_only_is_left_out(self, tmp_path: Path) -> None:
        # The three controls are independent, so a predictor the reader cannot combine
        # with either setup is not a predictor the reader can be offered.
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))
        _draw(tmp_path, "a-1", (TRAINED_ON_SPLIT, "graph"), *_both(PERSISTENCE))

        assert {predictor for _, predictor, _ in _offering(tmp_path)} == {PERSISTENCE}

    def test_a_split_the_predictor_never_ran_on_is_left_out(self, tmp_path: Path) -> None:
        # A rollout that failed at the first step still leaves a plot of the initial state
        # behind, and a plot of the prediction sitting on the truth shows no drift at all.
        results = (
            _predictor(PERSISTENCE, TRAINED_ON_SPLIT, horizon=0.1),
            _predictor(PERSISTENCE, HELD_OUT_SPLIT, horizon=0.1),
            _predictor("graph", TRAINED_ON_SPLIT, horizon=0.25),
            _predictor("graph", HELD_OUT_SPLIT, stop="failed", metrics=False),
        )
        _write(tmp_path, "a-1", _record(results, training=_history()))
        _draw(tmp_path, "a-1", *_both("graph"), *_both(PERSISTENCE))

        assert {predictor for _, predictor, _ in _offering(tmp_path)} == {PERSISTENCE}

    def test_a_predictor_with_no_plot_is_not_linked_to_a_missing_file(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))
        _draw(tmp_path, "a-1", *_both(PERSISTENCE))

        assert {predictor for _, predictor, _ in _offering(tmp_path)} == {PERSISTENCE}

    def test_every_path_it_offers_exists_on_disk(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))
        _draw(tmp_path, "a-1", *_both("graph"), *_both(PERSISTENCE))
        viewer = _viewer(tmp_path)

        assert viewer is not None
        assert all((tmp_path / frame.image).is_file() for frame in viewer.frames)

    def test_the_run_that_trained_a_predictor_supplies_its_plot(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-2",
            _record(
                _healthy(),
                run_id="2" * 16,
                created="2026-02-01T00:00:00+00:00",
                training=_history(model="operator"),
            ),
        )
        _draw(tmp_path, "a-1", *_both("graph"))
        _draw(tmp_path, "b-2", *_both("graph"))
        viewer = _viewer(tmp_path)

        assert viewer is not None
        assert all(frame.run_id == "1" * 16 for frame in viewer.frames)

    def test_a_baseline_comes_from_the_most_recent_run_that_drew_it(self, tmp_path: Path) -> None:
        # No run trains the free baseline, so the newest one wins: it was drawn by the
        # latest dataset and the latest drawing code.
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "b-2",
            _record(
                _healthy(),
                run_id="2" * 16,
                created="2026-02-01T00:00:00+00:00",
                training=_history(),
            ),
        )
        _draw(tmp_path, "a-1", *_both(PERSISTENCE))
        _draw(tmp_path, "b-2", *_both(PERSISTENCE))
        viewer = _viewer(tmp_path)

        assert viewer is not None
        assert all(frame.run_id == "2" * 16 for frame in viewer.frames)

    def test_a_fault_run_never_supplies_a_frame(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), run_id="1" * 16, training=_history()))
        _write(
            tmp_path,
            "a-fault-broken-2",
            _record(
                _healthy(),
                run_id="2" * 16,
                created="2026-02-01T00:00:00+00:00",
                training=_history(),
            ),
        )
        _draw(tmp_path, "a-1", *_both(PERSISTENCE))
        _draw(tmp_path, "a-fault-broken-2", *_both(PERSISTENCE))
        viewer = _viewer(tmp_path)

        assert viewer is not None
        assert all(frame.run_id == "1" * 16 for frame in viewer.frames)

    def test_a_root_with_no_state_plots_has_no_viewer(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))

        assert _viewer(tmp_path) is None

    def test_the_same_root_gives_the_same_viewer(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))
        _draw(tmp_path, "a-1", *_both("graph"), *_both(PERSISTENCE))

        assert _viewer(tmp_path) == _viewer(tmp_path)

    def test_the_learned_model_is_offered_before_the_baselines(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))
        _draw(tmp_path, "a-1", *_both("graph"), *_both(PERSISTENCE))
        viewer = _viewer(tmp_path)

        assert viewer is not None
        assert viewer.predictors("toy") == ("graph", PERSISTENCE)
        assert viewer.systems == ("toy",)
        assert viewer.splits == (TRAINED_ON_SPLIT, HELD_OUT_SPLIT)

    def test_a_combination_it_does_not_offer_is_not_found(self, tmp_path: Path) -> None:
        _write(tmp_path, "a-1", _record(_healthy(), training=_history()))
        _draw(tmp_path, "a-1", *_both("graph"))
        viewer = _viewer(tmp_path)

        assert viewer is not None
        assert viewer.frame("toy", "graph", TRAINED_ON_SPLIT) is not None
        assert viewer.frame("toy", NOISE, TRAINED_ON_SPLIT) is None
