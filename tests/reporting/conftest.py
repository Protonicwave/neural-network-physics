from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.evals.metrics import NEVER_REACHED, NOT_DETERMINED
from nnphysics.evals.result import (
    InvariantRecord,
    MetricRecord,
    PredictorResult,
    RolloutRecord,
    SuiteResult,
    SuiteSettings,
)
from nnphysics.evals.speed import (
    SpeedPoint,
    SpeedReport,
    cost_accounting,
    matched_speedup,
)
from nnphysics.reporting.environment import EnvironmentRecord
from nnphysics.reporting.record import RunRecord

METRICS = (
    "one_step_error",
    "rollout_error",
    "invariant_drift",
    "symmetry_violation",
    "distribution_drift",
)

CONFIG: dict[str, Any] = {
    "name": "example",
    "seed": 5,
    "system": {"name": "toy"},
    "data": {
        "n_trajectories": 8,
        "n_steps": 8,
        "regimes": ["hot"],
        "held_out_regimes": ["cold"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
    },
    "model": {"name": "placeholder"},
    "evaluation": {"name": "standard", "metrics": list(METRICS), "rollout_steps": 4},
}

ENVIRONMENT = EnvironmentRecord(
    python="3.12.0",
    implementation="CPython",
    platform="Linux 6.1",
    machine="x86_64",
    cpu_count=8,
    packages={"numpy": "2.1.0", "torch": "2.5.0"},
)
"""Fixed rather than read from the machine, so that a rendered report is the same
everywhere and a difference between two of them means something."""


def _metrics(scale: float) -> tuple[MetricRecord, ...]:
    """Every metric a default suite produces, with the errors scaled by a factor."""
    return (
        MetricRecord(
            name="one_step_error",
            scalars={"error": 0.001 * scale, "error.position": 0.002 * scale},
        ),
        MetricRecord(
            name="rollout_error",
            scalars={
                "error.final": 0.4 * scale,
                "error.max": 0.5 * scale,
                "error.mean": 0.2 * scale,
                "duration": 0.04,
                "horizon.0.01": 0.01,
                "horizon.1": NEVER_REACHED,
            },
            series={
                "error": (0.0, 0.1 * scale, 0.25 * scale, 0.4 * scale),
                "time": (0.0, 0.01, 0.02, 0.03),
            },
        ),
        MetricRecord(
            name="invariant_drift",
            scalars={
                "energy.drift": 0.01 * scale,
                "energy.reference_drift": 0.001,
                "energy.scale": 2.0,
                "energy.excess": 0.02 * scale,
                "energy.shortfall": 0.0,
                "energy.violation": 3.0 * scale,
                "worst_violation": 3.0 * scale,
            },
            series={
                "energy.predicted": (-1.0, -0.99, -0.97, -0.94),
                "energy.reference": (-1.0, -1.0, -1.0, -1.0),
            },
        ),
        MetricRecord(
            name="symmetry_violation",
            scalars={
                "rotation.max": 0.05 * scale,
                "rotation.final": 0.04 * scale,
                "rotation.mean": 0.03 * scale,
                "rotation.steps": 4.0,
                "worst": 0.05 * scale,
                "steps": 4.0,
            },
            series={"rotation": (0.0, 0.01, 0.03, 0.05)},
        ),
        MetricRecord(
            name="distribution_drift",
            scalars={
                "position.quantile_distance": 0.07 * scale,
                "position.spread_ratio": 1.1,
                "worst": 0.07 * scale,
                "window": 1.0,
            },
            series={"position.predicted": (-1.0, 0.0, 1.0), "position.reference": (-1.0, 0.0, 1.0)},
        ),
    )


def _rollouts(scale: float, split: str) -> tuple[RolloutRecord, ...]:
    """Two rollouts, each carrying the scalars a spread plot is drawn from."""
    return tuple(
        RolloutRecord(
            trajectory=f"{split}-{index}",
            regime="hot" if split != "held_out" else "cold",
            split=split,
            steps_requested=4,
            steps_completed=4,
            stop_reason="completed",
            seconds=0.01,
            scalars={"rollout_error.error.final": (0.3 + 0.2 * index) * scale},
        )
        for index in range(2)
    )


def _predictor_result(
    predictor: str, split: str, *, scale: float = 1.0, seconds: float = 0.002
) -> PredictorResult:
    """One predictor on one split, every number scaled by a factor."""
    return PredictorResult(
        predictor=predictor,
        spec=predictor,
        split=split,
        regimes=("hot",) if split != "held_out" else ("cold",),
        rollouts=_rollouts(scale, split),
        metrics=_metrics(scale),
        seconds_per_step=seconds,
        completed=True,
    )


def _make_result(
    *,
    scale: float = 1.0,
    seconds: float = 0.002,
    predictors: Sequence[str] = ("reference", "persistence"),
    splits: Sequence[str] = ("test", "held_out"),
    regime_gap: Mapping[str, float] | None = None,
) -> SuiteResult:
    """A suite result carrying every metric the default suite produces."""
    return SuiteResult(
        code_version="0.1.0",
        run_id="0123456789abcdef",
        dataset_id="fedcba9876543210",
        system="toy",
        seed=5,
        settings=SuiteSettings(
            name="standard",
            metrics=METRICS,
            rollout_steps=4,
            n_initial_conditions=2,
            error_thresholds=(0.01, 1.0),
            symmetry_steps=4,
            distribution_window=0.25,
            divergence_factor=1000.0,
        ),
        invariants={
            "hot": (
                InvariantRecord(
                    name="energy",
                    conservation="exact",
                    rtol=1e-6,
                    dimension="M L^2 T^-2",
                ),
            )
        },
        results=tuple(
            _predictor_result(predictor, split, scale=scale, seconds=seconds)
            for split in splits
            for predictor in predictors
        ),
        regime_gap=dict(
            {"persistence.rollout_error.error.final": 0.1 * scale}
            if regime_gap is None
            else regime_gap
        ),
    )


def _make_record(
    *,
    run_id: str = "0123456789abcdef",
    name: str = "example",
    created: str = "2026-01-01T00:00:00+00:00",
    **result: Any,
) -> RunRecord:
    """A run record whose every varying field is fixed."""
    evaluation = _make_result(**result)
    return RunRecord(
        run_id=run_id,
        name=name,
        created=created,
        code_version="0.1.0",
        commit="a" * 40,
        config=RunConfig.model_validate(CONFIG),
        environment=ENVIRONMENT,
        timings={"evaluation": 1.5},
        evaluation=evaluation,
        artefacts={"result": "evaluation-standard.json", "snapshots": "snapshots.npz"},
    )


@pytest.fixture(scope="session")
def make_result() -> Callable[..., SuiteResult]:
    """Build a suite result carrying every metric, with the errors scaled by a factor."""
    return _make_result


@pytest.fixture(scope="session")
def make_record() -> Callable[..., RunRecord]:
    """Build a run record whose every varying field is fixed."""
    return _make_record


@pytest.fixture
def record() -> RunRecord:
    """The baseline record every rendering test starts from."""
    return _make_record()


def _calibration(*, overconfident: bool) -> MetricRecord:
    """What a predictor that states an uncertainty produces, honest or not."""
    factor = 100.0 if overconfident else 1.0
    return MetricRecord(
        name="calibration",
        scalars={
            "steps": 4.0,
            "ece": 0.45 if overconfident else 0.02,
            "coverage": 0.01 if overconfident else 0.68,
            "sharpness": 0.004 / factor,
            "spread_error_correlation": 0.9,
            "horizon.error": 0.02,
            "horizon.spread": NEVER_REACHED if overconfident else 0.01,
            "warning_lead": NOT_DETERMINED if overconfident else 0.01,
        },
        series={
            "reliability.nominal": (0.25, 0.5, 0.75),
            "reliability.empirical": (0.0, 0.01, 0.02) if overconfident else (0.24, 0.51, 0.74),
            "spread": (0.0, 0.02, 0.05, 0.09),
            "error": (0.0, 0.03, 0.08, 0.15),
            "time": (0.0, 0.01, 0.02, 0.03),
        },
    )


def _speed_point(substeps: int, error: float, seconds: float, *, name: str = "reference") -> Any:
    return SpeedPoint(
        label=f"{name}:substeps={substeps}" if substeps else name,
        predictor=name,
        kind="solver" if substeps else "surrogate",
        substeps=substeps,
        error=error,
        seconds_per_step=seconds,
        iqr=seconds * 0.02,
        relative_spread=0.02,
        stable=True,
        completed=True,
    )


def _make_speed_report() -> SpeedReport:
    """A benchmark whose surrogate is faster than the solver setting it matches."""
    ladder = (
        _speed_point(1, 0.8, 0.004),
        _speed_point(2, 0.2, 0.008),
        _speed_point(4, 0.0, 0.016),
    )
    surrogate = _speed_point(0, 0.3, 0.002, name="operator")
    return SpeedReport(
        system="toy",
        split="test",
        steps=4,
        n_initial_conditions=2,
        threads=8,
        trials=7,
        warmup=2,
        steps_per_trial=16,
        dataset_substeps=4,
        ladder=ladder,
        surrogates=(surrogate,),
        matched=(matched_speedup(surrogate, ladder),),
        costs=(
            cost_accounting(
                matched_speedup(surrogate, ladder),
                training_seconds=100.0,
                generation_seconds=20.0,
                steps_per_rollout=4,
            ),
        ),
    )


@pytest.fixture(scope="session")
def make_speed_report() -> Callable[[], SpeedReport]:
    """Build a benchmark carrying a ladder, a surrogate and its accounting."""
    return _make_speed_report


@pytest.fixture
def benchmarked(record: RunRecord) -> RunRecord:
    """The baseline record with a benchmark attached."""
    return record.model_copy(update={"benchmark": _make_speed_report()})


@pytest.fixture
def uncertain_result() -> SuiteResult:
    """A suite result whose two predictors state an uncertainty, one of them dishonestly."""
    base = _make_result(predictors=("calibrated", "overconfident"), splits=("test",))
    return base.model_copy(
        update={
            "settings": base.settings.model_copy(update={"metrics": (*METRICS, "calibration")}),
            "results": tuple(
                entry.model_copy(
                    update={
                        "metrics": (
                            *entry.metrics,
                            _calibration(overconfident=entry.predictor == "overconfident"),
                        )
                    }
                )
                for entry in base.results
            ),
        }
    )
