from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nnphysics.agent.client import Reply, Usage
from nnphysics.core.config import RunConfig
from nnphysics.evals.result import (
    InvariantRecord,
    MetricRecord,
    PredictorResult,
    RolloutRecord,
    SuiteResult,
    SuiteSettings,
)
from nnphysics.reporting.environment import EnvironmentRecord
from nnphysics.reporting.record import RunRecord
from nnphysics.training.history import EpochRecord, TrainingHistory

FIXTURES = Path(__file__).parent / "fixtures"

METRICS = ("one_step_error", "rollout_error", "invariant_drift", "symmetry_violation")

CONFIG: dict[str, Any] = {
    "name": "example",
    "seed": 5,
    "system": {"name": "toy"},
    "data": {
        "n_trajectories": 8,
        "n_steps": 8,
        "substeps": 4,
        "regimes": ["hot"],
        "held_out_regimes": ["cold"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
    },
    "model": {"name": "placeholder"},
    "training": {"epochs": 4, "curriculum": [1, 2], "curriculum_epochs": [0, 2]},
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
"""Fixed rather than read from the machine, so a context built twice is the same twice."""


def _metrics(scale: float) -> tuple[MetricRecord, ...]:
    return (
        MetricRecord(name="one_step_error", scalars={"error": 0.001 * scale}),
        MetricRecord(
            name="rollout_error",
            scalars={
                "error.final": 0.4 * scale,
                "error.max": 0.5 * scale,
                "error.mean": 0.2 * scale,
                "duration": 0.04,
                "horizon.0.1": 0.02,
            },
            series={
                # Starts at zero, because every predictor is exactly right at the initial
                # condition. The reduction is supposed to drop that point.
                "error": (0.0, 0.1 * scale, 0.25 * scale, 0.4 * scale),
                "time": (0.0, 0.01, 0.02, 0.03),
            },
        ),
        MetricRecord(
            name="invariant_drift",
            scalars={
                "energy.drift": 0.01 * scale,
                "energy.violation": 3.0 * scale,
                "worst_violation": 3.0 * scale,
            },
        ),
        MetricRecord(
            name="symmetry_violation",
            scalars={"rotation.max": 0.05 * scale, "worst": 0.05 * scale, "steps": 4.0},
        ),
    )


def _rollouts(split: str, *, completed: int) -> tuple[RolloutRecord, ...]:
    return tuple(
        RolloutRecord(
            trajectory=f"{split}-{index}",
            regime="hot" if split != "held_out" else "cold",
            split=split,
            steps_requested=4,
            steps_completed=4 if index < completed else 2,
            stop_reason="completed" if index < completed else "diverged",
            seconds=0.01,
        )
        for index in range(2)
    )


def _result(
    *,
    scale: float = 1.0,
    seconds: float = 0.002,
    predictors: Sequence[str] = ("reference", "graph"),
    splits: Sequence[str] = ("test", "held_out"),
    completed: int = 2,
) -> SuiteResult:
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
            error_thresholds=(0.1,),
            symmetry_steps=4,
            distribution_window=0.25,
            divergence_factor=1000.0,
        ),
        invariants={
            "hot": (
                InvariantRecord(
                    name="energy", conservation="exact", rtol=1e-6, dimension="M L^2 T^-2"
                ),
            )
        },
        results=tuple(
            PredictorResult(
                predictor=predictor,
                spec=predictor,
                split=split,
                regimes=("hot",) if split != "held_out" else ("cold",),
                rollouts=_rollouts(split, completed=2 if predictor == "reference" else completed),
                metrics=_metrics(scale),
                seconds_per_step=seconds,
                completed=predictor == "reference" or completed == 2,
            )
            for split in splits
            for predictor in predictors
        ),
    )


def _history(*, loss: float, validation: float, curriculum: Sequence[int]) -> TrainingHistory:
    epochs = tuple(
        EpochRecord(
            epoch=index,
            curriculum_steps=curriculum[min(index, len(curriculum) - 1)],
            learning_rate=0.001,
            loss=loss * (1.0 - 0.1 * index),
            one_step_error=loss,
            multi_step_error=loss,
            physics_penalty=0.0,
            gradient_norm=0.5 + 0.1 * index,
            validation_error=validation * (1.0 - 0.05 * index),
            seconds=1.0,
            improved=index == 3,
        )
        for index in range(4)
    )
    return TrainingHistory(
        model="placeholder",
        n_parameters=1234,
        train_windows=8,
        validation_windows=2,
        validation_steps=2,
        epochs=epochs,
        best_epoch=3,
        best_validation_error=epochs[-1].validation_error,
        stopped_early=False,
        seconds=4.0,
    )


def _record(  # noqa: PLR0913
    # Every field a context reads, so that one test can vary one of them.
    *,
    run_id: str = "0123456789abcdef",
    name: str = "example",
    created: str = "2026-01-01T00:00:00+00:00",
    config: Mapping[str, Any] | None = None,
    trained: bool = True,
    loss: float = 0.5,
    validation: float = 0.2,
    curriculum: Sequence[int] = (1, 2),
    **result: Any,
) -> RunRecord:
    """A run record whose every varying field is fixed."""
    return RunRecord(
        run_id=run_id,
        name=name,
        created=created,
        code_version="0.1.0",
        commit="a" * 40,
        config=RunConfig.model_validate(dict(config or CONFIG)),
        environment=ENVIRONMENT,
        timings={"evaluation": 1.5},
        training=_history(loss=loss, validation=validation, curriculum=curriculum)
        if trained
        else None,
        evaluation=_result(**result),
        artefacts={"result": "evaluation-standard.json"},
    )


def _namespace(payload: Any) -> Any:
    """Turn a recorded JSON payload into something with the SDK's attribute access.

    The SDK returns objects, not dictionaries, and the client reads them with attribute
    access. Replaying a recording therefore means replaying its shape as well as its
    contents, or the test would be exercising a code path the real client never takes.
    """
    if isinstance(payload, dict):
        return SimpleNamespace(**{key: _namespace(value) for key, value in payload.items()})
    if isinstance(payload, list):
        return [_namespace(item) for item in payload]
    return payload


def _recorded_message(name: str = "diagnosis-reply") -> Any:
    """One recorded API response, as the SDK would have handed it over."""
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    message = _namespace(payload)
    # `input` on a tool use block is a plain mapping in the SDK, not an object.
    for block in message.content:
        if getattr(block, "type", "") == "tool_use":
            block.input = payload["content"][-1]["input"]
    return message


def _recorded_reply(name: str = "diagnosis-reply") -> Reply:
    """The same recording, already parsed, for a test that is not testing the parsing."""
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    arguments = next(block["input"] for block in payload["content"] if block["type"] == "tool_use")
    return Reply(
        arguments=arguments,
        usage=Usage(
            input_tokens=payload["usage"]["input_tokens"],
            output_tokens=payload["usage"]["output_tokens"],
        ),
        model=payload["model"],
        attempts=1,
    )


@pytest.fixture
def make_record() -> Callable[..., RunRecord]:
    """Build a run record whose every varying field is fixed."""
    return _record


@pytest.fixture
def recorded_message() -> Callable[..., Any]:
    """Load a recorded API response in the shape the SDK returns."""
    return _recorded_message


@pytest.fixture
def recorded_reply() -> Callable[..., Reply]:
    """Load a recorded API response, already parsed."""
    return _recorded_reply
