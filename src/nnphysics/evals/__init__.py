"""Evaluation harness: rollouts, reference and broken predictors, metrics and suites.

This layer imports `core` and `data`. It never imports `systems` and never imports
`models`, and a test enforces that rather than trusting the convention. Everything it
needs about a physical system arrives through the `System` protocol: the invariants the
system declares, the symmetries it declares and the solver it offers as ground truth. A
metric that had to ask which system it was looking at would be evidence that the
abstraction is wrong, and the fix would go there rather than into the metric.

Importing this module registers every metric and every predictor.
"""

from nnphysics.evals.metrics import (
    DEFAULT_ERROR_THRESHOLDS,
    DEFAULT_METRICS,
    METRICS,
    MetricContext,
    build_metrics,
)
from nnphysics.evals.predictors import (
    BROKEN_PREDICTORS,
    PREDICTORS,
    REFERENCE_NAME,
    PredictorContext,
    PredictorSpec,
    Substepped,
    build_predictor,
    parse_spec,
)
from nnphysics.evals.result import (
    RESULT_SCHEMA_VERSION,
    PredictorResult,
    SuiteResult,
    read_result,
    upgrade_result,
    write_result,
)
from nnphysics.evals.rollout import RolloutResult, StopReason, roll_out, roll_out_many
from nnphysics.evals.runner import (
    EvaluationCase,
    build_case_predictor,
    evaluate_predictor,
    load_cases,
    regime_gap,
    run_suite,
)
from nnphysics.evals.snapshots import (
    Snapshot,
    SnapshotSet,
    capture_snapshots,
    read_snapshots,
    write_snapshots,
)

__all__ = [
    "BROKEN_PREDICTORS",
    "DEFAULT_ERROR_THRESHOLDS",
    "DEFAULT_METRICS",
    "METRICS",
    "PREDICTORS",
    "REFERENCE_NAME",
    "RESULT_SCHEMA_VERSION",
    "EvaluationCase",
    "MetricContext",
    "PredictorContext",
    "PredictorResult",
    "PredictorSpec",
    "RolloutResult",
    "Snapshot",
    "SnapshotSet",
    "StopReason",
    "Substepped",
    "SuiteResult",
    "build_case_predictor",
    "build_metrics",
    "build_predictor",
    "capture_snapshots",
    "evaluate_predictor",
    "load_cases",
    "parse_spec",
    "read_result",
    "read_snapshots",
    "regime_gap",
    "roll_out",
    "roll_out_many",
    "run_suite",
    "upgrade_result",
    "write_result",
    "write_snapshots",
]
