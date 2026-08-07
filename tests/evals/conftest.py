from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.protocols import Predictor, System
from nnphysics.core.types import MetricResult, Regime, Rollout, State, Trajectory
from nnphysics.data.build import build_dataset
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import Manifest, read_manifest
from nnphysics.evals.metrics import MetricContext, build_metrics
from nnphysics.evals.predictors import PredictorContext, Substepped, build_predictor, parse_spec
from nnphysics.evals.rollout import roll_out
from nnphysics.systems import build_system

Score = Callable[..., dict[str, MetricResult]]

ALL_METRICS = (
    "one_step_error",
    "rollout_error",
    "invariant_drift",
    "symmetry_violation",
    "distribution_drift",
)

# The smallest settings that still resolve every declared regime: a 32 by 32 grid for the
# fluid, and for the N-body system a solver step short enough that its energy error is the
# integrator's rather than an artefact of the test.
SCENES = (
    ("nbody", {"softening": 0.05}, 0.02, 4),
    ("fluid", {"grid_size": 32}, 0.05, 4),
)

_DEFAULT_STEPS = 32
_DEFAULT_SYMMETRY_STEPS = 8
_DEFAULT_RESOLUTION_STEPS = 4
"""Shorter again: a refined rollout of a 32 by 32 field works on four times the data."""


@dataclass
class _Scene:
    """One system, one regime and the ground truth for it, built once and reused."""

    system: System
    regime: Regime
    dt: float
    substeps: int
    seed: int = 0
    _truth: dict[int, Trajectory] = field(default_factory=dict)

    @property
    def initial(self) -> State:
        return self.system.initial_state(self.regime, np.random.default_rng(self.seed))

    @property
    def reference(self) -> Predictor:
        return Substepped(
            self.system.reference_predictor(self.regime, self.dt / self.substeps), self.substeps
        )

    def truth(self, steps: int) -> Trajectory:
        if steps not in self._truth:
            self._truth[steps] = roll_out(self.reference, self.initial, steps).trajectory
        return self._truth[steps]

    def build(self, spec: str) -> Predictor:
        parsed = parse_spec(spec)
        return build_predictor(
            parsed,
            PredictorContext(
                reference=self.reference,
                state_spec=self.system.state_spec,
                symmetries=self.system.symmetries,
                seed=self.seed,
                stream=f"test.{parsed.name}",
            ),
        )

    def score(
        self,
        spec: str,
        *,
        steps: int = _DEFAULT_STEPS,
        metrics: Sequence[str] = ALL_METRICS,
        **settings: Any,
    ) -> dict[str, MetricResult]:
        predictor = self.build(spec)
        result = roll_out(predictor, self.initial, steps)
        rollout = Rollout(
            predicted=result.trajectory,
            reference=_prefix(self.truth(steps), len(result.trajectory)),
            predictor=result.predictor,
            system=self.system.name,
        )
        context = MetricContext(
            invariants=self.system.invariants(self.regime),
            symmetries=self.system.symmetries,
            refinements=self.system.refinements,
            predictor=predictor,
            **{
                "symmetry_steps": min(_DEFAULT_SYMMETRY_STEPS, steps),
                "resolution_steps": min(_DEFAULT_RESOLUTION_STEPS, steps),
                **settings,
            },
        )
        return {metric.name: metric.compute(rollout) for metric in build_metrics(metrics, context)}


def _prefix(trajectory: Trajectory, length: int) -> Trajectory:
    if length >= len(trajectory):
        return trajectory
    return Trajectory(
        fields={name: array[:length] for name, array in trajectory.fields.items()},
        times=trajectory.times[:length],
    )


def _make_scene(entry: tuple[str, dict[str, Any], float, int]) -> _Scene:
    name, parameters, dt, substeps = entry
    system = build_system(name, parameters)
    return _Scene(system=system, regime=system.regimes[0], dt=dt, substeps=substeps)


@pytest.fixture(scope="module", params=SCENES, ids=[entry[0] for entry in SCENES])
def score(request: pytest.FixtureRequest) -> Score:
    """Roll a predictor out and score it, against each supported system in turn."""
    return _make_scene(request.param).score


@pytest.fixture(scope="module")
def score_nbody() -> Score:
    """The N-body system alone, where every invariant is declared conserved."""
    return _make_scene(SCENES[0]).score


@pytest.fixture(scope="module")
def score_fluid() -> Score:
    """The fluid system alone, where the invariants are declared decaying."""
    return _make_scene(SCENES[1]).score


EVALUATION: dict[str, Any] = {
    "name": "test-suite",
    "metrics": list(ALL_METRICS),
    "predictors": ["reference", "persistence"],
    "rollout_steps": 8,
    "n_initial_conditions": 2,
    "symmetry_steps": 4,
}

DATASET: dict[str, Any] = {
    "name": "test-evals",
    "seed": 3,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 8,
        "n_steps": 10,
        "dt": 0.02,
        "substeps": 2,
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 4,
    },
    "model": {"name": "placeholder"},
    "evaluation": EVALUATION,
}


def make_config(root: Path, **evaluation: Any) -> RunConfig:
    """A run configuration pointed at a temporary data root."""
    return RunConfig.model_validate(
        {
            **DATASET,
            "data": {**DATASET["data"], "root": str(root)},
            "evaluation": {**EVALUATION, **evaluation},
        }
    )


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest, RunConfig]:
    """A small generated N-body dataset, shared by every test that only reads one."""
    config = make_config(tmp_path_factory.mktemp("evals-data"))
    directory = build_dataset(config)
    return directory, read_manifest(directory / MANIFEST_NAME), config


@pytest.fixture
def config_factory(tmp_path: Path) -> Callable[..., RunConfig]:
    """A configuration, with a data root of its own, that a test may vary the suite of."""

    def make(**evaluation: Any) -> RunConfig:
        return make_config(tmp_path, **evaluation)

    return make
