"""One suite, two systems, one changed word.

The plan's rule is that the harness never knows which system it is looking at. The test
of that is not a comment: it is running the same configuration against both systems with
only the system name and its parameters changed, and getting a result of the same shape
out of each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.data.build import build_dataset
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import read_manifest
from nnphysics.evals.result import SuiteResult
from nnphysics.evals.runner import run_suite
from nnphysics.systems import build_system

SUITE: dict[str, Any] = {
    "name": "both",
    "metrics": [
        "one_step_error",
        "rollout_error",
        "invariant_drift",
        "symmetry_violation",
        "distribution_drift",
    ],
    "predictors": ["reference", "persistence", "energy_injection", "symmetry_break"],
    "rollout_steps": 5,
    "n_initial_conditions": 2,
    "symmetry_steps": 3,
}

BASE: dict[str, Any] = {
    "name": "both-systems",
    "seed": 1,
    "data": {
        "n_trajectories": 4,
        "n_steps": 6,
        "dt": 0.05,
        "substeps": 4,
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 2,
    },
    "model": {"name": "placeholder"},
    "evaluation": SUITE,
}

# The only difference between the two runs. Everything else, the suite included, is shared.
SYSTEMS: dict[str, dict[str, Any]] = {
    "nbody": {
        "system": {"name": "nbody", "parameters": {"softening": 0.05}},
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
    },
    "fluid": {
        "system": {"name": "fluid", "parameters": {"grid_size": 32}},
        "regimes": ["taylor_green", "decaying_turbulence"],
        "held_out_regimes": ["shear_layer"],
    },
}


def evaluate(name: str, root: Path) -> SuiteResult:
    choice = SYSTEMS[name]
    config = RunConfig.model_validate(
        {
            **BASE,
            "system": choice["system"],
            "data": {
                **BASE["data"],
                "root": str(root),
                "regimes": choice["regimes"],
                "held_out_regimes": choice["held_out_regimes"],
            },
        }
    )
    directory = build_dataset(config)
    return run_suite(
        build_system(config.system.name, config.system.parameters),
        directory,
        read_manifest(directory / MANIFEST_NAME),
        config.evaluation,
        seed=config.seed,
        run_id=config.run_id,
    )


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SuiteResult]:
    root = tmp_path_factory.mktemp("both-systems")
    return {name: evaluate(name, root / name) for name in SYSTEMS}


class TestOneSuiteTwoSystems:
    def test_both_systems_produce_a_result(self, evaluated: dict[str, SuiteResult]) -> None:
        assert {result.system for result in evaluated.values()} == {"nbody", "fluid"}

    def test_the_same_predictors_and_splits_are_scored(
        self, evaluated: dict[str, SuiteResult]
    ) -> None:
        shapes = {
            name: sorted((entry.predictor, entry.split) for entry in result.results)
            for name, result in evaluated.items()
        }

        assert shapes["nbody"] == shapes["fluid"]

    def test_the_same_metrics_come_back(self, evaluated: dict[str, SuiteResult]) -> None:
        names = {
            name: [record.name for record in result.results[0].metrics]
            for name, result in evaluated.items()
        }

        assert names["nbody"] == names["fluid"] == list(SUITE["metrics"])

    def test_each_system_declares_its_own_invariants(
        self, evaluated: dict[str, SuiteResult]
    ) -> None:
        """The suite does not name them. The systems do, and they name different ones."""
        declared = {
            name: {record.name for records in result.invariants.values() for record in records}
            for name, result in evaluated.items()
        }

        assert declared["nbody"] == {"energy", "linear_momentum", "angular_momentum"}
        assert declared["fluid"] == {"energy", "enstrophy"}

    def test_each_system_declares_its_own_conservation(
        self, evaluated: dict[str, SuiteResult]
    ) -> None:
        """And the same metric reads both, which is the abstraction doing its job."""
        kinds = {
            name: {
                record.conservation for records in result.invariants.values() for record in records
            }
            for name, result in evaluated.items()
        }

        assert "decaying" in kinds["fluid"]
        assert "decaying" not in kinds["nbody"]

    @pytest.mark.parametrize("name", list(SYSTEMS))
    def test_the_reference_is_benign_on_both(
        self, evaluated: dict[str, SuiteResult], name: str
    ) -> None:
        entry = evaluated[name].result("reference", "test")

        assert entry.scalar("rollout_error", "error.final") == pytest.approx(0.0, abs=1e-12)
        assert entry.scalar("invariant_drift", "worst_violation") == 0.0
        assert entry.completed

    @pytest.mark.parametrize("name", list(SYSTEMS))
    def test_the_regime_gap_is_reported_on_both(
        self, evaluated: dict[str, SuiteResult], name: str
    ) -> None:
        gap = evaluated[name].regime_gap

        assert gap
        assert any(key.startswith("persistence.rollout_error.") for key in gap)
