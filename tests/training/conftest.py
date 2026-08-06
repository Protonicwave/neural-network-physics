from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nnphysics.core.config import RunConfig, TrainingConfig
from nnphysics.data.build import build_dataset
from nnphysics.data.fields import constant_fields
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import Manifest, Split, read_manifest
from nnphysics.data.normalisation import fit_normalisation
from nnphysics.models import ModelContext

# Small enough to train in a second and long enough to hold a curriculum window of four.
DATASET: dict[str, Any] = {
    "name": "test-training",
    "seed": 5,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 8,
        "n_steps": 12,
        "dt": 0.02,
        "substeps": 2,
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 4,
    },
    "model": {"name": "constant"},
    "evaluation": {
        "name": "test-suite",
        "metrics": ["one_step_error", "rollout_error"],
        "predictors": ["reference", "persistence"],
        "rollout_steps": 6,
        "n_initial_conditions": 2,
        "symmetry_steps": 3,
    },
}

TRAINING: dict[str, Any] = {
    "epochs": 2,
    "batch_size": 4,
    "learning_rate": 0.01,
    "warmup_epochs": 0,
    "validation_steps": 2,
    "window_stride": 4,
}

Dataset = tuple[Path, Manifest, RunConfig]
Trainer = Callable[..., TrainingConfig]


def make_config(root: Path, **overrides: Any) -> RunConfig:
    """A run configuration pointed at a temporary data root."""
    return RunConfig.model_validate(
        {
            **DATASET,
            "data": {**DATASET["data"], "root": str(root)},
            "training": {**TRAINING, **overrides},
        }
    )


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A small generated N-body dataset, shared by every test that only reads one."""
    config = make_config(tmp_path_factory.mktemp("training-data"))
    directory = build_dataset(config)
    return directory, read_manifest(directory / MANIFEST_NAME), config


@pytest.fixture(scope="session")
def context(dataset: Dataset) -> ModelContext:
    """A model context describing that dataset, fitted once."""
    directory, manifest, config = dataset
    return ModelContext(
        field_shapes=manifest.field_shapes(Split.TRAIN),
        static_fields=constant_fields(directory, manifest),
        normalisation=fit_normalisation(directory, manifest),
        dt=manifest.spec.dt,
        seed=config.seed,
    )


@pytest.fixture
def training() -> Trainer:
    """Training settings a test may vary one at a time."""

    def make(**overrides: Any) -> TrainingConfig:
        return TrainingConfig.model_validate({**TRAINING, **overrides})

    return make
