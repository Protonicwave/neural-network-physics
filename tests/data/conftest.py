from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.data.build import build_dataset
from nnphysics.data.layout import MANIFEST_NAME, dataset_dir
from nnphysics.data.manifest import Manifest, read_manifest

ConfigFactory = Callable[..., RunConfig]

NBODY: dict[str, Any] = {
    "name": "test-nbody",
    "seed": 3,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 8,
        "n_steps": 6,
        "dt": 0.05,
        "substeps": 2,
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 3,
    },
    "model": {"name": "placeholder"},
    "evaluation": {"metrics": ["placeholder"]},
}

FLUID: dict[str, Any] = {
    "name": "test-fluid",
    "seed": 5,
    # A 32 by 32 grid is the smallest that resolves every declared regime: the shear
    # layer must be thicker than a cell and the turbulence peak must sit below the
    # dealiasing cutoff.
    "system": {"name": "fluid", "parameters": {"grid_size": 32}},
    "data": {
        "n_trajectories": 4,
        "n_steps": 4,
        "dt": 0.05,
        # The tightest advective limit on a 32 grid is 0.0258, in the turbulent regime.
        "substeps": 4,
        "regimes": ["taylor_green", "decaying_turbulence"],
        "held_out_regimes": ["shear_layer"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 2,
    },
    "model": {"name": "placeholder"},
    "evaluation": {"metrics": ["placeholder"]},
}


def _build_config(base: dict[str, Any], root: Path, **overrides: Any) -> RunConfig:
    data = {**base["data"], "root": str(root), **overrides}
    return RunConfig.model_validate({**base, "data": data})


@pytest.fixture
def nbody_config(tmp_path: Path) -> ConfigFactory:
    def make(root: Path | None = None, **overrides: Any) -> RunConfig:
        return _build_config(NBODY, root or tmp_path, **overrides)

    return make


@pytest.fixture
def fluid_config(tmp_path: Path) -> ConfigFactory:
    def make(root: Path | None = None, **overrides: Any) -> RunConfig:
        return _build_config(FLUID, root or tmp_path, **overrides)

    return make


@pytest.fixture(scope="session")
def nbody_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    """A small generated N-body dataset, shared by every test that only reads one."""
    root = tmp_path_factory.mktemp("shared-nbody")
    config = _build_config(NBODY, root)
    directory = build_dataset(config)
    return directory, read_manifest(directory / MANIFEST_NAME)


@pytest.fixture
def built_nbody(nbody_config: ConfigFactory) -> tuple[Path, Manifest, RunConfig]:
    """A freshly generated N-body dataset that a test may modify."""
    config = nbody_config()
    directory = build_dataset(config)
    assert directory == dataset_dir(config)
    return directory, read_manifest(directory / MANIFEST_NAME), config
