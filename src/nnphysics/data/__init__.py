"""Data layer: generating trajectories, storing them, splitting them and loading them.

This layer imports `core` and `systems`. Nothing above it needs to know which system a
dataset holds: a manifest names the system, and the loader hands out named fields.
"""

from nnphysics.data.build import build_dataset, spec_of
from nnphysics.data.dataset import Sample, TrajectoryWindows, make_worker_init
from nnphysics.data.generation import (
    TrajectoryRequest,
    default_workers,
    find_regime,
    generate_trajectory,
)
from nnphysics.data.layout import (
    MANIFEST_NAME,
    NORMALISATION_NAME,
    dataset_dir,
    dataset_id,
    shard_name,
)
from nnphysics.data.manifest import Manifest, RegimeRole, Split, read_manifest, write_manifest
from nnphysics.data.normalisation import (
    FieldStats,
    Normalisation,
    fit_normalisation,
    read_normalisation,
    write_normalisation,
)
from nnphysics.data.spec import TrajectorySpec, trajectory_id, trajectory_stream
from nnphysics.data.splits import make_splits
from nnphysics.data.store import ShardReader, content_hash, write_shard
from nnphysics.data.verify import VerificationReport, verify_dataset

__all__ = [
    "MANIFEST_NAME",
    "NORMALISATION_NAME",
    "FieldStats",
    "Manifest",
    "Normalisation",
    "RegimeRole",
    "Sample",
    "ShardReader",
    "Split",
    "TrajectoryRequest",
    "TrajectorySpec",
    "TrajectoryWindows",
    "VerificationReport",
    "build_dataset",
    "content_hash",
    "dataset_dir",
    "dataset_id",
    "default_workers",
    "find_regime",
    "fit_normalisation",
    "generate_trajectory",
    "make_splits",
    "make_worker_init",
    "read_manifest",
    "read_normalisation",
    "shard_name",
    "spec_of",
    "trajectory_id",
    "trajectory_stream",
    "verify_dataset",
    "write_manifest",
    "write_normalisation",
    "write_shard",
]
