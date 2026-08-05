"""Where a dataset lives on disk.

A dataset directory is named after a hash of everything that decides its contents: the
system, its parameters, the generation settings and the seed. Two configurations that
would produce different data therefore cannot share a directory, and re-running the same
configuration finds the data it already generated. The root is deliberately left out of
the hash, so moving a data tree does not invalidate it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nnphysics.core.config import RunConfig

__all__ = [
    "MANIFEST_NAME",
    "NORMALISATION_NAME",
    "dataset_dir",
    "dataset_id",
    "shard_name",
]

MANIFEST_NAME = "manifest.json"
"""Provenance record. A dataset directory without one is not a dataset."""

NORMALISATION_NAME = "normalisation.json"
"""Statistics fitted to the training split, written separately from the manifest because
they are derived from the data rather than from the configuration."""

_ID_LENGTH = 8


def dataset_id(config: RunConfig) -> str:
    """Identify a dataset by everything that decides its contents.

    Args:
        config: The resolved run configuration.

    Returns:
        A short hexadecimal identifier.
    """
    data = config.data.model_dump(mode="json")
    data.pop("root", None)
    canonical = json.dumps(
        {"system": config.system.model_dump(mode="json"), "data": data, "seed": config.seed},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=_ID_LENGTH).hexdigest()


def dataset_dir(config: RunConfig) -> Path:
    """Directory the dataset for a configuration belongs in.

    Args:
        config: The resolved run configuration.

    Returns:
        The directory, which may not exist yet.
    """
    return config.data.root / f"{config.system.name}-{dataset_id(config)}"


def shard_name(regime: str, index: int) -> str:
    """Name one shard file.

    Shards do not span regimes: two regimes of the same system may give states of
    different shapes, for example a different number of bodies, and a shard holds one
    stacked array per field.

    Args:
        regime: Name of the regime the shard holds.
        index: Position of the shard within that regime.

    Returns:
        The file name.
    """
    return f"{regime}-{index:04d}.h5"
