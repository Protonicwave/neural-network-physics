"""Turning a configuration into a dataset on disk.

The order is fixed and matters: generate every regime, write shards as they fill, then
split, then write the manifest. The manifest is written last because it is the claim
that the dataset is complete, and a manifest that named a shard that was never written
would be a worse failure than no manifest at all.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from nnphysics import __version__
from nnphysics.core.errors import ValidationError
from nnphysics.core.seeding import numpy_generator
from nnphysics.data.generation import (
    batched,
    default_workers,
    find_regime,
    generate_many,
    iter_requests,
)
from nnphysics.data.layout import MANIFEST_NAME, dataset_dir, dataset_id, shard_name
from nnphysics.data.manifest import (
    Manifest,
    RegimeRecord,
    RegimeRole,
    ShardRecord,
    SpecRecord,
    SplitRecord,
    TrajectoryRecord,
    write_manifest,
)
from nnphysics.data.spec import SEED_STREAM_TEMPLATE, TrajectorySpec, trajectory_id
from nnphysics.data.splits import SPLIT_SEED_STREAM, make_splits
from nnphysics.data.store import content_hash, write_shard
from nnphysics.systems import build_system

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from nnphysics.core.config import RunConfig

__all__ = ["Progress", "build_dataset", "dataset_hash", "spec_of"]

type Progress = Callable[[str], None]
"""Where a build reports what it is doing. The CLI passes a printer, tests pass nothing."""


def spec_of(config: RunConfig) -> TrajectorySpec:
    """Read the trajectory specification out of a configuration.

    Args:
        config: The resolved run configuration.

    Returns:
        The specification.
    """
    return TrajectorySpec(
        n_steps=config.data.n_steps, dt=config.data.dt, substeps=config.data.substeps
    )


def dataset_hash(hashes: tuple[str, ...]) -> str:
    """Combine shard hashes into one hash for the dataset.

    Args:
        hashes: Shard hashes, in shard order.

    Returns:
        The combined hexadecimal digest.

    Raises:
        ValidationError: If no hashes were given.
    """
    if not hashes:
        raise ValidationError("a dataset hash needs at least one shard")
    digest = hashlib.blake2b(digest_size=16)
    for value in hashes:
        # Separated, so that two shards cannot hash the same as one shard whose hash is
        # their concatenation.
        digest.update(value.encode("ascii") + b"\n")
    return digest.hexdigest()


def build_dataset(config: RunConfig, *, progress: Progress | None = None) -> Path:
    """Generate a dataset, write it and record it.

    Args:
        config: The resolved run configuration.
        progress: Called with a line of progress, or `None` to stay quiet.

    Returns:
        The dataset directory.

    Raises:
        NumericalError: If the reference solver diverges.
        UnknownNameError: If a configured system or regime is not declared.
        ValidationError: If the configuration cannot produce a dataset.
    """
    report = progress or (lambda _: None)
    system = build_system(config.system.name, dict(config.system.parameters))
    spec = spec_of(config)
    directory = dataset_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    workers = config.data.workers or default_workers()

    roles = [(name, RegimeRole.TRAIN) for name in config.data.regimes]
    roles += [(name, RegimeRole.HELD_OUT) for name in config.data.held_out_regimes]
    # Resolved before any work starts, so a misspelled regime fails in a second rather
    # than after an hour of generation.
    resolved = {name: find_regime(system, name) for name, _ in roles}

    report(
        f"Generating {len(roles) * config.data.n_trajectories} trajectories of "
        f"{spec.n_steps} states for {system.name} on {workers} worker(s), "
        f"solver step {spec.solver_dt:g}, stored step {spec.dt:g}."
    )

    regimes: list[RegimeRecord] = []
    shards: list[ShardRecord] = []
    trajectories: list[TrajectoryRecord] = []
    by_regime: dict[str, list[str]] = {}

    for name, role in roles:
        regime = resolved[name]
        regimes.append(
            RegimeRecord(
                name=name,
                parameters=dict(regime.parameters),
                role=role,
                n_trajectories=config.data.n_trajectories,
            )
        )
        requests = iter_requests(
            system.name,
            config.system.parameters,
            name,
            config.data.n_trajectories,
            spec,
            config.seed,
        )
        for shard_index, batch in enumerate(batched(requests, config.data.shard_trajectories)):
            file_name = shard_name(name, shard_index)
            generated = generate_many(batch, workers=workers)
            shapes = write_shard(
                directory / file_name,
                generated,
                compression_level=config.data.compression_level,
            )
            shards.append(
                ShardRecord(
                    path=file_name,
                    regime=name,
                    n_trajectories=len(generated),
                    fields=shapes,
                    content_hash=content_hash(directory / file_name),
                )
            )
            for row, request in enumerate(batch):
                identifier = trajectory_id(name, request.index)
                trajectories.append(
                    TrajectoryRecord(
                        id=identifier,
                        regime=name,
                        index=request.index,
                        shard=file_name,
                        row=row,
                    )
                )
                by_regime.setdefault(name, []).append(identifier)
            report(f"  wrote {file_name} with {len(generated)} trajectories")

    members = make_splits(
        {name: by_regime[name] for name in config.data.regimes},
        {name: by_regime[name] for name in config.data.held_out_regimes},
        val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction,
        rng=numpy_generator(config.seed, SPLIT_SEED_STREAM),
    )

    manifest = Manifest(
        code_version=__version__,
        dataset_id=dataset_id(config),
        system=system.name,
        system_parameters=dict(config.system.parameters),
        seed=config.seed,
        seed_stream_template=SEED_STREAM_TEMPLATE,
        spec=SpecRecord(
            n_steps=spec.n_steps, dt=spec.dt, substeps=spec.substeps, solver_dt=spec.solver_dt
        ),
        regimes=tuple(regimes),
        shards=tuple(shards),
        trajectories=tuple(trajectories),
        splits=SplitRecord(
            seed_stream=SPLIT_SEED_STREAM,
            val_fraction=config.data.val_fraction,
            test_fraction=config.data.test_fraction,
            members=members,
        ),
        dataset_hash=dataset_hash(tuple(record.content_hash for record in shards)),
    )
    write_manifest(directory / MANIFEST_NAME, manifest)
    report(f"Wrote {directory / MANIFEST_NAME}, dataset hash {manifest.dataset_hash}.")
    return directory
