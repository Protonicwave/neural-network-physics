"""The parts of the pipeline that more than one command needs.

`nnp train` and the fault suite both have to find a dataset, fit or read its normalisation
statistics and build the context a model is constructed from. Those three steps have
nothing to do with which command asked for them, so they live here rather than in
whichever command was written first.

Everything here touches the filesystem, which is why it is in the command line layer and
not in the layers it calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nnphysics.data.build import build_dataset
from nnphysics.data.fields import constant_fields
from nnphysics.data.layout import MANIFEST_NAME, NORMALISATION_NAME, dataset_dir
from nnphysics.data.manifest import Manifest, Split, read_manifest
from nnphysics.data.normalisation import (
    Normalisation,
    fit_normalisation,
    read_normalisation,
    write_normalisation,
)
from nnphysics.models import ModelContext

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.core.config import RunConfig

__all__ = ["Echo", "dataset_normalisation", "ensure_dataset", "model_context"]

Echo = Callable[[str], None]
"""How a step says what it is doing. Passed in, so that nothing here prints on its own."""


def ensure_dataset(config: RunConfig, *, echo: Echo) -> tuple[Path, Manifest]:
    """Find the dataset a configuration names, generating it if it is not there.

    A dataset directory is named after a hash of everything that decides its contents, so
    a configuration that changed the regimes or the substeps names a directory that does
    not exist yet. Generating it rather than failing is what lets a fault that changes the
    data be run by the same command as one that does not.

    Args:
        config: The resolved run configuration.
        echo: How to report progress.

    Returns:
        The dataset directory and its manifest.

    Raises:
        ValidationError: If generation fails.
        ConfigurationError: If the manifest cannot be read.
    """
    directory = dataset_dir(config)
    if not (directory / MANIFEST_NAME).is_file():
        echo(f"No dataset at {directory}, generating it.")
        directory = build_dataset(config)
    return directory, read_manifest(directory / MANIFEST_NAME)


def dataset_normalisation(directory: Path, manifest: Manifest, *, echo: Echo) -> Normalisation:
    """The training split's statistics, fitted now if `nnp data stats` never was.

    Fitting here rather than failing keeps the pipeline to one command, and it cannot
    reach the wrong data: the fit reads the training split whichever path it is reached
    by, and the artefact it writes is the one `nnp data stats` would have written.

    Args:
        directory: The dataset directory.
        manifest: Its manifest.
        echo: How to report progress.

    Returns:
        The statistics.
    """
    path = directory / NORMALISATION_NAME
    if path.is_file():
        return read_normalisation(path)
    echo(f"No {NORMALISATION_NAME} beside the data, fitting it from the training split.")
    statistics = fit_normalisation(directory, manifest)
    write_normalisation(path, statistics)
    return statistics


def model_context(
    directory: Path,
    manifest: Manifest,
    config: RunConfig,
    normalisation: Normalisation,
) -> ModelContext:
    """Everything the model needs to know about the data it will be trained on.

    Args:
        directory: The dataset directory.
        manifest: Its manifest.
        config: The resolved run configuration.
        normalisation: Statistics the model normalises with. Passed in rather than read
            here, so that a caller can hand over statistics that are deliberately wrong.

    Returns:
        The context.
    """
    return ModelContext(
        field_shapes=manifest.field_shapes(Split.TRAIN),
        static_fields=constant_fields(directory, manifest),
        normalisation=normalisation,
        dt=manifest.spec.dt,
        seed=config.run_seed,
    )
