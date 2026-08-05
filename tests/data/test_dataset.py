from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import ValidationError
from nnphysics.data.build import build_dataset
from nnphysics.data.dataset import Sample, TrajectoryWindows, make_worker_init
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import Manifest, Split, read_manifest
from nnphysics.data.normalisation import fit_normalisation

ConfigFactory = Callable[..., RunConfig]
Shared = tuple[Path, Manifest]


def _positions(loader: DataLoader[Sample]) -> list[torch.Tensor]:
    return [batch["inputs"]["position"] for batch in loader]


def test_a_single_step_pair_is_one_input_and_one_target(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.TRAIN, sequence_length=1, manifest=manifest)
    sample = windows[0]
    assert sample["inputs"]["position"].shape == (32, 2)
    assert sample["targets"]["position"].shape == (1, 32, 2)
    assert sample["times"].shape == (2,)


def test_a_sequence_carries_the_states_that_follow(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.TRAIN, sequence_length=3, manifest=manifest)
    sample = windows[0]
    assert sample["targets"]["position"].shape == (3, 32, 2)
    assert sample["times"].shape == (4,)
    assert np.allclose(np.diff(sample["times"].numpy()), manifest.spec.dt)


def test_the_target_of_one_window_is_the_input_of_the_next(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.TRAIN, sequence_length=1, manifest=manifest)
    first, second = windows[0], windows[1]
    assert second["start"] == first["start"] + 1
    assert torch.equal(first["targets"]["position"][0], second["inputs"]["position"])


def test_the_dataset_covers_every_window_of_every_trajectory(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    length = 2
    windows = TrajectoryWindows(directory, Split.TRAIN, sequence_length=length, manifest=manifest)
    per_trajectory = manifest.spec.n_steps - length
    assert windows.windows_per_trajectory == per_trajectory
    assert len(windows) == len(manifest.split(Split.TRAIN)) * per_trajectory


def test_a_stride_takes_every_nth_window(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(
        directory, Split.TRAIN, sequence_length=1, stride=2, manifest=manifest
    )
    starts = {windows[index]["start"] for index in range(windows.windows_per_trajectory)}
    assert starts == set(range(0, manifest.spec.n_steps - 1, 2))


def test_normalisation_is_applied_at_load_time(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    normalisation = fit_normalisation(directory, manifest)
    raw = TrajectoryWindows(directory, Split.TRAIN, manifest=manifest)[0]
    normalised = TrajectoryWindows(
        directory, Split.TRAIN, normalisation=normalisation, manifest=manifest
    )[0]

    stats = normalisation.stats["position"]
    expected = (raw["inputs"]["position"].numpy() - stats.mean) / stats.scale
    assert np.allclose(normalised["inputs"]["position"].numpy(), expected, atol=1e-6)


def test_the_held_out_split_loads_through_the_same_class(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.HELD_OUT, manifest=manifest)
    assert windows.field_shapes == {"mass": (4,), "position": (4, 2), "velocity": (4, 2)}
    assert len(windows) == len(manifest.split(Split.HELD_OUT)) * (manifest.spec.n_steps - 1)


def test_the_fluid_dataset_loads_through_the_same_class(fluid_config: ConfigFactory) -> None:
    directory = build_dataset(fluid_config())
    manifest = read_manifest(directory / MANIFEST_NAME)
    windows = TrajectoryWindows(directory, Split.TRAIN, manifest=manifest)
    assert windows[0]["inputs"]["vorticity"].shape == (32, 32)


def test_worker_count_does_not_change_the_epoch(nbody_dataset: Shared) -> None:
    """The definition of done for the loader.

    The dataset draws no random numbers, so the order and contents of an epoch come from
    the sampler in the parent process alone. Two workers must therefore see exactly what
    none do.
    """
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.TRAIN, sequence_length=2, manifest=manifest)
    initialise = make_worker_init(0)

    without = _positions(
        DataLoader(windows, batch_size=5, shuffle=False, num_workers=0, worker_init_fn=initialise)
    )
    with_two = _positions(
        DataLoader(windows, batch_size=5, shuffle=False, num_workers=2, worker_init_fn=initialise)
    )

    assert len(without) == len(with_two)
    for one, other in zip(without, with_two, strict=True):
        assert torch.equal(one, other)


def test_a_shuffled_epoch_is_reproducible_from_a_generator(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.TRAIN, manifest=manifest)

    def epoch(seed: int) -> list[torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return _positions(
            DataLoader(windows, batch_size=4, shuffle=True, generator=generator, num_workers=0)
        )

    for one, other in zip(epoch(17), epoch(17), strict=True):
        assert torch.equal(one, other)


def test_a_batch_stacks_every_field(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.TRAIN, sequence_length=2, manifest=manifest)
    batch = next(iter(DataLoader(windows, batch_size=3, shuffle=False)))
    assert batch["inputs"]["position"].shape == (3, 32, 2)
    assert batch["targets"]["velocity"].shape == (3, 2, 32, 2)
    assert batch["times"].shape == (3, 3)


def test_states_are_single_precision_and_times_are_not(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    sample = TrajectoryWindows(directory, Split.TRAIN, manifest=manifest)[0]
    assert sample["inputs"]["position"].dtype is torch.float32
    assert sample["times"].dtype is torch.float64


def test_an_index_outside_the_dataset_is_rejected(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    windows = TrajectoryWindows(directory, Split.TRAIN, manifest=manifest)
    with pytest.raises(IndexError):
        windows[len(windows)]
    with pytest.raises(IndexError):
        windows[-1]


def test_a_window_longer_than_a_trajectory_is_rejected(nbody_dataset: Shared) -> None:
    directory, manifest = nbody_dataset
    with pytest.raises(ValidationError, match="does not fit"):
        TrajectoryWindows(
            directory, Split.TRAIN, sequence_length=manifest.spec.n_steps, manifest=manifest
        )


@pytest.mark.parametrize(("sequence_length", "stride"), [(0, 1), (1, 0)])
def test_unusable_window_settings_are_rejected(
    nbody_dataset: Shared, sequence_length: int, stride: int
) -> None:
    directory, manifest = nbody_dataset
    with pytest.raises(ValidationError):
        TrajectoryWindows(
            directory,
            Split.TRAIN,
            sequence_length=sequence_length,
            stride=stride,
            manifest=manifest,
        )


def test_worker_initialisation_is_deterministic() -> None:
    def first_draw(seed: int, worker: int) -> int:
        make_worker_init(seed)(worker)
        return int(torch.randint(0, 1 << 30, (1,)).item())

    assert first_draw(4, 0) == first_draw(4, 0)
    assert first_draw(4, 0) != first_draw(4, 1)
    assert first_draw(4, 0) != first_draw(5, 0)
