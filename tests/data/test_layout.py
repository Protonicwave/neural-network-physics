from collections.abc import Callable
from pathlib import Path

from nnphysics.core.config import RunConfig
from nnphysics.data.layout import dataset_dir, dataset_id, shard_name

ConfigFactory = Callable[..., RunConfig]


def test_the_same_configuration_gives_the_same_directory(nbody_config: ConfigFactory) -> None:
    assert dataset_dir(nbody_config()) == dataset_dir(nbody_config())


def test_a_change_that_changes_the_data_changes_the_identifier(
    nbody_config: ConfigFactory,
) -> None:
    baseline = dataset_id(nbody_config())
    assert dataset_id(nbody_config(n_steps=7)) != baseline
    assert dataset_id(nbody_config(substeps=3)) != baseline
    assert dataset_id(nbody_config(n_trajectories=12)) != baseline


def test_moving_the_root_does_not_change_the_identifier(
    nbody_config: ConfigFactory, tmp_path: Path
) -> None:
    here = nbody_config(tmp_path / "here")
    there = nbody_config(tmp_path / "there")
    assert dataset_id(here) == dataset_id(there)
    assert dataset_dir(here) != dataset_dir(there)


def test_the_seed_is_part_of_the_identifier(nbody_config: ConfigFactory) -> None:
    config = nbody_config()
    reseeded = config.model_copy(update={"seed": config.seed + 1})
    assert dataset_id(reseeded) != dataset_id(config)


def test_shard_names_sort_in_generation_order() -> None:
    names = [shard_name("regime", index) for index in (0, 1, 10)]
    assert names == sorted(names)
    assert names[0].endswith(".h5")
