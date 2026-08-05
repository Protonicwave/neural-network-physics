from collections.abc import Callable
from pathlib import Path

import pytest

from nnphysics.core.config import RunConfig
from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.data.build import build_dataset, dataset_hash, spec_of
from nnphysics.data.layout import MANIFEST_NAME
from nnphysics.data.manifest import Manifest, RegimeRole, Split, read_manifest

ConfigFactory = Callable[..., RunConfig]
Built = tuple[Path, Manifest, RunConfig]


def test_generating_twice_with_the_same_seed_gives_the_same_content(
    nbody_config: ConfigFactory, tmp_path: Path
) -> None:
    first = read_manifest(build_dataset(nbody_config(tmp_path / "one")) / MANIFEST_NAME)
    second = read_manifest(build_dataset(nbody_config(tmp_path / "two")) / MANIFEST_NAME)

    assert first.dataset_hash == second.dataset_hash
    assert [record.content_hash for record in first.shards] == [
        record.content_hash for record in second.shards
    ]


def test_a_different_seed_gives_different_content(
    nbody_config: ConfigFactory, tmp_path: Path
) -> None:
    baseline = nbody_config(tmp_path / "one")
    reseeded = nbody_config(tmp_path / "two").model_copy(update={"seed": baseline.seed + 1})
    first = read_manifest(build_dataset(baseline) / MANIFEST_NAME)
    second = read_manifest(build_dataset(reseeded) / MANIFEST_NAME)
    assert first.dataset_hash != second.dataset_hash


def test_the_worker_count_does_not_change_the_content(
    nbody_config: ConfigFactory, tmp_path: Path
) -> None:
    serial = read_manifest(build_dataset(nbody_config(tmp_path / "one")) / MANIFEST_NAME)
    parallel = read_manifest(
        build_dataset(nbody_config(tmp_path / "two", workers=2)) / MANIFEST_NAME
    )
    assert serial.dataset_hash == parallel.dataset_hash


def test_the_manifest_records_both_time_intervals(built_nbody: Built) -> None:
    _, manifest, config = built_nbody
    spec = spec_of(config)
    assert manifest.spec.dt == config.data.dt
    assert manifest.spec.substeps == config.data.substeps
    assert manifest.spec.solver_dt == pytest.approx(spec.solver_dt)
    assert manifest.spec.solver_dt < manifest.spec.dt


def test_the_manifest_records_provenance(built_nbody: Built) -> None:
    _, manifest, config = built_nbody
    assert manifest.system == config.system.name
    assert manifest.system_parameters == dict(config.system.parameters)
    assert manifest.seed == config.seed
    assert manifest.code_version
    assert "{regime}" in manifest.seed_stream_template


def test_every_generated_trajectory_is_recorded_and_stored(built_nbody: Built) -> None:
    directory, manifest, config = built_nbody
    expected = config.data.n_trajectories * (
        len(config.data.regimes) + len(config.data.held_out_regimes)
    )
    assert len(manifest.trajectories) == expected
    assert sum(record.n_trajectories for record in manifest.shards) == expected
    for record in manifest.shards:
        assert (directory / record.path).is_file()


def test_no_trajectory_appears_in_more_than_one_split(built_nbody: Built) -> None:
    _, manifest, _ = built_nbody
    placed = [name for split in Split for name in manifest.split(split)]
    assert len(set(placed)) == len(placed)
    assert set(placed) == {record.id for record in manifest.trajectories}


def test_no_held_out_regime_appears_in_train_validation_or_test(built_nbody: Built) -> None:
    _, manifest, config = built_nbody
    held_out = set(config.data.held_out_regimes)
    assert held_out
    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
        regimes = {manifest.trajectory(name).regime for name in manifest.split(split)}
        assert not regimes & held_out
    assert {manifest.trajectory(name).regime for name in manifest.split(Split.HELD_OUT)} == held_out


def test_regimes_are_recorded_with_the_role_they_were_generated_for(built_nbody: Built) -> None:
    _, manifest, config = built_nbody
    roles = {record.name: record.role for record in manifest.regimes}
    for name in config.data.regimes:
        assert roles[name] is RegimeRole.TRAIN
    for name in config.data.held_out_regimes:
        assert roles[name] is RegimeRole.HELD_OUT


def test_a_shard_holds_one_regime_only(built_nbody: Built) -> None:
    _, manifest, _ = built_nbody
    for record in manifest.shards:
        regimes = {entry.regime for entry in manifest.trajectories if entry.shard == record.path}
        assert regimes == {record.regime}


def test_shards_are_no_larger_than_configured(built_nbody: Built) -> None:
    _, manifest, config = built_nbody
    for record in manifest.shards:
        assert record.n_trajectories <= config.data.shard_trajectories


def test_the_fluid_system_builds_through_the_same_code(fluid_config: ConfigFactory) -> None:
    config = fluid_config()
    manifest = read_manifest(build_dataset(config) / MANIFEST_NAME)
    assert manifest.system == "fluid"
    shapes = manifest.field_shapes(Split.TRAIN)
    assert shapes == {"vorticity": (32, 32)}
    assert len(manifest.trajectories) == config.data.n_trajectories * 3


def test_an_undeclared_regime_fails_before_any_work(
    nbody_config: ConfigFactory, tmp_path: Path
) -> None:
    config = nbody_config(tmp_path / "nothing", regimes=["cold_collapse", "no_such_regime"])
    with pytest.raises(UnknownNameError):
        build_dataset(config)
    assert not any((tmp_path / "nothing").glob("**/*.h5"))


def test_progress_is_reported_when_asked_for(nbody_config: ConfigFactory) -> None:
    lines: list[str] = []
    build_dataset(nbody_config(), progress=lines.append)
    assert any("Generating" in line for line in lines)
    assert any(".h5" in line for line in lines)


def test_a_dataset_hash_needs_a_shard() -> None:
    with pytest.raises(ValidationError):
        dataset_hash(())


def test_the_dataset_hash_depends_on_every_shard() -> None:
    assert dataset_hash(("a", "b")) != dataset_hash(("b", "a"))
    assert dataset_hash(("a",)) != dataset_hash(("a", "b"))
    # Two shards must not hash the same as one shard whose hash is their concatenation.
    assert dataset_hash(("a", "b")) != dataset_hash(("ab",))
