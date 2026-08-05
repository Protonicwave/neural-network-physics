import json
from pathlib import Path
from typing import Any

import pytest

from nnphysics.core.errors import ConfigurationError
from nnphysics.core.errors import ValidationError as NNPValidationError
from nnphysics.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    RegimeRole,
    Split,
    read_manifest,
    write_manifest,
)

RAW: dict[str, Any] = {
    "schema_version": MANIFEST_SCHEMA_VERSION,
    "code_version": "0.1.0",
    "dataset_id": "abcd1234",
    "system": "nbody",
    "system_parameters": {"softening": 0.05},
    "seed": 3,
    "seed_stream_template": "data.trajectory.{system}.{regime}.{index}",
    "spec": {"n_steps": 4, "dt": 0.1, "substeps": 5, "solver_dt": 0.02},
    "regimes": [
        {"name": "alpha", "parameters": {"a": 1.0}, "role": "train", "n_trajectories": 2},
        {"name": "beta", "parameters": {"b": 2.0}, "role": "held_out", "n_trajectories": 1},
    ],
    "shards": [
        {
            "path": "alpha-0000.h5",
            "regime": "alpha",
            "n_trajectories": 2,
            "fields": {"position": [3, 2]},
            "content_hash": "aa",
        },
        {
            "path": "beta-0000.h5",
            "regime": "beta",
            "n_trajectories": 1,
            "fields": {"position": [5, 2]},
            "content_hash": "bb",
        },
    ],
    "trajectories": [
        {"id": "alpha/00000", "regime": "alpha", "index": 0, "shard": "alpha-0000.h5", "row": 0},
        {"id": "alpha/00001", "regime": "alpha", "index": 1, "shard": "alpha-0000.h5", "row": 1},
        {"id": "beta/00000", "regime": "beta", "index": 0, "shard": "beta-0000.h5", "row": 0},
    ],
    "splits": {
        "seed_stream": "data.split",
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "members": {
            "train": ["alpha/00000"],
            "validation": ["alpha/00001"],
            "held_out": ["beta/00000"],
        },
    },
    "dataset_hash": "cafe",
}


def _manifest(**overrides: Any) -> Manifest:
    return Manifest.model_validate({**RAW, **overrides})


def test_a_manifest_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, _manifest())
    assert read_manifest(path) == _manifest()


def test_lookups_find_what_the_manifest_lists() -> None:
    manifest = _manifest()
    assert manifest.trajectory("beta/00000").row == 0
    assert manifest.regime("beta").role is RegimeRole.HELD_OUT
    assert manifest.split(Split.TRAIN) == ("alpha/00000",)
    assert manifest.split(Split.TEST) == ()


def test_looking_up_something_absent_is_rejected() -> None:
    manifest = _manifest()
    with pytest.raises(NNPValidationError):
        manifest.trajectory("absent/00000")
    with pytest.raises(NNPValidationError):
        manifest.regime("absent")


def test_field_shapes_come_from_the_shards_a_split_uses() -> None:
    manifest = _manifest()
    assert manifest.field_shapes(Split.TRAIN) == {"position": (3, 2)}
    assert manifest.field_shapes(Split.HELD_OUT) == {"position": (5, 2)}


def test_a_split_that_mixes_shapes_is_rejected() -> None:
    manifest = _manifest(
        splits={
            **RAW["splits"],
            "members": {"train": ["alpha/00000", "beta/00000"], "validation": ["alpha/00001"]},
        }
    )
    with pytest.raises(NNPValidationError, match="different shapes"):
        manifest.field_shapes(Split.TRAIN)


def test_an_empty_split_has_no_shapes() -> None:
    with pytest.raises(NNPValidationError, match="no trajectories"):
        _manifest().field_shapes(Split.TEST)


def test_a_trajectory_in_two_splits_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than one split"):
        _manifest(
            splits={
                **RAW["splits"],
                "members": {
                    "train": ["alpha/00000", "alpha/00001"],
                    "validation": ["alpha/00001"],
                    "held_out": ["beta/00000"],
                },
            }
        )


def test_a_trajectory_in_no_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="belong to no split"):
        _manifest(splits={**RAW["splits"], "members": {"train": ["alpha/00000"]}})


def test_a_split_naming_an_unknown_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not list"):
        _manifest(
            splits={
                **RAW["splits"],
                "members": {
                    "train": ["alpha/00000", "ghost/00000"],
                    "validation": ["alpha/00001"],
                    "held_out": ["beta/00000"],
                },
            }
        )


def test_a_trajectory_naming_an_unlisted_shard_is_rejected() -> None:
    with pytest.raises(ValueError, match="unlisted shard"):
        _manifest(
            trajectories=[
                {**RAW["trajectories"][0], "shard": "ghost.h5"},
                *RAW["trajectories"][1:],
            ]
        )


def test_a_duplicated_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        _manifest(trajectories=[RAW["trajectories"][0], *RAW["trajectories"]])


def test_an_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="extra"):
        _manifest(unexpected=1)


def test_a_manifest_from_another_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**RAW, "schema_version": 99}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="schema version"):
        read_manifest(path)


def test_a_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        read_manifest(path)


def test_an_invalid_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**RAW, "seed": -1}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid manifest"):
        read_manifest(path)


def test_a_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        read_manifest(tmp_path / "absent.json")
