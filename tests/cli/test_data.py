from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nnphysics.cli.app import app
from nnphysics.core.config import RunConfig, load_run_config
from nnphysics.data.layout import MANIFEST_NAME, NORMALISATION_NAME, dataset_dir
from nnphysics.data.manifest import read_manifest

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

RAW: dict[str, Any] = {
    "name": "cli-nbody",
    "seed": 2,
    "system": {"name": "nbody", "parameters": {"softening": 0.05}},
    "data": {
        "n_trajectories": 8,
        "n_steps": 4,
        "dt": 0.05,
        "substeps": 2,
        "regimes": ["cold_collapse", "virialised_cluster"],
        "held_out_regimes": ["hierarchical_pair"],
        "val_fraction": 0.25,
        "test_fraction": 0.25,
        "workers": 1,
        "shard_trajectories": 4,
    },
    "model": {"name": "placeholder"},
    "evaluation": {"metrics": ["placeholder"]},
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    raw = {**RAW, "data": {**RAW["data"], "root": str(tmp_path / "data")}}
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.fixture
def generated(runner: CliRunner, config_file: Path) -> Path:
    result = runner.invoke(app, ["data", "generate", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    return dataset_dir(_load(config_file))


def _load(path: Path) -> RunConfig:
    return load_run_config(path)


def test_the_data_group_lists_its_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for name in ("generate", "verify", "stats"):
        assert name in result.stdout


def test_generate_writes_a_dataset_and_a_manifest(generated: Path) -> None:
    assert (generated / MANIFEST_NAME).is_file()
    manifest = read_manifest(generated / MANIFEST_NAME)
    assert len(manifest.trajectories) == 24
    assert all((generated / record.path).is_file() for record in manifest.shards)


def test_verify_passes_on_a_freshly_generated_dataset(
    runner: CliRunner, config_file: Path, generated: Path
) -> None:
    assert generated.is_dir()
    result = runner.invoke(app, ["data", "verify", "--config", str(config_file), "--sample", "2"])
    assert result.exit_code == 0, result.output
    assert "Dataset verified" in result.stdout


def test_verify_fails_on_a_dataset_altered_by_one_byte(
    runner: CliRunner, config_file: Path, generated: Path
) -> None:
    manifest = read_manifest(generated / MANIFEST_NAME)
    shard = generated / manifest.shards[0].path
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF
    shard.write_bytes(bytes(data))

    result = runner.invoke(app, ["data", "verify", "--config", str(config_file), "--sample", "0"])
    assert result.exit_code == 1
    assert "hashes to" in result.output


def test_verify_on_a_dataset_that_was_never_generated_exits_cleanly(
    runner: CliRunner, config_file: Path
) -> None:
    result = runner.invoke(app, ["data", "verify", "--config", str(config_file)])
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_stats_describes_the_dataset_and_writes_the_artefact(
    runner: CliRunner, config_file: Path, generated: Path
) -> None:
    result = runner.invoke(app, ["data", "stats", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "training split alone" in result.stdout
    assert "held_out" in result.stdout
    assert (generated / NORMALISATION_NAME).is_file()


def test_stats_can_describe_without_writing(
    runner: CliRunner, config_file: Path, generated: Path
) -> None:
    result = runner.invoke(app, ["data", "stats", "--config", str(config_file), "--no-write"])
    assert result.exit_code == 0, result.output
    assert not (generated / NORMALISATION_NAME).exists()


def test_a_missing_config_exits_with_a_message(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "generate", "--config", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("name", ["nbody.yaml", "fluid.yaml"])
def test_the_shipped_configurations_are_valid(name: str) -> None:
    config = _load(CONFIGS / name)
    assert config.data.substeps >= 1
    assert config.data.solver_dt < config.data.dt
    assert not set(config.data.regimes) & set(config.data.held_out_regimes)
