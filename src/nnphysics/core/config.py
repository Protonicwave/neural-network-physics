"""Run configuration: the Pydantic models, the YAML loader and the run identifier.

A run is defined by exactly two things, a configuration and a seed, and the seed lives
in the configuration. Everything downstream is reproducible from this object alone,
which is why it is frozen and why its identifier is a hash of its own contents.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nnphysics.core.errors import ConfigurationError

__all__ = [
    "ENV_NESTING",
    "ENV_PREFIX",
    "DataConfig",
    "EvaluationConfig",
    "ModelConfig",
    "RunConfig",
    "SystemConfig",
    "TrainingConfig",
    "load_run_config",
]

ENV_PREFIX = "NNP_"
"""Prefix marking an environment variable as a configuration override."""

ENV_NESTING = "__"
"""Separator for nested keys, so `NNP_TRAINING__EPOCHS` sets `training.epochs`."""

Fraction01 = Annotated[float, Field(gt=0.0, lt=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
Parameters = Mapping[str, float | int | bool | str]


class FrozenConfig(BaseModel):
    """Base for every configuration model: immutable and intolerant of unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class SystemConfig(FrozenConfig):
    """Which physical system a run uses, and with what parameters.

    Attributes:
        name: Registered system name.
        parameters: System level parameters that are not part of a regime.
    """

    name: NonEmptyStr
    parameters: Parameters = {}


class DataConfig(FrozenConfig):
    """Trajectory generation and splitting.

    Attributes:
        root: Directory holding generated shards and their manifest.
        n_trajectories: Trajectories generated per regime.
        n_steps: Steps recorded per trajectory.
        dt: Time between recorded steps.
        regimes: Regimes that training data is drawn from.
        held_out_regimes: Regimes never seen during training or model selection.
        val_fraction: Share of in distribution trajectories used for validation.
        test_fraction: Share of in distribution trajectories used for testing.
    """

    root: Path = Path("data")
    n_trajectories: PositiveInt = 64
    n_steps: PositiveInt = 256
    dt: PositiveFloat = 0.01
    regimes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    held_out_regimes: tuple[NonEmptyStr, ...] = Field(min_length=1)
    val_fraction: Fraction01 = 0.1
    test_fraction: Fraction01 = 0.1

    @model_validator(mode="after")
    def _check_splits_and_regimes(self) -> Self:
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction and test_fraction must leave a non empty training split")
        overlap = sorted(set(self.regimes) & set(self.held_out_regimes))
        if overlap:
            raise ValueError(f"regimes are both trained on and held out: {overlap}")
        return self


class ModelConfig(FrozenConfig):
    """Which predictor a run trains or evaluates.

    Attributes:
        name: Registered model name.
        hyperparameters: Model specific settings, validated by the model itself.
    """

    name: NonEmptyStr
    hyperparameters: Parameters = {}


class TrainingConfig(FrozenConfig):
    """The training loop.

    Attributes:
        epochs: Passes over the training split.
        batch_size: Samples per optimiser step.
        learning_rate: Initial learning rate.
        weight_decay: L2 penalty applied by the optimiser.
    """

    epochs: PositiveInt = 50
    batch_size: PositiveInt = 32
    learning_rate: PositiveFloat = 1e-3
    weight_decay: float = Field(default=0.0, ge=0.0)


class EvaluationConfig(FrozenConfig):
    """The evaluation suite.

    Attributes:
        metrics: Registered metric names to run.
        rollout_steps: Steps each predictor is rolled out for.
    """

    metrics: tuple[NonEmptyStr, ...] = Field(min_length=1)
    rollout_steps: PositiveInt = 256


class RunConfig(FrozenConfig):
    """Everything one run needs.

    Attributes:
        name: Human readable label for the run.
        seed: Root seed every generator is derived from.
        output_dir: Directory run artefacts are written under.
        system: System selection.
        data: Data generation and splitting.
        model: Model selection.
        training: Training loop settings.
        evaluation: Evaluation suite settings.
    """

    name: NonEmptyStr
    seed: int = Field(default=0, ge=0)
    output_dir: Path = Path("runs")
    system: SystemConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig

    @property
    def run_id(self) -> str:
        """Identifier derived from the resolved configuration, seed included.

        The same configuration always gives the same identifier, and any change to any
        field gives a different one.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()

    @property
    def run_dir(self) -> Path:
        """Directory this run's artefacts belong in."""
        return self.output_dir / f"{self.name}-{self.run_id}"


def load_run_config(path: Path, env: Mapping[str, str] | None = None) -> RunConfig:
    """Read a YAML configuration, apply environment overrides and validate it.

    Args:
        path: Path to the YAML file.
        env: Environment to take overrides from. Variables named `NNP_SECTION__KEY` set
            `section.key`. Pass `None` to apply no overrides.

    Returns:
        The validated, frozen configuration.

    Raises:
        ConfigurationError: If the file is missing, is not valid YAML, is not a mapping,
            carries an override that contradicts its structure, or fails validation.
    """
    raw = _read_yaml(path)
    if env is not None:
        raw = _apply_env_overrides(raw, env)
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid configuration in {path}: {error}") from error


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file that must contain a mapping."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read configuration {path}: {error}") from error
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"{path} must contain a mapping at the top level, got {type(loaded).__name__}"
        )
    return loaded


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Return a copy of `raw` with `NNP_` environment variables applied."""
    merged = json.loads(json.dumps(raw)) if raw else {}
    for variable, value in sorted(env.items()):
        if not variable.startswith(ENV_PREFIX):
            continue
        keys = [part.lower() for part in variable[len(ENV_PREFIX) :].split(ENV_NESTING)]
        if not all(keys):
            raise ConfigurationError(f"malformed configuration override {variable}")
        _set_nested(merged, keys, _parse_override(value), variable)
    return merged


def _parse_override(value: str) -> Any:  # noqa: ANN401
    """Interpret an override with YAML rules, so `3` is an int and `true` is a bool."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_nested(target: dict[str, Any], keys: list[str], value: Any, source: str) -> None:  # noqa: ANN401
    """Set a nested key, failing if the path runs through a non mapping."""
    node = target
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigurationError(
                f"override {source} cannot descend into {key!r}, which is not a mapping"
            )
        node = child
    node[keys[-1]] = value
