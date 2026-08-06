"""What a training run recorded about itself, epoch by epoch.

This is the structured log the plan asks for, and it is a value rather than a stream of
printed lines so that it can be embedded in the run record from phase 06 and compared
against another run's. Nothing here reads the clock: durations are measured by the loop
and handed in.

It can also be written on its own. A run that trains and evaluates in one command embeds
it in the record, but an ensemble member is trained without being evaluated, and what it
cost is exactly what the cost accounting in phase 09 needs. A member whose training time
was not written down would be accounted as free.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nnphysics.core.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["EpochRecord", "TrainingHistory", "read_history", "write_history"]


class EpochRecord(BaseModel):
    """One epoch.

    Attributes:
        epoch: Epoch number, counted from zero.
        curriculum_steps: Rollout length this epoch trained against.
        learning_rate: Rate the optimiser stepped at.
        loss: The full objective, averaged over the epoch's batches.
        one_step_error: Normalised mean squared error at the first predicted step.
        multi_step_error: The same, meaned over the rest of the curriculum window.
        physics_penalty: The model's own penalty before weighting, zero for a model that
            declares none.
        gradient_norm: Mean gradient norm before clipping, which is what says whether
            clipping was doing anything.
        validation_error: The number early stopping reads: normalised error over a fixed
            horizon, the same horizon in every epoch.
        seconds: Wall clock for the epoch, training and validation together.
        improved: Whether this epoch became the best so far.
    """

    # A run that diverged records a loss or a validation error that is not a finite
    # number, and that is the truth about it rather than a value to clean up. Pydantic
    # writes such a number as `null` by default, which then fails to read back as a
    # float, so a diverged run would write a record nobody could open. `constants` writes
    # `NaN` and `Infinity`, which the reader's `json.loads` accepts.
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    epoch: int = Field(ge=0)
    curriculum_steps: int = Field(ge=1)
    learning_rate: float = Field(ge=0.0)
    loss: float
    one_step_error: float
    multi_step_error: float
    physics_penalty: float
    gradient_norm: float
    validation_error: float
    seconds: float = Field(ge=0.0)
    improved: bool


class TrainingHistory(BaseModel):
    """Everything one training run is, apart from its weights.

    Attributes:
        model: Registered model name.
        n_parameters: Trainable parameters.
        train_windows: Windows the training split offered per epoch.
        validation_windows: Windows the validation split offered.
        validation_steps: Horizon the validation metric was measured over.
        epochs: One record per epoch actually run.
        best_epoch: Epoch with the lowest validation error.
        best_validation_error: That error.
        stopped_early: Whether patience ran out before the configured epochs did.
        seconds: Wall clock for the whole run.
    """

    # `constants` for the same reason `EpochRecord` uses it: a number that is not finite
    # is a measurement, and the default serialisation loses it.
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    model: str = Field(min_length=1)
    n_parameters: int = Field(ge=0)
    train_windows: int = Field(ge=1)
    validation_windows: int = Field(ge=1)
    validation_steps: int = Field(ge=1)
    epochs: tuple[EpochRecord, ...] = Field(min_length=1)
    best_epoch: int = Field(ge=0)
    best_validation_error: float
    stopped_early: bool
    seconds: float = Field(ge=0.0)

    @property
    def final(self) -> EpochRecord:
        """The last epoch run."""
        return self.epochs[-1]


def write_history(path: Path, history: TrainingHistory) -> None:
    """Write a training history on its own.

    Args:
        path: File to write. Its parent must exist.
        history: The history.
    """
    path.write_text(history.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_history(path: Path) -> TrainingHistory:
    """Read a training history written on its own.

    Args:
        path: File to read.

    Returns:
        The history.

    Raises:
        ConfigurationError: If the file is missing, is not valid JSON, or does not
            describe a training run.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"cannot read training history {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{path} is not valid JSON: {error}") from error
    try:
        return TrainingHistory.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid training history in {path}: {error}") from error
