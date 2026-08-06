"""Checkpoints that a run can be resumed from exactly.

Saving the weights alone is the usual mistake. An optimiser with momentum carries state
of the same size as the model, and a run resumed without it takes a different step from
the one it would have taken, so the resumed run and the uninterrupted one diverge from
the first batch. That failure is silent, which is why there is a test for it.

Everything a resume needs is here: the weights, the optimiser, the schedule position, the
epoch to start from, the best score so far, how long patience has been running out for,
and the history already recorded. What is deliberately absent is the state of any random
generator, because the loop derives its shuffling from the run seed and the epoch number
rather than from a generator it carries forward.

The payload is a superset of a plain model checkpoint, so `load_model` reads either file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nnphysics.core.errors import ConfigurationError
from nnphysics.models.base import CHECKPOINT_SCHEMA_VERSION, model_payload
from nnphysics.training.history import EpochRecord

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.models.base import SurrogateModel

__all__ = [
    "BEST_NAME",
    "LAST_NAME",
    "CheckpointPaths",
    "TrainingState",
    "load_checkpoint",
    "save_checkpoint",
]

BEST_NAME = "best.pt"
"""The epoch with the lowest validation error. This is the model that gets evaluated."""

LAST_NAME = "last.pt"
"""The most recent epoch. This is the one a resume continues from."""


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    """Where a run's checkpoints live.

    Attributes:
        root: Directory holding them, which may not exist yet.
    """

    root: Path

    @property
    def best(self) -> Path:
        """The best epoch."""
        return self.root / BEST_NAME

    @property
    def last(self) -> Path:
        """The most recent epoch."""
        return self.root / LAST_NAME

    def ensure(self) -> None:
        """Create the directory."""
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Where a run had got to when a checkpoint was written.

    Attributes:
        epoch: Epochs completed, which is the epoch a resume starts at.
        best_error: Lowest validation error seen.
        best_epoch: Epoch that produced it.
        stalled: Epochs since the last improvement.
        history: Every epoch recorded so far.
        seconds: Wall clock spent so far.
    """

    epoch: int
    best_error: float
    best_epoch: int
    stalled: int
    history: tuple[EpochRecord, ...]
    seconds: float


def save_checkpoint(
    path: Path,
    model: SurrogateModel,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
) -> None:
    """Write everything a resume needs.

    Args:
        path: File to write. Its parent must exist.
        model: The model.
        optimiser: The optimiser, whose state is the part most easily forgotten.
        scheduler: The learning rate schedule.
        state: Where the run had got to.
    """
    payload = model_payload(model)
    payload["training"] = {
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": state.epoch,
        "best_error": state.best_error,
        "best_epoch": state.best_epoch,
        "stalled": state.stalled,
        "seconds": state.seconds,
        "history": [record.model_dump() for record in state.history],
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: SurrogateModel,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> TrainingState:
    """Restore a run in place and report where it had got to.

    The model, the optimiser and the schedule are the ones already built, so a resume
    uses the configuration it was given and the checkpoint only supplies the numbers.
    A checkpoint that does not fit them is an error rather than a partial load.

    Args:
        path: File to read.
        model: The model to load weights into.
        optimiser: The optimiser to load state into.
        scheduler: The schedule to load state into.

    Returns:
        Where the run had got to.

    Raises:
        ConfigurationError: If the file is missing, is of another schema version, carries
            no training state, or does not fit what it is being loaded into.
    """
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise ConfigurationError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ConfigurationError(f"{path} is not a version {CHECKPOINT_SCHEMA_VERSION} checkpoint")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ConfigurationError(
            f"{path} carries weights but no training state, so it cannot be resumed from"
        )
    try:
        model.load_state_dict(payload["state_dict"])
        optimiser.load_state_dict(training["optimiser"])
        scheduler.load_state_dict(training["scheduler"])
        return TrainingState(
            epoch=int(training["epoch"]),
            best_error=float(training["best_error"]),
            best_epoch=int(training["best_epoch"]),
            stalled=int(training["stalled"]),
            history=tuple(EpochRecord.model_validate(entry) for entry in training["history"]),
            seconds=float(training["seconds"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"{path} is a malformed checkpoint: {error}") from error
    except RuntimeError as error:
        raise ConfigurationError(
            f"{path} does not fit the model it is loaded into: {error}"
        ) from error
