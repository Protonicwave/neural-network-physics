"""The rollout curriculum: how many steps an epoch trains against.

This is the part of the loop that decides whether a surrogate is stable over a long
horizon. A model trained only on single steps is optimised for a problem it will never be
asked: at evaluation it is handed its own output, which no single step of training ever
was, and the error it makes on a state slightly off the data distribution takes it
further off, and the rollout diverges. Training against a short rollout puts the model's
own mistakes in its input distribution, and lengthening that rollout stage by stage is
what makes the horizon grow.

The window is also what truncates the gradient. Backpropagation runs through the whole
window and stops there, so the cost and the memory of a stage are bounded by its length
rather than by the horizon the model is eventually rolled out to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from nnphysics.core.errors import ValidationError

__all__ = ["Stage", "stage_length", "stages"]


@dataclass(frozen=True, slots=True)
class Stage:
    """One curriculum stage.

    Attributes:
        steps: Rollout length trained against.
        first_epoch: First epoch of the stage.
        last_epoch: Last epoch of the stage, inclusive.
    """

    steps: int
    first_epoch: int
    last_epoch: int

    @property
    def epochs(self) -> int:
        """Epochs the stage spans."""
        return self.last_epoch - self.first_epoch + 1


def stages(curriculum: Sequence[int], starts: Sequence[int], epochs: int) -> tuple[Stage, ...]:
    """Lay a curriculum out over a run.

    Args:
        curriculum: Rollout lengths, in order.
        starts: Epoch each length starts at, the same length as `curriculum`, beginning
            at zero and strictly increasing.
        epochs: Total epochs.

    Returns:
        The stages, in order.

    Raises:
        ValidationError: If the two sequences disagree in length, are empty, do not start
            at zero, do not increase, name a non positive length, or start a stage the
            run never reaches.
    """
    if not curriculum or len(curriculum) != len(starts):
        raise ValidationError(
            f"a curriculum needs one start epoch per length, got {list(curriculum)} and "
            f"{list(starts)}"
        )
    if any(steps < 1 for steps in curriculum):
        raise ValidationError(f"every curriculum length must be positive, got {list(curriculum)}")
    if starts[0] != 0:
        raise ValidationError(f"the first curriculum stage must start at epoch 0, got {starts[0]}")
    if any(later <= earlier for earlier, later in pairwise(starts)):
        raise ValidationError(f"curriculum start epochs must increase, got {list(starts)}")
    if starts[-1] >= epochs:
        raise ValidationError(
            f"a curriculum stage starts at epoch {starts[-1]}, which {epochs} epochs never reach"
        )
    ends = [*(start - 1 for start in starts[1:]), epochs - 1]
    return tuple(
        Stage(steps=steps, first_epoch=start, last_epoch=end)
        for steps, start, end in zip(curriculum, starts, ends, strict=True)
    )


def stage_length(epoch: int, curriculum: Sequence[int], starts: Sequence[int], epochs: int) -> int:
    """The rollout length one epoch trains against.

    Args:
        epoch: Epoch number, counted from zero.
        curriculum: Rollout lengths, in order.
        starts: Epoch each length starts at.
        epochs: Total epochs.

    Returns:
        The number of steps.

    Raises:
        ValidationError: If the curriculum is malformed or the epoch is outside the run.
    """
    laid_out = stages(curriculum, starts, epochs)
    for stage in laid_out:
        if stage.first_epoch <= epoch <= stage.last_epoch:
            return stage.steps
    raise ValidationError(f"epoch {epoch} is outside a run of {epochs} epochs")
