"""The learning rate schedule, as a pure function of the epoch.

A schedule that reads a mutable counter cannot be resumed, because the counter is state
nobody thought to checkpoint. This one is a function from epoch number to a multiple of
the peak rate, so the rate at epoch forty is the same whether training reached it in one
run or in three.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from nnphysics.core.errors import ValidationError

__all__ = ["cosine_with_warmup", "learning_rate_factor"]


def learning_rate_factor(
    epoch: int, *, epochs: int, warmup_epochs: int, final_fraction: float
) -> float:
    """The multiple of the peak learning rate to use at one epoch.

    Warmup is linear and starts at one step's worth rather than at zero, because a first
    epoch at a rate of exactly zero is an epoch spent computing gradients and discarding
    them. After it the rate follows half a cosine down to `final_fraction` of the peak.

    Args:
        epoch: Epoch number, counted from zero.
        epochs: Total epochs the schedule is defined over.
        warmup_epochs: Epochs the rate rises over. Zero starts at the peak.
        final_fraction: Share of the peak the cosine decays towards.

    Returns:
        The factor, between `final_fraction` and one.

    Raises:
        ValidationError: If the epoch is outside the schedule, the total is not positive,
            warmup does not leave an epoch to decay over, or the fraction is not in the
            unit interval.
    """
    if epochs < 1:
        raise ValidationError(f"a schedule needs at least one epoch, got {epochs}")
    if not 0 <= epoch < epochs:
        raise ValidationError(f"epoch {epoch} is outside a schedule of {epochs} epochs")
    if not 0 <= warmup_epochs < epochs:
        raise ValidationError(
            f"warmup of {warmup_epochs} epochs leaves nothing to decay over in {epochs}"
        )
    if not 0.0 <= final_fraction <= 1.0:
        raise ValidationError(
            f"the final learning rate fraction must be in [0, 1], got {final_fraction}"
        )
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    return final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))


def cosine_with_warmup(
    *, epochs: int, warmup_epochs: int, final_fraction: float
) -> Callable[[int], float]:
    """Bind a schedule's settings, giving the function `LambdaLR` wants.

    Args:
        epochs: Total epochs the schedule is defined over.
        warmup_epochs: Epochs the rate rises over.
        final_fraction: Share of the peak the cosine decays towards.

    Returns:
        A function from epoch number to a multiple of the peak rate.
    """

    def factor(epoch: int) -> float:
        return learning_rate_factor(
            min(epoch, epochs - 1),
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            final_fraction=final_fraction,
        )

    return factor
