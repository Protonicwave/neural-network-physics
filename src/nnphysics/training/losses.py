"""What the loop minimises, and the one number early stopping reads.

Every error here is measured in normalised units. A loss in physical units would weight a
field by whatever units it happened to be written in, so a system whose velocities are a
thousand times its positions would be trained almost entirely on its velocities. The
statistics doing the normalising are the training split's, so the weighting is a property
of the data rather than of the system.

The loss has three parts and only the first is ever on by default. The one step term is
the model's accuracy at what it was asked to do. The multi step term is its accuracy at
what it will actually be asked to do, over the curriculum window. The physics penalty is
the model's own, and is weighted to zero unless a configuration says otherwise.

**The loss is robust and the metric is not, deliberately.** A rollout that has begun to
diverge produces residuals of tens of standard deviations, and a squared error on those
is a gradient thousands of times larger than anything the rest of the batch contributes.
Training on it does not teach the model to stop diverging, it destroys the accuracy the
model already had: measured here, one curriculum stage lengthening from one step to four
raised the one step error by a factor of three hundred within a single epoch, and the
validation rollout never recovered. So the loss counts a large residual linearly rather
than quadratically, past one standard deviation, which bounds its gradient.

The validation metric stays a plain mean squared error. It is a measurement rather than
an objective, nothing is optimised against it, and a number that quietly stopped growing
once a rollout went badly wrong would be the wrong number to select a model on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nnphysics.core.errors import ValidationError

if TYPE_CHECKING:
    from nnphysics.models.base import SurrogateModel

__all__ = ["HUBER_DELTA", "LossTerms", "rollout_error", "rollout_loss", "rollout_residual"]

HUBER_DELTA = 1.0
"""Where the loss stops being quadratic, in normalised units.

One standard deviation of the field. Past it a prediction is already wrong by as much as
the data itself varies, and how much further wrong it is says little about what the model
should do differently.
"""


@dataclass(frozen=True, slots=True)
class LossTerms:
    """The loss and the parts it was made of, for logging.

    Attributes:
        total: What is differentiated.
        one_step: Robust normalised error at the first predicted step.
        multi_step: The same, meaned over every later step of the window. Zero for a
            window of one step, which has no later steps rather than perfect ones.
        physics: The model's own penalty, before weighting.
    """

    total: torch.Tensor
    one_step: float
    multi_step: float
    physics: float


def rollout_error(
    model: SurrogateModel,
    inputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    steps: int,
) -> torch.Tensor:
    """Normalised mean squared error of an unroll, one number per step.

    This is the measurement, and it is what early stopping reads. It grows without bound
    on a rollout that has diverged, which is the honest thing for a metric to do.

    Args:
        model: The model, which owns the normalisation.
        inputs: The initial states of the batch, in physical units.
        targets: Ground truth for every step, shape `(batch, steps, *field_shape)`.
        steps: Steps to unroll, at most the number of steps the targets hold.

    Returns:
        A tensor of shape `(steps,)`, the error at each step meaned over the batch, over
        the elements of a field and over the predicted fields.

    Raises:
        ValidationError: If the targets are shorter than the unroll asked for.
    """
    return _per_step(model, inputs, targets, steps, delta=None)


def rollout_residual(
    model: SurrogateModel,
    inputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    steps: int,
    *,
    delta: float = HUBER_DELTA,
) -> torch.Tensor:
    """The same, counted linearly past `delta` so a divergence cannot swamp a batch.

    This is the objective. It agrees with `rollout_error` up to a constant factor while
    every residual is small, which is the regime a model that is working stays in.

    Args:
        model: The model, which owns the normalisation.
        inputs: The initial states of the batch, in physical units.
        targets: Ground truth for every step.
        steps: Steps to unroll.
        delta: Where the loss stops being quadratic, in normalised units.

    Returns:
        A tensor of shape `(steps,)`.

    Raises:
        ValidationError: If the targets are shorter than the unroll asked for, or the
            threshold is not positive.
    """
    if delta <= 0.0:
        raise ValidationError(f"the robust threshold must be positive, got {delta}")
    return _per_step(model, inputs, targets, steps, delta=delta)


def rollout_loss(  # noqa: PLR0913
    # The model, the batch it is scored on, the window and the two weights. The weights
    # come from the training configuration and the batch does not, so the object that
    # would group them does not exist.
    model: SurrogateModel,
    inputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    steps: int,
    *,
    multi_step_weight: float,
    physics_weight: float,
) -> LossTerms:
    """Score one batch and assemble the loss.

    The physics penalty is evaluated on the first predicted step alone. Over a window it
    would be a penalty on a state the model reached through its own errors, which is a
    different and much weaker statement than a penalty on the step itself.

    Args:
        model: The model.
        inputs: The initial states of the batch, in physical units.
        targets: Ground truth for every step of the window.
        steps: Length of the window.
        multi_step_weight: Weight on the mean error over the rest of the window.
        physics_weight: Weight on the model's own penalty.

    Returns:
        The loss and its parts.

    Raises:
        ValidationError: If a weight is negative or the window is empty.
    """
    if steps < 1:
        raise ValidationError(f"a loss window must hold at least one step, got {steps}")
    if multi_step_weight < 0.0 or physics_weight < 0.0:
        raise ValidationError(
            f"loss weights must not be negative, got {multi_step_weight} and {physics_weight}"
        )
    per_step = rollout_residual(model, inputs, targets, steps)
    one_step = per_step[0]
    multi_step = per_step[1:].mean() if steps > 1 else torch.zeros((), dtype=one_step.dtype)
    total = one_step + multi_step_weight * multi_step

    physics = torch.zeros((), dtype=one_step.dtype)
    if physics_weight > 0.0:
        first, _ = model.advance(inputs, None)
        physics = model.physics_penalty(inputs, first)
        total = total + physics_weight * physics
    return LossTerms(
        total=total,
        one_step=float(one_step.detach()),
        multi_step=float(multi_step.detach()),
        physics=float(physics.detach()),
    )


def _per_step(
    model: SurrogateModel,
    inputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    steps: int,
    *,
    delta: float | None,
) -> torch.Tensor:
    """One error per step, squared when `delta` is `None` and robust otherwise."""
    predicted = model.unroll(inputs, steps)
    normalised_prediction = model.normalise(predicted)
    normalised_target = model.normalise({name: targets[name] for name in model.predicted_fields})
    per_field: list[torch.Tensor] = []
    for name in model.predicted_fields:
        truth = normalised_target[name]
        if truth.shape[1] < steps:
            raise ValidationError(
                f"an unroll of {steps} steps needs {steps} target states for field "
                f"{name!r}, the batch holds {truth.shape[1]}"
            )
        residual = normalised_prediction[name] - truth[:, :steps]
        if delta is None:
            elementwise = residual.pow(2)
        else:
            # Huber, scaled so that it is the squared error while the residual is small
            # rather than half of it, which keeps the two comparable in a log.
            absolute = residual.abs()
            elementwise = torch.where(
                absolute <= delta,
                residual.pow(2),
                delta * (2.0 * absolute - delta),
            )
        per_field.append(elementwise.reshape(elementwise.shape[0], steps, -1).mean(dim=(0, 2)))
    return torch.stack(per_field).mean(dim=0)
