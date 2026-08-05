"""Building blocks shared by every model, initialised from an explicit generator.

PyTorch initialises a layer from the global random state. Nothing in this package is
allowed to, so every layer built here is re-initialised from a generator the run seed
determines. The distribution is the one `torch.nn.Linear` uses by default, so this
changes where the numbers come from and not what they look like.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn

from nnphysics.core.errors import ValidationError

__all__ = ["init_linear", "make_mlp"]

_MINIMUM_SIZES = 2


def init_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """Draw one linear layer's weights and bias from a generator.

    Both are uniform on plus or minus the inverse square root of the fan in, which is
    what `nn.Linear` does by way of Kaiming uniform with its default gain.

    Args:
        layer: The layer to initialise, modified in place.
        generator: Generator every number is drawn from.
    """
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


def make_mlp(
    sizes: tuple[int, ...],
    generator: torch.Generator,
    *,
    final_activation: bool = False,
) -> nn.Sequential:
    """Build a multilayer perceptron with a smooth activation.

    The activation is SiLU rather than ReLU because these networks are asked for a
    derivative of a smooth field, and a piecewise linear one puts a kink in every
    prediction that a long rollout accumulates.

    Args:
        sizes: Layer widths, from the input dimension to the output dimension. At least
            two entries.
        generator: Generator every weight is drawn from.
        final_activation: Whether the output layer is followed by an activation. False
            for a head that must be able to produce any real number.

    Returns:
        The network.

    Raises:
        ValidationError: If fewer than two sizes were given, or a size is not positive.
    """
    if len(sizes) < _MINIMUM_SIZES:
        raise ValidationError(f"a multilayer perceptron needs an input and an output, got {sizes}")
    if any(size < 1 for size in sizes):
        raise ValidationError(f"every layer width must be positive, got {sizes}")
    layers: list[nn.Module] = []
    for position, (fan_in, fan_out) in enumerate(pairwise(sizes)):
        linear = nn.Linear(fan_in, fan_out)
        init_linear(linear, generator)
        layers.append(linear)
        if final_activation or position < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)
