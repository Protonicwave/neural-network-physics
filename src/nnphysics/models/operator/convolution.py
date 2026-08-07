"""The control the operator is measured against: a plain periodic convolutional network.

This lives beside the operator rather than with the other baselines because it exists
only for the comparison, and it shares the operator's channel packing so that the two
differ in one thing. Without it there is no way to tell whether the spectral structure is
doing the work or the parameter count is, and a result that cannot separate those two is
not a result.

The padding is circular, which is what the domain is. A zero padded network would spend
capacity learning that the domain has no edges, and would be wrong at the boundary in a
way that has nothing to do with the comparison being made.

The default width is chosen to put the parameter count near the operator's default rather
than to be a round number. The two counts are reported in the README, because a claim of
equal capacity that nobody checked is not a control.

One property this deliberately does not have: a stencil is a statement about grid points,
so the receptive field of this network shrinks, in physical distance, when the grid is
refined. It runs at any resolution and answers a different question at each of them,
which is exactly the contrast the resolution metric is there to measure.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import torch
from torch import nn

from nnphysics.core.errors import ValidationError
from nnphysics.core.params import check_parameter_names, int_parameter
from nnphysics.models.base import Carry, ModelContext, ModelHyperparameters, SurrogateModel
from nnphysics.models.operator.fields import GridFields

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["CONVOLUTION_NAME", "ConvolutionalBaseline", "build_convolution"]

CONVOLUTION_NAME = "convolution"

_CONVOLUTION_PARAMETERS = ("width", "layers", "kernel_size")
_DEFAULT_WIDTH = 99
"""Chosen so the parameter count lands on the operator's default, not for its own sake:
266,014 against 264,417, which is six parts in a thousand."""
_DEFAULT_LAYERS = 4
_DEFAULT_KERNEL = 3

_MINIMUM_LAYERS = 1


class ConvolutionalBaseline(SurrogateModel):
    """A stack of circularly padded convolutions predicting the normalised update.

    Args:
        context: What the model is being built for.
        width: Channels the hidden layers work in.
        layers: Hidden convolutions, at least one.
        kernel_size: Stencil width, which must be odd so the stencil is centred.

    Raises:
        ValidationError: If the data is not a set of two dimensional fields, or a setting
            is out of range.
    """

    def __init__(
        self,
        context: ModelContext,
        *,
        width: int = _DEFAULT_WIDTH,
        layers: int = _DEFAULT_LAYERS,
        kernel_size: int = _DEFAULT_KERNEL,
    ) -> None:
        super().__init__(CONVOLUTION_NAME, context)
        if layers < _MINIMUM_LAYERS:
            raise ValidationError(f"a convolutional network needs a layer, got {layers}")
        if width < 1:
            raise ValidationError(f"the layer width must be positive, got {width}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValidationError(
                f"a centred stencil needs an odd width of at least one, got {kernel_size}"
            )
        self._width = width
        self._layers = layers
        self._kernel_size = kernel_size
        self._grid = GridFields.build(context, model=CONVOLUTION_NAME)

        generator = context.generator()
        widths = (self._grid.in_channels, *([width] * layers))
        self.blocks = nn.ModuleList(
            _periodic(fan_in, fan_out, kernel_size, generator)
            for fan_in, fan_out in pairwise(widths)
        )
        self.head = _periodic(width, self._grid.out_channels, 1, generator)
        # As with every model here, an untrained one is persistence rather than noise.
        with torch.no_grad():
            self.head.weight.zero_()
            if self.head.bias is not None:
                self.head.bias.zero_()

    @property
    def hyperparameters(self) -> ModelHyperparameters:
        """The settings this model was built with."""
        return {
            "width": self._width,
            "layers": self._layers,
            "kernel_size": self._kernel_size,
        }

    def advance(
        self, fields: Mapping[str, torch.Tensor], carry: Carry | None
    ) -> tuple[dict[str, torch.Tensor], Carry | None]:
        """Run the stack on the normalised state and add its output to it.

        Args:
            fields: The batch, in physical units.
            carry: Unused: every layer reads the whole state, so nothing survives a step.

        Returns:
            The predicted fields and no carry.

        Raises:
            ValidationError: If the fields do not share one grid.
        """
        del carry
        normalised = self.normalise(fields)
        hidden = self._grid.pack(normalised)
        for block in self.blocks:
            hidden = torch.nn.functional.silu(block(hidden))
        return self.denormalise(self._grid.unpack(self.head(hidden), normalised)), None


def _periodic(
    in_channels: int, out_channels: int, kernel_size: int, generator: torch.Generator
) -> nn.Conv2d:
    """One convolution that wraps at the edges, initialised from an explicit generator."""
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="circular",
    )
    bound = 1.0 / ((in_channels * kernel_size * kernel_size) ** 0.5)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)
    return layer


def build_convolution(
    context: ModelContext, hyperparameters: ModelHyperparameters
) -> SurrogateModel:
    """Build the convolutional baseline.

    Args:
        context: What the model is being built for.
        hyperparameters: Accepts `width`, `layers` and `kernel_size`.

    Returns:
        The model.

    Raises:
        ValidationError: If a hyperparameter is unknown or out of range, or the data is
            not a set of two dimensional fields.
    """
    check_parameter_names(hyperparameters, _CONVOLUTION_PARAMETERS, context=CONVOLUTION_NAME)
    return ConvolutionalBaseline(
        context,
        width=int_parameter(hyperparameters, "width", _DEFAULT_WIDTH, context=CONVOLUTION_NAME),
        layers=int_parameter(hyperparameters, "layers", _DEFAULT_LAYERS, context=CONVOLUTION_NAME),
        kernel_size=int_parameter(
            hyperparameters, "kernel_size", _DEFAULT_KERNEL, context=CONVOLUTION_NAME
        ),
    )
