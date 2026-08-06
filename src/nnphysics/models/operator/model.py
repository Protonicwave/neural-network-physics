"""The fluid surrogate: a Fourier neural operator on the vorticity field.

The design choice this phase turns on is what the model is a function of. A convolutional
network learns a stencil, which is a statement about grid points; this learns a multiplier
on a band of Fourier modes, which is a statement about wavenumbers. Only the second
survives a change of grid, and testing that it does is what the resolution metric is for.

Each block is a spectral path and a pointwise path added together. The spectral path is
the low band, filtered globally: one mode couples the whole domain, which is what an
elliptic problem needs and what a stencil of any fixed width cannot do in one layer.
The pointwise path is a one by one convolution, so it carries every wavenumber including
the ones the spectral path drops. Without it the model could not represent anything above
the cutoff at all; with it, the high band is passed through pointwise and only the low
band is learned as an operator.

The model predicts the update rather than the next state, in normalised units, and the
projection head starts at zero. An untrained operator is therefore exactly persistence,
so the first epochs are spent learning a step rather than unlearning a random one.

Unlike the N-body graph network, nothing about the physics is built in here. There is no
integrator to hand a derivative to: the vorticity equation is first order in time and its
right hand side is a nonlocal function of the whole field, so there is no cheap known half
to factor out the way velocity Verlet factors out the time stepping. The consequence is
visible in the results: this model conserves nothing by construction, and every invariant
number it produces is something it learned or failed to learn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from nnphysics.core.errors import ValidationError
from nnphysics.core.params import check_parameter_names, int_parameter
from nnphysics.models.base import Carry, ModelContext, ModelHyperparameters, SurrogateModel
from nnphysics.models.operator.fields import GridFields
from nnphysics.models.operator.spectral import SpectralConv2d

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["OPERATOR_NAME", "FourierNeuralOperator", "build_operator"]

OPERATOR_NAME = "operator"

_OPERATOR_PARAMETERS = ("modes", "width", "layers", "projection")
_DEFAULT_MODES = 8
"""Retained modes each way. The dealiasing cutoff of the default 64 point grid is 21, so
this keeps the band that carries the energy and leaves the tail to the pointwise path."""
_DEFAULT_WIDTH = 16
_DEFAULT_LAYERS = 4
_DEFAULT_PROJECTION = 64

_MINIMUM_LAYERS = 1


class FourierNeuralOperator(SurrogateModel):
    """Spectral convolution blocks on a lifted channel space, predicting the update.

    The model is tied to no grid size. Its weights are indexed by mode number, and the
    lifting, mixing and projection layers are one by one convolutions, so the same
    weights run on any grid large enough to hold the retained band.

    Args:
        context: What the model is being built for.
        modes: Modes retained along each axis in every spectral path.
        width: Channels the blocks work in.
        layers: Spectral blocks, at least one.
        projection: Width of the hidden layer of the projection head.

    Raises:
        ValidationError: If the data is not a set of two dimensional fields, or a setting
            is out of range.
    """

    def __init__(
        self,
        context: ModelContext,
        *,
        modes: int = _DEFAULT_MODES,
        width: int = _DEFAULT_WIDTH,
        layers: int = _DEFAULT_LAYERS,
        projection: int = _DEFAULT_PROJECTION,
    ) -> None:
        super().__init__(OPERATOR_NAME, context)
        if layers < _MINIMUM_LAYERS:
            raise ValidationError(f"a neural operator needs at least one block, got {layers}")
        if width < 1 or projection < 1:
            raise ValidationError(
                f"the block width and the projection width must be positive, got "
                f"{width} and {projection}"
            )
        self._modes = modes
        self._width = width
        self._layers = layers
        self._projection = projection
        self._grid = GridFields.build(context, model=OPERATOR_NAME)

        generator = context.generator()
        self.lift = _pointwise(self._grid.in_channels, width, generator)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes, generator) for _ in range(layers)
        )
        self.mixing = nn.ModuleList(_pointwise(width, width, generator) for _ in range(layers))
        self.project = _pointwise(width, projection, generator)
        self.head = _pointwise(projection, self._grid.out_channels, generator)
        # The head starts at zero, so an untrained operator is persistence.
        with torch.no_grad():
            self.head.weight.zero_()
            if self.head.bias is not None:
                self.head.bias.zero_()

    @property
    def hyperparameters(self) -> ModelHyperparameters:
        """The settings this model was built with."""
        return {
            "modes": self._modes,
            "width": self._width,
            "layers": self._layers,
            "projection": self._projection,
        }

    def advance(
        self, fields: Mapping[str, torch.Tensor], carry: Carry | None
    ) -> tuple[dict[str, torch.Tensor], Carry | None]:
        """Run the operator on the normalised state and add its output to it.

        Args:
            fields: The batch, in physical units.
            carry: Unused: every block reads the whole state, so nothing survives a step.

        Returns:
            The predicted fields and no carry.

        Raises:
            ValidationError: If the fields do not share one grid, or the grid is smaller
                than the retained band.
        """
        del carry
        normalised = self.normalise(fields)
        hidden = self.lift(self._grid.pack(normalised))
        for position, (spectral, mixing) in enumerate(zip(self.spectral, self.mixing, strict=True)):
            combined = spectral(hidden) + mixing(hidden)
            # No activation after the last block: the projection head is what turns the
            # channels into an update, and a nonlinearity in front of it only narrows
            # the range the head can start from.
            hidden = torch.nn.functional.silu(combined) if position < self._layers - 1 else combined
        update = self.head(torch.nn.functional.silu(self.project(hidden)))
        return self.denormalise(self._grid.unpack(update, normalised)), None


def _pointwise(in_channels: int, out_channels: int, generator: torch.Generator) -> nn.Conv2d:
    """A one by one convolution, which is a linear map across channels at every point.

    Initialised from an explicit generator with the distribution `nn.Linear` uses, which
    is what it is: a linear layer applied to every grid point independently.
    """
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    bound = 1.0 / (in_channels**0.5)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)
    return layer


def build_operator(context: ModelContext, hyperparameters: ModelHyperparameters) -> SurrogateModel:
    """Build the Fourier neural operator.

    Args:
        context: What the model is being built for.
        hyperparameters: Accepts `modes`, `width`, `layers` and `projection`.

    Returns:
        The model.

    Raises:
        ValidationError: If a hyperparameter is unknown or out of range, or the data is
            not a set of two dimensional fields.
    """
    check_parameter_names(hyperparameters, _OPERATOR_PARAMETERS, context=OPERATOR_NAME)
    return FourierNeuralOperator(
        context,
        modes=int_parameter(hyperparameters, "modes", _DEFAULT_MODES, context=OPERATOR_NAME),
        width=int_parameter(hyperparameters, "width", _DEFAULT_WIDTH, context=OPERATOR_NAME),
        layers=int_parameter(hyperparameters, "layers", _DEFAULT_LAYERS, context=OPERATOR_NAME),
        projection=int_parameter(
            hyperparameters, "projection", _DEFAULT_PROJECTION, context=OPERATOR_NAME
        ),
    )
