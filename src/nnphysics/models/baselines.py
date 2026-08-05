"""Two learned models that are weak on purpose.

They exist to set a floor and to exercise the training loop cheaply. A graph network that
beats neither of them has not been shown to work, and a training loop that cannot fit
them has a bug that no amount of capacity would reveal.

Both work in normalised units and predict an update rather than a next state. Predicting
the update is the weaker half of the choice the graph network makes: it means the model
starts from the identity, so an untrained one is the persistence baseline rather than
noise, and the numbers it has to produce are the small ones rather than the large ones.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from nnphysics.core.errors import ValidationError
from nnphysics.core.params import check_parameter_names, int_parameter
from nnphysics.models.base import Carry, ModelContext, ModelHyperparameters, SurrogateModel
from nnphysics.models.layers import make_mlp

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CONSTANT_NAME",
    "MLP_NAME",
    "ConstantModel",
    "MultilayerPerceptron",
    "build_constant",
    "build_mlp",
]

_MINIMUM_LAYERS = 2

CONSTANT_NAME = "constant"
MLP_NAME = "mlp"

_MLP_PARAMETERS = ("hidden", "layers")
_DEFAULT_HIDDEN = 256
_DEFAULT_LAYERS = 3


class ConstantModel(SurrogateModel):
    """The zero update model, with one learned number per predicted field.

    Every state is returned unchanged apart from a constant offset, the same for every
    element of a field, learned in normalised units and initialised at zero. Untrained it
    is exactly persistence. Trained it is the best constant drift, which is the least a
    model can learn and still have learned something.

    Args:
        context: What the model is being built for.
    """

    def __init__(self, context: ModelContext) -> None:
        super().__init__(CONSTANT_NAME, context)
        self.offset = nn.Parameter(torch.zeros(len(self.predicted_fields)))

    def advance(
        self, fields: Mapping[str, torch.Tensor], carry: Carry | None
    ) -> tuple[dict[str, torch.Tensor], Carry | None]:
        """Add the learned offset to every predicted field.

        Args:
            fields: The batch, in physical units.
            carry: Unused: nothing is worth reusing between steps.

        Returns:
            The predicted fields and no carry.
        """
        del carry
        normalised = self.normalise({name: fields[name] for name in self.predicted_fields})
        updated = {
            name: normalised[name] + self.offset[index]
            for index, name in enumerate(self.predicted_fields)
        }
        return self.denormalise(updated), None


class MultilayerPerceptron(SurrogateModel):
    """A small network on the flattened state, predicting the normalised update.

    Flattening is the weakness. The input layer has one weight per element of a state, so
    the model is tied to the exact shape it was trained on: a different number of bodies
    is not a harder case for it, it is a case it cannot represent at all. It fails loudly
    when handed one, which is the honest behaviour and is what the held out regime is for.

    Args:
        context: What the model is being built for.
        hidden: Width of each hidden layer.
        layers: Linear layers, at least two.

    Raises:
        ValidationError: If fewer than two layers were asked for.
    """

    def __init__(
        self,
        context: ModelContext,
        *,
        hidden: int = _DEFAULT_HIDDEN,
        layers: int = _DEFAULT_LAYERS,
    ) -> None:
        super().__init__(MLP_NAME, context)
        if layers < _MINIMUM_LAYERS:
            raise ValidationError(
                f"a multilayer perceptron needs at least two layers, got {layers}"
            )
        self._hidden = hidden
        self._layers = layers
        self._input_shapes = {name: tuple(shape) for name, shape in context.field_shapes.items()}
        self._order = tuple(sorted(self._input_shapes))
        fan_in = sum(_extent(self._input_shapes[name]) for name in self._order)
        fan_out = sum(_extent(self._input_shapes[name]) for name in self.predicted_fields)
        widths = (fan_in, *([hidden] * (layers - 1)), fan_out)
        self.network = make_mlp(widths, context.generator())
        # The head starts at zero, so an untrained model is persistence rather than noise
        # and the first epochs are spent learning the update instead of unlearning a
        # random one.
        head = self.network[-1]
        if isinstance(head, nn.Linear):
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()

    @property
    def hyperparameters(self) -> ModelHyperparameters:
        """The settings this model was built with."""
        return {"hidden": self._hidden, "layers": self._layers}

    def advance(
        self, fields: Mapping[str, torch.Tensor], carry: Carry | None
    ) -> tuple[dict[str, torch.Tensor], Carry | None]:
        """Run the network on the flattened normalised state.

        Args:
            fields: The batch, in physical units.
            carry: Unused: nothing is worth reusing between steps.

        Returns:
            The predicted fields and no carry.

        Raises:
            ValidationError: If a field does not have the shape the network was built for.
        """
        del carry
        self._check_shapes(fields)
        normalised = self.normalise(fields)
        batch = next(iter(normalised.values())).shape[0]
        flattened = torch.cat([normalised[name].reshape(batch, -1) for name in self._order], dim=1)
        update = self.network(flattened)
        predicted: dict[str, torch.Tensor] = {}
        offset = 0
        for name in self.predicted_fields:
            shape = self._input_shapes[name]
            width = _extent(shape)
            predicted[name] = normalised[name] + update[:, offset : offset + width].reshape(
                batch, *shape
            )
            offset += width
        return self.denormalise(predicted), None

    def _check_shapes(self, fields: Mapping[str, torch.Tensor]) -> None:
        """Reject a state the flattened input layer cannot represent."""
        for name in self._order:
            actual = tuple(fields[name].shape[1:])
            if actual != self._input_shapes[name]:
                raise ValidationError(
                    f"the multilayer perceptron was built for field {name!r} of shape "
                    f"{self._input_shapes[name]} and cannot take {actual}: a flattened "
                    f"state model does not transfer across a change of size"
                )


def _extent(shape: tuple[int, ...]) -> int:
    """Elements one state of a field holds."""
    return math.prod(shape) if shape else 1


def build_constant(context: ModelContext, hyperparameters: ModelHyperparameters) -> SurrogateModel:
    """Build the zero update model.

    Args:
        context: What the model is being built for.
        hyperparameters: None accepted.

    Returns:
        The model.

    Raises:
        ValidationError: If a hyperparameter is given.
    """
    check_parameter_names(hyperparameters, (), context=CONSTANT_NAME)
    return ConstantModel(context)


def build_mlp(context: ModelContext, hyperparameters: ModelHyperparameters) -> SurrogateModel:
    """Build the flattened state multilayer perceptron.

    Args:
        context: What the model is being built for.
        hyperparameters: Accepts `hidden` and `layers`.

    Returns:
        The model.

    Raises:
        ValidationError: If a hyperparameter is unknown or out of range.
    """
    check_parameter_names(hyperparameters, _MLP_PARAMETERS, context=MLP_NAME)
    return MultilayerPerceptron(
        context,
        hidden=int_parameter(hyperparameters, "hidden", _DEFAULT_HIDDEN, context=MLP_NAME),
        layers=int_parameter(hyperparameters, "layers", _DEFAULT_LAYERS, context=MLP_NAME),
    )
