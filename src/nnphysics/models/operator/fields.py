"""Reading a state as an image, and writing an image back as a state.

Both models in this package work on fields sampled on a two dimensional grid, and both
want the same thing: every field stacked as a channel, normalised, with the network's
output read back as an update to what went in. That is all this is.

It names no field. What makes a model applicable here is the shape of the data, not the
system that produced it: any dataset whose states are two dimensional arrays can be
packed this way, and one whose states are not is rejected at construction with a message
saying so. The extent of the grid is deliberately not checked, because a model that fixes
it cannot answer the question this phase exists to ask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nnphysics.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nnphysics.models.base import ModelContext

__all__ = ["GridFields"]

_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class GridFields:
    """The channel layout one grid model reads its data through.

    Attributes:
        names: Every field, sorted, which is the order the input channels are stacked in.
        predicted: Fields the network produces, sorted, which is the order the output
            channels are read in.
    """

    names: tuple[str, ...]
    predicted: tuple[str, ...]

    @classmethod
    def build(cls, context: ModelContext, *, model: str) -> GridFields:
        """Read the layout out of a model context.

        Args:
            context: What the model is being built for.
            model: Model name, used in the error message.

        Returns:
            The layout.

        Raises:
            ValidationError: If any field of the data is not a two dimensional grid.
        """
        offending = sorted(
            f"{name}{tuple(shape)}"
            for name, shape in context.field_shapes.items()
            if len(shape) != _DIMENSIONS
        )
        if offending:
            raise ValidationError(
                f"the {model} model works on fields sampled on a two dimensional grid, "
                f"but the data carries {offending}"
            )
        return cls(names=context.names, predicted=context.predicted_fields)

    @property
    def in_channels(self) -> int:
        """Channels the network reads."""
        return len(self.names)

    @property
    def out_channels(self) -> int:
        """Channels the network writes."""
        return len(self.predicted)

    def pack(self, normalised: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Stack normalised fields into one image.

        Args:
            normalised: Field name to a tensor of shape `(batch, height, width)`.

        Returns:
            One tensor of shape `(batch, channels, height, width)`.

        Raises:
            ValidationError: If the fields do not share one grid.
        """
        shapes = {tuple(normalised[name].shape) for name in self.names}
        if len(shapes) != 1:
            raise ValidationError(
                f"every field must be sampled on the same grid, got {sorted(shapes)}"
            )
        return torch.stack([normalised[name] for name in self.names], dim=1)

    def unpack(
        self, update: torch.Tensor, normalised: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Add the network's output to what it read, channel by channel.

        Predicting the update rather than the next state means an untrained model is
        persistence rather than noise, and that the numbers the network has to produce
        are the small ones.

        Args:
            update: The network's output, shape `(batch, channels, height, width)`.
            normalised: The normalised inputs the update applies to.

        Returns:
            The predicted fields, still normalised.
        """
        return {
            name: normalised[name] + update[:, index] for index, name in enumerate(self.predicted)
        }
