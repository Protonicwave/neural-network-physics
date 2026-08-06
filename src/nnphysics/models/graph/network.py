"""Message passing over a fully connected particle graph, producing an acceleration.

The graph is dense because the interaction is: gravity has no cutoff, so every pair is an
edge and a neighbourhood would be an approximation nobody asked for. At the sizes this
repository trains on, thirty two bodies, a dense pair tensor is also the fastest thing to
compute, and it keeps the whole forward pass free of any loop over particles.

Two properties are built in rather than learned, and both are worth stating plainly
because they are the reason the equivariance metric reads near zero for this model.

**Only relative positions enter.** Every scalar the network sees is a pair distance or a
mass, so translating the whole system cannot change any feature.

**The output is a sum of scalars times unit vectors.** The network chooses how strongly
body `i` is pulled along the direction of body `j`, and never chooses a direction of its
own. Rotating the input therefore rotates the output exactly. This is the form Newtonian
gravity already takes, so it costs nothing to impose and it removes a whole class of
error the model would otherwise have to spend capacity suppressing.

Velocity is deliberately absent. The acceleration of a gravitating body depends on where
the other bodies are and not on how fast anything is moving, so leaving velocity out is
a physical statement, and it makes invariance under a Galilean boost exact as well.
"""

from __future__ import annotations

import torch
from torch import nn

from nnphysics.core.errors import ValidationError
from nnphysics.models.layers import make_mlp

__all__ = ["EDGE_FEATURES", "InteractionNetwork"]

EDGE_FEATURES = 3
"""Distance, softened inverse distance and its square, all in units of the length scale.

The last two are the shapes a gravitational force law is built from, so the network is
given them rather than being made to synthesise them out of the first. Both are scaled by
the softening so that neither exceeds one however close a pair comes, which keeps the
input to the first layer bounded for the tightest binary in the held out regime.
"""


class InteractionNetwork(nn.Module):
    """Rounds of message passing that end in one acceleration per body.

    Args:
        hidden: Width of every node embedding, message and hidden layer.
        rounds: Message passing rounds. Each one lets information travel one more edge,
            which for a dense graph means one more order of many body effect.
        generator: Generator every weight is drawn from.
        softening: Softening length in units of the position scale, which bounds the
            inverse distance features and stops a close pair producing a large one.

    Raises:
        ValidationError: If the width, the round count or the softening is not positive.
    """

    def __init__(
        self,
        hidden: int,
        rounds: int,
        generator: torch.Generator,
        *,
        softening: float,
    ) -> None:
        super().__init__()
        if hidden < 1:
            raise ValidationError(f"the hidden width must be positive, got {hidden}")
        if rounds < 1:
            raise ValidationError(f"message passing needs at least one round, got {rounds}")
        if softening <= 0.0:
            raise ValidationError(f"the softening must be positive, got {softening}")
        self._hidden = hidden
        self._softening = softening
        self.embed = make_mlp((1, hidden, hidden), generator)
        self.edges = nn.ModuleList(
            make_mlp((2 * hidden + EDGE_FEATURES, hidden, hidden), generator) for _ in range(rounds)
        )
        self.nodes = nn.ModuleList(
            make_mlp((2 * hidden, hidden, hidden), generator) for _ in range(rounds)
        )
        self.head = make_mlp((2 * hidden + EDGE_FEATURES, hidden, 1), generator)
        # The head starts at zero, so an untrained model predicts no acceleration at all
        # and its velocity Verlet step is a straight line. That is a defensible state to
        # begin from, and it means the first epochs learn the force rather than unlearn
        # a random one.
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.weight.zero_()
                final.bias.zero_()

    def forward(self, position: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """Predict the acceleration of every body.

        Args:
            position: Positions in units of the length scale, shape `(batch, bodies, 2)`.
            mass: Masses in units of the mass scale, shape `(batch, bodies)`.

        Returns:
            Accelerations in units of the acceleration scale, shape `(batch, bodies, 2)`.
        """
        separation = position.unsqueeze(1) - position.unsqueeze(2)
        squared = separation.pow(2).sum(-1) + self._softening**2
        distance = squared.sqrt()
        scaled_inverse = self._softening / distance
        geometry = torch.stack([distance, scaled_inverse, scaled_inverse.pow(2)], dim=-1)
        # A body does not pull on itself, and the diagonal of the pair tensor is the one
        # place the softening would otherwise let it.
        mask = (
            (1.0 - torch.eye(position.shape[1], dtype=position.dtype, device=position.device))
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        embedded = self.embed(mass.unsqueeze(-1))
        for edge, node in zip(self.edges, self.nodes, strict=True):
            message = edge(_pair_features(embedded, geometry)) * mask
            embedded = embedded + node(torch.cat([embedded, message.sum(dim=2)], dim=-1))

        weight = self.head(_pair_features(embedded, geometry)) * mask
        direction = separation / distance.unsqueeze(-1)
        acceleration: torch.Tensor = (weight * direction).sum(dim=2)
        return acceleration


def _pair_features(embedded: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    """Lay every ordered pair of node embeddings alongside that pair's geometry."""
    batch, bodies, hidden = embedded.shape
    source = embedded.unsqueeze(2).expand(batch, bodies, bodies, hidden)
    target = embedded.unsqueeze(1).expand(batch, bodies, bodies, hidden)
    return torch.cat([source, target, geometry], dim=-1)
