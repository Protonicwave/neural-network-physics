from __future__ import annotations

import math

import pytest
import torch

from nnphysics.core.errors import ValidationError
from nnphysics.models.graph import InteractionNetwork

BODIES = 6
HIDDEN = 8
ROUNDS = 2
SOFTENING = 0.05


def build(seed: int = 0) -> InteractionNetwork:
    """A network whose head has been given weight, so it predicts something."""
    network = InteractionNetwork(
        HIDDEN, ROUNDS, torch.Generator().manual_seed(seed), softening=SOFTENING
    )
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in network.head.parameters():
            parameter.uniform_(-0.5, 0.5, generator=generator)
    return network


def sample(seed: int = 2, bodies: int = BODIES) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions and masses for one batch of two configurations."""
    generator = torch.Generator().manual_seed(seed)
    position = torch.rand((2, bodies, 2), generator=generator) * 2.0 - 1.0
    mass = torch.rand((2, bodies), generator=generator) + 0.5
    return position, mass


def rotation(angle: float) -> torch.Tensor:
    return torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


@pytest.mark.parametrize(
    ("hidden", "rounds", "softening"),
    [(0, ROUNDS, SOFTENING), (HIDDEN, 0, SOFTENING), (HIDDEN, ROUNDS, 0.0)],
)
def test_a_network_needs_a_width_a_round_and_a_softening(
    hidden: int, rounds: int, softening: float
) -> None:
    with pytest.raises(ValidationError):
        InteractionNetwork(hidden, rounds, torch.Generator(), softening=softening)


def test_an_untrained_network_predicts_no_acceleration_at_all() -> None:
    """The head starts at zero, so the first velocity Verlet step is a straight line."""
    network = InteractionNetwork(
        HIDDEN, ROUNDS, torch.Generator().manual_seed(0), softening=SOFTENING
    )
    position, mass = sample()
    assert torch.allclose(network(position, mass), torch.zeros_like(position))


def test_the_output_has_one_acceleration_per_body() -> None:
    position, mass = sample()
    assert build()(position, mass).shape == position.shape


def test_translating_the_system_leaves_every_acceleration_alone() -> None:
    network = build()
    position, mass = sample()
    shift = torch.tensor([3.0, -1.25])
    assert torch.allclose(network(position + shift, mass), network(position, mass), atol=1e-5)


def test_rotating_the_system_rotates_every_acceleration() -> None:
    network = build()
    position, mass = sample()
    turn = rotation(0.7)
    assert torch.allclose(
        network(position @ turn.T, mass), network(position, mass) @ turn.T, atol=1e-5
    )


def test_reordering_the_bodies_reorders_the_accelerations() -> None:
    network = build()
    position, mass = sample()
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    assert torch.allclose(
        network(position[:, order], mass[:, order]),
        network(position, mass)[:, order],
        atol=1e-5,
    )


def test_a_lone_body_pulls_on_nothing_including_itself() -> None:
    """The softening would otherwise make the diagonal of the pair tensor finite."""
    position, mass = sample(bodies=1)
    assert torch.allclose(build()(position, mass), torch.zeros_like(position))


def test_the_same_seed_gives_the_same_weights() -> None:
    first = {name: value.clone() for name, value in build(7).state_dict().items()}
    second = build(7).state_dict()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_a_different_seed_gives_different_weights() -> None:
    first = build(7).state_dict()
    second = build(8).state_dict()
    assert any(not torch.equal(first[name], second[name]) for name in first)
