from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from nnphysics.core.protocols import System
from nnphysics.core.types import Regime, State
from nnphysics.data.normalisation import FieldStats, Normalisation
from nnphysics.models import ModelContext
from nnphysics.systems import build_system
from nnphysics.systems.nbody.state import MASS_FIELD, POSITION_FIELD, VELOCITY_FIELD

BODIES = 8
"""Enough bodies for every pair term to matter, few enough to keep a test fast."""

DT = 0.01
SEED = 0

# Statistics of the shape a real fit produces, written down rather than fitted so that a
# model test never depends on generating a dataset.
STATISTICS = Normalisation(
    {
        POSITION_FIELD: FieldStats(mean=0.0, std=0.5, count=1000),
        VELOCITY_FIELD: FieldStats(mean=0.0, std=0.8, count=1000),
        MASS_FIELD: FieldStats(mean=1.0 / BODIES, std=0.01, count=1000),
    }
)


@pytest.fixture
def context() -> ModelContext:
    """A model context describing a small N-body dataset."""
    return ModelContext(
        field_shapes={
            POSITION_FIELD: (BODIES, 2),
            VELOCITY_FIELD: (BODIES, 2),
            MASS_FIELD: (BODIES,),
        },
        static_fields=(MASS_FIELD,),
        normalisation=STATISTICS,
        dt=DT,
        seed=SEED,
    )


def make_system(**parameters: Any) -> System:
    """The N-body system, with the softening a small test needs."""
    return build_system("nbody", {"softening": 0.05, **parameters})


@pytest.fixture
def state() -> State:
    """One virialised cluster of the size the context describes."""
    system = make_system()
    regime = Regime(
        name="virialised_cluster",
        parameters={
            "n_bodies": float(BODIES),
            "radius": 1.0,
            "mass_spread": 4.0,
            "virial_ratio": 1.0,
        },
    )
    return system.initial_state(regime, np.random.default_rng(SEED))


@pytest.fixture
def batch(state: State, as_batch: Batcher) -> dict[str, torch.Tensor]:
    """That state as a batch of one, in single precision."""
    return as_batch(state)


Batcher = Callable[..., dict[str, torch.Tensor]]
Exciter = Callable[..., None]


@pytest.fixture
def as_batch() -> Batcher:
    """Stack states into the batched tensors a model's `advance` takes."""

    def stack(*states: State) -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(
                np.stack([np.asarray(one.fields[name], dtype=np.float32) for one in states])
            )
            for name in states[0].names
        }

    return stack


@pytest.fixture
def excite() -> Exciter:
    """Give a zero initialised head some weight, so a test sees a model that predicts.

    Several models start at exactly the identity on purpose. That is the right place to
    begin training from and the wrong place to test equivariance from, because a model
    that predicts nothing is equivariant for a reason that has nothing to do with its
    design.
    """

    def fill(module: torch.nn.Module, seed: int = 1) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.uniform_(-0.3, 0.3, generator=generator)

    return fill
