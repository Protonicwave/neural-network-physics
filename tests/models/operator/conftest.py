from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from nnphysics.core.types import State
from nnphysics.data.normalisation import FieldStats, Normalisation
from nnphysics.models import ModelContext
from nnphysics.systems.fluid.grid import VORTICITY_FIELD, FluidGrid

SIZE = 32
"""Small enough to keep a test fast, large enough to hold the default retained band twice
over so that a refined case has somewhere to go."""

DT = 0.05
SEED = 0

STATISTICS = Normalisation({VORTICITY_FIELD: FieldStats(mean=0.0, std=2.0, count=1000)})
"""The shape a real fit produces on this system: vorticity has zero mean by construction."""

GridContext = Callable[..., ModelContext]


@pytest.fixture
def context() -> ModelContext:
    """A model context describing a small fluid dataset."""
    return make_context(SIZE)


@pytest.fixture
def grid_context() -> GridContext:
    """A context at whatever grid size a test asks for."""
    return make_context


def make_context(size: int) -> ModelContext:
    """A fluid context on a grid of the given size."""
    return ModelContext(
        field_shapes={VORTICITY_FIELD: (size, size)},
        static_fields=(),
        normalisation=STATISTICS,
        dt=DT,
        seed=SEED,
    )


def smooth_array(size: int, seed: int = 0) -> np.ndarray:
    """A band limited field, sampled on a grid of the given size.

    The same continuous function whatever the size, which is what lets a test compare a
    model on two grids and know the input was not the thing that changed.
    """
    grid = FluidGrid(size=size)
    x, y = grid.coordinates()
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size))
    for wavenumber in (1, 2, 3):
        field += rng.standard_normal() * np.sin(wavenumber * x + 0.3) * np.cos(wavenumber * y)
    return field - float(np.mean(field))


def smooth_field(size: int, seed: int = 0) -> torch.Tensor:
    """That field as a batch of one, in the precision a model works in."""
    return torch.from_numpy(smooth_array(size, seed).astype(np.float32)).unsqueeze(0)


def smooth_state(size: int, seed: int = 0) -> State:
    """That field as a state on its grid, in the precision the data is stored in."""
    return FluidGrid(size=size).make_state(smooth_array(size, seed))
