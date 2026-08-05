"""The single owner of random state.

Every stochastic function in the package takes a generator produced here. Streams are
named, and a stream's generator depends only on the root seed and the stream name, so
adding a new stream cannot shift the numbers drawn by an existing one.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

__all__ = [
    "make_deterministic",
    "numpy_generator",
    "seed_sequence",
    "torch_generator",
]

_MAX_UINT64 = (1 << 64) - 1


def seed_sequence(root_seed: int, stream: str) -> np.random.SeedSequence:
    """Derive an independent seed sequence for a named stream.

    Args:
        root_seed: The one seed a run is configured with.
        stream: Name of the stream, for example `data.train` or `model.init`.

    Returns:
        A seed sequence determined by the pair, independent of every other stream.

    Raises:
        ValueError: If the stream name is empty or the root seed is negative.
    """
    if root_seed < 0:
        raise ValueError(f"root seed must be non negative, got {root_seed}")
    if not stream:
        raise ValueError("stream name must not be empty")
    digest = hashlib.blake2b(stream.encode("utf-8"), digest_size=8).digest()
    return np.random.SeedSequence([root_seed, int.from_bytes(digest, "big")])


def numpy_generator(root_seed: int, stream: str) -> np.random.Generator:
    """Build the NumPy generator for a named stream.

    Args:
        root_seed: The one seed a run is configured with.
        stream: Name of the stream.

    Returns:
        A fresh generator, unaffected by any other stream.
    """
    return np.random.default_rng(seed_sequence(root_seed, stream))


def torch_generator(root_seed: int, stream: str) -> torch.Generator:
    """Build the PyTorch generator for a named stream.

    Args:
        root_seed: The one seed a run is configured with.
        stream: Name of the stream.

    Returns:
        A CPU generator, unaffected by any other stream.
    """
    state = seed_sequence(root_seed, stream).generate_state(2, dtype=np.uint64)
    generator = torch.Generator()
    generator.manual_seed(int(state[0]) & _MAX_UINT64)
    return generator


def make_deterministic(root_seed: int) -> None:
    """Pin the global random state of every library that has one.

    Package code never draws from a global generator. This exists for dependencies that
    do, and for the interpreter's own hash randomisation, which must be set before the
    process starts to have any effect.

    Args:
        root_seed: The one seed a run is configured with.

    Raises:
        ValueError: If the root seed is negative.
    """
    if root_seed < 0:
        raise ValueError(f"root seed must be non negative, got {root_seed}")
    os.environ["PYTHONHASHSEED"] = str(root_seed)
    random.seed(root_seed)
    np.random.seed(root_seed % (1 << 32))
    torch.manual_seed(root_seed & _MAX_UINT64)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
