"""Dividing trajectories into splits.

Two rules decide everything here.

Splitting is by trajectory and never by time step. A surrogate learns single step pairs,
so splitting by step would put step 100 of a trajectory in training and step 101 of the
same trajectory in test. The test error that follows measures interpolation within a
trajectory the model has already seen, which is not what anyone reads it as.

A held out regime is held out whole. It contributes to no training split, so a result on
it is a statement about generalisation to a region of parameter space rather than about
generalisation to new draws from the same one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nnphysics.core.errors import ValidationError
from nnphysics.data.manifest import Split

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["SPLIT_SEED_STREAM", "make_splits"]

SPLIT_SEED_STREAM = "data.split"
"""One stream for the whole split, so adding a regime does not reshuffle the others."""


def make_splits(
    trainable: Mapping[str, Sequence[str]],
    held_out: Mapping[str, Sequence[str]],
    *,
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> dict[Split, tuple[str, ...]]:
    """Divide trajectory identifiers into the four splits.

    Each trainable regime is divided separately, so every split holds every trainable
    regime in the same proportion. A split that drew regimes at random could leave one
    regime out of validation entirely and make the validation loss mean something
    different from the training loss.

    Args:
        trainable: Regime name to its trajectory identifiers.
        held_out: Regime name to its trajectory identifiers, for regimes held out whole.
        val_fraction: Share of each trainable regime used for validation.
        test_fraction: Share of each trainable regime used for testing.
        rng: Generator the permutation is drawn from.

    Returns:
        Split to its trajectory identifiers, sorted.

    Raises:
        ValidationError: If a regime appears on both sides, the fractions do not leave a
            non empty training split, or a regime is too small to divide.
    """
    if overlap := sorted(set(trainable) & set(held_out)):
        raise ValidationError(f"regimes are both trained on and held out: {overlap}")
    if not trainable:
        raise ValidationError("splitting needs at least one trainable regime")
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValidationError(
            f"split fractions must lie in (0, 1), got {val_fraction} and {test_fraction}"
        )
    if val_fraction + test_fraction >= 1.0:
        raise ValidationError(
            f"validation and test fractions {val_fraction} and {test_fraction} leave no "
            f"training split"
        )

    members: dict[Split, list[str]] = {split: [] for split in Split}
    # Sorted so that the order regimes were generated in cannot change the permutation.
    for regime in sorted(trainable):
        identifiers = sorted(trainable[regime])
        _check_divisible(regime, identifiers, val_fraction, test_fraction)
        shuffled = [identifiers[position] for position in rng.permutation(len(identifiers))]
        n_val = int(len(shuffled) * val_fraction)
        n_test = int(len(shuffled) * test_fraction)
        members[Split.VALIDATION].extend(shuffled[:n_val])
        members[Split.TEST].extend(shuffled[n_val : n_val + n_test])
        members[Split.TRAIN].extend(shuffled[n_val + n_test :])
    for regime in sorted(held_out):
        members[Split.HELD_OUT].extend(sorted(held_out[regime]))

    return {split: tuple(sorted(names)) for split, names in members.items() if names}


def _check_divisible(
    regime: str, identifiers: Sequence[str], val_fraction: float, test_fraction: float
) -> None:
    """Raise if a regime is too small to fill every split.

    Raises:
        ValidationError: If the regime has duplicates, or a split would come out empty.
    """
    if len(set(identifiers)) != len(identifiers):
        raise ValidationError(f"regime {regime!r} lists a trajectory more than once")
    count = len(identifiers)
    n_val = int(count * val_fraction)
    n_test = int(count * test_fraction)
    if n_val < 1 or n_test < 1 or count - n_val - n_test < 1:
        raise ValidationError(
            f"regime {regime!r} has {count} trajectories, too few to divide into "
            f"{val_fraction} validation and {test_fraction} test and leave a training split"
        )
