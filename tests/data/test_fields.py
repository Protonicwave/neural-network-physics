from pathlib import Path

import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.data.fields import constant_fields
from nnphysics.data.manifest import Manifest, Split
from nnphysics.systems.nbody.state import MASS_FIELD, POSITION_FIELD, VELOCITY_FIELD


def test_mass_is_the_only_field_an_nbody_trajectory_holds_still(
    nbody_dataset: tuple[Path, Manifest],
) -> None:
    directory, manifest = nbody_dataset
    assert constant_fields(directory, manifest) == (MASS_FIELD,)


def test_the_fields_that_move_are_the_ones_a_model_must_predict(
    nbody_dataset: tuple[Path, Manifest],
) -> None:
    directory, manifest = nbody_dataset
    constant = set(constant_fields(directory, manifest))
    assert POSITION_FIELD not in constant
    assert VELOCITY_FIELD not in constant


def test_the_answer_is_read_from_the_training_split_by_default(
    nbody_dataset: tuple[Path, Manifest],
) -> None:
    directory, manifest = nbody_dataset
    assert constant_fields(directory, manifest) == constant_fields(
        directory, manifest, split=Split.TRAIN
    )


def test_an_empty_split_is_an_error_rather_than_an_empty_answer(
    nbody_dataset: tuple[Path, Manifest],
) -> None:
    directory, manifest = nbody_dataset
    empty = manifest.model_copy(
        update={"splits": manifest.splits.model_copy(update={"members": {Split.TRAIN: ()}})}
    )
    with pytest.raises(ValidationError, match="empty split"):
        constant_fields(directory, empty)
