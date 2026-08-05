import numpy as np
import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.data.manifest import Split
from nnphysics.data.splits import make_splits

FRACTIONS = {"val_fraction": 0.25, "test_fraction": 0.25}


def _identifiers(regime: str, count: int) -> list[str]:
    return [f"{regime}/{index:05d}" for index in range(count)]


def _trainable(count: int = 8) -> dict[str, list[str]]:
    return {name: _identifiers(name, count) for name in ("alpha", "beta")}


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_every_trajectory_lands_in_exactly_one_split() -> None:
    trainable = _trainable()
    held_out = {"gamma": _identifiers("gamma", 4)}
    members = make_splits(trainable, held_out, **FRACTIONS, rng=_rng())

    placed = [name for names in members.values() for name in names]
    expected = [name for names in (*trainable.values(), *held_out.values()) for name in names]
    assert sorted(placed) == sorted(expected)
    assert len(set(placed)) == len(placed)


def test_a_held_out_regime_appears_in_no_training_split() -> None:
    held_out = {"gamma": _identifiers("gamma", 4)}
    members = make_splits(_trainable(), held_out, **FRACTIONS, rng=_rng())

    assert set(members[Split.HELD_OUT]) == set(held_out["gamma"])
    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
        assert not any(name.startswith("gamma/") for name in members[split])


def test_every_trainable_regime_is_represented_in_every_split() -> None:
    members = make_splits(_trainable(), {}, **FRACTIONS, rng=_rng())
    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
        regimes = {name.split("/")[0] for name in members[split]}
        assert regimes == {"alpha", "beta"}


def test_the_same_seed_gives_the_same_split() -> None:
    first = make_splits(_trainable(), {}, **FRACTIONS, rng=_rng(7))
    second = make_splits(_trainable(), {}, **FRACTIONS, rng=_rng(7))
    assert first == second


def test_a_different_seed_gives_a_different_split() -> None:
    first = make_splits(_trainable(count=16), {}, **FRACTIONS, rng=_rng(1))
    second = make_splits(_trainable(count=16), {}, **FRACTIONS, rng=_rng(2))
    assert first != second


def test_the_order_regimes_arrive_in_does_not_change_the_split() -> None:
    forward = dict(sorted(_trainable().items()))
    backward = dict(sorted(_trainable().items(), reverse=True))
    assert make_splits(forward, {}, **FRACTIONS, rng=_rng(4)) == make_splits(
        backward, {}, **FRACTIONS, rng=_rng(4)
    )


def test_the_fractions_are_applied_to_each_regime() -> None:
    members = make_splits(_trainable(count=8), {}, **FRACTIONS, rng=_rng())
    assert len(members[Split.VALIDATION]) == 4
    assert len(members[Split.TEST]) == 4
    assert len(members[Split.TRAIN]) == 8


def test_a_regime_too_small_to_divide_is_rejected() -> None:
    with pytest.raises(ValidationError, match="too few to divide"):
        make_splits({"alpha": _identifiers("alpha", 3)}, {}, **FRACTIONS, rng=_rng())


def test_a_regime_on_both_sides_is_rejected() -> None:
    identifiers = _identifiers("alpha", 8)
    with pytest.raises(ValidationError, match="both trained on and held out"):
        make_splits({"alpha": identifiers}, {"alpha": identifiers}, **FRACTIONS, rng=_rng())


def test_a_duplicated_trajectory_is_rejected() -> None:
    duplicated = [*_identifiers("alpha", 7), "alpha/00000"]
    with pytest.raises(ValidationError, match="more than once"):
        make_splits({"alpha": duplicated}, {}, **FRACTIONS, rng=_rng())


def test_splitting_nothing_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one trainable regime"):
        make_splits({}, {}, **FRACTIONS, rng=_rng())


@pytest.mark.parametrize(
    ("val_fraction", "test_fraction"), [(0.0, 0.25), (0.25, 1.0), (0.5, 0.5), (0.8, 0.3)]
)
def test_unusable_fractions_are_rejected(val_fraction: float, test_fraction: float) -> None:
    with pytest.raises(ValidationError):
        make_splits(
            _trainable(),
            {},
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            rng=_rng(),
        )
