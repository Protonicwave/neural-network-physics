import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.training import cosine_with_warmup, learning_rate_factor

EPOCHS = 20
WARMUP = 4
FINAL = 0.05


def factor(
    epoch: int,
    *,
    epochs: int = EPOCHS,
    warmup_epochs: int = WARMUP,
    final_fraction: float = FINAL,
) -> float:
    return learning_rate_factor(
        epoch, epochs=epochs, warmup_epochs=warmup_epochs, final_fraction=final_fraction
    )


def test_warmup_rises_and_reaches_the_peak_exactly_once() -> None:
    rising = [factor(epoch) for epoch in range(WARMUP)]
    assert rising == sorted(rising)
    assert rising[-1] == pytest.approx(1.0)


def test_the_first_epoch_is_not_a_wasted_one() -> None:
    """A rate of exactly zero would spend an epoch computing gradients and dropping them."""
    assert factor(0) > 0.0


def test_no_warmup_starts_at_the_peak() -> None:
    assert factor(0, warmup_epochs=0) == pytest.approx(1.0)


def test_the_cosine_falls_after_warmup() -> None:
    falling = [factor(epoch) for epoch in range(WARMUP, EPOCHS)]
    assert falling == sorted(falling, reverse=True)


def test_the_rate_stays_between_the_floor_and_the_peak() -> None:
    assert all(FINAL <= factor(epoch) <= 1.0 for epoch in range(EPOCHS))


def test_the_last_epoch_is_near_the_floor() -> None:
    assert factor(EPOCHS - 1) < FINAL + 0.05


def test_a_floor_of_one_never_decays() -> None:
    assert factor(EPOCHS - 1, warmup_epochs=0, final_fraction=1.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("epoch", "epochs", "warmup_epochs", "final_fraction"),
    [
        (EPOCHS, EPOCHS, WARMUP, FINAL),
        (-1, EPOCHS, WARMUP, FINAL),
        (0, 0, 0, FINAL),
        (0, EPOCHS, EPOCHS, FINAL),
        (0, EPOCHS, WARMUP, 1.5),
    ],
)
def test_a_schedule_outside_its_own_definition_is_refused(
    epoch: int, epochs: int, warmup_epochs: int, final_fraction: float
) -> None:
    with pytest.raises(ValidationError):
        factor(
            epoch,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            final_fraction=final_fraction,
        )


def test_the_bound_schedule_is_the_same_function() -> None:
    bound = cosine_with_warmup(epochs=EPOCHS, warmup_epochs=WARMUP, final_fraction=FINAL)
    assert [bound(epoch) for epoch in range(EPOCHS)] == [factor(epoch) for epoch in range(EPOCHS)]


def test_the_bound_schedule_holds_past_the_last_epoch() -> None:
    """`LambdaLR` steps once more than a loop runs, and must not raise for it."""
    bound = cosine_with_warmup(epochs=EPOCHS, warmup_epochs=WARMUP, final_fraction=FINAL)
    assert bound(EPOCHS) == pytest.approx(bound(EPOCHS - 1))
