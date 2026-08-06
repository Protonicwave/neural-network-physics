import pytest

from nnphysics.core.errors import ValidationError
from nnphysics.training import stage_length, stages

CURRICULUM = (1, 4, 8)
STARTS = (0, 10, 16)
EPOCHS = 20


def test_the_stages_cover_every_epoch_exactly_once() -> None:
    laid_out = stages(CURRICULUM, STARTS, EPOCHS)
    covered = [
        epoch for stage in laid_out for epoch in range(stage.first_epoch, stage.last_epoch + 1)
    ]
    assert covered == list(range(EPOCHS))


def test_a_stage_reports_the_epochs_it_spans() -> None:
    first, second, third = stages(CURRICULUM, STARTS, EPOCHS)
    assert (first.epochs, second.epochs, third.epochs) == (10, 6, 4)
    assert third.last_epoch == EPOCHS - 1


def test_the_length_of_an_epoch_is_the_stage_it_falls_in() -> None:
    lengths = [stage_length(epoch, CURRICULUM, STARTS, EPOCHS) for epoch in range(EPOCHS)]
    assert lengths[:10] == [1] * 10
    assert lengths[10:16] == [4] * 6
    assert lengths[16:] == [8] * 4


def test_a_single_stage_covers_the_whole_run() -> None:
    assert [stage_length(epoch, (1,), (0,), 3) for epoch in range(3)] == [1, 1, 1]


@pytest.mark.parametrize(
    ("curriculum", "starts", "epochs", "message"),
    [
        ((), (), EPOCHS, "one start epoch per length"),
        ((1, 4), (0,), EPOCHS, "one start epoch per length"),
        ((0,), (0,), EPOCHS, "must be positive"),
        ((1, 4), (2, 5), EPOCHS, "must start at epoch 0"),
        ((1, 4), (0, 0), EPOCHS, "must increase"),
        ((1, 4), (0, 5), 4, "never reach"),
    ],
)
def test_a_malformed_curriculum_is_refused(
    curriculum: tuple[int, ...], starts: tuple[int, ...], epochs: int, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        stages(curriculum, starts, epochs)


def test_an_epoch_outside_the_run_has_no_stage() -> None:
    with pytest.raises(ValidationError):
        stage_length(EPOCHS, CURRICULUM, STARTS, EPOCHS)
