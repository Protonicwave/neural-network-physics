from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nnphysics.core.errors import ConfigurationError
from nnphysics.models import ModelContext, SurrogateModel, build_model, load_model, save_model
from nnphysics.training import CheckpointPaths, TrainingState, load_checkpoint, save_checkpoint
from nnphysics.training.history import EpochRecord

RECORD = EpochRecord(
    epoch=0,
    curriculum_steps=1,
    learning_rate=0.01,
    loss=0.5,
    one_step_error=0.5,
    multi_step_error=0.0,
    physics_penalty=0.0,
    gradient_norm=1.25,
    validation_error=0.4,
    seconds=0.1,
    improved=True,
)

STATE = TrainingState(
    epoch=1, best_error=0.4, best_epoch=0, stalled=0, history=(RECORD,), seconds=0.1
)


def pieces(
    context: ModelContext,
) -> tuple[SurrogateModel, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """A model, an optimiser and a schedule, all freshly built."""
    model = build_model("constant", context)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: 0.5**epoch)
    return model, optimiser, scheduler


def test_the_paths_are_named_inside_the_directory(tmp_path: Path) -> None:
    paths = CheckpointPaths(tmp_path / "checkpoints")
    paths.ensure()
    assert paths.best.parent == paths.root
    assert paths.best != paths.last
    assert paths.root.is_dir()


def test_a_checkpoint_restores_where_a_run_had_got_to(
    context: ModelContext, tmp_path: Path
) -> None:
    model, optimiser, scheduler = pieces(context)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimiser, scheduler, STATE)

    restored = load_checkpoint(path, *pieces(context))
    assert restored == STATE


def test_the_optimiser_state_survives_the_round_trip(context: ModelContext, tmp_path: Path) -> None:
    """The failure this catches is silent: a resume without it takes a different step."""
    model, optimiser, scheduler = pieces(context)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.3)
    optimiser.step()
    expected = optimiser.state_dict()["state"]

    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimiser, scheduler, STATE)
    fresh_model, fresh_optimiser, fresh_scheduler = pieces(context)
    assert not fresh_optimiser.state_dict()["state"]

    load_checkpoint(path, fresh_model, fresh_optimiser, fresh_scheduler)
    actual = fresh_optimiser.state_dict()["state"]
    assert set(actual) == set(expected)
    for key, entry in expected.items():
        for name, value in entry.items():
            assert torch.equal(torch.as_tensor(actual[key][name]), torch.as_tensor(value))


def test_the_schedule_position_survives_the_round_trip(
    context: ModelContext, tmp_path: Path
) -> None:
    model, optimiser, scheduler = pieces(context)
    for _ in range(2):
        optimiser.step()
        scheduler.step()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimiser, scheduler, STATE)

    fresh = pieces(context)
    load_checkpoint(path, *fresh)
    assert fresh[2].get_last_lr() == scheduler.get_last_lr()


def test_a_training_checkpoint_is_also_a_model_checkpoint(
    context: ModelContext, tmp_path: Path
) -> None:
    """One payload, so the best checkpoint can be evaluated without unpacking a run."""
    model, optimiser, scheduler = pieces(context)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, optimiser, scheduler, STATE)
    restored = load_model(path)
    assert restored.name == model.name
    assert restored.context.static_fields == context.static_fields


def test_a_model_checkpoint_cannot_be_resumed_from(context: ModelContext, tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save_model(path, build_model("constant", context))
    with pytest.raises(ConfigurationError, match="no training state"):
        load_checkpoint(path, *pieces(context))


def test_a_missing_checkpoint_is_refused(context: ModelContext, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_checkpoint(tmp_path / "absent.pt", *pieces(context))


def test_a_checkpoint_of_another_version_is_refused(context: ModelContext, tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    torch.save({"schema_version": 99, "training": {}}, path)
    with pytest.raises(ConfigurationError, match="not a version"):
        load_checkpoint(path, *pieces(context))


def test_a_checkpoint_that_does_not_fit_the_model_is_refused(
    context: ModelContext, tmp_path: Path
) -> None:
    model, optimiser, scheduler = pieces(context)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimiser, scheduler, STATE)

    other = build_model("mlp", context)
    with pytest.raises(ConfigurationError):
        load_checkpoint(path, other, torch.optim.AdamW(other.parameters(), lr=0.01), scheduler)
