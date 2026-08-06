from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from nnphysics.core.config import RunConfig, TrainingConfig
from nnphysics.core.errors import ConfigurationError
from nnphysics.data.manifest import Manifest
from nnphysics.models import ModelContext, SurrogateModel, build_model
from nnphysics.training import CheckpointPaths, TrainingHistory, train_model

Dataset = tuple[Path, Manifest, RunConfig]
Trainer = Callable[..., TrainingConfig]


def run(  # noqa: PLR0913
    # A dataset, a context, settings, somewhere to put checkpoints and two switches.
    dataset: Dataset,
    context: ModelContext,
    config: TrainingConfig,
    root: Path,
    *,
    name: str = "mlp",
    resume: bool = False,
    model: SurrogateModel | None = None,
) -> tuple[SurrogateModel, TrainingHistory]:
    """Train one model, returning it and its history."""
    directory, manifest, run_config = dataset
    trained = model if model is not None else build_model(name, context)
    history = train_model(
        trained,
        config,
        directory,
        manifest,
        seed=run_config.seed,
        checkpoints=CheckpointPaths(root),
        resume=resume,
    )
    return trained, history


def weights(model: SurrogateModel) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in model.state_dict().items()}


def identical(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def test_the_same_configuration_and_seed_give_identical_weights(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    """Reproducibility is the claim the whole repository rests on."""
    config = training(epochs=3)
    first, _ = run(dataset, context, config, tmp_path / "one")
    second, _ = run(dataset, context, config, tmp_path / "two")
    assert identical(weights(first), weights(second))


def test_resuming_gives_the_run_that_was_never_interrupted(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    """This is the test that catches a forgotten optimiser state."""
    uninterrupted, _ = run(dataset, context, training(epochs=4), tmp_path / "whole")

    partial = tmp_path / "split"
    run(dataset, context, training(epochs=4, patience=2), partial)
    resumed, history = run(
        dataset,
        context,
        training(epochs=4),
        partial,
        resume=True,
        model=build_model("mlp", context),
    )
    assert history.epochs[-1].epoch == 3
    assert identical(weights(uninterrupted), weights(resumed))


def test_a_resume_needs_something_to_resume_from(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        run(dataset, context, training(), tmp_path / "empty", resume=True)


def test_one_record_is_written_for_every_epoch_run(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    _, history = run(dataset, context, training(epochs=3), tmp_path)
    assert [record.epoch for record in history.epochs] == [0, 1, 2]
    assert history.model == "mlp"
    assert history.n_parameters > 0
    assert history.validation_steps == 2


def test_the_curriculum_lengthens_the_window_it_trains_on(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    config = training(epochs=4, curriculum=(1, 3), curriculum_epochs=(0, 2))
    _, history = run(dataset, context, config, tmp_path)
    assert [record.curriculum_steps for record in history.epochs] == [1, 1, 3, 3]


def test_a_one_step_stage_has_no_multi_step_error_and_a_longer_one_does(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    config = training(epochs=4, curriculum=(1, 3), curriculum_epochs=(0, 2))
    _, history = run(dataset, context, config, tmp_path)
    assert history.epochs[0].multi_step_error == 0.0
    assert history.epochs[-1].multi_step_error > 0.0


def test_the_learning_rate_follows_the_schedule(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    config = training(epochs=4, warmup_epochs=2)
    _, history = run(dataset, context, config, tmp_path)
    rates = [record.learning_rate for record in history.epochs]
    assert rates[0] < rates[1] == pytest.approx(config.learning_rate)
    assert rates[2] > rates[3]


def test_the_best_checkpoint_is_the_best_epoch_not_the_last(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    _, history = run(dataset, context, training(epochs=3), tmp_path)
    errors = [record.validation_error for record in history.epochs]
    assert history.best_validation_error == pytest.approx(min(errors))
    assert history.best_epoch == errors.index(min(errors))
    assert history.epochs[history.best_epoch].improved


def test_both_checkpoints_are_written(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    run(dataset, context, training(epochs=2), tmp_path)
    paths = CheckpointPaths(tmp_path)
    assert paths.best.is_file()
    assert paths.last.is_file()


def test_training_stops_when_patience_runs_out(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    """Stopping reads the validation rollout, so a falling training loss cannot hide."""
    config = training(epochs=20, patience=1, learning_rate=1e-9)
    _, history = run(dataset, context, config, tmp_path, name="constant")
    assert history.stopped_early
    assert len(history.epochs) < config.epochs


def test_a_scheduled_curriculum_stage_always_runs(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    """A plateau is the reason the next stage exists, so it must not pre-empt it.

    The learning rate is small enough that no epoch improves, so patience of one would
    stop at epoch one if it applied while a longer window was still scheduled. Training
    reaches the four step stage instead, which is what the configuration asked for.
    """
    config = training(
        epochs=10,
        patience=1,
        learning_rate=1e-9,
        curriculum=(1, 4),
        curriculum_epochs=(0, 5),
    )

    _, history = run(dataset, context, config, tmp_path, name="constant")

    assert len(history.epochs) > 5
    assert history.epochs[5].curriculum_steps == 4


def test_the_last_stage_still_stops_when_its_own_patience_runs_out(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    """The other half: waiting for the last stage must not turn early stopping off."""
    config = training(
        epochs=10,
        patience=1,
        learning_rate=1e-9,
        curriculum=(1, 4),
        curriculum_epochs=(0, 5),
    )

    _, history = run(dataset, context, config, tmp_path, name="constant")

    assert history.stopped_early
    assert len(history.epochs) < config.epochs


def test_patience_of_none_runs_every_epoch(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    config = training(epochs=3, patience=None, learning_rate=1e-9)
    _, history = run(dataset, context, config, tmp_path, name="constant")
    assert not history.stopped_early
    assert len(history.epochs) == 3


def test_the_loop_trains_whatever_model_it_is_handed(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    """One loop, no branch on which model or which system it is driving."""
    for name in ("constant", "mlp", "graph"):
        model, history = run(dataset, context, training(epochs=2), tmp_path / name, name=name)
        assert history.model == name
        assert model.n_parameters == history.n_parameters


def test_training_reduces_the_loss_it_is_given(
    dataset: Dataset, context: ModelContext, training: Trainer, tmp_path: Path
) -> None:
    """A loop that cannot fit the weakest model has a bug capacity would not reveal."""
    _, history = run(dataset, context, training(epochs=6), tmp_path, name="constant")
    assert history.epochs[-1].loss < history.epochs[0].loss
