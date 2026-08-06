"""The training loop. One loop, driven by configuration, knowing no system.

Everything here works through the model interface and the dataset, so the same function
trains the N-body graph network and, in the next phase, the fluid operator. It never asks
what a field means.

Three decisions are worth stating, because each one is a bug that would otherwise be
found late or not at all.

**Shuffling is a function of the epoch, not a generator carried forward.** The order the
training windows are visited in is derived from the run seed and the epoch number, so
epoch seventeen sees the same order whether it was reached in one run or in three. A
loop that carried a generator would have to checkpoint it, and a resume that forgot to
would produce a run that was reproducible only if never interrupted.

**Early stopping reads a rollout, not the training loss.** The number it compares is the
normalised error over a fixed horizon on the validation split, measured the same way in
every epoch whatever the curriculum is doing. Stopping on the training loss would stop on
a number the model is being optimised to reduce, which it can do without getting better.

**The curriculum changes what an epoch trains on, so the loader is rebuilt when it does.**
A stage of length four reads windows of five states, which is a different dataset from a
stage of length one, not a different setting on the same one.

**Early stopping waits for the last curriculum stage, and each stage gets its own
patience.** Stopping early is a claim that more training of this kind will not help. A
scheduled longer window is not more of the same kind, and it is the thing most likely to
break the plateau that ran the counter up: measured on the fluid operator, the one step
stage flattened at a validation rollout of 0.844 and the four step stage reached 0.731 in
a single epoch. Without this, patience of ten against a stage change thirteen epochs later
stopped that run before the stage that justified the curriculum ever ran, and the number
reported would have been of a model the configuration never asked for.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from nnphysics.core.errors import ValidationError
from nnphysics.core.seeding import seed_sequence
from nnphysics.data.dataset import TrajectoryWindows, make_worker_init
from nnphysics.data.manifest import Split
from nnphysics.training.checkpoint import (
    CheckpointPaths,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from nnphysics.training.curriculum import stage_length
from nnphysics.training.history import EpochRecord, TrainingHistory
from nnphysics.training.losses import rollout_error, rollout_loss
from nnphysics.training.schedules import cosine_with_warmup

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from nnphysics.core.config import TrainingConfig
    from nnphysics.data.manifest import Manifest
    from nnphysics.models.base import SurrogateModel

__all__ = ["SHUFFLE_SEED_STREAM", "Progress", "train_model"]

SHUFFLE_SEED_STREAM = "training.shuffle.{epoch}"
"""How one epoch's visiting order is derived. It depends on the run seed and the epoch
alone, which is what makes a resumed run identical to an uninterrupted one."""

type Progress = Callable[[str], None]
"""Where the loop reports what it is doing. The CLI passes a printer, tests pass nothing."""

_MAX_UINT64 = (1 << 64) - 1


def train_model(  # noqa: PLR0913
    # The model, its settings, where the data is, what seeds it, where checkpoints go and
    # whether this is a resume. The object that would group them is the run configuration,
    # which this function deliberately does not take so that a test can train without
    # inventing one.
    model: SurrogateModel,
    config: TrainingConfig,
    directory: Path,
    manifest: Manifest,
    *,
    seed: int,
    checkpoints: CheckpointPaths,
    progress: Progress | None = None,
    resume: bool = False,
) -> TrainingHistory:
    """Train a model and record what happened.

    Args:
        model: The model, modified in place. It is left holding the weights of the last
            epoch run, not the best ones: the best ones are in the best checkpoint, and
            silently swapping them would make a resume continue from somewhere it never
            was.
        config: The training settings.
        directory: The dataset directory.
        manifest: Its manifest.
        seed: Root seed of the run.
        checkpoints: Where the best and last checkpoints go.
        progress: Called with a line of progress, or `None` to stay quiet.
        resume: Whether to continue from the last checkpoint. A missing checkpoint is an
            error rather than a fresh start, because a resume that quietly restarts is
            a resume that lost work.

    Returns:
        The history, epoch by epoch.

    Raises:
        ConfigurationError: If a resume was asked for and the checkpoint cannot be read.
        ValidationError: If the dataset cannot supply the windows the settings ask for.
    """
    report = progress or (lambda _: None)
    checkpoints.ensure()

    optimiser, scheduler = _optimiser(model, config)

    # Non overlapping windows: a validation set that stepped one state at a time would
    # measure the same rollout many times over and pay the horizon for each of them.
    validation_windows = _windows(
        directory,
        manifest,
        Split.VALIDATION,
        sequence_length=config.validation_steps,
        stride=config.validation_steps,
    )
    validation = _loader(validation_windows, config, seed=seed, shuffle=False)

    state = TrainingState(
        epoch=0, best_error=float("inf"), best_epoch=0, stalled=0, history=(), seconds=0.0
    )
    if resume:
        state = load_checkpoint(checkpoints.last, model, optimiser, scheduler)
        report(f"Resuming after epoch {state.epoch} with validation error {state.best_error:.4g}.")

    records = list(state.history)
    best_error = state.best_error
    best_epoch = state.best_epoch
    stalled = state.stalled
    elapsed = state.seconds
    stopped_early = False

    training: DataLoader[Any] | None = None
    current_length = 0
    previous_length: int | None = None
    windows = 0
    generator = torch.Generator()

    for epoch in range(state.epoch, config.epochs):
        started = time.perf_counter()
        length = stage_length(epoch, config.curriculum, config.curriculum_epochs, config.epochs)
        if previous_length is not None and length != previous_length:
            # The new stage gets its own patience. It has been given a harder problem
            # than the one whose plateau ran the counter up, so the old stage's stalled
            # epochs say nothing about it. The best error is not reset: model selection
            # still has to be beaten.
            stalled = 0
        previous_length = length
        if training is None or length != current_length:
            training, windows = _stage_loader(
                directory, manifest, config, length=length, seed=seed, generator=generator
            )
            current_length = length
            report(f"Curriculum stage of {length} step(s): {windows} training windows.")

        generator.manual_seed(_epoch_seed(seed, epoch))
        rate = float(scheduler.get_last_lr()[0])
        totals = _train_epoch(model, training, optimiser, config, length)
        error = _validate(model, validation, config.validation_steps)
        scheduler.step()

        improved = error < best_error - config.min_improvement
        if improved:
            best_error, best_epoch, stalled = error, epoch, 0
        else:
            stalled += 1
        elapsed += time.perf_counter() - started

        records.append(
            EpochRecord(
                epoch=epoch,
                curriculum_steps=length,
                learning_rate=rate,
                loss=totals["loss"],
                one_step_error=totals["one_step"],
                multi_step_error=totals["multi_step"],
                physics_penalty=totals["physics"],
                gradient_norm=totals["gradient_norm"],
                validation_error=error,
                seconds=time.perf_counter() - started,
                improved=improved,
            )
        )
        report(
            f"epoch {epoch + 1}/{config.epochs}  steps {length}  lr {rate:.3g}  "
            f"loss {totals['loss']:.4g}  validation {error:.4g}"
            f"{'  [best]' if improved else ''}"
        )

        _write(
            checkpoints,
            model,
            optimiser,
            scheduler,
            TrainingState(
                epoch=epoch + 1,
                best_error=best_error,
                best_epoch=best_epoch,
                stalled=stalled,
                history=tuple(records),
                seconds=elapsed,
            ),
            improved=improved,
        )

        if _should_stop(config, epoch=epoch, stalled=stalled):
            stopped_early = True
            report(f"Stopping: {stalled} epochs without an improvement.")
            break

    if not records:
        raise ValidationError(
            f"training ran no epochs: the checkpoint had already completed {state.epoch} "
            f"of {config.epochs}"
        )
    return TrainingHistory(
        model=model.name,
        n_parameters=model.n_parameters,
        train_windows=windows or 1,
        validation_windows=len(validation_windows),
        validation_steps=config.validation_steps,
        epochs=tuple(records),
        best_epoch=best_epoch,
        best_validation_error=best_error,
        stopped_early=stopped_early,
        seconds=elapsed,
    )


def _should_stop(config: TrainingConfig, *, epoch: int, stalled: int) -> bool:
    """Whether patience has run out, and whether it is yet entitled to.

    Stopping early claims that more training of this kind will not help. While a longer
    curriculum window is still scheduled, the next kind has not been tried, so patience
    is not entitled to end a run before the configuration's own last stage has begun.

    Args:
        config: The training settings.
        epoch: The epoch just completed.
        stalled: Epochs since the validation metric last improved.

    Returns:
        Whether to stop.
    """
    if config.patience is None or stalled < config.patience:
        return False
    return epoch >= config.curriculum_epochs[-1]


def _stage_loader(  # noqa: PLR0913
    # Where the data is, what to read, how long a window is, what seeds it and which
    # generator shuffles it. The generator is the loop's own, reseeded every epoch, so it
    # has to be passed in rather than made here.
    directory: Path,
    manifest: Manifest,
    config: TrainingConfig,
    *,
    length: int,
    seed: int,
    generator: torch.Generator,
) -> tuple[DataLoader[Any], int]:
    """The loader for one curriculum stage, and how many windows it holds."""
    stage = _windows(
        directory, manifest, Split.TRAIN, sequence_length=length, stride=config.window_stride
    )
    return _loader(stage, config, seed=seed, shuffle=True, generator=generator), len(stage)


def _train_epoch(
    model: SurrogateModel,
    loader: DataLoader[Any],
    optimiser: torch.optim.Optimizer,
    config: TrainingConfig,
    steps: int,
) -> dict[str, float]:
    """One pass over the training split, returning the epoch's mean of each term."""
    model.train()
    sums = {"loss": 0.0, "one_step": 0.0, "multi_step": 0.0, "physics": 0.0, "gradient_norm": 0.0}
    batches = 0
    for batch in loader:
        optimiser.zero_grad(set_to_none=True)
        terms = rollout_loss(
            model,
            batch["inputs"],
            batch["targets"],
            steps,
            multi_step_weight=config.multi_step_weight,
            physics_weight=config.physics_weight,
        )
        terms.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip if config.gradient_clip > 0.0 else float("inf"),
        )
        optimiser.step()
        sums["loss"] += float(terms.total.detach())
        sums["one_step"] += terms.one_step
        sums["multi_step"] += terms.multi_step
        sums["physics"] += terms.physics
        sums["gradient_norm"] += float(norm)
        batches += 1
    return {name: value / max(batches, 1) for name, value in sums.items()}


def _validate(model: SurrogateModel, loader: DataLoader[Any], steps: int) -> float:
    """Normalised error over a fixed horizon, meaned over the validation split."""
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            total += float(rollout_error(model, batch["inputs"], batch["targets"], steps).mean())
            batches += 1
    return total / max(batches, 1)


def _write(  # noqa: PLR0913
    # Everything a checkpoint is made of. They vary independently and the only object
    # that holds all of them is the loop itself.
    checkpoints: CheckpointPaths,
    model: SurrogateModel,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    *,
    improved: bool,
) -> None:
    """Write the last checkpoint, and the best one when this epoch became it."""
    if improved:
        save_checkpoint(checkpoints.best, model, optimiser, scheduler, state)
    save_checkpoint(checkpoints.last, model, optimiser, scheduler, state)


def _optimiser(
    model: SurrogateModel, config: TrainingConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """The optimiser and the schedule a run is driven by.

    AdamW rather than Adam: the weight decay a configuration asks for should be decay
    rather than an L2 term folded into the gradient, which is not the same thing once the
    optimiser is adaptive.
    """
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        cosine_with_warmup(
            epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            final_fraction=config.final_learning_rate_fraction,
        ),
    )
    return optimiser, scheduler


def _windows(
    directory: Path,
    manifest: Manifest,
    split: Split,
    *,
    sequence_length: int,
    stride: int,
) -> TrajectoryWindows:
    """Open one split as windows of consecutive states, in physical units.

    Normalisation is deliberately not applied here. The model owns it, so the loop hands
    a model the same physical units the evaluation harness will, and there is one code
    path rather than two.
    """
    return TrajectoryWindows(
        directory,
        split,
        sequence_length=sequence_length,
        stride=stride,
        normalisation=None,
        manifest=manifest,
    )


def _loader(
    windows: TrajectoryWindows,
    config: TrainingConfig,
    *,
    seed: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[Any]:
    """Batch a set of windows."""
    return DataLoader(
        windows,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config.loader_workers,
        worker_init_fn=make_worker_init(seed) if config.loader_workers else None,
        drop_last=False,
    )


def _epoch_seed(seed: int, epoch: int) -> int:
    """The seed one epoch's visiting order is drawn from."""
    stream = seed_sequence(seed, SHUFFLE_SEED_STREAM.format(epoch=epoch))
    return int(stream.generate_state(2, dtype=np.uint64)[0]) & _MAX_UINT64
