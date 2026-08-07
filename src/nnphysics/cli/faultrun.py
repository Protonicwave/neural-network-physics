"""Running one deliberately broken copy of a known good run.

The fault catalogue is declarative and says nothing about how a run is executed. This is
where that gets done: the configuration is transformed, the dataset it now names is found
or generated, the model is built, trained and scored, and a run record is written that is
indistinguishable in shape from one a real run would have produced. That last part is the
point. If a fault run were recognisable as a fault run, the diagnosis would be scored on
noticing the label rather than on reading the numbers.

Two faults cannot be expressed as a configuration and are injected here instead. One
builds the model with statistics that do not describe its data. The other trains the run
in two halves and throws away the optimiser moments in between, which is the failure the
phase 07 checkpoint tests exist to catch and the only one in the set that leaves no trace
anywhere except in the loss curve.

No snapshots are kept. They are the largest artefact a run writes and the diagnosis never
reads them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import torch

from nnphysics import __version__
from nnphysics.agent.faults import (
    Fault,
    Injection,
    RenamedPredictor,
    corrupt_normalisation,
    interrupted,
)
from nnphysics.cli.pipeline import Echo, dataset_normalisation, ensure_dataset, model_context
from nnphysics.cli.predictors import CHECKPOINT_DIR, ModelFactory
from nnphysics.core.errors import ValidationError
from nnphysics.core.seeding import make_deterministic
from nnphysics.evals.predictors.broken import SymmetryBreak
from nnphysics.evals.result import write_result
from nnphysics.evals.runner import run_suite
from nnphysics.models import build_model, load_model
from nnphysics.models.base import CHECKPOINT_SCHEMA_VERSION
from nnphysics.reporting.environment import describe_environment, git_commit
from nnphysics.reporting.layout import fault_paths
from nnphysics.reporting.record import RunRecord, write_record
from nnphysics.systems import build_system
from nnphysics.training import CheckpointPaths, train_model

if TYPE_CHECKING:
    from pathlib import Path

    from nnphysics.core.config import RunConfig
    from nnphysics.core.protocols import Predictor, System
    from nnphysics.data.manifest import Manifest
    from nnphysics.models import SurrogateModel
    from nnphysics.training import TrainingHistory

__all__ = ["BASELINE_LABEL", "FaultRun", "execute"]

BASELINE_LABEL = "baseline"
"""What the known good run of the suite is called. Not a fault, and it is run through the
same code path as the faults so that the two records differ only in the fault."""


@dataclass(frozen=True, slots=True)
class FaultRun:
    """One run of the fault suite.

    Attributes:
        label: The fault's name, or `baseline` for the known good run.
        injected: The fault, or `None` for the baseline.
        record: The run record it produced.
        directory: Where its artefacts were written.
    """

    label: str
    injected: Fault | None
    record: RunRecord
    directory: Path


def execute(config: RunConfig, injected: Fault | None, *, echo: Echo) -> FaultRun:
    """Run one copy of a configuration, broken or not, and record it.

    Args:
        config: The known good run configuration. Transformed by the fault, if there is
            one, before anything else happens.
        injected: The fault to inject, or `None` to run the configuration as it stands.
        echo: How to report progress.

    Returns:
        The run.

    Raises:
        ValidationError: If the fault cannot be applied to this configuration.
        ConfigurationError: If a dataset or checkpoint cannot be read.
    """
    label = injected.name if injected is not None else BASELINE_LABEL
    resolved = injected.apply(config) if injected is not None else config
    injection = injected.injection if injected is not None else Injection.NONE

    make_deterministic(resolved.run_seed)
    paths = fault_paths(resolved, label)
    paths.ensure()
    checkpoints = CheckpointPaths(paths.root / CHECKPOINT_DIR)

    directory, manifest = ensure_dataset(resolved, echo=echo)
    statistics = dataset_normalisation(directory, manifest, echo=echo)
    if injection is Injection.NORMALISATION:
        statistics = corrupt_normalisation(statistics)
        echo("Injected normalisation statistics that do not describe the training split.")

    model = build_model(
        resolved.model.name,
        model_context(directory, manifest, resolved, statistics),
        resolved.model.hyperparameters,
    )
    echo(
        f"[{label}] training {model.name}: {model.n_parameters} parameters, "
        f"{resolved.training.epochs} epochs."
    )

    started = time.perf_counter()
    if injection is Injection.OPTIMISER_STATE:
        history = _train_interrupted(model, resolved, directory, manifest, checkpoints, echo)
    else:
        history = train_model(
            model,
            resolved.training,
            directory,
            manifest,
            seed=resolved.run_seed,
            checkpoints=checkpoints,
            resume=False,
        )
    trained = time.perf_counter() - started

    started = time.perf_counter()
    system = build_system(resolved.system.name, resolved.system.parameters)
    best = load_model(checkpoints.best)
    predictor: Predictor = best
    if injection is Injection.SYMMETRY:
        predictor = _break_symmetry(best, system)
        echo(f"Injected a broken {system.symmetries[0].name} symmetry into the trained model.")

    result = run_suite(
        system,
        directory,
        manifest,
        resolved.evaluation,
        seed=resolved.run_seed,
        run_id=resolved.run_id,
        predictors=(*resolved.evaluation.predictors, best.name),
        factories={best.name: ModelFactory(predictor)},
    )
    evaluated = time.perf_counter() - started

    destination = paths.result(resolved.evaluation.name)
    write_result(destination, result)
    record = RunRecord(
        run_id=resolved.run_id,
        name=resolved.name,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        code_version=__version__,
        commit=git_commit(),
        config=resolved,
        environment=describe_environment(),
        timings={"training": trained, "evaluation": evaluated},
        training=history,
        evaluation=result,
        artefacts={
            "checkpoint": paths.relative(checkpoints.best),
            "result": paths.relative(destination),
        },
    )
    write_record(paths.record, record)
    echo(
        f"[{label}] validation error {history.best_validation_error:.4g} at epoch "
        f"{history.best_epoch}, wrote {paths.record}."
    )
    return FaultRun(label=label, injected=injected, record=record, directory=paths.root)


def _train_interrupted(  # noqa: PLR0913, PLR0917
    # Everything training needs, plus where to report progress. They vary independently
    # and the object that would group them is the run configuration, which supplies only
    # some of them.
    model: SurrogateModel,
    config: RunConfig,
    directory: Path,
    manifest: Manifest,
    checkpoints: CheckpointPaths,
    echo: Echo,
) -> TrainingHistory:
    """Train in two halves, losing the optimiser moments in between.

    The first half is a real run of a shorter schedule, so the checkpoint it leaves is a
    real checkpoint. What is then removed from it is exactly what a resume that saved only
    the weights would have failed to save.
    """
    first = interrupted(config)
    train_model(
        model,
        first.training,
        directory,
        manifest,
        seed=config.run_seed,
        checkpoints=checkpoints,
        resume=False,
    )
    _discard_optimiser_state(checkpoints.last)
    echo(
        f"Discarded the optimiser moments from {checkpoints.last.name} after "
        f"{first.training.epochs} of {config.training.epochs} epochs."
    )
    return train_model(
        model,
        config.training,
        directory,
        manifest,
        seed=config.run_seed,
        checkpoints=checkpoints,
        resume=True,
    )


def _discard_optimiser_state(path: Path) -> None:
    """Empty a checkpoint's optimiser moments, leaving everything else intact.

    The parameter groups stay, so the file still loads and the resumed run still uses the
    schedule it was configured with. Only the per parameter moment estimates go, which is
    the part a resume silently does without when nobody saved it.

    Args:
        path: The last checkpoint.

    Raises:
        ValidationError: If the file is not a checkpoint carrying training state.
    """
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValidationError(f"{path} is not a checkpoint this build can damage on purpose")
    training = payload.get("training")
    if not isinstance(training, dict) or "optimiser" not in training:
        raise ValidationError(f"{path} carries no optimiser state to discard")
    training["optimiser"] = {**training["optimiser"], "state": {}}
    torch.save(payload, path)


def _break_symmetry(model: Predictor, system: System) -> Predictor:
    """Wrap a trained model so it stops commuting with the system's other symmetries.

    Reported under the model's own name. The transformation has a name of its own, and it
    belongs to one of the suite's permanent fixtures; reporting the wrapped model under it
    would collide with the fixture, and under anything else the comparison would have
    nothing on the baseline side to line it up against.

    Raises:
        ValidationError: If the system declares no symmetry to break.
    """
    if not system.symmetries:
        raise ValidationError(
            f"system {system.name!r} declares no symmetry, so none of its can be broken"
        )
    return RenamedPredictor(
        inner=SymmetryBreak(inner=model, symmetry=system.symmetries[0]), name=model.name
    )
